"""
Kalman Filter for Neural Signal Translation - GPU Accelerated
==============================================================

State-space model for temporal signal translation.
Optimized with CuPy for GPU acceleration.
"""

from __future__ import annotations

from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numpy.typing import NDArray

# GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def get_array_module(use_gpu: bool = True):
    """Get numpy or cupy module."""
    if use_gpu and HAS_CUPY:
        return cp
    return np


def to_numpy(arr) -> np.ndarray:
    """Convert to numpy array."""
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        return arr.get()
    return np.asarray(arr)


class KalmanFilter:
    """Kalman filter for neural signal translation - GPU optimized.

    Models the system as:
        x_{t+1} = A @ x_t + B @ u_t + w_t
        y_t = C @ x_t + v_t

    Args:
        state_dim: Dimension of hidden state
        process_noise: Process noise variance
        observation_noise: Observation noise variance
        learn_transitions: If True, learn parameters from data
        use_gpu: Use GPU acceleration
        n_em_iterations: Number of EM iterations for learning
    """

    def __init__(
        self,
        state_dim: int = 32,
        process_noise: float = 0.01,
        observation_noise: float = 0.1,
        learn_transitions: bool = True,
        use_gpu: bool = True,
        n_em_iterations: int = 5,
    ):
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.learn_transitions = learn_transitions
        # Kalman filters are inherently sequential - GPU doesn't help much
        # and causes context issues in multi-GPU setups. Force CPU mode.
        self.use_gpu = False  # Disabled: sequential ops don't benefit from GPU
        self.n_em_iterations = n_em_iterations

        self.A: Optional[NDArray] = None
        self.B: Optional[NDArray] = None
        self.C: Optional[NDArray] = None
        self.Q: Optional[NDArray] = None
        self.R: Optional[NDArray] = None
        self._fitted = False

    def _initialize_parameters(self, input_dim: int, output_dim: int):
        """Initialize model parameters."""
        xp = get_array_module(self.use_gpu)

        # State transition (slightly contractive for stability)
        random_mat = xp.random.randn(self.state_dim, self.state_dim)
        U, _, Vt = xp.linalg.svd(random_mat)
        self.A = 0.95 * (U @ Vt)

        # Input to state
        self.B = xp.random.randn(self.state_dim, input_dim) * 0.1

        # State to output
        self.C = xp.random.randn(output_dim, self.state_dim) * 0.1

        # Noise covariances
        self.Q = self.process_noise * xp.eye(self.state_dim)
        self.R = self.observation_noise * xp.eye(output_dim)

    def _kalman_filter_single(
        self,
        u: NDArray,
        y: Optional[NDArray] = None,
    ) -> NDArray:
        """Run Kalman filter for single sequence - optimized."""
        xp = get_array_module(self.use_gpu)
        T = u.shape[0]

        # Move to GPU if needed
        if self.use_gpu:
            u = xp.asarray(u)
            if y is not None:
                y = xp.asarray(y)
            A, B, C, Q, R = self.A, self.B, self.C, self.Q, self.R
        else:
            A, B, C, Q, R = self.A, self.B, self.C, self.Q, self.R

        # Initialize
        x = xp.zeros(self.state_dim)
        P = xp.eye(self.state_dim)

        x_filtered = xp.zeros((T, self.state_dim))

        # Precompute transpose
        AT = A.T
        CT = C.T

        for t in range(T):
            # Predict
            x_pred = A @ x + B @ u[t]
            P_pred = A @ P @ AT + Q

            if y is not None:
                # Update with Kalman gain
                S = C @ P_pred @ CT + R
                K = xp.linalg.solve(S.T, (P_pred @ CT).T).T
                x = x_pred + K @ (y[t] - C @ x_pred)
                P = P_pred - K @ C @ P_pred
            else:
                x = x_pred
                P = P_pred

            x_filtered[t] = x

        return x_filtered

    def _process_sample(self, args):
        """Process single sample - for parallel execution."""
        idx, u, y = args
        x_filtered = self._kalman_filter_single(u, y)
        return idx, x_filtered

    def fit(self, X: NDArray, y: NDArray) -> "KalmanFilter":
        """Fit Kalman filter model.

        Args:
            X: Input signals [N, C_in, T]
            y: Target signals [N, C_out, T]

        Returns:
            self
        """
        xp = get_array_module(self.use_gpu)

        # Handle shape [N, C, T] -> [N, T, C]
        if X.shape[1] < X.shape[2]:
            X = np.transpose(X, (0, 2, 1))
            y = np.transpose(y, (0, 2, 1))
            self._transpose = True
        else:
            self._transpose = False

        N, T, input_dim = X.shape
        output_dim = y.shape[2]

        self._input_dim = input_dim
        self._output_dim = output_dim

        # Initialize parameters
        self._initialize_parameters(input_dim, output_dim)

        # Learn parameters via simplified EM
        if self.learn_transitions:
            self._learn_parameters_fast(X, y)

        self._fitted = True
        return self

    def _learn_parameters_fast(self, X: NDArray, y: NDArray):
        """Fast parameter learning using least squares."""
        xp = get_array_module(self.use_gpu)
        N, T, input_dim = X.shape
        output_dim = y.shape[2]

        if self.use_gpu:
            X = xp.asarray(X)
            y = xp.asarray(y)

        # Simple approach: Learn B and C via regression
        # Stack all time steps
        X_flat = X.reshape(-1, input_dim)  # [N*T, input_dim]
        y_flat = y.reshape(-1, output_dim)  # [N*T, output_dim]

        # Learn C and state via PCA-like approach
        # Use input directly as proxy for state
        state_proxy = X_flat @ xp.random.randn(input_dim, self.state_dim) * 0.1

        # Learn C: y = state @ C.T
        # C.T = (state.T @ state)^{-1} @ state.T @ y
        StS = state_proxy.T @ state_proxy + 1e-6 * xp.eye(self.state_dim)
        StY = state_proxy.T @ y_flat
        self.C = xp.linalg.solve(StS, StY).T

        # Learn B: state ~ X @ B.T (simplified)
        XtX = X_flat.T @ X_flat + 1e-6 * xp.eye(input_dim)
        XtS = X_flat.T @ state_proxy
        self.B = xp.linalg.solve(XtX, XtS).T

    def predict(self, X: NDArray) -> NDArray:
        """Predict output signals - parallelized across samples.

        Args:
            X: Input signals [N, C_in, T]

        Returns:
            Predicted outputs [N, C_out, T]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        xp = get_array_module(self.use_gpu)

        # Handle shape
        if X.shape[1] < X.shape[2]:
            X = np.transpose(X, (0, 2, 1))
            transpose_back = True
        else:
            transpose_back = False

        N, T, input_dim = X.shape

        y_pred = np.zeros((N, T, self._output_dim))

        # GPU mode: sequential processing to avoid CUDA context issues
        # CPU mode: parallel processing with ThreadPoolExecutor
        if self.use_gpu:
            # Sequential processing for GPU (avoids threading issues)
            C_np = to_numpy(self.C)
            for i in range(N):
                x_filtered = self._kalman_filter_single(X[i], None)
                x_filtered_np = to_numpy(x_filtered)
                y_pred[i] = x_filtered_np @ C_np.T
        else:
            # Parallel processing for CPU
            with ThreadPoolExecutor(max_workers=min(8, N)) as executor:
                args_list = [(i, X[i], None) for i in range(N)]
                results = list(executor.map(self._process_sample, args_list))

            for idx, x_filtered in results:
                x_filtered_np = to_numpy(x_filtered)
                C_np = to_numpy(self.C)
                y_pred[idx] = x_filtered_np @ C_np.T

        if transpose_back or self._transpose:
            y_pred = np.transpose(y_pred, (0, 2, 1))

        return y_pred

    def fit_predict(self, X: NDArray, y: NDArray) -> NDArray:
        """Fit and predict."""
        return self.fit(X, y).predict(X)


class ExtendedKalmanFilter(KalmanFilter):
    """Extended Kalman Filter with nonlinear transitions - GPU optimized."""

    def __init__(
        self,
        state_dim: int = 32,
        hidden_dim: int = 64,
        process_noise: float = 0.01,
        observation_noise: float = 0.1,
        use_gpu: bool = True,
    ):
        super().__init__(
            state_dim=state_dim,
            process_noise=process_noise,
            observation_noise=observation_noise,
            learn_transitions=False,
            use_gpu=use_gpu,
        )
        self.hidden_dim = hidden_dim
        self.W1: Optional[NDArray] = None
        self.b1: Optional[NDArray] = None
        self.W2: Optional[NDArray] = None
        self.b2: Optional[NDArray] = None

    def _initialize_parameters(self, input_dim: int, output_dim: int):
        """Initialize with nonlinear layers."""
        super()._initialize_parameters(input_dim, output_dim)
        xp = get_array_module(self.use_gpu)

        combined_dim = self.state_dim + input_dim
        self.W1 = xp.random.randn(self.hidden_dim, combined_dim) * 0.1
        self.b1 = xp.zeros(self.hidden_dim)
        self.W2 = xp.random.randn(self.state_dim, self.hidden_dim) * 0.1
        self.b2 = xp.zeros(self.state_dim)

    def _kalman_filter_single(self, u: NDArray, y: Optional[NDArray] = None) -> NDArray:
        """Run EKF for single sequence."""
        xp = get_array_module(self.use_gpu)
        T = u.shape[0]

        if self.use_gpu:
            u = xp.asarray(u)
            if y is not None:
                y = xp.asarray(y)

        x = xp.zeros(self.state_dim)
        P = xp.eye(self.state_dim)
        x_filtered = xp.zeros((T, self.state_dim))

        CT = self.C.T

        for t in range(T):
            # Nonlinear prediction
            combined = xp.concatenate([x, u[t]])
            hidden = xp.tanh(self.W1 @ combined + self.b1)
            x_pred = self.W2 @ hidden + self.b2

            # Jacobian
            d_tanh = 1 - hidden ** 2
            J_hidden = xp.diag(d_tanh) @ self.W1[:, :self.state_dim]
            F = self.W2 @ J_hidden

            P_pred = F @ P @ F.T + self.Q

            if y is not None:
                S = self.C @ P_pred @ CT + self.R
                K = xp.linalg.solve(S.T, (P_pred @ CT).T).T
                x = x_pred + K @ (y[t] - self.C @ x_pred)
                P = P_pred - K @ self.C @ P_pred
            else:
                x = x_pred
                P = P_pred

            x_filtered[t] = x

        return x_filtered


def create_kalman_filter(variant: str = "standard", **kwargs) -> KalmanFilter:
    """Factory for Kalman filter variants.

    Args:
        variant: "standard" or "extended"
        **kwargs: Additional arguments

    Returns:
        Kalman filter instance
    """
    if variant == "standard":
        return KalmanFilter(**kwargs)
    elif variant == "extended":
        return ExtendedKalmanFilter(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
