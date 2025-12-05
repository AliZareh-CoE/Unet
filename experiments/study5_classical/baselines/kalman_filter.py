"""
Kalman Filter for Neural Signal Translation
============================================

State-space model for temporal signal translation.
Provides principled handling of temporal dynamics and uncertainty.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class KalmanFilter:
    """Kalman filter for neural signal translation.

    Models the system as:
        x_{t+1} = A @ x_t + w_t     (state transition)
        y_t = C @ x_t + v_t         (observation)

    Where:
        x_t: Hidden state
        y_t: Observed output (target signal)
        A: State transition matrix
        C: Observation matrix
        w_t ~ N(0, Q): Process noise
        v_t ~ N(0, R): Observation noise

    For translation, we augment the state with input signal and
    learn mappings from input to output through the state.

    Args:
        state_dim: Dimension of hidden state
        process_noise: Process noise variance (Q)
        observation_noise: Observation noise variance (R)
        learn_transitions: If True, learn A and C from data
    """

    def __init__(
        self,
        state_dim: int = 32,
        process_noise: float = 0.01,
        observation_noise: float = 0.1,
        learn_transitions: bool = True,
    ):
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.learn_transitions = learn_transitions

        # Model parameters
        self.A: Optional[NDArray] = None  # State transition
        self.B: Optional[NDArray] = None  # Input to state
        self.C: Optional[NDArray] = None  # State to output
        self.Q: Optional[NDArray] = None  # Process noise covariance
        self.R: Optional[NDArray] = None  # Observation noise covariance

        self._fitted = False

    def _initialize_parameters(
        self,
        input_dim: int,
        output_dim: int,
    ):
        """Initialize model parameters."""
        # State transition (random orthogonal for stability)
        random_mat = np.random.randn(self.state_dim, self.state_dim)
        U, _, Vt = np.linalg.svd(random_mat)
        self.A = 0.95 * U @ Vt  # Slightly contractive

        # Input to state
        self.B = np.random.randn(self.state_dim, input_dim) * 0.1

        # State to output
        self.C = np.random.randn(output_dim, self.state_dim) * 0.1

        # Noise covariances
        self.Q = self.process_noise * np.eye(self.state_dim)
        self.R = self.observation_noise * np.eye(output_dim)

    def _kalman_filter(
        self,
        u: NDArray,
        y: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Run Kalman filter forward pass.

        Args:
            u: Input sequence [T, input_dim]
            y: Optional target sequence [T, output_dim] (for filtering)

        Returns:
            x_filtered: Filtered states [T, state_dim]
            P_filtered: Filtered covariances [T, state_dim, state_dim]
        """
        T, input_dim = u.shape
        output_dim = self.C.shape[0]

        # Initialize
        x = np.zeros(self.state_dim)
        P = np.eye(self.state_dim)

        x_filtered = np.zeros((T, self.state_dim))
        P_filtered = np.zeros((T, self.state_dim, self.state_dim))

        for t in range(T):
            # Predict
            x_pred = self.A @ x + self.B @ u[t]
            P_pred = self.A @ P @ self.A.T + self.Q

            if y is not None:
                # Update (Kalman gain)
                S = self.C @ P_pred @ self.C.T + self.R
                K = P_pred @ self.C.T @ np.linalg.inv(S)

                # Innovation
                innovation = y[t] - self.C @ x_pred

                # Update state
                x = x_pred + K @ innovation
                P = (np.eye(self.state_dim) - K @ self.C) @ P_pred
            else:
                x = x_pred
                P = P_pred

            x_filtered[t] = x
            P_filtered[t] = P

        return x_filtered, P_filtered

    def _kalman_smoother(
        self,
        x_filtered: NDArray,
        P_filtered: NDArray,
    ) -> NDArray:
        """Run Kalman smoother (backward pass).

        Args:
            x_filtered: Filtered states [T, state_dim]
            P_filtered: Filtered covariances [T, state_dim, state_dim]

        Returns:
            x_smoothed: Smoothed states [T, state_dim]
        """
        T = x_filtered.shape[0]
        x_smoothed = np.zeros_like(x_filtered)
        x_smoothed[-1] = x_filtered[-1]

        for t in range(T - 2, -1, -1):
            P_pred = self.A @ P_filtered[t] @ self.A.T + self.Q
            J = P_filtered[t] @ self.A.T @ np.linalg.inv(P_pred)
            x_smoothed[t] = x_filtered[t] + J @ (x_smoothed[t + 1] - self.A @ x_filtered[t])

        return x_smoothed

    def _learn_parameters_em(
        self,
        u_all: NDArray,
        y_all: NDArray,
        n_iterations: int = 10,
    ):
        """Learn parameters using EM algorithm.

        Args:
            u_all: All input sequences [N, T, input_dim]
            y_all: All target sequences [N, T, output_dim]
            n_iterations: Number of EM iterations
        """
        N, T, output_dim = y_all.shape
        input_dim = u_all.shape[2]

        for iteration in range(n_iterations):
            # E-step: Run filter and smoother for all sequences
            all_x_smoothed = []
            for n in range(N):
                x_filt, P_filt = self._kalman_filter(u_all[n], y_all[n])
                x_smooth = self._kalman_smoother(x_filt, P_filt)
                all_x_smoothed.append(x_smooth)

            all_x_smoothed = np.array(all_x_smoothed)  # [N, T, state_dim]

            # M-step: Update parameters

            # Update A (state transition)
            sum_xx_prev = np.zeros((self.state_dim, self.state_dim))
            sum_xx_next_prev = np.zeros((self.state_dim, self.state_dim))
            sum_Bu_x = np.zeros((self.state_dim, self.state_dim))

            for n in range(N):
                for t in range(T - 1):
                    x_t = all_x_smoothed[n, t]
                    x_t1 = all_x_smoothed[n, t + 1]
                    Bu_t = self.B @ u_all[n, t]

                    sum_xx_prev += np.outer(x_t, x_t)
                    sum_xx_next_prev += np.outer(x_t1 - Bu_t, x_t)

            # Regularize and solve
            sum_xx_prev += 1e-6 * np.eye(self.state_dim)
            self.A = sum_xx_next_prev @ np.linalg.inv(sum_xx_prev)

            # Update B (input mapping)
            sum_uu = np.zeros((input_dim, input_dim))
            sum_xu = np.zeros((self.state_dim, input_dim))

            for n in range(N):
                for t in range(T - 1):
                    x_t1 = all_x_smoothed[n, t + 1]
                    x_t = all_x_smoothed[n, t]
                    u_t = u_all[n, t]

                    residual = x_t1 - self.A @ x_t
                    sum_uu += np.outer(u_t, u_t)
                    sum_xu += np.outer(residual, u_t)

            sum_uu += 1e-6 * np.eye(input_dim)
            self.B = sum_xu @ np.linalg.inv(sum_uu)

            # Update C (observation mapping)
            sum_xx = np.zeros((self.state_dim, self.state_dim))
            sum_yx = np.zeros((output_dim, self.state_dim))

            for n in range(N):
                for t in range(T):
                    x_t = all_x_smoothed[n, t]
                    y_t = y_all[n, t]

                    sum_xx += np.outer(x_t, x_t)
                    sum_yx += np.outer(y_t, x_t)

            sum_xx += 1e-6 * np.eye(self.state_dim)
            self.C = sum_yx @ np.linalg.inv(sum_xx)

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        n_em_iterations: int = 10,
    ) -> "KalmanFilter":
        """Fit Kalman filter model.

        Args:
            X: Input signals [N, C_in, T] or [N, T, C_in]
            y: Target signals [N, C_out, T] or [N, T, C_out]
            n_em_iterations: Number of EM iterations for learning

        Returns:
            self
        """
        # Ensure shape is [N, T, C]
        if X.ndim == 3 and X.shape[1] < X.shape[2]:
            X = X.transpose(0, 2, 1)
            y = y.transpose(0, 2, 1)
            self._transpose = True
        else:
            self._transpose = False

        N, T, input_dim = X.shape
        _, _, output_dim = y.shape

        self._input_dim = input_dim
        self._output_dim = output_dim

        # Initialize parameters
        self._initialize_parameters(input_dim, output_dim)

        # Learn parameters
        if self.learn_transitions:
            self._learn_parameters_em(X, y, n_em_iterations)

        self._fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict output signals.

        Args:
            X: Input signals [N, C_in, T] or [N, T, C_in]

        Returns:
            Predicted outputs [N, C_out, T]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        # Handle shape
        if X.ndim == 3 and X.shape[1] < X.shape[2]:
            X = X.transpose(0, 2, 1)
            transpose_back = True
        else:
            transpose_back = False

        N, T, input_dim = X.shape

        # Predict for each sequence
        y_pred = np.zeros((N, T, self._output_dim))

        for n in range(N):
            x_filtered, _ = self._kalman_filter(X[n], y=None)
            y_pred[n] = x_filtered @ self.C.T

        # Transpose back if needed
        if transpose_back or self._transpose:
            y_pred = y_pred.transpose(0, 2, 1)

        return y_pred

    def fit_predict(self, X: NDArray, y: NDArray) -> NDArray:
        """Fit and predict in one step."""
        return self.fit(X, y).predict(X)


class ExtendedKalmanFilter(KalmanFilter):
    """Extended Kalman Filter with nonlinear state transitions.

    Uses a simple neural network for nonlinear dynamics:
        x_{t+1} = f(A @ x_t + B @ u_t)

    where f is a simple nonlinearity (tanh).

    Args:
        state_dim: Hidden state dimension
        hidden_dim: Hidden layer dimension for nonlinearity
        process_noise: Process noise variance
        observation_noise: Observation noise variance
    """

    def __init__(
        self,
        state_dim: int = 32,
        hidden_dim: int = 64,
        process_noise: float = 0.01,
        observation_noise: float = 0.1,
    ):
        super().__init__(
            state_dim=state_dim,
            process_noise=process_noise,
            observation_noise=observation_noise,
            learn_transitions=False,  # We'll use gradient descent
        )
        self.hidden_dim = hidden_dim

        # Nonlinear layers
        self.W1: Optional[NDArray] = None
        self.b1: Optional[NDArray] = None
        self.W2: Optional[NDArray] = None
        self.b2: Optional[NDArray] = None

    def _initialize_parameters(
        self,
        input_dim: int,
        output_dim: int,
    ):
        """Initialize with nonlinear layers."""
        super()._initialize_parameters(input_dim, output_dim)

        # Nonlinear transformation layers
        combined_dim = self.state_dim + input_dim
        self.W1 = np.random.randn(self.hidden_dim, combined_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.state_dim, self.hidden_dim) * 0.1
        self.b2 = np.zeros(self.state_dim)

    def _nonlinear_transition(
        self,
        x: NDArray,
        u: NDArray,
    ) -> NDArray:
        """Nonlinear state transition."""
        combined = np.concatenate([x, u])
        hidden = np.tanh(self.W1 @ combined + self.b1)
        x_new = self.W2 @ hidden + self.b2
        return x_new

    def _jacobian_transition(
        self,
        x: NDArray,
        u: NDArray,
    ) -> NDArray:
        """Compute Jacobian of transition function w.r.t. x."""
        combined = np.concatenate([x, u])
        hidden = np.tanh(self.W1 @ combined + self.b1)
        d_tanh = 1 - hidden ** 2  # Derivative of tanh

        # Jacobian: dh/dx * W2
        J_hidden = np.diag(d_tanh) @ self.W1[:, : self.state_dim]
        J = self.W2 @ J_hidden

        return J

    def _ekf_filter(
        self,
        u: NDArray,
        y: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Run Extended Kalman filter."""
        T, input_dim = u.shape

        x = np.zeros(self.state_dim)
        P = np.eye(self.state_dim)

        x_filtered = np.zeros((T, self.state_dim))
        P_filtered = np.zeros((T, self.state_dim, self.state_dim))

        for t in range(T):
            # Predict with nonlinear transition
            x_pred = self._nonlinear_transition(x, u[t])
            F = self._jacobian_transition(x, u[t])
            P_pred = F @ P @ F.T + self.Q

            if y is not None:
                # Update
                S = self.C @ P_pred @ self.C.T + self.R
                K = P_pred @ self.C.T @ np.linalg.inv(S)
                innovation = y[t] - self.C @ x_pred
                x = x_pred + K @ innovation
                P = (np.eye(self.state_dim) - K @ self.C) @ P_pred
            else:
                x = x_pred
                P = P_pred

            x_filtered[t] = x
            P_filtered[t] = P

        return x_filtered, P_filtered

    def predict(self, X: NDArray) -> NDArray:
        """Predict using EKF."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        if X.ndim == 3 and X.shape[1] < X.shape[2]:
            X = X.transpose(0, 2, 1)
            transpose_back = True
        else:
            transpose_back = False

        N, T, input_dim = X.shape
        y_pred = np.zeros((N, T, self._output_dim))

        for n in range(N):
            x_filtered, _ = self._ekf_filter(X[n], y=None)
            y_pred[n] = x_filtered @ self.C.T

        if transpose_back or self._transpose:
            y_pred = y_pred.transpose(0, 2, 1)

        return y_pred


def create_kalman_filter(
    variant: str = "standard",
    **kwargs,
) -> KalmanFilter:
    """Factory function for Kalman filter variants.

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
