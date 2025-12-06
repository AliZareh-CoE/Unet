"""
Vector Autoregressive (VAR) Model for Neural Signal Translation - GPU Accelerated
==================================================================================

VAR is a standard baseline in neuroscience for multi-channel time series.
Models temporal dependencies between channels. GPU optimized with CuPy.
"""

from __future__ import annotations

from typing import Optional
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


class VARModel:
    """Vector Autoregressive model for signal translation - GPU optimized.

    Models: y_t = sum_{i=1}^{p} A_i @ y_{t-i} + B @ x_t + noise

    Args:
        order: Number of lags (p)
        regularization: L2 regularization strength
        use_gpu: Use GPU acceleration
    """

    def __init__(
        self,
        order: int = 5,
        regularization: float = 1e-4,
        use_gpu: bool = True,
    ):
        self.order = order
        self.regularization = regularization
        self.use_gpu = use_gpu and HAS_CUPY
        self.A: Optional[NDArray] = None
        self.B: Optional[NDArray] = None
        self.intercept: Optional[NDArray] = None
        self._fitted = False

    def fit(self, X: NDArray, y: NDArray) -> "VARModel":
        """Fit VAR model - GPU accelerated.

        Args:
            X: Input signals [N, C_in, T]
            y: Target signals [N, C_out, T]

        Returns:
            self
        """
        xp = get_array_module(self.use_gpu)

        # Handle shape [N, C, T] -> [N, T, C]
        if X.ndim == 3:
            N, C_in, T = X.shape
            C_out = y.shape[1]
            X = np.transpose(X, (0, 2, 1))
            y = np.transpose(y, (0, 2, 1))
        else:
            raise ValueError("Expected 3D input [N, C, T]")

        self._input_shape = (C_in, T)
        self._output_shape = (C_out, T)

        # Move to GPU
        if self.use_gpu:
            X = xp.asarray(X)
            y = xp.asarray(y)

        p = self.order
        n_features = C_out * p + C_in

        # Build design matrix efficiently on GPU
        n_samples = N * (T - p)
        features = xp.zeros((n_samples, n_features))
        targets = xp.zeros((n_samples, C_out))

        idx = 0
        for n in range(N):
            for t in range(p, T):
                # Lagged outputs (vectorized)
                lags = y[n, t - p:t, :].flatten()[::-1]
                features[idx, :C_out * p] = lags
                features[idx, C_out * p:] = X[n, t, :]
                targets[idx] = y[n, t, :]
                idx += 1

        # Add intercept column
        features = xp.hstack([features, xp.ones((n_samples, 1))])

        # Ridge regression on GPU: W = (X'X + λI)^{-1} X'Y
        XtX = features.T @ features
        XtX += self.regularization * xp.eye(XtX.shape[0])
        XtY = features.T @ targets

        W = xp.linalg.solve(XtX, XtY)

        # Extract parameters
        self.A = W[:C_out * p, :].T
        self.B = W[C_out * p:C_out * p + C_in, :].T
        self.intercept = W[-1, :]

        self._fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict output signals - GPU accelerated.

        Args:
            X: Input signals [N, C_in, T]

        Returns:
            Predictions [N, C_out, T]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        xp = get_array_module(self.use_gpu)

        N, C_in, T = X.shape
        C_out = to_numpy(self.B).shape[0]
        p = self.order

        # Move to GPU
        if self.use_gpu:
            X = xp.asarray(np.transpose(X, (0, 2, 1)))
            A, B, intercept = self.A, self.B, self.intercept
        else:
            X = np.transpose(X, (0, 2, 1))
            A = to_numpy(self.A)
            B = to_numpy(self.B)
            intercept = to_numpy(self.intercept)

        y_pred = xp.zeros((N, T, C_out))

        # Initialize first p steps
        for t in range(p):
            y_pred[:, t, :] = X[:, t, :] @ B.T + intercept

        # Predict using AR (vectorized across samples)
        for t in range(p, T):
            lags = y_pred[:, t - p:t, :].reshape(N, -1)
            # Reverse to get most recent first
            lags = lags[:, ::-1]
            y_pred[:, t, :] = lags @ A.T + X[:, t, :] @ B.T + intercept

        # Move back to CPU and transpose
        y_pred = to_numpy(y_pred)
        return np.transpose(y_pred, (0, 2, 1))

    def fit_predict(self, X: NDArray, y: NDArray) -> NDArray:
        """Fit and predict."""
        return self.fit(X, y).predict(X)


class VARExogenous(VARModel):
    """VAR with exogenous inputs (VARX) - GPU optimized.

    Args:
        order: Number of lags for output
        input_order: Number of lags for input
        regularization: L2 regularization
        use_gpu: Use GPU acceleration
    """

    def __init__(
        self,
        order: int = 5,
        input_order: int = 3,
        regularization: float = 1e-4,
        use_gpu: bool = True,
    ):
        super().__init__(order=order, regularization=regularization, use_gpu=use_gpu)
        self.input_order = input_order

    def fit(self, X: NDArray, y: NDArray) -> "VARExogenous":
        """Fit VARX model - GPU accelerated."""
        xp = get_array_module(self.use_gpu)

        if X.ndim == 3:
            N, C_in, T = X.shape
            C_out = y.shape[1]
            X = np.transpose(X, (0, 2, 1))
            y = np.transpose(y, (0, 2, 1))
        else:
            raise ValueError("Expected 3D input")

        self._input_shape = (C_in, T)
        self._output_shape = (C_out, T)

        if self.use_gpu:
            X = xp.asarray(X)
            y = xp.asarray(y)

        p = self.order
        q = self.input_order
        max_lag = max(p, q)

        n_y_feats = C_out * p
        n_x_feats = C_in * (q + 1)
        n_features = n_y_feats + n_x_feats

        n_samples = N * (T - max_lag)
        features = xp.zeros((n_samples, n_features))
        targets = xp.zeros((n_samples, C_out))

        idx = 0
        for n in range(N):
            for t in range(max_lag, T):
                # Lagged outputs
                if p > 0:
                    y_lags = y[n, t - p:t, :].flatten()[::-1]
                    features[idx, :n_y_feats] = y_lags
                # Lagged + current inputs
                x_lags = X[n, t - q:t + 1, :].flatten()[::-1]
                features[idx, n_y_feats:n_y_feats + len(x_lags)] = x_lags
                targets[idx] = y[n, t, :]
                idx += 1

        features = xp.hstack([features, xp.ones((n_samples, 1))])

        XtX = features.T @ features + self.regularization * xp.eye(features.shape[1])
        W = xp.linalg.solve(XtX, features.T @ targets)

        self.A = W[:n_y_feats, :].T if p > 0 else None
        self.B = W[n_y_feats:n_y_feats + n_x_feats, :].T
        self.intercept = W[-1, :]
        self._max_lag = max_lag
        self._n_x_feats = n_x_feats

        self._fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict with VARX - GPU accelerated."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        xp = get_array_module(self.use_gpu)

        N, C_in, T = X.shape
        C_out = to_numpy(self.intercept).shape[0]
        p = self.order
        q = self.input_order
        max_lag = self._max_lag

        if self.use_gpu:
            X = xp.asarray(np.transpose(X, (0, 2, 1)))
            A = self.A
            B = self.B
            intercept = self.intercept
        else:
            X = np.transpose(X, (0, 2, 1))
            A = to_numpy(self.A) if self.A is not None else None
            B = to_numpy(self.B)
            intercept = to_numpy(self.intercept)

        y_pred = xp.zeros((N, T, C_out))

        # Initialize
        for t in range(max_lag):
            start = max(0, t - q)
            x_feat = X[:, start:t + 1, :].reshape(N, -1)
            if x_feat.shape[1] < self._n_x_feats:
                pad_size = self._n_x_feats - x_feat.shape[1]
                x_feat = xp.pad(x_feat, ((0, 0), (pad_size, 0)))
            x_feat = x_feat[:, ::-1]
            y_pred[:, t, :] = x_feat @ B.T + intercept

        for t in range(max_lag, T):
            x_lags = X[:, t - q:t + 1, :].reshape(N, -1)[:, ::-1]
            pred = x_lags @ B.T + intercept

            if A is not None and p > 0:
                y_lags = y_pred[:, t - p:t, :].reshape(N, -1)[:, ::-1]
                pred = pred + y_lags @ A.T

            y_pred[:, t, :] = pred

        y_pred = to_numpy(y_pred)
        return np.transpose(y_pred, (0, 2, 1))


def create_var_model(variant: str = "standard", **kwargs) -> VARModel:
    """Factory for VAR variants.

    Args:
        variant: "standard" or "exogenous"
        **kwargs: Additional arguments

    Returns:
        VAR model instance
    """
    if variant == "standard":
        return VARModel(**kwargs)
    elif variant == "exogenous":
        return VARExogenous(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
