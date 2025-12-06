"""
Vector Autoregressive (VAR) Model for Neural Signal Translation
================================================================

VAR is a standard baseline in neuroscience for multi-channel time series.
Models temporal dependencies between channels.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from numpy.typing import NDArray


class VARModel:
    """Vector Autoregressive model for signal translation.

    Models: y_t = sum_{i=1}^{p} A_i @ y_{t-i} + B @ x_t + noise

    Args:
        order: Number of lags (p)
        regularization: L2 regularization strength
    """

    def __init__(
        self,
        order: int = 5,
        regularization: float = 1e-4,
    ):
        self.order = order
        self.regularization = regularization
        self.A: Optional[NDArray] = None  # [output_dim, output_dim * order]
        self.B: Optional[NDArray] = None  # [output_dim, input_dim]
        self.intercept: Optional[NDArray] = None
        self._fitted = False

    def fit(self, X: NDArray, y: NDArray) -> "VARModel":
        """Fit VAR model.

        Args:
            X: Input signals [N, C_in, T]
            y: Target signals [N, C_out, T]

        Returns:
            self
        """
        # Handle shape [N, C, T] -> work with [N, T, C]
        if X.ndim == 3:
            N, C_in, T = X.shape
            C_out = y.shape[1]
            X = X.transpose(0, 2, 1)  # [N, T, C_in]
            y = y.transpose(0, 2, 1)  # [N, T, C_out]
        else:
            raise ValueError("Expected 3D input [N, C, T]")

        self._input_shape = (C_in, T)
        self._output_shape = (C_out, T)

        # Build design matrix with lagged y and current x
        # For each time t, features are: [y_{t-1}, y_{t-2}, ..., y_{t-p}, x_t]
        p = self.order
        n_features = C_out * p + C_in

        # Collect all valid time points
        all_features = []
        all_targets = []

        for n in range(N):
            for t in range(p, T):
                # Lagged outputs
                lags = y[n, t - p : t, :].flatten()[::-1]  # Most recent first
                # Current input
                feat = np.concatenate([lags, X[n, t, :]])
                all_features.append(feat)
                all_targets.append(y[n, t, :])

        features = np.array(all_features)  # [N_samples, n_features]
        targets = np.array(all_targets)    # [N_samples, C_out]

        # Add intercept
        features = np.hstack([features, np.ones((features.shape[0], 1))])

        # Ridge regression: W = (X'X + λI)^{-1} X'Y
        XtX = features.T @ features
        XtX += self.regularization * np.eye(XtX.shape[0])
        XtY = features.T @ targets

        W = np.linalg.solve(XtX, XtY)  # [n_features + 1, C_out]

        # Extract parameters
        self.A = W[:C_out * p, :].T  # [C_out, C_out * p]
        self.B = W[C_out * p : C_out * p + C_in, :].T  # [C_out, C_in]
        self.intercept = W[-1, :]  # [C_out]

        self._fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict output signals.

        Args:
            X: Input signals [N, C_in, T]

        Returns:
            Predictions [N, C_out, T]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        N, C_in, T = X.shape
        C_out = self.B.shape[0]
        p = self.order

        X = X.transpose(0, 2, 1)  # [N, T, C_in]
        y_pred = np.zeros((N, T, C_out))

        # Initialize first p steps with zeros or simple prediction
        for t in range(p):
            y_pred[:, t, :] = X[:, t, :] @ self.B.T + self.intercept

        # Predict using AR
        for t in range(p, T):
            # Lagged predictions
            lags = y_pred[:, t - p : t, :].reshape(N, -1)[:, ::-1]
            # VAR prediction
            y_pred[:, t, :] = (
                lags @ self.A.T +
                X[:, t, :] @ self.B.T +
                self.intercept
            )

        return y_pred.transpose(0, 2, 1)  # [N, C_out, T]

    def fit_predict(self, X: NDArray, y: NDArray) -> NDArray:
        """Fit and predict."""
        return self.fit(X, y).predict(X)


class VARExogenous(VARModel):
    """VAR with exogenous inputs (VARX).

    Better handles the input-output translation by treating
    input as exogenous variable with its own lags.

    Args:
        order: Number of lags for output
        input_order: Number of lags for input (0 = current only)
        regularization: L2 regularization
    """

    def __init__(
        self,
        order: int = 5,
        input_order: int = 3,
        regularization: float = 1e-4,
    ):
        super().__init__(order=order, regularization=regularization)
        self.input_order = input_order

    def fit(self, X: NDArray, y: NDArray) -> "VARExogenous":
        """Fit VARX model."""
        if X.ndim == 3:
            N, C_in, T = X.shape
            C_out = y.shape[1]
            X = X.transpose(0, 2, 1)
            y = y.transpose(0, 2, 1)
        else:
            raise ValueError("Expected 3D input")

        self._input_shape = (C_in, T)
        self._output_shape = (C_out, T)

        p = self.order
        q = self.input_order
        max_lag = max(p, q)

        all_features = []
        all_targets = []

        for n in range(N):
            for t in range(max_lag, T):
                # Lagged outputs
                y_lags = y[n, t - p : t, :].flatten()[::-1] if p > 0 else np.array([])
                # Lagged + current inputs
                x_lags = X[n, t - q : t + 1, :].flatten()[::-1]
                feat = np.concatenate([y_lags, x_lags])
                all_features.append(feat)
                all_targets.append(y[n, t, :])

        features = np.array(all_features)
        targets = np.array(all_targets)
        features = np.hstack([features, np.ones((features.shape[0], 1))])

        XtX = features.T @ features + self.regularization * np.eye(features.shape[1])
        W = np.linalg.solve(XtX, features.T @ targets)

        n_y_feats = C_out * p
        n_x_feats = C_in * (q + 1)

        self.A = W[:n_y_feats, :].T if p > 0 else None
        self.B = W[n_y_feats : n_y_feats + n_x_feats, :].T
        self.intercept = W[-1, :]
        self._max_lag = max_lag

        self._fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict with VARX."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        N, C_in, T = X.shape
        C_out = self.intercept.shape[0]
        p = self.order
        q = self.input_order
        max_lag = self._max_lag

        X = X.transpose(0, 2, 1)
        y_pred = np.zeros((N, T, C_out))

        # Initialize
        for t in range(max_lag):
            x_feat = X[:, max(0, t - q) : t + 1, :].reshape(N, -1)
            if x_feat.shape[1] < C_in * (q + 1):
                x_feat = np.pad(x_feat, ((0, 0), (C_in * (q + 1) - x_feat.shape[1], 0)))
            y_pred[:, t, :] = x_feat @ self.B.T + self.intercept

        for t in range(max_lag, T):
            y_lags = y_pred[:, t - p : t, :].reshape(N, -1)[:, ::-1] if p > 0 else np.zeros((N, 0))
            x_lags = X[:, t - q : t + 1, :].reshape(N, -1)[:, ::-1]

            pred = x_lags @ self.B.T + self.intercept
            if self.A is not None:
                pred += y_lags @ self.A.T
            y_pred[:, t, :] = pred

        return y_pred.transpose(0, 2, 1)


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
