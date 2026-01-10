"""
Vector Autoregressive (VAR) Models for Neural Signal Translation
================================================================

VAR models capture temporal dependencies in multi-channel time series.
Standard baseline in neuroscience for modeling neural dynamics.

Model: y[t] = sum_{i=1}^{p} A_i @ y[t-i] + B @ x[t] + c
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


class VARBaseline:
    """Vector Autoregressive model for neural signal translation.

    Models temporal dynamics:
        y[t] = A_1 @ y[t-1] + ... + A_p @ y[t-p] + B @ x[t] + c

    where:
        - A_i: Autoregressive coefficients for lag i
        - B: Input mapping coefficients
        - c: Intercept
        - p: Model order (number of lags)

    Args:
        order: Number of autoregressive lags (p)
        regularization: L2 regularization strength

    Attributes:
        A: Autoregressive coefficients [C_out, C_out * order]
        B: Input mapping [C_out, C_in]
        intercept: Bias term [C_out]

    Example:
        >>> model = VARBaseline(order=5)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        order: int = 5,
        regularization: float = 1e-4,
    ):
        self.order = order
        self.regularization = regularization

        # Model parameters
        self.A: Optional[NDArray] = None
        self.B: Optional[NDArray] = None
        self.intercept: Optional[NDArray] = None

        self._fitted = False
        self._input_shape: Optional[tuple] = None
        self._output_shape: Optional[tuple] = None

    def fit(self, X: NDArray, y: NDArray) -> "VARBaseline":
        """Fit VAR model using ridge regression.

        Constructs design matrix with lagged outputs and current input,
        then solves via regularized least squares.

        Args:
            X: Input signals [N, C_in, T]
            y: Target signals [N, C_out, T]

        Returns:
            self

        Raises:
            ValueError: If input not 3D
        """
        if X.ndim != 3:
            raise ValueError("Expected 3D input [N, C_in, T]")

        N, C_in, T = X.shape
        C_out = y.shape[1]
        p = self.order

        self._input_shape = (C_in, T)
        self._output_shape = (C_out, T)

        # Transpose to [N, T, C] for easier indexing
        X = X.transpose(0, 2, 1)  # [N, T, C_in]
        y = y.transpose(0, 2, 1)  # [N, T, C_out]

        # Number of features: lagged outputs + current input + intercept
        n_y_features = C_out * p
        n_x_features = C_in
        n_features = n_y_features + n_x_features + 1

        # Build design matrix
        n_samples = N * (T - p)
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        targets = np.zeros((n_samples, C_out), dtype=np.float32)

        idx = 0
        for n in range(N):
            for t in range(p, T):
                # Lagged outputs: y[t-p:t] flattened, most recent first
                y_lags = y[n, t-p:t, :].flatten()[::-1]
                features[idx, :n_y_features] = y_lags

                # Current input
                features[idx, n_y_features:n_y_features + n_x_features] = X[n, t, :]

                # Intercept (column of ones)
                features[idx, -1] = 1.0

                # Target
                targets[idx] = y[n, t, :]
                idx += 1

        # Solve via ridge regression: W = (X^T X + lambda*I)^{-1} X^T Y
        XtX = features.T @ features
        XtX_reg = XtX + self.regularization * np.eye(n_features)
        XtY = features.T @ targets

        W = np.linalg.solve(XtX_reg, XtY)

        # Extract parameters
        self.A = W[:n_y_features, :].T  # [C_out, C_out * p]
        self.B = W[n_y_features:n_y_features + n_x_features, :].T  # [C_out, C_in]
        self.intercept = W[-1, :]  # [C_out]

        self._fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict output signals using fitted VAR model.

        Uses autoregressive prediction: each time step depends on
        previous predictions.

        Args:
            X: Input signals [N, C_in, T]

        Returns:
            Predictions [N, C_out, T]

        Raises:
            RuntimeError: If model not fitted
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        N, C_in, T = X.shape
        C_out = self.B.shape[0]
        p = self.order

        # Transpose input
        X = X.transpose(0, 2, 1)  # [N, T, C_in]

        # Initialize output
        y_pred = np.zeros((N, T, C_out), dtype=np.float32)

        # Initialize first p steps using only input mapping
        for t in range(min(p, T)):
            y_pred[:, t, :] = X[:, t, :] @ self.B.T + self.intercept

        # Autoregressive prediction
        for t in range(p, T):
            # Get lagged outputs (most recent first)
            y_lags = y_pred[:, t-p:t, :].reshape(N, -1)[:, ::-1]

            # Predict: y[t] = A @ y_lags + B @ x[t] + c
            y_pred[:, t, :] = (
                y_lags @ self.A.T +
                X[:, t, :] @ self.B.T +
                self.intercept
            )

        # Transpose back to [N, C_out, T]
        return y_pred.transpose(0, 2, 1).astype(np.float32)

    def fit_predict(self, X: NDArray, y: NDArray) -> NDArray:
        """Fit and predict in one step."""
        return self.fit(X, y).predict(X)


class VARExogenous(VARBaseline):
    """VAR with Exogenous inputs (VARX).

    Extended VAR that includes lagged inputs as well:
        y[t] = sum_i A_i @ y[t-i] + sum_j B_j @ x[t-j] + c

    Args:
        order: Number of output lags (p)
        input_order: Number of input lags (q), includes current input
        regularization: L2 regularization

    Example:
        >>> model = VARExogenous(order=5, input_order=3)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        order: int = 5,
        input_order: int = 3,
        regularization: float = 1e-4,
    ):
        super().__init__(order=order, regularization=regularization)
        self.input_order = input_order
        self._max_lag: int = 0
        self._n_x_features: int = 0

    def fit(self, X: NDArray, y: NDArray) -> "VARExogenous":
        """Fit VARX model.

        Args:
            X: Input signals [N, C_in, T]
            y: Target signals [N, C_out, T]

        Returns:
            self
        """
        if X.ndim != 3:
            raise ValueError("Expected 3D input [N, C_in, T]")

        N, C_in, T = X.shape
        C_out = y.shape[1]
        p = self.order
        q = self.input_order
        max_lag = max(p, q)
        self._max_lag = max_lag

        self._input_shape = (C_in, T)
        self._output_shape = (C_out, T)

        # Transpose
        X = X.transpose(0, 2, 1)  # [N, T, C_in]
        y = y.transpose(0, 2, 1)  # [N, T, C_out]

        # Feature dimensions
        n_y_features = C_out * p if p > 0 else 0
        n_x_features = C_in * (q + 1)  # Current + q lags
        self._n_x_features = n_x_features
        n_features = n_y_features + n_x_features + 1

        # Build design matrix
        n_samples = N * (T - max_lag)
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        targets = np.zeros((n_samples, C_out), dtype=np.float32)

        idx = 0
        for n in range(N):
            for t in range(max_lag, T):
                feat_idx = 0

                # Lagged outputs: y[t-p:t]
                if p > 0:
                    y_lags = y[n, t-p:t, :].flatten()[::-1]
                    features[idx, feat_idx:feat_idx + n_y_features] = y_lags
                    feat_idx += n_y_features

                # Lagged + current inputs: x[t-q:t+1]
                x_lags = X[n, t-q:t+1, :].flatten()[::-1]
                features[idx, feat_idx:feat_idx + n_x_features] = x_lags
                feat_idx += n_x_features

                # Intercept
                features[idx, -1] = 1.0

                targets[idx] = y[n, t, :]
                idx += 1

        # Solve via ridge regression
        XtX = features.T @ features
        XtX_reg = XtX + self.regularization * np.eye(n_features)
        XtY = features.T @ targets

        W = np.linalg.solve(XtX_reg, XtY)

        # Extract parameters
        if p > 0:
            self.A = W[:n_y_features, :].T  # [C_out, C_out * p]
        else:
            self.A = None
        self.B = W[n_y_features:n_y_features + n_x_features, :].T  # [C_out, C_in * (q+1)]
        self.intercept = W[-1, :]

        self._fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict using VARX model.

        Args:
            X: Input signals [N, C_in, T]

        Returns:
            Predictions [N, C_out, T]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        N, C_in, T = X.shape
        C_out = self.intercept.shape[0]
        p = self.order
        q = self.input_order
        max_lag = self._max_lag

        # Transpose
        X = X.transpose(0, 2, 1)  # [N, T, C_in]

        # Initialize output
        y_pred = np.zeros((N, T, C_out), dtype=np.float32)

        # Initialize first max_lag steps
        for t in range(max_lag):
            # Collect available input lags
            start = max(0, t - q)
            x_feat = X[:, start:t+1, :].reshape(N, -1)

            # Pad if needed
            if x_feat.shape[1] < self._n_x_features:
                pad_size = self._n_x_features - x_feat.shape[1]
                x_feat = np.pad(x_feat, ((0, 0), (pad_size, 0)), mode='edge')

            # Reverse for most-recent-first ordering
            x_feat = x_feat[:, ::-1]

            y_pred[:, t, :] = x_feat @ self.B.T + self.intercept

        # Full prediction
        for t in range(max_lag, T):
            # Input features (most recent first)
            x_lags = X[:, t-q:t+1, :].reshape(N, -1)[:, ::-1]
            pred = x_lags @ self.B.T + self.intercept

            # Add AR component if present
            if self.A is not None and p > 0:
                y_lags = y_pred[:, t-p:t, :].reshape(N, -1)[:, ::-1]
                pred = pred + y_lags @ self.A.T

            y_pred[:, t, :] = pred

        return y_pred.transpose(0, 2, 1).astype(np.float32)
