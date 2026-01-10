"""
Ridge Regression Baselines for Neural Signal Translation
=========================================================

L2-regularized linear regression for mapping between neural signals.
Provides a simple but effective baseline.

Solution: W = (X^T X + alpha * I)^{-1} X^T Y
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class RidgeBaseline:
    """Ridge regression for pointwise neural signal translation.

    Maps input to output at each time point independently:
        y[t] = X[t] @ W + b

    Args:
        alpha: L2 regularization strength (larger = more regularization)
        fit_intercept: Whether to fit intercept term
        normalize: Whether to normalize features before fitting

    Attributes:
        W: Weight matrix [C_in, C_out]
        b: Bias vector [C_out]

    Example:
        >>> model = RidgeBaseline(alpha=1.0)
        >>> model.fit(X_train, y_train)  # X: [N, C, T]
        >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize: bool = False,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize

        # Model parameters
        self.W: Optional[NDArray] = None
        self.b: Optional[NDArray] = None

        # Normalization statistics
        self._X_mean: Optional[NDArray] = None
        self._X_std: Optional[NDArray] = None
        self._fitted = False
        self._orig_shape: Optional[Tuple] = None

    def _preprocess(self, X: NDArray, fit: bool = False) -> NDArray:
        """Preprocess input data (centering and optional normalization).

        Args:
            X: Input features [N, D]
            fit: Whether to compute statistics (True during training)

        Returns:
            Preprocessed features
        """
        if fit:
            self._X_mean = np.mean(X, axis=0, keepdims=True) if self.fit_intercept else np.zeros((1, X.shape[1]))
            if self.normalize:
                self._X_std = np.std(X, axis=0, keepdims=True)
                self._X_std[self._X_std < 1e-8] = 1.0
            else:
                self._X_std = np.ones((1, X.shape[1]))

        return (X - self._X_mean) / self._X_std

    def fit(self, X: NDArray, y: NDArray) -> "RidgeBaseline":
        """Fit ridge regression model.

        Solves: W = (X^T X + alpha * I)^{-1} X^T Y

        Args:
            X: Input features [N, C_in, T] or [N, D_in]
            y: Target values [N, C_out, T] or [N, D_out]

        Returns:
            self
        """
        # Handle 3D tensors: [N, C, T] -> [N*T, C]
        self._orig_shape = None
        if X.ndim == 3:
            N, C_in, T = X.shape
            _, C_out, _ = y.shape
            self._orig_shape = (N, C_in, T, C_out)
            # Reshape: [N, C, T] -> [N, T, C] -> [N*T, C]
            X = X.transpose(0, 2, 1).reshape(N * T, C_in)
            y = y.transpose(0, 2, 1).reshape(N * T, C_out)

        # Preprocess
        X = self._preprocess(X, fit=True)

        D_in = X.shape[1]
        D_out = y.shape[1]

        # Center targets if using intercept
        if self.fit_intercept:
            y_mean = np.mean(y, axis=0, keepdims=True)
            y_centered = y - y_mean
        else:
            y_mean = np.zeros((1, D_out))
            y_centered = y

        # Closed-form solution: W = (X^T X + alpha*I)^{-1} X^T y
        XtX = X.T @ X
        XtX_reg = XtX + self.alpha * np.eye(D_in)
        Xty = X.T @ y_centered

        self.W = np.linalg.solve(XtX_reg, Xty)

        # Compute intercept
        if self.fit_intercept:
            self.b = y_mean.flatten() - (self._X_mean / self._X_std).flatten() @ self.W
        else:
            self.b = np.zeros(D_out)

        self._fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict using fitted model.

        Args:
            X: Input features [N, C_in, T] or [N, D_in]

        Returns:
            Predictions [N, C_out, T] or [N, D_out]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Handle 3D
        reshape_back = False
        if X.ndim == 3:
            N, C_in, T = X.shape
            X = X.transpose(0, 2, 1).reshape(N * T, C_in)
            reshape_back = True

        # Preprocess and predict
        X = self._preprocess(X, fit=False)
        y = X @ self.W + self.b

        # Reshape back if needed
        if reshape_back and self._orig_shape is not None:
            _, _, _, C_out = self._orig_shape
            # y: [N*T, C_out] -> [N, T, C_out] -> [N, C_out, T]
            y = y.reshape(N, T, C_out).transpose(0, 2, 1)

        return y.astype(np.float32)

    def get_weights(self) -> Tuple[NDArray, NDArray]:
        """Get model weights.

        Returns:
            W: Weight matrix [D_in, D_out]
            b: Bias vector [D_out]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
        return self.W, self.b


class RidgeTemporal(RidgeBaseline):
    """Ridge regression with temporal context windows.

    Uses sliding window to incorporate temporal context:
        y[t] = concat(X[t-w:t+w]) @ W + b

    Args:
        alpha: L2 regularization strength
        window_size: Context window radius (total window = 2*w + 1)
        stride: Prediction stride (1 = predict every time point)
        fit_intercept: Whether to fit intercept

    Example:
        >>> model = RidgeTemporal(alpha=1.0, window_size=10)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        window_size: int = 10,
        stride: int = 1,
        fit_intercept: bool = True,
    ):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept)
        self.window_size = window_size
        self.stride = stride
        self._output_channels: Optional[int] = None

    def _create_temporal_features(self, X: NDArray) -> NDArray:
        """Create temporal feature matrix using sliding windows.

        Args:
            X: Input [N, C, T]

        Returns:
            Features [N * n_windows, C * window_length]
        """
        N, C, T = X.shape
        w = self.window_size
        window_len = 2 * w + 1

        # Pad signal with edge values
        X_padded = np.pad(X, ((0, 0), (0, 0), (w, w)), mode="edge")

        # Extract windows
        windows = []
        for t in range(0, T, self.stride):
            # Window: [N, C, window_len] -> [N, C * window_len]
            window = X_padded[:, :, t:t + window_len]
            windows.append(window.reshape(N, -1))

        # Stack: [n_windows, N, features] -> [N * n_windows, features]
        features = np.stack(windows, axis=1)  # [N, n_windows, features]
        return features.reshape(-1, features.shape[-1])

    def fit(self, X: NDArray, y: NDArray) -> "RidgeTemporal":
        """Fit temporal ridge regression.

        Args:
            X: Input signals [N, C_in, T]
            y: Target signals [N, C_out, T]

        Returns:
            self
        """
        if X.ndim != 3:
            raise ValueError("Expected 3D input [N, C, T]")

        N, C_in, T = X.shape
        _, C_out, _ = y.shape
        self._output_channels = C_out
        self._input_shape = (N, C_in, T)

        # Create temporal features
        X_feat = self._create_temporal_features(X)

        # Create matching targets (subsample to match windows)
        y_samples = []
        for t in range(0, T, self.stride):
            y_samples.append(y[:, :, t])  # [N, C_out]
        y_feat = np.stack(y_samples, axis=1)  # [N, n_windows, C_out]
        y_feat = y_feat.reshape(-1, C_out)  # [N * n_windows, C_out]

        # Store for original shapes
        self._orig_shape = None  # Don't use parent's 3D handling

        # Fit standard ridge on features
        return super().fit(X_feat, y_feat)

    def predict(self, X: NDArray) -> NDArray:
        """Predict using temporal features.

        Args:
            X: Input signals [N, C_in, T]

        Returns:
            Predictions [N, C_out, T]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        N, C_in, T = X.shape

        # Create temporal features
        X_feat = self._create_temporal_features(X)

        # Predict using parent's method (2D)
        self._orig_shape = None  # Ensure parent doesn't reshape
        y_feat = super().predict(X_feat)  # [N * n_windows, C_out]

        # Reshape back
        n_windows = (T - 1) // self.stride + 1
        y_feat = y_feat.reshape(N, n_windows, self._output_channels)

        # Interpolate if stride > 1
        if self.stride > 1:
            y = np.zeros((N, self._output_channels, T), dtype=np.float32)
            window_times = np.arange(0, T, self.stride)[:n_windows]
            for n in range(N):
                for c in range(self._output_channels):
                    y[n, c, :] = np.interp(np.arange(T), window_times, y_feat[n, :, c])
        else:
            # [N, n_windows, C_out] -> [N, C_out, n_windows]
            y = y_feat.transpose(0, 2, 1)

        return y.astype(np.float32)


class RidgeCV(RidgeBaseline):
    """Ridge regression with cross-validated regularization selection.

    Automatically selects optimal alpha via internal cross-validation.

    Args:
        alphas: List of alpha values to try
        cv: Number of internal CV folds
        fit_intercept: Whether to fit intercept

    Attributes:
        best_alpha: Selected regularization strength
        cv_scores: Mean CV scores for each alpha

    Example:
        >>> model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        >>> model.fit(X_train, y_train)
        >>> print(f"Best alpha: {model.best_alpha}")
        >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        alphas: Optional[List[float]] = None,
        cv: int = 5,
        fit_intercept: bool = True,
    ):
        super().__init__(alpha=1.0, fit_intercept=fit_intercept)
        self.alphas = alphas or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        self.cv = cv
        self.best_alpha: Optional[float] = None
        self.cv_scores: Optional[List[float]] = None

    def _compute_r2(self, y_true: NDArray, y_pred: NDArray) -> float:
        """Compute R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-8)

    def fit(self, X: NDArray, y: NDArray) -> "RidgeCV":
        """Fit with cross-validation to select optimal alpha.

        Args:
            X: Input features [N, C_in, T] or [N, D_in]
            y: Target values [N, C_out, T] or [N, D_out]

        Returns:
            self
        """
        # Handle 3D: [N, C, T] -> [N*T, C]
        orig_shape_3d = None
        if X.ndim == 3:
            N, C_in, T = X.shape
            _, C_out, _ = y.shape
            orig_shape_3d = (N, C_in, T, C_out)
            X = X.transpose(0, 2, 1).reshape(N * T, C_in)
            y = y.transpose(0, 2, 1).reshape(N * T, C_out)

        n_samples = X.shape[0]
        fold_size = n_samples // self.cv

        best_score = -np.inf
        self.cv_scores = []

        for alpha in self.alphas:
            fold_scores = []

            for fold in range(self.cv):
                # Create train/val split
                val_start = fold * fold_size
                val_end = val_start + fold_size

                X_val = X[val_start:val_end]
                y_val = y[val_start:val_end]
                X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
                y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)

                # Fit temporary model
                model = RidgeBaseline(alpha=alpha, fit_intercept=self.fit_intercept)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                fold_scores.append(self._compute_r2(y_val, y_pred))

            mean_score = np.mean(fold_scores)
            self.cv_scores.append(mean_score)

            if mean_score > best_score:
                best_score = mean_score
                self.best_alpha = alpha

        # Fit final model with best alpha on all data
        self.alpha = self.best_alpha
        self._orig_shape = orig_shape_3d

        # Restore 3D for parent fit
        if orig_shape_3d is not None:
            N, C_in, T, C_out = orig_shape_3d
            X = X.reshape(N, T, C_in).transpose(0, 2, 1)
            y = y.reshape(N, T, C_out).transpose(0, 2, 1)

        return super().fit(X, y)
