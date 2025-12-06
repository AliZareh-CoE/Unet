"""
Ridge Regression for Neural Signal Translation
===============================================

L2-regularized linear mapping between neural signals.
Simple but effective baseline for signal translation.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray


class RidgeRegression:
    """Ridge regression for neural signal translation.

    Solves: min ||Y - XW||^2 + alpha * ||W||^2

    Closed-form solution: W = (X^T X + alpha * I)^{-1} X^T Y

    Args:
        alpha: L2 regularization strength
        fit_intercept: Whether to fit an intercept term
        normalize: Whether to normalize input features
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

        # Parameters
        self.W: Optional[NDArray] = None
        self.b: Optional[NDArray] = None
        self.X_mean: Optional[NDArray] = None
        self.X_std: Optional[NDArray] = None
        self._fitted = False

    def _preprocess(self, X: NDArray, fit: bool = False) -> NDArray:
        """Preprocess input data."""
        if fit:
            if self.fit_intercept:
                self.X_mean = np.mean(X, axis=0, keepdims=True)
            else:
                self.X_mean = np.zeros((1, X.shape[1]))

            if self.normalize:
                self.X_std = np.std(X, axis=0, keepdims=True)
                self.X_std[self.X_std < 1e-8] = 1.0
            else:
                self.X_std = np.ones((1, X.shape[1]))

        X = (X - self.X_mean) / self.X_std
        return X

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> "RidgeRegression":
        """Fit ridge regression model.

        Args:
            X: Input features [N, D_in] or [N, C_in, T]
            y: Target values [N, D_out] or [N, C_out, T]

        Returns:
            self
        """
        # Handle 3D tensors - DON'T flatten time dimension (causes memory explosion)
        # Instead, treat each time point independently
        orig_shape = None
        if X.ndim == 3:
            N, C_in, T = X.shape
            _, C_out, _ = y.shape
            orig_shape = (C_in, T, C_out)
            # Transpose to [N, T, C] then reshape to [N*T, C]
            X = X.transpose(0, 2, 1).reshape(N * T, C_in)
            y = y.transpose(0, 2, 1).reshape(N * T, C_out)

        # Preprocess
        X = self._preprocess(X, fit=True)

        N_total, D_in = X.shape
        _, D_out = y.shape

        # Center y if fitting intercept
        if self.fit_intercept:
            y_mean = np.mean(y, axis=0, keepdims=True)
            y_centered = y - y_mean
        else:
            y_mean = np.zeros((1, D_out))
            y_centered = y

        # Closed-form solution with memory-efficient approach
        # W = (X^T X + alpha * I)^{-1} X^T y
        # For large datasets, use chunked computation to avoid memory issues
        D_in = X.shape[1]

        # X^T X is [D_in, D_in] which is manageable (32x32 = small)
        XtX = X.T @ X
        XtX_reg = XtX + self.alpha * np.eye(D_in)
        Xty = X.T @ y_centered

        self.W = np.linalg.solve(XtX_reg, Xty)

        # Compute intercept
        if self.fit_intercept:
            self.b = y_mean.flatten() - (self.X_mean / self.X_std) @ self.W
            self.b = self.b.flatten()
        else:
            self.b = np.zeros(D_out)

        self._fitted = True
        self._orig_shape = orig_shape
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict using fitted model.

        Args:
            X: Input features [N, D_in] or [N, C_in, T]

        Returns:
            Predictions [N, D_out] or [N, C_out, T]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Handle 3D - same reshape as fit
        reshape_back = False
        N_samples = None
        T_len = None
        if X.ndim == 3:
            N_samples, C_in, T_len = X.shape
            X = X.transpose(0, 2, 1).reshape(N_samples * T_len, C_in)
            reshape_back = True

        # Preprocess
        X = self._preprocess(X, fit=False)

        # Predict
        y = X @ self.W + self.b

        # Reshape back if needed
        if reshape_back and self._orig_shape is not None:
            C_in, T, C_out = self._orig_shape
            # y is [N*T, C_out], reshape to [N, T, C_out] then transpose to [N, C_out, T]
            y = y.reshape(N_samples, T_len, C_out).transpose(0, 2, 1)

        return y

    def fit_predict(self, X: NDArray, y: NDArray) -> NDArray:
        """Fit and predict in one step."""
        return self.fit(X, y).predict(X)

    def get_weights(self) -> Tuple[NDArray, NDArray]:
        """Get model weights.

        Returns:
            W: Weight matrix [D_in, D_out]
            b: Bias vector [D_out]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
        return self.W, self.b


class TemporalRidgeRegression(RidgeRegression):
    """Ridge regression with temporal context windows.

    Uses sliding window to include temporal context for prediction.

    Args:
        alpha: L2 regularization
        window_size: Number of time points for context (before and after)
        stride: Stride for window sliding
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

    def _create_temporal_features(
        self,
        X: NDArray,
        fit: bool = False,
    ) -> NDArray:
        """Create temporal context features using sliding window.

        Args:
            X: Input [N, C, T]
            fit: Whether this is during fitting

        Returns:
            Features [N * n_windows, C * (2*window_size + 1)]
        """
        N, C, T = X.shape

        # Pad signal
        pad_width = ((0, 0), (0, 0), (self.window_size, self.window_size))
        X_padded = np.pad(X, pad_width, mode="edge")

        # Create windows
        n_windows = (T - 1) // self.stride + 1
        window_len = 2 * self.window_size + 1

        features = []
        for i in range(0, T, self.stride):
            window = X_padded[:, :, i : i + window_len]
            features.append(window.reshape(N, -1))

        features = np.stack(features, axis=1)  # [N, n_windows, C * window_len]
        return features.reshape(N * n_windows, -1)

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> "TemporalRidgeRegression":
        """Fit temporal ridge regression.

        Args:
            X: Input signals [N, C_in, T]
            y: Target signals [N, C_out, T]

        Returns:
            self
        """
        N, C_in, T = X.shape
        _, C_out, _ = y.shape

        self._input_shape = (C_in, T)
        self._output_channels = C_out

        # Create temporal features
        X_feat = self._create_temporal_features(X, fit=True)

        # Create target (subsample to match windows)
        n_windows = (T - 1) // self.stride + 1
        y_feat = []
        for i in range(0, T, self.stride):
            y_feat.append(y[:, :, i])
        y_feat = np.stack(y_feat, axis=1)  # [N, n_windows, C_out]
        y_feat = y_feat.reshape(N * n_windows, C_out)

        # Fit standard ridge
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
        X_feat = self._create_temporal_features(X, fit=False)

        # Predict
        y_feat = super().predict(X_feat)

        # Reshape back
        n_windows = (T - 1) // self.stride + 1
        y_feat = y_feat.reshape(N, n_windows, self._output_channels)

        # Interpolate if stride > 1
        if self.stride > 1:
            y = np.zeros((N, self._output_channels, T))
            window_times = np.arange(0, T, self.stride)
            for n in range(N):
                for c in range(self._output_channels):
                    y[n, c, :] = np.interp(
                        np.arange(T),
                        window_times[:n_windows],
                        y_feat[n, :, c],
                    )
        else:
            y = y_feat.transpose(0, 2, 1)

        return y


class MultiOutputRidgeCV:
    """Ridge regression with cross-validated regularization.

    Automatically selects optimal alpha via cross-validation.

    Args:
        alphas: List of alpha values to try
        cv: Number of cross-validation folds
        fit_intercept: Whether to fit intercept
    """

    def __init__(
        self,
        alphas: Optional[List[float]] = None,
        cv: int = 5,
        fit_intercept: bool = True,
    ):
        self.alphas = alphas or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.cv = cv
        self.fit_intercept = fit_intercept

        self.best_alpha: Optional[float] = None
        self.model: Optional[RidgeRegression] = None
        self._fitted = False

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> "MultiOutputRidgeCV":
        """Fit with cross-validation to select alpha.

        Args:
            X: Input features [N, D_in]
            y: Target values [N, D_out]

        Returns:
            self
        """
        N = X.shape[0]
        fold_size = N // self.cv

        best_score = -np.inf
        best_alpha = self.alphas[0]

        for alpha in self.alphas:
            scores = []

            for fold in range(self.cv):
                # Create train/val split
                val_start = fold * fold_size
                val_end = val_start + fold_size

                X_val = X[val_start:val_end]
                y_val = y[val_start:val_end]
                X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
                y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)

                # Fit and evaluate
                model = RidgeRegression(
                    alpha=alpha,
                    fit_intercept=self.fit_intercept,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                # Compute R^2 score
                ss_res = np.sum((y_val - y_pred) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val, axis=0)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                scores.append(r2)

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

        # Fit final model with best alpha
        self.best_alpha = best_alpha
        self.model = RidgeRegression(
            alpha=best_alpha,
            fit_intercept=self.fit_intercept,
        )
        self.model.fit(X, y)
        self._fitted = True

        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict using fitted model."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
        return self.model.predict(X)


def create_ridge_regression(
    variant: str = "standard",
    **kwargs,
) -> RidgeRegression:
    """Factory function for ridge regression variants.

    Args:
        variant: "standard", "temporal", or "cv"
        **kwargs: Additional arguments

    Returns:
        Ridge regression instance
    """
    if variant == "standard":
        return RidgeRegression(**kwargs)
    elif variant == "temporal":
        return TemporalRidgeRegression(**kwargs)
    elif variant == "cv":
        return MultiOutputRidgeCV(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
