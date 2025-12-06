"""
Support Vector Regression (SVR) for Neural Signal Translation
==============================================================

Kernel-based nonlinear regression baseline.
Uses PCA for dimensionality reduction to handle large feature spaces.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class SVRBaseline:
    """SVR baseline for signal translation.

    Uses PCA to reduce dimensionality before SVR (essential for speed).

    Args:
        kernel: SVR kernel ('rbf', 'linear', 'poly')
        C: Regularization parameter
        epsilon: Epsilon in epsilon-SVR
        n_components: PCA components (None = auto)
        subsample: Subsample rate for training (1.0 = all data)
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        n_components: Optional[int] = 100,
        subsample: float = 1.0,
    ):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.n_components = n_components
        self.subsample = subsample

        self.pca_x: Optional[PCA] = None
        self.pca_y: Optional[PCA] = None
        self.scaler_x: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None
        self.model: Optional[MultiOutputRegressor] = None
        self._fitted = False

    def fit(self, X: NDArray, y: NDArray) -> "SVRBaseline":
        """Fit SVR model.

        Args:
            X: Input signals [N, C_in, T]
            y: Target signals [N, C_out, T]

        Returns:
            self
        """
        # Reshape to 2D
        if X.ndim == 3:
            N, C_in, T = X.shape
            C_out = y.shape[1]
            self._input_shape = (C_in, T)
            self._output_shape = (C_out, T)
            X = X.reshape(N, -1)
            y = y.reshape(N, -1)
        else:
            N = X.shape[0]

        # Subsample for speed
        if self.subsample < 1.0:
            n_samples = int(N * self.subsample)
            idx = np.random.choice(N, n_samples, replace=False)
            X = X[idx]
            y = y[idx]

        # Standardize
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        X = self.scaler_x.fit_transform(X)
        y = self.scaler_y.fit_transform(y)

        # PCA for dimensionality reduction
        n_comp_x = min(self.n_components or X.shape[1], X.shape[0] - 1, X.shape[1])
        n_comp_y = min(self.n_components or y.shape[1], y.shape[0] - 1, y.shape[1])

        self.pca_x = PCA(n_components=n_comp_x)
        self.pca_y = PCA(n_components=n_comp_y)

        X_pca = self.pca_x.fit_transform(X)
        y_pca = self.pca_y.fit_transform(y)

        # Fit SVR
        base_svr = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
        self.model = MultiOutputRegressor(base_svr, n_jobs=-1)
        self.model.fit(X_pca, y_pca)

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

        reshape_back = False
        if X.ndim == 3:
            N = X.shape[0]
            X = X.reshape(N, -1)
            reshape_back = True
        else:
            N = X.shape[0]

        # Transform
        X = self.scaler_x.transform(X)
        X_pca = self.pca_x.transform(X)

        # Predict in PCA space
        y_pca = self.model.predict(X_pca)

        # Inverse transform
        y = self.pca_y.inverse_transform(y_pca)
        y = self.scaler_y.inverse_transform(y)

        if reshape_back:
            C_out, T = self._output_shape
            y = y.reshape(N, C_out, T)

        return y

    def fit_predict(self, X: NDArray, y: NDArray) -> NDArray:
        """Fit and predict."""
        return self.fit(X, y).predict(X)


class LinearSVRBaseline(SVRBaseline):
    """Linear SVR - faster than RBF for large datasets."""

    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        n_components: Optional[int] = 100,
        subsample: float = 1.0,
    ):
        super().__init__(
            kernel="linear",
            C=C,
            epsilon=epsilon,
            n_components=n_components,
            subsample=subsample,
        )


def create_svr_baseline(variant: str = "rbf", **kwargs) -> SVRBaseline:
    """Factory for SVR variants.

    Args:
        variant: "rbf" or "linear"
        **kwargs: Additional arguments

    Returns:
        SVR baseline instance
    """
    if variant == "rbf":
        return SVRBaseline(kernel="rbf", **kwargs)
    elif variant == "linear":
        return LinearSVRBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
