"""
Support Vector Regression (SVR) for Neural Signal Translation - GPU Accelerated
=================================================================================

Kernel-based nonlinear regression baseline.
Uses GPU for preprocessing (PCA, scaling) and cuML for SVR if available.
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

# cuML for GPU SVR (RAPIDS)
try:
    from cuml.svm import SVR as cumlSVR
    from cuml.decomposition import PCA as cumlPCA
    from cuml.preprocessing import StandardScaler as cumlScaler
    HAS_CUML = True
except ImportError:
    HAS_CUML = False


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


class SVRBaseline:
    """SVR baseline for signal translation - GPU accelerated.

    Uses cuML for GPU SVR if available, falls back to sklearn.
    PCA preprocessing is GPU accelerated with CuPy.

    Args:
        kernel: SVR kernel ('rbf', 'linear', 'poly')
        C: Regularization parameter
        epsilon: Epsilon in epsilon-SVR
        n_components: PCA components for dimensionality reduction
        use_gpu: Use GPU acceleration
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        n_components: int = 100,
        use_gpu: bool = True,
    ):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.n_components = n_components
        self.use_gpu = use_gpu and (HAS_CUPY or HAS_CUML)

        self.pca_x = None
        self.pca_y = None
        self.scaler_x = None
        self.scaler_y = None
        self.model = None
        self._fitted = False

    def _gpu_pca_fit_transform(self, X: NDArray, n_components: int):
        """GPU-accelerated PCA fit_transform using CuPy SVD."""
        xp = get_array_module(self.use_gpu)

        if self.use_gpu and HAS_CUPY:
            X = xp.asarray(X)

        # Center
        mean = xp.mean(X, axis=0)
        X_centered = X - mean

        # SVD for PCA
        n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
        U, S, Vt = xp.linalg.svd(X_centered, full_matrices=False)

        components = Vt[:n_comp, :]
        X_transformed = U[:, :n_comp] * S[:n_comp]

        return to_numpy(X_transformed), to_numpy(components), to_numpy(mean)

    def _gpu_pca_transform(self, X: NDArray, components: NDArray, mean: NDArray):
        """GPU-accelerated PCA transform."""
        xp = get_array_module(self.use_gpu)

        if self.use_gpu and HAS_CUPY:
            X = xp.asarray(X)
            components = xp.asarray(components)
            mean = xp.asarray(mean)

        X_transformed = (X - mean) @ components.T
        return to_numpy(X_transformed)

    def _gpu_standardize_fit(self, X: NDArray):
        """GPU-accelerated standardization fit."""
        xp = get_array_module(self.use_gpu)

        if self.use_gpu and HAS_CUPY:
            X = xp.asarray(X)

        mean = xp.mean(X, axis=0)
        std = xp.std(X, axis=0)
        std = xp.where(std < 1e-8, 1.0, std)

        X_scaled = (X - mean) / std
        return to_numpy(X_scaled), to_numpy(mean), to_numpy(std)

    def _gpu_standardize_transform(self, X: NDArray, mean: NDArray, std: NDArray):
        """GPU-accelerated standardization transform."""
        xp = get_array_module(self.use_gpu)

        if self.use_gpu and HAS_CUPY:
            X = xp.asarray(X)
            mean = xp.asarray(mean)
            std = xp.asarray(std)

        return to_numpy((X - mean) / std)

    def fit(self, X: NDArray, y: NDArray) -> "SVRBaseline":
        """Fit SVR model - GPU accelerated.

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

        # Standardize on GPU
        X, self._x_mean, self._x_std = self._gpu_standardize_fit(X)
        y, self._y_mean, self._y_std = self._gpu_standardize_fit(y)

        # PCA on GPU
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        X_pca, self._pca_x_comp, self._pca_x_mean = self._gpu_pca_fit_transform(X, n_comp)
        y_pca, self._pca_y_comp, self._pca_y_mean = self._gpu_pca_fit_transform(y, n_comp)

        # Fit SVR - use cuML if available
        if HAS_CUML and self.use_gpu:
            # cuML multi-output SVR
            from cuml.svm import SVR as cumlSVR
            self.models = []
            for i in range(y_pca.shape[1]):
                svr = cumlSVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
                svr.fit(X_pca, y_pca[:, i])
                self.models.append(svr)
        else:
            # sklearn fallback
            from sklearn.svm import SVR
            from sklearn.multioutput import MultiOutputRegressor
            base_svr = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
            self.model = MultiOutputRegressor(base_svr, n_jobs=-1)
            self.model.fit(X_pca, y_pca)

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

        reshape_back = False
        if X.ndim == 3:
            N = X.shape[0]
            X = X.reshape(N, -1)
            reshape_back = True
        else:
            N = X.shape[0]

        # Standardize on GPU
        X = self._gpu_standardize_transform(X, self._x_mean, self._x_std)

        # PCA transform on GPU
        X_pca = self._gpu_pca_transform(X, self._pca_x_comp, self._pca_x_mean)

        # Predict
        if HAS_CUML and self.use_gpu and hasattr(self, 'models'):
            y_pca = np.column_stack([m.predict(X_pca) for m in self.models])
        else:
            y_pca = self.model.predict(X_pca)

        # Inverse PCA transform on GPU
        xp = get_array_module(self.use_gpu)
        if self.use_gpu and HAS_CUPY:
            y_pca = xp.asarray(y_pca)
            pca_y_comp = xp.asarray(self._pca_y_comp)
            pca_y_mean = xp.asarray(self._pca_y_mean)
            y = y_pca @ pca_y_comp + pca_y_mean
            y = to_numpy(y)
        else:
            y = y_pca @ self._pca_y_comp + self._pca_y_mean

        # Inverse standardize
        y = y * self._y_std + self._y_mean

        if reshape_back:
            C_out, T = self._output_shape
            y = y.reshape(N, C_out, T)

        return y

    def fit_predict(self, X: NDArray, y: NDArray) -> NDArray:
        """Fit and predict."""
        return self.fit(X, y).predict(X)


class LinearSVRBaseline(SVRBaseline):
    """Linear SVR - faster than RBF, GPU accelerated."""

    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        n_components: int = 100,
        use_gpu: bool = True,
    ):
        super().__init__(
            kernel="linear",
            C=C,
            epsilon=epsilon,
            n_components=n_components,
            use_gpu=use_gpu,
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
