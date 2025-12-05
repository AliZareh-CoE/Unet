"""
Canonical Correlation Analysis for Neural Signal Translation
============================================================

CCA finds linear projections that maximize correlation between
input and output signals. Useful baseline for multi-channel
neural signal translation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


class CCABaseline:
    """Canonical Correlation Analysis for signal translation.

    CCA finds linear projections W_x and W_y such that:
    corr(X @ W_x, Y @ W_y) is maximized

    For translation, we project input to canonical space and then
    back to output space.

    Args:
        n_components: Number of CCA components to use
        regularization: Regularization for covariance matrices
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        regularization: float = 1e-6,
    ):
        self.n_components = n_components
        self.regularization = regularization

        # Learned parameters
        self.W_x: Optional[NDArray] = None  # Input projection
        self.W_y: Optional[NDArray] = None  # Output projection
        self.correlations: Optional[NDArray] = None  # Canonical correlations
        self.x_mean: Optional[NDArray] = None
        self.y_mean: Optional[NDArray] = None
        self._fitted = False

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> "CCABaseline":
        """Fit CCA model.

        Args:
            X: Input data [N, D_x] or [N, C_in, T]
            y: Output data [N, D_y] or [N, C_out, T]

        Returns:
            self
        """
        # Handle 3D tensors
        self._orig_shape_x = None
        self._orig_shape_y = None

        if X.ndim == 3:
            N, C_in, T = X.shape
            self._orig_shape_x = (C_in, T)
            X = X.reshape(N, -1)

        if y.ndim == 3:
            N, C_out, T = y.shape
            self._orig_shape_y = (C_out, T)
            y = y.reshape(N, -1)

        N, D_x = X.shape
        _, D_y = y.shape

        # Center data
        self.x_mean = np.mean(X, axis=0, keepdims=True)
        self.y_mean = np.mean(y, axis=0, keepdims=True)
        X_c = X - self.x_mean
        y_c = y - self.y_mean

        # Compute covariance matrices
        C_xx = (X_c.T @ X_c) / (N - 1) + self.regularization * np.eye(D_x)
        C_yy = (y_c.T @ y_c) / (N - 1) + self.regularization * np.eye(D_y)
        C_xy = (X_c.T @ y_c) / (N - 1)

        # Compute CCA via generalized eigenvalue problem
        # Solve: C_xy @ C_yy^{-1} @ C_xy^T @ w = lambda * C_xx @ w

        C_xx_inv_sqrt = linalg.sqrtm(linalg.inv(C_xx))
        C_yy_inv_sqrt = linalg.sqrtm(linalg.inv(C_yy))

        # Form the matrix for SVD
        M = C_xx_inv_sqrt @ C_xy @ C_yy_inv_sqrt

        # SVD
        U, S, Vt = linalg.svd(M, full_matrices=False)

        # Number of components
        n_comp = self.n_components or min(D_x, D_y)
        n_comp = min(n_comp, len(S))

        # Extract components
        self.W_x = np.real(C_xx_inv_sqrt @ U[:, :n_comp])
        self.W_y = np.real(C_yy_inv_sqrt @ Vt[:n_comp, :].T)
        self.correlations = S[:n_comp]

        # For prediction: learn mapping from canonical X to original Y
        # X_canonical = X @ W_x
        # We want to predict Y from X_canonical
        X_canonical = X_c @ self.W_x
        self.reconstruction_weights = np.linalg.lstsq(
            X_canonical, y_c, rcond=None
        )[0]

        self._fitted = True
        return self

    def transform(
        self,
        X: NDArray,
        y: Optional[NDArray] = None,
    ) -> Tuple[NDArray, Optional[NDArray]]:
        """Transform data to canonical space.

        Args:
            X: Input data [N, D_x]
            y: Optional output data [N, D_y]

        Returns:
            X_canonical, Y_canonical (if y provided)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        X_c = X - self.x_mean
        X_canonical = X_c @ self.W_x

        if y is not None:
            y_c = y - self.y_mean
            y_canonical = y_c @ self.W_y
            return X_canonical, y_canonical

        return X_canonical, None

    def predict(self, X: NDArray) -> NDArray:
        """Predict output from input using CCA mapping.

        Args:
            X: Input data [N, D_x] or [N, C_in, T]

        Returns:
            Predictions [N, D_y] or [N, C_out, T]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        # Handle 3D
        reshape_back = False
        if X.ndim == 3:
            N, C_in, T = X.shape
            X = X.reshape(N, -1)
            reshape_back = True

        # Center and project to canonical space
        X_c = X - self.x_mean
        X_canonical = X_c @ self.W_x

        # Reconstruct output
        y = X_canonical @ self.reconstruction_weights + self.y_mean

        # Reshape if needed
        if reshape_back and self._orig_shape_y is not None:
            C_out, T = self._orig_shape_y
            y = y.reshape(-1, C_out, T)

        return y

    def fit_predict(self, X: NDArray, y: NDArray) -> NDArray:
        """Fit and predict in one step."""
        return self.fit(X, y).predict(X)

    def get_canonical_correlations(self) -> NDArray:
        """Get canonical correlations."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
        return self.correlations

    def score(self, X: NDArray, y: NDArray) -> float:
        """Compute R^2 score.

        Args:
            X: Input data
            y: Target data

        Returns:
            R^2 score
        """
        y_pred = self.predict(X)

        if y.ndim == 3:
            y = y.reshape(y.shape[0], -1)
            y_pred = y_pred.reshape(y_pred.shape[0], -1)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)

        return 1 - ss_res / (ss_tot + 1e-8)


class RegularizedCCA(CCABaseline):
    """CCA with explicit regularization types.

    Supports different regularization strategies:
    - ridge: Standard L2 regularization
    - pca: Reduce dimensionality with PCA first
    - elastic: Combination of L1 and L2

    Args:
        n_components: Number of CCA components
        regularization_type: "ridge", "pca", or "elastic"
        alpha: Regularization strength
        pca_components: Number of PCA components (for pca type)
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        regularization_type: str = "ridge",
        alpha: float = 0.1,
        pca_components: Optional[int] = None,
    ):
        super().__init__(n_components=n_components, regularization=alpha)
        self.regularization_type = regularization_type
        self.alpha = alpha
        self.pca_components = pca_components

        # PCA parameters
        self.pca_x: Optional[Tuple[NDArray, NDArray]] = None
        self.pca_y: Optional[Tuple[NDArray, NDArray]] = None

    def _pca_reduce(
        self,
        X: NDArray,
        n_components: int,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Reduce dimensionality with PCA.

        Returns:
            X_reduced, components, mean
        """
        mean = np.mean(X, axis=0)
        X_centered = X - mean

        # SVD for PCA
        U, S, Vt = linalg.svd(X_centered, full_matrices=False)
        components = Vt[:n_components, :]

        X_reduced = X_centered @ components.T
        return X_reduced, components, mean

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> "RegularizedCCA":
        """Fit regularized CCA.

        Args:
            X: Input data [N, D_x]
            y: Output data [N, D_y]

        Returns:
            self
        """
        if self.regularization_type == "pca" and self.pca_components is not None:
            # Apply PCA reduction first
            X_red, self.pca_x_comp, self.pca_x_mean = self._pca_reduce(
                X, self.pca_components
            )
            y_red, self.pca_y_comp, self.pca_y_mean = self._pca_reduce(
                y, self.pca_components
            )
            X, y = X_red, y_red

        return super().fit(X, y)

    def predict(self, X: NDArray) -> NDArray:
        """Predict with regularized CCA."""
        if self.regularization_type == "pca" and self.pca_x_comp is not None:
            X = (X - self.pca_x_mean) @ self.pca_x_comp.T

        y_pred = super().predict(X)

        if self.regularization_type == "pca" and self.pca_y_comp is not None:
            # Project back to original space
            y_pred = y_pred @ self.pca_y_comp + self.pca_y_mean

        return y_pred


class TemporalCCA(CCABaseline):
    """CCA with temporal embeddings.

    Uses delay embeddings to capture temporal dynamics.

    Args:
        n_components: Number of CCA components
        n_delays: Number of time delays to include
        regularization: Regularization strength
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        n_delays: int = 5,
        regularization: float = 1e-6,
    ):
        super().__init__(n_components=n_components, regularization=regularization)
        self.n_delays = n_delays

    def _create_delay_embedding(
        self,
        X: NDArray,
    ) -> NDArray:
        """Create delay embedding of signal.

        Args:
            X: Signal [N, C, T]

        Returns:
            Embedded signal [N, T - n_delays, C * (n_delays + 1)]
        """
        N, C, T = X.shape
        T_out = T - self.n_delays

        embedded = np.zeros((N, T_out, C * (self.n_delays + 1)))

        for d in range(self.n_delays + 1):
            start = self.n_delays - d
            end = T - d
            embedded[:, :, d * C : (d + 1) * C] = X[:, :, start:end].transpose(0, 2, 1)

        return embedded.reshape(N * T_out, -1)

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> "TemporalCCA":
        """Fit temporal CCA.

        Args:
            X: Input signals [N, C_in, T]
            y: Target signals [N, C_out, T]

        Returns:
            self
        """
        self._input_shape = X.shape[1:]
        self._output_shape = y.shape[1:]

        # Create delay embeddings
        X_emb = self._create_delay_embedding(X)
        y_emb = self._create_delay_embedding(y)

        return super().fit(X_emb, y_emb)

    def predict(self, X: NDArray) -> NDArray:
        """Predict using temporal CCA."""
        N, C_in, T = X.shape
        C_out = self._output_shape[0]

        # Create delay embedding
        X_emb = self._create_delay_embedding(X)

        # Predict
        y_emb = super().predict(X_emb)

        # Reshape to [N, T_out, C_out * (n_delays + 1)]
        T_out = T - self.n_delays
        y_emb = y_emb.reshape(N, T_out, -1)

        # Take only the current time step (delay=0)
        y = y_emb[:, :, :C_out]

        # Pad to original length
        y_full = np.zeros((N, C_out, T))
        y_full[:, :, self.n_delays :] = y.transpose(0, 2, 1)
        # Extrapolate beginning
        y_full[:, :, : self.n_delays] = y_full[:, :, self.n_delays : self.n_delays + 1]

        return y_full


def create_cca_baseline(
    variant: str = "standard",
    **kwargs,
) -> CCABaseline:
    """Factory function for CCA variants.

    Args:
        variant: "standard", "regularized", or "temporal"
        **kwargs: Additional arguments

    Returns:
        CCA baseline instance
    """
    if variant == "standard":
        return CCABaseline(**kwargs)
    elif variant == "regularized":
        return RegularizedCCA(**kwargs)
    elif variant == "temporal":
        return TemporalCCA(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
