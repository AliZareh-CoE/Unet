"""Euclidean Alignment (EA) for Cross-Session/Subject Transfer Learning.

This module implements Euclidean Alignment as described in:
- He & Wu (2020): "Transfer Learning for Brain-Computer Interfaces"
- "EEG-Bench: A Foundation for Assessing Deep Learning EEG Models"
  https://arxiv.org/html/2401.10746v3

EA aligns neural data by whitening each session to identity covariance:

    X_aligned = R^(-1/2) @ X

Where R is the mean covariance matrix for that session/subject.

Key insight: After EA, the mean covariance matrix of each domain equals the
identity matrix, making EEG data distributions from different domains more
consistent. EA focuses on the domain gap by re-centering each domain at the
identity matrix.

Key benefits:
- Reduces inter-session/subject data discrepancies
- Improves cross-subject model accuracy
- Accelerates convergence (~70% faster)
- Simple: no reference covariance needed

TO REMOVE THIS FEATURE:
1. Delete this file (euclidean_alignment.py)
2. Remove EA imports from data.py
3. Remove EA config options from train.py (use_euclidean_alignment, ea_*)
4. Remove --euclidean-alignment CLI args from train.py
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path
import warnings


def _matrix_sqrt(A: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Compute matrix square root (or inverse square root) via eigendecomposition.

    Args:
        A: Symmetric positive definite matrix [C, C]
        inverse: If True, compute A^(-1/2), else A^(1/2)

    Returns:
        Matrix square root [C, C]
    """
    # Eigendecomposition: A = V @ diag(w) @ V.T
    w, V = np.linalg.eigh(A)

    # Clamp small/negative eigenvalues for numerical stability
    w = np.maximum(w, 1e-10)

    if inverse:
        w_sqrt = 1.0 / np.sqrt(w)
    else:
        w_sqrt = np.sqrt(w)

    # Reconstruct: A^(1/2) = V @ diag(sqrt(w)) @ V.T
    return V @ np.diag(w_sqrt) @ V.T


def compute_session_covariance(
    data: np.ndarray,
    regularization: float = 1e-6,
) -> np.ndarray:
    """Compute regularized covariance matrix for a session.

    Args:
        data: Session data [trials, channels, samples] or [channels, samples]
        regularization: Regularization term added to diagonal (Tikhonov)

    Returns:
        Covariance matrix [channels, channels]
    """
    # Handle both 2D and 3D inputs
    if data.ndim == 2:
        # [channels, samples] -> treat as single trial
        data = data[np.newaxis, ...]

    trials, n_channels, n_samples = data.shape

    # Concatenate all trials: [channels, trials * samples]
    data_concat = data.transpose(1, 0, 2).reshape(n_channels, -1)

    # Center the data (zero mean per channel)
    data_centered = data_concat - data_concat.mean(axis=1, keepdims=True)

    # Compute sample covariance: (1/N) * X @ X.T
    n_total = data_centered.shape[1]
    cov = (data_centered @ data_centered.T) / n_total

    # Add Tikhonov regularization for numerical stability
    cov += regularization * np.eye(n_channels)

    return cov


class EuclideanAlignment:
    """Euclidean Alignment for cross-session/subject transfer learning.

    Aligns neural data by whitening each session to identity covariance:

        X_aligned = R^(-1/2) @ X

    Where R is the mean covariance matrix for that session.

    After EA, the mean covariance of each session equals the identity matrix,
    making data distributions from different sessions more consistent.

    Attributes:
        session_covariances: Dict of session covariances
        session_transforms: Dict of precomputed transforms (R^(-1/2))
        fitted: Whether fit() has been called
    """

    def __init__(
        self,
        regularization: float = 1e-6,
        reference_method: str = "arithmetic",  # Kept for API compatibility, not used
    ):
        """Initialize Euclidean Alignment.

        Args:
            regularization: Tikhonov regularization for covariance matrices
            reference_method: Kept for API compatibility (not used in true EA)
        """
        self.regularization = regularization
        self.reference_method = reference_method  # Not used, kept for compatibility

        self.session_covariances: Dict[str, np.ndarray] = {}
        self.session_transforms: Dict[str, np.ndarray] = {}
        self.fitted = False

    def fit(
        self,
        session_data: Dict[str, np.ndarray],
        verbose: bool = False,
    ) -> "EuclideanAlignment":
        """Fit EA on training sessions.

        For each session, computes the covariance matrix and its inverse
        square root transform that will whiten the data to identity covariance.

        Args:
            session_data: Dict mapping session_id -> data [trials, channels, samples]
            verbose: Print progress info

        Returns:
            self (for chaining)
        """
        if verbose:
            print(f"Fitting Euclidean Alignment on {len(session_data)} sessions...")

        # For each session: compute covariance and whitening transform
        for session_id, data in session_data.items():
            # Step 1: Compute session covariance
            cov = compute_session_covariance(data, self.regularization)
            self.session_covariances[session_id] = cov

            # Step 2: Compute whitening transform: R^(-1/2)
            # This maps the session's covariance to identity
            session_inv_sqrt = _matrix_sqrt(cov, inverse=True)
            self.session_transforms[session_id] = session_inv_sqrt

            if verbose:
                print(f"  Session {session_id}: cov shape {cov.shape}, "
                      f"trace={np.trace(cov):.2f} -> identity")

        self.fitted = True

        if verbose:
            print(f"Euclidean Alignment fitted successfully.")
            print(f"  All sessions will be whitened to identity covariance.")

        return self

    def fit_single_session(
        self,
        session_id: str,
        data: np.ndarray,
    ) -> None:
        """Add/update a single session's covariance and transform.

        Args:
            session_id: Session identifier
            data: Session data [trials, channels, samples]
        """
        cov = compute_session_covariance(data, self.regularization)
        self.session_covariances[session_id] = cov

        # Compute whitening transform: R^(-1/2)
        session_inv_sqrt = _matrix_sqrt(cov, inverse=True)
        self.session_transforms[session_id] = session_inv_sqrt

        self.fitted = True

    def transform(
        self,
        data: np.ndarray,
        session_id: str,
    ) -> np.ndarray:
        """Transform data using session-specific alignment.

        Applies: X_aligned = R^(-1/2) @ X

        Args:
            data: Data to transform [trials, channels, samples] or [channels, samples]
            session_id: Session identifier (must have been seen during fit)

        Returns:
            Aligned data (same shape as input) with identity covariance
        """
        if not self.fitted:
            raise RuntimeError("EuclideanAlignment not fitted. Call fit() first.")

        if session_id not in self.session_transforms:
            warnings.warn(
                f"Session '{session_id}' not seen during fit. "
                f"Using identity transform (no alignment)."
            )
            return data

        transform = self.session_transforms[session_id]

        # Handle both 2D and 3D inputs
        squeeze = False
        if data.ndim == 2:
            data = data[np.newaxis, ...]
            squeeze = True

        # Apply transform: [C, C] @ [trials, C, samples] -> need to handle properly
        # Transform each trial: transform @ data[i] for data[i] shape [C, T]
        trials, n_channels, n_samples = data.shape

        # Reshape for batch matrix multiply: [trials, C, T] -> [trials, C, T]
        # transform @ X where transform is [C, C] and X is [C, T]
        aligned = np.einsum('ij,njt->nit', transform, data)

        if squeeze:
            aligned = aligned[0]

        return aligned

    def transform_new_session(
        self,
        data: np.ndarray,
        compute_cov_from_data: bool = True,
    ) -> np.ndarray:
        """Transform data from a new (unseen) session.

        For new sessions not seen during training, we compute the session's
        covariance from the data itself and apply EA.

        Args:
            data: Data to transform [trials, channels, samples]
            compute_cov_from_data: If True, compute covariance from data.
                                   If False, return data unchanged.

        Returns:
            Aligned data (same shape as input) with identity covariance
        """
        if not compute_cov_from_data:
            return data

        # Compute session covariance from the data
        session_cov = compute_session_covariance(data, self.regularization)

        # Compute whitening transform: R^(-1/2)
        session_inv_sqrt = _matrix_sqrt(session_cov, inverse=True)

        # Apply transform
        squeeze = False
        if data.ndim == 2:
            data = data[np.newaxis, ...]
            squeeze = True

        aligned = np.einsum('ij,njt->nit', session_inv_sqrt, data)

        if squeeze:
            aligned = aligned[0]

        return aligned

    def save(self, path: Union[str, Path]) -> None:
        """Save fitted EA model to disk.

        Args:
            path: Path to save file (.npz)
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted model.")

        np.savez(
            path,
            regularization=self.regularization,
            reference_method=self.reference_method,
            session_ids=list(self.session_covariances.keys()),
            session_covs=np.array(list(self.session_covariances.values())),
            session_transforms=np.array(list(self.session_transforms.values())),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EuclideanAlignment":
        """Load fitted EA model from disk.

        Args:
            path: Path to saved file (.npz)

        Returns:
            Loaded EuclideanAlignment instance
        """
        data = np.load(path, allow_pickle=True)

        ea = cls(
            regularization=float(data["regularization"]),
            reference_method=str(data["reference_method"]),
        )

        session_ids = data["session_ids"]
        session_covs = data["session_covs"]
        session_transforms = data["session_transforms"]

        for i, session_id in enumerate(session_ids):
            ea.session_covariances[session_id] = session_covs[i]
            ea.session_transforms[session_id] = session_transforms[i]

        ea.fitted = True

        return ea


# =============================================================================
# Convenience functions for integration with data loading
# =============================================================================

def fit_euclidean_alignment(
    data_by_session: Dict[str, np.ndarray],
    regularization: float = 1e-6,
    reference_method: str = "arithmetic",
    verbose: bool = False,
) -> EuclideanAlignment:
    """Convenience function to fit EA on session data.

    Args:
        data_by_session: Dict mapping session_id -> data [trials, channels, samples]
        regularization: Tikhonov regularization
        reference_method: Kept for API compatibility (not used)
        verbose: Print progress

    Returns:
        Fitted EuclideanAlignment instance
    """
    ea = EuclideanAlignment(
        regularization=regularization,
        reference_method=reference_method,
    )
    ea.fit(data_by_session, verbose=verbose)
    return ea


def apply_euclidean_alignment(
    data: np.ndarray,
    session_id: str,
    ea: EuclideanAlignment,
) -> np.ndarray:
    """Convenience function to apply EA to data.

    Args:
        data: Data to transform [trials, channels, samples]
        session_id: Session identifier
        ea: Fitted EuclideanAlignment instance

    Returns:
        Aligned data with identity covariance
    """
    return ea.transform(data, session_id)
