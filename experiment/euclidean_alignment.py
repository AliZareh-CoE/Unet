"""Euclidean Alignment (EA) for Cross-Session/Subject Transfer Learning.

This module implements Euclidean Alignment as described in:
"EEG-Bench: A Foundation for Assessing Deep Learning EEG Models"
https://arxiv.org/html/2401.10746v3

EA aligns the marginal distributions of neural data across sessions/subjects
by whitening each session's data with its own covariance and re-coloring
with a reference covariance matrix.

Key benefits:
- Reduces inter-session/subject data discrepancies
- Improves cross-subject model accuracy
- Accelerates convergence (~70% faster)

Usage:
    # Compute reference matrix from all sessions
    ea = EuclideanAlignment()
    ea.fit(session_data_dict)  # {session_id: (trials, channels, samples)}

    # Transform new data
    aligned_data = ea.transform(data, session_id)

TO REMOVE THIS FEATURE:
1. Delete this file (euclidean_alignment.py)
2. Remove EA imports from data.py
3. Remove EA config options from train.py (use_euclidean_alignment, ea_*)
4. Remove --euclidean-alignment CLI args from train.py
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple, Union
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


def compute_reference_covariance(
    session_covariances: Dict[str, np.ndarray],
    method: str = "arithmetic",
) -> np.ndarray:
    """Compute reference covariance matrix from all sessions.

    Args:
        session_covariances: Dict mapping session_id -> covariance [C, C]
        method: Averaging method:
            - "arithmetic": Simple arithmetic mean (default, fastest)
            - "geometric": Riemannian geometric mean (more accurate but slower)

    Returns:
        Reference covariance matrix [C, C]
    """
    covs = list(session_covariances.values())

    if method == "arithmetic":
        # Simple arithmetic mean
        return np.mean(covs, axis=0)

    elif method == "geometric":
        # Riemannian geometric mean (iterative algorithm)
        # This is more principled for SPD matrices but slower
        ref = np.mean(covs, axis=0)  # Initialize with arithmetic mean

        for _ in range(10):  # Fixed iterations
            ref_inv_sqrt = _matrix_sqrt(ref, inverse=True)

            # Compute log-Euclidean mean
            log_sum = np.zeros_like(ref)
            for cov in covs:
                # Log map: ref^(-1/2) @ cov @ ref^(-1/2)
                transformed = ref_inv_sqrt @ cov @ ref_inv_sqrt
                # Matrix logarithm via eigendecomposition
                w, V = np.linalg.eigh(transformed)
                w = np.maximum(w, 1e-10)
                log_sum += V @ np.diag(np.log(w)) @ V.T

            log_mean = log_sum / len(covs)

            # Exp map back
            w, V = np.linalg.eigh(log_mean)
            exp_mean = V @ np.diag(np.exp(w)) @ V.T

            # Update reference
            ref_sqrt = _matrix_sqrt(ref, inverse=False)
            ref = ref_sqrt @ exp_mean @ ref_sqrt

        return ref

    else:
        raise ValueError(f"Unknown method: {method}. Use 'arithmetic' or 'geometric'")


class EuclideanAlignment:
    """Euclidean Alignment for cross-session/subject transfer learning.

    Aligns neural data by:
    1. Whitening each session with its own covariance (removes session-specific patterns)
    2. Re-coloring with reference covariance (applies shared structure)

    Formula: X_aligned = R^(1/2) @ Sigma^(-1/2) @ X

    Where:
        - Sigma: Session-specific covariance matrix
        - R: Reference covariance (mean across sessions)

    Attributes:
        reference_cov: Reference covariance matrix [C, C]
        session_covariances: Dict of session covariances
        session_transforms: Dict of precomputed transforms (R^(1/2) @ Sigma^(-1/2))
        fitted: Whether fit() has been called
    """

    def __init__(
        self,
        regularization: float = 1e-6,
        reference_method: str = "arithmetic",
    ):
        """Initialize Euclidean Alignment.

        Args:
            regularization: Tikhonov regularization for covariance matrices
            reference_method: Method for computing reference ("arithmetic" or "geometric")
        """
        self.regularization = regularization
        self.reference_method = reference_method

        self.reference_cov: Optional[np.ndarray] = None
        self.reference_sqrt: Optional[np.ndarray] = None
        self.session_covariances: Dict[str, np.ndarray] = {}
        self.session_transforms: Dict[str, np.ndarray] = {}
        self.fitted = False

    def fit(
        self,
        session_data: Dict[str, np.ndarray],
        verbose: bool = False,
    ) -> "EuclideanAlignment":
        """Fit EA on training sessions.

        Args:
            session_data: Dict mapping session_id -> data [trials, channels, samples]
            verbose: Print progress info

        Returns:
            self (for chaining)
        """
        if verbose:
            print(f"Fitting Euclidean Alignment on {len(session_data)} sessions...")

        # Step 1: Compute covariance for each session
        for session_id, data in session_data.items():
            cov = compute_session_covariance(data, self.regularization)
            self.session_covariances[session_id] = cov
            if verbose:
                print(f"  Session {session_id}: cov shape {cov.shape}, "
                      f"trace={np.trace(cov):.2f}")

        # Step 2: Compute reference covariance
        self.reference_cov = compute_reference_covariance(
            self.session_covariances,
            method=self.reference_method,
        )
        self.reference_sqrt = _matrix_sqrt(self.reference_cov, inverse=False)

        if verbose:
            print(f"  Reference cov: trace={np.trace(self.reference_cov):.2f}")

        # Step 3: Precompute transforms for each session
        for session_id, session_cov in self.session_covariances.items():
            session_inv_sqrt = _matrix_sqrt(session_cov, inverse=True)
            # Transform: R^(1/2) @ Sigma^(-1/2)
            self.session_transforms[session_id] = self.reference_sqrt @ session_inv_sqrt

        self.fitted = True

        if verbose:
            print(f"Euclidean Alignment fitted successfully.")

        return self

    def fit_single_session(
        self,
        session_id: str,
        data: np.ndarray,
    ) -> None:
        """Add/update a single session's covariance (for online updates).

        Note: After calling this, you should call recompute_reference() to
        update the reference matrix.

        Args:
            session_id: Session identifier
            data: Session data [trials, channels, samples]
        """
        cov = compute_session_covariance(data, self.regularization)
        self.session_covariances[session_id] = cov

    def recompute_reference(self) -> None:
        """Recompute reference covariance and all transforms.

        Call this after adding new sessions with fit_single_session().
        """
        if not self.session_covariances:
            raise ValueError("No sessions fitted. Call fit() or fit_single_session() first.")

        self.reference_cov = compute_reference_covariance(
            self.session_covariances,
            method=self.reference_method,
        )
        self.reference_sqrt = _matrix_sqrt(self.reference_cov, inverse=False)

        # Recompute all transforms
        for session_id, session_cov in self.session_covariances.items():
            session_inv_sqrt = _matrix_sqrt(session_cov, inverse=True)
            self.session_transforms[session_id] = self.reference_sqrt @ session_inv_sqrt

        self.fitted = True

    def transform(
        self,
        data: np.ndarray,
        session_id: str,
    ) -> np.ndarray:
        """Transform data using session-specific alignment.

        Args:
            data: Data to transform [trials, channels, samples] or [channels, samples]
            session_id: Session identifier (must have been seen during fit)

        Returns:
            Aligned data (same shape as input)
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

        For new sessions not seen during training, we can either:
        1. Compute the session's covariance from the data itself
        2. Use identity transform (no alignment)

        Args:
            data: Data to transform [trials, channels, samples]
            compute_cov_from_data: If True, compute covariance from data.
                                   If False, return data unchanged.

        Returns:
            Aligned data (same shape as input)
        """
        if not self.fitted:
            raise RuntimeError("EuclideanAlignment not fitted. Call fit() first.")

        if not compute_cov_from_data:
            return data

        # Compute session covariance from the data
        session_cov = compute_session_covariance(data, self.regularization)
        session_inv_sqrt = _matrix_sqrt(session_cov, inverse=True)
        transform = self.reference_sqrt @ session_inv_sqrt

        # Apply transform
        squeeze = False
        if data.ndim == 2:
            data = data[np.newaxis, ...]
            squeeze = True

        aligned = np.einsum('ij,njt->nit', transform, data)

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
            reference_cov=self.reference_cov,
            reference_sqrt=self.reference_sqrt,
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

        ea.reference_cov = data["reference_cov"]
        ea.reference_sqrt = data["reference_sqrt"]

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
        reference_method: "arithmetic" or "geometric"
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
        Aligned data
    """
    return ea.transform(data, session_id)
