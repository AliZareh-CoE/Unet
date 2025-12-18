#!/usr/bin/env python3
"""Augmentation Explorer: Find optimal augmentation combinations for cross-session generalization.

This script explores different augmentation strategies to find combinations that make
training data distributions "cover" validation session distributions, enabling better
cross-session generalization WITHOUT requiring expensive UNet training.

Algorithm:
1. Load train and validation data (session-split)
2. Extract multi-scale features (spectral, temporal, covariance, envelope)
3. For each augmentation configuration:
   a. Apply augmentations to training data (multiple passes for stochastic augs)
   b. Compute distribution distance metrics between augmented train and validation
   c. Score coverage (how well augmented train explains validation variance)
4. Search augmentation parameter space (grid search + smart combinations)
5. Report best configurations with visualization

Key Metrics:
- MMD (Maximum Mean Discrepancy): Measures distribution similarity in kernel space
- Wasserstein Distance: Earth mover's distance between distributions
- Feature Coverage: What fraction of validation feature space is covered by train
- Covariance Similarity: How well covariance structures match

Usage:
    python experiment/augmentation_explorer.py --n-val-sessions 4 --output-dir figures/augmentation_exploration

Author: Claude (Anthropic)
"""
from __future__ import annotations

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Limit threads BEFORE importing numpy/scipy
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ['NUMEXPR_MAX_THREADS'] = '16'

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from itertools import product
import multiprocessing as mp

import numpy as np
from scipy import signal as sp_signal
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm, eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# Data Loading
# =============================================================================

def load_data(
    n_val_sessions: int = 4,
    force_recreate_splits: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Load data with session-based splits."""
    try:
        from experiment.data import (
            load_signals, load_odor_labels, load_session_ids,
            load_or_create_session_splits, DATA_PATH, ODOR_CSV_PATH
        )
    except ModuleNotFoundError:
        from data import (
            load_signals, load_odor_labels, load_session_ids,
            load_or_create_session_splits, DATA_PATH, ODOR_CSV_PATH
        )

    print("Loading neural signals...")
    signals = load_signals(DATA_PATH)
    n_trials = signals.shape[0]
    print(f"  Loaded {n_trials} trials, shape: {signals.shape}")

    # Extract OB and PCx
    ob = signals[:, 0, :, :]  # (trials, 32, 5000)
    pcx = signals[:, 1, :, :]

    # Load labels and session info
    odors, odor_vocab = load_odor_labels(ODOR_CSV_PATH, n_trials)
    session_ids, session_to_idx, idx_to_session = load_session_ids(ODOR_CSV_PATH, num_trials=n_trials)

    n_sessions = len(idx_to_session)
    print(f"  Found {n_sessions} unique sessions")

    # Create session splits
    train_idx, val_idx, test_idx, split_info = load_or_create_session_splits(
        session_ids=session_ids,
        odors=odors,
        n_test_sessions=0,
        n_val_sessions=n_val_sessions,
        seed=seed,
        force_recreate=force_recreate_splits,
        idx_to_session=idx_to_session,
        no_test_set=True,
    )

    train_sessions = split_info.get('train_sessions', [])
    val_sessions = split_info.get('val_sessions', [])

    print(f"  Train: {len(train_idx)} trials from {len(train_sessions)} sessions")
    print(f"  Val: {len(val_idx)} trials from {len(val_sessions)} sessions")

    return {
        'ob': ob,
        'pcx': pcx,
        'odors': odors,
        'session_ids': session_ids,
        'idx_to_session': idx_to_session,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'train_sessions': train_sessions,
        'val_sessions': val_sessions,
    }


# =============================================================================
# Feature Extraction
# =============================================================================

@dataclass
class SignalFeatures:
    """Multi-scale features extracted from neural signals."""
    # Per-trial features (n_trials, n_features)
    band_power: np.ndarray        # Power in frequency bands (delta, theta, alpha, beta, gamma)
    envelope_stats: np.ndarray    # Envelope mean, std, skew, kurtosis per channel
    temporal_stats: np.ndarray    # Signal mean, std, gradient stats
    covariance_flat: np.ndarray   # Flattened upper triangle of covariance matrix
    cross_corr: np.ndarray        # Cross-correlation between channels

    # Raw for detailed analysis
    raw_covariance: np.ndarray    # (n_trials, n_channels, n_channels)
    raw_psd: np.ndarray           # (n_trials, n_channels, n_freqs)


def extract_features(
    ob: np.ndarray,
    pcx: np.ndarray,
    fs: int = 1000,
) -> SignalFeatures:
    """Extract comprehensive features from OB and PCx signals.

    Args:
        ob: OB signals (n_trials, n_channels, n_time)
        pcx: PCx signals (n_trials, n_channels, n_time)
        fs: Sampling frequency

    Returns:
        SignalFeatures dataclass with extracted features
    """
    n_trials, n_channels, n_time = ob.shape

    # Combine OB and PCx for joint analysis
    combined = np.concatenate([ob, pcx], axis=1)  # (n_trials, 64, n_time)

    # Frequency bands (Hz)
    bands = {
        'delta': (1, 4),
        'theta': (4, 12),
        'alpha': (12, 30),
        'beta': (30, 60),
        'gamma': (60, 100),
    }

    # Compute PSDs
    freqs, psd = sp_signal.welch(combined, fs=fs, nperseg=min(1024, n_time), axis=-1)
    # psd shape: (n_trials, 64, n_freqs)

    # Band power features
    band_power_list = []
    for band_name, (f_low, f_high) in bands.items():
        band_mask = (freqs >= f_low) & (freqs <= f_high)
        band_power = psd[:, :, band_mask].mean(axis=-1)  # (n_trials, 64)
        band_power_list.append(band_power)
    band_power = np.stack(band_power_list, axis=-1)  # (n_trials, 64, 5)
    band_power = band_power.reshape(n_trials, -1)  # (n_trials, 320)

    # Envelope features (using Hilbert transform)
    analytic = sp_signal.hilbert(combined, axis=-1)
    envelope = np.abs(analytic)

    envelope_mean = envelope.mean(axis=-1)  # (n_trials, 64)
    envelope_std = envelope.std(axis=-1)
    envelope_skew = np.nan_to_num(
        ((envelope - envelope_mean[:, :, None])**3).mean(axis=-1) / (envelope_std**3 + 1e-10)
    )
    envelope_kurt = np.nan_to_num(
        ((envelope - envelope_mean[:, :, None])**4).mean(axis=-1) / (envelope_std**4 + 1e-10) - 3
    )
    envelope_stats = np.concatenate([
        envelope_mean, envelope_std, envelope_skew, envelope_kurt
    ], axis=1)  # (n_trials, 256)

    # Temporal statistics
    signal_mean = combined.mean(axis=-1)  # (n_trials, 64)
    signal_std = combined.std(axis=-1)
    gradient = np.diff(combined, axis=-1)
    gradient_mean = np.abs(gradient).mean(axis=-1)
    gradient_std = gradient.std(axis=-1)
    temporal_stats = np.concatenate([
        signal_mean, signal_std, gradient_mean, gradient_std
    ], axis=1)  # (n_trials, 256)

    # Covariance matrices
    raw_covariance = np.zeros((n_trials, 64, 64), dtype=np.float32)
    for i in range(n_trials):
        raw_covariance[i] = np.cov(combined[i])

    # Flatten upper triangle
    triu_idx = np.triu_indices(64, k=1)
    covariance_flat = raw_covariance[:, triu_idx[0], triu_idx[1]]  # (n_trials, 2016)

    # Cross-correlation between OB and PCx (sample subset for speed)
    n_cross = 8  # Sample 8 OB-PCx pairs
    cross_corr = np.zeros((n_trials, n_cross * 21), dtype=np.float32)  # 21 lags
    lags = np.arange(-10, 11)  # -10ms to +10ms
    for i in range(n_trials):
        for j in range(n_cross):
            ob_ch = ob[i, j * 4, :]  # Sample every 4th channel
            pcx_ch = pcx[i, j * 4, :]
            corr = np.correlate(ob_ch, pcx_ch, mode='full')
            center = len(corr) // 2
            cross_corr[i, j*21:(j+1)*21] = corr[center-10:center+11] / (np.std(ob_ch) * np.std(pcx_ch) * n_time + 1e-10)

    return SignalFeatures(
        band_power=band_power.astype(np.float32),
        envelope_stats=envelope_stats.astype(np.float32),
        temporal_stats=temporal_stats.astype(np.float32),
        covariance_flat=covariance_flat.astype(np.float32),
        cross_corr=cross_corr.astype(np.float32),
        raw_covariance=raw_covariance,
        raw_psd=psd.astype(np.float32),
    )


def features_to_matrix(features: SignalFeatures) -> np.ndarray:
    """Convert SignalFeatures to a single feature matrix for distance computation."""
    # Normalize each feature type by its std
    parts = []
    for arr in [features.band_power, features.envelope_stats,
                features.temporal_stats, features.covariance_flat]:
        std = arr.std(axis=0, keepdims=True) + 1e-10
        parts.append(arr / std)
    return np.concatenate(parts, axis=1)


# =============================================================================
# Augmentation Functions
# =============================================================================

def aug_covariance_expansion(
    ob: np.ndarray,
    pcx: np.ndarray,
    strength: float = 0.3,
    mode: str = 'expand',
) -> Tuple[np.ndarray, np.ndarray]:
    """Covariance augmentation: perturb covariance structure to simulate session variability.

    This creates synthetic "sessions" by modifying the covariance structure of signals,
    simulating how electrode impedances and positions vary across recording sessions.

    Args:
        ob: OB signals (n_trials, n_channels, n_time)
        pcx: PCx signals (n_trials, n_channels, n_time)
        strength: Perturbation strength (0-1). Higher = more variation
        mode: 'expand' (increase variance), 'shrink' (decrease), or 'random'

    Returns:
        Augmented (ob, pcx) tuple
    """
    n_trials, n_channels, n_time = ob.shape

    # Process each trial
    ob_aug = np.zeros_like(ob)
    pcx_aug = np.zeros_like(pcx)

    for i in range(n_trials):
        # Compute covariance for OB
        ob_cov = np.cov(ob[i])
        pcx_cov = np.cov(pcx[i])

        # Eigendecomposition
        try:
            ob_eigvals, ob_eigvecs = eigh(ob_cov)
            pcx_eigvals, pcx_eigvecs = eigh(pcx_cov)
        except np.linalg.LinAlgError:
            # Fallback: no augmentation for this trial
            ob_aug[i] = ob[i]
            pcx_aug[i] = pcx[i]
            continue

        # Perturb eigenvalues
        if mode == 'expand':
            scale = 1 + strength * np.random.rand(n_channels)
        elif mode == 'shrink':
            scale = 1 - strength * np.random.rand(n_channels)
        else:  # random
            scale = 1 + strength * (2 * np.random.rand(n_channels) - 1)

        scale = np.clip(scale, 0.1, 10.0)  # Safety bounds

        # Reconstruct perturbed covariance
        ob_eigvals_new = np.maximum(ob_eigvals * scale, 1e-6)
        pcx_eigvals_new = np.maximum(pcx_eigvals * scale, 1e-6)

        # Compute transformation matrix: C_new = V * sqrt(D_new/D) * V^T * x
        # where x is whitened by C_old
        ob_transform = ob_eigvecs @ np.diag(np.sqrt(ob_eigvals_new / (ob_eigvals + 1e-10))) @ ob_eigvecs.T
        pcx_transform = pcx_eigvecs @ np.diag(np.sqrt(pcx_eigvals_new / (pcx_eigvals + 1e-10))) @ pcx_eigvecs.T

        # Apply transformation
        ob_centered = ob[i] - ob[i].mean(axis=-1, keepdims=True)
        pcx_centered = pcx[i] - pcx[i].mean(axis=-1, keepdims=True)

        ob_aug[i] = ob_transform @ ob_centered + ob[i].mean(axis=-1, keepdims=True)
        pcx_aug[i] = pcx_transform @ pcx_centered + pcx[i].mean(axis=-1, keepdims=True)

    return ob_aug, pcx_aug


def aug_cross_session_mixing(
    ob: np.ndarray,
    pcx: np.ndarray,
    session_ids: np.ndarray,
    alpha: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-session mixing: blend samples from different sessions.

    Creates virtual training examples by mixing samples from different sessions,
    encouraging the model to learn session-invariant features.

    Args:
        ob: OB signals (n_trials, n_channels, n_time)
        pcx: PCx signals (n_trials, n_channels, n_time)
        session_ids: Session ID for each trial
        alpha: Mixing coefficient (0-1). Higher = more mixing

    Returns:
        Augmented (ob, pcx) tuple
    """
    n_trials = ob.shape[0]
    unique_sessions = np.unique(session_ids)

    if len(unique_sessions) < 2:
        return ob, pcx

    ob_aug = np.zeros_like(ob)
    pcx_aug = np.zeros_like(pcx)

    for i in range(n_trials):
        my_session = session_ids[i]

        # Find trials from different sessions
        diff_session_mask = session_ids != my_session
        diff_indices = np.where(diff_session_mask)[0]

        if len(diff_indices) == 0:
            ob_aug[i] = ob[i]
            pcx_aug[i] = pcx[i]
            continue

        # Sample mixing coefficient
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5

        # Pick random partner from different session
        partner = np.random.choice(diff_indices)

        # Mix
        ob_aug[i] = lam * ob[i] + (1 - lam) * ob[partner]
        pcx_aug[i] = lam * pcx[i] + (1 - lam) * pcx[partner]

    return ob_aug, pcx_aug


def aug_euclidean_alignment(
    ob: np.ndarray,
    pcx: np.ndarray,
    reference_cov: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Euclidean Alignment: align covariance to reference (identity or mean).

    EA is a domain adaptation technique that whitens and re-colors signals
    to have a reference covariance structure.

    Args:
        ob: OB signals (n_trials, n_channels, n_time)
        pcx: PCx signals (n_trials, n_channels, n_time)
        reference_cov: Target covariance (default: identity matrix)

    Returns:
        Aligned (ob, pcx) tuple
    """
    n_trials, n_channels, n_time = ob.shape

    # Compute mean covariance across all trials
    ob_covs = np.array([np.cov(ob[i]) for i in range(n_trials)])
    pcx_covs = np.array([np.cov(pcx[i]) for i in range(n_trials)])

    ob_mean_cov = ob_covs.mean(axis=0)
    pcx_mean_cov = pcx_covs.mean(axis=0)

    # Reference is identity if not specified
    if reference_cov is None:
        reference_cov = np.eye(n_channels)

    # Compute alignment matrices using matrix square root
    try:
        ob_sqrt_inv = np.linalg.inv(sqrtm(ob_mean_cov + 1e-6 * np.eye(n_channels)))
        pcx_sqrt_inv = np.linalg.inv(sqrtm(pcx_mean_cov + 1e-6 * np.eye(n_channels)))
        ref_sqrt = sqrtm(reference_cov)
    except (np.linalg.LinAlgError, ValueError):
        return ob, pcx

    # Alignment: x_aligned = ref_sqrt @ mean_cov_sqrt_inv @ x
    ob_align = np.real(ref_sqrt @ ob_sqrt_inv)
    pcx_align = np.real(ref_sqrt @ pcx_sqrt_inv)

    ob_aligned = np.zeros_like(ob)
    pcx_aligned = np.zeros_like(pcx)

    for i in range(n_trials):
        ob_centered = ob[i] - ob[i].mean(axis=-1, keepdims=True)
        pcx_centered = pcx[i] - pcx[i].mean(axis=-1, keepdims=True)
        ob_aligned[i] = ob_align @ ob_centered + ob[i].mean(axis=-1, keepdims=True)
        pcx_aligned[i] = pcx_align @ pcx_centered + pcx[i].mean(axis=-1, keepdims=True)

    return ob_aligned, pcx_aligned


def aug_amplitude_jitter(
    ob: np.ndarray,
    pcx: np.ndarray,
    scale_range: Tuple[float, float] = (0.7, 1.3),
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel amplitude jittering."""
    n_trials, n_channels, _ = ob.shape

    scales_ob = np.random.uniform(scale_range[0], scale_range[1], (n_trials, n_channels, 1))
    scales_pcx = np.random.uniform(scale_range[0], scale_range[1], (n_trials, n_channels, 1))

    return ob * scales_ob, pcx * scales_pcx


def aug_noise_injection(
    ob: np.ndarray,
    pcx: np.ndarray,
    noise_std: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise relative to signal std."""
    ob_std = ob.std(axis=-1, keepdims=True)
    pcx_std = pcx.std(axis=-1, keepdims=True)

    ob_noise = np.random.randn(*ob.shape) * noise_std * ob_std
    pcx_noise = np.random.randn(*pcx.shape) * noise_std * pcx_std

    return ob + ob_noise, pcx + pcx_noise


def aug_time_warp(
    ob: np.ndarray,
    pcx: np.ndarray,
    warp_factor: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Local time warping using interpolation."""
    n_trials, n_channels, n_time = ob.shape

    ob_warped = np.zeros_like(ob)
    pcx_warped = np.zeros_like(pcx)

    t_orig = np.linspace(0, 1, n_time)

    for i in range(n_trials):
        # Create smooth random warp field
        n_knots = 5
        knot_positions = np.linspace(0, 1, n_knots)
        knot_offsets = np.random.randn(n_knots) * warp_factor
        knot_offsets[0] = 0  # Keep endpoints fixed
        knot_offsets[-1] = 0

        # Interpolate to full time axis
        from scipy.interpolate import interp1d
        warp_interp = interp1d(knot_positions, knot_offsets, kind='cubic', fill_value=0, bounds_error=False)
        t_warped = np.clip(t_orig + warp_interp(t_orig), 0, 1)

        # Resample signals
        for ch in range(n_channels):
            ob_interp = interp1d(t_orig, ob[i, ch], kind='linear', fill_value='extrapolate')
            pcx_interp = interp1d(t_orig, pcx[i, ch], kind='linear', fill_value='extrapolate')
            ob_warped[i, ch] = ob_interp(t_warped)
            pcx_warped[i, ch] = pcx_interp(t_warped)

    return ob_warped, pcx_warped


# =============================================================================
# Distance Metrics
# =============================================================================

def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """Compute Maximum Mean Discrepancy between two sets of samples.

    MMD measures the distance between distributions in a reproducing kernel
    Hilbert space. Lower MMD = more similar distributions.

    Args:
        X: Samples from distribution P (n_samples, n_features)
        Y: Samples from distribution Q (m_samples, n_features)
        gamma: RBF kernel bandwidth (default: 1/n_features)

    Returns:
        MMD^2 estimate
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # Subsample if too large
    max_samples = 500
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
    if len(Y) > max_samples:
        idx = np.random.choice(len(Y), max_samples, replace=False)
        Y = Y[idx]

    # RBF kernel
    def rbf_kernel(A, B):
        dists = cdist(A, B, 'sqeuclidean')
        return np.exp(-gamma * dists)

    K_XX = rbf_kernel(X, X)
    K_YY = rbf_kernel(Y, Y)
    K_XY = rbf_kernel(X, Y)

    n, m = len(X), len(Y)

    # Unbiased MMD^2 estimate
    mmd2 = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
    mmd2 += (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
    mmd2 -= 2 * K_XY.mean()

    return float(max(0, mmd2))


def compute_wasserstein_multivariate(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute approximate Wasserstein distance using sliced 1D projections.

    Uses random 1D projections and averages 1D Wasserstein distances.
    """
    n_projections = 50
    n_features = X.shape[1]

    distances = []
    for _ in range(n_projections):
        # Random unit vector
        direction = np.random.randn(n_features)
        direction /= np.linalg.norm(direction)

        # Project data
        X_proj = X @ direction
        Y_proj = Y @ direction

        # 1D Wasserstein
        dist = wasserstein_distance(X_proj, Y_proj)
        distances.append(dist)

    return float(np.mean(distances))


def compute_coverage(
    X_train: np.ndarray,
    X_val: np.ndarray,
    k: int = 5,
) -> float:
    """Compute coverage: fraction of validation points that have a training neighbor nearby.

    For each validation point, check if it has at least one training point within
    a distance threshold (based on k-th nearest neighbor in training set).

    Returns:
        Coverage fraction (0-1)
    """
    from scipy.spatial import KDTree

    # Subsample if too large
    max_samples = 1000
    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[idx]
    if len(X_val) > max_samples:
        idx = np.random.choice(len(X_val), max_samples, replace=False)
        X_val = X_val[idx]

    # Build KD-tree for training points
    tree = KDTree(X_train)

    # Compute threshold from training set k-NN distances
    train_dists, _ = tree.query(X_train, k=k+1)  # +1 because self is included
    threshold = np.percentile(train_dists[:, -1], 95)  # 95th percentile of k-NN distances

    # For each validation point, check if within threshold
    val_dists, _ = tree.query(X_val, k=1)
    covered = (val_dists <= threshold).mean()

    return float(covered)


def compute_covariance_distance(cov1: np.ndarray, cov2: np.ndarray) -> float:
    """Compute Frobenius distance between mean covariance matrices."""
    mean_cov1 = cov1.mean(axis=0)
    mean_cov2 = cov2.mean(axis=0)

    # Frobenius norm of difference
    diff = mean_cov1 - mean_cov2
    fro_dist = np.sqrt(np.sum(diff ** 2))

    # Normalize by size
    return float(fro_dist / np.sqrt(mean_cov1.shape[0] * mean_cov1.shape[1]))


# =============================================================================
# Augmentation Configuration
# =============================================================================

@dataclass
class AugmentationConfig:
    """Configuration for a set of augmentations."""
    name: str

    # Covariance augmentation
    use_cov_aug: bool = False
    cov_strength: float = 0.3
    cov_mode: str = 'random'  # 'expand', 'shrink', 'random'

    # Cross-session mixing
    use_cross_session_mix: bool = False
    cross_session_alpha: float = 0.3

    # Euclidean alignment
    use_euclidean_alignment: bool = False

    # Amplitude jitter
    use_amplitude_jitter: bool = False
    amplitude_range: Tuple[float, float] = (0.7, 1.3)

    # Noise injection
    use_noise: bool = False
    noise_std: float = 0.1

    # Time warp
    use_time_warp: bool = False
    warp_factor: float = 0.1

    # Number of augmentation passes (for averaging stochastic augmentations)
    n_passes: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'use_cov_aug': self.use_cov_aug,
            'cov_strength': self.cov_strength,
            'cov_mode': self.cov_mode,
            'use_cross_session_mix': self.use_cross_session_mix,
            'cross_session_alpha': self.cross_session_alpha,
            'use_euclidean_alignment': self.use_euclidean_alignment,
            'use_amplitude_jitter': self.use_amplitude_jitter,
            'amplitude_range': self.amplitude_range,
            'use_noise': self.use_noise,
            'noise_std': self.noise_std,
            'use_time_warp': self.use_time_warp,
            'warp_factor': self.warp_factor,
        }


def apply_augmentation(
    ob: np.ndarray,
    pcx: np.ndarray,
    session_ids: np.ndarray,
    config: AugmentationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply augmentation configuration to signals."""
    ob_aug, pcx_aug = ob.copy(), pcx.copy()

    # Euclidean alignment (applied first, deterministic)
    if config.use_euclidean_alignment:
        ob_aug, pcx_aug = aug_euclidean_alignment(ob_aug, pcx_aug)

    # Covariance augmentation
    if config.use_cov_aug:
        ob_aug, pcx_aug = aug_covariance_expansion(
            ob_aug, pcx_aug,
            strength=config.cov_strength,
            mode=config.cov_mode
        )

    # Cross-session mixing
    if config.use_cross_session_mix:
        ob_aug, pcx_aug = aug_cross_session_mixing(
            ob_aug, pcx_aug,
            session_ids=session_ids,
            alpha=config.cross_session_alpha
        )

    # Amplitude jitter
    if config.use_amplitude_jitter:
        ob_aug, pcx_aug = aug_amplitude_jitter(
            ob_aug, pcx_aug,
            scale_range=config.amplitude_range
        )

    # Noise injection
    if config.use_noise:
        ob_aug, pcx_aug = aug_noise_injection(
            ob_aug, pcx_aug,
            noise_std=config.noise_std
        )

    # Time warp
    if config.use_time_warp:
        ob_aug, pcx_aug = aug_time_warp(
            ob_aug, pcx_aug,
            warp_factor=config.warp_factor
        )

    return ob_aug, pcx_aug


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class EvaluationResult:
    """Results from evaluating an augmentation configuration."""
    config: AugmentationConfig

    # Distance metrics (lower is better for these)
    mmd: float
    wasserstein: float
    cov_distance: float

    # Coverage metric (higher is better)
    coverage: float

    # Combined score (higher is better)
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': self.config.to_dict(),
            'mmd': self.mmd,
            'wasserstein': self.wasserstein,
            'cov_distance': self.cov_distance,
            'coverage': self.coverage,
            'score': self.score,
        }


def evaluate_augmentation(
    ob_train: np.ndarray,
    pcx_train: np.ndarray,
    ob_val: np.ndarray,
    pcx_val: np.ndarray,
    session_ids_train: np.ndarray,
    val_features: SignalFeatures,
    config: AugmentationConfig,
) -> EvaluationResult:
    """Evaluate how well an augmentation configuration covers validation distribution."""

    # Apply augmentation multiple times and aggregate features
    all_features = []
    all_covs = []

    for _ in range(config.n_passes):
        ob_aug, pcx_aug = apply_augmentation(
            ob_train, pcx_train, session_ids_train, config
        )

        # Extract features
        aug_features = extract_features(ob_aug, pcx_aug)
        all_features.append(features_to_matrix(aug_features))
        all_covs.append(aug_features.raw_covariance)

    # Combine all passes
    train_feat_matrix = np.concatenate(all_features, axis=0)
    train_cov = np.concatenate(all_covs, axis=0)

    # Validation feature matrix
    val_feat_matrix = features_to_matrix(val_features)

    # Compute metrics
    mmd = compute_mmd(train_feat_matrix, val_feat_matrix)
    wasserstein = compute_wasserstein_multivariate(train_feat_matrix, val_feat_matrix)
    coverage = compute_coverage(train_feat_matrix, val_feat_matrix)
    cov_distance = compute_covariance_distance(train_cov, val_features.raw_covariance)

    # Combined score: maximize coverage, minimize distances
    # Normalize and combine (higher is better)
    score = coverage - 0.3 * mmd - 0.2 * wasserstein - 0.1 * cov_distance

    return EvaluationResult(
        config=config,
        mmd=mmd,
        wasserstein=wasserstein,
        cov_distance=cov_distance,
        coverage=coverage,
        score=score,
    )


# =============================================================================
# Search Space
# =============================================================================

def generate_search_space() -> List[AugmentationConfig]:
    """Generate comprehensive search space of augmentation configurations."""
    configs = []

    # Baseline: no augmentation
    configs.append(AugmentationConfig(name='baseline'))

    # Individual augmentations
    configs.append(AugmentationConfig(name='ea_only', use_euclidean_alignment=True))

    for strength in [0.1, 0.2, 0.3, 0.5, 0.7]:
        configs.append(AugmentationConfig(
            name=f'cov_expand_{strength}',
            use_cov_aug=True, cov_strength=strength, cov_mode='expand'
        ))
        configs.append(AugmentationConfig(
            name=f'cov_shrink_{strength}',
            use_cov_aug=True, cov_strength=strength, cov_mode='shrink'
        ))
        configs.append(AugmentationConfig(
            name=f'cov_random_{strength}',
            use_cov_aug=True, cov_strength=strength, cov_mode='random'
        ))

    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
        configs.append(AugmentationConfig(
            name=f'cross_session_{alpha}',
            use_cross_session_mix=True, cross_session_alpha=alpha
        ))

    for noise in [0.05, 0.1, 0.15, 0.2]:
        configs.append(AugmentationConfig(
            name=f'noise_{noise}',
            use_noise=True, noise_std=noise
        ))

    # Combinations: EA + covariance
    for strength in [0.2, 0.3, 0.5]:
        configs.append(AugmentationConfig(
            name=f'ea_cov_{strength}',
            use_euclidean_alignment=True,
            use_cov_aug=True, cov_strength=strength, cov_mode='random'
        ))

    # Combinations: EA + cross-session mixing
    for alpha in [0.2, 0.3, 0.4]:
        configs.append(AugmentationConfig(
            name=f'ea_csm_{alpha}',
            use_euclidean_alignment=True,
            use_cross_session_mix=True, cross_session_alpha=alpha
        ))

    # Combinations: covariance + cross-session
    for strength in [0.2, 0.3]:
        for alpha in [0.2, 0.3]:
            configs.append(AugmentationConfig(
                name=f'cov_{strength}_csm_{alpha}',
                use_cov_aug=True, cov_strength=strength, cov_mode='random',
                use_cross_session_mix=True, cross_session_alpha=alpha
            ))

    # Full combinations
    for strength in [0.2, 0.3, 0.5]:
        for alpha in [0.2, 0.3, 0.4]:
            configs.append(AugmentationConfig(
                name=f'ea_cov_{strength}_csm_{alpha}',
                use_euclidean_alignment=True,
                use_cov_aug=True, cov_strength=strength, cov_mode='random',
                use_cross_session_mix=True, cross_session_alpha=alpha
            ))

    # Kitchen sink (all augmentations)
    for strength in [0.2, 0.3]:
        for alpha in [0.2, 0.3]:
            for noise in [0.05, 0.1]:
                configs.append(AugmentationConfig(
                    name=f'full_{strength}_{alpha}_{noise}',
                    use_euclidean_alignment=True,
                    use_cov_aug=True, cov_strength=strength, cov_mode='random',
                    use_cross_session_mix=True, cross_session_alpha=alpha,
                    use_noise=True, noise_std=noise,
                    use_amplitude_jitter=True, amplitude_range=(0.8, 1.2),
                ))

    return configs


# =============================================================================
# Visualization
# =============================================================================

def plot_results(
    results: List[EvaluationResult],
    output_dir: Path,
):
    """Generate visualization of exploration results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by score
    results_sorted = sorted(results, key=lambda r: r.score, reverse=True)

    # Top 20 configurations
    top_n = min(20, len(results_sorted))
    top_results = results_sorted[:top_n]

    # Figure 1: Score bar chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    names = [r.config.name for r in top_results]
    scores = [r.score for r in top_results]

    ax = axes[0, 0]
    bars = ax.barh(range(top_n), scores)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Combined Score (higher is better)')
    ax.set_title(f'Top {top_n} Augmentation Configurations')
    ax.invert_yaxis()

    # Color by score
    for bar, score in zip(bars, scores):
        bar.set_color(plt.cm.RdYlGn((score - min(scores)) / (max(scores) - min(scores) + 1e-10)))

    # Figure 2: MMD comparison
    ax = axes[0, 1]
    mmds = [r.mmd for r in top_results]
    bars = ax.barh(range(top_n), mmds)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('MMD (lower is better)')
    ax.set_title('Maximum Mean Discrepancy')
    ax.invert_yaxis()

    for bar, mmd in zip(bars, mmds):
        bar.set_color(plt.cm.RdYlGn_r((mmd - min(mmds)) / (max(mmds) - min(mmds) + 1e-10)))

    # Figure 3: Coverage
    ax = axes[1, 0]
    coverages = [r.coverage for r in top_results]
    bars = ax.barh(range(top_n), coverages)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Coverage (higher is better)')
    ax.set_title('Validation Coverage by Augmented Training')
    ax.invert_yaxis()

    for bar, cov in zip(bars, coverages):
        bar.set_color(plt.cm.RdYlGn((cov - min(coverages)) / (max(coverages) - min(coverages) + 1e-10)))

    # Figure 4: Wasserstein
    ax = axes[1, 1]
    wass = [r.wasserstein for r in top_results]
    bars = ax.barh(range(top_n), wass)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Wasserstein Distance (lower is better)')
    ax.set_title('Sliced Wasserstein Distance')
    ax.invert_yaxis()

    for bar, w in zip(bars, wass):
        bar.set_color(plt.cm.RdYlGn_r((w - min(wass)) / (max(wass) - min(wass) + 1e-10)))

    plt.tight_layout()
    plt.savefig(output_dir / 'augmentation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 5: Metric correlations scatter plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    all_scores = [r.score for r in results]
    all_mmds = [r.mmd for r in results]
    all_coverages = [r.coverage for r in results]
    all_wass = [r.wasserstein for r in results]

    ax = axes[0]
    ax.scatter(all_mmds, all_scores, alpha=0.5)
    ax.set_xlabel('MMD')
    ax.set_ylabel('Score')
    ax.set_title('Score vs MMD')

    ax = axes[1]
    ax.scatter(all_coverages, all_scores, alpha=0.5)
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Score')
    ax.set_title('Score vs Coverage')

    ax = axes[2]
    ax.scatter(all_wass, all_scores, alpha=0.5)
    ax.set_xlabel('Wasserstein')
    ax.set_ylabel('Score')
    ax.set_title('Score vs Wasserstein')

    plt.tight_layout()
    plt.savefig(output_dir / 'metric_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


def plot_feature_space(
    train_features: SignalFeatures,
    val_features: SignalFeatures,
    aug_features: SignalFeatures,
    config_name: str,
    output_dir: Path,
):
    """Plot PCA/UMAP of feature space showing train, val, and augmented."""
    from sklearn.decomposition import PCA

    train_matrix = features_to_matrix(train_features)
    val_matrix = features_to_matrix(val_features)
    aug_matrix = features_to_matrix(aug_features)

    # Combine for PCA
    combined = np.vstack([train_matrix, val_matrix, aug_matrix])

    # Fit PCA
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined)

    n_train = len(train_matrix)
    n_val = len(val_matrix)

    train_pca = combined_pca[:n_train]
    val_pca = combined_pca[n_train:n_train+n_val]
    aug_pca = combined_pca[n_train+n_val:]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(train_pca[:, 0], train_pca[:, 1], alpha=0.3, label='Train (original)', c='blue', s=20)
    ax.scatter(val_pca[:, 0], val_pca[:, 1], alpha=0.5, label='Validation', c='red', s=30)
    ax.scatter(aug_pca[:, 0], aug_pca[:, 1], alpha=0.3, label=f'Train (augmented: {config_name})', c='green', s=20)

    ax.legend()
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'Feature Space: {config_name}')

    plt.tight_layout()
    plt.savefig(output_dir / f'feature_space_{config_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Augmentation Explorer for Cross-Session Generalization')
    parser.add_argument('--n-val-sessions', type=int, default=4,
                        help='Number of sessions to hold out for validation')
    parser.add_argument('--output-dir', type=str, default='figures/augmentation_exploration',
                        help='Output directory for results')
    parser.add_argument('--force-recreate-splits', action='store_true',
                        help='Force recreation of session splits')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs (experimental)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AUGMENTATION EXPLORER FOR CROSS-SESSION GENERALIZATION")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    data = load_data(
        n_val_sessions=args.n_val_sessions,
        force_recreate_splits=args.force_recreate_splits,
        seed=args.seed,
    )

    ob = data['ob']
    pcx = data['pcx']
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    session_ids = data['session_ids']

    ob_train, pcx_train = ob[train_idx], pcx[train_idx]
    ob_val, pcx_val = ob[val_idx], pcx[val_idx]
    session_ids_train = session_ids[train_idx]

    print(f"   Train: {len(train_idx)} trials")
    print(f"   Val: {len(val_idx)} trials")

    # Extract baseline features
    print("\n2. Extracting features...")
    print("   Training features...")
    train_features = extract_features(ob_train, pcx_train)
    print("   Validation features...")
    val_features = extract_features(ob_val, pcx_val)

    # Generate search space
    print("\n3. Generating augmentation search space...")
    configs = generate_search_space()
    print(f"   {len(configs)} configurations to evaluate")

    # Evaluate each configuration
    print("\n4. Evaluating augmentation configurations...")
    results = []

    for config in tqdm(configs, desc="Evaluating"):
        result = evaluate_augmentation(
            ob_train, pcx_train, ob_val, pcx_val,
            session_ids_train, val_features, config
        )
        results.append(result)

    # Sort by score
    results_sorted = sorted(results, key=lambda r: r.score, reverse=True)

    # Print top results
    print("\n" + "=" * 70)
    print("TOP 10 AUGMENTATION CONFIGURATIONS")
    print("=" * 70)

    for i, result in enumerate(results_sorted[:10]):
        print(f"\n{i+1}. {result.config.name}")
        print(f"   Score: {result.score:.4f}")
        print(f"   Coverage: {result.coverage:.4f}")
        print(f"   MMD: {result.mmd:.4f}")
        print(f"   Wasserstein: {result.wasserstein:.4f}")
        print(f"   Cov Distance: {result.cov_distance:.4f}")

    # Save results
    print("\n5. Saving results...")
    results_dict = [r.to_dict() for r in results_sorted]
    with open(output_dir / 'augmentation_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Generate plots
    print("\n6. Generating visualizations...")
    plot_results(results, output_dir)

    # Plot feature space for top 3 configurations
    print("   Plotting feature spaces for top configurations...")
    for result in results_sorted[:3]:
        ob_aug, pcx_aug = apply_augmentation(
            ob_train, pcx_train, session_ids_train, result.config
        )
        aug_features = extract_features(ob_aug, pcx_aug)
        plot_feature_space(train_features, val_features, aug_features, result.config.name, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best = results_sorted[0]
    baseline = next(r for r in results if r.config.name == 'baseline')

    print(f"\nBaseline (no augmentation):")
    print(f"  Score: {baseline.score:.4f}, Coverage: {baseline.coverage:.4f}, MMD: {baseline.mmd:.4f}")

    print(f"\nBest configuration: {best.config.name}")
    print(f"  Score: {best.score:.4f} (+{best.score - baseline.score:.4f})")
    print(f"  Coverage: {best.coverage:.4f} (+{best.coverage - baseline.coverage:.4f})")
    print(f"  MMD: {best.mmd:.4f} ({best.mmd - baseline.mmd:+.4f})")

    print(f"\nConfiguration details:")
    for key, value in best.config.to_dict().items():
        if key != 'name':
            print(f"  {key}: {value}")

    print(f"\nResults saved to: {output_dir}/")
    print(f"  - augmentation_results.json (full results)")
    print(f"  - augmentation_comparison.png (comparison charts)")
    print(f"  - metric_correlations.png (metric scatter plots)")
    print(f"  - feature_space_*.png (PCA visualizations)")


if __name__ == '__main__':
    main()
