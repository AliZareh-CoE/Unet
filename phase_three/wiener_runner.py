#!/usr/bin/env python3
"""
Wiener Filter Runner for Phase 3 Evaluation
============================================

Tests the Wiener filter baseline (without neural network) using the same
3-fold cross-validation strategy as the Phase 3 ablation study.

This provides a pure linear baseline to compare against neural network
and Wiener+Neural hybrid approaches.

Theory:
    The Wiener filter computes H(f) = S_xy(f) / S_xx(f) in frequency domain,
    providing the optimal linear prediction minimizing MSE.

3-Fold Cross-Validation:
    - 3-fold session-based CV (9 sessions -> 3 groups of 3)
    - Multiple seeds for variance estimation
    - Proper train/val/test separation (no leakage)

Usage:
    python -m phase_three.wiener_runner
    python -m phase_three.wiener_runner --n-seeds 3
    python -m phase_three.wiener_runner --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Custom JSON Encoder for NumPy types
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class WienerFoldResult:
    """Result from a single fold of Wiener filter evaluation.

    Metrics:
    - train_r2: R^2 on training data (fitting quality)
    - val_r2: R^2 on validation data (held-out from training sessions)
    - test_r2: R^2 on test sessions (PRIMARY - true generalization)
    """
    fold_idx: int
    seed: int
    seed_idx: int
    test_sessions: List[str]
    train_sessions: List[str]

    # Metrics
    train_r2: float = 0.0
    val_r2: float = 0.0
    test_r2: float = 0.0

    train_corr: float = 0.0
    val_corr: float = 0.0
    test_corr: float = 0.0

    train_mae: float = 0.0
    val_mae: float = 0.0
    test_mae: float = 0.0

    # Per-session test metrics
    per_session_test_r2: Dict[str, float] = field(default_factory=dict)

    # Timing
    fit_time: float = 0.0
    total_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_idx": self.fold_idx,
            "seed": self.seed,
            "seed_idx": self.seed_idx,
            "test_sessions": self.test_sessions,
            "train_sessions": self.train_sessions,
            "train_r2": self.train_r2,
            "val_r2": self.val_r2,
            "test_r2": self.test_r2,
            "train_corr": self.train_corr,
            "val_corr": self.val_corr,
            "test_corr": self.test_corr,
            "train_mae": self.train_mae,
            "val_mae": self.val_mae,
            "test_mae": self.test_mae,
            "per_session_test_r2": self.per_session_test_r2,
            "fit_time": self.fit_time,
            "total_time": self.total_time,
        }


@dataclass
class WienerResult:
    """Aggregated results across all folds."""
    fold_results: List[WienerFoldResult] = field(default_factory=list)

    # Statistics (computed from fold results)
    mean_test_r2: float = 0.0
    std_test_r2: float = 0.0
    mean_val_r2: float = 0.0
    std_val_r2: float = 0.0
    mean_train_r2: float = 0.0

    # Per-fold test R2s for plotting
    test_r2s: List[float] = field(default_factory=list)
    val_r2s: List[float] = field(default_factory=list)

    def compute_statistics(self):
        """Compute aggregate statistics from fold results."""
        if not self.fold_results:
            return

        self.test_r2s = [r.test_r2 for r in self.fold_results]
        self.val_r2s = [r.val_r2 for r in self.fold_results]
        train_r2s = [r.train_r2 for r in self.fold_results]

        self.mean_test_r2 = float(np.mean(self.test_r2s))
        self.std_test_r2 = float(np.std(self.test_r2s))
        self.mean_val_r2 = float(np.mean(self.val_r2s))
        self.std_val_r2 = float(np.std(self.val_r2s))
        self.mean_train_r2 = float(np.mean(train_r2s))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_test_r2": self.mean_test_r2,
            "std_test_r2": self.std_test_r2,
            "mean_val_r2": self.mean_val_r2,
            "std_val_r2": self.std_val_r2,
            "mean_train_r2": self.mean_train_r2,
            "test_r2s": self.test_r2s,
            "val_r2s": self.val_r2s,
            "fold_results": [r.to_dict() for r in self.fold_results],
        }


# =============================================================================
# 3-Fold Session Splits (Same as Phase 3 runner)
# =============================================================================

def get_3fold_session_splits(all_sessions: List[str]) -> List[Dict[str, Any]]:
    """Create 3-fold CV splits where each fold holds out ~3 sessions.

    For 9 sessions [s0, s1, ..., s8]:
    - Fold 0: test=[s0,s1,s2], train=[s3,s4,s5,s6,s7,s8]
    - Fold 1: test=[s3,s4,s5], train=[s0,s1,s2,s6,s7,s8]
    - Fold 2: test=[s6,s7,s8], train=[s0,s1,s2,s3,s4,s5]
    """
    n_sessions = len(all_sessions)
    if n_sessions < 6:
        raise ValueError(f"Need at least 6 sessions for 3-fold CV, got {n_sessions}")

    sessions_per_fold = n_sessions // 3
    remainder = n_sessions % 3

    splits = []
    start_idx = 0

    for fold_idx in range(3):
        n_test = sessions_per_fold + (1 if fold_idx < remainder else 0)
        end_idx = start_idx + n_test

        test_sessions = all_sessions[start_idx:end_idx]
        train_sessions = all_sessions[:start_idx] + all_sessions[end_idx:]

        splits.append({
            "fold_idx": fold_idx,
            "test_sessions": test_sessions,
            "train_sessions": train_sessions,
        })

        start_idx = end_idx

    return splits


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_olfactory_data() -> Tuple[NDArray, NDArray, NDArray, Dict[int, str]]:
    """Load raw olfactory data with session info for LOSO-style evaluation.

    Returns:
        X: Raw input signals (OB) [N, C, T]
        y: Raw target signals (PCx) [N, C, T]
        session_ids: Session ID per trial [N]
        idx_to_session: Map from session int ID to session name
    """
    from data import load_signals, extract_window, load_session_ids, DATA_PATH, ODOR_CSV_PATH

    print("Loading raw olfactory data...")

    # Load raw signals
    signals = load_signals(DATA_PATH)
    windowed = extract_window(signals)  # [N, 2, C, T]
    X = windowed[:, 0]  # OB
    y = windowed[:, 1]  # PCx

    # Load session info
    session_ids, session_to_idx, idx_to_session = load_session_ids(
        ODOR_CSV_PATH, session_column="recording_id", num_trials=X.shape[0]
    )

    print(f"  Loaded: OB {X.shape} -> PCx {y.shape}")
    print(f"  Sessions: {len(idx_to_session)} unique")

    return X, y, session_ids, idx_to_session


def load_raw_pfc_data() -> Tuple[NDArray, NDArray, NDArray, Dict[int, str]]:
    """Load raw PFC->HPC data with session info.

    Returns:
        X: Raw input signals (PFC) [N, C, T]
        y: Raw target signals (HPC) [N, C, T]
        session_ids: Session ID per trial [N]
        idx_to_session: Map from session int ID to session name
    """
    from data import load_pfc_signals, load_pfc_session_ids, PFC_DATA_PATH, PFC_META_PATH

    print("Loading raw PFC->HPC data...")

    # Load raw signals
    signals = load_pfc_signals(PFC_DATA_PATH)
    X = signals[:, 0]  # PFC
    y = signals[:, 1]  # HPC

    # Load session info
    session_ids, session_to_idx, idx_to_session = load_pfc_session_ids(PFC_META_PATH)

    print(f"  Loaded: PFC {X.shape} -> HPC {y.shape}")
    print(f"  Sessions: {len(idx_to_session)} unique")

    return X, y, session_ids, idx_to_session


def load_raw_pcx1_data(
    window_size: int = 5000,
    stride_ratio: float = 0.5,
) -> Tuple[NDArray, NDArray, NDArray, Dict[int, str]]:
    """Load raw PCx1 continuous data with session info.

    Args:
        window_size: Window size in samples
        stride_ratio: Stride as fraction of window size

    Returns:
        X: Raw input signals (OB) [N, C, T]
        y: Raw target signals (PCx) [N, C, T]
        session_ids: Session ID per window [N]
        idx_to_session: Map from session int ID to session name
    """
    from data import list_pcx1_sessions, load_pcx1_session

    print("Loading raw PCx1 continuous data...")

    sessions = list_pcx1_sessions()
    stride = int(window_size * stride_ratio)

    all_X = []
    all_y = []
    all_session_ids = []
    session_to_idx = {s: i for i, s in enumerate(sessions)}
    idx_to_session = {i: s for i, s in enumerate(sessions)}

    for sess_idx, sess_name in enumerate(sessions):
        ob, pcx = load_pcx1_session(sess_name)  # [C, T_total]

        # Sliding window
        n_windows = (ob.shape[1] - window_size) // stride + 1
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            all_X.append(ob[:, start:end])
            all_y.append(pcx[:, start:end])
            all_session_ids.append(sess_idx)

    X = np.stack(all_X, axis=0)  # [N, C, T]
    y = np.stack(all_y, axis=0)
    session_ids = np.array(all_session_ids)

    print(f"  Loaded: OB {X.shape} -> PCx {y.shape}")
    print(f"  Sessions: {len(idx_to_session)} unique, {len(X)} windows")

    return X, y, session_ids, idx_to_session


def load_raw_dandi_data(
    source_region: str = "amygdala",
    target_region: str = "hippocampus",
    window_size: int = 5000,
    stride_ratio: float = 0.5,
) -> Tuple[NDArray, NDArray, NDArray, Dict[int, str]]:
    """Load raw DANDI movie data with subject info.

    Args:
        source_region: Source brain region
        target_region: Target brain region
        window_size: Window size in samples
        stride_ratio: Stride as fraction of window size

    Returns:
        X: Raw input signals [N, C, T]
        y: Raw target signals [N, C, T]
        subject_ids: Subject ID per window [N]
        idx_to_subject: Map from subject int ID to subject name
    """
    from data import _DANDI_DATA_DIR, list_dandi_nwb_files, load_dandi_regions

    print(f"Loading raw DANDI data ({source_region} -> {target_region})...")

    nwb_files = list_dandi_nwb_files(_DANDI_DATA_DIR)
    stride = int(window_size * stride_ratio)

    # Extract unique subjects
    subjects = []
    for f in nwb_files:
        stem = f.stem
        if "sub-" in stem:
            subj_id = stem.split("_")[0]
        else:
            subj_id = stem
        if subj_id not in subjects:
            subjects.append(subj_id)
    subjects = sorted(subjects)

    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    idx_to_subject = {i: s for i, s in enumerate(subjects)}

    all_X = []
    all_y = []
    all_subject_ids = []

    for nwb_file in nwb_files:
        stem = nwb_file.stem
        if "sub-" in stem:
            subj_id = stem.split("_")[0]
        else:
            subj_id = stem
        subj_idx = subject_to_idx[subj_id]

        try:
            source, target = load_dandi_regions(
                nwb_file, source_region, target_region
            )  # [C, T_total]
        except Exception as e:
            print(f"  Skipping {nwb_file.name}: {e}")
            continue

        # Sliding window
        min_len = min(source.shape[1], target.shape[1])
        n_windows = (min_len - window_size) // stride + 1
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            all_X.append(source[:, start:end])
            all_y.append(target[:, start:end])
            all_subject_ids.append(subj_idx)

    if not all_X:
        raise RuntimeError("No valid windows loaded from DANDI data")

    X = np.stack(all_X, axis=0)
    y = np.stack(all_y, axis=0)
    subject_ids = np.array(all_subject_ids)

    print(f"  Loaded: {source_region} {X.shape} -> {target_region} {y.shape}")
    print(f"  Subjects: {len(idx_to_subject)} unique, {len(X)} windows")

    return X, y, subject_ids, idx_to_subject


def load_dataset_data(
    dataset: str,
    pcx1_window_size: int = 5000,
    pcx1_stride_ratio: float = 0.5,
    dandi_source_region: str = "amygdala",
    dandi_target_region: str = "hippocampus",
    dandi_window_size: int = 5000,
    dandi_stride_ratio: float = 0.5,
) -> Tuple[NDArray, NDArray, NDArray, Dict[int, str]]:
    """Load data for any supported dataset.

    Args:
        dataset: Dataset name (olfactory, pfc, pcx1, dandi)
        pcx1_window_size: Window size for pcx1
        pcx1_stride_ratio: Stride ratio for pcx1
        dandi_source_region: Source region for dandi
        dandi_target_region: Target region for dandi
        dandi_window_size: Window size for dandi
        dandi_stride_ratio: Stride ratio for dandi

    Returns:
        X, y, session_ids, idx_to_session
    """
    if dataset == "olfactory":
        return load_raw_olfactory_data()
    elif dataset == "pfc":
        return load_raw_pfc_data()
    elif dataset == "pcx1":
        return load_raw_pcx1_data(pcx1_window_size, pcx1_stride_ratio)
    elif dataset == "dandi":
        return load_raw_dandi_data(
            dandi_source_region, dandi_target_region,
            dandi_window_size, dandi_stride_ratio
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: olfactory, pfc, pcx1, dandi")


# =============================================================================
# Normalization (per-channel, computed on training data only)
# =============================================================================

def per_channel_normalize(
    X: NDArray,
    mean: Optional[NDArray] = None,
    std: Optional[NDArray] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Per-channel z-score normalization.

    Args:
        X: Input array [N, C, T]
        mean: Optional precomputed mean [C] (if None, compute from X)
        std: Optional precomputed std [C] (if None, compute from X)

    Returns:
        X_norm: Normalized array [N, C, T]
        mean: Mean used [C]
        std: Std used [C]
    """
    if mean is None:
        mean = X.mean(axis=(0, 2), keepdims=True)  # [1, C, 1]
    if std is None:
        std = X.std(axis=(0, 2), keepdims=True)  # [1, C, 1]

    # Avoid division by zero
    std = np.where(std < 1e-8, 1.0, std)

    X_norm = (X - mean) / std
    return X_norm, mean.squeeze(), std.squeeze()


# =============================================================================
# Wiener Filter Implementation (NumPy version for baselines)
# =============================================================================

class WienerFilter:
    """Wiener filter for neural signal translation.

    Computes optimal linear filter H(f) = S_xy(f) / S_xx(f) in frequency domain.
    Each output channel is predicted from its corresponding input channel.
    """

    def __init__(
        self,
        n_fft: Optional[int] = None,
        regularization: float = 1e-6,
    ):
        self.n_fft_init = n_fft
        self.n_fft: Optional[int] = None
        self.regularization = regularization
        self.H: Optional[NDArray] = None
        self._fitted = False

    def fit(self, X: NDArray, y: NDArray) -> "WienerFilter":
        """Fit Wiener filter from input-output pairs.

        Args:
            X: Input signals [N, C_in, T]
            y: Target signals [N, C_out, T]

        Returns:
            self
        """
        N, C_in, T = X.shape
        _, C_out, _ = y.shape

        # Auto-determine FFT size
        if self.n_fft_init is None:
            self.n_fft = int(2 ** np.ceil(np.log2(T)))
        else:
            self.n_fft = self.n_fft_init

        # Compute FFT
        X_fft = np.fft.rfft(X, n=self.n_fft, axis=-1)  # [N, C_in, n_freq]
        y_fft = np.fft.rfft(y, n=self.n_fft, axis=-1)  # [N, C_out, n_freq]

        n_freq = X_fft.shape[-1]

        # Initialize filter: diagonal matching (channel i -> channel i)
        self.H = np.zeros((C_out, C_in, n_freq), dtype=np.complex128)
        n_matched = min(C_out, C_in)

        # Compute Wiener filter for matched channels
        for c in range(n_matched):
            # Cross-spectral density: E[Y(f) * X(f)*]
            S_xy = np.mean(y_fft[:, c, :] * np.conj(X_fft[:, c, :]), axis=0)
            # Power spectral density: E[|X(f)|^2]
            S_xx = np.mean(np.abs(X_fft[:, c, :]) ** 2, axis=0)
            # Wiener filter
            self.H[c, c, :] = S_xy / (S_xx + self.regularization)

        self._fitted = True
        self._signal_length = T
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Apply Wiener filter to input data.

        Args:
            X: Input signals [N, C_in, T]

        Returns:
            Filtered signals [N, C_out, T]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        N, C_in, T = X.shape
        C_out = self.H.shape[0]

        # FFT of input
        X_fft = np.fft.rfft(X, n=self.n_fft, axis=-1)

        # Apply filter via einsum
        y_fft = np.einsum('oif,nif->nof', self.H, X_fft)

        # Inverse FFT
        y = np.fft.irfft(y_fft, n=self.n_fft, axis=-1)

        # Truncate to original length
        y = y[..., :T]

        return y.astype(np.float32)


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_r2(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute R^2 (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_correlation(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute Pearson correlation (averaged across channels)."""
    # Flatten to [N*C, T] for correlation
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

    corrs = []
    for i in range(y_true_flat.shape[0]):
        if np.std(y_true_flat[i]) > 1e-8 and np.std(y_pred_flat[i]) > 1e-8:
            corr = np.corrcoef(y_true_flat[i], y_pred_flat[i])[0, 1]
            if not np.isnan(corr):
                corrs.append(corr)

    return float(np.mean(corrs)) if corrs else 0.0


def compute_mae(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# =============================================================================
# Single Fold Evaluation
# =============================================================================

def run_single_fold(
    X: NDArray,
    y: NDArray,
    session_ids: NDArray,
    idx_to_session: Dict[int, str],
    fold_split: Dict[str, Any],
    seed: int = 42,
    seed_idx: int = 0,
    verbose: bool = True,
) -> WienerFoldResult:
    """Run Wiener filter evaluation for a single fold.

    Args:
        X: Input signals [N, C, T]
        y: Target signals [N, C, T]
        session_ids: Session ID per trial [N]
        idx_to_session: Map from session int ID to session name
        fold_split: Dict with fold_idx, test_sessions, train_sessions
        seed: Random seed (for consistency with neural network runs)
        seed_idx: Which seed iteration this is
        verbose: Print progress

    Returns:
        WienerFoldResult with metrics
    """
    np.random.seed(seed)

    fold_idx = fold_split["fold_idx"]
    test_session_names = fold_split["test_sessions"]
    train_session_names = fold_split["train_sessions"]

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Wiener Filter | Fold {fold_idx + 1}/3 | Seed {seed_idx + 1}")
        print(f"{'='*70}")
        print(f"  Test sessions:  {test_session_names}")
        print(f"  Train sessions: {train_session_names}")

    start_time = time.time()

    # Build session name to ID mapping
    session_to_idx = {v: k for k, v in idx_to_session.items()}

    # Get indices for train and test
    test_session_ids = [session_to_idx[s] for s in test_session_names if s in session_to_idx]
    train_session_ids = [session_to_idx[s] for s in train_session_names if s in session_to_idx]

    test_mask = np.isin(session_ids, test_session_ids)
    train_mask = np.isin(session_ids, train_session_ids)

    X_train_all = X[train_mask]
    y_train_all = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    test_session_ids_arr = session_ids[test_mask]

    if verbose:
        print(f"  Train samples: {len(X_train_all)}, Test samples: {len(X_test)}")

    # Split training data into train/val (70/30)
    n_train_total = len(X_train_all)
    indices = np.random.permutation(n_train_total)
    n_train = int(0.7 * n_train_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = X_train_all[train_idx]
    y_train = y_train_all[train_idx]
    X_val = X_train_all[val_idx]
    y_val = y_train_all[val_idx]

    if verbose:
        print(f"  Train split: {len(X_train)} train, {len(X_val)} val")

    # Normalize using ONLY training statistics (no leakage)
    X_train_norm, X_mean, X_std = per_channel_normalize(X_train)
    y_train_norm, y_mean, y_std = per_channel_normalize(y_train)

    # Apply same normalization to val and test
    X_val_norm, _, _ = per_channel_normalize(X_val, X_mean[:, np.newaxis, np.newaxis], X_std[:, np.newaxis, np.newaxis])
    y_val_norm, _, _ = per_channel_normalize(y_val, y_mean[:, np.newaxis, np.newaxis], y_std[:, np.newaxis, np.newaxis])
    X_test_norm, _, _ = per_channel_normalize(X_test, X_mean[:, np.newaxis, np.newaxis], X_std[:, np.newaxis, np.newaxis])
    y_test_norm, _, _ = per_channel_normalize(y_test, y_mean[:, np.newaxis, np.newaxis], y_std[:, np.newaxis, np.newaxis])

    # Fit Wiener filter on training data
    fit_start = time.time()
    wiener = WienerFilter(regularization=1e-6)
    wiener.fit(X_train_norm, y_train_norm)
    fit_time = time.time() - fit_start

    if verbose:
        print(f"  Wiener filter fit in {fit_time:.2f}s")

    # Predict on all splits
    y_train_pred = wiener.predict(X_train_norm)
    y_val_pred = wiener.predict(X_val_norm)
    y_test_pred = wiener.predict(X_test_norm)

    # Compute metrics
    train_r2 = compute_r2(y_train_norm, y_train_pred)
    val_r2 = compute_r2(y_val_norm, y_val_pred)
    test_r2 = compute_r2(y_test_norm, y_test_pred)

    train_corr = compute_correlation(y_train_norm, y_train_pred)
    val_corr = compute_correlation(y_val_norm, y_val_pred)
    test_corr = compute_correlation(y_test_norm, y_test_pred)

    train_mae = compute_mae(y_train_norm, y_train_pred)
    val_mae = compute_mae(y_val_norm, y_val_pred)
    test_mae = compute_mae(y_test_norm, y_test_pred)

    # Per-session test R2
    per_session_test_r2 = {}
    for sess_id in test_session_ids:
        sess_name = idx_to_session[sess_id]
        sess_mask = test_session_ids_arr == sess_id
        if sess_mask.sum() > 0:
            sess_r2 = compute_r2(y_test_norm[sess_mask], y_test_pred[sess_mask])
            per_session_test_r2[sess_name] = sess_r2

    total_time = time.time() - start_time

    if verbose:
        print(f"\n  Results:")
        print(f"    Train R2: {train_r2:.4f}")
        print(f"    Val R2:   {val_r2:.4f}")
        print(f"    Test R2:  {test_r2:.4f} (PRIMARY)")
        print(f"    Test Corr: {test_corr:.4f}")
        print(f"    Per-session Test R2:")
        for sess, r2 in per_session_test_r2.items():
            print(f"      {sess}: {r2:.4f}")
        print(f"    Time: {total_time:.1f}s")

    return WienerFoldResult(
        fold_idx=fold_idx,
        seed=seed,
        seed_idx=seed_idx,
        test_sessions=test_session_names,
        train_sessions=train_session_names,
        train_r2=train_r2,
        val_r2=val_r2,
        test_r2=test_r2,
        train_corr=train_corr,
        val_corr=val_corr,
        test_corr=test_corr,
        train_mae=train_mae,
        val_mae=val_mae,
        test_mae=test_mae,
        per_session_test_r2=per_session_test_r2,
        fit_time=fit_time,
        total_time=total_time,
    )


# =============================================================================
# Main Runner
# =============================================================================

def run_wiener_evaluation(
    output_dir: Path,
    dataset: str = "olfactory",
    seed: int = 42,
    n_seeds: int = 1,
    folds_to_run: Optional[List[int]] = None,
    verbose: bool = True,
    dry_run: bool = False,
    # Dataset-specific options
    pcx1_window_size: int = 5000,
    pcx1_stride_ratio: float = 0.5,
    dandi_source_region: str = "amygdala",
    dandi_target_region: str = "hippocampus",
    dandi_window_size: int = 5000,
    dandi_stride_ratio: float = 0.5,
) -> WienerResult:
    """Run complete Wiener filter evaluation with 3-fold CV.

    Args:
        output_dir: Directory to save results
        dataset: Dataset to evaluate (olfactory, pfc, pcx1, dandi)
        seed: Base random seed
        n_seeds: Number of seeds per fold
        folds_to_run: Optional list of specific fold indices to run
        verbose: Print progress
        dry_run: If True, skip actual computation
        pcx1_window_size: Window size for pcx1 dataset
        pcx1_stride_ratio: Stride ratio for pcx1 dataset
        dandi_source_region: Source region for dandi dataset
        dandi_target_region: Target region for dandi dataset
        dandi_window_size: Window size for dandi dataset
        dandi_stride_ratio: Stride ratio for dandi dataset

    Returns:
        WienerResult with all fold results
    """
    print("\n" + "#"*70)
    print("# WIENER FILTER BASELINE EVALUATION")
    print(f"# Dataset: {dataset}")
    print("# 3-Fold Cross-Validation")
    print("#"*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, session_ids, idx_to_session = load_dataset_data(
        dataset=dataset,
        pcx1_window_size=pcx1_window_size,
        pcx1_stride_ratio=pcx1_stride_ratio,
        dandi_source_region=dandi_source_region,
        dandi_target_region=dandi_target_region,
        dandi_window_size=dandi_window_size,
        dandi_stride_ratio=dandi_stride_ratio,
    )

    # Get session names in order
    all_sessions = sorted(idx_to_session.values())
    print(f"\nAll sessions: {all_sessions}")

    # Create 3-fold splits
    fold_splits = get_3fold_session_splits(all_sessions)

    if dry_run:
        print("\n[DRY RUN] Would run the following folds:")
        for split in fold_splits:
            print(f"  Fold {split['fold_idx']}: test={split['test_sessions']}")
        return WienerResult()

    # Run all folds
    fold_results = []

    for split in fold_splits:
        fold_idx = split["fold_idx"]

        # Skip if not in folds_to_run
        if folds_to_run is not None and fold_idx not in folds_to_run:
            continue

        # Run multiple seeds per fold
        for seed_idx in range(n_seeds):
            current_seed = seed + fold_idx * n_seeds + seed_idx

            result = run_single_fold(
                X=X,
                y=y,
                session_ids=session_ids,
                idx_to_session=idx_to_session,
                fold_split=split,
                seed=current_seed,
                seed_idx=seed_idx,
                verbose=verbose,
            )

            fold_results.append(result)

            # Save fold result
            fold_file = output_dir / f"wiener_fold{fold_idx}_seed{seed_idx}_result.json"
            with open(fold_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, cls=NumpyEncoder)

    # Aggregate results
    wiener_result = WienerResult(fold_results=fold_results)
    wiener_result.compute_statistics()

    # Print summary
    print("\n" + "="*70)
    print("WIENER FILTER RESULTS SUMMARY")
    print("="*70)
    print(f"  Folds completed: {len(fold_results)}")
    print(f"  Mean Test R2: {wiener_result.mean_test_r2:.4f} +/- {wiener_result.std_test_r2:.4f}")
    print(f"  Mean Val R2:  {wiener_result.mean_val_r2:.4f} +/- {wiener_result.std_val_r2:.4f}")
    print(f"  Mean Train R2: {wiener_result.mean_train_r2:.4f}")
    print(f"\n  Per-fold Test R2: {wiener_result.test_r2s}")
    print("="*70)

    # Save final results
    result_file = output_dir / "wiener_results.json"
    with open(result_file, 'w') as f:
        json.dump(wiener_result.to_dict(), f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {result_file}")

    return wiener_result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Wiener Filter Baseline Evaluation (Phase 3 style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="olfactory",
        choices=["olfactory", "pfc", "pcx1", "dandi"],
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase_three" / "wiener_baseline",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of seeds per fold (for variance estimation)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific fold indices to run (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )

    # PCx1-specific arguments
    parser.add_argument(
        "--pcx1-window-size",
        type=int,
        default=5000,
        help="Window size for PCx1 dataset",
    )
    parser.add_argument(
        "--pcx1-stride-ratio",
        type=float,
        default=0.5,
        help="Stride ratio for PCx1 dataset",
    )

    # DANDI-specific arguments
    parser.add_argument(
        "--dandi-source-region",
        type=str,
        default="amygdala",
        help="Source region for DANDI dataset",
    )
    parser.add_argument(
        "--dandi-target-region",
        type=str,
        default="hippocampus",
        help="Target region for DANDI dataset",
    )
    parser.add_argument(
        "--dandi-window-size",
        type=int,
        default=5000,
        help="Window size for DANDI dataset",
    )
    parser.add_argument(
        "--dandi-stride-ratio",
        type=float,
        default=0.5,
        help="Stride ratio for DANDI dataset",
    )

    args = parser.parse_args()

    verbose = not args.quiet

    result = run_wiener_evaluation(
        output_dir=args.output_dir,
        dataset=args.dataset,
        seed=args.seed,
        n_seeds=args.n_seeds,
        folds_to_run=args.folds,
        verbose=verbose,
        dry_run=args.dry_run,
        pcx1_window_size=args.pcx1_window_size,
        pcx1_stride_ratio=args.pcx1_stride_ratio,
        dandi_source_region=args.dandi_source_region,
        dandi_target_region=args.dandi_target_region,
        dandi_window_size=args.dandi_window_size,
        dandi_stride_ratio=args.dandi_stride_ratio,
    )

    # Exit with appropriate code
    if result.mean_test_r2 > 0:
        print(f"\nWiener baseline complete. Test R2: {result.mean_test_r2:.4f}")
        sys.exit(0)
    else:
        print("\nWiener evaluation failed or produced zero R2.")
        sys.exit(1)


if __name__ == "__main__":
    main()
