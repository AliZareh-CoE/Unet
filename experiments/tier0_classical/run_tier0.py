#!/usr/bin/env python3
"""
Tier 0: Classical Floor (GATE) - PERFORMANCE OPTIMIZED
=======================================================

Establishes the minimum performance bar that neural methods must beat.
Uses 5-fold CV for robust estimation.

GATE CONDITION: Neural methods in Tier 1+ must beat best classical by 0.10 R²

PERFORMANCE FEATURES:
- Parallel CV fold processing (via joblib)
- Parallel baseline evaluation (optional)
- GPU acceleration for Wiener/Ridge (via CuPy)
- Vectorized metrics and bootstrap CI
- OpenBLAS/MKL threading for NumPy operations

Usage:
    python experiments/tier0_classical/run_tier0.py
    python experiments/tier0_classical/run_tier0.py --dry-run
    python experiments/tier0_classical/run_tier0.py --parallel-baselines --n-jobs -1
    python experiments/tier0_classical/run_tier0.py --use-gpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# PERFORMANCE: Configure NumPy/BLAS threading BEFORE importing NumPy
# =============================================================================
# These must be set before numpy import for OpenBLAS/MKL
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())

# =============================================================================
# CRITICAL: Fix Python path for imports
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import multiprocessing as mp

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from experiments.common.cross_validation import create_data_splits
from experiments.common.config_registry import (
    ClassicalResult,
    Tier0Result,
    get_registry,
)
from experiments.study5_classical.baselines import create_baseline, BASELINE_REGISTRY


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier0_classical"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# All baselines to evaluate - these are deterministic, no seeds needed
BASELINES = list(BASELINE_REGISTRY.keys())

# Fast baselines for dry-run
FAST_BASELINES = ["ridge", "wiener"]

# Memory configuration for large datasets
def get_optimal_batch_config(n_samples: int, n_channels: int, time_steps: int, available_gb: float = None) -> Dict[str, int]:
    """Determine optimal batch sizes based on available memory.

    Args:
        n_samples: Number of samples
        n_channels: Number of channels
        time_steps: Time steps per sample
        available_gb: Available RAM in GB (auto-detect if None)

    Returns:
        Dict with optimal configuration
    """
    import psutil

    if available_gb is None:
        mem = psutil.virtual_memory()
        available_gb = mem.available / 1e9

    # Estimate bytes per sample (float32 = 4 bytes)
    bytes_per_sample = n_channels * time_steps * 4 * 2  # *2 for X and y

    # Target 50% memory usage for data, leaving room for operations
    target_memory = available_gb * 0.5 * 1e9
    max_samples_in_memory = int(target_memory / bytes_per_sample)

    # Ensure at least 1 sample per batch
    batch_size = max(1, min(n_samples, max_samples_in_memory))

    # For metrics computation, we can use full batches
    metrics_batch = batch_size

    print(f"  Memory configuration: {available_gb:.1f} GB available")
    print(f"  Optimal batch size: {batch_size} samples ({batch_size * bytes_per_sample / 1e9:.2f} GB)")

    return {
        "batch_size": batch_size,
        "metrics_batch": metrics_batch,
        "max_samples_in_memory": max_samples_in_memory,
    }


# =============================================================================
# Metrics (Publication-Quality for Nature Methods)
# =============================================================================

# Neural frequency bands
NEURAL_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 100),
}


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, sample_rate: float = 1000.0) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for Nature Methods.

    FULLY VECTORIZED for maximum CPU/memory efficiency.

    Includes:
    - R² (per-channel, then averaged - correct multi-output formulation)
    - MAE
    - Pearson correlation (all channels)
    - Per-frequency-band R² (delta, theta, alpha, beta, gamma)
    - PSD error in dB

    Args:
        y_pred: Predicted values [N, C, T]
        y_true: True values [N, C, T]
        sample_rate: Sampling rate in Hz (for PSD computation)

    Returns:
        Dictionary of metrics
    """
    # Ensure 3D shape [N, C, T]
    if y_pred.ndim == 2:
        y_pred = y_pred[:, np.newaxis, :]
        y_true = y_true[:, np.newaxis, :]

    N, C, T = y_pred.shape

    # =========================================================================
    # R² - FULLY VECTORIZED per-channel computation
    # =========================================================================
    # Reshape to [C, N*T] for vectorized computation across all channels at once
    pred_flat = y_pred.transpose(1, 0, 2).reshape(C, -1)  # [C, N*T]
    true_flat = y_true.transpose(1, 0, 2).reshape(C, -1)  # [C, N*T]

    # Compute SS_res and SS_tot for all channels simultaneously
    ss_res = np.sum((true_flat - pred_flat) ** 2, axis=1)  # [C]
    true_means = np.mean(true_flat, axis=1, keepdims=True)  # [C, 1]
    ss_tot = np.sum((true_flat - true_means) ** 2, axis=1)  # [C]

    r2_per_channel = 1 - ss_res / (ss_tot + 1e-8)  # [C]
    r2 = float(np.mean(r2_per_channel))
    r2_std_across_channels = float(np.std(r2_per_channel))

    # =========================================================================
    # MAE - already vectorized
    # =========================================================================
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # =========================================================================
    # Pearson correlation - FULLY VECTORIZED
    # =========================================================================
    # Compute for all channels at once
    pred_centered = pred_flat - np.mean(pred_flat, axis=1, keepdims=True)  # [C, N*T]
    true_centered = true_flat - np.mean(true_flat, axis=1, keepdims=True)  # [C, N*T]

    numerator = np.sum(pred_centered * true_centered, axis=1)  # [C]
    denominator = np.sqrt(np.sum(pred_centered ** 2, axis=1) * np.sum(true_centered ** 2, axis=1))  # [C]

    correlations = numerator / (denominator + 1e-8)  # [C]
    valid_mask = np.isfinite(correlations)
    pearson = float(np.mean(correlations[valid_mask])) if valid_mask.any() else 0.0
    pearson_std = float(np.std(correlations[valid_mask])) if valid_mask.sum() > 1 else 0.0

    # =========================================================================
    # PSD Error in dB (spectral fidelity) - vectorized via scipy
    # =========================================================================
    try:
        from scipy import signal as scipy_signal

        # Compute PSD using Welch's method (already vectorized over all axes)
        _, psd_pred = scipy_signal.welch(y_pred, fs=sample_rate, axis=-1, nperseg=min(256, T))
        _, psd_true = scipy_signal.welch(y_true, fs=sample_rate, axis=-1, nperseg=min(256, T))

        # Log-space error (dB)
        log_pred = np.log10(psd_pred + 1e-10)
        log_true = np.log10(psd_true + 1e-10)
        psd_error_db = float(10 * np.mean(np.abs(log_pred - log_true)))
    except Exception:
        psd_error_db = np.nan

    # =========================================================================
    # Per-Frequency-Band R² - VECTORIZED band filtering
    # =========================================================================
    band_r2 = {}
    try:
        from scipy import signal as scipy_signal

        # Pre-design all filters
        filters = {}
        nyq = sample_rate / 2
        for band_name, (f_low, f_high) in NEURAL_BANDS.items():
            if f_low >= nyq:
                continue
            low = f_low / nyq
            high = min(f_high / nyq, 0.99)
            if low < high:
                try:
                    filters[band_name] = scipy_signal.butter(4, [low, high], btype='band')
                except Exception:
                    pass

        # Apply all filters (scipy.filtfilt is already vectorized over non-axis dims)
        for band_name, (b, a) in filters.items():
            try:
                pred_filt = scipy_signal.filtfilt(b, a, y_pred, axis=-1)
                true_filt = scipy_signal.filtfilt(b, a, y_true, axis=-1)

                # Vectorized R² computation for band
                pred_band_flat = pred_filt.transpose(1, 0, 2).reshape(C, -1)
                true_band_flat = true_filt.transpose(1, 0, 2).reshape(C, -1)

                ss_res_band = np.sum((true_band_flat - pred_band_flat) ** 2, axis=1)
                true_band_means = np.mean(true_band_flat, axis=1, keepdims=True)
                ss_tot_band = np.sum((true_band_flat - true_band_means) ** 2, axis=1)

                r2_band = 1 - ss_res_band / (ss_tot_band + 1e-8)
                band_r2[band_name] = float(np.mean(r2_band))
            except Exception:
                pass
    except Exception:
        pass

    # =========================================================================
    # Assemble results
    # =========================================================================
    results = {
        "r2": r2,
        "r2_std_channels": r2_std_across_channels,
        "mae": mae,
        "pearson": pearson,
        "pearson_std": pearson_std,
        "psd_error_db": psd_error_db,
    }

    # Add per-band R²
    for band_name, r2_val in band_r2.items():
        results[f"r2_{band_name}"] = r2_val

    return results


# =============================================================================
# Baseline Evaluation with CV
# =============================================================================

@dataclass
class DetailedBaselineResult:
    """Extended baseline result with all metrics for Nature Methods."""
    method: str
    # Core metrics
    r2_mean: float
    r2_std: float
    r2_ci_lower: float
    r2_ci_upper: float
    mae_mean: float
    mae_std: float
    pearson_mean: float
    pearson_std: float
    psd_error_db: float
    # Per-band R²
    r2_delta: Optional[float] = None
    r2_theta: Optional[float] = None
    r2_alpha: Optional[float] = None
    r2_beta: Optional[float] = None
    r2_gamma: Optional[float] = None
    # Fold-level data (for statistical tests)
    fold_r2s: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "r2_mean": self.r2_mean,
            "r2_std": self.r2_std,
            "r2_ci_lower": self.r2_ci_lower,
            "r2_ci_upper": self.r2_ci_upper,
            "mae_mean": self.mae_mean,
            "mae_std": self.mae_std,
            "pearson_mean": self.pearson_mean,
            "pearson_std": self.pearson_std,
            "psd_error_db": self.psd_error_db,
            "r2_delta": self.r2_delta,
            "r2_theta": self.r2_theta,
            "r2_alpha": self.r2_alpha,
            "r2_beta": self.r2_beta,
            "r2_gamma": self.r2_gamma,
        }


def _evaluate_single_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    baseline_name: str,
    sample_rate: float,
    use_gpu: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single CV fold (designed for parallel execution).

    This function is designed to be called in parallel for each fold.

    Args:
        fold_idx: Index of the fold
        train_indices: Indices for training
        val_indices: Indices for validation
        X: Full input data
        y: Full target data
        baseline_name: Name of baseline method
        sample_rate: Sampling rate in Hz
        use_gpu: Whether to enable GPU for supported baselines

    Returns:
        Dictionary with fold results
    """
    # Get fold data
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    try:
        # Create and fit baseline with optional GPU support
        baseline_kwargs = {}
        if use_gpu and baseline_name in ["wiener", "wiener_mimo", "ridge"]:
            baseline_kwargs["use_gpu"] = True

        baseline = create_baseline(baseline_name, **baseline_kwargs)
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_val)

        metrics = compute_metrics(y_pred, y_val, sample_rate=sample_rate)

        return {
            "fold_idx": fold_idx,
            "success": True,
            "metrics": metrics,
        }

    except Exception as e:
        return {
            "fold_idx": fold_idx,
            "success": False,
            "error": str(e),
            "metrics": {},
        }


def evaluate_baseline_cv(
    baseline_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    sample_rate: float = 1000.0,
    n_jobs: int = -1,
    use_gpu: bool = False,
) -> Tuple[ClassicalResult, DetailedBaselineResult]:
    """Evaluate a baseline with 5-fold cross-validation (PARALLEL).

    Uses joblib for parallel fold evaluation to maximize CPU utilization.

    Returns BOTH ClassicalResult (for registry compatibility) and
    DetailedBaselineResult (for Nature Methods reporting).

    Args:
        baseline_name: Name of baseline method
        X: Input data [N, C, T]
        y: Target data [N, C, T]
        n_folds: Number of CV folds
        seed: Random seed for splits
        sample_rate: Sampling rate in Hz
        n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = sequential)
        use_gpu: Whether to enable GPU for supported baselines

    Returns:
        Tuple of (ClassicalResult, DetailedBaselineResult)
    """
    n_samples = X.shape[0]
    splits = create_data_splits(
        n_samples=n_samples,
        n_folds=n_folds,
        holdout_fraction=0.0,  # No holdout for Tier 0
        seed=seed,
    )

    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = min(n_folds, mp.cpu_count())

    # Parallel fold evaluation using joblib
    if HAS_JOBLIB and n_jobs > 1 and not use_gpu:
        # Note: GPU baselines should run sequentially to avoid CUDA context issues
        fold_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_evaluate_single_fold)(
                fold_idx=split.fold_idx,
                train_indices=split.train_indices,
                val_indices=split.val_indices,
                X=X,
                y=y,
                baseline_name=baseline_name,
                sample_rate=sample_rate,
                use_gpu=False,  # GPU disabled for parallel
            )
            for split in splits.cv_splits
        )
    else:
        # Sequential (for GPU or when joblib unavailable)
        fold_results = [
            _evaluate_single_fold(
                fold_idx=split.fold_idx,
                train_indices=split.train_indices,
                val_indices=split.val_indices,
                X=X,
                y=y,
                baseline_name=baseline_name,
                sample_rate=sample_rate,
                use_gpu=use_gpu,
            )
            for split in splits.cv_splits
        ]

    # Track all metrics per fold
    fold_r2s = []
    fold_maes = []
    fold_pearsons = []
    fold_psd_errors = []
    fold_band_r2s = {band: [] for band in NEURAL_BANDS.keys()}

    for result in fold_results:
        if result["success"]:
            metrics = result["metrics"]
            fold_r2s.append(metrics["r2"])
            fold_maes.append(metrics["mae"])
            fold_pearsons.append(metrics.get("pearson", np.nan))
            fold_psd_errors.append(metrics.get("psd_error_db", np.nan))

            for band in NEURAL_BANDS.keys():
                fold_band_r2s[band].append(metrics.get(f"r2_{band}", np.nan))
        else:
            print(f"    Fold {result['fold_idx']} failed: {result.get('error', 'Unknown')}")
            fold_r2s.append(np.nan)
            fold_maes.append(np.nan)
            fold_pearsons.append(np.nan)
            fold_psd_errors.append(np.nan)
            for band in NEURAL_BANDS.keys():
                fold_band_r2s[band].append(np.nan)

    # Aggregate across folds
    valid_r2s = np.array([r for r in fold_r2s if not np.isnan(r)])
    valid_maes = np.array([m for m in fold_maes if not np.isnan(m)])
    valid_pearsons = np.array([p for p in fold_pearsons if not np.isnan(p)])
    valid_psd_errors = np.array([p for p in fold_psd_errors if not np.isnan(p)])

    # Compute 95% bootstrap CI for R²
    ci = compute_bootstrap_ci(np.array(fold_r2s))

    # Create ClassicalResult (for registry compatibility)
    classical_result = ClassicalResult(
        method=baseline_name,
        r2_mean=float(np.mean(valid_r2s)) if len(valid_r2s) > 0 else np.nan,
        r2_std=float(np.std(valid_r2s, ddof=1)) if len(valid_r2s) > 1 else 0.0,
        mae_mean=float(np.mean(valid_maes)) if len(valid_maes) > 0 else np.nan,
        mae_std=float(np.std(valid_maes, ddof=1)) if len(valid_maes) > 1 else 0.0,
    )

    # Create DetailedBaselineResult (for Nature Methods)
    detailed_result = DetailedBaselineResult(
        method=baseline_name,
        r2_mean=float(np.mean(valid_r2s)) if len(valid_r2s) > 0 else np.nan,
        r2_std=float(np.std(valid_r2s, ddof=1)) if len(valid_r2s) > 1 else 0.0,
        r2_ci_lower=ci["ci_lower"],
        r2_ci_upper=ci["ci_upper"],
        mae_mean=float(np.mean(valid_maes)) if len(valid_maes) > 0 else np.nan,
        mae_std=float(np.std(valid_maes, ddof=1)) if len(valid_maes) > 1 else 0.0,
        pearson_mean=float(np.mean(valid_pearsons)) if len(valid_pearsons) > 0 else np.nan,
        pearson_std=float(np.std(valid_pearsons, ddof=1)) if len(valid_pearsons) > 1 else 0.0,
        psd_error_db=float(np.mean(valid_psd_errors)) if len(valid_psd_errors) > 0 else np.nan,
        r2_delta=float(np.nanmean(fold_band_r2s["delta"])) if fold_band_r2s["delta"] else None,
        r2_theta=float(np.nanmean(fold_band_r2s["theta"])) if fold_band_r2s["theta"] else None,
        r2_alpha=float(np.nanmean(fold_band_r2s["alpha"])) if fold_band_r2s["alpha"] else None,
        r2_beta=float(np.nanmean(fold_band_r2s["beta"])) if fold_band_r2s["beta"] else None,
        r2_gamma=float(np.nanmean(fold_band_r2s["gamma"])) if fold_band_r2s["gamma"] else None,
        fold_r2s=fold_r2s,
    )

    return classical_result, detailed_result


def _load_checkpoint() -> Dict[str, ClassicalResult]:
    """Load checkpoint for intra-tier resume."""
    checkpoint_file = ARTIFACTS_DIR / "tier0_checkpoint.json"
    if not checkpoint_file.exists():
        return {}

    with open(checkpoint_file) as f:
        data = json.load(f)

    results = {}
    for name, r in data.items():
        results[name] = ClassicalResult(
            method=r["method"],
            r2_mean=r["r2_mean"],
            r2_std=r["r2_std"],
            mae_mean=r["mae_mean"],
            mae_std=r["mae_std"],
        )
    return results


def _save_checkpoint(results: Dict[str, ClassicalResult]) -> None:
    """Save checkpoint for intra-tier resume."""
    checkpoint_file = ARTIFACTS_DIR / "tier0_checkpoint.json"
    data = {}
    for name, r in results.items():
        data[name] = {
            "method": r.method,
            "r2_mean": r.r2_mean,
            "r2_std": r.r2_std,
            "mae_mean": r.mae_mean,
            "mae_std": r.mae_std,
        }
    with open(checkpoint_file, "w") as f:
        json.dump(data, f, indent=2)


def _clear_checkpoint() -> None:
    """Clear checkpoint after successful tier completion."""
    checkpoint_file = ARTIFACTS_DIR / "tier0_checkpoint.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()


def compute_statistical_comparisons(
    fold_results: Dict[str, List[float]],
    metric_name: str = "R²",
) -> Dict[str, Any]:
    """Compute pairwise statistical comparisons between baselines.

    For Nature Methods, we need:
    - Paired t-tests (since same CV folds)
    - Effect sizes (Cohen's d)
    - Bonferroni correction for multiple comparisons
    - 95% bootstrap confidence intervals

    Args:
        fold_results: Dict mapping baseline name -> list of fold R² values
        metric_name: Name of metric for reporting

    Returns:
        Dictionary with statistical analysis results
    """
    from scipy import stats

    baselines = list(fold_results.keys())
    n_baselines = len(baselines)
    n_comparisons = n_baselines * (n_baselines - 1) // 2

    comparisons = []

    for i in range(n_baselines):
        for j in range(i + 1, n_baselines):
            name1, name2 = baselines[i], baselines[j]
            values1 = np.array(fold_results[name1])
            values2 = np.array(fold_results[name2])

            # Filter NaNs
            valid_mask = ~(np.isnan(values1) | np.isnan(values2))
            if valid_mask.sum() < 2:
                continue

            v1 = values1[valid_mask]
            v2 = values2[valid_mask]

            # Paired t-test (same folds)
            t_stat, p_value = stats.ttest_rel(v1, v2)

            # Cohen's d (paired)
            diff = v1 - v2
            d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)

            # Effect size interpretation
            d_abs = abs(d)
            if d_abs < 0.2:
                effect_size = "negligible"
            elif d_abs < 0.5:
                effect_size = "small"
            elif d_abs < 0.8:
                effect_size = "medium"
            else:
                effect_size = "large"

            comparisons.append({
                "baseline1": name1,
                "baseline2": name2,
                "mean1": float(np.mean(v1)),
                "mean2": float(np.mean(v2)),
                "diff": float(np.mean(v1) - np.mean(v2)),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "cohens_d": float(d),
                "effect_size": effect_size,
            })

    # Bonferroni correction
    if comparisons:
        p_values = np.array([c["p_value"] for c in comparisons])
        adjusted_p = np.minimum(p_values * n_comparisons, 1.0)
        for i, comp in enumerate(comparisons):
            comp["p_adjusted"] = float(adjusted_p[i])
            comp["significant"] = adjusted_p[i] < 0.05

    return {
        "comparisons": comparisons,
        "n_comparisons": n_comparisons,
        "correction": "bonferroni",
        "alpha": 0.05,
    }


def compute_bootstrap_ci(values: np.ndarray, n_bootstrap: int = 10000, ci_level: float = 0.95) -> Dict[str, float]:
    """Compute bootstrap 95% confidence interval - FULLY VECTORIZED.

    Uses NumPy vectorization to compute all bootstrap samples at once,
    eliminating the Python loop overhead for 10,000 iterations.

    Args:
        values: Array of observations (e.g., R² values from folds)
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level

    Returns:
        Dictionary with CI bounds and standard error
    """
    valid_values = values[~np.isnan(values)]

    if len(valid_values) < 2:
        return {"ci_lower": np.nan, "ci_upper": np.nan, "se": np.nan}

    n = len(valid_values)
    rng = np.random.RandomState(42)

    # VECTORIZED: Generate all bootstrap sample indices at once [n_bootstrap, n]
    bootstrap_indices = rng.randint(0, n, size=(n_bootstrap, n))

    # VECTORIZED: Index into values and compute means in one shot
    bootstrap_samples = valid_values[bootstrap_indices]  # [n_bootstrap, n]
    bootstrap_means = np.mean(bootstrap_samples, axis=1)  # [n_bootstrap]

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    se = float(np.std(bootstrap_means, ddof=1))

    return {"ci_lower": ci_lower, "ci_upper": ci_upper, "se": se}


def _evaluate_baseline_for_parallel(
    baseline_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
    sample_rate: float,
) -> Tuple[str, ClassicalResult, DetailedBaselineResult]:
    """Wrapper for parallel baseline evaluation."""
    classical_result, detailed_result = evaluate_baseline_cv(
        baseline_name=baseline_name,
        X=X,
        y=y,
        n_folds=n_folds,
        seed=seed,
        sample_rate=sample_rate,
        n_jobs=1,  # Sequential folds when running baselines in parallel
        use_gpu=False,
    )
    return baseline_name, classical_result, detailed_result


def run_tier0(
    X: np.ndarray,
    y: np.ndarray,
    baselines: Optional[List[str]] = None,
    n_folds: int = 5,
    seed: int = 42,
    register: bool = True,
    resume: bool = True,
    sample_rate: float = 1000.0,
    n_jobs: int = -1,
    use_gpu: bool = False,
    parallel_baselines: bool = False,
) -> Tuple[Tier0Result, Dict[str, Any]]:
    """Run Tier 0: Classical Floor evaluation (Nature Methods publication quality).

    PERFORMANCE OPTIMIZED:
    - Parallel CV fold processing (via n_jobs)
    - Optional parallel baseline evaluation (via parallel_baselines)
    - GPU acceleration for supported baselines (via use_gpu)
    - Vectorized metrics computation and bootstrap CI

    Args:
        X: Input data [N, C, T]
        y: Target data [N, C, T]
        baselines: List of baseline names (default: all)
        n_folds: Number of CV folds
        seed: Random seed
        register: Whether to register results with ConfigRegistry
        resume: Whether to resume from checkpoint (skip completed experiments)
        sample_rate: Sampling rate in Hz for spectral analysis
        n_jobs: Number of parallel jobs for fold CV (-1 = all CPUs)
        use_gpu: Whether to use GPU for supported baselines
        parallel_baselines: Whether to run baselines in parallel (uses more memory)

    Returns:
        Tuple of:
        - Tier0Result (for registry)
        - Dict with detailed results and statistical analysis (for publication)
    """
    baselines = baselines or BASELINES

    # Load checkpoint for intra-tier resume
    completed_results = {}
    if resume:
        completed_results = _load_checkpoint()
        if completed_results:
            print(f"\n🔄 Resuming from checkpoint: {len(completed_results)} baselines already done")

    results = []  # ClassicalResults for registry
    detailed_results = []  # DetailedBaselineResults for publication
    fold_results_for_stats = {}  # baseline_name -> list of fold R² values

    # Filter out already completed baselines
    baselines_to_run = [b for b in baselines if b not in completed_results]

    # Add completed results first
    for baseline_name in baselines:
        if baseline_name in completed_results:
            result = completed_results[baseline_name]
            print(f"  {baseline_name}... SKIPPED (already done) R² = {result.r2_mean:.4f} ± {result.r2_std:.4f}")
            results.append(result)

    n_parallel_jobs = min(len(baselines_to_run), mp.cpu_count()) if parallel_baselines else 1
    mode_str = f"parallel ({n_parallel_jobs} baselines)" if n_parallel_jobs > 1 else "sequential"
    gpu_str = " + GPU" if use_gpu else ""

    print(f"\nEvaluating {len(baselines_to_run)} classical baselines with {n_folds}-fold CV...")
    print(f"  Mode: {mode_str}{gpu_str}, {n_jobs} jobs per baseline for fold CV")
    print(f"  Data shape: {X.shape}, Memory: ~{X.nbytes * 2 / 1e9:.1f} GB")

    # Parallel baseline evaluation
    if parallel_baselines and HAS_JOBLIB and n_parallel_jobs > 1:
        print(f"\n  Running {len(baselines_to_run)} baselines in parallel...")
        start_all = time.time()

        parallel_results = Parallel(n_jobs=n_parallel_jobs, prefer="processes")(
            delayed(_evaluate_baseline_for_parallel)(
                baseline_name=baseline_name,
                X=X,
                y=y,
                n_folds=n_folds,
                seed=seed,
                sample_rate=sample_rate,
            )
            for baseline_name in baselines_to_run
        )

        for baseline_name, classical_result, detailed_result in parallel_results:
            results.append(classical_result)
            detailed_results.append(detailed_result)
            if detailed_result.fold_r2s:
                fold_results_for_stats[baseline_name] = detailed_result.fold_r2s
            completed_results[baseline_name] = classical_result

            ci_str = f"[{detailed_result.r2_ci_lower:.4f}, {detailed_result.r2_ci_upper:.4f}]"
            print(f"  {baseline_name}: R² = {classical_result.r2_mean:.4f} ± {classical_result.r2_std:.4f} {ci_str}")

        elapsed_all = time.time() - start_all
        print(f"  Total parallel time: {elapsed_all:.1f}s ({elapsed_all/len(baselines_to_run):.1f}s avg per baseline)")

        # Save checkpoint after all complete
        _save_checkpoint(completed_results)
    else:
        # Sequential baseline evaluation
        for baseline_name in baselines_to_run:
            print(f"  {baseline_name}...", end=" ", flush=True)
            start = time.time()

            classical_result, detailed_result = evaluate_baseline_cv(
                baseline_name=baseline_name,
                X=X,
                y=y,
                n_folds=n_folds,
                seed=seed,
                sample_rate=sample_rate,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
            )
            results.append(classical_result)
            detailed_results.append(detailed_result)

            # Store fold-level results for statistical tests
            if detailed_result.fold_r2s:
                fold_results_for_stats[baseline_name] = detailed_result.fold_r2s

            elapsed = time.time() - start
            ci_str = f"[{detailed_result.r2_ci_lower:.4f}, {detailed_result.r2_ci_upper:.4f}]"
            print(f"R² = {classical_result.r2_mean:.4f} ± {classical_result.r2_std:.4f} {ci_str} ({elapsed:.1f}s)")

            # Save checkpoint after each baseline
            completed_results[baseline_name] = classical_result
            _save_checkpoint(completed_results)

    # Find best
    valid_results = [r for r in results if not np.isnan(r.r2_mean)]
    if not valid_results:
        raise RuntimeError("All baselines failed!")

    best = max(valid_results, key=lambda r: r.r2_mean)

    tier0_result = Tier0Result(
        results=results,
        best_method=best.method,
        best_r2=best.r2_mean,
    )

    # Register with global registry
    if register:
        registry = get_registry(ARTIFACTS_DIR)
        registry.register_tier0(tier0_result)

    # Clear checkpoint on successful completion
    _clear_checkpoint()

    # Compute statistical comparisons between baselines
    stats_comparisons = {}
    if len(fold_results_for_stats) >= 2:
        stats_comparisons = compute_statistical_comparisons(fold_results_for_stats)

    # Assemble publication-quality results
    publication_results = {
        "detailed_results": [r.to_dict() for r in detailed_results],
        "statistical_comparisons": stats_comparisons,
        "best_method": best.method,
        "best_r2": best.r2_mean,
        "n_baselines": len(baselines),
        "n_folds": n_folds,
    }

    return tier0_result, publication_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tier 0: Classical Floor - PERFORMANCE OPTIMIZED",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Performance Tips:
  --n-jobs -1           Use all CPUs for parallel CV folds (default)
  --parallel-baselines  Run baselines in parallel (uses more RAM)
  --use-gpu             Enable GPU for supported baselines (Wiener, Ridge)

Example for maximum performance with 2TB RAM:
  python run_tier0.py --parallel-baselines --n-jobs -1

Example for GPU acceleration:
  python run_tier0.py --use-gpu --n-jobs 1
        """
    )
    parser.add_argument("--dry-run", action="store_true", help="Use small synthetic data")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    # Performance options
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of parallel jobs for fold CV (-1 = all CPUs, default: -1)"
    )
    parser.add_argument(
        "--parallel-baselines", action="store_true",
        help="Run baselines in parallel (uses more RAM but faster)"
    )
    parser.add_argument(
        "--use-gpu", action="store_true",
        help="Use GPU for supported baselines (Wiener, Ridge)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't resume from checkpoint, start fresh"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TIER 0: Classical Floor")
    print("=" * 60)
    print("Purpose: Establish minimum bar for neural methods")
    print("Gate: Neural must beat classical by R² >= 0.10")
    print()

    # Load data
    print("Loading data...")

    if args.dry_run:
        # Small synthetic data
        N = 100
        C_in, C_out, T = 32, 32, 300
        X = np.random.randn(N, C_in, T).astype(np.float32)
        y = np.random.randn(N, C_out, T).astype(np.float32)
        baselines = FAST_BASELINES
        print(f"  [DRY-RUN] Using synthetic data: {X.shape}")
    else:
        try:
            from data import prepare_data
            data = prepare_data()

            # Combine train and val for CV (holdout is separate)
            X_train = data["train"]["ob"].numpy()
            y_train = data["train"]["hp"].numpy()
            X_val = data["val"]["ob"].numpy()
            y_val = data["val"]["hp"].numpy()

            X = np.concatenate([X_train, X_val], axis=0)
            y = np.concatenate([y_train, y_val], axis=0)
            baselines = BASELINES
            print(f"  Loaded real data: {X.shape}")
        except Exception as e:
            print(f"  Warning: Could not load data ({e}), using synthetic")
            N = 200
            C_in, C_out, T = 32, 32, 500
            X = np.random.randn(N, C_in, T).astype(np.float32)
            y = np.random.randn(N, C_out, T).astype(np.float32)
            baselines = BASELINES

    # Print performance configuration
    print(f"\nPerformance configuration:")
    print(f"  Parallel fold CV jobs: {args.n_jobs}")
    print(f"  Parallel baselines: {args.parallel_baselines}")
    print(f"  GPU acceleration: {args.use_gpu}")
    print(f"  Resume from checkpoint: {not args.no_resume}")

    # Run
    result, publication_results = run_tier0(
        X=X,
        y=y,
        baselines=baselines,
        n_folds=args.folds,
        seed=args.seed,
        resume=not args.no_resume,
        n_jobs=args.n_jobs,
        use_gpu=args.use_gpu,
        parallel_baselines=args.parallel_baselines,
    )

    # Summary
    print("\n" + "=" * 60)
    print("TIER 0 RESULTS: Classical Floor (Nature Methods)")
    print("=" * 60)

    # Main results table with 95% CI
    print("\n📊 MAIN RESULTS (5-fold CV)")
    print("-" * 80)
    print(f"{'Method':<20} {'R²':>10} {'95% CI':>20} {'Pearson':>10} {'PSD Err(dB)':>12}")
    print("-" * 80)

    sorted_detailed = sorted(
        publication_results.get("detailed_results", []),
        key=lambda r: r.get("r2_mean", 0),
        reverse=True,
    )

    for r in sorted_detailed:
        ci_str = f"[{r.get('r2_ci_lower', 0):.3f}, {r.get('r2_ci_upper', 0):.3f}]"
        pearson = r.get('pearson_mean', 0)
        psd_err = r.get('psd_error_db', float('nan'))
        marker = " *" if r.get("method") == result.best_method else ""
        print(f"{r.get('method', 'N/A'):<20} {r.get('r2_mean', 0):>10.4f} {ci_str:>20} {pearson:>10.4f} {psd_err:>12.2f}{marker}")

    # Per-frequency-band R²
    print("\n📈 PER-BAND R² (neuroscience-relevant)")
    print("-" * 80)
    print(f"{'Method':<20} {'Delta':>8} {'Theta':>8} {'Alpha':>8} {'Beta':>8} {'Gamma':>8}")
    print("-" * 80)

    for r in sorted_detailed[:5]:  # Top 5 only
        delta = r.get('r2_delta', float('nan'))
        theta = r.get('r2_theta', float('nan'))
        alpha = r.get('r2_alpha', float('nan'))
        beta = r.get('r2_beta', float('nan'))
        gamma = r.get('r2_gamma', float('nan'))

        def fmt(v):
            return f"{v:.4f}" if v is not None and not np.isnan(v) else "N/A"

        print(f"{r.get('method', 'N/A'):<20} {fmt(delta):>8} {fmt(theta):>8} {fmt(alpha):>8} {fmt(beta):>8} {fmt(gamma):>8}")

    # Statistical comparisons
    stats = publication_results.get("statistical_comparisons", {})
    if stats and stats.get("comparisons"):
        print("\n📉 STATISTICAL COMPARISONS (Bonferroni-corrected)")
        print("-" * 80)
        sig_comparisons = [c for c in stats["comparisons"] if c.get("significant")]
        if sig_comparisons:
            print(f"Significant differences (p < 0.05, {stats.get('n_comparisons', 0)} comparisons):")
            for comp in sorted(sig_comparisons, key=lambda x: -abs(x.get("cohens_d", 0)))[:10]:
                print(f"  {comp['baseline1']} vs {comp['baseline2']}: "
                      f"Δ = {comp['diff']:.4f}, d = {comp['cohens_d']:.2f} ({comp['effect_size']}), "
                      f"p_adj = {comp['p_adjusted']:.4f}")
        else:
            print("  No significant differences found between baselines")

    # Classical floor summary
    print("\n" + "=" * 60)
    print(f"CLASSICAL FLOOR: R² = {result.best_r2:.4f} ({result.best_method})")
    print(f"GATE THRESHOLD: Neural must achieve R² >= {result.best_r2 + 0.10:.4f}")
    print("=" * 60)

    # Save both results
    output_file = args.output or (ARTIFACTS_DIR / f"tier0_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_file = Path(output_file)

    # Combine both for saving
    full_results = {
        "tier0_summary": result.to_dict(),
        "publication_results": publication_results,
    }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return result, publication_results


if __name__ == "__main__":
    main()
