"""
Comprehensive Metrics for Phase 1
=================================

Publication-quality evaluation metrics for neural signal translation.
Includes per-channel, per-frequency-band, and statistical analysis.

Metrics computed:
    - R² (coefficient of determination)
    - MAE (mean absolute error)
    - Pearson correlation
    - Per-frequency-band R² (delta, theta, alpha, beta, gamma)
    - PSD error in dB (spectral fidelity)
    - Bootstrap confidence intervals
    - Statistical comparisons (paired tests, effect sizes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal
from scipy import stats

from .config import NEURAL_BANDS


# =============================================================================
# Metric Dataclasses
# =============================================================================

@dataclass
class Phase1Metrics:
    """Complete metrics for a single baseline evaluation.

    Attributes:
        method: Baseline method name
        r2_mean: Mean R² across folds
        r2_std: Standard deviation of R² across folds
        r2_ci: 95% confidence interval (lower, upper)
        mae_mean: Mean absolute error
        mae_std: MAE standard deviation
        pearson_mean: Mean Pearson correlation
        pearson_std: Pearson correlation std
        psd_error_db: PSD error in decibels
        band_r2: Per-frequency-band R² values
        fold_r2s: R² values for each fold (for statistical tests)
        n_folds: Number of CV folds
        n_samples: Number of samples evaluated
    """

    method: str

    # Core metrics
    r2_mean: float
    r2_std: float
    r2_ci: Tuple[float, float]

    mae_mean: float
    mae_std: float

    pearson_mean: float
    pearson_std: float

    psd_error_db: float

    # Per-band metrics
    band_r2: Dict[str, float] = field(default_factory=dict)

    # Fold-level data
    fold_r2s: List[float] = field(default_factory=list)
    fold_maes: List[float] = field(default_factory=list)
    fold_pearsons: List[float] = field(default_factory=list)

    # Metadata
    n_folds: int = 5
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method,
            "r2_mean": self.r2_mean,
            "r2_std": self.r2_std,
            "r2_ci_lower": self.r2_ci[0],
            "r2_ci_upper": self.r2_ci[1],
            "mae_mean": self.mae_mean,
            "mae_std": self.mae_std,
            "pearson_mean": self.pearson_mean,
            "pearson_std": self.pearson_std,
            "psd_error_db": self.psd_error_db,
            **{f"r2_{band}": r2 for band, r2 in self.band_r2.items()},
            "n_folds": self.n_folds,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Phase1Metrics":
        """Create from dictionary."""
        band_r2 = {band: d.get(f"r2_{band}", np.nan) for band in NEURAL_BANDS.keys()}
        return cls(
            method=d["method"],
            r2_mean=d["r2_mean"],
            r2_std=d["r2_std"],
            r2_ci=(d["r2_ci_lower"], d["r2_ci_upper"]),
            mae_mean=d["mae_mean"],
            mae_std=d["mae_std"],
            pearson_mean=d["pearson_mean"],
            pearson_std=d["pearson_std"],
            psd_error_db=d["psd_error_db"],
            band_r2=band_r2,
            n_folds=d.get("n_folds", 5),
            n_samples=d.get("n_samples", 0),
        )


@dataclass
class StatisticalComparison:
    """Pairwise statistical comparison between two methods."""

    method1: str
    method2: str
    mean_diff: float      # method1 - method2
    t_stat: float         # Paired t-test statistic
    p_value: float        # Raw p-value
    p_adjusted: float     # Bonferroni-corrected p-value
    cohens_d: float       # Effect size
    effect_size: str      # Interpretation: negligible/small/medium/large
    significant: bool     # Whether p_adjusted < 0.05


# =============================================================================
# Metrics Calculator
# =============================================================================

class MetricsCalculator:
    """Calculator for comprehensive evaluation metrics.

    Computes all metrics required for Nature Methods publication:
    - Per-channel and overall R²
    - Per-frequency-band R² (neural oscillation bands)
    - Pearson correlations
    - PSD error for spectral fidelity
    - Bootstrap confidence intervals

    Args:
        sample_rate: Sampling rate in Hz (for spectral analysis)
        n_bootstrap: Number of bootstrap samples for CI

    Example:
        >>> calc = MetricsCalculator(sample_rate=1000)
        >>> metrics = calc.compute(y_pred, y_true)
    """

    def __init__(
        self,
        sample_rate: float = 1000.0,
        n_bootstrap: int = 10000,
        ci_level: float = 0.95,
    ):
        self.sample_rate = sample_rate
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level

    def compute(self, y_pred: NDArray, y_true: NDArray) -> Dict[str, float]:
        """Compute all metrics for a single evaluation.

        Args:
            y_pred: Predictions [N, C, T]
            y_true: Ground truth [N, C, T]

        Returns:
            Dictionary of metric name -> value
        """
        # Ensure 3D
        if y_pred.ndim == 2:
            y_pred = y_pred[:, np.newaxis, :]
            y_true = y_true[:, np.newaxis, :]

        N, C, T = y_pred.shape

        metrics = {}

        # R² (per-channel, then averaged)
        r2, r2_std = self._compute_r2(y_pred, y_true)
        metrics["r2"] = r2
        metrics["r2_std_channels"] = r2_std

        # MAE
        metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))

        # Pearson correlation
        pearson, pearson_std = self._compute_pearson(y_pred, y_true)
        metrics["pearson"] = pearson
        metrics["pearson_std"] = pearson_std

        # PSD error in dB
        metrics["psd_error_db"] = self._compute_psd_error(y_pred, y_true)

        # Per-frequency-band R²
        band_r2 = self._compute_band_r2(y_pred, y_true)
        for band, r2_val in band_r2.items():
            metrics[f"r2_{band}"] = r2_val

        return metrics

    def _compute_r2(self, y_pred: NDArray, y_true: NDArray) -> Tuple[float, float]:
        """Compute per-channel R², then average.

        R² = 1 - SS_res / SS_tot

        Returns:
            (mean_r2, std_r2_across_channels)
        """
        N, C, T = y_pred.shape

        # Reshape to [C, N*T] for vectorized computation
        pred_flat = y_pred.transpose(1, 0, 2).reshape(C, -1)
        true_flat = y_true.transpose(1, 0, 2).reshape(C, -1)

        # Compute SS_res and SS_tot per channel
        ss_res = np.sum((true_flat - pred_flat) ** 2, axis=1)  # [C]
        ss_tot = np.sum((true_flat - np.mean(true_flat, axis=1, keepdims=True)) ** 2, axis=1)  # [C]

        r2_per_channel = 1 - ss_res / (ss_tot + 1e-10)

        return float(np.mean(r2_per_channel)), float(np.std(r2_per_channel))

    def _compute_pearson(self, y_pred: NDArray, y_true: NDArray) -> Tuple[float, float]:
        """Compute Pearson correlation per channel, then average."""
        N, C, T = y_pred.shape

        pred_flat = y_pred.transpose(1, 0, 2).reshape(C, -1)
        true_flat = y_true.transpose(1, 0, 2).reshape(C, -1)

        # Center
        pred_centered = pred_flat - np.mean(pred_flat, axis=1, keepdims=True)
        true_centered = true_flat - np.mean(true_flat, axis=1, keepdims=True)

        # Correlation
        num = np.sum(pred_centered * true_centered, axis=1)
        denom = np.sqrt(np.sum(pred_centered ** 2, axis=1) * np.sum(true_centered ** 2, axis=1))

        correlations = num / (denom + 1e-10)
        valid = np.isfinite(correlations)

        if valid.any():
            return float(np.mean(correlations[valid])), float(np.std(correlations[valid]))
        return 0.0, 0.0

    def _compute_psd_error(self, y_pred: NDArray, y_true: NDArray) -> float:
        """Compute PSD error in decibels (spectral fidelity)."""
        try:
            T = y_pred.shape[-1]
            nperseg = min(256, T)

            _, psd_pred = scipy_signal.welch(y_pred, fs=self.sample_rate, axis=-1, nperseg=nperseg)
            _, psd_true = scipy_signal.welch(y_true, fs=self.sample_rate, axis=-1, nperseg=nperseg)

            # Log-space error (in dB)
            log_pred = np.log10(psd_pred + 1e-10)
            log_true = np.log10(psd_true + 1e-10)

            return float(10 * np.mean(np.abs(log_pred - log_true)))
        except Exception:
            return np.nan

    def _compute_band_r2(self, y_pred: NDArray, y_true: NDArray) -> Dict[str, float]:
        """Compute R² for each neural frequency band."""
        band_r2 = {}
        nyq = self.sample_rate / 2

        for band_name, (f_low, f_high) in NEURAL_BANDS.items():
            if f_low >= nyq:
                band_r2[band_name] = np.nan
                continue

            try:
                # Design bandpass filter
                low = f_low / nyq
                high = min(f_high / nyq, 0.99)

                if low >= high:
                    band_r2[band_name] = np.nan
                    continue

                b, a = scipy_signal.butter(4, [low, high], btype='band')

                # Filter signals
                pred_filt = scipy_signal.filtfilt(b, a, y_pred, axis=-1)
                true_filt = scipy_signal.filtfilt(b, a, y_true, axis=-1)

                # Compute R² on filtered
                r2, _ = self._compute_r2(pred_filt, true_filt)
                band_r2[band_name] = r2

            except Exception:
                band_r2[band_name] = np.nan

        return band_r2

    def bootstrap_ci(
        self,
        values: NDArray,
        n_bootstrap: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval.

        Args:
            values: Array of observations (e.g., R² from folds)
            n_bootstrap: Number of bootstrap samples (default: self.n_bootstrap)

        Returns:
            (ci_lower, ci_upper)
        """
        n_bootstrap = n_bootstrap or self.n_bootstrap
        valid = values[~np.isnan(values)]

        if len(valid) < 2:
            return (np.nan, np.nan)

        rng = np.random.RandomState(42)

        # Vectorized bootstrap
        n = len(valid)
        indices = rng.randint(0, n, size=(n_bootstrap, n))
        bootstrap_means = np.mean(valid[indices], axis=1)

        alpha = 1 - self.ci_level
        ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

        return (ci_lower, ci_upper)


# =============================================================================
# Statistical Analysis
# =============================================================================

def compute_statistical_comparisons(
    method_fold_r2s: Dict[str, List[float]],
    alpha: float = 0.05,
) -> List[StatisticalComparison]:
    """Compute pairwise statistical comparisons between methods.

    Uses paired t-tests (same CV folds) with Bonferroni correction.

    Args:
        method_fold_r2s: Dict mapping method name -> list of fold R² values
        alpha: Significance level

    Returns:
        List of StatisticalComparison objects
    """
    methods = list(method_fold_r2s.keys())
    n_methods = len(methods)
    n_comparisons = n_methods * (n_methods - 1) // 2

    comparisons = []

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            name1, name2 = methods[i], methods[j]
            values1 = np.array(method_fold_r2s[name1])
            values2 = np.array(method_fold_r2s[name2])

            # Filter NaNs
            valid_mask = ~(np.isnan(values1) | np.isnan(values2))
            if valid_mask.sum() < 2:
                continue

            v1 = values1[valid_mask]
            v2 = values2[valid_mask]

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(v1, v2)

            # Cohen's d (paired)
            diff = v1 - v2
            d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

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

            # Bonferroni correction
            p_adjusted = min(p_value * n_comparisons, 1.0)

            comparisons.append(StatisticalComparison(
                method1=name1,
                method2=name2,
                mean_diff=float(np.mean(v1) - np.mean(v2)),
                t_stat=float(t_stat),
                p_value=float(p_value),
                p_adjusted=float(p_adjusted),
                cohens_d=float(d),
                effect_size=effect_size,
                significant=p_adjusted < alpha,
            ))

    return comparisons


def rank_methods(metrics_list: List[Phase1Metrics]) -> List[Tuple[int, Phase1Metrics]]:
    """Rank methods by R² (highest first).

    Args:
        metrics_list: List of Phase1Metrics objects

    Returns:
        List of (rank, metrics) tuples, sorted by R²
    """
    sorted_metrics = sorted(metrics_list, key=lambda m: m.r2_mean, reverse=True)
    return [(i + 1, m) for i, m in enumerate(sorted_metrics)]


def format_metrics_table(metrics_list: List[Phase1Metrics]) -> str:
    """Format metrics as a publication-ready table.

    Args:
        metrics_list: List of Phase1Metrics objects

    Returns:
        Formatted string table
    """
    ranked = rank_methods(metrics_list)

    lines = []
    lines.append("=" * 90)
    lines.append(f"{'Rank':<6}{'Method':<18}{'R²':>10}{'95% CI':>22}{'Pearson':>10}{'PSD Err(dB)':>12}")
    lines.append("-" * 90)

    for rank, m in ranked:
        ci_str = f"[{m.r2_ci[0]:.4f}, {m.r2_ci[1]:.4f}]"
        lines.append(
            f"{rank:<6}{m.method:<18}{m.r2_mean:>10.4f}{ci_str:>22}"
            f"{m.pearson_mean:>10.4f}{m.psd_error_db:>12.2f}"
        )

    lines.append("=" * 90)
    return "\n".join(lines)
