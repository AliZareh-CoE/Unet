"""
Comprehensive Metrics for Phase 1
=================================

Publication-quality evaluation metrics for neural signal translation.
Includes per-channel, per-frequency-band, DSP metrics, and statistical analysis.

Metrics computed:
    - R² (coefficient of determination)
    - MAE (mean absolute error)
    - Pearson/Spearman correlations
    - Per-frequency-band R² (delta, theta, alpha, beta, gamma)
    - PSD error in dB (spectral fidelity)
    - DSP Metrics: PLV, PLI, wPLI, coherence, SNR, envelope correlation
    - Cross-correlation analysis
    - Mutual information
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

# Import DSP metrics
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from utils.dsp_metrics import (
    compute_plv, compute_pli, compute_wpli,
    compute_coherence, compute_reconstruction_snr,
    compute_cross_correlation, compute_envelope_correlation,
    compute_mutual_information, compute_normalized_mi,
    compare_psd, FREQUENCY_BANDS,
)


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
        spearman_mean: Mean Spearman correlation
        spearman_std: Spearman correlation std
        psd_error_db: PSD error in decibels
        band_r2: Per-frequency-band R² values
        fold_r2s: R² values for each fold (for statistical tests)
        n_folds: Number of CV folds
        n_samples: Number of samples evaluated

        # DSP Metrics
        plv_mean: Phase Locking Value mean
        pli_mean: Phase Lag Index mean
        wpli_mean: Weighted Phase Lag Index mean
        coherence_mean: Mean coherence across frequencies
        coherence_bands: Per-band coherence values
        snr_db_mean: Signal-to-noise ratio mean (dB)
        reconstruction_snr_db: SNR of reconstruction vs original
        cross_corr_max: Maximum cross-correlation
        cross_corr_lag: Lag at maximum cross-correlation
        envelope_corr_mean: Envelope correlation mean
        mutual_info_mean: Mutual information mean
        normalized_mi_mean: Normalized mutual information mean
        psd_correlation: Correlation between PSDs
        band_power_errors: Per-band power errors
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

    # Additional correlation
    spearman_mean: float = 0.0
    spearman_std: float = 0.0

    # DSP Metrics - Phase connectivity
    plv_mean: float = 0.0
    plv_std: float = 0.0
    pli_mean: float = 0.0
    pli_std: float = 0.0
    wpli_mean: float = 0.0
    wpli_std: float = 0.0

    # DSP Metrics - Coherence
    coherence_mean: float = 0.0
    coherence_std: float = 0.0
    coherence_bands: Dict[str, float] = field(default_factory=dict)

    # DSP Metrics - SNR
    snr_db_mean: float = 0.0
    reconstruction_snr_db: float = 0.0

    # DSP Metrics - Cross-correlation
    cross_corr_max: float = 0.0
    cross_corr_lag: int = 0

    # DSP Metrics - Envelope
    envelope_corr_mean: float = 0.0
    envelope_corr_std: float = 0.0

    # DSP Metrics - Information theory
    mutual_info_mean: float = 0.0
    normalized_mi_mean: float = 0.0

    # DSP Metrics - PSD comparison
    psd_correlation: float = 0.0
    band_power_errors: Dict[str, float] = field(default_factory=dict)

    # Fold-level DSP metrics for statistical analysis
    fold_plvs: List[float] = field(default_factory=list)
    fold_plis: List[float] = field(default_factory=list)
    fold_coherences: List[float] = field(default_factory=list)
    fold_envelope_corrs: List[float] = field(default_factory=list)
    fold_reconstruction_snrs: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "method": self.method,
            "r2_mean": self.r2_mean,
            "r2_std": self.r2_std,
            "r2_ci": list(self.r2_ci),
            "r2_ci_lower": self.r2_ci[0],
            "r2_ci_upper": self.r2_ci[1],
            "mae_mean": self.mae_mean,
            "mae_std": self.mae_std,
            "pearson_mean": self.pearson_mean,
            "pearson_std": self.pearson_std,
            "spearman_mean": self.spearman_mean,
            "spearman_std": self.spearman_std,
            "psd_error_db": self.psd_error_db,
            "band_r2": self.band_r2,
            "n_folds": self.n_folds,
            "n_samples": self.n_samples,
            # DSP metrics
            "plv_mean": self.plv_mean,
            "plv_std": self.plv_std,
            "pli_mean": self.pli_mean,
            "pli_std": self.pli_std,
            "wpli_mean": self.wpli_mean,
            "wpli_std": self.wpli_std,
            "coherence_mean": self.coherence_mean,
            "coherence_std": self.coherence_std,
            "coherence_bands": self.coherence_bands,
            "snr_db_mean": self.snr_db_mean,
            "reconstruction_snr_db": self.reconstruction_snr_db,
            "cross_corr_max": self.cross_corr_max,
            "cross_corr_lag": self.cross_corr_lag,
            "envelope_corr_mean": self.envelope_corr_mean,
            "envelope_corr_std": self.envelope_corr_std,
            "mutual_info_mean": self.mutual_info_mean,
            "normalized_mi_mean": self.normalized_mi_mean,
            "psd_correlation": self.psd_correlation,
            "band_power_errors": self.band_power_errors,
            # Fold-level data
            "fold_r2s": self.fold_r2s,
            "fold_maes": self.fold_maes,
            "fold_pearsons": self.fold_pearsons,
            "fold_plvs": self.fold_plvs,
            "fold_plis": self.fold_plis,
            "fold_coherences": self.fold_coherences,
            "fold_envelope_corrs": self.fold_envelope_corrs,
            "fold_reconstruction_snrs": self.fold_reconstruction_snrs,
        }
        # Add band R² individually for backward compatibility
        for band, r2 in self.band_r2.items():
            result[f"r2_{band}"] = r2
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Phase1Metrics":
        """Create from dictionary."""
        band_r2 = d.get("band_r2", {band: d.get(f"r2_{band}", np.nan) for band in NEURAL_BANDS.keys()})
        r2_ci = d.get("r2_ci", (d.get("r2_ci_lower", 0.0), d.get("r2_ci_upper", 0.0)))
        if isinstance(r2_ci, list):
            r2_ci = tuple(r2_ci)
        return cls(
            method=d["method"],
            r2_mean=d["r2_mean"],
            r2_std=d["r2_std"],
            r2_ci=r2_ci,
            mae_mean=d["mae_mean"],
            mae_std=d["mae_std"],
            pearson_mean=d["pearson_mean"],
            pearson_std=d["pearson_std"],
            spearman_mean=d.get("spearman_mean", 0.0),
            spearman_std=d.get("spearman_std", 0.0),
            psd_error_db=d["psd_error_db"],
            band_r2=band_r2,
            n_folds=d.get("n_folds", 5),
            n_samples=d.get("n_samples", 0),
            # DSP metrics
            plv_mean=d.get("plv_mean", 0.0),
            plv_std=d.get("plv_std", 0.0),
            pli_mean=d.get("pli_mean", 0.0),
            pli_std=d.get("pli_std", 0.0),
            wpli_mean=d.get("wpli_mean", 0.0),
            wpli_std=d.get("wpli_std", 0.0),
            coherence_mean=d.get("coherence_mean", 0.0),
            coherence_std=d.get("coherence_std", 0.0),
            coherence_bands=d.get("coherence_bands", {}),
            snr_db_mean=d.get("snr_db_mean", 0.0),
            reconstruction_snr_db=d.get("reconstruction_snr_db", 0.0),
            cross_corr_max=d.get("cross_corr_max", 0.0),
            cross_corr_lag=d.get("cross_corr_lag", 0),
            envelope_corr_mean=d.get("envelope_corr_mean", 0.0),
            envelope_corr_std=d.get("envelope_corr_std", 0.0),
            mutual_info_mean=d.get("mutual_info_mean", 0.0),
            normalized_mi_mean=d.get("normalized_mi_mean", 0.0),
            psd_correlation=d.get("psd_correlation", 0.0),
            band_power_errors=d.get("band_power_errors", {}),
            # Fold-level data
            fold_r2s=d.get("fold_r2s", []),
            fold_maes=d.get("fold_maes", []),
            fold_pearsons=d.get("fold_pearsons", []),
            fold_plvs=d.get("fold_plvs", []),
            fold_plis=d.get("fold_plis", []),
            fold_coherences=d.get("fold_coherences", []),
            fold_envelope_corrs=d.get("fold_envelope_corrs", []),
            fold_reconstruction_snrs=d.get("fold_reconstruction_snrs", []),
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

    def compute(self, y_pred: NDArray, y_true: NDArray) -> Dict[str, Any]:
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

        # Spearman correlation
        spearman, spearman_std = self._compute_spearman(y_pred, y_true)
        metrics["spearman"] = spearman
        metrics["spearman_std"] = spearman_std

        # PSD error in dB
        metrics["psd_error_db"] = self._compute_psd_error(y_pred, y_true)

        # Per-frequency-band R²
        band_r2 = self._compute_band_r2(y_pred, y_true)
        for band, r2_val in band_r2.items():
            metrics[f"r2_{band}"] = r2_val

        # DSP Metrics
        dsp_metrics = self._compute_dsp_metrics(y_pred, y_true)
        metrics.update(dsp_metrics)

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

    def _compute_spearman(self, y_pred: NDArray, y_true: NDArray) -> Tuple[float, float]:
        """Compute Spearman correlation per channel, then average."""
        N, C, T = y_pred.shape

        correlations = []
        for c in range(C):
            pred_flat = y_pred[:, c, :].flatten()
            true_flat = y_true[:, c, :].flatten()
            try:
                corr, _ = stats.spearmanr(pred_flat, true_flat)
                if np.isfinite(corr):
                    correlations.append(corr)
            except Exception:
                pass

        if correlations:
            return float(np.mean(correlations)), float(np.std(correlations))
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

    def _compute_dsp_metrics(self, y_pred: NDArray, y_true: NDArray) -> Dict[str, Any]:
        """Compute comprehensive DSP metrics for signal reconstruction quality.

        Computes phase connectivity, coherence, SNR, cross-correlation,
        envelope correlation, mutual information, and PSD comparison metrics.

        Args:
            y_pred: Predictions [N, C, T]
            y_true: Ground truth [N, C, T]

        Returns:
            Dictionary of DSP metric name -> value
        """
        N, C, T = y_pred.shape
        dsp = {}

        # Sample a subset of channels/samples for computational efficiency
        max_samples = min(N, 50)
        max_channels = min(C, 8)
        sample_indices = np.linspace(0, N - 1, max_samples, dtype=int)
        channel_indices = np.linspace(0, C - 1, max_channels, dtype=int)

        # Collect per-sample metrics
        plvs, plis, wplis = [], [], []
        coherences = []
        reconstruction_snrs = []
        cross_corr_maxs, cross_corr_lags = [], []
        envelope_corrs = []
        mutual_infos, normalized_mis = [], []
        psd_corrs = []
        band_coherences = {band: [] for band in FREQUENCY_BANDS.keys()}
        band_power_errs = {band: [] for band in FREQUENCY_BANDS.keys()}

        for n in sample_indices:
            for c in channel_indices:
                pred_signal = y_pred[n, c, :]
                true_signal = y_true[n, c, :]

                try:
                    # Phase connectivity metrics
                    plv = compute_plv(true_signal, pred_signal, fs=self.sample_rate)
                    pli = compute_pli(true_signal, pred_signal, fs=self.sample_rate)
                    wpli = compute_wpli(true_signal, pred_signal, fs=self.sample_rate)
                    plvs.append(plv)
                    plis.append(pli)
                    wplis.append(wpli)

                    # Coherence
                    nperseg = min(256, T // 4) if T >= 256 else T // 2
                    freqs, coh, band_coh = compute_coherence(
                        true_signal, pred_signal,
                        fs=self.sample_rate, nperseg=nperseg
                    )
                    coherences.append(np.mean(coh))
                    for band, val in band_coh.items():
                        if not np.isnan(val):
                            band_coherences[band].append(val)

                    # Reconstruction SNR
                    recon_snr = compute_reconstruction_snr(true_signal, pred_signal)
                    reconstruction_snrs.append(recon_snr)

                    # Cross-correlation
                    max_corr, lag, _ = compute_cross_correlation(
                        true_signal, pred_signal,
                        max_lag=min(int(self.sample_rate), T // 2)
                    )
                    cross_corr_maxs.append(max_corr)
                    cross_corr_lags.append(lag)

                    # Envelope correlation
                    env_corr = compute_envelope_correlation(
                        true_signal, pred_signal, fs=self.sample_rate
                    )
                    envelope_corrs.append(env_corr)

                    # Mutual information
                    mi = compute_mutual_information(true_signal, pred_signal, bins=32)
                    nmi = compute_normalized_mi(true_signal, pred_signal, bins=32)
                    mutual_infos.append(mi)
                    normalized_mis.append(nmi)

                    # PSD comparison
                    psd_comp = compare_psd(true_signal, pred_signal,
                                          fs=self.sample_rate, nperseg=nperseg)
                    psd_corrs.append(psd_comp['psd_correlation'])
                    for band, err in psd_comp['band_power_errors'].items():
                        if not np.isnan(err):
                            band_power_errs[band].append(err)

                except Exception:
                    # Skip failed computations
                    pass

        # Aggregate metrics
        def safe_mean(arr):
            arr = [x for x in arr if np.isfinite(x)]
            return float(np.mean(arr)) if arr else 0.0

        def safe_std(arr):
            arr = [x for x in arr if np.isfinite(x)]
            return float(np.std(arr)) if len(arr) > 1 else 0.0

        dsp['plv'] = safe_mean(plvs)
        dsp['plv_std'] = safe_std(plvs)
        dsp['pli'] = safe_mean(plis)
        dsp['pli_std'] = safe_std(plis)
        dsp['wpli'] = safe_mean(wplis)
        dsp['wpli_std'] = safe_std(wplis)

        dsp['coherence'] = safe_mean(coherences)
        dsp['coherence_std'] = safe_std(coherences)
        dsp['coherence_bands'] = {band: safe_mean(vals) for band, vals in band_coherences.items()}

        dsp['reconstruction_snr_db'] = safe_mean(reconstruction_snrs)

        dsp['cross_corr_max'] = safe_mean(cross_corr_maxs)
        dsp['cross_corr_lag'] = int(np.round(safe_mean(cross_corr_lags)))

        dsp['envelope_corr'] = safe_mean(envelope_corrs)
        dsp['envelope_corr_std'] = safe_std(envelope_corrs)

        dsp['mutual_info'] = safe_mean(mutual_infos)
        dsp['normalized_mi'] = safe_mean(normalized_mis)

        dsp['psd_correlation'] = safe_mean(psd_corrs)
        dsp['band_power_errors'] = {band: safe_mean(errs) for band, errs in band_power_errs.items()}

        return dsp


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
