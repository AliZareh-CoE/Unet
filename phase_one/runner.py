#!/usr/bin/env python3
"""
Phase 1 Runner: Classical Baselines Evaluation
===============================================

Main execution script for Phase 1 of the Nature Methods study.
Evaluates 7 classical signal processing baselines on the olfactory dataset.

Usage:
    python -m phase_one.runner --dataset olfactory
    python -m phase_one.runner --dry-run  # Quick test with synthetic data

Output:
    - JSON results file with all metrics
    - PDF figures for publication
    - Console summary
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
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

from phase_one.config import (
    Phase1Config,
    BASELINE_METHODS,
    FAST_BASELINES,
    GATE_THRESHOLD,
    get_gate_threshold,
    NEURAL_BANDS,
)
from phase_one.baselines import create_baseline, list_baselines, BASELINE_REGISTRY
from phase_one.metrics import (
    MetricsCalculator,
    Phase1Metrics,
    StatisticalComparison,
    compute_statistical_comparisons,
    format_metrics_table,
    rank_methods,
)
from phase_one.visualization import Phase1Visualizer, create_summary_table

# Import shared statistical utilities
from utils import (
    compare_methods,
    compare_multiple_methods,
    holm_correction,
    fdr_correction,
    check_assumptions,
    format_mean_ci,
    TestResult,
    ComparisonResult,
)


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class Phase1Result:
    """Complete results from Phase 1 evaluation.

    Attributes:
        metrics: List of metrics for each baseline
        best_method: Name of best performing method
        best_r2: R² of best method
        gate_threshold: Minimum R² for neural methods to pass
        statistical_comparisons: Pairwise method comparisons
        comprehensive_stats: Enhanced statistical analysis with both parametric/non-parametric tests
        assumption_checks: Results of normality and homogeneity tests
        models: Dictionary of trained models (method_name -> model object)
        config: Configuration used
        timestamp: Evaluation timestamp
    """

    metrics: List[Phase1Metrics]
    best_method: str
    best_r2: float
    gate_threshold: float
    statistical_comparisons: List[StatisticalComparison] = field(default_factory=list)
    comprehensive_stats: Optional[Dict[str, Any]] = None
    assumption_checks: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Any]] = None  # Trained models for each method
    config: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "best_method": self.best_method,
            "best_r2": self.best_r2,
            "gate_threshold": self.gate_threshold,
            "statistical_comparisons": [
                {
                    "method1": c.method1,
                    "method2": c.method2,
                    "mean_diff": c.mean_diff,
                    "t_stat": c.t_stat,
                    "p_value": c.p_value,
                    "p_adjusted": c.p_adjusted,
                    "cohens_d": c.cohens_d,
                    "effect_size": c.effect_size,
                    "significant": c.significant,
                }
                for c in self.statistical_comparisons
            ],
            "comprehensive_stats": self.comprehensive_stats,
            "assumption_checks": self.assumption_checks,
            "has_models": self.models is not None,
            "config": self.config,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def save_models(self, path: Path) -> None:
        """Save trained models to pickle file.

        Args:
            path: Path to save models (will add .pkl extension if not present)
        """
        if self.models is None:
            print("No models to save")
            return

        model_path = Path(path)
        if model_path.suffix != '.pkl':
            model_path = model_path.with_suffix('.pkl')

        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(self.models, f)

        print(f"Models saved to: {model_path}")

    @staticmethod
    def load_models(path: Path) -> Dict[str, Any]:
        """Load trained models from pickle file.

        Args:
            path: Path to models pickle file

        Returns:
            Dictionary of method_name -> model object
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def load(cls, path: Path) -> "Phase1Result":
        """Load results from JSON file.

        Args:
            path: Path to JSON results file

        Returns:
            Phase1Result object

        Example:
            >>> result = Phase1Result.load("results/phase1/phase1_results_20240101.json")
            >>> print(result.best_method, result.best_r2)
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct Phase1Metrics objects
        metrics = []
        for m in data["metrics"]:
            metrics.append(Phase1Metrics(
                method=m["method"],
                r2_mean=m["r2_mean"],
                r2_std=m["r2_std"],
                r2_ci=tuple(m["r2_ci"]),
                mae_mean=m["mae_mean"],
                mae_std=m["mae_std"],
                pearson_mean=m["pearson_mean"],
                pearson_std=m["pearson_std"],
                psd_error_db=m.get("psd_error_db", 0.0),
                band_r2=m.get("band_r2", {}),
                fold_r2s=m.get("fold_r2s", []),
                fold_maes=m.get("fold_maes", []),
                fold_pearsons=m.get("fold_pearsons", []),
                n_folds=m.get("n_folds", 5),
                n_samples=m.get("n_samples", 0),
            ))

        # Reconstruct StatisticalComparison objects
        comparisons = []
        for c in data.get("statistical_comparisons", []):
            comparisons.append(StatisticalComparison(
                method1=c["method1"],
                method2=c["method2"],
                mean_diff=c["mean_diff"],
                t_stat=c["t_stat"],
                p_value=c["p_value"],
                p_adjusted=c["p_adjusted"],
                cohens_d=c["cohens_d"],
                effect_size=c["effect_size"],
                significant=c["significant"],
            ))

        return cls(
            metrics=metrics,
            best_method=data["best_method"],
            best_r2=data["best_r2"],
            gate_threshold=data["gate_threshold"],
            statistical_comparisons=comparisons,
            config=data.get("config"),
            timestamp=data.get("timestamp", ""),
        )


# =============================================================================
# Data Loading
# =============================================================================

def load_olfactory_data() -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Load olfactory dataset (OB -> PCx translation).

    Returns:
        X: Input signals (OB) [N, C, T]
        y: Target signals (PCx) [N, C, T]
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices (held out)
    """
    from data import prepare_data

    print("Loading olfactory dataset...")
    data = prepare_data()

    ob = data["ob"]    # [N, C, T] - Olfactory Bulb
    pcx = data["pcx"]  # [N, C, T] - Piriform Cortex

    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    print(f"  Loaded: OB {ob.shape} -> PCx {pcx.shape}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    return ob, pcx, train_idx, val_idx, test_idx


def create_synthetic_data(
    n_samples: int = 200,
    n_channels: int = 32,
    time_points: int = 1000,
    seed: int = 42,
) -> Tuple[NDArray, NDArray]:
    """Create synthetic data for testing.

    Generates correlated signals to simulate neural translation task.

    Args:
        n_samples: Number of samples
        n_channels: Number of channels
        time_points: Time points per sample
        seed: Random seed

    Returns:
        X: Input signals [N, C, T]
        y: Target signals [N, C, T]
    """
    np.random.seed(seed)

    # Generate correlated signals
    X = np.random.randn(n_samples, n_channels, time_points).astype(np.float32)

    # Target is filtered/transformed version of input + noise
    y = np.zeros_like(X)
    for i in range(n_samples):
        for c in range(n_channels):
            # Simple smoothing + delay + noise
            kernel = np.hanning(50) / np.hanning(50).sum()
            y[i, c, :] = np.convolve(X[i, c, :], kernel, mode='same')
            y[i, c, :] += 0.3 * np.random.randn(time_points)

    return X, y


# =============================================================================
# Cross-Validation
# =============================================================================

def create_cv_splits(
    n_samples: int,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[NDArray, NDArray]]:
    """Create K-fold cross-validation splits.

    Args:
        n_samples: Total number of samples
        n_folds: Number of folds
        seed: Random seed

    Returns:
        List of (train_indices, val_indices) tuples
    """
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // n_folds

    splits = []
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n_samples

        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        splits.append((train_idx, val_idx))

    return splits


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_baseline(
    baseline_name: str,
    X: NDArray,
    y: NDArray,
    config: Phase1Config,
    return_model: bool = True,
) -> Tuple[Phase1Metrics, Optional[NDArray], Optional[Any]]:
    """Evaluate a single baseline with cross-validation.

    Args:
        baseline_name: Name of baseline method
        X: Input signals [N, C, T]
        y: Target signals [N, C, T]
        config: Configuration object
        return_model: Whether to return a trained model on full data

    Returns:
        Phase1Metrics object, optional predictions on last fold, and optional trained model
    """
    n_samples = X.shape[0]
    splits = create_cv_splits(n_samples, config.n_folds, config.seed)

    metrics_calc = MetricsCalculator(
        sample_rate=config.sample_rate,
        n_bootstrap=config.n_bootstrap,
        ci_level=config.ci_level,
    )

    fold_results = []
    last_predictions = None

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        try:
            # Create and fit model
            model = create_baseline(baseline_name)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Compute metrics (including DSP metrics)
            metrics = metrics_calc.compute(y_pred, y_val)
            fold_results.append(metrics)

            # Save last fold predictions
            if fold_idx == len(splits) - 1:
                last_predictions = (y_pred, y_val)

        except Exception as e:
            print(f"    Warning: Fold {fold_idx} failed for {baseline_name}: {e}")
            fold_results.append({"r2": np.nan, "mae": np.nan, "pearson": np.nan})

    # Aggregate across folds - basic metrics
    fold_r2s = [r.get("r2", np.nan) for r in fold_results]
    fold_maes = [r.get("mae", np.nan) for r in fold_results]
    fold_pearsons = [r.get("pearson", np.nan) for r in fold_results]
    fold_spearmans = [r.get("spearman", np.nan) for r in fold_results]

    valid_r2s = np.array([r for r in fold_r2s if not np.isnan(r)])
    valid_maes = np.array([m for m in fold_maes if not np.isnan(m)])
    valid_pearsons = np.array([p for p in fold_pearsons if not np.isnan(p)])
    valid_spearmans = np.array([s for s in fold_spearmans if not np.isnan(s)])

    # Compute bootstrap CI
    ci = metrics_calc.bootstrap_ci(np.array(fold_r2s))

    # Aggregate band R² (average across folds)
    band_r2 = {}
    for band in NEURAL_BANDS.keys():
        band_vals = [r.get(f"r2_{band}", np.nan) for r in fold_results]
        valid_band = [v for v in band_vals if not np.isnan(v)]
        band_r2[band] = float(np.mean(valid_band)) if valid_band else np.nan

    # PSD error
    psd_errors = [r.get("psd_error_db", np.nan) for r in fold_results]
    valid_psd = [p for p in psd_errors if not np.isnan(p)]
    psd_error_db = float(np.mean(valid_psd)) if valid_psd else np.nan

    # Aggregate DSP metrics across folds
    def aggregate_metric(key):
        vals = [r.get(key, np.nan) for r in fold_results]
        valid = [v for v in vals if not np.isnan(v)]
        return (float(np.mean(valid)) if valid else 0.0,
                float(np.std(valid)) if len(valid) > 1 else 0.0,
                vals)

    plv_mean, plv_std, fold_plvs = aggregate_metric("plv")
    pli_mean, pli_std, fold_plis = aggregate_metric("pli")
    wpli_mean, wpli_std, _ = aggregate_metric("wpli")
    coherence_mean, coherence_std, fold_coherences = aggregate_metric("coherence")
    recon_snr, _, fold_snrs = aggregate_metric("reconstruction_snr_db")
    cross_corr_max, _, _ = aggregate_metric("cross_corr_max")
    cross_corr_lag = int(np.nanmean([r.get("cross_corr_lag", 0) for r in fold_results]))
    env_corr_mean, env_corr_std, fold_env_corrs = aggregate_metric("envelope_corr")
    mi_mean, _, _ = aggregate_metric("mutual_info")
    nmi_mean, _, _ = aggregate_metric("normalized_mi")
    psd_corr_mean, _, _ = aggregate_metric("psd_correlation")

    # Aggregate coherence bands
    coherence_bands = {}
    for band in NEURAL_BANDS.keys():
        band_vals = []
        for r in fold_results:
            coh_bands = r.get("coherence_bands", {})
            if band in coh_bands and not np.isnan(coh_bands[band]):
                band_vals.append(coh_bands[band])
        coherence_bands[band] = float(np.mean(band_vals)) if band_vals else 0.0

    # Aggregate band power errors
    band_power_errors = {}
    for band in NEURAL_BANDS.keys():
        band_vals = []
        for r in fold_results:
            bp_errors = r.get("band_power_errors", {})
            if band in bp_errors and not np.isnan(bp_errors[band]):
                band_vals.append(bp_errors[band])
        band_power_errors[band] = float(np.mean(band_vals)) if band_vals else 0.0

    result = Phase1Metrics(
        method=baseline_name,
        r2_mean=float(np.mean(valid_r2s)) if len(valid_r2s) > 0 else np.nan,
        r2_std=float(np.std(valid_r2s, ddof=1)) if len(valid_r2s) > 1 else 0.0,
        r2_ci=ci,
        mae_mean=float(np.mean(valid_maes)) if len(valid_maes) > 0 else np.nan,
        mae_std=float(np.std(valid_maes, ddof=1)) if len(valid_maes) > 1 else 0.0,
        pearson_mean=float(np.mean(valid_pearsons)) if len(valid_pearsons) > 0 else np.nan,
        pearson_std=float(np.std(valid_pearsons, ddof=1)) if len(valid_pearsons) > 1 else 0.0,
        spearman_mean=float(np.mean(valid_spearmans)) if len(valid_spearmans) > 0 else np.nan,
        spearman_std=float(np.std(valid_spearmans, ddof=1)) if len(valid_spearmans) > 1 else 0.0,
        psd_error_db=psd_error_db,
        band_r2=band_r2,
        fold_r2s=fold_r2s,
        fold_maes=fold_maes,
        fold_pearsons=fold_pearsons,
        n_folds=config.n_folds,
        n_samples=n_samples,
        # DSP metrics
        plv_mean=plv_mean,
        plv_std=plv_std,
        pli_mean=pli_mean,
        pli_std=pli_std,
        wpli_mean=wpli_mean,
        wpli_std=wpli_std,
        coherence_mean=coherence_mean,
        coherence_std=coherence_std,
        coherence_bands=coherence_bands,
        reconstruction_snr_db=recon_snr,
        cross_corr_max=cross_corr_max,
        cross_corr_lag=cross_corr_lag,
        envelope_corr_mean=env_corr_mean,
        envelope_corr_std=env_corr_std,
        mutual_info_mean=mi_mean,
        normalized_mi_mean=nmi_mean,
        psd_correlation=psd_corr_mean,
        band_power_errors=band_power_errors,
        # Fold-level DSP data
        fold_plvs=fold_plvs,
        fold_plis=fold_plis,
        fold_coherences=fold_coherences,
        fold_envelope_corrs=fold_env_corrs,
        fold_reconstruction_snrs=fold_snrs,
    )

    # Train final model on full data if requested
    final_model = None
    if return_model:
        try:
            final_model = create_baseline(baseline_name)
            final_model.fit(X, y)
        except Exception as e:
            print(f"    Warning: Could not train final model for {baseline_name}: {e}")

    return result, last_predictions, final_model


# =============================================================================
# Main Runner
# =============================================================================

def run_phase1(
    config: Phase1Config,
    X: NDArray,
    y: NDArray,
) -> Phase1Result:
    """Run complete Phase 1 evaluation.

    Args:
        config: Configuration object
        X: Input signals [N, C, T]
        y: Target signals [N, C, T]

    Returns:
        Phase1Result with all metrics and analysis
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Classical Baselines Evaluation")
    print("=" * 70)
    print(f"Dataset: {config.dataset}")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Methods: {', '.join(config.baselines)}")
    print(f"CV folds: {config.n_folds}")
    print()

    all_metrics = []
    all_predictions = {}
    all_models = {}
    method_fold_r2s = {}

    # Evaluate each baseline
    for i, baseline_name in enumerate(config.baselines):
        print(f"[{i+1}/{len(config.baselines)}] Evaluating {baseline_name}...")
        start_time = time.time()

        result, predictions, model = evaluate_baseline(baseline_name, X, y, config, return_model=True)
        all_metrics.append(result)

        if predictions is not None:
            all_predictions[baseline_name] = predictions[0]  # y_pred

        if model is not None:
            all_models[baseline_name] = model

        method_fold_r2s[baseline_name] = result.fold_r2s

        elapsed = time.time() - start_time
        ci_str = f"[{result.r2_ci[0]:.4f}, {result.r2_ci[1]:.4f}]"
        print(f"    R² = {result.r2_mean:.4f} ± {result.r2_std:.4f} {ci_str} ({elapsed:.1f}s)")
        # Print key DSP metrics
        print(f"    PLV = {result.plv_mean:.4f}, Coherence = {result.coherence_mean:.4f}, SNR = {result.reconstruction_snr_db:.1f} dB")

    # Find best method
    valid_metrics = [m for m in all_metrics if not np.isnan(m.r2_mean)]
    if not valid_metrics:
        raise RuntimeError("All baselines failed!")

    best = max(valid_metrics, key=lambda m: m.r2_mean)
    gate_threshold = get_gate_threshold(best.r2_mean)

    # Statistical comparisons (legacy)
    comparisons = compute_statistical_comparisons(method_fold_r2s)

    # Comprehensive statistical analysis using new utilities
    comprehensive_stats = {}
    assumption_checks = {}

    if len(method_fold_r2s) >= 2:
        # Compare all methods against the best method
        print("\nComputing comprehensive statistical analysis...")

        for method_name, fold_values in method_fold_r2s.items():
            if method_name == best.method:
                continue

            values_best = np.array(method_fold_r2s[best.method])
            values_other = np.array(fold_values)

            # Skip if not enough data
            if len(values_best) < 2 or len(values_other) < 2:
                continue

            # Comprehensive comparison (parametric + non-parametric)
            comp_result = compare_methods(
                values_best, values_other,
                name_a=best.method, name_b=method_name,
                paired=True, alpha=0.05
            )

            comprehensive_stats[method_name] = comp_result.to_dict()

            # Check assumptions
            assumptions = check_assumptions(values_best, values_other, alpha=0.05)
            assumption_checks[method_name] = assumptions

        # Apply multiple comparison correction (Holm method)
        if comprehensive_stats:
            p_values = [comprehensive_stats[m]["parametric_test"]["p_value"]
                       for m in comprehensive_stats]
            significant_holm = holm_correction(p_values, alpha=0.05)

            for i, method_name in enumerate(comprehensive_stats.keys()):
                comprehensive_stats[method_name]["significant_holm"] = significant_holm[i]

            # Also apply FDR correction
            significant_fdr, adjusted_p = fdr_correction(p_values, alpha=0.05)
            for i, method_name in enumerate(comprehensive_stats.keys()):
                comprehensive_stats[method_name]["significant_fdr"] = significant_fdr[i]
                comprehensive_stats[method_name]["p_fdr_adjusted"] = adjusted_p[i]

    # Create result
    result = Phase1Result(
        metrics=all_metrics,
        best_method=best.method,
        best_r2=best.r2_mean,
        gate_threshold=gate_threshold,
        statistical_comparisons=comparisons,
        comprehensive_stats=comprehensive_stats,
        assumption_checks=assumption_checks,
        models=all_models if all_models else None,
        config=config.to_dict(),
    )

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 1 RESULTS SUMMARY")
    print("=" * 70)
    print(format_metrics_table(all_metrics))

    # DSP Metrics Summary
    print("\nDSP Metrics Summary:")
    print("-" * 90)
    print(f"{'Method':<18}{'PLV':>8}{'PLI':>8}{'Coh':>8}{'Env.Corr':>10}{'SNR(dB)':>10}{'NMI':>8}")
    print("-" * 90)
    for m in sorted(all_metrics, key=lambda x: -x.r2_mean):
        print(f"{m.method:<18}{m.plv_mean:>8.4f}{m.pli_mean:>8.4f}{m.coherence_mean:>8.4f}"
              f"{m.envelope_corr_mean:>10.4f}{m.reconstruction_snr_db:>10.2f}{m.normalized_mi_mean:>8.4f}")

    # Per-band breakdown for top 3
    print("\nPer-Band R² (Top 3 Methods):")
    print("-" * 70)
    print(f"{'Method':<18}" + "".join(f"{b.capitalize():>10}" for b in NEURAL_BANDS.keys()))
    print("-" * 70)

    ranked = rank_methods(all_metrics)[:3]
    for _, m in ranked:
        bands = "".join(f"{m.band_r2.get(b, np.nan):>10.4f}" for b in NEURAL_BANDS.keys())
        print(f"{m.method:<18}{bands}")

    # Statistical summary
    sig_comparisons = [c for c in comparisons if c.significant]
    if sig_comparisons:
        print(f"\nSignificant Differences ({len(sig_comparisons)} comparisons, p < 0.05 Bonferroni):")
        for c in sorted(sig_comparisons, key=lambda x: -abs(x.cohens_d))[:5]:
            print(f"  {c.method1} vs {c.method2}: d={c.cohens_d:.2f} ({c.effect_size}), p_adj={c.p_adjusted:.4f}")

    # Comprehensive stats summary
    if comprehensive_stats:
        print("\nComprehensive Statistical Analysis (vs best method):")
        print("-" * 70)
        print("{:<15}{:<12}{:<12}{:<12}{:<8}{:<8}".format("Method", "t-test p", "Wilcoxon p", "Cohen's d", "Holm", "FDR"))
        print("-" * 70)
        for method_name, stats in comprehensive_stats.items():
            t_p = stats["parametric_test"]["p_value"]
            w_p = stats["nonparametric_test"]["p_value"]
            d = stats["parametric_test"]["effect_size"] or 0
            holm_sig = "Yes" if stats.get("significant_holm", False) else "No"
            fdr_sig = "Yes" if stats.get("significant_fdr", False) else "No"
            print(f"{method_name:<15}{t_p:<12.4f}{w_p:<12.4f}{d:<12.3f}{holm_sig:<8}{fdr_sig:<8}")

        # Assumption check summary
        print("\nAssumption Checks (Normality of differences):")
        for method_name, checks in assumption_checks.items():
            norm_ok = "OK" if checks["normality_diff"]["ok"] else "VIOLATED"
            rec = checks["recommendation"]
            print(f"  {method_name}: {norm_ok} (recommended: {rec})")

    # Gate threshold
    print("\n" + "=" * 70)
    print(f"CLASSICAL FLOOR: R² = {best.r2_mean:.4f} ({best.method})")
    print(f"GATE THRESHOLD: Neural methods must achieve R² >= {gate_threshold:.4f}")
    print("=" * 70)

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for Phase 1."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Classical Baselines Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m phase_one.runner --dataset olfactory
    python -m phase_one.runner --dry-run  # Quick test
    python -m phase_one.runner --output results/phase1/
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="olfactory",
        choices=["olfactory", "pfc", "dandi"],
        help="Dataset to evaluate on (default: olfactory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic data for quick testing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase1",
        help="Output directory for results",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Only run fast baselines (ridge, wiener)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--load-results",
        type=str,
        default=None,
        help="Load previous results and regenerate figures (path to JSON)",
    )
    parser.add_argument(
        "--figure-format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg", "eps"],
        help="Output format for figures",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
        help="DPI for raster figures (png)",
    )

    args = parser.parse_args()

    # Handle result loading mode (figures only)
    if args.load_results:
        print(f"Loading results from: {args.load_results}")
        result = Phase1Result.load(Path(args.load_results))

        print(f"Loaded Phase 1 results:")
        print(f"  Best method: {result.best_method} (R² = {result.best_r2:.4f})")
        print(f"  Gate threshold: {result.gate_threshold:.4f}")
        print(f"  Methods evaluated: {len(result.metrics)}")

        # Generate figures
        output_dir = Path(args.output)
        viz = Phase1Visualizer(
            output_dir=output_dir / "figures",
            sample_rate=result.config.get("sample_rate", 1000.0) if result.config else 1000.0,
        )

        fig_path = viz.plot_main_figure(
            metrics_list=result.metrics,
            predictions=None,  # No predictions when loading
            ground_truth=None,
            filename=f"figure_1_1_classical_baselines.{args.figure_format}",
        )
        print(f"\nFigure regenerated: {fig_path}")

        # Stats figure (simple)
        stats_path = viz.plot_statistical_summary(
            result.metrics,
            filename=f"figure_1_1_stats.{args.figure_format}",
        )
        print(f"Stats figure: {stats_path}")

        # Comprehensive statistical analysis figure
        comprehensive_path = viz.plot_comprehensive_stats_figure(
            result.metrics,
            result.statistical_comparisons,
            filename=f"figure_1_2_statistical_analysis.{args.figure_format}",
        )
        print(f"Comprehensive stats figure: {comprehensive_path}")

        # LaTeX table
        latex_table = create_summary_table(result.metrics)
        latex_path = output_dir / "table_classical_baselines.tex"
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table: {latex_path}")

        print("\nFigure regeneration complete!")
        return result

    # Create configuration
    baselines = FAST_BASELINES if args.fast else BASELINE_METHODS
    config = Phase1Config(
        dataset=args.dataset,
        n_folds=args.folds,
        seed=args.seed,
        output_dir=Path(args.output),
        baselines=baselines,
    )

    # Load data
    if args.dry_run:
        print("[DRY-RUN] Using synthetic data")
        X, y = create_synthetic_data(n_samples=100, time_points=500)
        ground_truth = y
    else:
        try:
            X, y, train_idx, val_idx, test_idx = load_olfactory_data()
            # Use train + val for CV (test is held out for Phase 4)
            cv_idx = np.concatenate([train_idx, val_idx])
            X = X[cv_idx]
            y = y[cv_idx]
            ground_truth = y
        except Exception as e:
            print(f"Warning: Could not load data ({e}). Using synthetic data.")
            X, y = create_synthetic_data()
            ground_truth = y

    # Run evaluation
    result = run_phase1(config, X, y)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = config.output_dir / f"phase1_results_{timestamp}.json"
    result.save(results_file)
    print(f"\nResults saved to: {results_file}")

    # Save models separately (they can be large)
    if result.models:
        models_file = config.output_dir / f"phase1_models_{timestamp}.pkl"
        result.save_models(models_file)

    # Generate figures
    if not args.no_figures:
        print("\nGenerating figures...")
        viz = Phase1Visualizer(
            output_dir=config.output_dir / "figures",
            sample_rate=config.sample_rate,
        )

        # Collect predictions for visualization
        predictions = {}
        for m in result.metrics:
            # Re-run best method to get predictions for visualization
            if m.method == result.best_method:
                model = create_baseline(m.method)
                model.fit(X, y)
                predictions[m.method] = model.predict(X)

        fig_path = viz.plot_main_figure(
            metrics_list=result.metrics,
            predictions=predictions if predictions else None,
            ground_truth=ground_truth,
        )
        print(f"Figure saved to: {fig_path}")

        # Comprehensive statistical analysis figure
        comprehensive_path = viz.plot_comprehensive_stats_figure(
            result.metrics,
            result.statistical_comparisons,
        )
        print(f"Comprehensive stats figure: {comprehensive_path}")

        # LaTeX table
        latex_table = create_summary_table(result.metrics)
        latex_path = config.output_dir / "table_classical_baselines.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {latex_path}")

    print("\nPhase 1 complete!")
    return result


if __name__ == "__main__":
    main()
