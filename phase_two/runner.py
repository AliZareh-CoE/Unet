#!/usr/bin/env python3
"""
Phase 2 Runner: Architecture Screening (5-Fold Cross-Validation)
=================================================================

Main execution script for Phase 2 of the Nature Methods study.
Compares 8 neural architectures on the olfactory dataset using
rigorous 5-fold cross-validation for robust performance estimates.

Usage:
    python -m phase_two.runner --dataset olfactory --epochs 60
    python -m phase_two.runner --dry-run  # Quick test
    python -m phase_two.runner --arch condunet wavenet --n-folds 5

Cross-Validation:
    Each architecture is evaluated using 5-fold CV, where the data
    is split into 5 parts and each part is used as validation once.
    Results are reported as mean ± std across all folds.

Auto-loading Phase 1 Results:
    If --classical-r2 is not specified, Phase 2 will automatically
    search for Phase 1 results in results/phase1/ and use the best
    classical R² as the gate threshold baseline.

Output:
    - JSON results with per-architecture metrics (5-fold CV)
    - Learning curves and comparison figures
    - Model checkpoints per fold
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from phase_two.config import (
    Phase2Config,
    TrainingConfig,
    ARCHITECTURE_LIST,
    FAST_ARCHITECTURES,
    check_gate_threshold,
)
from phase_two.architectures import create_architecture, list_architectures
from phase_two.trainer import Trainer, TrainingResult, create_dataloaders

# Import shared statistical utilities
from utils import (
    compare_methods,
    compare_multiple_methods,
    holm_correction,
    fdr_correction,
    check_assumptions,
    format_mean_ci,
    confidence_interval,
)


# =============================================================================
# Phase 1 Result Loading
# =============================================================================

def load_phase1_classical_r2(phase1_dir: Path = None) -> Optional[float]:
    """Auto-load best classical R² from Phase 1 results.

    Searches for Phase 1 result files and extracts the best R².

    Args:
        phase1_dir: Directory containing Phase 1 results.
                    Defaults to results/phase1/

    Returns:
        Best R² value from Phase 1, or None if not found.
    """
    if phase1_dir is None:
        phase1_dir = PROJECT_ROOT / "results" / "phase1"

    phase1_dir = Path(phase1_dir)

    if not phase1_dir.exists():
        return None

    # Find phase1 result files (sorted by modification time, newest first)
    result_files = sorted(
        phase1_dir.glob("phase1_results*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not result_files:
        return None

    # Load the most recent result file
    result_file = result_files[0]
    try:
        with open(result_file, "r") as f:
            data = json.load(f)

        # Extract best R²
        best_r2 = data.get("best_r2")
        best_method = data.get("best_method", "unknown")

        if best_r2 is not None:
            print(f"[Auto-loaded] Phase 1 best R²: {best_r2:.4f} ({best_method})")
            print(f"  Source: {result_file}")
            return float(best_r2)
    except Exception as e:
        print(f"Warning: Could not load Phase 1 results from {result_file}: {e}")

    return None


# =============================================================================
# Phase 2 Result
# =============================================================================

@dataclass
class Phase2Result:
    """Complete results from Phase 2 architecture screening."""

    results: List[TrainingResult]
    aggregated: Dict[str, Dict[str, float]]  # arch -> {r2_mean, r2_std, ...}
    best_architecture: str
    best_r2: float
    gate_passed: bool
    classical_baseline_r2: float
    config: Dict[str, Any]
    comprehensive_stats: Optional[Dict[str, Any]] = None
    assumption_checks: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Any]] = None  # Architecture -> model state dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "aggregated": self.aggregated,
            "best_architecture": self.best_architecture,
            "best_r2": self.best_r2,
            "gate_passed": self.gate_passed,
            "classical_baseline_r2": self.classical_baseline_r2,
            "config": self.config,
            "comprehensive_stats": self.comprehensive_stats,
            "assumption_checks": self.assumption_checks,
            "has_models": self.models is not None,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def save_models(self, path: Path) -> None:
        """Save trained models to pickle file."""
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
        """Load trained models from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def load(cls, path: Path) -> "Phase2Result":
        """Load results from JSON file.

        Args:
            path: Path to JSON results file

        Returns:
            Phase2Result object

        Example:
            >>> result = Phase2Result.load("results/phase2/phase2_results_20240101.json")
            >>> print(result.best_architecture, result.best_r2)
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct TrainingResult objects
        results = []
        for r in data["results"]:
            results.append(TrainingResult(
                architecture=r["architecture"],
                fold=r.get("fold", r.get("seed", 0)),  # Support both old and new format
                best_val_r2=r["best_val_r2"],
                best_val_mae=r["best_val_mae"],
                best_epoch=r["best_epoch"],
                train_losses=r["train_losses"],
                val_losses=r["val_losses"],
                val_r2s=r["val_r2s"],
                total_time=r["total_time"],
                epochs_trained=r["epochs_trained"],
                n_parameters=r["n_parameters"],
                nan_detected=r.get("nan_detected", False),
                nan_recovery_count=r.get("nan_recovery_count", 0),
                early_stopped=r.get("early_stopped", False),
                error_message=r.get("error_message"),
                peak_gpu_memory_mb=r.get("peak_gpu_memory_mb"),
                completed_successfully=r.get("completed_successfully", True),
            ))

        return cls(
            results=results,
            aggregated=data["aggregated"],
            best_architecture=data["best_architecture"],
            best_r2=data["best_r2"],
            gate_passed=data["gate_passed"],
            classical_baseline_r2=data["classical_baseline_r2"],
            config=data.get("config", {}),
            timestamp=data.get("timestamp", ""),
        )


# =============================================================================
# Data Loading
# =============================================================================

def load_olfactory_data():
    """Load olfactory dataset (full data for cross-validation).

    Returns:
        X: Input signals [N, C, T]
        y: Target signals [N, C, T]
    """
    from data import prepare_data

    print("Loading olfactory dataset...")
    data = prepare_data()

    ob = data["ob"]    # [N, C, T]
    pcx = data["pcx"]  # [N, C, T]

    # Combine train+val for CV (exclude test set for final evaluation)
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]

    all_idx = np.concatenate([train_idx, val_idx])
    X = ob[all_idx]
    y = pcx[all_idx]

    print(f"  Loaded: {X.shape} samples for cross-validation")

    return X, y


def create_synthetic_data(
    n_samples: int = 250,
    n_channels: int = 32,
    time_steps: int = 1000,
    seed: int = 42,
):
    """Create synthetic data for testing (full data for CV)."""
    np.random.seed(seed)

    X = np.random.randn(n_samples, n_channels, time_steps).astype(np.float32)
    # Target is smoothed + nonlinear transform
    y = np.zeros_like(X)
    for i in range(n_samples):
        for c in range(n_channels):
            kernel = np.hanning(30) / np.hanning(30).sum()
            y[i, c] = np.convolve(X[i, c], kernel, mode='same')
            y[i, c] = np.tanh(y[i, c])
    y += 0.1 * np.random.randn(*y.shape).astype(np.float32)

    return X, y


# =============================================================================
# Cross-Validation
# =============================================================================

def create_cv_splits(
    n_samples: int,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create K-fold cross-validation splits.

    Args:
        n_samples: Total number of samples
        n_folds: Number of folds
        seed: Random seed for reproducibility

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
# Main Runner
# =============================================================================

def run_phase2(
    config: Phase2Config,
    X: np.ndarray,
    y: np.ndarray,
    classical_r2: float = 0.35,  # From Phase 1
) -> Phase2Result:
    """Run Phase 2 architecture screening with 5-fold cross-validation.

    Args:
        config: Phase 2 configuration
        X: Input signals [N, C, T]
        y: Target signals [N, C, T]
        classical_r2: Best R² from Phase 1 (for gate check)

    Returns:
        Phase2Result with all metrics
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Architecture Screening (5-Fold Cross-Validation)")
    print("=" * 70)
    print(f"Dataset: {config.dataset}")
    print(f"Architectures: {', '.join(config.architectures)}")
    print(f"Cross-validation: {config.n_folds} folds")
    print(f"Total runs: {config.total_runs}")
    print(f"Device: {config.device}")
    print()

    n_samples = X.shape[0]
    in_channels = X.shape[1]
    out_channels = y.shape[1]
    time_steps = X.shape[2]

    # Create CV splits
    cv_splits = create_cv_splits(n_samples, config.n_folds, config.cv_seed)
    print(f"CV splits created: {[len(s[1]) for s in cv_splits]} validation samples per fold")

    all_results = []
    arch_r2s = {arch: [] for arch in config.architectures}
    all_models = {}  # Store best model for each architecture

    run_idx = 0
    total_runs = config.total_runs

    for arch_name in config.architectures:
        print(f"\n{'='*60}")
        print(f"Architecture: {arch_name.upper()}")
        print("=" * 60)

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] {arch_name} (fold {fold_idx + 1}/{config.n_folds})")

            # Get fold data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Set seed for reproducibility (use fold index as seed offset)
            seed = config.cv_seed + fold_idx
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Create model
            model = create_architecture(
                arch_name,
                in_channels=in_channels,
                out_channels=out_channels,
                time_steps=time_steps,
            )

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Parameters: {n_params:,}")
            print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

            # Create data loaders
            train_loader, val_loader = create_dataloaders(
                X_train, y_train, X_val, y_val,
                batch_size=config.training.batch_size,
                n_workers=config.n_workers,
            )

            # Create trainer with logging
            trainer = Trainer(
                model=model,
                config=config.training,
                device=config.device,
                checkpoint_dir=config.checkpoint_dir / arch_name / f"fold{fold_idx}" if config.save_checkpoints else None,
                log_dir=config.log_dir if hasattr(config, 'log_dir') else None,
            )

            # Train with robustness features
            result = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                arch_name=arch_name,
                fold=fold_idx,
                verbose=config.verbose > 0,
                checkpoint_every=config.training.checkpoint_every,
            )

            all_results.append(result)
            arch_r2s[arch_name].append(result.best_val_r2)

            # Store best model for this architecture (keep the best across folds)
            if arch_name not in all_models or result.best_val_r2 > all_models[arch_name]['r2']:
                all_models[arch_name] = {
                    'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'r2': result.best_val_r2,
                    'fold': fold_idx,
                    'n_parameters': n_params,
                }

            print(f"  Best R²: {result.best_val_r2:.4f} (epoch {result.best_epoch})")
            print(f"  Time: {result.total_time:.1f}s")
            if result.dsp_metrics:
                print(f"  DSP: PLV={result.dsp_metrics.get('plv_mean', 0):.4f}, "
                      f"Coh={result.dsp_metrics.get('coherence_mean', 0):.4f}, "
                      f"SNR={result.dsp_metrics.get('reconstruction_snr_db', 0):.1f}dB")

            # Show robustness info if any issues
            if result.nan_detected:
                print(f"  Warning: {result.nan_recovery_count} NaN batches recovered")
            if not result.completed_successfully:
                print(f"  Error: {result.error_message}")

            # Clean up GPU memory between folds/architectures
            del model, trainer, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Aggregate results per architecture (across CV folds)
    aggregated = {}
    for arch_name in config.architectures:
        r2s = arch_r2s[arch_name]
        valid_r2s = [r for r in r2s if not np.isnan(r)]

        # Get DSP metrics from results
        arch_results = [r for r in all_results if r.architecture == arch_name]
        dsp_agg = {}
        if arch_results and arch_results[0].dsp_metrics:
            dsp_keys = arch_results[0].dsp_metrics.keys()
            for key in dsp_keys:
                vals = [r.dsp_metrics.get(key, 0) for r in arch_results if r.dsp_metrics]
                if vals:
                    dsp_agg[f"{key}_mean"] = float(np.mean(vals))
                    dsp_agg[f"{key}_std"] = float(np.std(vals))

        aggregated[arch_name] = {
            "r2_mean": float(np.mean(valid_r2s)) if valid_r2s else np.nan,
            "r2_std": float(np.std(valid_r2s, ddof=1)) if len(valid_r2s) > 1 else 0.0,
            "r2_min": float(np.min(valid_r2s)) if valid_r2s else np.nan,
            "r2_max": float(np.max(valid_r2s)) if valid_r2s else np.nan,
            "n_folds": len(valid_r2s),
            "fold_r2s": r2s,  # Store individual fold results
            **dsp_agg,  # Include aggregated DSP metrics
        }

    # Find best
    valid_archs = [k for k in aggregated.keys() if not np.isnan(aggregated[k]["r2_mean"])]
    best_arch = max(valid_archs, key=lambda k: aggregated[k]["r2_mean"]) if valid_archs else config.architectures[0]
    best_r2 = aggregated[best_arch]["r2_mean"]

    # Check gate
    gate_passed = check_gate_threshold(best_r2, classical_r2)

    # Comprehensive statistical analysis
    comprehensive_stats = {}
    assumption_checks = {}

    if len(valid_archs) >= 2:
        print("\nComputing comprehensive statistical analysis...")

        best_r2s = aggregated[best_arch].get("fold_r2s", [])

        for arch_name in aggregated.keys():
            if arch_name == best_arch:
                continue

            arch_r2s_list = aggregated[arch_name].get("fold_r2s", [])

            if len(best_r2s) >= 2 and len(arch_r2s_list) >= 2:
                values_best = np.array(best_r2s)
                values_other = np.array(arch_r2s_list)

                # Comprehensive comparison
                comp_result = compare_methods(
                    values_best, values_other,
                    name_a=best_arch, name_b=arch_name,
                    paired=True, alpha=0.05
                )
                comprehensive_stats[arch_name] = comp_result.to_dict()

                # Check assumptions
                assumptions = check_assumptions(values_best, values_other, alpha=0.05)
                assumption_checks[arch_name] = assumptions

        # Apply multiple comparison corrections
        if comprehensive_stats:
            p_values = [comprehensive_stats[m]["parametric_test"]["p_value"]
                       for m in comprehensive_stats]
            significant_holm = holm_correction(p_values, alpha=0.05)
            significant_fdr, adjusted_p = fdr_correction(p_values, alpha=0.05)

            for i, arch_name in enumerate(comprehensive_stats.keys()):
                comprehensive_stats[arch_name]["significant_holm"] = significant_holm[i]
                comprehensive_stats[arch_name]["significant_fdr"] = significant_fdr[i]
                comprehensive_stats[arch_name]["p_fdr_adjusted"] = adjusted_p[i]

    result = Phase2Result(
        results=all_results,
        aggregated=aggregated,
        best_architecture=best_arch,
        best_r2=best_r2,
        gate_passed=gate_passed,
        classical_baseline_r2=classical_r2,
        config=config.to_dict(),
        comprehensive_stats=comprehensive_stats,
        assumption_checks=assumption_checks,
        models=all_models if all_models else None,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 2 RESULTS SUMMARY (5-Fold Cross-Validation)")
    print("=" * 70)

    print("\n{:<15} {:>10} {:>10} {:>12}".format(
        "Architecture", "R² Mean", "R² Std", "Parameters"
    ))
    print("-" * 50)

    # Sort by R²
    sorted_archs = sorted(aggregated.keys(), key=lambda k: aggregated[k]["r2_mean"], reverse=True)
    for arch in sorted_archs:
        stats = aggregated[arch]
        # Find params from results
        arch_results = [r for r in all_results if r.architecture == arch]
        n_params = arch_results[0].n_parameters if arch_results else 0
        marker = " *" if arch == best_arch else ""
        print(f"{arch:<15} {stats['r2_mean']:>10.4f} {stats['r2_std']:>10.4f} {n_params:>12,}{marker}")

    print("-" * 50)

    # Gate check
    print(f"\nClassical baseline R²: {classical_r2:.4f}")
    print(f"Gate threshold: {classical_r2 + 0.10:.4f}")
    print(f"Best neural R²: {best_r2:.4f} ({best_arch})")

    if gate_passed:
        print(f"\n✓ GATE PASSED: {best_arch} beats classical by {best_r2 - classical_r2:.4f}")
    else:
        print(f"\n✗ GATE FAILED: Best neural ({best_r2:.4f}) did not beat classical + 0.10")

    # Print comprehensive statistics summary
    if comprehensive_stats:
        print("\n" + "=" * 70)
        print("STATISTICAL ANALYSIS (vs best architecture)")
        print("=" * 70)
        print("{:<15}{:<12}{:<12}{:<12}{:<8}{:<8}".format("Architecture", "t-test p", "Wilcoxon p", "Cohen's d", "Holm", "FDR"))
        print("-" * 70)
        for arch_name, stats in comprehensive_stats.items():
            t_p = stats["parametric_test"]["p_value"]
            w_p = stats["nonparametric_test"]["p_value"]
            d = stats["parametric_test"]["effect_size"] or 0
            holm_sig = "Yes" if stats.get("significant_holm", False) else "No"
            fdr_sig = "Yes" if stats.get("significant_fdr", False) else "No"
            print(f"{arch_name:<15}{t_p:<12.4f}{w_p:<12.4f}{d:<12.3f}{holm_sig:<8}{fdr_sig:<8}")

        # Assumption check summary
        print("\nAssumption Checks (Normality of differences):")
        for arch_name, checks in assumption_checks.items():
            norm_ok = "OK" if checks["normality_diff"]["ok"] else "VIOLATED"
            rec = checks["recommendation"]
            print(f"  {arch_name}: {norm_ok} (recommended: {rec})")

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Architecture Screening (5-Fold Cross-Validation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m phase_two.runner --dataset olfactory --epochs 60
    python -m phase_two.runner --dry-run
    python -m phase_two.runner --arch condunet wavenet --n-folds 5
        """
    )

    parser.add_argument("--dataset", type=str, default="olfactory",
                       choices=["olfactory", "pfc", "dandi"])
    parser.add_argument("--dry-run", action="store_true",
                       help="Quick test with synthetic data")
    parser.add_argument("--output", type=str, default="results/phase2",
                       help="Output directory")
    parser.add_argument("--arch", type=str, nargs="+", default=None,
                       help="Specific architectures to run")
    parser.add_argument("--n-folds", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--cv-seed", type=int, default=42,
                       help="Random seed for CV splits")
    parser.add_argument("--epochs", type=int, default=60,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--fast", action="store_true",
                       help="Only run fast architectures")
    parser.add_argument("--no-checkpoints", action="store_true",
                       help="Don't save model checkpoints")
    parser.add_argument("--classical-r2", type=float, default=None,
                       help="Phase 1 classical baseline R² (auto-loads from results/phase1/ if not specified)")
    parser.add_argument("--phase1-dir", type=str, default=None,
                       help="Directory containing Phase 1 results for auto-loading")
    parser.add_argument("--load-results", type=str, default=None,
                       help="Load previous results and regenerate figures (path to JSON)")
    parser.add_argument("--figure-format", type=str, default="pdf",
                       choices=["pdf", "png", "svg", "eps"],
                       help="Output format for figures")
    parser.add_argument("--figure-dpi", type=int, default=300,
                       help="DPI for raster figures (png)")

    args = parser.parse_args()

    # Handle result loading mode (figures only)
    if args.load_results:
        from phase_two.visualization import Phase2Visualizer, create_summary_table

        print(f"Loading results from: {args.load_results}")
        result = Phase2Result.load(Path(args.load_results))

        print(f"Loaded Phase 2 results:")
        print(f"  Best architecture: {result.best_architecture} (R² = {result.best_r2:.4f})")
        print(f"  Gate passed: {result.gate_passed}")
        print(f"  Architectures evaluated: {len(result.aggregated)}")

        # Generate figures
        output_dir = Path(args.output)
        viz = Phase2Visualizer(
            output_dir=output_dir / "figures",
            dpi=args.figure_dpi,
        )

        paths = viz.plot_all(result, format=args.figure_format)
        print(f"\nFigures regenerated:")
        for p in paths:
            print(f"  {p}")

        # LaTeX table
        latex_table = create_summary_table(result)
        latex_path = output_dir / "table_architecture_comparison.tex"
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table: {latex_path}")

        print("\nFigure regeneration complete!")
        return result

    # Determine architectures
    if args.arch:
        architectures = args.arch
    elif args.fast:
        architectures = FAST_ARCHITECTURES
    else:
        architectures = ARCHITECTURE_LIST

    # Create config
    training_config = TrainingConfig(
        epochs=args.epochs if not args.dry_run else 5,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    config = Phase2Config(
        dataset=args.dataset,
        architectures=architectures,
        n_folds=args.n_folds if not args.dry_run else 2,  # Use 2 folds for dry-run
        cv_seed=args.cv_seed,
        training=training_config,
        output_dir=Path(args.output),
        save_checkpoints=not args.no_checkpoints,
    )

    # Auto-load classical R² from Phase 1 if not specified
    classical_r2 = args.classical_r2
    if classical_r2 is None:
        phase1_dir = Path(args.phase1_dir) if args.phase1_dir else None
        classical_r2 = load_phase1_classical_r2(phase1_dir)

        if classical_r2 is None:
            classical_r2 = 0.35  # Fallback default
            print(f"Warning: Could not auto-load Phase 1 results. Using default R²={classical_r2}")
            print("  Run Phase 1 first, or specify --classical-r2 manually")
    else:
        print(f"Using specified classical R²: {classical_r2:.4f}")

    # Load data (full dataset for cross-validation)
    if args.dry_run:
        print("[DRY-RUN] Using synthetic data")
        X, y = create_synthetic_data(
            n_samples=100, time_steps=500
        )
    else:
        try:
            X, y = load_olfactory_data()
        except Exception as e:
            print(f"Warning: Could not load data ({e}). Using synthetic.")
            X, y = create_synthetic_data()

    # Run with 5-fold cross-validation
    result = run_phase2(
        config=config,
        X=X,
        y=y,
        classical_r2=classical_r2,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = config.output_dir / f"phase2_results_{timestamp}.json"
    result.save(results_file)
    print(f"\nResults saved to: {results_file}")

    # Save models separately
    if result.models:
        models_file = config.output_dir / f"phase2_models_{timestamp}.pkl"
        result.save_models(models_file)

    print("\nPhase 2 complete!")
    return result


if __name__ == "__main__":
    main()
