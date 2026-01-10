#!/usr/bin/env python3
"""
Phase 2 Runner: Architecture Screening
======================================

Main execution script for Phase 2 of the Nature Methods study.
Compares 8 neural architectures on the olfactory dataset.

Usage:
    python -m phase_two.runner --dataset olfactory --epochs 60
    python -m phase_two.runner --dry-run  # Quick test
    python -m phase_two.runner --arch condunet --seeds 42 43 44

Output:
    - JSON results with per-architecture metrics
    - Learning curves and comparison figures
    - Model checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

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
                seed=r["seed"],
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
    """Load olfactory dataset."""
    from data import prepare_data

    print("Loading olfactory dataset...")
    data = prepare_data()

    ob = data["ob"]    # [N, C, T]
    pcx = data["pcx"]  # [N, C, T]

    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    X_train = ob[train_idx]
    y_train = pcx[train_idx]
    X_val = ob[val_idx]
    y_val = pcx[val_idx]

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    return X_train, y_train, X_val, y_val


def create_synthetic_data(
    n_train: int = 200,
    n_val: int = 50,
    n_channels: int = 32,
    time_steps: int = 1000,
    seed: int = 42,
):
    """Create synthetic data for testing."""
    np.random.seed(seed)

    def generate(n):
        X = np.random.randn(n, n_channels, time_steps).astype(np.float32)
        # Target is smoothed + nonlinear transform
        y = np.zeros_like(X)
        for i in range(n):
            for c in range(n_channels):
                kernel = np.hanning(30) / np.hanning(30).sum()
                y[i, c] = np.convolve(X[i, c], kernel, mode='same')
                y[i, c] = np.tanh(y[i, c])
        y += 0.1 * np.random.randn(*y.shape).astype(np.float32)
        return X, y

    X_train, y_train = generate(n_train)
    X_val, y_val = generate(n_val)

    return X_train, y_train, X_val, y_val


# =============================================================================
# Main Runner
# =============================================================================

def run_phase2(
    config: Phase2Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    classical_r2: float = 0.35,  # From Phase 1
) -> Phase2Result:
    """Run Phase 2 architecture screening.

    Args:
        config: Phase 2 configuration
        X_train, y_train: Training data
        X_val, y_val: Validation data
        classical_r2: Best R² from Phase 1 (for gate check)

    Returns:
        Phase2Result with all metrics
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Architecture Screening")
    print("=" * 70)
    print(f"Dataset: {config.dataset}")
    print(f"Architectures: {', '.join(config.architectures)}")
    print(f"Seeds: {config.seeds}")
    print(f"Total runs: {config.total_runs}")
    print(f"Device: {config.device}")
    print()

    in_channels = X_train.shape[1]
    out_channels = y_train.shape[1]
    time_steps = X_train.shape[2]

    all_results = []
    arch_r2s = {arch: [] for arch in config.architectures}

    run_idx = 0
    total_runs = config.total_runs

    for arch_name in config.architectures:
        print(f"\n{'='*60}")
        print(f"Architecture: {arch_name.upper()}")
        print("=" * 60)

        for seed in config.seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] {arch_name} (seed={seed})")

            # Set seed
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
                checkpoint_dir=config.checkpoint_dir / arch_name if config.save_checkpoints else None,
                log_dir=config.log_dir if hasattr(config, 'log_dir') else None,
            )

            # Train with robustness features
            result = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                arch_name=arch_name,
                seed=seed,
                verbose=config.verbose > 0,
                checkpoint_every=config.training.checkpoint_every,
            )

            all_results.append(result)
            arch_r2s[arch_name].append(result.best_val_r2)

            print(f"  Best R²: {result.best_val_r2:.4f} (epoch {result.best_epoch})")
            print(f"  Time: {result.total_time:.1f}s")

            # Show robustness info if any issues
            if result.nan_detected:
                print(f"  Warning: {result.nan_recovery_count} NaN batches recovered")
            if not result.completed_successfully:
                print(f"  Error: {result.error_message}")

    # Aggregate results per architecture
    aggregated = {}
    for arch_name in config.architectures:
        r2s = arch_r2s[arch_name]
        aggregated[arch_name] = {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "r2_min": float(np.min(r2s)),
            "r2_max": float(np.max(r2s)),
            "n_runs": len(r2s),
        }

    # Find best
    best_arch = max(aggregated.keys(), key=lambda k: aggregated[k]["r2_mean"])
    best_r2 = aggregated[best_arch]["r2_mean"]

    # Check gate
    gate_passed = check_gate_threshold(best_r2, classical_r2)

    result = Phase2Result(
        results=all_results,
        aggregated=aggregated,
        best_architecture=best_arch,
        best_r2=best_r2,
        gate_passed=gate_passed,
        classical_baseline_r2=classical_r2,
        config=config.to_dict(),
    )

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 2 RESULTS SUMMARY")
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

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Architecture Screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m phase_two.runner --dataset olfactory --epochs 60
    python -m phase_two.runner --dry-run
    python -m phase_two.runner --arch condunet wavenet --seeds 42 43 44
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                       help="Random seeds")
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
    parser.add_argument("--classical-r2", type=float, default=0.35,
                       help="Phase 1 classical baseline R²")
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
        seeds=args.seeds if not args.dry_run else [42],
        training=training_config,
        output_dir=Path(args.output),
        save_checkpoints=not args.no_checkpoints,
    )

    # Load data
    if args.dry_run:
        print("[DRY-RUN] Using synthetic data")
        X_train, y_train, X_val, y_val = create_synthetic_data(
            n_train=100, n_val=30, time_steps=500
        )
    else:
        try:
            X_train, y_train, X_val, y_val = load_olfactory_data()
        except Exception as e:
            print(f"Warning: Could not load data ({e}). Using synthetic.")
            X_train, y_train, X_val, y_val = create_synthetic_data()

    # Run
    result = run_phase2(
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        classical_r2=args.classical_r2,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = config.output_dir / f"phase2_results_{timestamp}.json"
    result.save(results_file)
    print(f"\nResults saved to: {results_file}")

    print("\nPhase 2 complete!")
    return result


if __name__ == "__main__":
    main()
