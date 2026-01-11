"""
Runner for Phase 5: Real-Time Continuous
=========================================

Orchestrates sliding window experiments and benchmarks.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import (
    Phase5Config,
    TrainingConfig,
    ExperimentConfig,
    SlidingWindowConfig,
    BenchmarkConfig,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STRIDE_RATIO,
)
from .sliding_window import (
    SlidingWindowDataset,
    generate_continuous_signal,
    continuous_inference,
)
from .benchmark import LatencyBenchmark, BenchmarkResults
from .trainer import Phase5Trainer

# Import shared statistical utilities
from utils import (
    compare_methods,
    holm_correction,
    fdr_correction,
    confidence_interval,
)


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class ExperimentResult:
    """Result from a single experiment (one fold)."""

    dataset: str
    window_size: int
    stride_ratio: float
    fold: int  # CV fold index

    # Performance
    train_r2: float
    val_r2: float
    test_r2: float

    # Continuous evaluation
    continuous_r2: Optional[float] = None

    # Metadata
    n_windows: int = 0
    epochs_trained: int = 0
    total_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "window_size": self.window_size,
            "stride_ratio": self.stride_ratio,
            "fold": self.fold,
            "train_r2": self.train_r2,
            "val_r2": self.val_r2,
            "test_r2": self.test_r2,
            "continuous_r2": self.continuous_r2,
            "n_windows": self.n_windows,
            "epochs_trained": self.epochs_trained,
            "total_time": self.total_time,
        }


@dataclass
class Phase5Result:
    """Aggregated results from Phase 5."""

    results: List[ExperimentResult]
    benchmark_results: Dict[str, BenchmarkResults]
    summary: Dict[str, Any]
    config: Dict[str, Any]
    comprehensive_stats: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "benchmark_results": {k: v.to_dict() for k, v in self.benchmark_results.items()},
            "summary": self.summary,
            "config": self.config,
            "comprehensive_stats": self.comprehensive_stats,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path):
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Results saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "Phase5Result":
        """Load results from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        results = [
            ExperimentResult(**r) for r in data["results"]
        ]

        # Reconstruct benchmark results (simplified)
        benchmark_results = {}

        return cls(
            results=results,
            benchmark_results=benchmark_results,
            summary=data["summary"],
            config=data["config"],
            timestamp=data.get("timestamp", ""),
        )


# =============================================================================
# Data Loading
# =============================================================================

def create_synthetic_sliding_data(
    window_size: int,
    stride_ratio: float,
    n_channels: int = 32,
    duration_sec: float = 300.0,  # 5 minutes
) -> Tuple[SlidingWindowDataset, SlidingWindowDataset, SlidingWindowDataset, np.ndarray, np.ndarray]:
    """Create synthetic sliding window datasets."""
    # Generate continuous signal
    source, target = generate_continuous_signal(
        duration_sec=duration_sec,
        n_channels=n_channels,
    )

    stride = int(window_size * stride_ratio)
    n_samples = source.shape[1]

    # Split: 70% train, 15% val, 15% test
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)

    train_dataset = SlidingWindowDataset(
        source[:, :train_end].T,
        target[:, :train_end].T,
        window_size=window_size,
        stride=stride,
    )

    val_dataset = SlidingWindowDataset(
        source[:, train_end:val_end].T,
        target[:, train_end:val_end].T,
        window_size=window_size,
        stride=stride,
    )

    test_dataset = SlidingWindowDataset(
        source[:, val_end:].T,
        target[:, val_end:].T,
        window_size=window_size,
        stride=stride,
    )

    # Keep continuous test signal for evaluation
    test_source = source[:, val_end:]
    test_target = target[:, val_end:]

    return train_dataset, val_dataset, test_dataset, test_source, test_target


def build_model(in_channels: int, out_channels: int, model_config: Dict[str, Any]) -> Any:
    """Build model for Phase 5."""
    try:
        from models import CondUNet1D
        return CondUNet1D(
            in_channels=in_channels,
            out_channels=out_channels,
            base=model_config.get("base_channels", 64),
            n_downsample=model_config.get("n_downsample", 2),
            attention_type=model_config.get("attention_type", "cross_freq_v2"),
            cond_mode="none",
        )
    except ImportError:
        from phase_three.model_builder import SimplifiedCondUNet
        return SimplifiedCondUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=model_config.get("base_channels", 64),
            n_downsample=model_config.get("n_downsample", 2),
        )


# =============================================================================
# Experiment Functions
# =============================================================================

def run_experiment(
    dataset: str,
    window_size: int,
    stride_ratio: float,
    fold: int,
    cv_seed: int,
    training_config: TrainingConfig,
    model_config: Dict[str, Any],
    device: str = "cuda",
    verbose: int = 1,
) -> ExperimentResult:
    """Run a single real-time experiment (one fold)."""
    seed = cv_seed + fold
    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose >= 1:
        print(f"\n  [{dataset}] fold {fold + 1}")

    # Create data
    train_ds, val_ds, test_ds, test_source, test_target = create_synthetic_sliding_data(
        window_size=window_size,
        stride_ratio=stride_ratio,
    )

    if verbose >= 1:
        print(f"    Windows: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=training_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=training_config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=training_config.batch_size, shuffle=False)

    # Build model
    in_channels = 32
    out_channels = 32
    model = build_model(in_channels, out_channels, model_config)

    # Train
    trainer = Phase5Trainer(model, training_config, device=device)
    history = trainer.train(train_loader, val_loader)

    # Evaluate
    _, train_r2 = trainer.evaluate(train_loader)
    _, test_r2 = trainer.evaluate(test_loader)

    # Continuous evaluation
    stride = int(window_size * stride_ratio)
    continuous_pred = continuous_inference(
        model, test_source, window_size, stride, device=device
    )

    # Compute continuous R²
    pred_flat = continuous_pred.flatten()
    target_flat = test_target.flatten()
    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - target_flat.mean()) ** 2)
    continuous_r2 = 1 - ss_res / (ss_tot + 1e-8)

    if verbose >= 1:
        print(f"    Test R²: {test_r2:.4f} | Continuous R²: {continuous_r2:.4f}")

    return ExperimentResult(
        dataset=dataset,
        window_size=window_size,
        stride_ratio=stride_ratio,
        fold=fold,
        train_r2=train_r2,
        val_r2=history["best_val_r2"],
        test_r2=test_r2,
        continuous_r2=float(continuous_r2),
        n_windows=len(train_ds) + len(val_ds) + len(test_ds),
        epochs_trained=history["epochs_trained"],
        total_time=history["total_time"],
    )


def run_benchmarks(
    model_config: Dict[str, Any],
    benchmark_config: BenchmarkConfig,
    verbose: int = 1,
) -> BenchmarkResults:
    """Run latency/throughput/memory benchmarks."""
    if verbose >= 1:
        print("\nRunning benchmarks...")

    model = build_model(32, 32, model_config)

    benchmark = LatencyBenchmark(model, benchmark_config)
    results = benchmark.run_all()

    return results


# =============================================================================
# Main Runner
# =============================================================================

def run_phase5(config: Phase5Config) -> Phase5Result:
    """Run all Phase 5 experiments with 5-fold cross-validation."""
    print("\n" + "=" * 60)
    print("Phase 5: Real-Time Continuous Processing (5-Fold CV)")
    print("=" * 60)
    print(f"Datasets: {config.datasets}")
    print(f"Window size: {config.window_size} samples")
    print(f"Stride ratio: {config.stride_ratio}")
    print(f"Cross-validation: {config.n_folds} folds")
    print(f"Total runs: {config.total_runs}")
    print("=" * 60)

    all_results = []

    # Run experiments for each dataset
    for dataset in config.datasets:
        print(f"\n{'=' * 40}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'=' * 40}")

        for fold_idx in range(config.n_folds):
            result = run_experiment(
                dataset=dataset,
                window_size=config.window_size,
                stride_ratio=config.stride_ratio,
                fold=fold_idx,
                cv_seed=config.cv_seed,
                training_config=config.training,
                model_config=config.model_config,
                device=config.device,
                verbose=config.verbose,
            )
            all_results.append(result)

    # Run benchmarks
    benchmark_results = {}
    benchmark_results["default"] = run_benchmarks(
        config.model_config,
        config.benchmark,
        verbose=config.verbose,
    )

    # Create summary
    summary = create_summary(all_results, benchmark_results)

    # Compute comprehensive statistics
    comprehensive_stats = {}
    print("\nComputing statistical analysis...")

    # Compare test vs continuous R² across all datasets
    all_test_r2 = [r.test_r2 for r in all_results]
    all_cont_r2 = [r.continuous_r2 for r in all_results if r.continuous_r2 is not None]

    if len(all_test_r2) >= 2 and len(all_cont_r2) >= 2:
        # Paired comparison (same folds)
        min_len = min(len(all_test_r2), len(all_cont_r2))
        comp = compare_methods(
            np.array(all_test_r2[:min_len]),
            np.array(all_cont_r2[:min_len]),
            name_a="test",
            name_b="continuous",
            paired=True,
            alpha=0.05
        )
        comprehensive_stats["test_vs_continuous"] = comp.to_dict()

    # Per-dataset comparisons (vs overall average)
    for ds, stats in summary.get("by_dataset", {}).items():
        fold_test = stats.get("fold_test_r2s", [])
        fold_cont = stats.get("fold_cont_r2s", [])

        if len(fold_test) >= 2 and len(fold_cont) >= 2:
            comp = compare_methods(
                np.array(fold_test),
                np.array(fold_cont),
                name_a="test",
                name_b="continuous",
                paired=True,
                alpha=0.05
            )
            comprehensive_stats[f"{ds}_test_vs_cont"] = comp.to_dict()

    # Create result
    phase5_result = Phase5Result(
        results=all_results,
        benchmark_results=benchmark_results,
        summary=summary,
        config=config.to_dict(),
        comprehensive_stats=comprehensive_stats,
    )

    # Print summary
    print_summary(phase5_result)

    # Save
    output_path = config.output_dir / "phase5_results.json"
    phase5_result.save(output_path)

    return phase5_result


def create_summary(
    results: List[ExperimentResult],
    benchmark_results: Dict[str, BenchmarkResults],
) -> Dict[str, Any]:
    """Create summary statistics."""
    summary = {
        "by_dataset": {},
        "overall": {},
        "realtime_feasible": False,
    }

    # Group by dataset
    for result in results:
        ds = result.dataset
        if ds not in summary["by_dataset"]:
            summary["by_dataset"][ds] = {
                "test_r2_values": [],
                "continuous_r2_values": [],
            }
        summary["by_dataset"][ds]["test_r2_values"].append(result.test_r2)
        if result.continuous_r2 is not None:
            summary["by_dataset"][ds]["continuous_r2_values"].append(result.continuous_r2)

    # Compute statistics per dataset
    for ds in summary["by_dataset"]:
        test_r2s = summary["by_dataset"][ds]["test_r2_values"]
        cont_r2s = summary["by_dataset"][ds]["continuous_r2_values"]

        summary["by_dataset"][ds] = {
            "test_r2_mean": float(np.mean(test_r2s)),
            "test_r2_std": float(np.std(test_r2s)),
            "fold_test_r2s": test_r2s,  # Store for statistical analysis
            "continuous_r2_mean": float(np.mean(cont_r2s)) if cont_r2s else None,
            "continuous_r2_std": float(np.std(cont_r2s)) if cont_r2s else None,
            "fold_cont_r2s": cont_r2s if cont_r2s else [],
            "n_runs": len(test_r2s),
        }

    # Overall statistics
    all_test_r2 = [r.test_r2 for r in results]
    all_cont_r2 = [r.continuous_r2 for r in results if r.continuous_r2 is not None]

    summary["overall"] = {
        "test_r2_mean": float(np.mean(all_test_r2)),
        "test_r2_std": float(np.std(all_test_r2)),
        "continuous_r2_mean": float(np.mean(all_cont_r2)) if all_cont_r2 else None,
    }

    # Check real-time feasibility
    if "default" in benchmark_results:
        summary["realtime_feasible"] = benchmark_results["default"].meets_all_requirements

    return summary


def print_summary(result: Phase5Result):
    """Print Phase 5 summary."""
    print("\n" + "=" * 60)
    print("Phase 5 Summary")
    print("=" * 60)

    print("\nResults by Dataset:")
    print("-" * 60)
    print(f"{'Dataset':<15} {'Test R²':<20} {'Continuous R²':<20}")
    print("-" * 60)

    for ds, stats in result.summary.get("by_dataset", {}).items():
        test_str = f"{stats['test_r2_mean']:.4f} ± {stats['test_r2_std']:.4f}"
        cont_mean = stats.get('continuous_r2_mean')
        cont_str = f"{cont_mean:.4f}" if cont_mean is not None else "N/A"
        print(f"{ds:<15} {test_str:<20} {cont_str:<20}")

    overall = result.summary.get("overall", {})
    print("-" * 60)
    print(f"{'Overall':<15} {overall.get('test_r2_mean', 0):.4f}")

    print(f"\nReal-time Feasible: {result.summary.get('realtime_feasible', False)}")

    # Print comprehensive statistics
    if result.comprehensive_stats:
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS")
        print("=" * 60)
        for key, stats in result.comprehensive_stats.items():
            t_p = stats["parametric_test"]["p_value"]
            d = stats["parametric_test"]["effect_size"] or 0
            sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
            print(f"{key}: p={t_p:.4f} {sig}, Cohen's d={d:.3f}")

    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Real-Time Continuous")

    parser.add_argument("--datasets", nargs="+", default=["olfactory", "pfc", "dandi"])
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
                       help="Window size in samples (default: 5000)")
    parser.add_argument("--stride-ratio", type=float, default=DEFAULT_STRIDE_RATIO,
                       help="Stride ratio (default: 0.5)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--cv-seed", type=int, default=42, help="Random seed for CV")
    parser.add_argument("--output-dir", type=str, default="results/phase5")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--load-results", type=str, default=None)
    parser.add_argument("--figure-format", type=str, default="pdf")
    parser.add_argument("--figure-dpi", type=int, default=300)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast", action="store_true")

    args = parser.parse_args()

    if args.load_results:
        print(f"Loading results from {args.load_results}")
        result = Phase5Result.load(args.load_results)

        from .visualization import Phase5Visualizer
        viz = Phase5Visualizer(output_dir=args.output_dir, dpi=args.figure_dpi)
        figures = viz.plot_all(result, format=args.figure_format)
        print(f"Generated {len(figures)} figures")
        return

    training_config = TrainingConfig(
        epochs=5 if args.dry_run else args.epochs,
        batch_size=args.batch_size,
        patience=3 if args.dry_run else 15,
    )

    benchmark_config = BenchmarkConfig(
        n_iterations=10 if args.dry_run else 100,
        device=args.device,
    )

    datasets = ["olfactory"] if args.fast else args.datasets

    config = Phase5Config(
        datasets=datasets,
        window_size=args.window_size,
        stride_ratio=args.stride_ratio,
        n_folds=2 if args.dry_run else args.n_folds,
        cv_seed=args.cv_seed,
        training=training_config,
        benchmark=benchmark_config,
        output_dir=Path(args.output_dir),
        device=args.device,
    )

    result = run_phase5(config)

    from .visualization import Phase5Visualizer
    viz = Phase5Visualizer(output_dir=config.output_dir, dpi=args.figure_dpi)
    figures = viz.plot_all(result, format=args.figure_format)
    print(f"\nGenerated {len(figures)} figures in {config.output_dir}")


if __name__ == "__main__":
    main()
