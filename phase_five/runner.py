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
    WINDOW_SIZES,
    STRIDE_RATIOS,
)
from .sliding_window import (
    SlidingWindowDataset,
    generate_continuous_signal,
    continuous_inference,
)
from .benchmark import LatencyBenchmark, BenchmarkResults
from .trainer import Phase5Trainer


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class WindowAblationResult:
    """Result from a single window/stride ablation."""

    dataset: str
    window_size: int
    stride_ratio: float
    seed: int

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
            "seed": self.seed,
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

    window_ablations: List[WindowAblationResult]
    stride_ablations: List[WindowAblationResult]
    benchmark_results: Dict[str, BenchmarkResults]
    summary: Dict[str, Any]
    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_ablations": [r.to_dict() for r in self.window_ablations],
            "stride_ablations": [r.to_dict() for r in self.stride_ablations],
            "benchmark_results": {k: v.to_dict() for k, v in self.benchmark_results.items()},
            "summary": self.summary,
            "config": self.config,
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

        window_ablations = [
            WindowAblationResult(**r) for r in data["window_ablations"]
        ]
        stride_ablations = [
            WindowAblationResult(**r) for r in data["stride_ablations"]
        ]

        # Reconstruct benchmark results (simplified)
        benchmark_results = {}

        return cls(
            window_ablations=window_ablations,
            stride_ablations=stride_ablations,
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

def run_window_ablation(
    dataset: str,
    window_size: int,
    stride_ratio: float,
    seed: int,
    training_config: TrainingConfig,
    model_config: Dict[str, Any],
    device: str = "cuda",
    verbose: int = 1,
) -> WindowAblationResult:
    """Run a single window/stride ablation experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose >= 1:
        print(f"\n  [{dataset}] window={window_size}, stride={stride_ratio}")

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

    return WindowAblationResult(
        dataset=dataset,
        window_size=window_size,
        stride_ratio=stride_ratio,
        seed=seed,
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
    """Run all Phase 5 experiments."""
    print("\n" + "=" * 60)
    print("Phase 5: Real-Time Continuous Processing")
    print("=" * 60)
    print(f"Datasets: {config.datasets}")
    print(f"Window sizes: {config.window_sizes}")
    print(f"Stride ratios: {config.stride_ratios}")
    print(f"Total runs: {config.total_runs}")
    print("=" * 60)

    window_ablations = []
    stride_ablations = []

    # Window size ablations (fixed stride=0.5)
    print("\n" + "-" * 40)
    print("Window Size Ablations")
    print("-" * 40)

    for dataset in config.datasets:
        for window_size in config.window_sizes:
            for seed in config.seeds:
                result = run_window_ablation(
                    dataset=dataset,
                    window_size=window_size,
                    stride_ratio=0.5,  # Fixed stride
                    seed=seed,
                    training_config=config.training,
                    model_config=config.model_config,
                    device=config.device,
                    verbose=config.verbose,
                )
                window_ablations.append(result)

    # Stride ablations (fixed window=5000)
    print("\n" + "-" * 40)
    print("Stride Ratio Ablations")
    print("-" * 40)

    for dataset in config.datasets:
        for stride_ratio in config.stride_ratios:
            for seed in config.seeds:
                result = run_window_ablation(
                    dataset=dataset,
                    window_size=5000,  # Fixed window
                    stride_ratio=stride_ratio,
                    seed=seed,
                    training_config=config.training,
                    model_config=config.model_config,
                    device=config.device,
                    verbose=config.verbose,
                )
                stride_ablations.append(result)

    # Run benchmarks
    benchmark_results = {}
    benchmark_results["default"] = run_benchmarks(
        config.model_config,
        config.benchmark,
        verbose=config.verbose,
    )

    # Create summary
    summary = create_summary(window_ablations, stride_ablations, benchmark_results)

    # Create result
    phase5_result = Phase5Result(
        window_ablations=window_ablations,
        stride_ablations=stride_ablations,
        benchmark_results=benchmark_results,
        summary=summary,
        config=config.to_dict(),
    )

    # Print summary
    print_summary(phase5_result)

    # Save
    output_path = config.output_dir / "phase5_results.json"
    phase5_result.save(output_path)

    return phase5_result


def create_summary(
    window_ablations: List[WindowAblationResult],
    stride_ablations: List[WindowAblationResult],
    benchmark_results: Dict[str, BenchmarkResults],
) -> Dict[str, Any]:
    """Create summary statistics."""
    summary = {
        "window_ablation": {},
        "stride_ablation": {},
        "best_config": {},
        "realtime_feasible": False,
    }

    # Window ablation summary
    for result in window_ablations:
        key = f"{result.dataset}_{result.window_size}"
        if key not in summary["window_ablation"]:
            summary["window_ablation"][key] = {
                "r2_values": [],
                "window_size": result.window_size,
                "dataset": result.dataset,
            }
        summary["window_ablation"][key]["r2_values"].append(result.test_r2)

    for key in summary["window_ablation"]:
        r2s = summary["window_ablation"][key]["r2_values"]
        summary["window_ablation"][key]["r2_mean"] = float(np.mean(r2s))
        summary["window_ablation"][key]["r2_std"] = float(np.std(r2s))

    # Stride ablation summary
    for result in stride_ablations:
        key = f"{result.dataset}_{result.stride_ratio}"
        if key not in summary["stride_ablation"]:
            summary["stride_ablation"][key] = {
                "r2_values": [],
                "stride_ratio": result.stride_ratio,
                "dataset": result.dataset,
            }
        summary["stride_ablation"][key]["r2_values"].append(result.test_r2)

    for key in summary["stride_ablation"]:
        r2s = summary["stride_ablation"][key]["r2_values"]
        summary["stride_ablation"][key]["r2_mean"] = float(np.mean(r2s))
        summary["stride_ablation"][key]["r2_std"] = float(np.std(r2s))

    # Find best config
    all_results = window_ablations + stride_ablations
    if all_results:
        best = max(all_results, key=lambda x: x.test_r2)
        summary["best_config"] = {
            "window_size": best.window_size,
            "stride_ratio": best.stride_ratio,
            "r2": best.test_r2,
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

    print("\nWindow Size Ablation (stride=0.5):")
    print("-" * 40)
    for key, stats in result.summary.get("window_ablation", {}).items():
        print(f"  {stats['dataset']} w={stats['window_size']}: "
              f"R²={stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}")

    print("\nStride Ratio Ablation (window=5000):")
    print("-" * 40)
    for key, stats in result.summary.get("stride_ablation", {}).items():
        print(f"  {stats['dataset']} s={stats['stride_ratio']}: "
              f"R²={stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}")

    print("\nBest Configuration:")
    best = result.summary.get("best_config", {})
    print(f"  Window: {best.get('window_size', 'N/A')}")
    print(f"  Stride: {best.get('stride_ratio', 'N/A')}")
    print(f"  R²: {best.get('r2', 'N/A')}")

    print(f"\nReal-time Feasible: {result.summary.get('realtime_feasible', False)}")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Real-Time Continuous")

    parser.add_argument("--datasets", nargs="+", default=["olfactory", "pfc", "dandi"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
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
    window_sizes = [1000] if args.fast else WINDOW_SIZES
    stride_ratios = [0.5] if args.fast else STRIDE_RATIOS

    config = Phase5Config(
        datasets=datasets,
        window_sizes=window_sizes,
        stride_ratios=stride_ratios,
        seeds=args.seeds,
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
