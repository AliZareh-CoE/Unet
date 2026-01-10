"""
Runner for Phase 3: CondUNet Ablation Studies
==============================================

Orchestrates ablation experiments comparing different CondUNet configurations.
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
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from .config import (
    Phase3Config,
    AblationConfig,
    TrainingConfig,
    ABLATION_STUDIES,
    get_baseline_config,
)
from .trainer import AblationTrainer
from .model_builder import build_condunet, count_parameters


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class AblationResult:
    """Result from a single ablation experiment."""

    study: str
    variant: str
    seed: int

    # Performance
    best_r2: float
    best_loss: float

    # Training history
    train_losses: List[float]
    val_losses: List[float]
    val_r2s: List[float]

    # Metadata
    n_parameters: int
    epochs_trained: int
    total_time: float
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study": self.study,
            "variant": self.variant,
            "seed": self.seed,
            "best_r2": self.best_r2,
            "best_loss": self.best_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_r2s": self.val_r2s,
            "n_parameters": self.n_parameters,
            "epochs_trained": self.epochs_trained,
            "total_time": self.total_time,
            "config": self.config,
        }


@dataclass
class Phase3Result:
    """Aggregated results from Phase 3 ablation studies."""

    results: List[AblationResult]
    aggregated: Dict[str, Dict[str, Dict[str, float]]]  # study -> variant -> stats
    baseline_r2: float
    optimal_config: Dict[str, str]
    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "aggregated": self.aggregated,
            "baseline_r2": self.baseline_r2,
            "optimal_config": self.optimal_config,
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
    def load(cls, path: Path) -> "Phase3Result":
        """Load results from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        results = [
            AblationResult(
                study=r["study"],
                variant=r["variant"],
                seed=r["seed"],
                best_r2=r["best_r2"],
                best_loss=r["best_loss"],
                train_losses=r["train_losses"],
                val_losses=r["val_losses"],
                val_r2s=r["val_r2s"],
                n_parameters=r["n_parameters"],
                epochs_trained=r["epochs_trained"],
                total_time=r["total_time"],
                config=r["config"],
            )
            for r in data["results"]
        ]

        return cls(
            results=results,
            aggregated=data["aggregated"],
            baseline_r2=data["baseline_r2"],
            optimal_config=data["optimal_config"],
            config=data["config"],
            timestamp=data.get("timestamp", ""),
        )


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(
    dataset_name: str,
    batch_size: int = 32,
    n_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Load dataset and create dataloaders.

    Args:
        dataset_name: Name of dataset ('olfactory', etc.)
        batch_size: Batch size
        n_workers: DataLoader workers

    Returns:
        train_loader, val_loader, in_channels, out_channels
    """
    try:
        # Try to load real dataset
        from data import get_dataloader

        train_loader = get_dataloader(dataset_name, split="train", batch_size=batch_size)
        val_loader = get_dataloader(dataset_name, split="val", batch_size=batch_size)

        # Get channel info from first batch
        sample = next(iter(train_loader))
        in_channels = sample[0].shape[1]
        out_channels = sample[1].shape[1] if len(sample) > 1 else in_channels

        return train_loader, val_loader, in_channels, out_channels

    except (ImportError, FileNotFoundError):
        print(f"Dataset '{dataset_name}' not available, using synthetic data")
        return create_synthetic_data(batch_size, n_workers)


def create_synthetic_data(
    batch_size: int = 32,
    n_workers: int = 4,
    n_samples: int = 1000,
    seq_len: int = 5000,
    in_channels: int = 32,
    out_channels: int = 32,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Create synthetic data for testing."""
    # Training data
    x_train = torch.randn(n_samples, in_channels, seq_len)
    # Create target as smoothed transformation of input
    y_train = x_train + 0.5 * torch.randn_like(x_train)
    # Smooth with simple conv
    kernel = torch.ones(1, 1, 51) / 51
    for c in range(out_channels):
        y_train[:, c:c+1, :] = torch.nn.functional.conv1d(
            y_train[:, c:c+1, :], kernel, padding=25
        )
    odor_ids_train = torch.randint(0, 7, (n_samples,))

    # Validation data (smaller)
    n_val = n_samples // 5
    x_val = torch.randn(n_val, in_channels, seq_len)
    y_val = x_val + 0.5 * torch.randn_like(x_val)
    for c in range(out_channels):
        y_val[:, c:c+1, :] = torch.nn.functional.conv1d(
            y_val[:, c:c+1, :], kernel, padding=25
        )
    odor_ids_val = torch.randint(0, 7, (n_val,))

    train_dataset = TensorDataset(x_train, y_train, odor_ids_train)
    val_dataset = TensorDataset(x_val, y_val, odor_ids_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    return train_loader, val_loader, in_channels, out_channels


# =============================================================================
# Main Runner
# =============================================================================

def run_single_ablation(
    ablation_config: AblationConfig,
    training_config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    in_channels: int,
    out_channels: int,
    seed: int,
    device: str = "cuda",
    checkpoint_dir: Optional[Path] = None,
    verbose: int = 1,
) -> AblationResult:
    """Run a single ablation experiment.

    Args:
        ablation_config: Ablation configuration
        training_config: Training configuration
        train_loader: Training dataloader
        val_loader: Validation dataloader
        in_channels: Input channels
        out_channels: Output channels
        seed: Random seed
        device: Device to use
        checkpoint_dir: Directory for checkpoints
        verbose: Verbosity level

    Returns:
        AblationResult with training metrics
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build model
    model = build_condunet(ablation_config, in_channels, out_channels)
    n_params = count_parameters(model)

    if verbose >= 1:
        print(f"\n  [{ablation_config.study}] {ablation_config.variant}")
        print(f"    Parameters: {n_params:,}")

    # Create trainer
    trainer = AblationTrainer(
        model=model,
        ablation_config=ablation_config,
        training_config=training_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # Train
    start_time = time.time()
    history = trainer.train(train_loader, val_loader)
    total_time = time.time() - start_time

    if verbose >= 1:
        print(f"    Best R²: {history['best_val_r2']:.4f} | Time: {total_time/60:.1f} min")

    return AblationResult(
        study=ablation_config.study,
        variant=ablation_config.variant,
        seed=seed,
        best_r2=history["best_val_r2"],
        best_loss=history["best_val_loss"],
        train_losses=history["train_losses"],
        val_losses=history["val_losses"],
        val_r2s=history["val_r2s"],
        n_parameters=n_params,
        epochs_trained=history["epochs_trained"],
        total_time=total_time,
        config=ablation_config.to_dict(),
    )


def run_phase3(config: Phase3Config) -> Phase3Result:
    """Run all Phase 3 ablation studies.

    Args:
        config: Phase 3 configuration

    Returns:
        Phase3Result with all ablation results
    """
    print("\n" + "=" * 60)
    print("Phase 3: CondUNet Ablation Studies")
    print("=" * 60)
    print(f"Dataset: {config.dataset}")
    print(f"Studies: {config.studies}")
    print(f"Seeds: {config.seeds}")
    print(f"Total runs: {config.total_runs}")
    print("=" * 60)

    # Load data
    train_loader, val_loader, in_channels, out_channels = load_dataset(
        config.dataset,
        batch_size=config.training.batch_size,
        n_workers=config.n_workers,
    )

    # First, run baseline
    print("\n[Baseline] Running full CondUNet configuration...")
    baseline_config = get_baseline_config()
    baseline_results = []

    for seed in config.seeds:
        result = run_single_ablation(
            ablation_config=baseline_config,
            training_config=config.training,
            train_loader=train_loader,
            val_loader=val_loader,
            in_channels=in_channels,
            out_channels=out_channels,
            seed=seed,
            device=config.device,
            checkpoint_dir=config.checkpoint_dir / "baseline",
            verbose=config.verbose,
        )
        baseline_results.append(result)

    baseline_r2 = np.mean([r.best_r2 for r in baseline_results])
    print(f"\nBaseline R²: {baseline_r2:.4f}")

    # Run ablation studies
    all_results = baseline_results.copy()

    for study in config.studies:
        print(f"\n{'=' * 40}")
        print(f"Study: {ABLATION_STUDIES[study]['description']}")
        print(f"{'=' * 40}")

        ablation_configs = [
            AblationConfig(study=study, variant=variant)
            for variant in ABLATION_STUDIES[study]["variants"]
        ]

        for ablation_config in ablation_configs:
            for seed in config.seeds:
                result = run_single_ablation(
                    ablation_config=ablation_config,
                    training_config=config.training,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    seed=seed,
                    device=config.device,
                    checkpoint_dir=config.checkpoint_dir / study,
                    verbose=config.verbose,
                )
                all_results.append(result)

    # Aggregate results
    aggregated = aggregate_results(all_results)

    # Determine optimal configuration
    optimal_config = determine_optimal_config(aggregated)

    # Create final result
    phase3_result = Phase3Result(
        results=all_results,
        aggregated=aggregated,
        baseline_r2=baseline_r2,
        optimal_config=optimal_config,
        config=config.to_dict(),
    )

    # Print summary
    print_summary(phase3_result)

    # Save results
    output_path = config.output_dir / "phase3_results.json"
    phase3_result.save(output_path)

    return phase3_result


def aggregate_results(
    results: List[AblationResult],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate results by study and variant."""
    aggregated = {}

    for result in results:
        study = result.study
        variant = result.variant

        if study not in aggregated:
            aggregated[study] = {}
        if variant not in aggregated[study]:
            aggregated[study][variant] = {
                "r2_values": [],
                "loss_values": [],
                "times": [],
            }

        aggregated[study][variant]["r2_values"].append(result.best_r2)
        aggregated[study][variant]["loss_values"].append(result.best_loss)
        aggregated[study][variant]["times"].append(result.total_time)

    # Compute statistics
    for study in aggregated:
        for variant in aggregated[study]:
            r2_values = aggregated[study][variant]["r2_values"]
            loss_values = aggregated[study][variant]["loss_values"]
            times = aggregated[study][variant]["times"]

            aggregated[study][variant] = {
                "r2_mean": float(np.mean(r2_values)),
                "r2_std": float(np.std(r2_values)),
                "loss_mean": float(np.mean(loss_values)),
                "loss_std": float(np.std(loss_values)),
                "time_mean": float(np.mean(times)),
                "n_runs": len(r2_values),
            }

    return aggregated


def determine_optimal_config(
    aggregated: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, str]:
    """Determine the optimal configuration from ablation results."""
    optimal = {}

    for study, variants in aggregated.items():
        if study == "baseline":
            continue

        # Find variant with highest R²
        best_variant = max(variants.keys(), key=lambda v: variants[v]["r2_mean"])
        optimal[study] = best_variant

    return optimal


def print_summary(result: Phase3Result):
    """Print summary of ablation results."""
    print("\n" + "=" * 60)
    print("Phase 3 Summary")
    print("=" * 60)

    print(f"\nBaseline R²: {result.baseline_r2:.4f}")

    print("\nAblation Results by Study:")
    print("-" * 60)

    for study, variants in result.aggregated.items():
        if study == "baseline":
            continue

        print(f"\n{study.upper()}:")
        sorted_variants = sorted(
            variants.items(), key=lambda x: x[1]["r2_mean"], reverse=True
        )
        for variant, stats in sorted_variants:
            delta = stats["r2_mean"] - result.baseline_r2
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            print(
                f"  {variant:20s} | R²: {stats['r2_mean']:.4f} ± {stats['r2_std']:.4f} | "
                f"Δ: {delta_str}"
            )

    print("\n" + "-" * 60)
    print("Optimal Configuration:")
    for study, variant in result.optimal_config.items():
        print(f"  {study}: {variant}")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: CondUNet Ablation Studies")

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="olfactory", help="Dataset name"
    )

    # Studies to run
    parser.add_argument(
        "--studies",
        type=str,
        nargs="+",
        default=None,
        help="Ablation studies to run (default: all)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")

    # Seeds
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42], help="Random seeds"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase3",
        help="Output directory",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")

    # Load existing results
    parser.add_argument(
        "--load-results",
        type=str,
        default=None,
        help="Load existing results and regenerate figures",
    )

    # Figure options
    parser.add_argument(
        "--figure-format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Figure format",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
        help="Figure DPI",
    )

    # Testing
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick run with minimal epochs",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only attention ablation for quick testing",
    )

    args = parser.parse_args()

    # Handle result loading
    if args.load_results:
        print(f"Loading results from {args.load_results}")
        result = Phase3Result.load(args.load_results)

        # Generate figures
        from .visualization import Phase3Visualizer

        viz = Phase3Visualizer(output_dir=args.output_dir, dpi=args.figure_dpi)
        figures = viz.plot_all(result, format=args.figure_format)
        print(f"Generated {len(figures)} figures")
        return

    # Build configuration
    training_config = TrainingConfig(
        epochs=5 if args.dry_run else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=3 if args.dry_run else args.patience,
    )

    studies = ["attention"] if args.fast else (args.studies or list(ABLATION_STUDIES.keys()))

    config = Phase3Config(
        dataset=args.dataset,
        studies=studies,
        seeds=args.seeds,
        training=training_config,
        output_dir=Path(args.output_dir),
        device=args.device,
        n_workers=args.workers,
    )

    # Run ablations
    result = run_phase3(config)

    # Generate figures
    from .visualization import Phase3Visualizer

    viz = Phase3Visualizer(output_dir=config.output_dir, dpi=args.figure_dpi)
    figures = viz.plot_all(result, format=args.figure_format)
    print(f"\nGenerated {len(figures)} figures in {config.output_dir}")


if __name__ == "__main__":
    main()
