"""
Runner for Phase 4: Inter vs Intra Session
==========================================

Orchestrates generalization experiments across datasets and split modes.
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
    from torch.utils.data import DataLoader, TensorDataset, Subset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import (
    Phase4Config,
    TrainingConfig,
    ExperimentConfig,
    SplitMode,
    DATASET_CONFIGS,
)
from .data_splitter import (
    SessionSplitter,
    SessionSplit,
    compute_session_statistics,
    identify_easy_hard_sessions,
)
from .trainer import Phase4Trainer


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class ExperimentResult:
    """Result from a single experiment (one fold)."""

    dataset: str
    split_mode: str  # "intra" or "inter"
    fold: int  # CV fold index

    # Performance
    train_r2: float
    val_r2: float
    test_r2: float

    # Training history
    train_losses: List[float]
    val_losses: List[float]
    val_r2s: List[float]

    # Session analysis
    session_stats: Dict[int, Dict[str, float]]
    train_sessions: List[int]
    test_sessions: List[int]

    # Metadata
    n_train: int
    n_val: int
    n_test: int
    epochs_trained: int
    total_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split_mode": self.split_mode,
            "fold": self.fold,
            "train_r2": self.train_r2,
            "val_r2": self.val_r2,
            "test_r2": self.test_r2,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_r2s": self.val_r2s,
            "session_stats": {str(k): v for k, v in self.session_stats.items()},
            "train_sessions": self.train_sessions,
            "test_sessions": self.test_sessions,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "epochs_trained": self.epochs_trained,
            "total_time": self.total_time,
        }


@dataclass
class Phase4Result:
    """Aggregated results from Phase 4."""

    results: List[ExperimentResult]
    summary: Dict[str, Dict[str, Dict[str, float]]]  # dataset -> mode -> stats
    generalization_gaps: Dict[str, float]  # dataset -> gap
    easy_hard_sessions: Dict[str, Tuple[List[int], List[int]]]
    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "generalization_gaps": self.generalization_gaps,
            "easy_hard_sessions": {
                k: {"easy": v[0], "hard": v[1]}
                for k, v in self.easy_hard_sessions.items()
            },
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
    def load(cls, path: Path) -> "Phase4Result":
        """Load results from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        results = [
            ExperimentResult(
                dataset=r["dataset"],
                split_mode=r["split_mode"],
                fold=r.get("fold", r.get("seed", 0)),  # Support old and new format
                train_r2=r["train_r2"],
                val_r2=r["val_r2"],
                test_r2=r["test_r2"],
                train_losses=r["train_losses"],
                val_losses=r["val_losses"],
                val_r2s=r["val_r2s"],
                session_stats={int(k): v for k, v in r["session_stats"].items()},
                train_sessions=r["train_sessions"],
                test_sessions=r["test_sessions"],
                n_train=r["n_train"],
                n_val=r["n_val"],
                n_test=r["n_test"],
                epochs_trained=r["epochs_trained"],
                total_time=r["total_time"],
            )
            for r in data["results"]
        ]

        easy_hard = {
            k: (v["easy"], v["hard"])
            for k, v in data.get("easy_hard_sessions", {}).items()
        }

        return cls(
            results=results,
            summary=data["summary"],
            generalization_gaps=data["generalization_gaps"],
            easy_hard_sessions=easy_hard,
            config=data["config"],
            timestamp=data.get("timestamp", ""),
        )


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset_with_sessions(
    dataset_name: str,
    n_samples: int = 1000,
    seq_len: int = 5000,
) -> Tuple[Any, np.ndarray, int, int]:
    """Load dataset with session information.

    Returns:
        dataset, session_ids, in_channels, out_channels
    """
    try:
        # Try to load real dataset
        from data import load_dataset_with_metadata

        dataset, metadata = load_dataset_with_metadata(dataset_name)
        session_ids = metadata["session_ids"]
        in_channels = metadata["in_channels"]
        out_channels = metadata["out_channels"]

        return dataset, session_ids, in_channels, out_channels

    except (ImportError, FileNotFoundError):
        # Create synthetic data with sessions
        print(f"  Dataset '{dataset_name}' not available, using synthetic data")
        return create_synthetic_dataset_with_sessions(
            dataset_name, n_samples, seq_len
        )


def create_synthetic_dataset_with_sessions(
    dataset_name: str,
    n_samples: int = 1000,
    seq_len: int = 5000,
) -> Tuple[Any, np.ndarray, int, int]:
    """Create synthetic dataset with session labels."""
    config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["olfactory"])
    in_channels = config.in_channels
    out_channels = config.out_channels
    n_sessions = config.n_sessions

    # Create data
    x = torch.randn(n_samples, in_channels, seq_len)
    y = x + 0.3 * torch.randn_like(x)

    # Smooth targets
    kernel = torch.ones(1, 1, 51) / 51
    for c in range(out_channels):
        y[:, c:c+1, :] = torch.nn.functional.conv1d(
            y[:, c:c+1, :], kernel, padding=25
        )

    # Assign sessions
    session_ids = np.random.randint(0, n_sessions, size=n_samples)

    # Add session-specific bias to make sessions distinguishable
    for session in range(n_sessions):
        mask = session_ids == session
        session_bias = np.random.randn() * 0.2
        y[mask] = y[mask] + session_bias

    dataset = TensorDataset(x, y)

    return dataset, session_ids, in_channels, out_channels


def build_model(
    in_channels: int,
    out_channels: int,
    model_config: Dict[str, Any],
) -> Any:
    """Build CondUNet model."""
    try:
        from models import CondUNet1D

        return CondUNet1D(
            in_channels=in_channels,
            out_channels=out_channels,
            base=model_config.get("base_channels", 64),
            n_odors=7,
            emb_dim=128,
            dropout=0.1,
            use_attention=True,
            attention_type=model_config.get("attention_type", "cross_freq_v2"),
            norm_type="instance",
            cond_mode=model_config.get("cond_mode", "none"),
            n_downsample=model_config.get("n_downsample", 2),
        )
    except ImportError:
        # Fallback to simple model
        from phase_three.model_builder import SimplifiedCondUNet

        return SimplifiedCondUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=model_config.get("base_channels", 64),
            n_downsample=model_config.get("n_downsample", 2),
        )


# =============================================================================
# Main Runner
# =============================================================================

def run_single_experiment(
    dataset_name: str,
    split_mode: SplitMode,
    fold: int,
    cv_seed: int,
    training_config: TrainingConfig,
    model_config: Dict[str, Any],
    device: str = "cuda",
    checkpoint_dir: Optional[Path] = None,
    verbose: int = 1,
    n_samples: int = 1000,
) -> ExperimentResult:
    """Run a single experiment (one fold)."""
    # Set seed based on fold
    seed = cv_seed + fold
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    dataset, session_ids, in_channels, out_channels = load_dataset_with_sessions(
        dataset_name, n_samples=n_samples
    )

    # Create splits (use fold as random state for different splits)
    splitter = SessionSplitter(session_ids, random_state=seed)
    split = splitter.split(split_mode)

    if verbose >= 1:
        print(f"\n  [{dataset_name}] {split_mode.value} | fold {fold + 1}")
        print(f"    Train: {split.n_train} | Val: {split.n_val} | Test: {split.n_test}")
        print(f"    Train sessions: {len(split.train_sessions)} | Test sessions: {len(split.test_sessions)}")

    # Create dataloaders
    train_dataset = Subset(dataset, split.train_indices)
    val_dataset = Subset(dataset, split.val_indices)
    test_dataset = Subset(dataset, split.test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Build model
    model = build_model(in_channels, out_channels, model_config)

    # Create experiment config
    exp_config = ExperimentConfig(
        dataset=dataset_name,
        split_mode=split_mode,
        seed=cv_seed + fold,
    )

    # Train
    trainer = Phase4Trainer(
        model=model,
        experiment_config=exp_config,
        training_config=training_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    start_time = time.time()
    history = trainer.train(train_loader, val_loader)
    total_time = time.time() - start_time

    # Evaluate on test set
    test_loss, test_r2, test_preds, test_targets = trainer.evaluate(test_loader)

    # Evaluate on train set (for train R²)
    _, train_r2, _, _ = trainer.evaluate(train_loader)

    if verbose >= 1:
        print(f"    Test R²: {test_r2:.4f} | Train R²: {train_r2:.4f}")

    # Per-session statistics on test set
    test_session_ids = session_ids[split.test_indices]
    session_stats = compute_session_statistics(
        test_session_ids, test_preds, test_targets
    )

    return ExperimentResult(
        dataset=dataset_name,
        split_mode=split_mode.value,
        fold=fold,
        train_r2=train_r2,
        val_r2=history["best_val_r2"],
        test_r2=test_r2,
        train_losses=history["train_losses"],
        val_losses=history["val_losses"],
        val_r2s=history["val_r2s"],
        session_stats=session_stats,
        train_sessions=split.train_sessions,
        test_sessions=split.test_sessions,
        n_train=split.n_train,
        n_val=split.n_val,
        n_test=split.n_test,
        epochs_trained=history["epochs_trained"],
        total_time=total_time,
    )


def run_phase4(config: Phase4Config) -> Phase4Result:
    """Run all Phase 4 experiments with 5-fold cross-validation."""
    print("\n" + "=" * 60)
    print("Phase 4: Inter vs Intra Session Generalization (5-Fold CV)")
    print("=" * 60)
    print(f"Datasets: {config.datasets}")
    print(f"Split modes: {[m.value for m in config.split_modes]}")
    print(f"Cross-validation: {config.n_folds} folds")
    print(f"Total runs: {config.total_runs}")
    print("=" * 60)

    all_results = []

    for dataset in config.datasets:
        print(f"\n{'=' * 40}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'=' * 40}")

        for split_mode in config.split_modes:
            for fold_idx in range(config.n_folds):
                result = run_single_experiment(
                    dataset_name=dataset,
                    split_mode=split_mode,
                    fold=fold_idx,
                    cv_seed=config.cv_seed,
                    training_config=config.training,
                    model_config=config.model_config,
                    device=config.device,
                    checkpoint_dir=config.checkpoint_dir / dataset / f"fold{fold_idx}",
                    verbose=config.verbose,
                )
                all_results.append(result)

    # Aggregate results
    summary = aggregate_results(all_results)
    generalization_gaps = compute_generalization_gaps(summary)
    easy_hard = identify_all_easy_hard_sessions(all_results)

    # Create final result
    phase4_result = Phase4Result(
        results=all_results,
        summary=summary,
        generalization_gaps=generalization_gaps,
        easy_hard_sessions=easy_hard,
        config=config.to_dict(),
    )

    # Print summary
    print_summary(phase4_result)

    # Save results
    output_path = config.output_dir / "phase4_results.json"
    phase4_result.save(output_path)

    return phase4_result


def aggregate_results(
    results: List[ExperimentResult],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate results by dataset and split mode."""
    summary = {}

    for result in results:
        dataset = result.dataset
        mode = result.split_mode

        if dataset not in summary:
            summary[dataset] = {}
        if mode not in summary[dataset]:
            summary[dataset][mode] = {
                "test_r2_values": [],
                "train_r2_values": [],
                "times": [],
            }

        summary[dataset][mode]["test_r2_values"].append(result.test_r2)
        summary[dataset][mode]["train_r2_values"].append(result.train_r2)
        summary[dataset][mode]["times"].append(result.total_time)

    # Compute statistics
    for dataset in summary:
        for mode in summary[dataset]:
            test_r2s = summary[dataset][mode]["test_r2_values"]
            train_r2s = summary[dataset][mode]["train_r2_values"]
            times = summary[dataset][mode]["times"]

            summary[dataset][mode] = {
                "test_r2_mean": float(np.mean(test_r2s)),
                "test_r2_std": float(np.std(test_r2s)),
                "train_r2_mean": float(np.mean(train_r2s)),
                "train_r2_std": float(np.std(train_r2s)),
                "time_mean": float(np.mean(times)),
                "n_runs": len(test_r2s),
            }

    return summary


def compute_generalization_gaps(
    summary: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, float]:
    """Compute generalization gap (intra - inter) for each dataset."""
    gaps = {}

    for dataset, modes in summary.items():
        if "intra" in modes and "inter" in modes:
            intra_r2 = modes["intra"]["test_r2_mean"]
            inter_r2 = modes["inter"]["test_r2_mean"]
            gaps[dataset] = intra_r2 - inter_r2

    return gaps


def identify_all_easy_hard_sessions(
    results: List[ExperimentResult],
) -> Dict[str, Tuple[List[int], List[int]]]:
    """Identify easy/hard sessions for each dataset."""
    easy_hard = {}

    # Group session stats by dataset
    dataset_stats = {}
    for result in results:
        if result.dataset not in dataset_stats:
            dataset_stats[result.dataset] = {}
        for session, stats in result.session_stats.items():
            if session not in dataset_stats[result.dataset]:
                dataset_stats[result.dataset][session] = []
            dataset_stats[result.dataset][session].append(stats["r2"])

    # Average across runs and identify easy/hard
    for dataset, sessions in dataset_stats.items():
        avg_stats = {
            session: {"r2": float(np.mean(r2s))}
            for session, r2s in sessions.items()
        }
        easy, hard = identify_easy_hard_sessions(avg_stats)
        easy_hard[dataset] = (easy, hard)

    return easy_hard


def print_summary(result: Phase4Result):
    """Print summary of Phase 4 results."""
    print("\n" + "=" * 60)
    print("Phase 4 Summary")
    print("=" * 60)

    print("\nTest R² by Dataset and Split Mode:")
    print("-" * 60)
    print(f"{'Dataset':<15} {'Intra R²':<20} {'Inter R²':<20} {'Gap':<10}")
    print("-" * 60)

    for dataset, modes in result.summary.items():
        intra = modes.get("intra", {})
        inter = modes.get("inter", {})

        intra_str = f"{intra.get('test_r2_mean', 0):.4f} ± {intra.get('test_r2_std', 0):.4f}" if intra else "N/A"
        inter_str = f"{inter.get('test_r2_mean', 0):.4f} ± {inter.get('test_r2_std', 0):.4f}" if inter else "N/A"
        gap = result.generalization_gaps.get(dataset, 0)
        gap_str = f"{gap:.4f}"

        print(f"{dataset:<15} {intra_str:<20} {inter_str:<20} {gap_str:<10}")

    print("\nGeneralization Gaps:")
    for dataset, gap in result.generalization_gaps.items():
        print(f"  {dataset}: {gap:.4f}")

    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Inter vs Intra Session")

    # Datasets
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["olfactory", "pfc", "dandi"],
        help="Datasets to evaluate",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")

    # Cross-validation
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Number of CV folds"
    )
    parser.add_argument(
        "--cv-seed", type=int, default=42, help="Random seed for CV splits"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase4",
        help="Output directory",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )

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
    )
    parser.add_argument("--figure-dpi", type=int, default=300)

    # Testing
    parser.add_argument("--dry-run", action="store_true", help="Quick run with minimal epochs")
    parser.add_argument("--fast", action="store_true", help="Run only olfactory dataset")

    args = parser.parse_args()

    # Handle result loading
    if args.load_results:
        print(f"Loading results from {args.load_results}")
        result = Phase4Result.load(args.load_results)

        from .visualization import Phase4Visualizer
        viz = Phase4Visualizer(output_dir=args.output_dir, dpi=args.figure_dpi)
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

    datasets = ["olfactory"] if args.fast else args.datasets

    config = Phase4Config(
        datasets=datasets,
        n_folds=2 if args.dry_run else args.n_folds,
        cv_seed=args.cv_seed,
        training=training_config,
        output_dir=Path(args.output_dir),
        device=args.device,
    )

    # Run experiments
    result = run_phase4(config)

    # Generate figures
    from .visualization import Phase4Visualizer
    viz = Phase4Visualizer(output_dir=config.output_dir, dpi=args.figure_dpi)
    figures = viz.plot_all(result, format=args.figure_format)
    print(f"\nGenerated {len(figures)} figures in {config.output_dir}")


if __name__ == "__main__":
    main()
