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
    INCREMENTAL_STAGES,
    get_baseline_config,
    get_simple_baseline_config,
    print_protocol_summary,
)
from .trainer import AblationTrainer
from .model_builder import build_condunet, count_parameters

# Import shared statistical utilities
from utils import (
    compare_methods,
    holm_correction,
    fdr_correction,
    check_assumptions,
    confidence_interval,
)

import pickle
import subprocess


# =============================================================================
# train.py Subprocess Support (for FSDP)
# =============================================================================

def save_fold_indices(
    output_dir: Path,
    study: str,
    variant: str,
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Path:
    """Save fold indices to pickle file for train.py.

    Args:
        output_dir: Directory to save indices
        study: Ablation study name
        variant: Ablation variant name
        fold_idx: Fold index
        train_idx: Training sample indices
        val_idx: Validation sample indices

    Returns:
        Path to saved pickle file
    """
    fold_dir = output_dir / "fold_indices"
    fold_dir.mkdir(parents=True, exist_ok=True)

    indices_file = fold_dir / f"{study}_{variant}_fold{fold_idx}_indices.pkl"
    with open(indices_file, 'wb') as f:
        pickle.dump({
            'train_idx': train_idx,
            'val_idx': val_idx,
        }, f)

    return indices_file


def run_train_subprocess(
    study: str,
    variant: str,
    fold_idx: int,
    fold_indices_file: Path,
    output_results_file: Path,
    ablation_config: "AblationConfig",
    epochs: int = 60,
    batch_size: int = 32,
    use_fsdp: bool = False,
    fsdp_strategy: str = "grad_op",
    seed: int = 42,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Run train.py as subprocess for a single ablation experiment.

    This ensures Phase 3 uses the EXACT same training loop as train.py,
    including all augmentations, loss functions, and hyperparameters.

    Args:
        study: Ablation study name
        variant: Ablation variant name
        fold_idx: CV fold index (0-indexed)
        fold_indices_file: Path to pickle file with train/val indices
        output_results_file: Path to JSON file for results output
        ablation_config: Ablation configuration with model parameters
        epochs: Number of training epochs
        batch_size: Batch size (total, will be divided by num GPUs)
        use_fsdp: Whether to use FSDP distributed training
        fsdp_strategy: FSDP sharding strategy
        seed: Random seed
        verbose: Print subprocess output

    Returns:
        Results dict from train.py, or None on failure
    """
    # Build command
    if use_fsdp:
        # Use torchrun for distributed training
        nproc = torch.cuda.device_count() if torch.cuda.is_available() else 1
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            str(PROJECT_ROOT / "train.py"),
        ]
    else:
        cmd = [sys.executable, str(PROJECT_ROOT / "train.py")]

    # Add base arguments (always condunet for ablation studies)
    cmd.extend([
        "--arch", "condunet",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seed", str(seed + fold_idx),  # Different seed per fold
        "--fold-indices-file", str(fold_indices_file),
        "--output-results-file", str(output_results_file),
        "--fold", str(fold_idx),
        "--no-plots",  # Skip plot generation for speed
    ])

    # Add ablation-specific parameters
    config = ablation_config.to_dict()

    # Attention type
    if "attention_type" in config:
        cmd.extend(["--attention-type", config["attention_type"]])

    # Conditioning mode
    if "cond_mode" in config:
        cmd.extend(["--cond-mode", config["cond_mode"]])

    # Base channels (capacity)
    if "base_channels" in config:
        cmd.extend(["--base-channels", str(config["base_channels"])])

    # Loss type
    if "loss_type" in config:
        loss_type = config["loss_type"]
        if loss_type in ["l1", "huber", "wavelet", "l1_wavelet", "huber_wavelet"]:
            cmd.extend(["--loss", loss_type])

    # Convolution type
    if "conv_type" in config:
        cmd.extend(["--conv-type", config["conv_type"]])

    # Augmentation
    if config.get("use_augmentation") is False:
        cmd.append("--no-aug")

    # Bidirectional training
    if config.get("bidirectional") is False:
        cmd.append("--no-bidirectional")

    # FSDP flags
    if use_fsdp:
        cmd.extend(["--fsdp", "--fsdp-strategy", fsdp_strategy])

    # Run subprocess
    start_time = time.time()
    try:
        if verbose:
            # Stream output in real-time
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                check=True,
                text=True,
            )
        else:
            # Capture output silently
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: train.py failed for {study}/{variant} fold {fold_idx}")
        print(f"  Return code: {e.returncode}")
        if e.stderr:
            print(f"  stderr: {e.stderr[:500]}...")
        return None

    elapsed = time.time() - start_time

    # Load results from JSON file
    if output_results_file.exists():
        with open(output_results_file, 'r') as f:
            results = json.load(f)
        results["total_time"] = elapsed
        return results
    else:
        print(f"WARNING: Results file not found: {output_results_file}")
        return None


# =============================================================================
# Checkpoint/Resume Support
# =============================================================================

def get_checkpoint_path(output_dir: Path) -> Path:
    """Get the checkpoint file path for incremental saving."""
    return output_dir / "phase3_checkpoint.pkl"


def save_checkpoint(
    checkpoint_path: Path,
    all_results: List,
    baseline_results: List,
    completed_studies: set,
    baseline_completed: bool,
) -> None:
    """Save checkpoint after each study for resume capability."""
    checkpoint = {
        "all_results": [r.to_dict() if hasattr(r, 'to_dict') else vars(r) for r in all_results],
        "baseline_results": [r.to_dict() if hasattr(r, 'to_dict') else vars(r) for r in baseline_results],
        "completed_studies": list(completed_studies),
        "baseline_completed": baseline_completed,
        "timestamp": datetime.now().isoformat(),
    }

    temp_path = checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    temp_path.rename(checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Load checkpoint if it exists."""
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"\n*** RESUMING FROM CHECKPOINT ***")
        print(f"    Baseline completed: {checkpoint['baseline_completed']}")
        print(f"    Completed studies: {len(checkpoint['completed_studies'])}")
        print(f"    Checkpoint from: {checkpoint['timestamp']}")
        return checkpoint
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None


def reconstruct_from_checkpoint(checkpoint: Dict[str, Any]) -> Tuple[List, List, set, bool]:
    """Reconstruct state from checkpoint."""
    # Reconstruct AblationResult objects
    all_results = []
    for r_dict in checkpoint["all_results"]:
        result = AblationResult(
            study=r_dict["study"],
            variant=r_dict["variant"],
            fold=r_dict["fold"],
            best_r2=r_dict["best_r2"],
            best_loss=r_dict["best_loss"],
            train_history=r_dict.get("train_history", []),
            val_r2_history=r_dict.get("val_r2_history", []),
            n_parameters=r_dict.get("n_parameters", 0),
            training_time=r_dict.get("training_time", 0.0),
        )
        all_results.append(result)

    baseline_results = []
    for r_dict in checkpoint["baseline_results"]:
        result = AblationResult(
            study=r_dict["study"],
            variant=r_dict["variant"],
            fold=r_dict["fold"],
            best_r2=r_dict["best_r2"],
            best_loss=r_dict["best_loss"],
            train_history=r_dict.get("train_history", []),
            val_r2_history=r_dict.get("val_r2_history", []),
            n_parameters=r_dict.get("n_parameters", 0),
            training_time=r_dict.get("training_time", 0.0),
        )
        baseline_results.append(result)

    completed_studies = set(checkpoint["completed_studies"])
    baseline_completed = checkpoint["baseline_completed"]

    return all_results, baseline_results, completed_studies, baseline_completed


# =============================================================================
# JSON Serialization Helper
# =============================================================================

def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class AblationResult:
    """Result from a single ablation experiment (one fold)."""

    study: str
    variant: str
    fold: int  # CV fold index

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
            "fold": self.fold,
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
    comprehensive_stats: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "aggregated": self.aggregated,
            "baseline_r2": self.baseline_r2,
            "optimal_config": self.optimal_config,
            "config": self.config,
            "comprehensive_stats": self.comprehensive_stats,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path):
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_serializer)
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
                fold=r.get("fold", r.get("seed", 0)),  # Support old and new format
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
# Data Loading and Cross-Validation
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


def load_dataset_raw(
    dataset_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Load dataset as raw tensors for cross-validation.

    Args:
        dataset_name: Name of dataset ('olfactory', etc.)

    Returns:
        X, y, odor_ids, in_channels, out_channels
    """
    try:
        from data import prepare_data

        print(f"Loading {dataset_name} dataset...")
        data = prepare_data()

        ob = data["ob"]    # [N, C, T]
        pcx = data["pcx"]  # [N, C, T]

        # Combine train+val for CV
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        all_idx = np.concatenate([train_idx, val_idx])

        X = torch.from_numpy(ob[all_idx]).float()
        y = torch.from_numpy(pcx[all_idx]).float()

        # Create dummy odor IDs if not available
        odor_ids = torch.zeros(len(all_idx), dtype=torch.long)

        in_channels = X.shape[1]
        out_channels = y.shape[1]

        print(f"  Loaded: {X.shape} samples for cross-validation")

        return X, y, odor_ids, in_channels, out_channels

    except (ImportError, FileNotFoundError) as e:
        print(f"Dataset '{dataset_name}' not available ({e}), using synthetic data")
        return create_synthetic_data_raw()


def create_synthetic_data_raw(
    n_samples: int = 1000,
    seq_len: int = 5000,
    in_channels: int = 32,
    out_channels: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Create synthetic data as raw tensors."""
    X = torch.randn(n_samples, in_channels, seq_len)
    y = X + 0.5 * torch.randn_like(X)

    # Smooth with simple conv
    kernel = torch.ones(1, 1, 51) / 51
    for c in range(out_channels):
        y[:, c:c+1, :] = torch.nn.functional.conv1d(
            y[:, c:c+1, :], kernel, padding=25
        )

    odor_ids = torch.randint(0, 7, (n_samples,))

    return X, y, odor_ids, in_channels, out_channels


def create_dataloaders_from_indices(
    X: torch.Tensor,
    y: torch.Tensor,
    odor_ids: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 32,
    n_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders from data and indices."""
    train_dataset = TensorDataset(
        X[train_idx], y[train_idx], odor_ids[train_idx]
    )
    val_dataset = TensorDataset(
        X[val_idx], y[val_idx], odor_ids[val_idx]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    return train_loader, val_loader


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
    fold: int,
    cv_seed: int = 42,
    device: str = "cuda",
    checkpoint_dir: Optional[Path] = None,
    verbose: int = 1,
) -> AblationResult:
    """Run a single ablation experiment (one fold).

    Args:
        ablation_config: Ablation configuration
        training_config: Training configuration
        train_loader: Training dataloader
        val_loader: Validation dataloader
        in_channels: Input channels
        out_channels: Output channels
        fold: Cross-validation fold index
        cv_seed: Base seed for reproducibility
        device: Device to use
        checkpoint_dir: Directory for checkpoints
        verbose: Verbosity level

    Returns:
        AblationResult with training metrics
    """
    # Set seed based on fold for reproducibility
    seed = cv_seed + fold
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build model
    model = build_condunet(ablation_config, in_channels, out_channels)
    n_params = count_parameters(model)

    if verbose >= 1:
        print(f"\n  [{ablation_config.study}] {ablation_config.variant} (fold {fold + 1})")
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
        fold=fold,
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


def run_phase3(
    config: Phase3Config,
    use_train_py: bool = False,
    use_fsdp: bool = False,
    fsdp_strategy: str = "grad_op",
) -> Phase3Result:
    """Run all Phase 3 ablation studies with 5-fold cross-validation.

    Supports two protocols:
    - ADDITIVE (build-up): Start with simple baseline, add components incrementally
    - SUBTRACTIVE (traditional): Start with full model, remove components

    Args:
        config: Phase 3 configuration
        use_train_py: Use train.py subprocess for exact training consistency
        use_fsdp: Enable FSDP distributed training (requires use_train_py)
        fsdp_strategy: FSDP sharding strategy

    Returns:
        Phase3Result with all ablation results
    """
    print("\n" + "=" * 60)
    print("Phase 3: CondUNet Ablation Studies (5-Fold Cross-Validation)")
    print("=" * 60)
    print(f"Protocol: {config.protocol.upper()}")
    print(f"Dataset: {config.dataset}")
    if config.protocol == "additive":
        print(f"Stages: {config.stages} ({len(config.stages)} stages)")
    else:
        print(f"Studies: {config.studies}")
    print(f"Cross-validation: {config.n_folds} folds")
    print(f"Total runs: {config.total_runs}")
    if use_train_py:
        print(f"Training mode: train.py subprocess (exact consistency)")
        if use_fsdp:
            print(f"  FSDP passed to train.py: strategy={fsdp_strategy}")
    print("=" * 60)

    # Route to appropriate protocol handler
    if config.protocol == "additive":
        return _run_additive_protocol(config, use_train_py, use_fsdp, fsdp_strategy)
    else:
        return _run_subtractive_protocol(config, use_train_py, use_fsdp, fsdp_strategy)


def _run_additive_protocol(
    config: Phase3Config,
    use_train_py: bool = False,
    use_fsdp: bool = False,
    fsdp_strategy: str = "grad_op",
) -> Phase3Result:
    """Run ADDITIVE (build-up) ablation protocol.

    Starts with simple baseline (Stage 0) and incrementally adds components.
    Each stage is compared to the previous stage to show incremental gain.
    """
    # Setup checkpoint for resume capability
    checkpoint_path = get_checkpoint_path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load checkpoint for resume
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is not None:
        all_results, baseline_results, completed_studies, _ = reconstruct_from_checkpoint(checkpoint)
    else:
        all_results = []
        baseline_results = []  # For additive: Stage 0 results
        completed_studies = set()  # stage names like "stage0", "stage1", etc.

    # Load raw data for CV
    X, y, odor_ids, in_channels, out_channels = load_dataset_raw(config.dataset)

    # Create CV splits
    cv_splits = create_cv_splits(len(X), config.n_folds, config.cv_seed)
    print(f"CV splits: {[len(s[1]) for s in cv_splits]} validation samples per fold")

    # Get ablation configs (one per stage)
    ablation_configs = config.get_ablation_configs()
    skipped_stages = 0

    for ablation_config in ablation_configs:
        stage_name = ablation_config.study  # e.g., "stage0", "stage1"
        stage_info = INCREMENTAL_STAGES[ablation_config.stage]

        # Check if already completed
        if stage_name in completed_studies:
            skipped_stages += 1
            print(f"\n[{stage_name}] SKIPPED (already completed)")
            continue

        print(f"\n{'=' * 50}")
        print(f"Stage {ablation_config.stage}: {stage_info['name']}")
        print(f"  {stage_info['description']}")
        print(f"{'=' * 50}")

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            if use_train_py:
                # Use train.py subprocess for FSDP support
                fold_indices_file = save_fold_indices(
                    config.output_dir, stage_name, ablation_config.variant, fold_idx, train_idx, val_idx
                )
                output_results_file = config.output_dir / "train_results" / f"{stage_name}_{ablation_config.variant}_fold{fold_idx}_results.json"
                output_results_file.parent.mkdir(parents=True, exist_ok=True)

                print(f"\n  [{stage_name}] {ablation_config.variant} (fold {fold_idx + 1}/{config.n_folds})")
                print(f"    Train: {len(train_idx)}, Val: {len(val_idx)}")
                print(f"    Running train.py subprocess...")

                train_results = run_train_subprocess(
                    study=stage_name,
                    variant=ablation_config.variant,
                    fold_idx=fold_idx,
                    fold_indices_file=fold_indices_file,
                    output_results_file=output_results_file,
                    ablation_config=ablation_config,
                    epochs=config.training.epochs,
                    batch_size=config.training.batch_size,
                    use_fsdp=use_fsdp,
                    fsdp_strategy=fsdp_strategy,
                    seed=config.cv_seed,
                    verbose=True,
                )

                if train_results is not None:
                    result = AblationResult(
                        study=stage_name,
                        variant=ablation_config.variant,
                        fold=fold_idx,
                        best_r2=train_results.get("best_val_r2", 0.0),
                        best_loss=train_results.get("best_val_loss", 0.0),
                        train_losses=train_results.get("train_losses", []),
                        val_losses=train_results.get("val_losses", []),
                        val_r2s=train_results.get("val_r2s", []),
                        n_parameters=train_results.get("n_parameters", 0),
                        epochs_trained=train_results.get("epochs_trained", 0),
                        total_time=train_results.get("total_time", 0.0),
                        config=ablation_config.to_dict(),
                    )
                    print(f"    Best R²: {result.best_r2:.4f}")
                else:
                    result = AblationResult(
                        study=stage_name,
                        variant=ablation_config.variant,
                        fold=fold_idx,
                        best_r2=0.0,
                        best_loss=float('inf'),
                        train_losses=[],
                        val_losses=[],
                        val_r2s=[],
                        n_parameters=0,
                        epochs_trained=0,
                        total_time=0.0,
                        config=ablation_config.to_dict(),
                    )
                    print(f"    FAILED")
            else:
                # Use internal trainer
                train_loader, val_loader = create_dataloaders_from_indices(
                    X, y, odor_ids, train_idx, val_idx,
                    batch_size=config.training.batch_size,
                    n_workers=config.n_workers,
                )

                result = run_single_ablation(
                    ablation_config=ablation_config,
                    training_config=config.training,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    fold=fold_idx,
                    cv_seed=config.cv_seed,
                    device=config.device,
                    checkpoint_dir=config.checkpoint_dir / stage_name / f"fold{fold_idx}",
                    verbose=config.verbose,
                )

            all_results.append(result)

            # Keep track of baseline (stage 0) results for stats
            if ablation_config.stage == 0:
                baseline_results.append(result)

        # Save checkpoint after each stage
        completed_studies.add(stage_name)
        save_checkpoint(checkpoint_path, all_results, baseline_results, completed_studies, True)
        print(f"Checkpoint saved ({len(completed_studies)}/{len(ablation_configs)} stages completed)")

    if skipped_stages > 0:
        print(f"\n*** Resumed run: skipped {skipped_stages} already-completed stages ***")

    # Aggregate results
    aggregated = aggregate_results(all_results)

    # For additive: baseline is Stage 0
    baseline_r2 = 0.0
    if baseline_results:
        baseline_r2 = np.mean([r.best_r2 for r in baseline_results])
    elif "stage0" in aggregated:
        baseline_r2 = aggregated["stage0"]["simple_baseline"]["r2_mean"]

    # Compute incremental statistics (each stage vs previous stage)
    comprehensive_stats = _compute_incremental_stats(aggregated, all_results)

    # Determine optimal config (for additive: final stage is optimal)
    optimal_config = {"final_stage": len(config.stages) - 1}

    # Create final result
    phase3_result = Phase3Result(
        results=all_results,
        aggregated=aggregated,
        baseline_r2=baseline_r2,
        optimal_config=optimal_config,
        config=config.to_dict(),
        comprehensive_stats=comprehensive_stats,
    )

    # Print summary
    _print_additive_summary(phase3_result, config)

    # Save results
    output_path = config.output_dir / "phase3_results.json"
    phase3_result.save(output_path)

    return phase3_result


def _compute_incremental_stats(
    aggregated: Dict[str, Dict[str, Dict[str, float]]],
    all_results: List[AblationResult],
) -> Dict[str, Any]:
    """Compute statistics comparing each stage to the previous stage."""
    comprehensive_stats = {}

    # Group results by stage
    stage_results = {}
    for result in all_results:
        stage = result.study  # e.g., "stage0"
        if stage not in stage_results:
            stage_results[stage] = []
        stage_results[stage].append(result)

    # Get ordered stage names
    stage_names = sorted([s for s in stage_results.keys() if s.startswith("stage")],
                        key=lambda x: int(x.replace("stage", "")))

    # Compare consecutive stages
    for i in range(1, len(stage_names)):
        prev_stage = stage_names[i - 1]
        curr_stage = stage_names[i]

        prev_r2s = [r.best_r2 for r in stage_results.get(prev_stage, [])]
        curr_r2s = [r.best_r2 for r in stage_results.get(curr_stage, [])]

        if len(prev_r2s) >= 2 and len(curr_r2s) >= 2:
            try:
                comp = compare_methods(
                    np.array(prev_r2s),
                    np.array(curr_r2s),
                    name_a=prev_stage,
                    name_b=curr_stage,
                    paired=True,
                    alpha=0.05
                )
                comprehensive_stats[f"{curr_stage}_vs_{prev_stage}"] = comp.to_dict()
            except Exception:
                pass

    # Also compare each stage vs stage0 (baseline)
    if "stage0" in stage_results:
        baseline_r2s = [r.best_r2 for r in stage_results["stage0"]]
        for stage_name in stage_names[1:]:
            curr_r2s = [r.best_r2 for r in stage_results.get(stage_name, [])]
            if len(curr_r2s) >= 2 and len(baseline_r2s) >= 2:
                try:
                    comp = compare_methods(
                        np.array(baseline_r2s),
                        np.array(curr_r2s),
                        name_a="stage0",
                        name_b=stage_name,
                        paired=True,
                        alpha=0.05
                    )
                    comprehensive_stats[f"{stage_name}_vs_stage0"] = comp.to_dict()
                except Exception:
                    pass

    return comprehensive_stats


def _print_additive_summary(result: Phase3Result, config: Phase3Config):
    """Print summary for additive protocol."""
    print("\n" + "=" * 70)
    print("INCREMENTAL COMPONENT ANALYSIS RESULTS")
    print("=" * 70)

    print(f"\nBaseline (Stage 0) R²: {result.baseline_r2:.4f}")
    print("\nProgressive Component Addition:")
    print("-" * 70)
    print(f"{'Stage':<8}{'Component':<25}{'R² Mean ± Std':<20}{'Δ from prev':<15}")
    print("-" * 70)

    prev_r2 = None
    for stage_idx in sorted(config.stages):
        stage_name = f"stage{stage_idx}"
        stage_info = INCREMENTAL_STAGES[stage_idx]

        if stage_name in result.aggregated:
            variant_name = stage_info["name"]
            if variant_name in result.aggregated[stage_name]:
                stats = result.aggregated[stage_name][variant_name]
                r2_mean = stats["r2_mean"]
                r2_std = stats["r2_std"]

                if prev_r2 is not None:
                    delta = r2_mean - prev_r2
                    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                else:
                    delta_str = "baseline"

                print(f"{stage_idx:<8}{stage_info['name']:<25}{r2_mean:.4f} ± {r2_std:.4f}    {delta_str:<15}")
                prev_r2 = r2_mean

    print("-" * 70)

    # Final improvement over baseline
    final_stage = max(config.stages)
    final_name = f"stage{final_stage}"
    if final_name in result.aggregated:
        final_info = INCREMENTAL_STAGES[final_stage]
        if final_info["name"] in result.aggregated[final_name]:
            final_r2 = result.aggregated[final_name][final_info["name"]]["r2_mean"]
            total_gain = final_r2 - result.baseline_r2
            print(f"\nTOTAL IMPROVEMENT: {result.baseline_r2:.4f} → {final_r2:.4f} (Δ = +{total_gain:.4f})")

    print("=" * 70 + "\n")


def _run_subtractive_protocol(
    config: Phase3Config,
    use_train_py: bool = False,
    use_fsdp: bool = False,
    fsdp_strategy: str = "grad_op",
) -> Phase3Result:
    """Run SUBTRACTIVE (traditional) ablation protocol.

    Starts with full model (baseline) and removes components one at a time.
    """

    # Setup checkpoint for resume capability
    checkpoint_path = get_checkpoint_path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load checkpoint for resume
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is not None:
        all_results, baseline_results, completed_studies, baseline_completed = reconstruct_from_checkpoint(checkpoint)
    else:
        all_results = []
        baseline_results = []
        completed_studies = set()
        baseline_completed = False

    # Load raw data for CV
    X, y, odor_ids, in_channels, out_channels = load_dataset_raw(config.dataset)

    # Create CV splits
    cv_splits = create_cv_splits(len(X), config.n_folds, config.cv_seed)
    print(f"CV splits: {[len(s[1]) for s in cv_splits]} validation samples per fold")

    # First, run baseline across all folds (if not already done)
    baseline_config = get_baseline_config()

    if baseline_completed:
        print("\n[Baseline] SKIPPED (already completed)")
        baseline_r2 = np.mean([r.best_r2 for r in baseline_results])
        baseline_std = np.std([r.best_r2 for r in baseline_results])
        print(f"Baseline R²: {baseline_r2:.4f} ± {baseline_std:.4f}")
    else:
        print("\n[Baseline] Running full CondUNet configuration...")
        baseline_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            if use_train_py:
                # Use train.py subprocess for FSDP support
                fold_indices_file = save_fold_indices(
                    config.output_dir, "baseline", "full", fold_idx, train_idx, val_idx
                )
                output_results_file = config.output_dir / "train_results" / f"baseline_full_fold{fold_idx}_results.json"
                output_results_file.parent.mkdir(parents=True, exist_ok=True)

                print(f"\n  [baseline] full (fold {fold_idx + 1}/{config.n_folds})")
                print(f"    Train: {len(train_idx)}, Val: {len(val_idx)}")
                print(f"    Running train.py subprocess...")

                train_results = run_train_subprocess(
                    study="baseline",
                    variant="full",
                    fold_idx=fold_idx,
                    fold_indices_file=fold_indices_file,
                    output_results_file=output_results_file,
                    ablation_config=baseline_config,
                    epochs=config.training.epochs,
                    batch_size=config.training.batch_size,
                    use_fsdp=use_fsdp,
                    fsdp_strategy=fsdp_strategy,
                    seed=config.cv_seed,
                    verbose=True,
                )

                if train_results is not None:
                    result = AblationResult(
                        study="baseline",
                        variant="full",
                        fold=fold_idx,
                        best_r2=train_results.get("best_val_r2", 0.0),
                        best_loss=train_results.get("best_val_loss", 0.0),
                        train_losses=train_results.get("train_losses", []),
                        val_losses=train_results.get("val_losses", []),
                        val_r2s=train_results.get("val_r2s", []),
                        n_parameters=train_results.get("n_parameters", 0),
                        epochs_trained=train_results.get("epochs_trained", 0),
                        total_time=train_results.get("total_time", 0.0),
                        config=baseline_config.to_dict(),
                    )
                    print(f"    Best R²: {result.best_r2:.4f}")
                else:
                    # Failed - create placeholder result
                    result = AblationResult(
                        study="baseline",
                        variant="full",
                        fold=fold_idx,
                        best_r2=0.0,
                        best_loss=float('inf'),
                        train_losses=[],
                        val_losses=[],
                        val_r2s=[],
                        n_parameters=0,
                        epochs_trained=0,
                        total_time=0.0,
                        config=baseline_config.to_dict(),
                    )
                    print(f"    FAILED")
            else:
                # Use internal trainer
                train_loader, val_loader = create_dataloaders_from_indices(
                    X, y, odor_ids, train_idx, val_idx,
                    batch_size=config.training.batch_size,
                    n_workers=config.n_workers,
                )

                result = run_single_ablation(
                    ablation_config=baseline_config,
                    training_config=config.training,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    fold=fold_idx,
                    cv_seed=config.cv_seed,
                    device=config.device,
                    checkpoint_dir=config.checkpoint_dir / "baseline" / f"fold{fold_idx}",
                    verbose=config.verbose,
                )
            baseline_results.append(result)

        baseline_r2 = np.mean([r.best_r2 for r in baseline_results])
        baseline_std = np.std([r.best_r2 for r in baseline_results])
        print(f"\nBaseline R²: {baseline_r2:.4f} ± {baseline_std:.4f}")

        # Save checkpoint after baseline
        baseline_completed = True
        all_results = baseline_results.copy()
        save_checkpoint(checkpoint_path, all_results, baseline_results, completed_studies, baseline_completed)
        print("Checkpoint saved after baseline")

    # Run ablation studies (one ablated variant per study)
    if not baseline_completed:
        # Already set above
        pass
    else:
        # When resuming, start with loaded results
        if len(all_results) == 0:
            all_results = baseline_results.copy()

    ablation_configs = config.get_ablation_configs()  # One per study
    skipped_studies = 0

    for ablation_config in ablation_configs:
        study_name = ablation_config.study

        # Check if already completed
        if study_name in completed_studies:
            skipped_studies += 1
            print(f"\n[{study_name}] SKIPPED (already completed)")
            continue

        print(f"\n{'=' * 40}")
        print(f"Study: {ABLATION_STUDIES[ablation_config.study]['description']}")
        print(f"Ablated variant: {ablation_config.variant}")
        print(f"{'=' * 40}")

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            if use_train_py:
                # Use train.py subprocess for FSDP support
                fold_indices_file = save_fold_indices(
                    config.output_dir, study_name, ablation_config.variant, fold_idx, train_idx, val_idx
                )
                output_results_file = config.output_dir / "train_results" / f"{study_name}_{ablation_config.variant}_fold{fold_idx}_results.json"
                output_results_file.parent.mkdir(parents=True, exist_ok=True)

                print(f"\n  [{study_name}] {ablation_config.variant} (fold {fold_idx + 1}/{config.n_folds})")
                print(f"    Train: {len(train_idx)}, Val: {len(val_idx)}")
                print(f"    Running train.py subprocess...")

                train_results = run_train_subprocess(
                    study=study_name,
                    variant=ablation_config.variant,
                    fold_idx=fold_idx,
                    fold_indices_file=fold_indices_file,
                    output_results_file=output_results_file,
                    ablation_config=ablation_config,
                    epochs=config.training.epochs,
                    batch_size=config.training.batch_size,
                    use_fsdp=use_fsdp,
                    fsdp_strategy=fsdp_strategy,
                    seed=config.cv_seed,
                    verbose=True,
                )

                if train_results is not None:
                    result = AblationResult(
                        study=study_name,
                        variant=ablation_config.variant,
                        fold=fold_idx,
                        best_r2=train_results.get("best_val_r2", 0.0),
                        best_loss=train_results.get("best_val_loss", 0.0),
                        train_losses=train_results.get("train_losses", []),
                        val_losses=train_results.get("val_losses", []),
                        val_r2s=train_results.get("val_r2s", []),
                        n_parameters=train_results.get("n_parameters", 0),
                        epochs_trained=train_results.get("epochs_trained", 0),
                        total_time=train_results.get("total_time", 0.0),
                        config=ablation_config.to_dict(),
                    )
                    print(f"    Best R²: {result.best_r2:.4f}")
                else:
                    # Failed - create placeholder result
                    result = AblationResult(
                        study=study_name,
                        variant=ablation_config.variant,
                        fold=fold_idx,
                        best_r2=0.0,
                        best_loss=float('inf'),
                        train_losses=[],
                        val_losses=[],
                        val_r2s=[],
                        n_parameters=0,
                        epochs_trained=0,
                        total_time=0.0,
                        config=ablation_config.to_dict(),
                    )
                    print(f"    FAILED")
            else:
                # Use internal trainer
                train_loader, val_loader = create_dataloaders_from_indices(
                    X, y, odor_ids, train_idx, val_idx,
                    batch_size=config.training.batch_size,
                    n_workers=config.n_workers,
                )

                result = run_single_ablation(
                    ablation_config=ablation_config,
                    training_config=config.training,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    fold=fold_idx,
                    cv_seed=config.cv_seed,
                    device=config.device,
                    checkpoint_dir=config.checkpoint_dir / ablation_config.study / f"fold{fold_idx}",
                    verbose=config.verbose,
                )
            all_results.append(result)

        # Save checkpoint after each study
        completed_studies.add(study_name)
        save_checkpoint(checkpoint_path, all_results, baseline_results, completed_studies, baseline_completed)
        print(f"Checkpoint saved ({len(completed_studies)}/{len(ablation_configs)} studies completed)")

    if skipped_studies > 0:
        print(f"\n*** Resumed run: skipped {skipped_studies} already-completed studies ***")

    # Aggregate results
    aggregated = aggregate_results(all_results)

    # Determine optimal configuration
    optimal_config = determine_optimal_config(aggregated)

    # Compute comprehensive statistics
    comprehensive_stats = {}
    baseline_fold_r2s = [r.best_r2 for r in baseline_results]

    if len(baseline_fold_r2s) >= 2:
        print("\nComputing statistical analysis...")

        for study, variants in aggregated.items():
            if study == "baseline":
                continue

            for variant, stats in variants.items():
                fold_r2s = stats.get("fold_r2s", [])

                if len(fold_r2s) >= 2:
                    comp = compare_methods(
                        np.array(baseline_fold_r2s),
                        np.array(fold_r2s),
                        name_a="baseline",
                        name_b=f"{study}:{variant}",
                        paired=True,
                        alpha=0.05
                    )
                    comprehensive_stats[f"{study}:{variant}"] = comp.to_dict()

        # Apply multiple comparison corrections
        if comprehensive_stats:
            p_values = [comprehensive_stats[k]["parametric_test"]["p_value"]
                       for k in comprehensive_stats]
            significant_holm = holm_correction(p_values, alpha=0.05)
            significant_fdr, adjusted_p = fdr_correction(p_values, alpha=0.05)

            for i, key in enumerate(comprehensive_stats.keys()):
                comprehensive_stats[key]["significant_holm"] = significant_holm[i]
                comprehensive_stats[key]["significant_fdr"] = significant_fdr[i]
                comprehensive_stats[key]["p_fdr_adjusted"] = adjusted_p[i]

    # Create final result
    phase3_result = Phase3Result(
        results=all_results,
        aggregated=aggregated,
        baseline_r2=baseline_r2,
        optimal_config=optimal_config,
        config=config.to_dict(),
        comprehensive_stats=comprehensive_stats,
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
                "fold_r2s": r2_values,  # Store for statistical analysis
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

    # Print comprehensive statistics if available
    if result.comprehensive_stats:
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS (vs baseline)")
        print("=" * 60)
        print("{:<25}{:<12}{:<12}{:<8}{:<8}".format("Ablation", "t-test p", "Cohen's d", "Holm", "FDR"))
        print("-" * 60)
        for key, stats in result.comprehensive_stats.items():
            t_p = stats["parametric_test"]["p_value"]
            d = stats["parametric_test"]["effect_size"] or 0
            holm_sig = "Yes" if stats.get("significant_holm", False) else "No"
            fdr_sig = "Yes" if stats.get("significant_fdr", False) else "No"
            print(f"{key:<25}{t_p:<12.4f}{d:<12.3f}{holm_sig:<8}{fdr_sig:<8}")

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

    # Ablation protocol
    parser.add_argument(
        "--protocol",
        type=str,
        default="additive",
        choices=["additive", "subtractive"],
        help="Ablation protocol: 'additive' (build-up, recommended) or 'subtractive' (traditional)",
    )

    # Studies to run (for subtractive protocol)
    parser.add_argument(
        "--studies",
        type=str,
        nargs="+",
        default=None,
        help="Ablation studies to run for subtractive protocol (default: all)",
    )

    # Stages to run (for additive protocol)
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=None,
        help="Stages to run for additive protocol (0-6, default: all)",
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
        "--fresh",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only attention ablation for quick testing",
    )

    # train.py integration (for FSDP multi-GPU support)
    parser.add_argument(
        "--use-train-py",
        action="store_true",
        help="Use train.py via subprocess for EXACT same training loop. "
             "This ensures Phase 3 results are perfectly comparable to train.py. "
             "When enabled, --fsdp is passed to train.py subprocess.",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Enable FSDP distributed training (requires --use-train-py)",
    )
    parser.add_argument(
        "--fsdp-strategy",
        type=str,
        default="grad_op",
        choices=["full", "grad_op", "no_shard", "hybrid"],
        help="FSDP sharding strategy (default: grad_op)",
    )

    args = parser.parse_args()

    # Validate FSDP requires --use-train-py
    if args.fsdp and not args.use_train_py:
        print("Note: --fsdp requires --use-train-py, enabling it automatically")
        args.use_train_py = True

    # When using --use-train-py, runner runs as single process
    if args.use_train_py:
        if args.fsdp:
            print("Note: --use-train-py mode passes FSDP to train.py subprocess")
            print("      Runner itself runs as single process")

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

    # Handle protocol-specific options
    if args.protocol == "additive":
        if args.fast:
            stages = [0, 1]  # Just baseline and first addition for quick test
        else:
            stages = args.stages if args.stages else list(range(len(INCREMENTAL_STAGES)))
        studies = list(ABLATION_STUDIES.keys())  # Not used in additive, but required
    else:
        stages = list(range(len(INCREMENTAL_STAGES)))  # Not used in subtractive
        studies = ["attention"] if args.fast else (args.studies or list(ABLATION_STUDIES.keys()))

    config = Phase3Config(
        dataset=args.dataset,
        protocol=args.protocol,
        studies=studies,
        stages=stages,
        n_folds=2 if args.dry_run else args.n_folds,
        cv_seed=args.cv_seed,
        training=training_config,
        output_dir=Path(args.output_dir),
        device=args.device,
        n_workers=args.workers,
    )

    # Print protocol summary
    print_protocol_summary(args.protocol)

    # Handle --fresh flag: delete existing checkpoint
    if args.fresh:
        checkpoint_path = get_checkpoint_path(config.output_dir)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("Deleted existing checkpoint (--fresh mode)")

    # Run ablations
    result = run_phase3(
        config,
        use_train_py=args.use_train_py,
        use_fsdp=args.fsdp,
        fsdp_strategy=args.fsdp_strategy,
    )

    # Clean up checkpoint after successful completion
    checkpoint_path = get_checkpoint_path(config.output_dir)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint cleaned up after successful completion")

    # Generate figures
    from .visualization import Phase3Visualizer

    viz = Phase3Visualizer(output_dir=config.output_dir, dpi=args.figure_dpi)
    figures = viz.plot_all(result, format=args.figure_format)
    print(f"\nGenerated {len(figures)} figures in {config.output_dir}")


if __name__ == "__main__":
    main()
