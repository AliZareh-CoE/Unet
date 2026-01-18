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
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress PyTorch C++ warnings BEFORE importing torch
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("TORCH_LOGS", "-all")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "0")
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "0")

import numpy as np
import torch

# Disable PyTorch's internal warnings
torch.set_warn_always(False)

# Suppress harmless warnings
warnings.filterwarnings("ignore", message=".*found in sys.modules after import.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*FSDP.state_dict_type.*")
warnings.filterwarnings("ignore", message=".*Deallocating Tensor.*")

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
from phase_two.trainer import (
    Trainer,
    TrainingResult,
    create_dataloaders,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    is_distributed,
    FSDP_AVAILABLE,
)

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
# train.py Subprocess Integration
# =============================================================================

def run_train_py(
    arch_name: str,
    fold_idx: int,
    fold_indices_file: Path,
    output_results_file: Path,
    epochs: int = 80,
    batch_size: int = 64,
    use_fsdp: bool = False,
    fsdp_strategy: str = "grad_op",
    seed: int = 42,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Run train.py as subprocess for a single architecture/fold.

    This ensures Phase 2 uses the EXACT same training loop as train.py,
    including all augmentations, loss functions, and hyperparameters.

    Args:
        arch_name: Architecture name (e.g., 'wavenet', 'vit')
        fold_idx: CV fold index (0-indexed)
        fold_indices_file: Path to pickle file with train/val indices
        output_results_file: Path to JSON file for results output
        epochs: Number of training epochs
        batch_size: Batch size (total, will be divided by num GPUs)
        use_fsdp: Whether to use FSDP distributed training
        fsdp_strategy: FSDP sharding strategy
        seed: Random seed
        verbose: Print subprocess output

    Returns:
        Results dict from train.py, or None on failure
    """
    import subprocess
    import shutil

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

    # Add arguments
    cmd.extend([
        "--arch", arch_name,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seed", str(seed + fold_idx),  # Different seed per fold
        "--fold-indices-file", str(fold_indices_file),
        "--output-results-file", str(output_results_file),
        "--fold", str(fold_idx),
        "--no-bidirectional",  # Disable for fair comparison
        "--no-plots",  # Skip plot generation for speed
    ])

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
        print(f"ERROR: train.py failed for {arch_name} fold {fold_idx}")
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


def save_fold_indices(
    output_dir: Path,
    arch_name: str,
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Path:
    """Save fold indices to pickle file for train.py.

    Args:
        output_dir: Directory to save indices
        arch_name: Architecture name
        fold_idx: Fold index
        train_idx: Training sample indices
        val_idx: Validation sample indices

    Returns:
        Path to saved pickle file
    """
    fold_dir = output_dir / "fold_indices"
    fold_dir.mkdir(parents=True, exist_ok=True)

    indices_file = fold_dir / f"{arch_name}_fold{fold_idx}_indices.pkl"
    with open(indices_file, 'wb') as f:
        pickle.dump({
            "train_idx": train_idx,
            "val_idx": val_idx,
        }, f)

    return indices_file


# =============================================================================
# Phase 1 Result Loading
# =============================================================================

def load_phase1_classical_r2(phase1_dir: Path = None, verbose: bool = True) -> Optional[float]:
    """Auto-load best classical R² from Phase 1 results.

    Searches for Phase 1 result files and extracts the best R².

    Args:
        phase1_dir: Directory containing Phase 1 results.
                    Defaults to results/phase1/
        verbose: Whether to print loading messages (set False for non-main ranks)

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
            if verbose:
                print(f"[Auto-loaded] Phase 1 best R²: {best_r2:.4f} ({best_method})")
                print(f"  Source: {result_file}")
            return float(best_r2)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not load Phase 1 results from {result_file}: {e}")

    return None


# =============================================================================
# Checkpoint/Resume Support
# =============================================================================

def get_checkpoint_path(output_dir: Path) -> Path:
    """Get the checkpoint file path for incremental saving."""
    return output_dir / "phase2_checkpoint.pkl"


def save_checkpoint(
    checkpoint_path: Path,
    all_results: List,
    arch_r2s: Dict[str, List[float]],
    all_models: Dict[str, Any],
    completed_runs: set,
) -> None:
    """Save checkpoint after each fold for resume capability.

    Args:
        checkpoint_path: Path to save checkpoint
        all_results: List of TrainingResult objects
        arch_r2s: Dict mapping arch -> list of R2 values
        all_models: Dict mapping arch -> best model info
        completed_runs: Set of (arch_name, fold_idx) tuples that are done
    """
    checkpoint = {
        "all_results": [r.to_dict() for r in all_results],
        "arch_r2s": arch_r2s,
        "all_models": all_models,
        "completed_runs": list(completed_runs),
        "timestamp": datetime.now().isoformat(),
    }

    # Save atomically (write to temp file, then rename)
    temp_path = checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    temp_path.rename(checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Load checkpoint if it exists.

    Returns:
        Checkpoint dict or None if no checkpoint exists
    """
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"\n*** RESUMING FROM CHECKPOINT ***")
        print(f"    Found {len(checkpoint['completed_runs'])} completed runs")
        # Handle checkpoints created by create_checkpoint_from_results.py (no timestamp)
        if 'timestamp' in checkpoint:
            print(f"    Checkpoint from: {checkpoint['timestamp']}")
        else:
            print(f"    Checkpoint reconstructed from existing results")
        return checkpoint
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None


def reconstruct_results_from_checkpoint(
    checkpoint: Dict[str, Any]
) -> Tuple[List, Dict[str, List[float]], Dict[str, Any], set]:
    """Reconstruct state from checkpoint.

    Returns:
        Tuple of (all_results, arch_r2s, all_models, completed_runs)
    """
    from phase_two.trainer import TrainingResult

    # Reconstruct TrainingResult objects
    all_results = []
    for r_item in checkpoint["all_results"]:
        # Handle both TrainingResult objects and dicts
        if isinstance(r_item, TrainingResult):
            # Already a TrainingResult (from create_checkpoint_from_results.py)
            all_results.append(r_item)
        else:
            # Dict format (from original checkpoint saving)
            r_dict = r_item
            result = TrainingResult(
                architecture=r_dict["architecture"],
                fold=r_dict["fold"],
                best_val_r2=r_dict["best_val_r2"],
                best_val_mae=r_dict["best_val_mae"],
                best_epoch=r_dict["best_epoch"],
                train_losses=r_dict["train_losses"],
                val_losses=r_dict["val_losses"],
                val_r2s=r_dict["val_r2s"],
                total_time=r_dict["total_time"],
                epochs_trained=r_dict["epochs_trained"],
                n_parameters=r_dict["n_parameters"],
                nan_detected=r_dict.get("nan_detected", False),
                nan_recovery_count=r_dict.get("nan_recovery_count", 0),
                early_stopped=r_dict.get("early_stopped", False),
                error_message=r_dict.get("error_message"),
                peak_gpu_memory_mb=r_dict.get("peak_gpu_memory_mb"),
                completed_successfully=r_dict.get("completed_successfully", True),
                dsp_metrics=r_dict.get("dsp_metrics"),
            )
            all_results.append(result)

    arch_r2s = checkpoint["arch_r2s"]
    all_models = checkpoint["all_models"]
    completed_runs = set(tuple(x) for x in checkpoint["completed_runs"])

    return all_results, arch_r2s, all_models, completed_runs


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

def load_olfactory_data(verbose: bool = True):
    """Load olfactory dataset (cross-subject session-based split).

    Uses session-based split where entire sessions are held out for
    validation and test, ensuring true cross-subject generalization.

    Args:
        verbose: Whether to print loading messages (set False for non-main ranks)

    Returns:
        X: Input signals [N, C, T]
        y: Target signals [N, C, T]
    """
    from data import prepare_data

    if verbose:
        print("Loading olfactory dataset (cross-subject: 3 held-out sessions, no test set)...")
    data = prepare_data(
        split_by_session=True,
        n_test_sessions=0,   # No separate test set (matches Phase 3)
        n_val_sessions=3,    # 3 sessions held out for validation (matches Phase 3)
        no_test_set=True,    # All held-out sessions for validation
    )

    ob = data["ob"]    # [N, C, T]
    pcx = data["pcx"]  # [N, C, T]

    # Combine train+val for CV (exclude test set for final evaluation)
    # Note: With session-based split, train/val sessions are already separate
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]

    # Log session split info if available
    if verbose and "split_info" in data:
        split_info = data["split_info"]
        print(f"  Train sessions: {split_info.get('train_sessions', 'N/A')}")
        print(f"  Val sessions: {split_info.get('val_sessions', 'N/A')}")
        print(f"  Test sessions: {split_info.get('test_sessions', 'N/A')}")

    all_idx = np.concatenate([train_idx, val_idx])
    X = ob[all_idx]
    y = pcx[all_idx]

    if verbose:
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
    use_fsdp: bool = False,
    fsdp_strategy: str = "grad_op",
    fsdp_cpu_offload: bool = False,
    use_train_py: bool = False,
) -> Phase2Result:
    """Run Phase 2 architecture screening with 5-fold cross-validation.

    Args:
        config: Phase 2 configuration
        X: Input signals [N, C, T]
        y: Target signals [N, C, T]
        classical_r2: Best R² from Phase 1 (for gate check)
        use_fsdp: Enable FSDP for distributed training
        fsdp_strategy: FSDP sharding strategy ('full', 'grad_op', 'no_shard', 'hybrid')
        fsdp_cpu_offload: Enable CPU offloading for FSDP
        use_train_py: Use train.py subprocess for exact training consistency

    Returns:
        Phase2Result with all metrics

    FSDP Usage:
        torchrun --nproc_per_node=8 -m phase_two.runner --fsdp --epochs 80 --batch-size 64

    train.py Integration:
        python -m phase_two.runner --use-train-py --fsdp --epochs 80
    """
    # Only print on main process
    _is_main = is_main_process()

    if _is_main:
        print("\n" + "=" * 70)
        print("PHASE 2: Architecture Screening (5-Fold Cross-Validation)")
        print("=" * 70)
        print(f"Dataset: {config.dataset}")
        print(f"Architectures: {', '.join(config.architectures)}")
        print(f"Cross-validation: {config.n_folds} folds")
        print(f"Total runs: {config.total_runs}")
        print(f"Device: {config.device}")
        if use_train_py:
            print(f"Training mode: train.py subprocess (exact consistency)")
            if use_fsdp:
                print(f"  FSDP passed to train.py: strategy={fsdp_strategy}")
        elif use_fsdp:
            print(f"FSDP: Enabled (strategy={fsdp_strategy}, cpu_offload={fsdp_cpu_offload})")
        print()

    n_samples = X.shape[0]
    in_channels = X.shape[1]
    out_channels = y.shape[1]
    time_steps = X.shape[2]

    # Create CV splits
    cv_splits = create_cv_splits(n_samples, config.n_folds, config.cv_seed)
    if _is_main:
        print(f"CV splits created: {[len(s[1]) for s in cv_splits]} validation samples per fold")

    # Setup checkpoint for resume capability (only main process)
    checkpoint_path = get_checkpoint_path(config.output_dir)
    if _is_main:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load checkpoint for resume (only main process loads, then broadcasts state)
    checkpoint = None
    completed_runs = set()

    if _is_main:
        checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint is not None:
        all_results, arch_r2s, all_models, completed_runs = reconstruct_results_from_checkpoint(checkpoint)
        # Ensure all architectures have entries in arch_r2s
        for arch in config.architectures:
            if arch not in arch_r2s:
                arch_r2s[arch] = []
    else:
        all_results = []
        arch_r2s = {arch: [] for arch in config.architectures}
        all_models = {}  # Store best model for each architecture
        completed_runs = set()

    # Broadcast completed_runs to all ranks so they all skip the same runs
    if use_fsdp and is_distributed():
        import torch.distributed as dist
        # Convert to list for broadcasting
        completed_runs_list = [list(completed_runs)]
        dist.broadcast_object_list(completed_runs_list, src=0)
        completed_runs = set(tuple(x) for x in completed_runs_list[0])

    run_idx = 0
    total_runs = config.total_runs
    skipped_runs = 0

    for arch_name in config.architectures:
        if _is_main:
            print(f"\n{'='*60}")
            print(f"Architecture: {arch_name.upper()}")
            print("=" * 60)

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            run_idx += 1

            # Check if this run was already completed
            if (arch_name, fold_idx) in completed_runs:
                skipped_runs += 1
                if _is_main:
                    print(f"\n[{run_idx}/{total_runs}] {arch_name} (fold {fold_idx + 1}/{config.n_folds}) - SKIPPED (already completed)")
                continue

            if _is_main:
                print(f"\n[{run_idx}/{total_runs}] {arch_name} (fold {fold_idx + 1}/{config.n_folds})")

            # ================================================================
            # Training: use train.py subprocess OR internal trainer
            # ================================================================
            if use_train_py:
                # Use train.py subprocess for EXACT training consistency
                # This ensures Phase 2 uses the same training loop as train.py

                # Save fold indices for train.py
                fold_indices_file = save_fold_indices(
                    config.output_dir, arch_name, fold_idx, train_idx, val_idx
                )

                # Output file for results
                results_dir = config.output_dir / "train_results"
                results_dir.mkdir(parents=True, exist_ok=True)
                output_results_file = results_dir / f"{arch_name}_fold{fold_idx}_results.json"

                if _is_main:
                    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
                    print(f"  Running train.py subprocess...")

                # Run train.py
                train_results = run_train_py(
                    arch_name=arch_name,
                    fold_idx=fold_idx,
                    fold_indices_file=fold_indices_file,
                    output_results_file=output_results_file,
                    epochs=config.training.epochs,
                    batch_size=config.training.batch_size,
                    use_fsdp=use_fsdp,
                    fsdp_strategy=fsdp_strategy,
                    seed=config.cv_seed,
                    verbose=config.verbose > 0,
                )

                if train_results is None:
                    # Training failed
                    result = TrainingResult(
                        architecture=arch_name,
                        fold=fold_idx,
                        best_val_r2=float('nan'),
                        best_val_mae=float('nan'),
                        best_epoch=0,
                        train_losses=[],
                        val_losses=[],
                        val_r2s=[],
                        total_time=0.0,
                        epochs_trained=0,
                        n_parameters=0,
                        completed_successfully=False,
                        error_message="train.py subprocess failed",
                    )
                else:
                    # Convert train.py results to TrainingResult
                    result = TrainingResult(
                        architecture=train_results.get("architecture", arch_name),
                        fold=train_results.get("fold", fold_idx),
                        best_val_r2=train_results.get("best_val_r2", 0.0),
                        best_val_mae=train_results.get("best_val_mae", 0.0),
                        best_epoch=train_results.get("best_epoch", 0),
                        train_losses=train_results.get("train_losses", []),
                        val_losses=train_results.get("val_losses", []),
                        val_r2s=train_results.get("val_r2s", []),
                        total_time=train_results.get("total_time", 0.0),
                        epochs_trained=train_results.get("epochs_trained", 0),
                        n_parameters=train_results.get("n_parameters", 0),
                        completed_successfully=train_results.get("completed_successfully", True),
                    )

                n_params = result.n_parameters
                all_results.append(result)
                arch_r2s[arch_name].append(result.best_val_r2)

                # Note: model state not saved in train.py mode (models saved separately by train.py)
                if _is_main:
                    if arch_name not in all_models or result.best_val_r2 > all_models.get(arch_name, {}).get('r2', float('-inf')):
                        all_models[arch_name] = {
                            'r2': result.best_val_r2,
                            'fold': fold_idx,
                            'n_parameters': n_params,
                            'checkpoint': str(PROJECT_ROOT / "artifacts" / "checkpoints" / "best_model.pt"),
                        }

                if _is_main:
                    print(f"  Best R²: {result.best_val_r2:.4f} (epoch {result.best_epoch})")
                    print(f"  Time: {result.total_time:.1f}s")
                    if not result.completed_successfully:
                        print(f"  Error: {result.error_message}")

            else:
                # Use internal trainer (original code path)

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
                if _is_main:
                    print(f"  Parameters: {n_params:,}")
                    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
                    if use_fsdp:
                        print(f"  FSDP: Enabled with strategy={fsdp_strategy}")

                # Create data loaders (with distributed sampler if using FSDP)
                train_loader, val_loader = create_dataloaders(
                    X_train, y_train, X_val, y_val,
                    batch_size=config.training.batch_size,
                    n_workers=config.n_workers,
                    use_distributed=use_fsdp,
                )

                # Create trainer with logging (FSDP-aware)
                trainer = Trainer(
                    model=model,
                    config=config.training,
                    device=config.device,
                    checkpoint_dir=config.checkpoint_dir / arch_name / f"fold{fold_idx}" if config.save_checkpoints else None,
                    log_dir=config.log_dir if hasattr(config, 'log_dir') else None,
                    use_fsdp=use_fsdp,
                    fsdp_strategy=fsdp_strategy,
                    fsdp_cpu_offload=fsdp_cpu_offload,
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
                # For FSDP, trainer._best_model_state already has the full state dict on rank 0
                if _is_main:
                    if arch_name not in all_models or result.best_val_r2 > all_models[arch_name]['r2']:
                        if hasattr(trainer, '_best_model_state') and trainer._best_model_state is not None:
                            all_models[arch_name] = {
                                'state_dict': {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in trainer._best_model_state.items()},
                                'r2': result.best_val_r2,
                                'fold': fold_idx,
                                'n_parameters': n_params,
                            }

                if _is_main:
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

            # Mark this run as completed and save checkpoint (only main process)
            completed_runs.add((arch_name, fold_idx))
            if _is_main:
                save_checkpoint(checkpoint_path, all_results, arch_r2s, all_models, completed_runs)
                print(f"  Checkpoint saved ({len(completed_runs)}/{total_runs} runs completed)")

    # Print resume summary if any runs were skipped
    if skipped_runs > 0 and _is_main:
        print(f"\n*** Resumed run: skipped {skipped_runs} already-completed runs ***")

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
        if _is_main:
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

    # Print summary (only on main process)
    if _is_main:
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
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint (default behavior, this flag confirms intent)")
    parser.add_argument("--fresh", action="store_true",
                       help="Start fresh, ignoring any existing checkpoint")
    parser.add_argument("--figure-format", type=str, default="pdf",
                       choices=["pdf", "png", "svg", "eps"],
                       help="Output format for figures")
    parser.add_argument("--figure-dpi", type=int, default=300,
                       help="DPI for raster figures (png)")

    # FSDP arguments
    parser.add_argument("--fsdp", action="store_true",
                       help="Enable FSDP for distributed multi-GPU training")
    parser.add_argument("--fsdp-strategy", type=str, default="grad_op",
                       choices=["full", "grad_op", "no_shard", "hybrid"],
                       help="FSDP sharding strategy (default: grad_op)")
    parser.add_argument("--fsdp-cpu-offload", action="store_true",
                       help="Enable CPU offloading for FSDP (saves GPU memory)")

    # train.py integration (for exact training consistency)
    parser.add_argument("--use-train-py", action="store_true",
                       help="Use train.py via subprocess for EXACT same training loop. "
                            "This ensures Phase 2 results are perfectly comparable to train.py. "
                            "When enabled, --fsdp is passed to train.py subprocess.")

    args = parser.parse_args()

    # When using --use-train-py, the runner runs as single process
    # and calls train.py subprocess with FSDP if requested
    if args.use_train_py:
        if args.fsdp:
            print("Note: --use-train-py mode passes FSDP to train.py subprocess")
            print("      Runner itself runs as single process")
        rank, world_size = 0, 1
    elif args.fsdp:
        # Direct FSDP mode (runner handles FSDP internally)
        if not FSDP_AVAILABLE:
            print("Error: FSDP not available. Please install PyTorch >= 1.12")
            sys.exit(1)
        rank, world_size = setup_distributed()
        if is_main_process():
            print(f"FSDP: Initialized {world_size} processes (rank={rank})")
    else:
        rank, world_size = 0, 1

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

    # Determine if this is the main process for printing
    _is_main = is_main_process()

    # Auto-load classical R² from Phase 1 if not specified
    classical_r2 = args.classical_r2
    if classical_r2 is None:
        phase1_dir = Path(args.phase1_dir) if args.phase1_dir else None
        classical_r2 = load_phase1_classical_r2(phase1_dir, verbose=_is_main)

        if classical_r2 is None:
            classical_r2 = 0.35  # Fallback default
            if _is_main:
                print(f"Warning: Could not auto-load Phase 1 results. Using default R²={classical_r2}")
                print("  Run Phase 1 first, or specify --classical-r2 manually")
    else:
        if _is_main:
            print(f"Using specified classical R²: {classical_r2:.4f}")

    # Load data (full dataset for cross-validation)
    if args.dry_run:
        if _is_main:
            print("[DRY-RUN] Using synthetic data")
        X, y = create_synthetic_data(
            n_samples=100, time_steps=500
        )
    else:
        try:
            X, y = load_olfactory_data(verbose=_is_main)
        except Exception as e:
            if _is_main:
                print(f"Warning: Could not load data ({e}). Using synthetic.")
            X, y = create_synthetic_data()

    # Handle --fresh flag: delete existing checkpoint (only main process)
    if args.fresh and _is_main:
        checkpoint_path = get_checkpoint_path(config.output_dir)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("Deleted existing checkpoint (--fresh mode)")

    # Handle --resume flag: check if checkpoint exists
    if args.resume and _is_main:
        checkpoint_path = get_checkpoint_path(config.output_dir)
        if checkpoint_path.exists():
            print(f"Resume mode: Found checkpoint at {checkpoint_path}")
        else:
            print("Resume mode: No checkpoint found, starting fresh")

    try:
        # Run with 5-fold cross-validation
        result = run_phase2(
            config=config,
            X=X,
            y=y,
            classical_r2=classical_r2,
            use_fsdp=args.fsdp,
            fsdp_strategy=args.fsdp_strategy,
            fsdp_cpu_offload=args.fsdp_cpu_offload,
            use_train_py=args.use_train_py,
        )

        # Only main process handles file I/O
        if _is_main:
            # Clean up checkpoint after successful completion
            checkpoint_path = get_checkpoint_path(config.output_dir)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print("Checkpoint cleaned up after successful completion")

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

    finally:
        # Clean up distributed training
        if args.fsdp:
            cleanup_distributed()

    return result


if __name__ == "__main__":
    main()
