"""LOSO (Leave-One-Subject-Out) Cross-Validation Runner.

Implements Leave-One-Session-Out cross-validation for neural signal
translation models. Systematically holds out each recording session
for testing while training on all others.

Usage:
    # Run LOSO cross-validation with default settings
    python -m LOSO.runner

    # Specify dataset and output directory
    python -m LOSO.runner --dataset olfactory --output-dir artifacts/loso_results

    # Resume from checkpoint
    python -m LOSO.runner --resume

    # Run specific folds only
    python -m LOSO.runner --folds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from LOSO.config import LOSOConfig, LOSOFoldResult, LOSOResult


# =============================================================================
# Logging Setup
# =============================================================================

class TeeLogger:
    """Duplicate stdout to both console and log file."""

    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = log_file
        self.log = open(log_file, 'w', buffering=1)  # Line buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_logging(output_dir: Path) -> Tuple[Path, TeeLogger]:
    """Setup logging to capture all output to a log file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"loso_run_{timestamp}.log"

    tee = TeeLogger(log_path)
    sys.stdout = tee

    print(f"Logging to: {log_path}")
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    return log_path, tee


# =============================================================================
# Session Discovery
# =============================================================================

def get_all_sessions(dataset: str) -> List[str]:
    """Get list of all session names for a dataset.

    Args:
        dataset: Dataset name (olfactory, pfc_hpc, dandi_movie)

    Returns:
        List of session names/identifiers
    """
    if dataset == "olfactory":
        from data import load_session_ids, ODOR_CSV_PATH
        _, session_to_idx, _ = load_session_ids(ODOR_CSV_PATH)
        return sorted(session_to_idx.keys())

    elif dataset == "pfc_hpc":
        from data import load_pfc_session_ids, PFC_META_PATH
        _, session_to_idx, _ = load_pfc_session_ids(PFC_META_PATH)
        return sorted(session_to_idx.keys())

    elif dataset == "dandi_movie":
        # DANDI dataset - get sessions from data directory
        from data import _DANDI_DATA_DIR
        if _DANDI_DATA_DIR.exists():
            sessions = []
            for session_dir in sorted(_DANDI_DATA_DIR.iterdir()):
                if session_dir.is_dir() and session_dir.name.startswith("sub-"):
                    sessions.append(session_dir.name)
            return sessions
        raise FileNotFoundError(f"DANDI data directory not found: {_DANDI_DATA_DIR}")

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# =============================================================================
# Checkpoint Management
# =============================================================================

def get_checkpoint_path(output_dir: Path) -> Path:
    """Get the checkpoint file path."""
    return output_dir / "loso_checkpoint.pkl"


def save_checkpoint(
    checkpoint_path: Path,
    config: LOSOConfig,
    fold_results: List[LOSOFoldResult],
    completed_folds: List[int],
    all_sessions: List[str],
) -> None:
    """Save checkpoint after each fold for resume capability."""
    checkpoint = {
        "config": config.to_dict(),
        "fold_results": [r.to_dict() for r in fold_results],
        "completed_folds": completed_folds,
        "all_sessions": all_sessions,
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
        print(f"    Completed folds: {len(checkpoint['completed_folds'])}")
        print(f"    Checkpoint from: {checkpoint['timestamp']}")
        return checkpoint
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None


def reconstruct_from_checkpoint(checkpoint: Dict[str, Any]) -> Tuple[List[LOSOFoldResult], List[int], List[str]]:
    """Reconstruct state from checkpoint."""
    fold_results = []
    for r_dict in checkpoint["fold_results"]:
        result = LOSOFoldResult(
            fold_idx=r_dict["fold_idx"],
            test_session=r_dict["test_session"],
            train_sessions=r_dict["train_sessions"],
            val_r2=r_dict["val_r2"],
            val_loss=r_dict["val_loss"],
            train_loss=r_dict.get("train_loss", 0.0),
            per_session_r2=r_dict.get("per_session_r2", {}),
            per_session_loss=r_dict.get("per_session_loss", {}),
            epochs_trained=r_dict.get("epochs_trained", 0),
            total_time=r_dict.get("total_time", 0.0),
            config=r_dict.get("config", {}),
        )
        fold_results.append(result)

    completed_folds = checkpoint["completed_folds"]
    all_sessions = checkpoint["all_sessions"]

    return fold_results, completed_folds, all_sessions


# =============================================================================
# Single Fold Training
# =============================================================================

def run_single_fold(
    fold_idx: int,
    test_session: str,
    all_sessions: List[str],
    config: LOSOConfig,
    output_results_file: Path,
) -> Optional[LOSOFoldResult]:
    """Run a single LOSO fold by calling train.py.

    Args:
        fold_idx: Fold index (0-indexed)
        test_session: Session name to hold out for testing
        all_sessions: List of all session names
        config: LOSO configuration
        output_results_file: Path to save train.py results

    Returns:
        LOSOFoldResult or None on failure
    """
    train_sessions = [s for s in all_sessions if s != test_session]

    print(f"\n{'='*70}")
    print(f"LOSO FOLD {fold_idx + 1}/{len(all_sessions)}")
    print(f"{'='*70}")
    print(f"Test session:  {test_session}")
    print(f"Train sessions: {train_sessions}")
    print()

    # Build train.py command
    if config.use_fsdp:
        nproc = torch.cuda.device_count() if torch.cuda.is_available() else 1
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            str(PROJECT_ROOT / "train.py"),
        ]
    else:
        cmd = [sys.executable, str(PROJECT_ROOT / "train.py")]

    # Base arguments
    cmd.extend([
        "--arch", config.arch,
        "--epochs", str(config.epochs),
        "--batch-size", str(config.batch_size),
        "--lr", str(config.learning_rate),
        "--seed", str(config.seed + fold_idx),  # Different seed per fold
        "--output-results-file", str(output_results_file),
        "--fold", str(fold_idx),
    ])

    # Skip plots for speed (unless enabled)
    if not config.generate_plots:
        cmd.append("--no-plots")

    # Dataset selection
    if config.dataset != "olfactory":
        cmd.extend(["--dataset", config.dataset])

    # Session-based splitting: hold out one session for validation (LOSO)
    cmd.append("--split-by-session")
    cmd.append("--force-recreate-splits")
    cmd.extend(["--val-sessions", test_session])  # Held-out session becomes validation
    cmd.append("--no-test-set")  # No separate test set needed

    # Model architecture arguments
    if config.base_channels:
        cmd.extend(["--base-channels", str(config.base_channels)])
    if config.n_downsample:
        cmd.extend(["--n-downsample", str(config.n_downsample)])
    if config.attention_type:
        cmd.extend(["--attention-type", config.attention_type])
    if config.cond_mode:
        cmd.extend(["--cond-mode", config.cond_mode])
    if config.conv_type:
        cmd.extend(["--conv-type", config.conv_type])
    if config.activation:
        cmd.extend(["--activation", config.activation])
    if config.skip_type:
        cmd.extend(["--skip-type", config.skip_type])
    if config.n_heads is not None:
        cmd.extend(["--n-heads", str(config.n_heads)])
    # Only pass --conditioning if it's not "none" (to disable conditioning, omit the arg)
    if config.conditioning and config.conditioning.lower() != "none":
        cmd.extend(["--conditioning", config.conditioning])

    # Training arguments
    if config.optimizer:
        cmd.extend(["--optimizer", config.optimizer])
    if config.lr_schedule:
        cmd.extend(["--lr-schedule", config.lr_schedule])
    if config.weight_decay > 0:
        cmd.extend(["--weight-decay", str(config.weight_decay)])
    if config.dropout > 0:
        cmd.extend(["--dropout", str(config.dropout)])

    if not config.use_bidirectional:
        cmd.append("--no-bidirectional")

    # Session adaptation
    if config.use_session_stats:
        cmd.append("--use-session-stats")
    if config.session_use_spectral:
        cmd.append("--session-use-spectral")
    if config.use_adaptive_scaling:
        cmd.append("--use-adaptive-scaling")

    # FSDP
    if config.use_fsdp:
        cmd.extend(["--fsdp", "--fsdp-strategy", config.fsdp_strategy])

    # Run subprocess
    start_time = time.time()
    try:
        if config.verbose:
            # Stream output in real-time
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                print(line, end='', flush=True)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        else:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: train.py failed for fold {fold_idx} (test={test_session})")
        print(f"  Return code: {e.returncode}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"  stderr: {e.stderr[:1000]}...")
        return None

    elapsed = time.time() - start_time

    # Load results from JSON file
    if output_results_file.exists():
        with open(output_results_file, 'r') as f:
            results = json.load(f)

        fold_result = LOSOFoldResult(
            fold_idx=fold_idx,
            test_session=test_session,
            train_sessions=train_sessions,
            val_r2=results.get("best_val_r2", results.get("val_r2", 0.0)),
            val_loss=results.get("best_val_loss", results.get("val_loss", float('inf'))),
            train_loss=results.get("final_train_loss", 0.0),
            per_session_r2=results.get("per_session_r2", {}),
            per_session_loss=results.get("per_session_loss", {}),
            epochs_trained=results.get("epochs_trained", config.epochs),
            total_time=elapsed,
            config=config.to_dict(),
        )

        print(f"\nFold {fold_idx} completed:")
        print(f"  Test session: {test_session}")
        print(f"  Val R²: {fold_result.val_r2:.4f}")
        print(f"  Val Loss: {fold_result.val_loss:.4f}")
        print(f"  Time: {elapsed/60:.1f} minutes")

        return fold_result
    else:
        print(f"WARNING: Results file not found: {output_results_file}")
        return None


# =============================================================================
# Main LOSO Runner
# =============================================================================

def run_loso(
    config: LOSOConfig,
    folds_to_run: Optional[List[int]] = None,
) -> LOSOResult:
    """Run Leave-One-Session-Out cross-validation.

    Args:
        config: LOSO configuration
        folds_to_run: Optional list of specific fold indices to run.
                      If None, runs all folds.

    Returns:
        LOSOResult with aggregated results
    """
    # Setup output directory and logging
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path, tee = setup_logging(config.output_dir)

    print("=" * 70)
    print("LOSO (Leave-One-Session-Out) Cross-Validation")
    print("=" * 70)
    print(f"Dataset: {config.dataset}")
    print(f"Output directory: {config.output_dir}")
    print()

    # Get all sessions
    all_sessions = get_all_sessions(config.dataset)
    n_folds = len(all_sessions)
    print(f"Found {n_folds} sessions: {all_sessions}")
    print()

    # Check for checkpoint to resume
    checkpoint_path = get_checkpoint_path(config.output_dir)
    fold_results: List[LOSOFoldResult] = []
    completed_folds: List[int] = []

    if config.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint is not None:
            fold_results, completed_folds, _ = reconstruct_from_checkpoint(checkpoint)
            print(f"Resuming from {len(completed_folds)} completed folds")

    # Determine which folds to run
    if folds_to_run is not None:
        folds_remaining = [f for f in folds_to_run if f not in completed_folds]
    else:
        folds_remaining = [i for i in range(n_folds) if i not in completed_folds]

    print(f"Folds to run: {folds_remaining}")
    print()

    # Print configuration summary
    print("Configuration:")
    print("-" * 40)
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print()

    # Run each fold
    results_dir = config.output_dir / "fold_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx in folds_remaining:
        test_session = all_sessions[fold_idx]
        output_results_file = results_dir / f"fold_{fold_idx}_{test_session}_results.json"

        result = run_single_fold(
            fold_idx=fold_idx,
            test_session=test_session,
            all_sessions=all_sessions,
            config=config,
            output_results_file=output_results_file,
        )

        if result is not None:
            fold_results.append(result)
            completed_folds.append(fold_idx)

            # Save checkpoint after each fold
            save_checkpoint(
                checkpoint_path,
                config,
                fold_results,
                completed_folds,
                all_sessions,
            )

            print(f"\nCheckpoint saved after fold {fold_idx}")

    # Create final result
    loso_result = LOSOResult(
        config=config,
        fold_results=fold_results,
        all_sessions=all_sessions,
    )
    loso_result.compute_statistics()

    # Print summary
    loso_result.print_summary()

    # Save final results
    final_results_path = config.output_dir / "loso_results.json"
    with open(final_results_path, 'w') as f:
        json.dump(loso_result.to_dict(), f, indent=2, default=_json_serializer)
    print(f"\nFinal results saved to: {final_results_path}")

    # Close logger
    tee.close()
    sys.stdout = tee.terminal

    return loso_result


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
# CLI Interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LOSO (Leave-One-Session-Out) Cross-Validation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset and output
    parser.add_argument(
        "--dataset",
        type=str,
        default="olfactory",
        choices=["olfactory", "pfc_hpc", "dandi_movie"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/loso",
        help="Output directory for results",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model architecture
    parser.add_argument("--arch", type=str, default="condunet", help="Model architecture")
    parser.add_argument("--base-channels", type=int, default=128, help="Base channel count")
    parser.add_argument("--n-downsample", type=int, default=2, help="Downsample layers")
    parser.add_argument(
        "--attention-type",
        type=str,
        default="cross_freq_v2",
        choices=["none", "basic", "cross_freq_v2"],
        help="Attention type",
    )
    parser.add_argument(
        "--cond-mode",
        type=str,
        default="cross_attn_gated",
        help="Conditioning mode",
    )
    parser.add_argument(
        "--conv-type",
        type=str,
        default="modern",
        choices=["standard", "modern"],
        help="Convolution type",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function")
    parser.add_argument("--skip-type", type=str, default="add", choices=["add", "concat"], help="Skip connection type")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--conditioning", type=str, default="spectro_temporal", help="Auto-conditioning type (none, spectro_temporal, etc.)")

    # Training configuration
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer")
    parser.add_argument("--lr-schedule", type=str, default="cosine_warmup", help="LR schedule")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    # Session adaptation
    parser.add_argument("--use-session-stats", action="store_true", help="Use session statistics")
    parser.add_argument("--session-use-spectral", action="store_true", help="Include spectral features")
    parser.add_argument("--use-adaptive-scaling", action="store_true", help="Use adaptive scaling")

    # Training options
    parser.add_argument("--no-bidirectional", action="store_true", help="Disable bidirectional training")

    # FSDP
    parser.add_argument("--fsdp", action="store_true", help="Enable FSDP")
    parser.add_argument("--fsdp-strategy", type=str, default="grad_op", help="FSDP strategy")

    # Execution control
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore checkpoint)")
    parser.add_argument("--folds", type=int, nargs="+", help="Specific fold indices to run")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--generate-plots", action="store_true", help="Generate validation plots")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create config from args
    config = LOSOConfig(
        dataset=args.dataset,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        arch=args.arch,
        base_channels=args.base_channels,
        n_downsample=args.n_downsample,
        attention_type=args.attention_type,
        cond_mode=args.cond_mode,
        conv_type=args.conv_type,
        activation=args.activation,
        skip_type=args.skip_type,
        n_heads=args.n_heads,
        conditioning=args.conditioning,
        optimizer=args.optimizer,
        lr_schedule=args.lr_schedule,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        use_session_stats=args.use_session_stats,
        session_use_spectral=args.session_use_spectral,
        use_adaptive_scaling=args.use_adaptive_scaling,
        use_bidirectional=not args.no_bidirectional,
        use_fsdp=args.fsdp,
        fsdp_strategy=args.fsdp_strategy,
        resume=not args.no_resume,
        verbose=not args.quiet,
        generate_plots=args.generate_plots,
    )

    # Run LOSO
    result = run_loso(config, folds_to_run=args.folds)

    # Print final summary
    print(f"\nLOSO cross-validation complete!")
    print(f"Mean R²: {result.mean_r2:.4f} ± {result.std_r2:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
