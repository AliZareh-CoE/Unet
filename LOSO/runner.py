"""LOSO (Leave-One-Subject-Out) Cross-Validation Runner.

Implements Leave-One-Session-Out cross-validation for neural signal
translation models. Systematically holds out each recording session
for testing while training on all others.

Usage:
    # Run LOSO cross-validation with default settings
    python -m LOSO.runner

    # Specify dataset and output directory
    python -m LOSO.runner --dataset olfactory --output-dir results/loso

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

from LOSO.config import (
    LOSOConfig,
    LOSOFoldResult,
    LOSOResult,
    DatasetConfig,
    DATASET_CONFIGS,
    get_dataset_config,
)


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

def get_all_sessions(
    dataset: str,
    config: Optional[LOSOConfig] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Get list of all session/subject names for a dataset with metadata.

    This function discovers all available sessions or subjects for LOSO
    cross-validation. For session-based datasets (olfactory, pfc_hpc), it
    returns recording session IDs. For subject-based datasets (dandi_movie),
    it returns subject IDs.

    Args:
        dataset: Dataset name (olfactory, pfc_hpc, dandi_movie)
        config: Optional LOSOConfig for dataset-specific parameters

    Returns:
        Tuple of:
            - sessions: List of session/subject identifiers
            - metadata: Dict with dataset-specific info:
                - session_type: "session" or "subject"
                - trials_per_session: Dict mapping session -> trial count (if available)
                - total_trials: Total number of trials (if available)
                - description: Human-readable dataset description

    Raises:
        ValueError: If dataset is not recognized
        FileNotFoundError: If required data files are not found
    """
    ds_config = get_dataset_config(dataset)

    if dataset == "olfactory":
        from data import load_session_ids, ODOR_CSV_PATH

        session_ids, session_to_idx, idx_to_session = load_session_ids(ODOR_CSV_PATH)
        sessions = sorted(session_to_idx.keys())

        # Count trials per session
        unique_ids, counts = np.unique(session_ids, return_counts=True)
        trials_per_session = {
            idx_to_session[int(sid)]: int(cnt)
            for sid, cnt in zip(unique_ids, counts)
        }

        metadata = {
            "session_type": ds_config.session_type,
            "trials_per_session": trials_per_session,
            "total_trials": int(len(session_ids)),
            "description": ds_config.description,
            "source_region": ds_config.source_region,
            "target_region": ds_config.target_region,
        }
        return sessions, metadata

    elif dataset == "pfc_hpc":
        from data import load_pfc_session_ids, PFC_META_PATH

        session_ids, session_to_idx, idx_to_session = load_pfc_session_ids(PFC_META_PATH)
        sessions = sorted(session_to_idx.keys())

        # Count trials per session
        unique_ids, counts = np.unique(session_ids, return_counts=True)
        trials_per_session = {
            idx_to_session[int(sid)]: int(cnt)
            for sid, cnt in zip(unique_ids, counts)
        }

        metadata = {
            "session_type": ds_config.session_type,
            "trials_per_session": trials_per_session,
            "total_trials": int(len(session_ids)),
            "description": ds_config.description,
            "source_region": ds_config.source_region,
            "target_region": ds_config.target_region,
        }
        return sessions, metadata

    elif dataset == "dandi_movie":
        from data import _DANDI_DATA_DIR, list_dandi_nwb_files

        if not _DANDI_DATA_DIR.exists():
            raise FileNotFoundError(
                f"DANDI data directory not found: {_DANDI_DATA_DIR}\n"
                f"Please download DANDI 000623 dataset to this location."
            )

        # Discover subjects from NWB files
        nwb_files = list_dandi_nwb_files(_DANDI_DATA_DIR)
        if not nwb_files:
            raise FileNotFoundError(
                f"No NWB files found in DANDI data directory: {_DANDI_DATA_DIR}"
            )

        # Extract unique subject IDs
        subjects = []
        for f in nwb_files:
            stem = f.stem
            if "sub-" in stem:
                subj_id = stem.split("_")[0]  # Get sub-CSXX part
            else:
                subj_id = stem
            if subj_id not in subjects:
                subjects.append(subj_id)

        subjects = sorted(subjects)

        # Get source/target regions from config if available
        source_region = ds_config.source_region
        target_region = ds_config.target_region
        if config is not None:
            source_region = config.dandi_source_region
            target_region = config.dandi_target_region

        metadata = {
            "session_type": ds_config.session_type,  # "subject"
            "description": ds_config.description,
            "source_region": source_region,
            "target_region": target_region,
            "n_subjects": len(subjects),
            "note": "LOSO holds out entire subjects (Leave-One-Subject-Out)",
        }
        return subjects, metadata

    else:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset}'. Available datasets: {available}"
        )


def validate_sessions(
    sessions: List[str],
    dataset: str,
    config: LOSOConfig,
) -> None:
    """Validate that we have enough sessions for LOSO.

    Args:
        sessions: List of session/subject identifiers
        dataset: Dataset name
        config: LOSO configuration

    Raises:
        ValueError: If validation fails
    """
    ds_config = get_dataset_config(dataset)
    session_label = "subjects" if ds_config.session_type == "subject" else "sessions"

    # Minimum sessions check
    if len(sessions) < 3:
        raise ValueError(
            f"LOSO requires at least 3 {session_label}. "
            f"Dataset '{dataset}' has only {len(sessions)}."
        )

    # Dataset-specific validations
    if dataset == "dandi_movie":
        # Validate DANDI source/target regions
        valid_regions = ["amygdala", "hippocampus", "medial_frontal_cortex"]

        if config.dandi_source_region == config.dandi_target_region:
            raise ValueError(
                f"Source and target regions must be different. "
                f"Both are set to '{config.dandi_source_region}'."
            )

        for region_name, region_val in [
            ("source", config.dandi_source_region),
            ("target", config.dandi_target_region),
        ]:
            if region_val not in valid_regions:
                raise ValueError(
                    f"Invalid DANDI {region_name} region: '{region_val}'. "
                    f"Valid options: {valid_regions}"
                )

    print(f"✓ Configuration validated for {dataset}")
    print(f"  {len(sessions)} {session_label} available for LOSO")


def check_data_leakage(
    fold_idx: int,
    test_session: str,
    train_sessions: List[str],
    dataset: str,
) -> None:
    """Verify no data leakage between train and test.

    This is a critical safety check for LOSO cross-validation.
    Raises RuntimeError if the test session appears in training set.

    Args:
        fold_idx: Current fold index
        test_session: Session/subject held out for testing
        train_sessions: List of sessions/subjects used for training
        dataset: Dataset name (for error message)

    Raises:
        RuntimeError: If data leakage is detected
    """
    ds_config = get_dataset_config(dataset)
    session_label = "subject" if ds_config.session_type == "subject" else "session"

    # Check for leakage
    if test_session in train_sessions:
        raise RuntimeError(
            f"CRITICAL DATA LEAKAGE DETECTED in fold {fold_idx}!\n"
            f"  Test {session_label} '{test_session}' found in training set!\n"
            f"  Training {session_label}s: {train_sessions}\n"
            f"  This would invalidate all results. Aborting."
        )

    # Also check that test session exists
    if not test_session:
        raise RuntimeError(
            f"Empty test {session_label} in fold {fold_idx}. "
            f"This indicates a bug in session assignment."
        )

    # Check train sessions are not empty
    if len(train_sessions) == 0:
        raise RuntimeError(
            f"No training {session_label}s in fold {fold_idx}. "
            f"Cannot train without any training data."
        )


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
            val_corr=r_dict.get("val_corr", 0.0),
            val_loss=r_dict["val_loss"],
            train_loss=r_dict.get("train_loss", 0.0),
            per_session_r2=r_dict.get("per_session_r2", {}),
            per_session_corr=r_dict.get("per_session_corr", {}),
            per_session_loss=r_dict.get("per_session_loss", {}),
            epochs_trained=r_dict.get("epochs_trained", 0),
            total_time=r_dict.get("total_time", 0.0),
            config=r_dict.get("config", {}),
            # Dataset metadata (may not exist in old checkpoints)
            dataset=r_dict.get("dataset", ""),
            session_type=r_dict.get("session_type", ""),
            n_train_samples=r_dict.get("n_train_samples", 0),
            n_val_samples=r_dict.get("n_val_samples", 0),
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
    non_test_sessions = [s for s in all_sessions if s != test_session]

    # CRITICAL: Split non-test sessions into train and validation
    # This prevents data leakage by NOT using the test session for model selection
    # Validation sessions are used for best epoch selection / early stopping
    n_val = max(1, int(len(non_test_sessions) * config.val_ratio))
    # Use deterministic split based on fold_idx for reproducibility
    rng = np.random.RandomState(config.seed + fold_idx)
    shuffled_sessions = non_test_sessions.copy()
    rng.shuffle(shuffled_sessions)
    val_sessions = shuffled_sessions[:n_val]
    train_sessions = shuffled_sessions[n_val:]

    # CRITICAL: Check for data leakage before proceeding
    check_data_leakage(fold_idx, test_session, train_sessions + val_sessions, config.dataset)

    # Get dataset config for proper labeling
    ds_config = config.get_dataset_config()
    session_label = "Subject" if ds_config.session_type == "subject" else "Session"
    session_label_lower = session_label.lower()

    print(f"\n{'='*70}")
    print(f"LOSO FOLD {fold_idx + 1}/{len(all_sessions)}")
    print(f"{'='*70}")
    print(f"Dataset: {config.dataset} ({ds_config.source_region} → {ds_config.target_region})")
    print(f"Test {session_label} (held-out): {test_session}")
    print(f"Validation {session_label_lower}s ({len(val_sessions)}): {val_sessions}")
    print(f"Training {session_label_lower}s ({len(train_sessions)}): {train_sessions}")
    print(f"✓ No data leakage detected")
    print()

    # Build train.py command
    if config.use_fsdp:
        nproc = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # Use unique port per fold to avoid "address already in use" errors
        # when running sequential folds with FSDP
        master_port = 29500 + fold_idx
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            f"--master_port={master_port}",
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

    # Dataset selection - map LOSO dataset names to train.py dataset names
    ds_config = config.get_dataset_config()
    train_py_dataset = ds_config.train_py_dataset_name
    if train_py_dataset != "olfactory":
        cmd.extend(["--dataset", train_py_dataset])

    # Dataset-specific arguments
    if config.dataset == "dandi_movie":
        # DANDI-specific: source/target regions and window settings
        cmd.extend([
            "--dandi-source-region", config.dandi_source_region,
            "--dandi-target-region", config.dandi_target_region,
            "--dandi-window-size", str(config.dandi_window_size),
            "--dandi-stride-ratio", str(config.dandi_stride_ratio),
        ])
        # For DANDI, the --val-sessions will be treated as subject IDs

    elif config.dataset == "pfc_hpc":
        # PFC-specific: resampling and sliding window options
        if config.pfc_resample_to_1khz:
            cmd.append("--resample-pfc")
        if config.pfc_sliding_window:
            cmd.extend([
                "--pfc-sliding-window",
                "--pfc-window-size", str(config.pfc_window_size),
                "--pfc-stride-ratio", str(config.pfc_stride_ratio),
            ])

    # Session-based splitting with proper train/val/test separation
    # CRITICAL: This prevents data leakage in LOSO cross-validation
    # - test_session: Held out completely (LOSO), only for final evaluation
    # - val_sessions: Used for model selection (best epoch, early stopping)
    # - train_sessions: Used for training (remaining sessions)
    cmd.append("--split-by-session")
    cmd.append("--force-recreate-splits")
    cmd.extend(["--test-sessions", test_session])  # LOSO held-out session for final eval
    cmd.extend(["--val-sessions"] + val_sessions)  # For model selection (no leakage)

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

    # Noise augmentation (optimized for LOSO)
    if config.use_noise_augmentation:
        cmd.append("--use-noise-augmentation")
        cmd.extend(["--noise-gaussian-std", str(config.noise_gaussian_std)])
        if config.noise_pink:
            cmd.append("--noise-pink")
            cmd.extend(["--noise-pink-std", str(config.noise_pink_std)])
        if config.noise_channel_dropout > 0:
            cmd.extend(["--noise-channel-dropout", str(config.noise_channel_dropout)])
        if config.noise_temporal_dropout > 0:
            cmd.extend(["--noise-temporal-dropout", str(config.noise_temporal_dropout)])
        cmd.extend(["--noise-prob", str(config.noise_prob)])

    # FSDP
    if config.use_fsdp:
        cmd.extend(["--fsdp", "--fsdp-strategy", config.fsdp_strategy])

    # Set NCCL environment variables for stability with FSDP
    env = os.environ.copy()
    env["TORCH_NCCL_ENABLE_MONITORING"] = "0"  # Disable watchdog completely
    env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1800"  # 30 minutes
    env["NCCL_TIMEOUT"] = "1800"
    env["NCCL_DEBUG"] = "WARN"

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
                env=env,
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
                env=env,
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

        # Get dataset info for the result
        ds_config = config.get_dataset_config()

        # CRITICAL: Use TEST metrics for LOSO evaluation (not validation metrics)
        # Validation metrics were used for model selection (best epoch)
        # Test metrics are from the held-out LOSO session - the unbiased evaluation
        test_r2 = results.get("test_avg_r2", results.get("best_val_r2", 0.0))
        test_corr = results.get("test_avg_corr", results.get("best_val_corr", 0.0))

        # Per-session test results (for the held-out test session)
        per_session_test = results.get("per_session_test_results", [])
        per_session_r2 = {}
        per_session_corr = {}
        per_session_loss = {}
        for session_result in per_session_test:
            session_name = session_result.get("session", "")
            if session_name:
                per_session_r2[session_name] = session_result.get("r2", 0.0)
                per_session_corr[session_name] = session_result.get("corr", 0.0)
                per_session_loss[session_name] = session_result.get("delta", 0.0)

        fold_result = LOSOFoldResult(
            fold_idx=fold_idx,
            test_session=test_session,
            train_sessions=train_sessions,
            val_r2=test_r2,  # LOSO metric: test R² from held-out session
            val_corr=test_corr,  # LOSO metric: test correlation from held-out session
            val_loss=results.get("test_avg_delta", results.get("best_val_loss", float('inf'))),
            train_loss=results.get("final_train_loss", 0.0),
            per_session_r2=per_session_r2,
            per_session_corr=per_session_corr,
            per_session_loss=per_session_loss,
            epochs_trained=results.get("epochs_trained", config.epochs),
            total_time=elapsed,
            config=config.to_dict(),
            # Dataset metadata
            dataset=config.dataset,
            session_type=ds_config.session_type,
            n_train_samples=results.get("n_train_samples", 0),
            n_val_samples=results.get("n_val_samples", 0),
        )

        # Use appropriate label (session vs subject)
        session_label = "Subject" if ds_config.session_type == "subject" else "Session"

        print(f"\nFold {fold_idx} completed:")
        print(f"  Held-out {session_label} (TEST): {test_session}")
        print(f"  Test R²: {fold_result.val_r2:.4f}")
        if test_corr != 0.0:
            print(f"  Test Corr: {fold_result.val_corr:.4f}")
        print(f"  Test Loss: {fold_result.val_loss:.4f}")
        print(f"  Time: {elapsed/60:.1f} minutes")

        # Print per-session test results if available
        if fold_result.per_session_corr:
            print(f"  Per-session test results:")
            for sess, corr in sorted(fold_result.per_session_corr.items()):
                r2 = fold_result.per_session_r2.get(sess, 0.0)
                print(f"    {sess}: Corr={corr:.4f}, R²={r2:.4f}")

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

    # Get dataset configuration for display
    ds_config = config.get_dataset_config()
    session_label = config.get_session_type_label()

    print("=" * 70)
    print("LOSO (Leave-One-Session-Out) Cross-Validation")
    print("=" * 70)
    print(f"Dataset: {config.dataset}")
    print(f"Description: {ds_config.description}")
    print(f"Translation: {ds_config.source_region} → {ds_config.target_region}")
    print(f"CV Type: Leave-One-{session_label}-Out")
    print(f"Output directory: {config.output_dir}")
    print()

    # Get all sessions/subjects with metadata
    all_sessions, session_metadata = get_all_sessions(config.dataset, config)
    n_folds = len(all_sessions)

    # Validate configuration
    validate_sessions(all_sessions, config.dataset, config)

    print(f"\nFound {n_folds} {session_label.lower()}s: {all_sessions}")
    if "trials_per_session" in session_metadata:
        total = session_metadata.get("total_trials", "?")
        print(f"Total trials: {total}")
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

            # Small delay between folds when using FSDP to ensure
            # distributed processes fully terminate and release ports
            if config.use_fsdp and fold_idx < folds_remaining[-1]:
                time.sleep(2)

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
        default="results/loso",
        help="Output directory for results",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs per fold (no early stopping)")
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

    # =========================================================================
    # Dataset-specific arguments
    # =========================================================================

    # DANDI Movie dataset options (human iEEG)
    dandi_group = parser.add_argument_group(
        "DANDI Movie Dataset",
        "Options for DANDI 000623 human iEEG movie watching dataset"
    )
    dandi_group.add_argument(
        "--dandi-source-region",
        type=str,
        default="amygdala",
        choices=["amygdala", "hippocampus", "medial_frontal_cortex"],
        help="Source brain region for translation",
    )
    dandi_group.add_argument(
        "--dandi-target-region",
        type=str,
        default="hippocampus",
        choices=["amygdala", "hippocampus", "medial_frontal_cortex"],
        help="Target brain region for translation",
    )
    dandi_group.add_argument(
        "--dandi-window-size",
        type=int,
        default=5000,
        help="Window size in samples (at 1kHz)",
    )
    dandi_group.add_argument(
        "--dandi-stride-ratio",
        type=float,
        default=0.5,
        help="Stride as fraction of window size (0.5 = 50%% overlap)",
    )

    # PFC/HPC dataset options
    pfc_group = parser.add_argument_group(
        "PFC/HPC Dataset",
        "Options for PFC to hippocampus (CA1) translation dataset"
    )
    pfc_group.add_argument(
        "--pfc-resample",
        action="store_true",
        help="Resample PFC data from 1.25kHz to 1kHz",
    )
    pfc_group.add_argument(
        "--pfc-sliding-window",
        action="store_true",
        help="Use sliding window for PFC dataset",
    )
    pfc_group.add_argument(
        "--pfc-window-size",
        type=int,
        default=2500,
        help="Window size in samples for sliding window",
    )
    pfc_group.add_argument(
        "--pfc-stride-ratio",
        type=float,
        default=0.5,
        help="Stride as fraction of window size",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create config from args
    config = LOSOConfig(
        # Dataset
        dataset=args.dataset,
        output_dir=Path(args.output_dir),
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        # Model architecture
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
        # Optimizer
        optimizer=args.optimizer,
        lr_schedule=args.lr_schedule,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        # Session adaptation
        use_session_stats=args.use_session_stats,
        session_use_spectral=args.session_use_spectral,
        use_adaptive_scaling=args.use_adaptive_scaling,
        use_bidirectional=not args.no_bidirectional,
        # FSDP
        use_fsdp=args.fsdp,
        fsdp_strategy=args.fsdp_strategy,
        # Execution
        resume=not args.no_resume,
        verbose=not args.quiet,
        generate_plots=args.generate_plots,
        # DANDI-specific
        dandi_source_region=args.dandi_source_region,
        dandi_target_region=args.dandi_target_region,
        dandi_window_size=args.dandi_window_size,
        dandi_stride_ratio=args.dandi_stride_ratio,
        # PFC-specific
        pfc_resample_to_1khz=args.pfc_resample,
        pfc_sliding_window=args.pfc_sliding_window,
        pfc_window_size=args.pfc_window_size,
        pfc_stride_ratio=args.pfc_stride_ratio,
    )

    # Run LOSO
    result = run_loso(config, folds_to_run=args.folds)

    # Print final summary
    ds_config = config.get_dataset_config()
    session_label = config.get_session_type_label().lower()
    print(f"\nLOSO cross-validation complete!")
    print(f"Dataset: {config.dataset} ({len(result.all_sessions)} {session_label}s)")
    print(f"Mean R²: {result.mean_r2:.4f} ± {result.std_r2:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
