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

    # Run ECoG LOSO folds in parallel (1 GPU per fold, 8 GPUs)
    python -m LOSO.runner --dataset ecog --parallel-folds --n-gpus 8

    # Parallel folds with auto-detected GPU count
    python -m LOSO.runner --dataset ecog --parallel-folds
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pickle
import socket
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
# Port management for distributed training
# =============================================================================

def _find_free_port(start: int = 29500, max_tries: int = 100) -> int:
    """Find a free TCP port starting from `start`.

    Tries sequential ports until one is available, avoiding EADDRINUSE errors
    when launching torchrun for sequential LOSO folds.
    """
    for offset in range(max_tries):
        port = start + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", port))
                return port
        except OSError:
            continue
    # Fallback: let the OS pick
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


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

    elif dataset == "pcx1":
        from data import list_pcx1_sessions

        sessions = list_pcx1_sessions()

        metadata = {
            "session_type": ds_config.session_type,
            "description": ds_config.description,
            "source_region": ds_config.source_region,
            "target_region": ds_config.target_region,
            "n_sessions": len(sessions),
        }
        return sessions, metadata

    elif dataset == "ecog":
        from data import (
            _ECOG_DATA_DIR, _enumerate_ecog_recordings,
            load_ecog_subject,
        )

        # Get experiment and region from config
        experiment = "fingerflex"
        source_region = ds_config.source_region
        target_region = ds_config.target_region
        if config is not None:
            experiment = config.ecog_experiment
            source_region = config.ecog_source_region
            target_region = config.ecog_target_region

        if not _ECOG_DATA_DIR.exists():
            raise FileNotFoundError(
                f"ECoG data directory not found: {_ECOG_DATA_DIR}\n"
                f"Run: python scripts/download_ecog.py --experiments {experiment}"
            )

        # Load NPZ and enumerate recordings with correct (subject_idx, block_idx) pairs
        filepath = _ECOG_DATA_DIR / f"{experiment}.npz"
        alldat = np.load(filepath, allow_pickle=True)["dat"]
        recordings = _enumerate_ecog_recordings(alldat)

        # Filter subjects that have enough channels in both regions
        valid_subjects = []
        for si, bi, rec_id in recordings:
            subj_data = load_ecog_subject(
                subject_idx=si,
                block_idx=bi,
                data_dir=_ECOG_DATA_DIR,
                experiment=experiment,
                source_region=source_region,
                target_region=target_region,
                _alldat=alldat,
            )
            if subj_data is not None:
                valid_subjects.append(rec_id)

        metadata = {
            "session_type": ds_config.session_type,  # "subject"
            "description": ds_config.description,
            "source_region": source_region,
            "target_region": target_region,
            "experiment": experiment,
            "n_subjects": len(valid_subjects),
            "note": "LOSO holds out entire subjects (Leave-One-Subject-Out)",
        }
        return valid_subjects, metadata

    elif dataset == "boran_mtl":
        from data import list_boran_subjects, validate_boran_subject, _BORAN_DATA_DIR

        source_region = ds_config.source_region
        target_region = ds_config.target_region
        min_channels = 4
        exclude_soz = False
        if config is not None:
            source_region = config.boran_source_region
            target_region = config.boran_target_region
            min_channels = config.boran_min_channels
            exclude_soz = config.boran_exclude_soz

        if not _BORAN_DATA_DIR.exists():
            raise FileNotFoundError(
                f"Boran MTL data directory not found: {_BORAN_DATA_DIR}\n"
                f"Download from: https://gin.g-node.org/USZ_NCH/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM"
            )

        # Get all subjects and filter using lightweight metadata-only check
        all_subjects = list_boran_subjects(data_dir=_BORAN_DATA_DIR)
        print(f"  Boran MTL: found {len(all_subjects)} subjects in {_BORAN_DATA_DIR}")
        print(f"  Checking {source_region} -> {target_region} (min_channels={min_channels}, exclude_soz={exclude_soz})")

        valid_subjects = []
        for subj_id in all_subjects:
            is_valid = validate_boran_subject(
                subject_id=subj_id,
                data_dir=_BORAN_DATA_DIR,
                source_region=source_region,
                target_region=target_region,
                min_channels=min_channels,
                exclude_soz=exclude_soz,
            )
            if is_valid:
                valid_subjects.append(subj_id)
            print(f"    {subj_id}: {'VALID' if is_valid else 'skipped'}")

        metadata = {
            "session_type": ds_config.session_type,  # "subject"
            "description": ds_config.description,
            "source_region": source_region,
            "target_region": target_region,
            "min_channels": min_channels,
            "exclude_soz": exclude_soz,
            "n_subjects": len(valid_subjects),
            "note": "LOSO holds out entire subjects (Leave-One-Subject-Out)",
        }
        return valid_subjects, metadata

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
    if dataset == "ecog":
        # Validate ECoG source/target regions
        if config.ecog_source_region == config.ecog_target_region:
            raise ValueError(
                f"Source and target regions must be different. "
                f"Both are set to '{config.ecog_source_region}'."
            )

    elif dataset == "dandi_movie":
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
    seed_idx: int = 0,
    gpu_id: Optional[int] = None,
) -> Optional[LOSOFoldResult]:
    """Run a single LOSO fold by calling train.py.

    Args:
        fold_idx: Fold index (0-indexed)
        test_session: Session name to hold out for testing
        all_sessions: List of all session names
        config: LOSO configuration
        output_results_file: Path to save train.py results
        seed_idx: Seed index within this fold (for multi-seed runs)
        gpu_id: GPU index to pin this fold to (sets CUDA_VISIBLE_DEVICES).
                If None, no GPU pinning is done.

    Returns:
        LOSOFoldResult or None on failure
    """
    train_sessions = [s for s in all_sessions if s != test_session]

    # CRITICAL: Check for data leakage before proceeding
    check_data_leakage(fold_idx, test_session, train_sessions, config.dataset)

    # Get dataset config for proper labeling
    ds_config = config.get_dataset_config()
    session_label = "Subject" if ds_config.session_type == "subject" else "Session"
    session_label_lower = session_label.lower()

    print(f"\n{'='*70}")
    print(f"LOSO FOLD {fold_idx + 1}/{len(all_sessions)}")
    print(f"{'='*70}")
    print(f"Dataset: {config.dataset} ({ds_config.source_region} → {ds_config.target_region})")
    print(f"Test {session_label} (held-out LOSO): {test_session}")
    print(f"Train/Val {session_label_lower}s ({len(train_sessions)}): {train_sessions}")
    print(f"  -> Train/Val split: 70/30 random at trial level (NOT session-wise)")
    print(f"✓ No data leakage detected")
    print()

    # Build train.py command
    if config.use_fsdp:
        # Use explicit nproc if provided, otherwise auto-detect
        if config.nproc is not None:
            nproc = config.nproc
        else:
            nproc = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # Find a free port to avoid "address already in use" errors
        # when running sequential folds/seeds with FSDP
        master_port = _find_free_port(29500 + fold_idx * 10 + seed_idx)
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            f"--master_port={master_port}",
            str(PROJECT_ROOT / "train.py"),
        ]
        print(f"Launching with torchrun: {nproc} GPUs, port {master_port}")
    else:
        cmd = [sys.executable, str(PROJECT_ROOT / "train.py")]

    # Base arguments
    # Seed formula ensures unique seeds across all folds and seed runs
    run_seed = config.seed + fold_idx * config.n_seeds + seed_idx
    # Use absolute paths so train.py (which runs with cwd=PROJECT_ROOT) resolves correctly
    abs_output_results_file = output_results_file.absolute()
    cmd.extend([
        "--arch", config.arch,
        "--epochs", str(config.epochs),
        "--batch-size", str(config.batch_size),
        "--lr", str(config.learning_rate),
        "--seed", str(run_seed),
        "--output-results-file", str(abs_output_results_file),
        "--fold", str(fold_idx),
    ])

    # Always give each fold its own output directory so checkpoints/logs
    # don't overwrite each other (critical for concurrent external runners)
    fold_output_dir = config.output_dir.absolute() / "fold_artifacts" / f"fold_{fold_idx}_{test_session}_seed{seed_idx}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--output-dir", str(fold_output_dir)])

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
    elif config.dataset == "pcx1":
        # PCx1-specific: window settings for continuous data
        cmd.extend([
            "--pcx1-window-size", str(config.pcx1_window_size),
            "--pcx1-stride-ratio", str(config.pcx1_stride_ratio),
        ])

    elif config.dataset == "pfc_hpc":
        # PFC-specific: resampling, sliding window, and direction
        if config.pfc_resample_to_1khz:
            cmd.append("--resample-pfc")
        if config.pfc_reverse:
            cmd.append("--pfc-reverse")
        if config.pfc_sliding_window:
            cmd.extend([
                "--pfc-sliding-window",
                "--pfc-window-size", str(config.pfc_window_size),
                "--pfc-stride-ratio", str(config.pfc_stride_ratio),
            ])

    elif config.dataset == "ecog":
        # ECoG-specific: experiment, regions, and window settings
        cmd.extend([
            "--ecog-experiment", config.ecog_experiment,
            "--ecog-source-region", config.ecog_source_region,
            "--ecog-target-region", config.ecog_target_region,
            "--ecog-window-size", str(config.ecog_window_size),
            "--ecog-stride-ratio", str(config.ecog_stride_ratio),
            "--ecog-channel-selection", config.ecog_channel_selection,
        ])
        # For ECoG, the --test-sessions will be treated as subject IDs

    elif config.dataset == "boran_mtl":
        # Boran MTL-specific: regions, window settings, SOZ exclusion
        cmd.extend([
            "--boran-source-region", config.boran_source_region,
            "--boran-target-region", config.boran_target_region,
            "--boran-window-size", str(config.boran_window_size),
            "--boran-stride-ratio", str(config.boran_stride_ratio),
            "--boran-min-channels", str(config.boran_min_channels),
        ])
        if config.boran_exclude_soz:
            cmd.append("--boran-exclude-soz")
        # For Boran, the --test-sessions will be treated as subject IDs

    # LOSO test session holdout with random train/val split
    # CRITICAL: This prevents data leakage in LOSO cross-validation
    # - test_session: Held out completely by SESSION (LOSO), only for final evaluation
    # - train/val: 70/30 random split at TRIAL level from remaining sessions
    # NOTE: We do NOT use --split-by-session here - train/val is random, not session-wise
    cmd.append("--force-recreate-splits")
    cmd.extend(["--test-sessions", test_session])  # LOSO held-out session for final eval
    # No --val-sessions: train.py will do 70/30 random trial split from remaining data

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

    # Wiener residual learning
    if config.wiener_residual:
        cmd.append("--wiener-residual")
        if config.wiener_alpha != 1.0:
            cmd.extend(["--wiener-alpha", str(config.wiener_alpha)])

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

    # Pin to specific GPU when running parallel folds
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
        # Brief delay to let torchrun child processes and sockets clean up
        if config.use_fsdp:
            time.sleep(5)
        print(f"ERROR: train.py failed for fold {fold_idx} (test={test_session})")
        print(f"  Return code: {e.returncode}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"  stderr: {e.stderr[:1000]}...")
        return None

    # Brief delay to let torchrun child processes and sockets clean up
    if config.use_fsdp:
        time.sleep(5)

    elapsed = time.time() - start_time

    # Load results from JSON file
    if output_results_file.exists():
        with open(output_results_file, 'r') as f:
            results = json.load(f)

        # Get dataset info for the result
        ds_config = config.get_dataset_config()

        # CRITICAL: Use TEST metrics for LOSO evaluation (NOT validation metrics!)
        # Validation metrics were used for model selection (best epoch)
        # Test metrics are from the held-out LOSO session - the unbiased evaluation
        #
        # WARNING: Do NOT fall back to validation metrics - that would be data leakage!
        # Use explicit None checks (not `or`) because 0.0 is a valid metric value
        test_r2 = results.get("test_avg_r2")
        test_corr = results.get("test_avg_corr")
        test_loss = results.get("test_avg_delta")

        # Validate that we got actual test metrics
        if test_r2 is None:
            print(f"  WARNING: test_avg_r2 is None in results! Check train.py output.")
            print(f"  Available keys: {list(results.keys())}")
            test_r2 = 0.0
        if test_corr is None:
            test_corr = 0.0
        if test_loss is None:
            test_loss = float('inf')

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
            val_loss=test_loss,
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
        print(f"  Test R²: {fold_result.val_r2:.4f}" if fold_result.val_r2 else "  Test R²: N/A")
        if fold_result.val_corr and fold_result.val_corr != 0.0:
            print(f"  Test Corr: {fold_result.val_corr:.4f}")
        print(f"  Test Loss: {fold_result.val_loss:.4f}" if fold_result.val_loss != float('inf') else "  Test Loss: N/A")
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


def _load_fold_result_from_json(
    results_file: Path,
    fold_idx: int,
    test_session: str,
    all_sessions: List[str],
    config: 'LOSOConfig',
) -> Optional['LOSOFoldResult']:
    """Load a LOSOFoldResult from an existing results JSON file.

    Used to skip re-running folds whose results already exist on disk.

    Args:
        results_file: Path to the fold results JSON
        fold_idx: Fold index
        test_session: Name of the held-out test session
        all_sessions: All session names
        config: LOSO configuration

    Returns:
        LOSOFoldResult or None if file doesn't exist or can't be parsed
    """
    if not results_file.exists():
        return None

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: Could not read {results_file}: {e}")
        return None

    train_sessions = [s for s in all_sessions if s != test_session]
    ds_config = config.get_dataset_config()

    test_r2 = results.get("test_avg_r2")
    test_corr = results.get("test_avg_corr")
    test_loss = results.get("test_avg_delta")

    if test_r2 is None:
        test_r2 = 0.0
    if test_corr is None:
        test_corr = 0.0
    if test_loss is None:
        test_loss = float('inf')

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

    return LOSOFoldResult(
        fold_idx=fold_idx,
        test_session=test_session,
        train_sessions=train_sessions,
        val_r2=test_r2,
        val_corr=test_corr,
        val_loss=test_loss,
        train_loss=results.get("final_train_loss", 0.0),
        per_session_r2=per_session_r2,
        per_session_corr=per_session_corr,
        per_session_loss=per_session_loss,
        epochs_trained=results.get("epochs_trained", config.epochs),
        total_time=results.get("total_time", 0.0),
        config=config.to_dict(),
        dataset=config.dataset,
        session_type=ds_config.session_type,
        n_train_samples=results.get("n_train_samples", 0),
        n_val_samples=results.get("n_val_samples", 0),
    )


def _aggregate_seed_results(
    seed_results: List[LOSOFoldResult],
    fold_idx: int,
    test_session: str,
    config: LOSOConfig,
) -> LOSOFoldResult:
    """Aggregate results from multiple seeds into a single fold result.

    Takes the mean of metrics across seeds. The aggregated result represents
    the expected performance for this fold with training variance accounted for.

    Args:
        seed_results: List of results from each seed run
        fold_idx: Fold index
        test_session: Test session name
        config: LOSO configuration

    Returns:
        Aggregated LOSOFoldResult with mean metrics across seeds
    """
    # Compute mean metrics across seeds
    r2_values = [r.val_r2 for r in seed_results if r.val_r2 is not None]
    corr_values = [r.val_corr for r in seed_results if r.val_corr is not None]
    loss_values = [r.val_loss for r in seed_results if r.val_loss is not None and r.val_loss != float('inf')]

    mean_r2 = float(np.mean(r2_values)) if r2_values else 0.0
    mean_corr = float(np.mean(corr_values)) if corr_values else 0.0
    mean_loss = float(np.mean(loss_values)) if loss_values else float('inf')

    # Compute std for reporting
    std_r2 = float(np.std(r2_values)) if len(r2_values) > 1 else 0.0
    std_corr = float(np.std(corr_values)) if len(corr_values) > 1 else 0.0

    print(f"\n  Fold {fold_idx} aggregated ({len(seed_results)} seeds):")
    print(f"    Test R²: {mean_r2:.4f} ± {std_r2:.4f}")
    if mean_corr != 0.0:
        print(f"    Test Corr: {mean_corr:.4f} ± {std_corr:.4f}")

    # Use the first result as template for other fields
    template = seed_results[0]
    ds_config = config.get_dataset_config()

    return LOSOFoldResult(
        fold_idx=fold_idx,
        test_session=test_session,
        train_sessions=template.train_sessions,
        val_r2=mean_r2,
        val_corr=mean_corr,
        val_loss=mean_loss,
        train_loss=float(np.mean([r.train_loss for r in seed_results])),
        per_session_r2=template.per_session_r2,  # Just use first seed's per-session
        per_session_corr=template.per_session_corr,
        per_session_loss=template.per_session_loss,
        epochs_trained=int(np.mean([r.epochs_trained for r in seed_results])),
        total_time=sum(r.total_time for r in seed_results),
        config=config.to_dict(),
        dataset=config.dataset,
        session_type=ds_config.session_type,
        n_train_samples=template.n_train_samples,
        n_val_samples=template.n_val_samples,
    )


# =============================================================================
# Parallel fold execution helper
# =============================================================================

def _run_fold_all_seeds(
    fold_idx: int,
    all_sessions: List[str],
    config: LOSOConfig,
    results_dir: Path,
    gpu_id: int,
) -> Tuple[int, List[LOSOFoldResult]]:
    """Run all seeds for a single fold on a specific GPU.

    This is the unit of work for parallel fold execution. Each call
    runs one fold (all seeds) on one GPU via CUDA_VISIBLE_DEVICES.

    Args:
        fold_idx: Fold index (0-indexed)
        all_sessions: List of all session names
        config: LOSO configuration
        results_dir: Directory for per-fold result JSON files
        gpu_id: GPU index to pin to

    Returns:
        Tuple of (fold_idx, list of LOSOFoldResult for each seed)
    """
    test_session = all_sessions[fold_idx]
    seed_results = []

    for seed_idx in range(config.n_seeds):
        output_results_file = results_dir / f"fold_{fold_idx}_{test_session}_seed{seed_idx}_results.json"

        print(f"[GPU {gpu_id}] Fold {fold_idx}, Seed {seed_idx + 1}/{config.n_seeds} "
              f"(test={test_session})", flush=True)

        # Skip if result already exists on disk
        existing_result = _load_fold_result_from_json(
            output_results_file, fold_idx, test_session, all_sessions, config,
        )
        if existing_result is not None:
            print(f"[GPU {gpu_id}] Fold {fold_idx} seed {seed_idx}: "
                  f"SKIPPING (exists, Corr={existing_result.val_corr:.4f})", flush=True)
            seed_results.append(existing_result)
            continue

        result = run_single_fold(
            fold_idx=fold_idx,
            test_session=test_session,
            all_sessions=all_sessions,
            config=config,
            output_results_file=output_results_file,
            seed_idx=seed_idx,
            gpu_id=gpu_id,
        )

        if result is not None:
            seed_results.append(result)
            print(f"[GPU {gpu_id}] Fold {fold_idx} seed {seed_idx}: "
                  f"Corr={result.val_corr:.4f}, R²={result.val_r2:.4f}", flush=True)
        else:
            print(f"[GPU {gpu_id}] Fold {fold_idx} seed {seed_idx}: FAILED", flush=True)

    return (fold_idx, seed_results)


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
    # Setup output directory and logging (resolve to absolute path early
    # so all downstream paths are absolute and work correctly regardless
    # of subprocess cwd differences)
    config.output_dir = config.output_dir.absolute()
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

    # Run each fold with multiple seeds
    results_dir = config.output_dir / "fold_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if config.parallel_folds and len(folds_remaining) > 0:
        # =================================================================
        # PARALLEL EXECUTION: run folds across GPUs (1 GPU per fold)
        # =================================================================
        n_gpus = config.n_gpus
        if n_gpus is None:
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        n_gpus = min(n_gpus, len(folds_remaining))

        print(f"PARALLEL MODE: {len(folds_remaining)} folds across {n_gpus} GPUs")
        print(f"  Each GPU trains one fold at a time (no FSDP)")
        print()

        # Process folds in batches of n_gpus
        for batch_start in range(0, len(folds_remaining), n_gpus):
            batch = folds_remaining[batch_start : batch_start + n_gpus]
            print(f"\n{'='*70}")
            print(f"BATCH {batch_start // n_gpus + 1}: Folds {batch} "
                  f"(GPUs 0-{len(batch)-1})")
            print(f"{'='*70}\n")

            # Launch all folds in this batch concurrently
            # Use ThreadPoolExecutor because actual compute is in subprocesses
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {}
                for gpu_idx, fold_idx in enumerate(batch):
                    future = executor.submit(
                        _run_fold_all_seeds,
                        fold_idx=fold_idx,
                        all_sessions=all_sessions,
                        config=config,
                        results_dir=results_dir,
                        gpu_id=gpu_idx,
                    )
                    futures[future] = (fold_idx, gpu_idx)

                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    fold_idx, gpu_idx = futures[future]
                    test_session = all_sessions[fold_idx]
                    try:
                        _, seed_results = future.result()
                    except Exception as e:
                        print(f"\n[GPU {gpu_idx}] Fold {fold_idx} EXCEPTION: {e}")
                        continue

                    if seed_results:
                        aggregated_result = _aggregate_seed_results(
                            seed_results, fold_idx, test_session, config,
                        )
                        fold_results.append(aggregated_result)
                        completed_folds.append(fold_idx)

                        save_checkpoint(
                            checkpoint_path, config, fold_results,
                            completed_folds, all_sessions,
                        )
                        print(f"[GPU {gpu_idx}] Fold {fold_idx} checkpoint saved "
                              f"({len(seed_results)}/{config.n_seeds} seeds)", flush=True)

            print(f"\nBatch complete. {len(completed_folds)}/{n_folds} folds done.")

    else:
        # =================================================================
        # SEQUENTIAL EXECUTION: original behavior
        # =================================================================
        for fold_idx in folds_remaining:
            test_session = all_sessions[fold_idx]

            # Run multiple seeds for this fold
            seed_results = []
            for seed_idx in range(config.n_seeds):
                output_results_file = results_dir / f"fold_{fold_idx}_{test_session}_seed{seed_idx}_results.json"

                print(f"\n--- Fold {fold_idx}, Seed {seed_idx + 1}/{config.n_seeds} ---")

                # Skip if result file already exists on disk (allows resuming without checkpoint)
                existing_result = _load_fold_result_from_json(
                    output_results_file, fold_idx, test_session, all_sessions, config,
                )
                if existing_result is not None:
                    print(f"  SKIPPING: Result file already exists: {output_results_file.name}")
                    print(f"  (loaded: Corr={existing_result.val_corr:.4f}, R²={existing_result.val_r2:.4f})")
                    seed_results.append(existing_result)
                    continue

                result = run_single_fold(
                    fold_idx=fold_idx,
                    test_session=test_session,
                    all_sessions=all_sessions,
                    config=config,
                    output_results_file=output_results_file,
                    seed_idx=seed_idx,
                )

                if result is not None:
                    seed_results.append(result)

                # Small delay between runs when using FSDP
                if config.use_fsdp and seed_idx < config.n_seeds - 1:
                    time.sleep(2)

            # Aggregate results across seeds for this fold
            if seed_results:
                aggregated_result = _aggregate_seed_results(seed_results, fold_idx, test_session, config)
                fold_results.append(aggregated_result)
                completed_folds.append(fold_idx)

                # Save checkpoint after each fold
                save_checkpoint(
                    checkpoint_path,
                    config,
                    fold_results,
                    completed_folds,
                    all_sessions,
                )

                print(f"\nCheckpoint saved after fold {fold_idx} ({len(seed_results)}/{config.n_seeds} seeds completed)")

                # Small delay between folds when using FSDP
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
        choices=["olfactory", "pfc_hpc", "dandi_movie", "pcx1", "ecog", "boran_mtl"],
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
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (optimized: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model architecture (defaults from optimized LOSOConfig)
    parser.add_argument("--arch", type=str, default="condunet", help="Model architecture")
    parser.add_argument("--base-channels", type=int, default=256, help="Base channel count (optimized: 256)")
    parser.add_argument("--n-downsample", type=int, default=2, help="Downsample layers")
    parser.add_argument(
        "--attention-type",
        type=str,
        default="none",
        choices=["none", "basic", "cross_freq_v2"],
        help="Attention type (optimized: none)",
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
    parser.add_argument("--nproc", type=int, default=None, help="Number of GPUs for distributed training (default: auto-detect)")

    # Parallel fold execution (1 GPU per fold, mutually exclusive with FSDP)
    parser.add_argument(
        "--parallel-folds",
        action="store_true",
        help="Run LOSO folds in parallel across GPUs (1 fold per GPU, no FSDP). "
             "E.g., with 8 GPUs and 15 folds, runs 8 folds simultaneously.",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=None,
        help="Number of GPUs for parallel fold execution (default: auto-detect). "
             "Only used with --parallel-folds.",
    )

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

    # PCx1 dataset options (continuous OB->PCx)
    pcx1_group = parser.add_argument_group(
        "PCx1 Dataset",
        "Options for continuous OB to PCx translation (1kHz LFP)"
    )
    pcx1_group.add_argument(
        "--pcx1-window-size",
        type=int,
        default=5000,
        help="Window size in samples (at 1kHz)",
    )
    pcx1_group.add_argument(
        "--pcx1-stride-ratio",
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
        "--pfc-reverse",
        action="store_true",
        help="Reverse direction: translate CA1 -> PFC instead of PFC -> CA1",
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

    # Miller ECoG Library dataset options
    ecog_group = parser.add_argument_group(
        "Miller ECoG Library",
        "Options for Miller ECoG Library inter-region cortical translation"
    )
    ecog_group.add_argument(
        "--ecog-experiment",
        type=str,
        default="fingerflex",
        choices=["fingerflex", "faceshouses", "motor_imagery",
                 "joystick_track", "memory_nback"],
        help="ECoG Library experiment",
    )
    ecog_group.add_argument(
        "--ecog-source-region",
        type=str,
        default="frontal",
        choices=["frontal", "temporal", "parietal", "occipital", "limbic"],
        help="Source brain lobe for translation",
    )
    ecog_group.add_argument(
        "--ecog-target-region",
        type=str,
        default="temporal",
        choices=["frontal", "temporal", "parietal", "occipital", "limbic"],
        help="Target brain lobe for translation",
    )
    ecog_group.add_argument(
        "--ecog-window-size",
        type=int,
        default=5000,
        help="Window size in samples (at 1kHz)",
    )
    ecog_group.add_argument(
        "--ecog-stride-ratio",
        type=float,
        default=0.5,
        help="Stride as fraction of window size (0.5 = 50%% overlap)",
    )

    # Boran MTL Working Memory dataset options
    boran_group = parser.add_argument_group(
        "Boran MTL Working Memory",
        "Options for Boran MTL depth electrode inter-region translation"
    )
    boran_group.add_argument(
        "--boran-source-region",
        type=str,
        default="hippocampus",
        choices=["hippocampus", "entorhinal_cortex", "amygdala"],
        help="Source MTL region for translation",
    )
    boran_group.add_argument(
        "--boran-target-region",
        type=str,
        default="entorhinal_cortex",
        choices=["hippocampus", "entorhinal_cortex", "amygdala"],
        help="Target MTL region for translation",
    )
    boran_group.add_argument(
        "--boran-window-size",
        type=int,
        default=5000,
        help="Window size in samples (at 1kHz, preprocessed)",
    )
    boran_group.add_argument(
        "--boran-stride-ratio",
        type=float,
        default=0.5,
        help="Stride as fraction of window size (0.5 = 50%% overlap)",
    )
    boran_group.add_argument(
        "--boran-min-channels",
        type=int,
        default=4,
        help="Minimum channels per region",
    )
    boran_group.add_argument(
        "--boran-exclude-soz",
        action="store_true",
        help="Exclude Seizure Onset Zone probes",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate mutually exclusive options
    if args.parallel_folds and args.fsdp:
        print("ERROR: --parallel-folds and --fsdp are mutually exclusive.")
        print("  --parallel-folds: 1 GPU per fold (parallel LOSO folds)")
        print("  --fsdp: all GPUs per fold (distributed single model)")
        return 1

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
        nproc=args.nproc,
        # Parallel folds
        parallel_folds=args.parallel_folds,
        n_gpus=args.n_gpus,
        # Execution
        resume=not args.no_resume,
        verbose=not args.quiet,
        generate_plots=args.generate_plots,
        # DANDI-specific
        dandi_source_region=args.dandi_source_region,
        dandi_target_region=args.dandi_target_region,
        dandi_window_size=args.dandi_window_size,
        dandi_stride_ratio=args.dandi_stride_ratio,
        # PCx1-specific
        pcx1_window_size=args.pcx1_window_size,
        pcx1_stride_ratio=args.pcx1_stride_ratio,
        # PFC-specific
        pfc_resample_to_1khz=args.pfc_resample,
        pfc_sliding_window=args.pfc_sliding_window,
        pfc_window_size=args.pfc_window_size,
        pfc_stride_ratio=args.pfc_stride_ratio,
        pfc_reverse=args.pfc_reverse,
        # ECoG-specific
        ecog_experiment=args.ecog_experiment,
        ecog_source_region=args.ecog_source_region,
        ecog_target_region=args.ecog_target_region,
        ecog_window_size=args.ecog_window_size,
        ecog_stride_ratio=args.ecog_stride_ratio,
        # Boran MTL-specific
        boran_source_region=args.boran_source_region,
        boran_target_region=args.boran_target_region,
        boran_window_size=args.boran_window_size,
        boran_stride_ratio=args.boran_stride_ratio,
        boran_min_channels=args.boran_min_channels,
        boran_exclude_soz=args.boran_exclude_soz,
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
