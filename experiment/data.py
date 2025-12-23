"""Data loading and preprocessing for neural signal translation.

This module handles all data operations:
- Loading OB/PCx LFP recordings from .npy files (olfactory dataset)
- Loading PFC/CA1 recordings from .npy files (prefrontal-hippocampal dataset)
- Label loading from CSV (odor labels or trial type)
- Stratified train/val/test splits
- Normalization (z-score per channel)
- PyTorch Dataset classes for paired and unpaired data
- Temporal ablation utilities

Supported Datasets:
    1. Olfactory (OB/PCx): 
       - Shape: (trials, 2, 32, 5000) at 1kHz
       - Labels: odor_name (7 classes)
       
    2. PFC/Hippocampus (PFC/CA1):
       - Shape: (trials, 6250, 96) at 1.25kHz
       - Channels 0-63: PFC, 64-95: CA1
       - Labels: trial_type (Right/Left)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import StratifiedShuffleSplit

# =============================================================================
# Distributed Training Utilities
# =============================================================================

def _is_primary_rank() -> bool:
    """Check if current process is rank 0 (primary) for printing.

    In distributed training, only rank 0 should print to avoid duplicate output.
    Returns True if not in distributed mode or if rank == 0.
    """
    import os
    rank = int(os.environ.get("RANK", 0))
    return rank == 0


# =============================================================================
# Dataset Type Enum
# =============================================================================

class DatasetType(Enum):
    """Supported dataset types."""
    OLFACTORY = "olfactory"      # OB/PCx dataset
    PFC_HPC = "pfc_hpc"          # PFC/Hippocampus (CA1) dataset


# =============================================================================
# Constants - Olfactory Dataset (OB/PCx)
# =============================================================================
# Data directory - /data is the primary location on server
_DATA_DIR = Path("/data")

DATA_PATH = _DATA_DIR / "signal_windows_1khz.npy"
ODOR_CSV_PATH = _DATA_DIR / "signal_windows_meta_1khz.csv"
TRAIN_SPLIT_PATH = _DATA_DIR / "train_indices.npy"
VAL_SPLIT_PATH = _DATA_DIR / "val_indices.npy"
TEST_SPLIT_PATH = _DATA_DIR / "test_indices.npy"

# Session-based split paths (for held-out session evaluation)
SESSION_TRAIN_SPLIT_PATH = _DATA_DIR / "session_train_indices.npy"
SESSION_VAL_SPLIT_PATH = _DATA_DIR / "session_val_indices.npy"
SESSION_TEST_SPLIT_PATH = _DATA_DIR / "session_test_indices.npy"
SESSION_SPLIT_INFO_PATH = _DATA_DIR / "session_split_info.json"

SAMPLING_RATE_HZ = 1000
T_INPUT_START_S = 0.0
T_INPUT_END_S = 5.0
INPUT_START_IDX = int(T_INPUT_START_S * SAMPLING_RATE_HZ)
INPUT_END_IDX = int(T_INPUT_END_S * SAMPLING_RATE_HZ)
INPUT_WINDOW = INPUT_END_IDX - INPUT_START_IDX

T_TARGET_START_S = 0.5
T_TARGET_END_S = 4.5
TARGET_START_IDX = int(T_TARGET_START_S * SAMPLING_RATE_HZ)
TARGET_END_IDX = int(T_TARGET_END_S * SAMPLING_RATE_HZ)
TARGET_WINDOW = TARGET_END_IDX - TARGET_START_IDX
CROP_START = TARGET_START_IDX - INPUT_START_IDX
CROP_END = CROP_START + TARGET_WINDOW

NUM_ODORS = 7

# =============================================================================
# Constants - PFC/Hippocampus Dataset
# =============================================================================
_PFC_DATA_DIR = Path("/data/pfc/processed_data")
PFC_DATA_PATH = _PFC_DATA_DIR / "neural_data.npy"
PFC_META_PATH = _PFC_DATA_DIR / "metadata.csv"
PFC_TRAIN_SPLIT_PATH = _PFC_DATA_DIR / "train_indices.npy"
PFC_VAL_SPLIT_PATH = _PFC_DATA_DIR / "val_indices.npy"
PFC_TEST_SPLIT_PATH = _PFC_DATA_DIR / "test_indices.npy"

# PFC dataset specifics
PFC_SAMPLING_RATE_HZ = 1250
PFC_DURATION_S = 5.0
PFC_TIME_POINTS = 6250  # 5.0 * 1250
PFC_TOTAL_CHANNELS = 96
PFC_CHANNELS = 64       # Channels 0-63
CA1_CHANNELS = 32       # Channels 64-95
PFC_CHANNEL_START = 0
PFC_CHANNEL_END = 64
CA1_CHANNEL_START = 64
CA1_CHANNEL_END = 96

# Trial type labels for PFC dataset
NUM_TRIAL_TYPES = 2  # Right, Left


# =============================================================================
# Normalization Stats
# =============================================================================

@dataclass
class NormalizationStats:
    """Statistics for z-score normalization."""
    mean: np.ndarray
    std: np.ndarray

    def save(self, path: Path) -> None:
        payload = {"mean": self.mean.tolist(), "std": self.std.tolist()}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "NormalizationStats":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return NormalizationStats(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_signals(path: Path = DATA_PATH) -> np.ndarray:
    """Load neural signals from .npy file.

    Expected shape: (trials, 2, channels, time)
    - trials: number of recording trials
    - 2: [OB, PCx] regions
    - channels: 32 electrodes
    - time: 5000 samples at 1kHz
    """
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 4 or arr.shape[1] != 2:
        raise ValueError(f"Expected shape (trials, 2, channels, time), got {arr.shape}")
    return arr


def load_pfc_signals(path: Path = PFC_DATA_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Load PFC/Hippocampus neural signals from .npy file.

    Input shape from file: (trials, time, channels) = (494, 6250, 96)
    - trials: 494 distinct trials
    - time: 6250 time steps (5.0 seconds at 1250 Hz)
    - channels: 96 total
        - Indices 0-63: Prefrontal Cortex (PFC)
        - Indices 64-95: Hippocampus (CA1)

    Returns:
        pfc: PFC signals array [n_samples, channels, time] = [494, 64, 6250]
        ca1: CA1 signals array [n_samples, channels, time] = [494, 32, 6250]
    """
    if not path.exists():
        raise FileNotFoundError(f"PFC signal file not found: {path}")

    arr = np.load(path).astype(np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected shape (trials, time, channels), got {arr.shape}")

    n_trials, n_time, n_channels = arr.shape
    if _is_primary_rank():
        print(f"Loaded PFC data: {n_trials} trials, {n_time} time points, {n_channels} channels")

    if n_channels != PFC_TOTAL_CHANNELS:
        raise ValueError(f"Expected {PFC_TOTAL_CHANNELS} channels, got {n_channels}")

    # Transpose from (trials, time, channels) to (trials, channels, time)
    arr = arr.transpose(0, 2, 1)  # Now: (494, 96, 6250)

    # Split into PFC and CA1
    pfc = arr[:, PFC_CHANNEL_START:PFC_CHANNEL_END, :]  # (494, 64, 6250)
    ca1 = arr[:, CA1_CHANNEL_START:CA1_CHANNEL_END, :]  # (494, 32, 6250)

    if _is_primary_rank():
        print(f"Split into PFC: {pfc.shape}, CA1: {ca1.shape}")

    return pfc, ca1


def load_pfc_metadata(
    csv_path: Path = PFC_META_PATH,
    num_trials: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, int], pd.DataFrame]:
    """Load PFC trial metadata from CSV file.

    The CSV contains:
        - session: Recording session ID (e.g., 'EE.049')
        - rat: Subject ID (e.g., 'EE')
        - trial_id: Original trial index within session
        - trial_type: Task condition ('Right' or 'Left')
        - original_duration: Raw length before processing
        - status: How trial was fitted to 5s window

    Returns:
        trial_types: Array of integer trial type IDs (0=Left, 1=Right or similar)
        vocab: Mapping from trial type name to ID
        df: Full metadata DataFrame for additional filtering
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"PFC metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "trial_type" not in df.columns:
        raise ValueError("CSV must contain 'trial_type' column")

    if num_trials is not None and len(df) != num_trials:
        raise ValueError(f"Metadata count ({len(df)}) must match trials ({num_trials})")

    # Create vocabulary for trial types
    trial_type_names = df["trial_type"].astype(str).tolist()
    vocab: Dict[str, int] = {}
    ids = np.empty(len(trial_type_names), dtype=np.int64)

    for idx, name in enumerate(trial_type_names):
        if name not in vocab:
            vocab[name] = len(vocab)
        ids[idx] = vocab[name]

    if _is_primary_rank():
        print(f"PFC trial types: {vocab}")
        print(f"Trial type distribution: {np.bincount(ids)}")

    return ids, vocab, df


def load_pfc_session_ids(
    csv_path: Path = PFC_META_PATH,
    num_trials: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """Load session IDs from PFC metadata for session-based splitting.

    Args:
        csv_path: Path to metadata CSV
        num_trials: Expected number of trials (for validation)

    Returns:
        session_ids: Array of integer session indices for each trial
        session_to_idx: Dict mapping session name -> integer index
        idx_to_session: Dict mapping integer index -> session name
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "session" not in df.columns:
        raise ValueError("CSV must contain 'session' column")

    raw_sessions = df["session"].astype(str).values

    if num_trials is not None and len(raw_sessions) != num_trials:
        raise ValueError(f"Session count ({len(raw_sessions)}) must match trials ({num_trials})")

    # Map unique session names to integer indices
    unique_sessions = sorted(set(raw_sessions))
    session_to_idx = {name: idx for idx, name in enumerate(unique_sessions)}
    idx_to_session = {idx: name for name, idx in session_to_idx.items()}

    session_ids = np.array([session_to_idx[s] for s in raw_sessions], dtype=np.int64)

    if _is_primary_rank():
        print(f"Found {len(unique_sessions)} unique PFC sessions: {unique_sessions[:5]}{'...' if len(unique_sessions) > 5 else ''}")

    return session_ids, session_to_idx, idx_to_session


def load_pfc_rat_ids(
    csv_path: Path = PFC_META_PATH,
    num_trials: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """Load rat/subject IDs from PFC metadata for subject-based splitting.

    This enables leave-one-subject-out cross-validation.

    Returns:
        rat_ids: Array of integer rat indices for each trial
        rat_to_idx: Dict mapping rat name -> integer index
        idx_to_rat: Dict mapping integer index -> rat name
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "rat" not in df.columns:
        raise ValueError("CSV must contain 'rat' column")

    raw_rats = df["rat"].astype(str).values

    if num_trials is not None and len(raw_rats) != num_trials:
        raise ValueError(f"Rat count ({len(raw_rats)}) must match trials ({num_trials})")

    # Map unique rat names to integer indices
    unique_rats = sorted(set(raw_rats))
    rat_to_idx = {name: idx for idx, name in enumerate(unique_rats)}
    idx_to_rat = {idx: name for name, idx in rat_to_idx.items()}

    rat_ids = np.array([rat_to_idx[r] for r in raw_rats], dtype=np.int64)

    if _is_primary_rank():
        print(f"Found {len(unique_rats)} unique rats: {unique_rats}")

    return rat_ids, rat_to_idx, idx_to_rat


def load_odor_labels(csv_path: Path = ODOR_CSV_PATH, num_trials: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load odor labels from CSV file.

    Returns:
        odor_ids: Array of integer odor IDs
        vocab: Mapping from odor name to ID
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Odor CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "odor_name" not in df.columns:
        raise ValueError("CSV must contain 'odor_name' column")
    odor_names = df["odor_name"].astype(str).tolist()
    if num_trials is not None and len(odor_names) != num_trials:
        raise ValueError(f"Odor metadata count ({len(odor_names)}) must match trials ({num_trials})")

    vocab: Dict[str, int] = {}
    ids = np.empty(len(odor_names), dtype=np.int64)
    for idx, name in enumerate(odor_names):
        if name not in vocab:
            vocab[name] = len(vocab)
        ids[idx] = vocab[name]
    return ids, vocab


def load_session_ids(
    csv_path: Path = ODOR_CSV_PATH,
    session_column: str = "recording_id",
    num_trials: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """Load session/recording IDs from CSV file.

    Handles both integer and string session IDs (e.g., '141208-1' date-based format).

    Args:
        csv_path: Path to metadata CSV
        session_column: Column name containing session IDs (default: "recording_id")
        num_trials: Expected number of trials (for validation)

    Returns:
        session_ids: Array of integer session indices for each trial
        session_to_idx: Dict mapping original session name -> integer index
        idx_to_session: Dict mapping integer index -> original session name
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # Try multiple possible column names for session
    possible_columns = [session_column, "recording_id", "session", "session_id", "rec_id"]
    found_column = None
    for col in possible_columns:
        if col in df.columns:
            found_column = col
            break

    if found_column is None:
        raise ValueError(
            f"CSV must contain a session column. Tried: {possible_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Get raw session values (could be strings like '141208-1' or integers)
    raw_sessions = df[found_column].astype(str).values

    if num_trials is not None and len(raw_sessions) != num_trials:
        raise ValueError(f"Session count ({len(raw_sessions)}) must match trials ({num_trials})")

    # Map unique session names to integer indices
    unique_sessions = sorted(set(raw_sessions))
    session_to_idx = {name: idx for idx, name in enumerate(unique_sessions)}
    idx_to_session = {idx: name for name, idx in session_to_idx.items()}

    # Convert to integer indices
    session_ids = np.array([session_to_idx[s] for s in raw_sessions], dtype=np.int64)

    if _is_primary_rank():
        print(f"Found {len(unique_sessions)} unique sessions: {unique_sessions[:5]}{'...' if len(unique_sessions) > 5 else ''}")

    return session_ids, session_to_idx, idx_to_session


def load_or_create_session_splits(
    session_ids: np.ndarray,
    odors: np.ndarray,
    n_test_sessions: int = 1,
    n_val_sessions: int = 1,
    seed: int = 42,
    force_recreate: bool = False,
    idx_to_session: Optional[Dict[int, str]] = None,
    test_sessions: Optional[List[str]] = None,  # Explicit test session names
    val_sessions: Optional[List[str]] = None,   # Explicit val session names
    no_test_set: bool = False,  # If True, no test set - all held-out sessions are validation
    separate_val_sessions: bool = False,  # If True, return per-session val indices
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create train/val/test splits by holding out entire sessions.

    This is a more rigorous evaluation than random splitting because it tests
    generalization to completely new recording sessions.

    Args:
        session_ids: Array of integer session indices for each trial
        odors: Array of odor labels for each trial
        n_test_sessions: Number of sessions to hold out for testing (ignored if no_test_set=True)
        n_val_sessions: Number of sessions to hold out for validation
        seed: Random seed for reproducibility
        force_recreate: If True, recreate splits even if they exist
        idx_to_session: Optional mapping from integer index to original session name
        test_sessions: Explicit list of session names to use for test (overrides n_test_sessions)
        val_sessions: Explicit list of session names to use for val (overrides n_val_sessions)
        no_test_set: If True, no test set - all held-out sessions are used for validation
        separate_val_sessions: If True, return per-session validation indices as dict

    Returns:
        train_idx: Indices of training trials
        val_idx: Indices of validation trials (or dict if separate_val_sessions=True)
        test_idx: Indices of test trials (empty array if no_test_set=True)
        split_info: Dictionary with split metadata
    """
    # Check if using explicit session specification
    use_explicit_sessions = test_sessions is not None or val_sessions is not None

    # Check if splits already exist (skip if using explicit sessions)
    if (not force_recreate and not use_explicit_sessions and
        SESSION_TRAIN_SPLIT_PATH.exists() and
        SESSION_VAL_SPLIT_PATH.exists() and
        SESSION_TEST_SPLIT_PATH.exists() and
        SESSION_SPLIT_INFO_PATH.exists()):

        train_idx = np.load(SESSION_TRAIN_SPLIT_PATH)
        val_idx = np.load(SESSION_VAL_SPLIT_PATH)
        test_idx = np.load(SESSION_TEST_SPLIT_PATH)
        split_info = json.loads(SESSION_SPLIT_INFO_PATH.read_text())

        # VALIDATION: Check that loaded indices don't exceed data size
        max_idx = len(session_ids) - 1
        if train_idx.max() > max_idx or val_idx.max() > max_idx or test_idx.max() > max_idx:
            if _is_primary_rank():
                print(f"WARNING: Cached split indices exceed data size! Recreating splits...")
            # Don't return - fall through to recreate splits
        else:
            # Validate no overlap
            train_set = set(train_idx.tolist())
            val_set = set(val_idx.tolist())
            test_set = set(test_idx.tolist())
            has_overlap = bool(train_set & val_set) or bool(train_set & test_set) or bool(val_set & test_set)
            if has_overlap:
                if _is_primary_rank():
                    print(f"WARNING: Loaded splits have overlapping indices! Recreating splits...")
            else:
                # Verify sessions are separate
                train_sessions_check = set(session_ids[train_idx])
                val_sessions_check = set(session_ids[val_idx])
                test_sessions_check = set(session_ids[test_idx])

                session_overlap = (
                    bool(train_sessions_check & val_sessions_check) or
                    bool(train_sessions_check & test_sessions_check) or
                    bool(val_sessions_check & test_sessions_check)
                )
                if session_overlap and _is_primary_rank():
                    print(f"WARNING: Loaded splits have overlapping sessions! This shouldn't happen!")
                    print(f"  Train sessions: {train_sessions_check}")
                    print(f"  Val sessions: {val_sessions_check}")
                    print(f"  Test sessions: {test_sessions_check}")

                if _is_primary_rank():
                    print(f"Loaded existing session splits: {split_info['n_train_sessions']} train, "
                          f"{split_info['n_val_sessions']} val, {split_info['n_test_sessions']} test sessions")
                return train_idx, val_idx, test_idx, split_info

    rng = np.random.default_rng(seed)

    # Get unique sessions and their trial counts
    unique_sessions = np.unique(session_ids)
    n_sessions = len(unique_sessions)

    # Build reverse mapping from session name to index
    session_to_idx = {}
    if idx_to_session is not None:
        session_to_idx = {name: idx for idx, name in idx_to_session.items()}

    # Handle explicit session specification
    if use_explicit_sessions:
        # Validate and convert explicit session names to indices
        if test_sessions is not None:
            test_session_ids = set()
            for name in test_sessions:
                if name not in session_to_idx:
                    available = list(session_to_idx.keys())
                    raise ValueError(f"Test session '{name}' not found. Available: {available}")
                test_session_ids.add(session_to_idx[name])
        else:
            test_session_ids = set()

        if val_sessions is not None:
            val_session_ids = set()
            for name in val_sessions:
                if name not in session_to_idx:
                    available = list(session_to_idx.keys())
                    raise ValueError(f"Val session '{name}' not found. Available: {available}")
                val_session_ids.add(session_to_idx[name])
        else:
            val_session_ids = set()

        # Remaining sessions go to train
        all_session_ids = set(unique_sessions.tolist())
        train_session_ids = all_session_ids - test_session_ids - val_session_ids

        if len(train_session_ids) == 0:
            raise ValueError("No sessions left for training after specifying test/val sessions!")

        if _is_primary_rank():
            print(f"\n[Explicit Session Split]")
            print(f"  Test sessions: {test_sessions}")
            print(f"  Val sessions: {val_sessions}")
    else:
        # Random session selection (original behavior)
        # If no_test_set, all held-out sessions go to validation
        n_holdout = n_val_sessions if no_test_set else (n_test_sessions + n_val_sessions)
        if n_holdout >= n_sessions:
            raise ValueError(
                f"Cannot hold out {n_holdout} sessions "
                f"from only {n_sessions} total sessions. Need at least 1 for training."
            )

        # Shuffle sessions
        shuffled_sessions = unique_sessions.copy()
        rng.shuffle(shuffled_sessions)

        # Assign sessions to splits (these are integer indices)
        if no_test_set:
            # No test set - all held-out sessions are validation
            test_session_ids = set()
            val_session_ids = set(shuffled_sessions[:n_val_sessions].tolist())
            train_session_ids = set(shuffled_sessions[n_val_sessions:].tolist())
        else:
            test_session_ids = set(shuffled_sessions[:n_test_sessions].tolist())
            val_session_ids = set(shuffled_sessions[n_test_sessions:n_test_sessions + n_val_sessions].tolist())
            train_session_ids = set(shuffled_sessions[n_test_sessions + n_val_sessions:].tolist())

    # Create index arrays
    all_indices = np.arange(len(session_ids))
    train_idx = all_indices[np.isin(session_ids, list(train_session_ids))]
    test_idx = all_indices[np.isin(session_ids, list(test_session_ids))]

    # For validation, either combined or per-session
    if separate_val_sessions:
        # Create per-session validation indices (dict: session_name -> indices)
        val_idx_per_session = {}
        for sess_id in val_session_ids:
            sess_name = idx_to_session[sess_id] if idx_to_session is not None else str(sess_id)
            sess_indices = all_indices[session_ids == sess_id]
            val_idx_per_session[sess_name] = sess_indices
        # Also create combined val_idx for backward compatibility
        val_idx = all_indices[np.isin(session_ids, list(val_session_ids))]
    else:
        val_idx = all_indices[np.isin(session_ids, list(val_session_ids))]
        val_idx_per_session = None

    # CRITICAL VALIDATION: Ensure no overlap between splits
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())

    if train_set & val_set:
        raise ValueError(f"BUG: Train-Val overlap detected! {len(train_set & val_set)} overlapping indices")
    if train_set & test_set:
        raise ValueError(f"BUG: Train-Test overlap detected! {len(train_set & test_set)} overlapping indices")
    if val_set & test_set:
        raise ValueError(f"BUG: Val-Test overlap detected! {len(val_set & test_set)} overlapping indices")

    # Verify that sessions are truly separate
    train_sessions_check = set(session_ids[train_idx])
    val_sessions_check = set(session_ids[val_idx])
    test_sessions_check = set(session_ids[test_idx])

    if _is_primary_rank():
        if train_sessions_check & val_sessions_check:
            print(f"WARNING: Train and Val have overlapping sessions: {train_sessions_check & val_sessions_check}")
        if train_sessions_check & test_sessions_check:
            print(f"WARNING: Train and Test have overlapping sessions: {train_sessions_check & test_sessions_check}")
        if val_sessions_check & test_sessions_check:
            print(f"WARNING: Val and Test have overlapping sessions: {val_sessions_check & test_sessions_check}")

    # Shuffle within splits
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    # Convert integer indices back to original session names for display
    def ids_to_names(ids):
        if idx_to_session is not None:
            return sorted([idx_to_session[i] for i in ids])
        return sorted(ids)

    train_session_names = ids_to_names(train_session_ids)
    val_session_names = ids_to_names(val_session_ids)
    test_session_names = ids_to_names(test_session_ids)

    # Compute split statistics
    def get_odor_distribution(indices):
        odor_subset = odors[indices]
        unique, counts = np.unique(odor_subset, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    split_info = {
        "split_type": "session_holdout",
        "seed": seed,
        "n_total_sessions": n_sessions,
        "n_train_sessions": len(train_session_ids),
        "n_val_sessions": len(val_session_ids),
        "n_test_sessions": len(test_session_ids),
        "train_sessions": train_session_names,
        "val_sessions": val_session_names,
        "test_sessions": test_session_names,
        "n_train_trials": len(train_idx),
        "n_val_trials": len(val_idx),
        "n_test_trials": len(test_idx),
        "train_odor_distribution": get_odor_distribution(train_idx),
        "val_odor_distribution": get_odor_distribution(val_idx),
        "test_odor_distribution": get_odor_distribution(test_idx),
        "no_test_set": no_test_set,
        "separate_val_sessions": separate_val_sessions,
    }

    # Add per-session validation info
    if separate_val_sessions and val_idx_per_session is not None:
        split_info["val_sessions_detail"] = {
            sess_name: {
                "n_trials": len(indices),
                "odor_distribution": get_odor_distribution(indices)
            }
            for sess_name, indices in val_idx_per_session.items()
        }

    # Print summary (only on primary rank to avoid duplicate output)
    if _is_primary_rank():
        print(f"\n{'='*60}")
        if no_test_set:
            print("SESSION-BASED SPLIT (No Test Set - All Held-Out Sessions for Validation)")
        else:
            print("SESSION-BASED SPLIT (Held-Out Session Evaluation)")
        print(f"{'='*60}")
        print(f"Total sessions: {n_sessions}")
        print(f"Train sessions: {train_session_names} ({len(train_idx)} trials)")
        if separate_val_sessions and val_idx_per_session is not None:
            print(f"Val sessions (SEPARATE):")
            for sess_name, indices in val_idx_per_session.items():
                print(f"  - {sess_name}: {len(indices)} trials")
        else:
            print(f"Val sessions:   {val_session_names} ({len(val_idx)} trials)")
        if not no_test_set:
            print(f"Test sessions:  {test_session_names} ({len(test_idx)} trials)")
        print(f"{'='*60}\n")

    # Save splits (skip saving if using separate_val_sessions - it's not serializable)
    if not separate_val_sessions:
        SESSION_TRAIN_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(SESSION_TRAIN_SPLIT_PATH, train_idx)
        np.save(SESSION_VAL_SPLIT_PATH, val_idx)
        np.save(SESSION_TEST_SPLIT_PATH, test_idx)
        SESSION_SPLIT_INFO_PATH.write_text(json.dumps(split_info, indent=2))

    # Add per-session indices to split_info for return
    if separate_val_sessions and val_idx_per_session is not None:
        split_info["val_idx_per_session"] = val_idx_per_session

    return train_idx, val_idx, test_idx, split_info


def extract_window(signals: np.ndarray) -> np.ndarray:
    """Extract input window from full signals."""
    return signals[..., INPUT_START_IDX:INPUT_END_IDX]


def load_or_create_stratified_splits(
    odors: np.ndarray,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    force_balanced: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create or load stratified train/val/test splits.

    Splits are stratified by odor label to ensure balanced representation.

    Args:
        odors: Array of odor labels for each trial
        seed: Random seed for reproducibility
        train_ratio: Fraction of data for training (default: 0.7)
        val_ratio: Fraction of data for validation (default: 0.15)
        force_balanced: If True, ensures EQUAL samples per odor in each split.
                       If False, uses proportional stratification.
    """
    # Check if splits already exist
    if TRAIN_SPLIT_PATH.exists() and VAL_SPLIT_PATH.exists() and TEST_SPLIT_PATH.exists():
        train_idx = np.load(TRAIN_SPLIT_PATH)
        val_idx = np.load(VAL_SPLIT_PATH)
        test_idx = np.load(TEST_SPLIT_PATH)
        return train_idx, val_idx, test_idx

    rng = np.random.default_rng(seed)
    total_trials = odors.shape[0]
    all_indices = np.arange(total_trials)

    if force_balanced:
        # BALANCED SPLIT: Equal absolute samples per odor in each split
        unique_odors = np.unique(odors)
        n_odors = len(unique_odors)

        # Find minimum count across all odors
        min_count = min(np.sum(odors == oid) for oid in unique_odors)

        # Calculate how many samples per odor for each split
        n_per_odor_train = int(min_count * train_ratio)
        n_per_odor_val = int(min_count * val_ratio)
        n_per_odor_test = min_count - n_per_odor_train - n_per_odor_val

        # Ensure at least 1 sample per odor in test
        if n_per_odor_test < 1:
            n_per_odor_test = 1
            n_per_odor_val = max(1, n_per_odor_val)
            n_per_odor_train = min_count - n_per_odor_val - n_per_odor_test

        train_idx_list, val_idx_list, test_idx_list = [], [], []

        for odor_id in unique_odors:
            # Get all indices for this odor
            odor_mask = odors == odor_id
            odor_indices = all_indices[odor_mask].copy()

            # Shuffle indices for this odor
            rng.shuffle(odor_indices)

            # Take equal counts from each odor
            train_idx_list.append(odor_indices[:n_per_odor_train])
            val_idx_list.append(odor_indices[n_per_odor_train:n_per_odor_train + n_per_odor_val])
            test_idx_list.append(odor_indices[n_per_odor_train + n_per_odor_val:
                                              n_per_odor_train + n_per_odor_val + n_per_odor_test])

        train_idx = np.concatenate(train_idx_list)
        val_idx = np.concatenate(val_idx_list)
        test_idx = np.concatenate(test_idx_list)

        # Shuffle within each split to avoid odor ordering effects
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

        # Log the balanced split info
        if _is_primary_rank():
            print(f"Balanced split: {n_per_odor_train} train, {n_per_odor_val} val, "
                  f"{n_per_odor_test} test per odor (x{n_odors} odors)")
            print(f"Total: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    else:
        # PROPORTIONAL SPLIT: Maintains original class proportions
        # First split: train vs (val+test)
        test_size = 1.0 - train_ratio
        primary_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx_mask, temp_idx_mask = next(primary_split.split(all_indices.reshape(-1, 1), odors))
        train_idx = all_indices[train_idx_mask]
        temp_idx = all_indices[temp_idx_mask]

        # Second split: val vs test
        temp_labels = odors[temp_idx]
        val_fraction = val_ratio / (val_ratio + (1.0 - train_ratio - val_ratio))
        secondary_split = StratifiedShuffleSplit(n_splits=1, test_size=1-val_fraction, random_state=seed + 1)
        val_mask, test_mask = next(secondary_split.split(temp_idx.reshape(-1, 1), temp_labels))
        val_idx = temp_idx[val_mask]
        test_idx = temp_idx[test_mask]

    # Save splits
    TRAIN_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(TRAIN_SPLIT_PATH, train_idx)
    np.save(VAL_SPLIT_PATH, val_idx)
    np.save(TEST_SPLIT_PATH, test_idx)

    return train_idx, val_idx, test_idx


def load_or_create_pfc_stratified_splits(
    trial_types: np.ndarray,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    force_recreate: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create or load stratified train/val/test splits for PFC dataset.

    Splits are stratified by trial type (Right/Left) to ensure balanced representation.

    Args:
        trial_types: Array of trial type labels for each trial
        seed: Random seed for reproducibility
        train_ratio: Fraction of data for training (default: 0.7)
        val_ratio: Fraction of data for validation (default: 0.15)
        force_recreate: If True, recreate splits even if they exist
    """
    # Check if splits already exist
    if (not force_recreate and
        PFC_TRAIN_SPLIT_PATH.exists() and
        PFC_VAL_SPLIT_PATH.exists() and
        PFC_TEST_SPLIT_PATH.exists()):
        train_idx = np.load(PFC_TRAIN_SPLIT_PATH)
        val_idx = np.load(PFC_VAL_SPLIT_PATH)
        test_idx = np.load(PFC_TEST_SPLIT_PATH)
        if _is_primary_rank():
            print(f"Loaded existing PFC splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        return train_idx, val_idx, test_idx

    rng = np.random.default_rng(seed)
    total_trials = trial_types.shape[0]
    all_indices = np.arange(total_trials)

    # Use stratified splitting to maintain class balance
    # First split: train vs (val+test)
    test_size = 1.0 - train_ratio
    primary_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx_mask, temp_idx_mask = next(primary_split.split(all_indices.reshape(-1, 1), trial_types))
    train_idx = all_indices[train_idx_mask]
    temp_idx = all_indices[temp_idx_mask]

    # Second split: val vs test
    temp_labels = trial_types[temp_idx]
    val_fraction = val_ratio / (val_ratio + (1.0 - train_ratio - val_ratio))
    secondary_split = StratifiedShuffleSplit(n_splits=1, test_size=1-val_fraction, random_state=seed + 1)
    val_mask, test_mask = next(secondary_split.split(temp_idx.reshape(-1, 1), temp_labels))
    val_idx = temp_idx[val_mask]
    test_idx = temp_idx[test_mask]

    # Print split statistics
    if _is_primary_rank():
        print(f"\nPFC Dataset Splits:")
        print(f"  Train: {len(train_idx)} samples")
        for tt in np.unique(trial_types):
            count = np.sum(trial_types[train_idx] == tt)
            print(f"    - Type {tt}: {count}")
        print(f"  Val: {len(val_idx)} samples")
        print(f"  Test: {len(test_idx)} samples")

    # Save splits
    PFC_TRAIN_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(PFC_TRAIN_SPLIT_PATH, train_idx)
    np.save(PFC_VAL_SPLIT_PATH, val_idx)
    np.save(PFC_TEST_SPLIT_PATH, test_idx)

    return train_idx, val_idx, test_idx


def load_or_create_pfc_session_splits(
    session_ids: np.ndarray,
    trial_types: np.ndarray,
    n_test_sessions: int = 1,
    n_val_sessions: int = 1,
    seed: int = 42,
    force_recreate: bool = False,
    idx_to_session: Optional[Dict[int, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create train/val/test splits by holding out entire PFC recording sessions.

    Args:
        session_ids: Array of integer session indices for each trial
        trial_types: Array of trial type labels for each trial
        n_test_sessions: Number of sessions to hold out for testing
        n_val_sessions: Number of sessions to hold out for validation
        seed: Random seed for reproducibility
        force_recreate: If True, recreate splits even if they exist
        idx_to_session: Optional mapping from integer index to original session name

    Returns:
        train_idx, val_idx, test_idx: Split indices
        split_info: Dictionary with split metadata
    """
    # Session split paths for PFC
    pfc_session_train = _PFC_DATA_DIR / "session_train_indices.npy"
    pfc_session_val = _PFC_DATA_DIR / "session_val_indices.npy"
    pfc_session_test = _PFC_DATA_DIR / "session_test_indices.npy"
    pfc_session_info = _PFC_DATA_DIR / "session_split_info.json"

    if (not force_recreate and
        pfc_session_train.exists() and
        pfc_session_val.exists() and
        pfc_session_test.exists() and
        pfc_session_info.exists()):

        train_idx = np.load(pfc_session_train)
        val_idx = np.load(pfc_session_val)
        test_idx = np.load(pfc_session_test)
        split_info = json.loads(pfc_session_info.read_text())
        if _is_primary_rank():
            print(f"Loaded existing PFC session splits: {split_info['n_train_sessions']} train, "
                  f"{split_info['n_val_sessions']} val, {split_info['n_test_sessions']} test sessions")
        return train_idx, val_idx, test_idx, split_info

    rng = np.random.default_rng(seed)
    unique_sessions = np.unique(session_ids)
    n_sessions = len(unique_sessions)

    if n_test_sessions + n_val_sessions >= n_sessions:
        raise ValueError(
            f"Cannot hold out {n_test_sessions} test + {n_val_sessions} val sessions "
            f"from only {n_sessions} total sessions."
        )

    # Shuffle sessions
    shuffled_sessions = unique_sessions.copy()
    rng.shuffle(shuffled_sessions)

    # Assign sessions to splits
    test_session_ids = set(shuffled_sessions[:n_test_sessions].tolist())
    val_session_ids = set(shuffled_sessions[n_test_sessions:n_test_sessions + n_val_sessions].tolist())
    train_session_ids = set(shuffled_sessions[n_test_sessions + n_val_sessions:].tolist())

    # Create index arrays
    all_indices = np.arange(len(session_ids))
    train_idx = all_indices[np.isin(session_ids, list(train_session_ids))]
    val_idx = all_indices[np.isin(session_ids, list(val_session_ids))]
    test_idx = all_indices[np.isin(session_ids, list(test_session_ids))]

    # Shuffle within splits
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    # Convert to session names for display
    def ids_to_names(ids):
        if idx_to_session is not None:
            return sorted([idx_to_session[i] for i in ids])
        return sorted(ids)

    split_info = {
        "split_type": "pfc_session_holdout",
        "seed": seed,
        "n_total_sessions": n_sessions,
        "n_train_sessions": len(train_session_ids),
        "n_val_sessions": len(val_session_ids),
        "n_test_sessions": len(test_session_ids),
        "train_sessions": ids_to_names(train_session_ids),
        "val_sessions": ids_to_names(val_session_ids),
        "test_sessions": ids_to_names(test_session_ids),
        "n_train_trials": len(train_idx),
        "n_val_trials": len(val_idx),
        "n_test_trials": len(test_idx),
    }

    if _is_primary_rank():
        print(f"\nPFC SESSION-BASED SPLIT:")
        print(f"  Train sessions: {split_info['train_sessions']} ({len(train_idx)} trials)")
        print(f"  Val sessions: {split_info['val_sessions']} ({len(val_idx)} trials)")
        print(f"  Test sessions: {split_info['test_sessions']} ({len(test_idx)} trials)")

    # Save
    pfc_session_train.parent.mkdir(parents=True, exist_ok=True)
    np.save(pfc_session_train, train_idx)
    np.save(pfc_session_val, val_idx)
    np.save(pfc_session_test, test_idx)
    pfc_session_info.write_text(json.dumps(split_info, indent=2))

    return train_idx, val_idx, test_idx, split_info


def compute_normalization(windowed: np.ndarray, train_idx: np.ndarray) -> NormalizationStats:
    """Compute normalization statistics from training set only.
    
    Works for both dataset formats:
    - Olfactory: (trials, 2, channels, time) - normalizes over axis (0, 3)
    - PFC: (trials, channels, time) - normalizes over axis (0, 2)
    """
    subset = windowed[train_idx]
    if subset.ndim == 4:
        # Olfactory format: (trials, 2, channels, time)
        mean = subset.mean(axis=(0, 3), keepdims=True).astype(np.float32)
        std = subset.std(axis=(0, 3), keepdims=True).astype(np.float32)
    elif subset.ndim == 3:
        # PFC format: (trials, channels, time)
        mean = subset.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        std = subset.std(axis=(0, 2), keepdims=True).astype(np.float32)
    else:
        raise ValueError(f"Unexpected array shape: {subset.shape}")
    
    std[std < 1e-6] = 1e-6
    return NormalizationStats(mean=mean, std=std)


def normalize(windowed: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Apply z-score normalization."""
    normalized = (windowed - stats.mean) / stats.std
    return normalized.astype(np.float32)


def denormalize(normalized: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Reverse z-score normalization."""
    return normalized * stats.std + stats.mean


# =============================================================================
# Per-Segment Normalization (for temporal ablation validation)
# =============================================================================

@dataclass
class PerSegmentNormStats:
    """Per-segment normalization statistics for temporal ablation validation.

    This addresses the potential data leakage concern where global normalization
    (across all time points) might confound temporal ablation experiments.

    Segments:
        - pre_onset: 0-2000ms (baseline period)
        - onset: 2000-2500ms (immediate response)
        - early: 2500-3000ms (early response)
        - sustained: 3000-4000ms (sustained response)
        - late: 4000-5000ms (late response)
    """
    # Shape: (1, 2, channels, 1) for each segment
    pre_onset_mean: np.ndarray
    pre_onset_std: np.ndarray
    onset_mean: np.ndarray
    onset_std: np.ndarray
    early_mean: np.ndarray
    early_std: np.ndarray
    sustained_mean: np.ndarray
    sustained_std: np.ndarray
    late_mean: np.ndarray
    late_std: np.ndarray

    # Segment boundaries
    SEGMENTS = {
        "pre_onset": (0, 2000),
        "onset": (2000, 2500),
        "early": (2500, 3000),
        "sustained": (3000, 4000),
        "late": (4000, 5000),
    }

    def get_segment_stats(self, segment_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get mean/std for a specific segment."""
        return (
            getattr(self, f"{segment_name}_mean"),
            getattr(self, f"{segment_name}_std"),
        )


def compute_per_segment_normalization(
    windowed: np.ndarray,
    train_idx: np.ndarray
) -> PerSegmentNormStats:
    """Compute normalization statistics per temporal segment from training set only.

    This allows validation of temporal ablation experiments by normalizing each
    segment independently, avoiding cross-segment statistical information leakage.

    Args:
        windowed: Full data array, shape (trials, 2, channels, time)
        train_idx: Indices of training samples

    Returns:
        PerSegmentNormStats with per-segment mean/std
    """
    subset = windowed[train_idx]  # (n_train, 2, channels, 5000)

    stats_dict = {}
    for seg_name, (start, end) in PerSegmentNormStats.SEGMENTS.items():
        seg_data = subset[..., start:end]  # (n_train, 2, channels, seg_len)
        # Compute mean/std across trials and time within this segment
        seg_mean = seg_data.mean(axis=(0, 3), keepdims=True).astype(np.float32)
        seg_std = seg_data.std(axis=(0, 3), keepdims=True).astype(np.float32)
        seg_std[seg_std < 1e-6] = 1e-6
        stats_dict[f"{seg_name}_mean"] = seg_mean
        stats_dict[f"{seg_name}_std"] = seg_std

    return PerSegmentNormStats(**stats_dict)


def normalize_per_segment(
    windowed: np.ndarray,
    stats: PerSegmentNormStats
) -> np.ndarray:
    """Apply per-segment z-score normalization.

    Each temporal segment is normalized using only statistics computed
    from that segment, preventing cross-segment information leakage.
    """
    normalized = np.zeros_like(windowed, dtype=np.float32)

    for seg_name, (start, end) in PerSegmentNormStats.SEGMENTS.items():
        mean, std = stats.get_segment_stats(seg_name)
        normalized[..., start:end] = (windowed[..., start:end] - mean) / std

    return normalized


def validate_normalization_impact(
    windowed: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, Any]:
    """Validate that global vs per-segment normalization doesn't confound ablation.

    Computes statistics to check if there's significant difference in per-segment
    distributions that could bias temporal ablation experiments.

    Returns:
        Dictionary with per-segment statistics comparing global vs segment-specific
        normalization. Key metrics:
        - segment_mean_diff: How much segment means differ from global mean
        - segment_std_ratio: Ratio of segment std to global std
        - normalization_bias: Potential bias introduced by global normalization
    """
    # Global normalization stats
    global_stats = compute_normalization(windowed, train_idx)

    # Per-segment stats
    segment_stats = compute_per_segment_normalization(windowed, train_idx)

    results = {
        "segments": {},
        "recommendation": "",
    }

    max_mean_diff = 0.0
    max_std_ratio_deviation = 0.0

    for seg_name, (start, end) in PerSegmentNormStats.SEGMENTS.items():
        seg_mean, seg_std = segment_stats.get_segment_stats(seg_name)

        # How different is segment mean from global mean?
        mean_diff = np.abs(seg_mean - global_stats.mean).mean()

        # How different is segment std from global std?
        std_ratio = (seg_std / global_stats.std).mean()

        results["segments"][seg_name] = {
            "mean_diff_from_global": float(mean_diff),
            "std_ratio_to_global": float(std_ratio),
            "segment_duration_ms": end - start,
        }

        max_mean_diff = max(max_mean_diff, mean_diff)
        max_std_ratio_deviation = max(max_std_ratio_deviation, abs(1.0 - std_ratio))

    # Provide recommendation
    if max_mean_diff < 0.1 and max_std_ratio_deviation < 0.1:
        results["recommendation"] = (
            "LOW RISK: Segment statistics are similar to global statistics. "
            "Global normalization is unlikely to confound temporal ablation results."
        )
        results["risk_level"] = "low"
    elif max_mean_diff < 0.3 and max_std_ratio_deviation < 0.3:
        results["recommendation"] = (
            "MODERATE RISK: Some segments differ from global statistics. "
            "Consider reporting results with both normalization schemes for validation."
        )
        results["risk_level"] = "moderate"
    else:
        results["recommendation"] = (
            "HIGH RISK: Significant differences between segment and global statistics. "
            "Temporal ablation results should be validated with per-segment normalization."
        )
        results["risk_level"] = "high"

    results["max_mean_diff"] = float(max_mean_diff)
    results["max_std_ratio_deviation"] = float(max_std_ratio_deviation)

    return results


# =============================================================================
# Temporal Ablation
# =============================================================================

class TemporalAblation:
    """Temporal masking for ablation studies.

    Supports various masking patterns:
    - pre_only: Keep only pre-onset baseline
    - post_only: Keep only post-onset response
    - pre_N: Remove N ms before onset
    - post_N: Remove N ms after onset
    - baseline_only, onset_only, early_only, sustained_only, late_only
    - no_pre_buffer, no_post_buffer, no_buffers, large_buffers
    """

    ONSET_IDX = int(2.0 * SAMPLING_RATE_HZ)  # Odor onset at 2s

    ABLATION_CONFIGS = {
        # Pre-onset patterns
        "pre_only": {"keep_start": 0, "keep_end": ONSET_IDX},
        "pre_500": {"mask_start": ONSET_IDX - 500, "mask_end": ONSET_IDX},
        "pre_1000": {"mask_start": ONSET_IDX - 1000, "mask_end": ONSET_IDX},
        "pre_1500": {"mask_start": ONSET_IDX - 1500, "mask_end": ONSET_IDX},
        "pre_2000": {"mask_start": 0, "mask_end": ONSET_IDX},

        # Post-onset patterns
        "post_only": {"keep_start": ONSET_IDX, "keep_end": INPUT_WINDOW},
        "post_500": {"mask_start": ONSET_IDX, "mask_end": ONSET_IDX + 500},
        "post_1000": {"mask_start": ONSET_IDX, "mask_end": ONSET_IDX + 1000},
        "post_1500": {"mask_start": ONSET_IDX, "mask_end": ONSET_IDX + 1500},
        "post_2000": {"mask_start": ONSET_IDX, "mask_end": ONSET_IDX + 2000},

        # Window isolation patterns
        "baseline_only": {"keep_start": 0, "keep_end": 2000},
        "onset_only": {"keep_start": 2000, "keep_end": 2500},
        "early_only": {"keep_start": 2500, "keep_end": 3000},
        "sustained_only": {"keep_start": 3000, "keep_end": 4000},
        "late_only": {"keep_start": 4000, "keep_end": INPUT_WINDOW},

        # Buffer patterns
        "no_pre_buffer": {"mask_start": 0, "mask_end": 500},
        "no_post_buffer": {"mask_start": 4500, "mask_end": INPUT_WINDOW},
        "no_buffers": {"mask_regions": [(0, 500), (4500, INPUT_WINDOW)]},
        "large_buffers": {"keep_start": 1000, "keep_end": 4000},
    }

    def __init__(self, ablation_type: str):
        self.ablation_type = ablation_type
        if ablation_type not in self.ABLATION_CONFIGS:
            raise ValueError(f"Unknown ablation type: {ablation_type}. "
                           f"Available: {list(self.ABLATION_CONFIGS.keys())}")
        self.config = self.ABLATION_CONFIGS[ablation_type]

    def apply_mask(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply temporal mask to signal. Masked regions are set to 0."""
        masked = signal.clone()

        if "keep_start" in self.config and "keep_end" in self.config:
            # Zero everything outside the keep window
            keep_start = self.config["keep_start"]
            keep_end = self.config["keep_end"]
            if keep_start > 0:
                masked[..., :keep_start] = 0
            if keep_end < signal.shape[-1]:
                masked[..., keep_end:] = 0

        elif "mask_start" in self.config and "mask_end" in self.config:
            # Zero the mask window
            mask_start = self.config["mask_start"]
            mask_end = min(self.config["mask_end"], signal.shape[-1])
            masked[..., mask_start:mask_end] = 0

        elif "mask_regions" in self.config:
            # Zero multiple regions
            for start, end in self.config["mask_regions"]:
                end = min(end, signal.shape[-1])
                masked[..., start:end] = 0

        return masked


# =============================================================================
# PyTorch Datasets
# =============================================================================

class PairedNeuralDataset(Dataset):
    """Generic dataset for paired neural signal translation.
    
    Works for both datasets:
    - Olfactory: OB -> PCx translation with odor conditioning
    - PFC/HPC: PFC -> CA1 translation with trial type conditioning
    
    Args:
        source: Source region signals [n_samples, channels, time]
        target: Target region signals [n_samples, channels, time]
        labels: Condition labels array [n_samples]
        indices: Sample indices to use from the arrays
        filter_label_id: If provided, only keep samples with this label ID
        temporal_ablation: TemporalAblation instance to apply masking
        data_fraction: Fraction of data to use (for data scaling experiments)
    """
    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        filter_label_id: Optional[int] = None,
        temporal_ablation: Optional[TemporalAblation] = None,
        data_fraction: float = 1.0,
    ):
        # Apply label filtering first if requested
        if filter_label_id is not None:
            label_mask = labels[indices] == filter_label_id
            indices = indices[label_mask]

        # Apply data subsampling if requested
        if data_fraction < 1.0:
            n_samples = len(indices)
            n_keep = max(1, int(n_samples * data_fraction))
            rng = np.random.RandomState(42)
            keep_idx = rng.choice(n_samples, size=n_keep, replace=False)
            keep_idx = np.sort(keep_idx)
            indices = indices[keep_idx]

        self.source = torch.from_numpy(source[indices]).float()
        self.target = torch.from_numpy(target[indices]).float()
        self.labels = torch.from_numpy(labels[indices]).long()
        self.temporal_ablation = temporal_ablation

    def __len__(self) -> int:
        return self.source.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = self.source[idx]
        tgt = self.target[idx]
        label = self.labels[idx]

        if self.temporal_ablation is not None:
            src = self.temporal_ablation.apply_mask(src)
            tgt = self.temporal_ablation.apply_mask(tgt)

        return src, tgt, label


class PairedConditionalDataset(Dataset):
    """Dataset for paired OB-PCx signals with odor conditioning.

    Args:
        ob: OB signals array [n_samples, channels, time]
        pcx: PCx signals array [n_samples, channels, time]
        odors: Odor labels array [n_samples]
        indices: Sample indices to use from the arrays
        filter_odor_id: If provided, only keep samples with this odor ID
        temporal_ablation: TemporalAblation instance to apply masking
        data_fraction: Fraction of data to use (for data scaling experiments)
        session_ids: Optional session IDs for each sample (for session-aware augmentation)
    """
    def __init__(
        self,
        ob: np.ndarray,
        pcx: np.ndarray,
        odors: np.ndarray,
        indices: np.ndarray,
        filter_odor_id: Optional[int] = None,
        temporal_ablation: Optional[TemporalAblation] = None,
        data_fraction: float = 1.0,
        session_ids: Optional[np.ndarray] = None,
    ):
        # Apply odor filtering first if requested
        if filter_odor_id is not None:
            odor_mask = odors[indices] == filter_odor_id
            indices = indices[odor_mask]

        # Apply data subsampling if requested
        if data_fraction < 1.0:
            n_samples = len(indices)
            n_keep = max(1, int(n_samples * data_fraction))
            rng = np.random.RandomState(42)
            keep_idx = rng.choice(n_samples, size=n_keep, replace=False)
            keep_idx = np.sort(keep_idx)
            indices = indices[keep_idx]

        self.ob = torch.from_numpy(ob[indices]).float()
        self.pcx = torch.from_numpy(pcx[indices]).float()
        self.odors = torch.from_numpy(odors[indices]).long()
        self.temporal_ablation = temporal_ablation

        # Session IDs for session-aware augmentation
        if session_ids is not None:
            self.session_ids = torch.from_numpy(session_ids[indices]).long()
        else:
            self.session_ids = None

    def __len__(self) -> int:
        return self.ob.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ob = self.ob[idx]
        pcx = self.pcx[idx]
        odor = self.odors[idx]

        if self.temporal_ablation is not None:
            ob = self.temporal_ablation.apply_mask(ob)
            pcx = self.temporal_ablation.apply_mask(pcx)

        # Return session_id if available (for session-aware augmentation)
        if self.session_ids is not None:
            return ob, pcx, odor, self.session_ids[idx]
        return ob, pcx, odor


class UnpairedDataset(Dataset):
    """Dataset for unpaired translation (CycleGAN style).

    Samples OB and PCx signals independently without pairing.
    """
    def __init__(
        self,
        signals: np.ndarray,  # [n_samples, 2, channels, time]
        indices: np.ndarray,
        domain: int = 0,  # 0=OB, 1=PCx
    ):
        self.data = torch.from_numpy(signals[indices, domain]).float()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class MultiDomainDataset(Dataset):
    """Dataset for multi-domain translation (StarGAN style).

    Returns signals with their domain (odor) label for domain-to-domain translation.
    """
    def __init__(
        self,
        signals: np.ndarray,  # [n_samples, 2, channels, time]
        odors: np.ndarray,
        indices: np.ndarray,
        region: int = 0,  # 0=OB, 1=PCx
    ):
        self.data = torch.from_numpy(signals[indices, region]).float()
        self.odors = torch.from_numpy(odors[indices]).long()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.odors[idx]


# =============================================================================
# Covariance Augmentation for Cross-Session Robustness
# =============================================================================

def _matrix_sqrt(A: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Compute matrix square root (or inverse square root) via eigendecomposition.

    Args:
        A: Symmetric positive definite matrix [C, C]
        inverse: If True, compute A^(-1/2), else A^(1/2)

    Returns:
        Matrix square root [C, C]
    """
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 1e-10)  # Numerical stability

    if inverse:
        w_sqrt = 1.0 / np.sqrt(w)
    else:
        w_sqrt = np.sqrt(w)

    return V @ np.diag(w_sqrt) @ V.T


def compute_covariance(data: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
    """Compute regularized covariance matrix.

    Args:
        data: Data array [trials, channels, samples] or [channels, samples]
        regularization: Tikhonov regularization for numerical stability

    Returns:
        Covariance matrix [channels, channels]
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    trials, n_channels, n_samples = data.shape

    # Concatenate all trials: [channels, trials * samples]
    data_concat = data.transpose(1, 0, 2).reshape(n_channels, -1)

    # Center the data
    data_centered = data_concat - data_concat.mean(axis=1, keepdims=True)

    # Compute covariance
    n_total = data_centered.shape[1]
    cov = (data_centered @ data_centered.T) / n_total

    # Regularize
    cov += regularization * np.eye(n_channels)

    return cov


def perturb_covariance(
    cov: np.ndarray,
    strength: float = 0.3,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Perturb a covariance matrix to simulate session variability.

    Args:
        cov: Original covariance matrix [C, C]
        strength: Perturbation strength (0 = no change, 1 = large change)
        rng: Random number generator

    Returns:
        Perturbed covariance matrix [C, C]
    """
    if rng is None:
        rng = np.random.default_rng()

    n_channels = cov.shape[0]

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    # Perturb eigenvalues (log-normal perturbation to keep positive)
    log_eigenvalues = np.log(eigenvalues)
    perturbation = rng.normal(0, strength, size=n_channels)
    perturbed_log_eigenvalues = log_eigenvalues + perturbation
    perturbed_eigenvalues = np.exp(perturbed_log_eigenvalues)

    # Small rotation of eigenvectors (orthogonal perturbation)
    # Generate antisymmetric matrix and exponentiate for orthogonal matrix
    A = rng.normal(0, strength * 0.1, size=(n_channels, n_channels))
    A = (A - A.T) / 2  # Make antisymmetric

    # Approximate matrix exponential for small perturbations: exp(A) ≈ I + A
    rotation = np.eye(n_channels) + A
    # Re-orthogonalize via QR
    rotation, _ = np.linalg.qr(rotation)

    # Reconstruct perturbed covariance
    perturbed_eigenvectors = eigenvectors @ rotation
    perturbed_cov = perturbed_eigenvectors @ np.diag(perturbed_eigenvalues) @ perturbed_eigenvectors.T

    # Ensure symmetry
    perturbed_cov = (perturbed_cov + perturbed_cov.T) / 2

    return perturbed_cov


def generate_synthetic_session(
    data: np.ndarray,
    original_cov: np.ndarray,
    target_cov: np.ndarray,
) -> np.ndarray:
    """Transform data from original covariance to target covariance.

    Formula: X_new = Σ_target^(1/2) @ Σ_original^(-1/2) @ X

    Args:
        data: Original data [trials, channels, samples]
        original_cov: Original covariance [C, C]
        target_cov: Target covariance [C, C]

    Returns:
        Transformed data [trials, channels, samples]
    """
    # Compute transformation matrix
    original_inv_sqrt = _matrix_sqrt(original_cov, inverse=True)
    target_sqrt = _matrix_sqrt(target_cov, inverse=False)
    transform = target_sqrt @ original_inv_sqrt

    # Apply transformation
    # data: [trials, channels, samples]
    # transform: [channels, channels]
    transformed = np.einsum('ij,njt->nit', transform, data)

    return transformed


def apply_covariance_augmentation(
    source: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    strength: float = 0.3,
    n_synthetic: int = 3,
    seed: int = 42,
    verbose: bool = True,
    session_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Apply covariance augmentation to generate synthetic training sessions.

    For each training sample, generates n_synthetic augmented versions by
    perturbing the covariance structure, simulating cross-session variability.

    Args:
        source: Source signal array [trials, channels, samples]
        target: Target signal array [trials, channels, samples]
        labels: Label array [trials]
        train_idx: Indices of training samples
        strength: Perturbation strength (0-1)
        n_synthetic: Number of synthetic sessions per real session
        seed: Random seed
        verbose: Print progress
        session_ids: Optional session ID array [trials] to expand alongside data

    Returns:
        Tuple of (aug_source, aug_target, aug_labels, aug_train_idx, aug_session_ids)
        Arrays are expanded with synthetic samples appended
    """
    rng = np.random.default_rng(seed)

    if verbose and _is_primary_rank():
        print(f"\n{'='*60}")
        print("Applying Covariance Augmentation")
        print(f"{'='*60}")
        print(f"  Strength: {strength}")
        print(f"  Synthetic sessions per real: {n_synthetic}")

    # Get training data
    train_source = source[train_idx]
    train_target = target[train_idx]
    train_labels = labels[train_idx]
    train_session_ids = session_ids[train_idx] if session_ids is not None else None

    # Compute original covariances from training data
    cov_source = compute_covariance(train_source)
    cov_target = compute_covariance(train_target)

    if verbose and _is_primary_rank():
        print(f"  Source cov trace: {np.trace(cov_source):.2f}")
        print(f"  Target cov trace: {np.trace(cov_target):.2f}")

    # Generate synthetic sessions
    synthetic_sources = []
    synthetic_targets = []
    synthetic_labels = []
    synthetic_session_ids = []

    for i in range(n_synthetic):
        # Perturb covariances
        perturbed_cov_source = perturb_covariance(cov_source, strength, rng)
        perturbed_cov_target = perturb_covariance(cov_target, strength, rng)

        # Generate synthetic data
        syn_source = generate_synthetic_session(train_source, cov_source, perturbed_cov_source)
        syn_target = generate_synthetic_session(train_target, cov_target, perturbed_cov_target)

        synthetic_sources.append(syn_source)
        synthetic_targets.append(syn_target)
        synthetic_labels.append(train_labels.copy())
        if train_session_ids is not None:
            synthetic_session_ids.append(train_session_ids.copy())

        if verbose and _is_primary_rank():
            print(f"  Generated synthetic session {i+1}/{n_synthetic}")

    # Concatenate original + synthetic
    aug_source = np.concatenate([source] + synthetic_sources, axis=0)
    aug_target = np.concatenate([target] + synthetic_targets, axis=0)
    aug_labels = np.concatenate([labels] + synthetic_labels, axis=0)

    # Concatenate session_ids if provided
    if session_ids is not None and len(synthetic_session_ids) > 0:
        aug_session_ids = np.concatenate([session_ids] + synthetic_session_ids, axis=0)
    else:
        aug_session_ids = session_ids

    # Update train indices to include synthetic samples
    n_original = len(source)
    n_train = len(train_idx)

    # New train indices: original train_idx + all synthetic indices
    synthetic_indices = []
    for i in range(n_synthetic):
        start_idx = n_original + i * n_train
        synthetic_indices.extend(range(start_idx, start_idx + n_train))

    aug_train_idx = np.concatenate([train_idx, np.array(synthetic_indices)])

    if verbose and _is_primary_rank():
        print(f"  Original training samples: {n_train}")
        print(f"  Total training samples: {len(aug_train_idx)} ({n_synthetic + 1}x)")
        print(f"{'='*60}\n")

    return aug_source, aug_target, aug_labels, aug_train_idx, aug_session_ids


# =============================================================================
# Dataset Expansion Augmentations
# =============================================================================

def expand_session_mixup(
    source: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    session_ids: np.ndarray,
    n_samples_per_pair: int = 2,
    alpha: float = 0.5,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-generate cross-session mixup samples to expand the dataset.

    For each pair of different sessions, generates interpolated samples.
    This expands the training set with synthetic cross-session data.

    Args:
        source: Source signal array [trials, channels, samples]
        target: Target signal array [trials, channels, samples]
        labels: Label array [trials]
        train_idx: Indices of training samples
        session_ids: Session ID array [trials]
        n_samples_per_pair: Number of samples to generate per session pair
        alpha: Beta distribution alpha for mixup weights
        seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (aug_source, aug_target, aug_labels, aug_train_idx, aug_session_ids)
    """
    rng = np.random.default_rng(seed)

    if verbose and _is_primary_rank():
        print(f"\n{'='*60}")
        print("Applying Session Mixup Dataset Expansion")
        print(f"{'='*60}")
        print(f"  Samples per session pair: {n_samples_per_pair}")
        print(f"  Mixup alpha: {alpha}")

    # Get unique sessions in training data
    train_source = source[train_idx]
    train_target = target[train_idx]
    train_labels = labels[train_idx]
    train_sessions = session_ids[train_idx]

    unique_sessions = np.unique(train_sessions)
    n_sessions = len(unique_sessions)

    if verbose and _is_primary_rank():
        print(f"  Unique training sessions: {n_sessions}")

    # Generate cross-session mixup samples
    synthetic_sources = []
    synthetic_targets = []
    synthetic_labels = []
    synthetic_session_ids = []

    # For each pair of different sessions
    session_pairs = [(i, j) for i in range(n_sessions) for j in range(i+1, n_sessions)]

    for sess_i, sess_j in session_pairs:
        sess_id_i = unique_sessions[sess_i]
        sess_id_j = unique_sessions[sess_j]

        # Get samples from each session
        mask_i = train_sessions == sess_id_i
        mask_j = train_sessions == sess_id_j

        samples_i = train_source[mask_i]
        samples_j = train_source[mask_j]
        targets_i = train_target[mask_i]
        targets_j = train_target[mask_j]
        labels_i = train_labels[mask_i]
        labels_j = train_labels[mask_j]

        # Generate n_samples_per_pair mixup samples
        for _ in range(n_samples_per_pair):
            # Random sample from each session
            idx_i = rng.integers(0, len(samples_i))
            idx_j = rng.integers(0, len(samples_j))

            # Mixup weight from beta distribution
            lam = rng.beta(alpha, alpha)

            # Create mixed sample
            mixed_source = lam * samples_i[idx_i] + (1 - lam) * samples_j[idx_j]
            mixed_target = lam * targets_i[idx_i] + (1 - lam) * targets_j[idx_j]

            # Use label from dominant sample
            mixed_label = labels_i[idx_i] if lam > 0.5 else labels_j[idx_j]

            # Assign to new synthetic session ID (use max + pair index)
            max_session_id = np.max(session_ids)
            synthetic_session_id = max_session_id + sess_i * n_sessions + sess_j + 1

            synthetic_sources.append(mixed_source)
            synthetic_targets.append(mixed_target)
            synthetic_labels.append(mixed_label)
            synthetic_session_ids.append(synthetic_session_id)

    n_synthetic = len(synthetic_sources)

    if n_synthetic > 0:
        # Stack synthetic samples
        syn_source = np.stack(synthetic_sources, axis=0)
        syn_target = np.stack(synthetic_targets, axis=0)
        syn_labels = np.array(synthetic_labels)
        syn_session_ids = np.array(synthetic_session_ids)

        # Concatenate with original
        aug_source = np.concatenate([source, syn_source], axis=0)
        aug_target = np.concatenate([target, syn_target], axis=0)
        aug_labels = np.concatenate([labels, syn_labels], axis=0)
        aug_session_ids = np.concatenate([session_ids, syn_session_ids], axis=0)

        # Update train indices
        n_original = len(source)
        new_indices = np.arange(n_original, n_original + n_synthetic)
        aug_train_idx = np.concatenate([train_idx, new_indices])
    else:
        aug_source = source
        aug_target = target
        aug_labels = labels
        aug_session_ids = session_ids
        aug_train_idx = train_idx

    if verbose and _is_primary_rank():
        print(f"  Session pairs: {len(session_pairs)}")
        print(f"  Generated synthetic samples: {n_synthetic}")
        print(f"  Original training samples: {len(train_idx)}")
        print(f"  Total training samples: {len(aug_train_idx)}")
        print(f"{'='*60}\n")

    return aug_source, aug_target, aug_labels, aug_train_idx, aug_session_ids


def expand_geodesic_warp(
    source: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    session_ids: np.ndarray,
    n_samples_per_source: int = 2,
    t_range: Tuple[float, float] = (0.3, 0.7),
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-generate geodesic warp samples to expand the dataset.

    Generates samples along geodesic paths on the SPD manifold between
    different session covariance structures.

    Args:
        source: Source signal array [trials, channels, samples]
        target: Target signal array [trials, channels, samples]
        labels: Label array [trials]
        train_idx: Indices of training samples
        session_ids: Session ID array [trials]
        n_samples_per_source: Number of geodesic samples per source sample
        t_range: Range of interpolation parameter (0=source, 1=target session)
        seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (aug_source, aug_target, aug_labels, aug_train_idx, aug_session_ids)
    """
    rng = np.random.default_rng(seed)

    if verbose and _is_primary_rank():
        print(f"\n{'='*60}")
        print("Applying Geodesic Warp Dataset Expansion")
        print(f"{'='*60}")
        print(f"  Samples per source: {n_samples_per_source}")
        print(f"  Interpolation range: {t_range}")

    # Get training data
    train_source = source[train_idx]
    train_target = target[train_idx]
    train_labels = labels[train_idx]
    train_sessions = session_ids[train_idx]

    unique_sessions = np.unique(train_sessions)
    n_sessions = len(unique_sessions)

    if n_sessions < 2:
        if verbose and _is_primary_rank():
            print("  WARNING: Need at least 2 sessions for geodesic warp, skipping")
            print(f"{'='*60}\n")
        return source, target, labels, train_idx, session_ids

    if verbose and _is_primary_rank():
        print(f"  Unique training sessions: {n_sessions}")

    # Compute per-session covariances
    session_covs_source = {}
    session_covs_target = {}

    for sess_id in unique_sessions:
        mask = train_sessions == sess_id
        session_covs_source[sess_id] = compute_covariance(train_source[mask])
        session_covs_target[sess_id] = compute_covariance(train_target[mask])

    # Generate geodesic warp samples
    synthetic_sources = []
    synthetic_targets = []
    synthetic_labels = []
    synthetic_session_ids = []

    for i, idx in enumerate(train_idx):
        sess_id = session_ids[idx]
        src = source[idx]
        tgt = target[idx]
        lbl = labels[idx]

        # Pick a random different session
        other_sessions = [s for s in unique_sessions if s != sess_id]
        if len(other_sessions) == 0:
            continue

        for _ in range(n_samples_per_source):
            other_sess = rng.choice(other_sessions)

            # Interpolation parameter
            t = rng.uniform(t_range[0], t_range[1])

            # Compute geodesic interpolation of covariances
            cov_src_source = session_covs_source[sess_id]
            cov_src_target = session_covs_target[sess_id]
            cov_other_source = session_covs_source[other_sess]
            cov_other_target = session_covs_target[other_sess]

            # Interpolate covariance (simple linear for now - proper geodesic is complex)
            # A full geodesic would use SPD manifold exponential/log maps
            interp_cov_source = (1 - t) * cov_src_source + t * cov_other_source
            interp_cov_target = (1 - t) * cov_src_target + t * cov_other_target

            # Apply whitening-coloring transform
            # whiten with source session, color with interpolated covariance
            try:
                # Regularize covariances for numerical stability
                eps = 1e-6
                cov_src_source_reg = cov_src_source + eps * np.eye(cov_src_source.shape[0])
                cov_src_target_reg = cov_src_target + eps * np.eye(cov_src_target.shape[0])
                interp_cov_source_reg = interp_cov_source + eps * np.eye(interp_cov_source.shape[0])
                interp_cov_target_reg = interp_cov_target + eps * np.eye(interp_cov_target.shape[0])

                # Whitening transform (inverse sqrt of source cov)
                L_src = np.linalg.cholesky(cov_src_source_reg)
                L_interp_source = np.linalg.cholesky(interp_cov_source_reg)
                L_tgt = np.linalg.cholesky(cov_src_target_reg)
                L_interp_target = np.linalg.cholesky(interp_cov_target_reg)

                # Transform: L_interp @ L_src^-1 @ data
                transform_source = L_interp_source @ np.linalg.inv(L_src)
                transform_target = L_interp_target @ np.linalg.inv(L_tgt)

                warped_src = np.einsum('ij,jt->it', transform_source, src)
                warped_tgt = np.einsum('ij,jt->it', transform_target, tgt)

                synthetic_sources.append(warped_src)
                synthetic_targets.append(warped_tgt)
                synthetic_labels.append(lbl)

                # Assign to synthetic session ID
                max_session_id = np.max(session_ids)
                synthetic_session_ids.append(max_session_id + i * n_samples_per_source + 1)

            except np.linalg.LinAlgError:
                # Skip if Cholesky fails (non-PSD matrix)
                continue

    n_synthetic = len(synthetic_sources)

    if n_synthetic > 0:
        # Stack synthetic samples
        syn_source = np.stack(synthetic_sources, axis=0)
        syn_target = np.stack(synthetic_targets, axis=0)
        syn_labels = np.array(synthetic_labels)
        syn_session_ids = np.array(synthetic_session_ids)

        # Concatenate with original
        aug_source = np.concatenate([source, syn_source], axis=0)
        aug_target = np.concatenate([target, syn_target], axis=0)
        aug_labels = np.concatenate([labels, syn_labels], axis=0)
        aug_session_ids = np.concatenate([session_ids, syn_session_ids], axis=0)

        # Update train indices
        n_original = len(source)
        new_indices = np.arange(n_original, n_original + n_synthetic)
        aug_train_idx = np.concatenate([train_idx, new_indices])
    else:
        aug_source = source
        aug_target = target
        aug_labels = labels
        aug_session_ids = session_ids
        aug_train_idx = train_idx

    if verbose and _is_primary_rank():
        print(f"  Generated geodesic warp samples: {n_synthetic}")
        print(f"  Original training samples: {len(train_idx)}")
        print(f"  Total training samples: {len(aug_train_idx)}")
        print(f"{'='*60}\n")

    return aug_source, aug_target, aug_labels, aug_train_idx, aug_session_ids


# =============================================================================
# Data Preparation Pipeline
# =============================================================================

def prepare_data(
    data_path: Path = DATA_PATH,
    odor_csv_path: Path = ODOR_CSV_PATH,
    seed: int = 42,
    split_by_session: bool = False,
    n_test_sessions: int = 1,
    n_val_sessions: int = 1,
    session_column: str = "recording_id",
    force_recreate_splits: bool = False,
    test_sessions: Optional[List[str]] = None,  # Explicit test session names
    val_sessions: Optional[List[str]] = None,   # Explicit val session names
    no_test_set: bool = False,  # If True, no test set - all held-out sessions for validation
    separate_val_sessions: bool = False,  # If True, return per-session val indices
    # Covariance augmentation
    use_covariance_augmentation: bool = False,
    cov_aug_strength: float = 0.3,
    cov_aug_n_synthetic: int = 3,
    # Dataset expansion augmentations
    aug_expand_session_mixup: bool = False,
    aug_expand_session_mixup_n: int = 2,
    aug_expand_session_mixup_alpha: float = 0.5,
    aug_expand_geodesic: bool = False,
    aug_expand_geodesic_n: int = 2,
    aug_expand_geodesic_range: Tuple[float, float] = (0.3, 0.7),
) -> Dict[str, Any]:
    """Complete data preparation pipeline.

    Args:
        data_path: Path to signal .npy file
        odor_csv_path: Path to metadata CSV
        seed: Random seed for reproducibility
        split_by_session: If True, use session-based holdout instead of random splits
        n_test_sessions: Number of sessions to hold out for testing (if split_by_session=True)
        n_val_sessions: Number of sessions to hold out for validation (if split_by_session=True)
        session_column: Column name in CSV containing session IDs
        force_recreate_splits: If True, recreate splits even if they exist on disk
        test_sessions: Explicit list of session names for test (overrides n_test_sessions)
        val_sessions: Explicit list of session names for val (overrides n_val_sessions)
        no_test_set: If True, no test set - all held-out sessions are for validation only
        separate_val_sessions: If True, return per-session validation indices
        use_covariance_augmentation: If True, generate synthetic sessions via cov perturbation
        cov_aug_strength: Perturbation strength (0-1)
        cov_aug_n_synthetic: Number of synthetic sessions per real session

    Returns dictionary with:
    - ob, pcx: Normalized signal arrays (augmented if use_covariance_augmentation)
    - odors: Odor label array
    - vocab: Odor name to ID mapping
    - train_idx, val_idx, test_idx: Split indices
    - norm_stats: Normalization statistics
    - split_info: (only if split_by_session) metadata about session splits
    - val_idx_per_session: (only if separate_val_sessions) dict of session_name -> indices
    """
    # Load raw signals
    signals = load_signals(data_path)
    num_trials = signals.shape[0]

    # Load odor labels
    odors, vocab = load_odor_labels(odor_csv_path, num_trials)

    # Extract window
    windowed = extract_window(signals)

    # Create splits
    split_info = None
    if split_by_session:
        # Session-based holdout: entire sessions held out for test/val
        session_ids, session_to_idx, idx_to_session = load_session_ids(
            odor_csv_path, session_column, num_trials
        )
        train_idx, val_idx, test_idx, split_info = load_or_create_session_splits(
            session_ids=session_ids,
            odors=odors,
            n_test_sessions=n_test_sessions,
            n_val_sessions=n_val_sessions,
            seed=seed,
            force_recreate=force_recreate_splits,
            idx_to_session=idx_to_session,
            test_sessions=test_sessions,
            val_sessions=val_sessions,
            no_test_set=no_test_set,
            separate_val_sessions=separate_val_sessions,
        )
    else:
        # Random stratified splits (original behavior)
        train_idx, val_idx, test_idx = load_or_create_stratified_splits(odors, seed)

    # Compute normalization from training set ONLY
    norm_stats = compute_normalization(windowed, train_idx)

    # Normalize
    normalized = normalize(windowed, norm_stats)

    # Split into OB and PCx
    ob = normalized[:, 0]  # [trials, channels, time]
    pcx = normalized[:, 1]

    # Apply covariance augmentation if enabled
    if use_covariance_augmentation:
        ob, pcx, odors, train_idx, session_ids = apply_covariance_augmentation(
            source=ob,
            target=pcx,
            labels=odors,
            train_idx=train_idx,
            strength=cov_aug_strength,
            n_synthetic=cov_aug_n_synthetic,
            seed=seed,
            verbose=True,
            session_ids=session_ids if split_by_session else None,
        )

    # Apply dataset expansion augmentations (only if session info available)
    if split_by_session and session_ids is not None:
        # Session mixup expansion
        if aug_expand_session_mixup:
            ob, pcx, odors, train_idx, session_ids = expand_session_mixup(
                source=ob,
                target=pcx,
                labels=odors,
                train_idx=train_idx,
                session_ids=session_ids,
                n_samples_per_pair=aug_expand_session_mixup_n,
                alpha=aug_expand_session_mixup_alpha,
                seed=seed,
                verbose=True,
            )

        # Geodesic warp expansion
        if aug_expand_geodesic:
            ob, pcx, odors, train_idx, session_ids = expand_geodesic_warp(
                source=ob,
                target=pcx,
                labels=odors,
                train_idx=train_idx,
                session_ids=session_ids,
                n_samples_per_source=aug_expand_geodesic_n,
                t_range=aug_expand_geodesic_range,
                seed=seed,
                verbose=True,
            )

    result = {
        "ob": ob,
        "pcx": pcx,
        "odors": odors,
        "vocab": vocab,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "norm_stats": norm_stats,
        "n_odors": len(vocab),
    }

    if split_info is not None:
        result["split_info"] = split_info
        # Include per-session validation indices if available
        if "val_idx_per_session" in split_info:
            result["val_idx_per_session"] = split_info["val_idx_per_session"]

    # Include session info for per-session evaluation
    if split_by_session:
        result["session_ids"] = session_ids  # Integer session ID per trial
        result["idx_to_session"] = idx_to_session  # Map from int ID to session name

    return result


def prepare_pfc_data(
    data_path: Path = PFC_DATA_PATH,
    meta_path: Path = PFC_META_PATH,
    seed: int = 42,
    split_by_session: bool = False,
    split_by_rat: bool = False,
    n_test_sessions: int = 1,
    n_val_sessions: int = 1,
    force_recreate_splits: bool = False,
    resample_to_1khz: bool = False,
) -> Dict[str, Any]:
    """Complete data preparation pipeline for PFC/Hippocampus dataset.

    This dataset contains prefrontal cortex (PFC) and hippocampal CA1 recordings
    during a left/right decision task.

    Args:
        data_path: Path to neural_data.npy file
        meta_path: Path to metadata.csv file
        seed: Random seed for reproducibility
        split_by_session: If True, hold out entire sessions for test/val
        split_by_rat: If True, hold out entire subjects (rats) for test/val
        n_test_sessions: Number of sessions/rats to hold out for testing
        n_val_sessions: Number of sessions/rats to hold out for validation
        force_recreate_splits: If True, recreate splits even if they exist
        resample_to_1khz: If True, downsample from 1250Hz to 1000Hz

    Returns dictionary with:
    - pfc: Normalized PFC signal array [trials, 64, time]
    - ca1: Normalized CA1 signal array [trials, 32, time]
    - trial_types: Trial type label array (0=Left, 1=Right or similar)
    - vocab: Trial type name to ID mapping
    - metadata: Full metadata DataFrame
    - train_idx, val_idx, test_idx: Split indices
    - norm_stats_pfc, norm_stats_ca1: Normalization statistics for each region
    - dataset_type: DatasetType.PFC_HPC
    - split_info: (if split_by_session/rat) metadata about splits
    """
    if _is_primary_rank():
        print(f"\n{'='*60}")
        print("Loading PFC/Hippocampus Dataset")
        print(f"{'='*60}")

    # Load raw signals
    pfc, ca1 = load_pfc_signals(data_path)
    num_trials = pfc.shape[0]

    # Optionally resample to match olfactory dataset
    if resample_to_1khz:
        from scipy.signal import resample
        target_len = int(PFC_TIME_POINTS * SAMPLING_RATE_HZ / PFC_SAMPLING_RATE_HZ)
        if _is_primary_rank():
            print(f"Resampling from {PFC_TIME_POINTS} to {target_len} time points (1250Hz -> 1000Hz)")
        pfc_resampled = np.zeros((num_trials, PFC_CHANNELS, target_len), dtype=np.float32)
        ca1_resampled = np.zeros((num_trials, CA1_CHANNELS, target_len), dtype=np.float32)
        for i in range(num_trials):
            for c in range(PFC_CHANNELS):
                pfc_resampled[i, c] = resample(pfc[i, c], target_len)
            for c in range(CA1_CHANNELS):
                ca1_resampled[i, c] = resample(ca1[i, c], target_len)
        pfc = pfc_resampled
        ca1 = ca1_resampled

    # Load metadata
    trial_types, vocab, metadata = load_pfc_metadata(meta_path, num_trials)

    # Create splits
    split_info = None
    if split_by_rat:
        # Subject-based holdout (leave-one-subject-out style)
        rat_ids, rat_to_idx, idx_to_rat = load_pfc_rat_ids(meta_path, num_trials)
        train_idx, val_idx, test_idx, split_info = load_or_create_pfc_session_splits(
            session_ids=rat_ids,  # Reuse session split logic with rat IDs
            trial_types=trial_types,
            n_test_sessions=n_test_sessions,
            n_val_sessions=n_val_sessions,
            seed=seed,
            force_recreate=force_recreate_splits,
            idx_to_session=idx_to_rat,
        )
        split_info["split_type"] = "pfc_rat_holdout"
    elif split_by_session:
        # Session-based holdout
        session_ids, session_to_idx, idx_to_session = load_pfc_session_ids(meta_path, num_trials)
        train_idx, val_idx, test_idx, split_info = load_or_create_pfc_session_splits(
            session_ids=session_ids,
            trial_types=trial_types,
            n_test_sessions=n_test_sessions,
            n_val_sessions=n_val_sessions,
            seed=seed,
            force_recreate=force_recreate_splits,
            idx_to_session=idx_to_session,
        )
    else:
        # Random stratified splits
        train_idx, val_idx, test_idx = load_or_create_pfc_stratified_splits(
            trial_types, seed, force_recreate=force_recreate_splits
        )

    # Compute normalization from training set ONLY (separately for each region)
    norm_stats_pfc = compute_normalization(pfc, train_idx)
    norm_stats_ca1 = compute_normalization(ca1, train_idx)

    # Normalize
    pfc_normalized = normalize(pfc, norm_stats_pfc)
    ca1_normalized = normalize(ca1, norm_stats_ca1)

    if _is_primary_rank():
        print(f"\nFinal shapes:")
        print(f"  PFC: {pfc_normalized.shape}")
        print(f"  CA1: {ca1_normalized.shape}")
        print(f"  Trial types: {len(vocab)} classes - {vocab}")
        print(f"  Splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    result = {
        # Signal arrays (analogous to ob/pcx in olfactory dataset)
        "pfc": pfc_normalized,
        "ca1": ca1_normalized,
        # For compatibility with existing code expecting source/target naming
        "source": pfc_normalized,  # PFC -> CA1 translation
        "target": ca1_normalized,
        # Alternative naming for CA1 -> PFC translation
        "region_a": pfc_normalized,
        "region_b": ca1_normalized,
        # Labels
        "trial_types": trial_types,
        "labels": trial_types,  # Generic name for compatibility
        "vocab": vocab,
        "metadata": metadata,
        # Splits
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        # Normalization
        "norm_stats_pfc": norm_stats_pfc,
        "norm_stats_ca1": norm_stats_ca1,
        "norm_stats": norm_stats_pfc,  # Default to PFC stats for source
        # Dataset info
        "dataset_type": DatasetType.PFC_HPC,
        "n_labels": len(vocab),
        "n_channels_source": PFC_CHANNELS,
        "n_channels_target": CA1_CHANNELS,
        "sampling_rate": PFC_SAMPLING_RATE_HZ if not resample_to_1khz else SAMPLING_RATE_HZ,
    }

    if split_info is not None:
        result["split_info"] = split_info

    return result


def create_dataloaders(
    data: Dict[str, Any],
    batch_size: int = 16,
    num_workers: int = 4,
    filter_odor_id: Optional[int] = None,
    temporal_ablation: Optional[str] = None,
    data_fraction: float = 1.0,
    distributed: bool = False,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for train/val/test splits.

    Returns dictionary with 'train', 'val', 'test' DataLoaders.
    
    Args:
        seed: Random seed for reproducibility (used for worker initialization).
    """
    # Worker init function for reproducibility
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    # Generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Create temporal ablation if specified
    temp_ablation = TemporalAblation(temporal_ablation) if temporal_ablation else None

    # Get session_ids if available (for session-aware augmentation)
    session_ids = data.get("session_ids", None)

    # Create datasets
    train_dataset = PairedConditionalDataset(
        data["ob"], data["pcx"], data["odors"], data["train_idx"],
        filter_odor_id=filter_odor_id,
        temporal_ablation=temp_ablation,
        data_fraction=data_fraction,
        session_ids=session_ids,
    )
    val_dataset = PairedConditionalDataset(
        data["ob"], data["pcx"], data["odors"], data["val_idx"],
        filter_odor_id=filter_odor_id,
        temporal_ablation=temp_ablation,
        session_ids=session_ids,
    )
    test_dataset = PairedConditionalDataset(
        data["ob"], data["pcx"], data["odors"], data["test_idx"],
        filter_odor_id=filter_odor_id,
        temporal_ablation=temp_ablation,
        session_ids=session_ids,
    )

    # Create samplers for distributed training (seed=42 for reproducibility)
    train_sampler = DistributedSampler(train_dataset, seed=42) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=42) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=42) if distributed else None

    # Create dataloaders with optimized settings
    pin_memory = torch.cuda.is_available()
    # persistent_workers keeps worker processes alive between epochs (faster)
    # prefetch_factor=2 prefetches 2 batches per worker in advance
    persistent = num_workers > 0

    # For small datasets, reduce batch size for val/test to ensure we get at least 1 batch
    val_batch_size = batch_size
    test_batch_size = batch_size
    if distributed:
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        val_per_gpu = len(val_dataset) // world_size
        test_per_gpu = len(test_dataset) // world_size
        if val_per_gpu < batch_size:
            val_batch_size = max(1, val_per_gpu)
        if test_per_gpu < batch_size:
            test_batch_size = max(1, test_per_gpu)

    # Reproducibility: create generator and worker_init_fn for deterministic data loading
    g = torch.Generator()
    g.manual_seed(42)
    
    def worker_init_fn(worker_id):
        worker_seed = 42 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Don't drop for val - we want all samples evaluated
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Don't drop for test - we want all samples evaluated
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    result = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_sampler": train_sampler,
    }

    # Add per-session validation loaders if available
    if "val_idx_per_session" in data and data["val_idx_per_session"] is not None:
        val_session_loaders = {}
        for sess_name, sess_indices in data["val_idx_per_session"].items():
            val_session_loaders[sess_name] = create_single_session_dataloader(
                data, sess_name, sess_indices,
                batch_size=val_batch_size,
                num_workers=num_workers,
                distributed=distributed,
                seed=seed,
            )
        result["val_sessions"] = val_session_loaders

    return result


def create_single_session_dataloader(
    data: Dict[str, Any],
    session_name: str,
    indices: np.ndarray,
    batch_size: int = 16,
    num_workers: int = 4,
    distributed: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader for a single session's data.

    Args:
        data: Data dictionary from prepare_data()
        session_name: Name of the session (for logging)
        indices: Trial indices belonging to this session
        batch_size: Batch size
        num_workers: Number of data loading workers
        distributed: Whether to use distributed sampler
        seed: Random seed

    Returns:
        DataLoader for the specified session's data
    """
    # Create dataset for this session's indices
    dataset = PairedConditionalDataset(
        data["ob"], data["pcx"], data["odors"], indices,
    )

    # Create sampler for distributed training
    sampler = DistributedSampler(dataset, shuffle=False, seed=seed) if distributed else None

    # Handle small datasets
    effective_batch_size = batch_size
    if distributed:
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        per_gpu = len(dataset) // world_size
        if per_gpu < batch_size:
            effective_batch_size = max(1, per_gpu)

    # Worker init for reproducibility
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    return loader


def create_pfc_dataloaders(
    data: Dict[str, Any],
    batch_size: int = 16,
    num_workers: int = 4,
    filter_label_id: Optional[int] = None,
    temporal_ablation: Optional[str] = None,
    data_fraction: float = 1.0,
    distributed: bool = False,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for PFC/CA1 train/val/test splits.

    Args:
        data: Output from prepare_pfc_data()
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        filter_label_id: If provided, only include samples with this trial type
        temporal_ablation: Temporal ablation pattern to apply
        data_fraction: Fraction of training data to use
        distributed: If True, use DistributedSampler

    Returns dictionary with 'train', 'val', 'test' DataLoaders.
    """
    # Create temporal ablation if specified
    temp_ablation = TemporalAblation(temporal_ablation) if temporal_ablation else None

    # Use PFC as source and CA1 as target
    pfc = data["pfc"]
    ca1 = data["ca1"]
    trial_types = data["trial_types"]

    # Create datasets using the generic PairedNeuralDataset
    train_dataset = PairedNeuralDataset(
        pfc, ca1, trial_types, data["train_idx"],
        filter_label_id=filter_label_id,
        temporal_ablation=temp_ablation,
        data_fraction=data_fraction,
    )
    val_dataset = PairedNeuralDataset(
        pfc, ca1, trial_types, data["val_idx"],
        filter_label_id=filter_label_id,
        temporal_ablation=temp_ablation,
    )
    test_dataset = PairedNeuralDataset(
        pfc, ca1, trial_types, data["test_idx"],
        filter_label_id=filter_label_id,
        temporal_ablation=temp_ablation,
    )

    # Create samplers for distributed training (seed=42 for reproducibility)
    train_sampler = DistributedSampler(train_dataset, seed=42) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=42) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=42) if distributed else None

    # Create dataloaders with optimized settings
    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0

    # For small datasets, reduce batch size for val/test to ensure we get at least 1 batch
    # This is especially important for distributed training where samples are split across GPUs
    val_batch_size = batch_size
    test_batch_size = batch_size
    if distributed:
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        # Samples per GPU
        val_per_gpu = len(val_dataset) // world_size
        test_per_gpu = len(test_dataset) // world_size
        # Ensure at least 1 batch per GPU
        if val_per_gpu < batch_size:
            val_batch_size = max(1, val_per_gpu)
        if test_per_gpu < batch_size:
            test_batch_size = max(1, test_per_gpu)

    # Reproducibility: create generator and worker_init_fn for deterministic data loading
    g = torch.Generator()
    g.manual_seed(42)
    
    def worker_init_fn(worker_id):
        worker_seed = 42 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Don't drop for val - we want all samples evaluated
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Don't drop for test - we want all samples evaluated
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_sampler": train_sampler,
    }


# =============================================================================
# Utility Functions
# =============================================================================

def crop_to_target_torch(x: torch.Tensor) -> torch.Tensor:
    """Crop signal to target window (remove edge artifacts)."""
    return x[..., CROP_START:CROP_END]


def crop_to_target_np(x: np.ndarray) -> np.ndarray:
    """Crop numpy array to target window."""
    return x[..., CROP_START:CROP_END]


def time_axis(num_points: int, start_time: float = T_TARGET_START_S) -> np.ndarray:
    """Generate time axis for plotting."""
    return np.arange(num_points) / SAMPLING_RATE_HZ + start_time


# =============================================================================
# Data Module Wrapper (for compatibility)
# =============================================================================

class OlfactoryDataModule:
    """Wrapper class to provide a consistent interface for data loading."""

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        normalize_per_segment: bool = False,
        segment_length_sec: float = 2.0,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize_per_segment = normalize_per_segment
        self.segment_length_sec = segment_length_sec
        self.data = None
        self.loaders = None

    def setup(self, stage: Optional[str] = None):
        """Prepare data for training/validation/testing."""
        # prepare_data only takes seed parameter, not the segment params
        self.data = prepare_data()

        self.loaders = create_dataloaders(
            self.data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        """Return training dataloader."""
        if self.loaders is None:
            self.setup()
        return self.loaders["train"]

    def val_dataloader(self):
        """Return validation dataloader."""
        if self.loaders is None:
            self.setup()
        return self.loaders["val"]

    def test_dataloader(self):
        """Return test dataloader."""
        if self.loaders is None:
            self.setup()
        return self.loaders["test"]


class PFCDataModule:
    """Wrapper class for PFC/Hippocampus dataset loading.
    
    Provides consistent interface for loading PFC -> CA1 translation data.
    """

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        split_by_session: bool = False,
        split_by_rat: bool = False,
        resample_to_1khz: bool = False,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.split_by_session = split_by_session
        self.split_by_rat = split_by_rat
        self.resample_to_1khz = resample_to_1khz
        self.data = None
        self.loaders = None

    def setup(self, stage: Optional[str] = None):
        """Prepare data for training/validation/testing."""
        self.data = prepare_pfc_data(
            split_by_session=self.split_by_session,
            split_by_rat=self.split_by_rat,
            resample_to_1khz=self.resample_to_1khz,
        )

        self.loaders = create_pfc_dataloaders(
            self.data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        """Return training dataloader."""
        if self.loaders is None:
            self.setup()
        return self.loaders["train"]

    def val_dataloader(self):
        """Return validation dataloader."""
        if self.loaders is None:
            self.setup()
        return self.loaders["val"]

    def test_dataloader(self):
        """Return test dataloader."""
        if self.loaders is None:
            self.setup()
        return self.loaders["test"]
    
    @property
    def n_channels_source(self) -> int:
        """Number of channels in source region (PFC)."""
        return PFC_CHANNELS
    
    @property
    def n_channels_target(self) -> int:
        """Number of channels in target region (CA1)."""
        return CA1_CHANNELS
    
    @property
    def n_labels(self) -> int:
        """Number of condition labels (trial types)."""
        if self.data is None:
            self.setup()
        return self.data["n_labels"]


def get_odor_name(vocab: Dict[str, int], odor_id: int) -> str:
    """Get odor name from ID."""
    for name, idx in vocab.items():
        if idx == odor_id:
            return name
    return f"odor_{odor_id}"


def get_label_name(vocab: Dict[str, int], label_id: int) -> str:
    """Get label name from ID (works for odors, trial types, etc)."""
    for name, idx in vocab.items():
        if idx == label_id:
            return name
    return f"label_{label_id}"


def get_data_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """Get summary information about loaded data.
    
    Works for both olfactory and PFC datasets.
    """
    # Determine dataset type
    if "ob" in data:
        # Olfactory dataset
        return {
            "dataset_type": "olfactory",
            "n_trials": data["ob"].shape[0],
            "n_channels_source": data["ob"].shape[1],
            "n_channels_target": data["pcx"].shape[1],
            "n_timepoints": data["ob"].shape[2],
            "n_labels": data.get("n_odors", len(data["vocab"])),
            "label_names": list(data["vocab"].keys()),
            "train_size": len(data["train_idx"]),
            "val_size": len(data["val_idx"]),
            "test_size": len(data["test_idx"]),
        }
    elif "pfc" in data:
        # PFC dataset
        return {
            "dataset_type": "pfc_hpc",
            "n_trials": data["pfc"].shape[0],
            "n_channels_source": data["pfc"].shape[1],
            "n_channels_target": data["ca1"].shape[1],
            "n_timepoints": data["pfc"].shape[2],
            "n_labels": data.get("n_labels", len(data["vocab"])),
            "label_names": list(data["vocab"].keys()),
            "train_size": len(data["train_idx"]),
            "val_size": len(data["val_idx"]),
            "test_size": len(data["test_idx"]),
            "sampling_rate": data.get("sampling_rate", PFC_SAMPLING_RATE_HZ),
        }
    else:
        raise ValueError("Unknown data format")
