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
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import random

# Suppress hdmf namespace version warnings (harmless version mismatch in NWB files)
warnings.filterwarnings("ignore", message="Ignoring the following cached namespace")
warnings.filterwarnings("ignore", module="hdmf.spec.namespace")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import StratifiedShuffleSplit


# =============================================================================
# Distributed Training Helpers (to avoid 8x printing on multi-GPU)
# =============================================================================

def _is_primary_rank() -> bool:
    """Check if this is the primary rank (rank 0) or not in distributed mode.

    Returns True if:
    - Not in distributed training mode
    - In distributed mode and rank == 0

    Use this to gate print statements so they only execute once.
    """
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _print_primary(*args, **kwargs):
    """Print only on primary rank to avoid duplicate output in distributed training."""
    if _is_primary_rank():
        print(*args, **kwargs)


# =============================================================================
# Dataset Type Enum
# =============================================================================

class DatasetType(Enum):
    """Supported dataset types."""
    OLFACTORY = "olfactory"      # OB/PCx dataset
    PFC_HPC = "pfc_hpc"          # PFC/Hippocampus (CA1) dataset
    DANDI_MOVIE = "dandi_movie"  # DANDI 000623: Human iEEG movie watching
    ECOG = "ecog"                # Miller ECoG Library: inter-region cortical translation
    BORAN_MTL = "boran_mtl"      # Boran et al. MTL Working Memory (depth electrodes)


# =============================================================================
# Constants - Olfactory Dataset (OB/PCx)
# =============================================================================
# Data directories - configurable via environment variables
# Set UNET_DATA_DIR to override the default data location
_DATA_DIR = Path(os.environ.get("UNET_DATA_DIR", "/data"))

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
_PFC_DATA_DIR = _DATA_DIR / "pfc" / "processed_data"
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
# Constants - DANDI 000623 Movie Dataset (Human iEEG)
# =============================================================================
# Reference: Keles et al., 2024, Scientific Data
# "Multimodal single-neuron, intracranial EEG, and fMRI brain responses
# during movie watching in human patients"
# DANDI Archive: https://dandiarchive.org/dandiset/000623
# GitHub: https://github.com/rutishauserlab/bmovie-release-NWB-BIDS

_DANDI_DATA_DIR = _DATA_DIR / "movie"
DANDI_DANDISET_ID = "000623"
DANDI_SAMPLING_RATE_HZ = 1000  # LFP/iEEG downsampled to 1000 Hz
DANDI_MOVIE_DURATION_S = 480.0  # ~8 minutes movie clip

# Brain regions in the dataset
DANDI_BRAIN_REGIONS = ["amygdala", "hippocampus", "medial_frontal_cortex"]
DANDI_REGION_ABBREVIATIONS = {"amygdala": "AMY", "hippocampus": "HPC", "medial_frontal_cortex": "MFC"}

# Data paths (NWB files will be downloaded here)
DANDI_RAW_PATH = _DANDI_DATA_DIR  # NWB files directly in /data/movie
DANDI_PROCESSED_PATH = _DANDI_DATA_DIR / "processed"
DANDI_CACHE_PATH = _DANDI_DATA_DIR / "cache"

# Subject info (18 subjects in the dataset)
DANDI_N_SUBJECTS = 18
DANDI_SUBJECT_IDS = [f"sub-CS{i}" for i in range(41, 63) if i not in [43, 44, 48, 50]]

# Train/val/test split paths
DANDI_TRAIN_SPLIT_PATH = _DANDI_DATA_DIR / "train_indices.npy"
DANDI_VAL_SPLIT_PATH = _DANDI_DATA_DIR / "val_indices.npy"
DANDI_TEST_SPLIT_PATH = _DANDI_DATA_DIR / "test_indices.npy"


# =============================================================================
# Constants - Miller ECoG Library (Human Subdural ECoG)
# =============================================================================
# Reference: Miller, K.J. (2019). A library of human electrocorticographic
# data and analyses. Nature Human Behaviour, 3(11), 1225-1235.
# DOI: 10.1038/s41562-019-0678-3
# Repository: https://purl.stanford.edu/zk881ps0522
# Preprocessed data via Neuromatch Academy (OSF)

_ECOG_DATA_DIR = _DATA_DIR / "ECoG"
ECOG_SAMPLING_RATE_HZ = 1000  # All recordings at 1kHz (verified from actual data, fallback only)

# Available experiments with OSF download URLs
ECOG_EXPERIMENTS = {
    "fingerflex": "https://osf.io/5m47z/download",
    "faceshouses": "https://osf.io/argh7/download",
    "motor_imagery": "https://osf.io/ksqv8/download",
    "joystick_track": "https://osf.io/6jncm/download",
    "memory_nback": "https://osf.io/xfc7e/download",
}

# Brain lobes for region-based source/target pairing
# Raw lobe annotations in the data are e.g. "Frontal Lobe", "Temporal Lobe" etc.
# We normalize to lowercase single-word keys.
ECOG_BRAIN_LOBES = ["frontal", "temporal", "parietal", "occipital", "limbic"]


# =============================================================================
# Constants - Boran MTL Working Memory (Depth Electrodes)
# =============================================================================
# Boran et al., Sci Data 2020; DOI: 10.12751/g-node.d76994
# 9 subjects, 37 sessions, ~1827 iEEG trials, 2000 Hz
# Depth electrodes: 8 contacts per probe
# Regions: hippocampus (AH/PH probes), entorhinal_cortex (EC probes), amygdala (A probes)

_BORAN_DATA_DIR = _DATA_DIR / "boran_mtl_wm" / "processed_1khz"
BORAN_SAMPLING_RATE_HZ = 1000  # Preprocessed: downsampled from 2kHz to 1kHz

# Canonical MTL regions for inter-region translation
BORAN_MTL_REGIONS = ["hippocampus", "entorhinal_cortex", "amygdala"]

# Probe prefix -> canonical region mapping (longest-first matching)
BORAN_PROBE_REGION_MAP = {
    "AH":  "hippocampus",       # Anterior Hippocampus
    "PH":  "hippocampus",       # Posterior Hippocampus
    "EC":  "entorhinal_cortex", # Entorhinal Cortex
    "AL":  "amygdala",          # Amygdala Left
    "AR":  "amygdala",          # Amygdala Right
    "A":   "amygdala",          # Amygdala (generic)
    "TB":  "other",             # Temporal basal
    "PHC": "other",             # Parahippocampal
    "DR":  "other",             # Non-MTL
    "LR":  "other",             # Non-MTL
}


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
    _print_primary(f"Loaded PFC data: {n_trials} trials, {n_time} time points, {n_channels} channels")

    if n_channels != PFC_TOTAL_CHANNELS:
        raise ValueError(f"Expected {PFC_TOTAL_CHANNELS} channels, got {n_channels}")

    # Transpose from (trials, time, channels) to (trials, channels, time)
    arr = arr.transpose(0, 2, 1)  # Now: (494, 96, 6250)

    # Split into PFC and CA1
    pfc = arr[:, PFC_CHANNEL_START:PFC_CHANNEL_END, :]  # (494, 64, 6250)
    ca1 = arr[:, CA1_CHANNEL_START:CA1_CHANNEL_END, :]  # (494, 32, 6250)

    _print_primary(f"Split into PFC: {pfc.shape}, CA1: {ca1.shape}")

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

    _print_primary(f"PFC trial types: {vocab}")
    _print_primary(f"Trial type distribution: {np.bincount(ids)}")

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

    _print_primary(f"Found {len(unique_sessions)} unique PFC sessions: {unique_sessions[:5]}{'...' if len(unique_sessions) > 5 else ''}")

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

    _print_primary(f"Found {len(unique_rats)} unique rats: {unique_rats}")

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

    _print_primary(f"Found {len(unique_sessions)} unique sessions: {unique_sessions[:5]}{'...' if len(unique_sessions) > 5 else ''}")

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
        # Handle empty arrays (e.g., no test set)
        train_max = train_idx.max() if len(train_idx) > 0 else -1
        val_max = val_idx.max() if len(val_idx) > 0 else -1
        test_max = test_idx.max() if len(test_idx) > 0 else -1
        if train_max > max_idx or val_max > max_idx or test_max > max_idx:
            _print_primary(f"WARNING: Cached split indices exceed data size! Recreating splits...")
            # Don't return - fall through to recreate splits
        else:
            # Validate no overlap
            train_set = set(train_idx.tolist())
            val_set = set(val_idx.tolist())
            test_set = set(test_idx.tolist())
            has_overlap = bool(train_set & val_set) or bool(train_set & test_set) or bool(val_set & test_set)
            if has_overlap:
                _print_primary(f"WARNING: Loaded splits have overlapping indices! Recreating splits...")
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
                if session_overlap:
                    _print_primary(f"WARNING: Loaded splits have overlapping sessions! This shouldn't happen!")
                    _print_primary(f"  Train sessions: {train_sessions_check}")
                    _print_primary(f"  Val sessions: {val_sessions_check}")
                    _print_primary(f"  Test sessions: {test_sessions_check}")

                _print_primary(f"Loaded existing session splits: {split_info['n_train_sessions']} train, "
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

        _print_primary(f"\n[Explicit Session Split]")
        _print_primary(f"  Test sessions: {test_sessions}")
        _print_primary(f"  Val sessions: {val_sessions}")
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

    # Create combined val_idx first (will be shuffled later)
    val_idx = all_indices[np.isin(session_ids, list(val_session_ids))]

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

    if train_sessions_check & val_sessions_check:
        _print_primary(f"WARNING: Train and Val have overlapping sessions: {train_sessions_check & val_sessions_check}")
    if train_sessions_check & test_sessions_check:
        _print_primary(f"WARNING: Train and Test have overlapping sessions: {train_sessions_check & test_sessions_check}")
    if val_sessions_check & test_sessions_check:
        _print_primary(f"WARNING: Val and Test have overlapping sessions: {val_sessions_check & test_sessions_check}")

    # Shuffle within splits
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    # CRITICAL FIX: Create per-session validation indices AFTER shuffling val_idx
    # This ensures that per-session indices have the same ordering as val_idx,
    # so DistributedSampler gives each rank the same samples for both the
    # combined val loader and per-session loaders. Without this, the R² metrics
    # would differ because rank 0 would see different samples from each loader.
    if separate_val_sessions:
        val_idx_per_session = {}
        for sess_id in val_session_ids:
            sess_name = idx_to_session[sess_id] if idx_to_session is not None else str(sess_id)
            # Extract indices from SHUFFLED val_idx that belong to this session
            # This preserves the random order established by the shuffle
            sess_mask = session_ids[val_idx] == sess_id
            sess_indices = val_idx[sess_mask]
            val_idx_per_session[sess_name] = sess_indices
    else:
        val_idx_per_session = None

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

    # Print summary
    _print_primary(f"\n{'='*60}")
    if no_test_set:
        _print_primary("SESSION-BASED SPLIT (No Test Set - All Held-Out Sessions for Validation)")
    else:
        _print_primary("SESSION-BASED SPLIT (Held-Out Session Evaluation)")
    _print_primary(f"{'='*60}")
    _print_primary(f"Total sessions: {n_sessions}")
    _print_primary(f"Train sessions: {train_session_names} ({len(train_idx)} trials)")
    if separate_val_sessions and val_idx_per_session is not None:
        _print_primary(f"Val sessions (SEPARATE):")
        for sess_name, indices in val_idx_per_session.items():
            _print_primary(f"  - {sess_name}: {len(indices)} trials")
    else:
        _print_primary(f"Val sessions:   {val_session_names} ({len(val_idx)} trials)")
    if not no_test_set:
        _print_primary(f"Test sessions:  {test_session_names} ({len(test_idx)} trials)")
    _print_primary(f"{'='*60}\n")

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
    force_recreate: bool = False,
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
        force_recreate: If True, recreate splits even if they exist on disk.
    """
    # Check if splits already exist (skip if force_recreate)
    if not force_recreate and TRAIN_SPLIT_PATH.exists() and VAL_SPLIT_PATH.exists() and TEST_SPLIT_PATH.exists():
        train_idx = np.load(TRAIN_SPLIT_PATH)
        val_idx = np.load(VAL_SPLIT_PATH)
        test_idx = np.load(TEST_SPLIT_PATH)
        # Validate indices don't exceed data size
        max_idx = len(odors) - 1
        if train_idx.max() > max_idx or val_idx.max() > max_idx or test_idx.max() > max_idx:
            _print_primary(f"WARNING: Cached stratified splits have indices exceeding data size. Recreating...")
        else:
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
        _print_primary(f"Balanced split: {n_per_odor_train} train, {n_per_odor_val} val, "
              f"{n_per_odor_test} test per odor (x{n_odors} odors)")
        _print_primary(f"Total: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

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
        _print_primary(f"Loaded existing PFC splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
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
    _print_primary(f"\nPFC Dataset Splits:")
    _print_primary(f"  Train: {len(train_idx)} samples")
    for tt in np.unique(trial_types):
        count = np.sum(trial_types[train_idx] == tt)
        _print_primary(f"    - Type {tt}: {count}")
    _print_primary(f"  Val: {len(val_idx)} samples")
    _print_primary(f"  Test: {len(test_idx)} samples")

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
    # Explicit session holdout for LOSO cross-validation
    val_sessions: Optional[List[str]] = None,
    test_sessions: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create train/val/test splits by holding out entire PFC recording sessions.

    Args:
        session_ids: Array of integer session indices for each trial
        trial_types: Array of trial type labels for each trial
        n_test_sessions: Number of sessions to hold out for testing (ignored if val_sessions provided)
        n_val_sessions: Number of sessions to hold out for validation (ignored if val_sessions provided)
        seed: Random seed for reproducibility
        force_recreate: If True, recreate splits even if they exist
        idx_to_session: Optional mapping from integer index to original session name
        val_sessions: Explicit list of session names for validation (for LOSO). Overrides n_val_sessions.
        test_sessions: Explicit list of session names for testing (for LOSO). Overrides n_test_sessions.

    Returns:
        train_idx, val_idx, test_idx: Split indices
        split_info: Dictionary with split metadata

    Note:
        For LOSO cross-validation, use val_sessions to specify the held-out session.
        All other sessions will be used for training. Example:
            load_or_create_pfc_session_splits(session_ids, trial_types, val_sessions=["session_1"])
    """
    rng = np.random.default_rng(seed)
    unique_sessions = np.unique(session_ids)
    n_sessions = len(unique_sessions)

    # Build mapping from session name to integer ID if not provided
    if idx_to_session is None:
        idx_to_session = {i: str(i) for i in unique_sessions}
    session_to_idx = {name: idx for idx, name in idx_to_session.items()}

    # Check for explicit session holdout (LOSO mode)
    if val_sessions is not None or test_sessions is not None:
        # Explicit session holdout mode (for LOSO)
        val_sessions = val_sessions or []
        test_sessions = test_sessions or []

        # Validate that specified sessions exist
        available_sessions = set(idx_to_session.values())
        for sess in val_sessions + test_sessions:
            if sess not in available_sessions:
                raise ValueError(
                    f"Session '{sess}' not found in dataset. "
                    f"Available sessions: {sorted(available_sessions)}"
                )

        # Check for overlap between val and test
        overlap = set(val_sessions) & set(test_sessions)
        if overlap:
            raise ValueError(
                f"Overlap between val and test sessions: {overlap}. "
                f"This would cause data leakage!"
            )

        # Convert session names to integer IDs
        val_session_ids = set(session_to_idx[s] for s in val_sessions)
        test_session_ids = set(session_to_idx[s] for s in test_sessions)
        holdout_ids = val_session_ids | test_session_ids
        train_session_ids = set(i for i in unique_sessions if i not in holdout_ids)

        _print_primary(f"[LOSO MODE] Explicit PFC session holdout:")
        _print_primary(f"  Train sessions ({len(train_session_ids)}): {sorted([idx_to_session[i] for i in train_session_ids])}")
        _print_primary(f"  Val sessions ({len(val_session_ids)}): {val_sessions}")
        _print_primary(f"  Test sessions ({len(test_session_ids)}): {test_sessions}")

        # Don't load/save cached splits in LOSO mode - always create fresh
    else:
        # Random split mode - check for cached splits
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
            _print_primary(f"Loaded existing PFC session splits: {split_info['n_train_sessions']} train, "
                  f"{split_info['n_val_sessions']} val, {split_info['n_test_sessions']} test sessions")
            return train_idx, val_idx, test_idx, split_info

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

    _print_primary(f"\nPFC SESSION-BASED SPLIT:")
    _print_primary(f"  Train sessions: {split_info['train_sessions']} ({len(train_idx)} trials)")
    _print_primary(f"  Val sessions: {split_info['val_sessions']} ({len(val_idx)} trials)")
    _print_primary(f"  Test sessions: {split_info['test_sessions']} ({len(test_idx)} trials)")

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
    """Dataset for paired OB-PCx signals with odor conditioning and session info.

    Args:
        ob: OB signals array [n_samples, channels, time]
        pcx: PCx signals array [n_samples, channels, time]
        odors: Odor labels array [n_samples]
        indices: Sample indices to use from the arrays
        session_ids: Optional session IDs array [n_samples] for session embedding.
                    If None, odor IDs are used as session IDs for backwards compatibility.
        filter_odor_id: If provided, only keep samples with this odor ID
        temporal_ablation: TemporalAblation instance to apply masking
        data_fraction: Fraction of data to use (for data scaling experiments)
    """
    def __init__(
        self,
        ob: np.ndarray,
        pcx: np.ndarray,
        odors: np.ndarray,
        indices: np.ndarray,
        session_ids: Optional[np.ndarray] = None,
        filter_odor_id: Optional[int] = None,
        temporal_ablation: Optional[TemporalAblation] = None,
        data_fraction: float = 1.0,
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
        # Session IDs: use provided session_ids or fall back to odors for backwards compat
        if session_ids is not None:
            self.session_ids = torch.from_numpy(session_ids[indices]).long()
        else:
            # Backwards compatibility: if no session_ids, use odors (wrong but matches old behavior)
            self.session_ids = self.odors
        self.temporal_ablation = temporal_ablation

    def __len__(self) -> int:
        return self.ob.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ob = self.ob[idx]
        pcx = self.pcx[idx]
        odor = self.odors[idx]
        session_id = self.session_ids[idx]

        if self.temporal_ablation is not None:
            ob = self.temporal_ablation.apply_mask(ob)
            pcx = self.temporal_ablation.apply_mask(pcx)

        return ob, pcx, odor, session_id


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
    exclude_sessions: Optional[List[str]] = None,  # Sessions to exclude entirely
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
        exclude_sessions: Sessions to exclude entirely from all data (for held-out test)

    Returns dictionary with:
    - ob, pcx: Normalized signal arrays
    - odors: Odor label array
    - vocab: Odor name to ID mapping
    - train_idx, val_idx, test_idx: Split indices
    - norm_stats: Normalization statistics
    - split_info: (only if split_by_session) metadata about session splits
    - val_idx_per_session: (only if separate_val_sessions) dict of session_name -> indices
    - excluded_sessions: (only if exclude_sessions) list of excluded session names
    """
    # Load raw signals
    signals = load_signals(data_path)
    num_trials = signals.shape[0]

    # Load odor labels
    odors, vocab = load_odor_labels(odor_csv_path, num_trials)

    # Filter out excluded sessions if specified
    excluded_session_names = []
    session_ids = None  # Only load if needed
    session_to_idx = None
    idx_to_session = None

    if exclude_sessions:
        # Load session IDs for filtering
        session_ids, session_to_idx, idx_to_session = load_session_ids(
            odor_csv_path, session_column, num_trials
        )
        # Get indices of trials to keep (not in excluded sessions)
        excluded_session_ids = set()
        for sess_name in exclude_sessions:
            if sess_name in session_to_idx:
                excluded_session_ids.add(session_to_idx[sess_name])
                excluded_session_names.append(sess_name)
            else:
                _print_primary(f"Warning: Session '{sess_name}' not found, skipping")

        if excluded_session_ids:
            keep_mask = ~np.isin(session_ids, list(excluded_session_ids))
            signals = signals[keep_mask]
            odors = odors[keep_mask]
            session_ids = session_ids[keep_mask]
            num_trials = signals.shape[0]
            _print_primary(f"Excluded {len(excluded_session_names)} sessions: {excluded_session_names}")
            _print_primary(f"Remaining trials: {num_trials}")

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
    elif test_sessions:
        # Hybrid mode: session-wise test holdout + trial-wise train/val
        # Load session IDs to identify test trials
        session_ids, session_to_idx, idx_to_session = load_session_ids(
            odor_csv_path, session_column, num_trials
        )

        # Identify test session indices
        test_session_ids = set()
        test_session_names = []
        for sess_name in test_sessions:
            if sess_name in session_to_idx:
                test_session_ids.add(session_to_idx[sess_name])
                test_session_names.append(sess_name)
            else:
                _print_primary(f"Warning: Test session '{sess_name}' not found, skipping")

        # Split trials into test vs non-test
        test_mask = np.isin(session_ids, list(test_session_ids))
        test_idx = np.where(test_mask)[0]
        non_test_idx = np.where(~test_mask)[0]

        # Do trial-wise stratified split on non-test trials (70/30 train/val)
        non_test_odors = odors[non_test_idx]
        rng = np.random.default_rng(seed)

        # Stratified split by odor
        train_idx_local = []
        val_idx_local = []
        for odor_id in np.unique(non_test_odors):
            odor_mask = non_test_odors == odor_id
            odor_indices = non_test_idx[odor_mask]
            rng.shuffle(odor_indices)
            n_train = int(len(odor_indices) * 0.7)
            train_idx_local.extend(odor_indices[:n_train])
            val_idx_local.extend(odor_indices[n_train:])

        train_idx = np.array(sorted(train_idx_local))
        val_idx = np.array(sorted(val_idx_local))

        # Identify which sessions are in train/val for per-session reporting
        train_session_ids = set(session_ids[train_idx])
        train_session_names = [idx_to_session[sid] for sid in sorted(train_session_ids)]

        split_info = {
            "mode": "hybrid_test_sessions",
            "test_sessions": test_session_names,
            "train_val_sessions": train_session_names,
            "n_test_trials": len(test_idx),
            "n_train_trials": len(train_idx),
            "n_val_trials": len(val_idx),
            "seed": seed,
        }

        _print_primary(f"Hybrid split: {len(test_session_names)} test sessions ({len(test_idx)} trials)")
        _print_primary(f"  Train/val sessions: {train_session_names}")
        _print_primary(f"  Train: {len(train_idx)}, Val: {len(val_idx)} (70/30 trial-wise)")
    else:
        # Random stratified splits (original behavior)
        if no_test_set:
            # 70/30 train/val split, no test set
            train_idx, val_idx, test_idx = load_or_create_stratified_splits(
                odors, seed, train_ratio=0.7, val_ratio=0.3,
                force_recreate=force_recreate_splits
            )
            test_idx = np.array([], dtype=int)  # Empty test set
            _print_primary(f"Stratified split (no test): Train {len(train_idx)}, Val {len(val_idx)} (70/30)")
        else:
            train_idx, val_idx, test_idx = load_or_create_stratified_splits(
                odors, seed, force_recreate=force_recreate_splits
            )

    # Compute normalization from training set ONLY
    norm_stats = compute_normalization(windowed, train_idx)

    # Normalize
    normalized = normalize(windowed, norm_stats)

    # Split into OB and PCx
    ob = normalized[:, 0]  # [trials, channels, time]
    pcx = normalized[:, 1]

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

    # Include excluded sessions info
    if excluded_session_names:
        result["excluded_sessions"] = excluded_session_names

    # Include session info for per-session evaluation
    if split_by_session or test_sessions:
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
    # Explicit session holdout for LOSO cross-validation
    val_sessions: Optional[List[str]] = None,
    test_sessions: Optional[List[str]] = None,
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
        n_test_sessions: Number of sessions/rats to hold out for testing (ignored if val_sessions provided)
        n_val_sessions: Number of sessions/rats to hold out for validation (ignored if val_sessions provided)
        force_recreate_splits: If True, recreate splits even if they exist
        resample_to_1khz: If True, downsample from 1250Hz to 1000Hz
        val_sessions: Explicit list of session names for validation (for LOSO). Overrides n_val_sessions.
        test_sessions: Explicit list of session names for testing (for LOSO). Overrides n_test_sessions.

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
    _print_primary(f"\n{'='*60}")
    _print_primary("Loading PFC/Hippocampus Dataset")
    _print_primary(f"{'='*60}")

    # Load raw signals
    pfc, ca1 = load_pfc_signals(data_path)
    num_trials = pfc.shape[0]

    # Optionally resample to match olfactory dataset
    if resample_to_1khz:
        from scipy.signal import resample
        target_len = int(PFC_TIME_POINTS * SAMPLING_RATE_HZ / PFC_SAMPLING_RATE_HZ)
        _print_primary(f"Resampling from {PFC_TIME_POINTS} to {target_len} time points (1250Hz -> 1000Hz)")
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
            val_sessions=val_sessions,
            test_sessions=test_sessions,
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

    _print_primary(f"\nFinal shapes:")
    _print_primary(f"  PFC: {pfc_normalized.shape}")
    _print_primary(f"  CA1: {ca1_normalized.shape}")
    _print_primary(f"  Trial types: {len(vocab)} classes - {vocab}")
    _print_primary(f"  Splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

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

    # Get session_ids if available (for proper session embedding)
    session_ids = data.get("session_ids", None)

    # Create datasets
    train_dataset = PairedConditionalDataset(
        data["ob"], data["pcx"], data["odors"], data["train_idx"],
        session_ids=session_ids,
        filter_odor_id=filter_odor_id,
        temporal_ablation=temp_ablation,
        data_fraction=data_fraction,
    )
    val_dataset = PairedConditionalDataset(
        data["ob"], data["pcx"], data["odors"], data["val_idx"],
        session_ids=session_ids,
        filter_odor_id=filter_odor_id,
        temporal_ablation=temp_ablation,
    )
    test_dataset = PairedConditionalDataset(
        data["ob"], data["pcx"], data["odors"], data["test_idx"],
        session_ids=session_ids,
        filter_odor_id=filter_odor_id,
        temporal_ablation=temp_ablation,
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

    # For small datasets, reduce batch size to ensure we get at least 1 batch per GPU
    # This is especially important for distributed training where samples are split across GPUs
    train_batch_size = batch_size
    val_batch_size = batch_size
    test_batch_size = batch_size
    train_drop_last = True  # Default: drop incomplete batches for training stability
    if distributed:
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        # Samples per GPU
        train_per_gpu = len(train_dataset) // world_size
        val_per_gpu = len(val_dataset) // world_size
        test_per_gpu = len(test_dataset) // world_size
        # Ensure at least 1 batch per GPU for training
        if train_per_gpu < batch_size:
            # Either reduce batch size or set drop_last=False
            if train_per_gpu >= batch_size // 2:
                train_batch_size = max(1, train_per_gpu)
            else:
                # Very few samples - don't drop last batch
                train_drop_last = False
        # Ensure at least 1 batch per GPU for val/test
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
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train_drop_last,
        persistent_workers=persistent,
        prefetch_factor=4 if num_workers > 0 else None,
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
        prefetch_factor=4 if num_workers > 0 else None,
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
        prefetch_factor=4 if num_workers > 0 else None,
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

        # DEBUG: Check if val_idx matches per-session indices (for single session case)
        val_idx = data["val_idx"]
        if len(data["val_idx_per_session"]) == 1:
            sess_name_debug, sess_indices_debug = list(data["val_idx_per_session"].items())[0]
            indices_match = np.array_equal(val_idx, sess_indices_debug)
            _print_primary(f"\n[DEBUG R² Gap] Index comparison:")
            _print_primary(f"  val_idx shape: {val_idx.shape}, per_session shape: {sess_indices_debug.shape}")
            _print_primary(f"  Indices are IDENTICAL: {indices_match}")
            if not indices_match:
                _print_primary(f"  val_idx[:10]: {val_idx[:10]}")
                _print_primary(f"  per_session[:10]: {sess_indices_debug[:10]}")
                overlap = len(set(val_idx.tolist()) & set(sess_indices_debug.tolist()))
                _print_primary(f"  Overlap: {overlap}/{len(val_idx)} ({100*overlap/len(val_idx):.1f}%)")

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
    # Pass session_ids for consistency with main val loader (fixes R² gap issue)
    dataset = PairedConditionalDataset(
        data["ob"], data["pcx"], data["odors"], indices,
        session_ids=data.get("session_ids"),
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
        prefetch_factor=4 if num_workers > 0 else None,
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

    # For small datasets, reduce batch size to ensure we get at least 1 batch per GPU
    # This is especially important for distributed training where samples are split across GPUs
    train_batch_size = batch_size
    val_batch_size = batch_size
    test_batch_size = batch_size
    train_drop_last = True  # Default: drop incomplete batches for training stability
    if distributed:
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        # Samples per GPU
        train_per_gpu = len(train_dataset) // world_size
        val_per_gpu = len(val_dataset) // world_size
        test_per_gpu = len(test_dataset) // world_size
        # Ensure at least 1 batch per GPU for training
        if train_per_gpu < batch_size:
            # Either reduce batch size or set drop_last=False
            if train_per_gpu >= batch_size // 2:
                train_batch_size = max(1, train_per_gpu)
            else:
                # Very few samples - don't drop last batch
                train_drop_last = False
        # Ensure at least 1 batch per GPU for val/test
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
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train_drop_last,
        persistent_workers=persistent,
        prefetch_factor=4 if num_workers > 0 else None,
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
        prefetch_factor=4 if num_workers > 0 else None,
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
        prefetch_factor=4 if num_workers > 0 else None,
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


# =============================================================================
# PFC Sliding Window Dataset (for continuous-style training from trial data)
# =============================================================================

class SlidingWindowPFCDataset(Dataset):
    """PyTorch Dataset for PFC data with sliding windows within trials.

    Creates overlapping windows from trial-based recordings for training
    neural translation models with more data augmentation.

    Args:
        pfc: PFC signals [n_trials, n_channels, n_samples] = [N, 64, 6250]
        ca1: CA1 signals [n_trials, n_channels, n_samples] = [N, 32, 6250]
        trial_indices: Which trials to include (e.g., train_idx)
        trial_types: Trial type labels for each trial
        window_size: Window size in samples (default: 2500 = 2 seconds at 1250 Hz)
        stride: Stride between windows in samples (default: window_size // 2)
        zscore_per_window: Whether to z-score each window independently
        session_ids: Optional session IDs for each trial
    """
    def __init__(
        self,
        pfc: np.ndarray,
        ca1: np.ndarray,
        trial_indices: np.ndarray,
        trial_types: Optional[np.ndarray] = None,
        window_size: int = 2500,
        stride: Optional[int] = None,
        zscore_per_window: bool = False,
        session_ids: Optional[np.ndarray] = None,
    ):
        self.pfc = pfc.astype(np.float32)
        self.ca1 = ca1.astype(np.float32)
        self.trial_indices = trial_indices
        self.trial_types = trial_types
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size // 2
        self.zscore_per_window = zscore_per_window
        self.session_ids = session_ids

        # Get trial length
        self.trial_length = pfc.shape[2]

        # Calculate number of windows per trial
        self.windows_per_trial = max(0, (self.trial_length - window_size) // self.stride + 1)

        # Build index mapping: global_idx -> (trial_idx, local_window_idx)
        self.window_mapping = []
        for trial_idx in trial_indices:
            for local_idx in range(self.windows_per_trial):
                self.window_mapping.append((trial_idx, local_idx))

        _print_primary(f"SlidingWindowPFCDataset: {len(trial_indices)} trials, "
              f"{self.windows_per_trial} windows/trial, "
              f"{len(self.window_mapping)} total windows")

    def __len__(self) -> int:
        return len(self.window_mapping)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        trial_idx, local_idx = self.window_mapping[idx]

        start = local_idx * self.stride
        end = start + self.window_size

        pfc_window = self.pfc[trial_idx, :, start:end].copy()
        ca1_window = self.ca1[trial_idx, :, start:end].copy()

        if self.zscore_per_window:
            pfc_window = (pfc_window - pfc_window.mean(axis=1, keepdims=True)) / \
                         (pfc_window.std(axis=1, keepdims=True) + 1e-8)
            ca1_window = (ca1_window - ca1_window.mean(axis=1, keepdims=True)) / \
                         (ca1_window.std(axis=1, keepdims=True) + 1e-8)

        # Return trial type as label (for compatibility with training loop)
        label = self.trial_types[trial_idx] if self.trial_types is not None else 0
        return torch.from_numpy(pfc_window), torch.from_numpy(ca1_window), int(label)

    def get_session_id(self, idx: int) -> Optional[int]:
        """Get session ID for a given window index."""
        if self.session_ids is None:
            return None
        trial_idx, _ = self.window_mapping[idx]
        return self.session_ids[trial_idx]


class MultiSessionSlidingWindowPFCDataset(Dataset):
    """Dataset combining multiple PFC sessions with sliding windows.

    Groups trials by session and provides session-aware batching.

    Args:
        pfc: PFC signals [n_trials, n_channels, n_samples]
        ca1: CA1 signals [n_trials, n_channels, n_samples]
        trial_indices: Which trials to include
        session_ids: Session ID for each trial
        trial_types: Trial type labels
        window_size: Window size in samples
        stride: Stride between windows
        zscore_per_window: Whether to z-score each window
        idx_to_session: Mapping from session index to session name
    """
    def __init__(
        self,
        pfc: np.ndarray,
        ca1: np.ndarray,
        trial_indices: np.ndarray,
        session_ids: np.ndarray,
        trial_types: Optional[np.ndarray] = None,
        window_size: int = 2500,
        stride: Optional[int] = None,
        zscore_per_window: bool = False,
        idx_to_session: Optional[Dict[int, str]] = None,
    ):
        self.pfc = pfc.astype(np.float32)
        self.ca1 = ca1.astype(np.float32)
        self.trial_indices = trial_indices
        self.session_ids = session_ids
        self.trial_types = trial_types
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size // 2
        self.zscore_per_window = zscore_per_window
        self.idx_to_session = idx_to_session or {}

        self.trial_length = pfc.shape[2]
        self.windows_per_trial = max(0, (self.trial_length - window_size) // self.stride + 1)

        # Build index mapping: global_idx -> (trial_idx, local_window_idx, session_idx)
        self.window_mapping = []
        session_window_counts = {}

        for trial_idx in trial_indices:
            sess_idx = session_ids[trial_idx]
            for local_idx in range(self.windows_per_trial):
                self.window_mapping.append((trial_idx, local_idx, sess_idx))
                session_window_counts[sess_idx] = session_window_counts.get(sess_idx, 0) + 1

        # Store unique sessions
        self.unique_sessions = sorted(set(session_ids[trial_indices]))

        _print_primary(f"MultiSessionSlidingWindowPFCDataset: {len(trial_indices)} trials, "
              f"{len(self.unique_sessions)} sessions, "
              f"{len(self.window_mapping)} total windows")
        for sess_idx, count in sorted(session_window_counts.items()):
            sess_name = self.idx_to_session.get(sess_idx, f"session_{sess_idx}")
            _print_primary(f"  {sess_name}: {count} windows")

    def __len__(self) -> int:
        return len(self.window_mapping)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        trial_idx, local_idx, sess_idx = self.window_mapping[idx]

        start = local_idx * self.stride
        end = start + self.window_size

        pfc_window = self.pfc[trial_idx, :, start:end].copy()
        ca1_window = self.ca1[trial_idx, :, start:end].copy()

        if self.zscore_per_window:
            pfc_window = (pfc_window - pfc_window.mean(axis=1, keepdims=True)) / \
                         (pfc_window.std(axis=1, keepdims=True) + 1e-8)
            ca1_window = (ca1_window - ca1_window.mean(axis=1, keepdims=True)) / \
                         (ca1_window.std(axis=1, keepdims=True) + 1e-8)

        label = self.trial_types[trial_idx] if self.trial_types is not None else 0
        return torch.from_numpy(pfc_window), torch.from_numpy(ca1_window), int(label)

    def get_session_name(self, sess_idx: int) -> str:
        """Get session name by index."""
        return self.idx_to_session.get(sess_idx, f"session_{sess_idx}")


def create_pfc_sliding_window_dataloaders(
    data: Dict[str, Any],
    window_size: int = 2500,
    stride: Optional[int] = None,
    val_stride: Optional[int] = None,
    batch_size: int = 32,
    zscore_per_window: bool = False,
    num_workers: int = 4,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    use_sessions: bool = False,
    distributed: bool = False,
) -> Dict[str, Any]:
    """Create DataLoaders for PFC data with sliding windows.

    Args:
        data: Output from prepare_pfc_data()
        window_size: Window size in samples (default: 2500 = 2s at 1250Hz)
        stride: Training stride (default: window_size // 2)
        val_stride: Validation stride (default: window_size for non-overlapping)
        batch_size: Batch size for DataLoader
        zscore_per_window: Whether to z-score each window
        num_workers: Number of DataLoader workers
        persistent_workers: Keep workers alive between batches
        prefetch_factor: Prefetch multiplier
        use_sessions: If True, use session-aware dataset
        distributed: If True, use DistributedSampler for DDP

    Returns:
        Dictionary with train/val/test loaders and metadata
    """
    pfc = data["pfc"]
    ca1 = data["ca1"]
    trial_types = data.get("trial_types")
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    stride = stride if stride is not None else window_size // 2
    val_stride = val_stride if val_stride is not None else window_size

    # Get session info if available
    session_ids = None
    idx_to_session = None
    if use_sessions and "split_info" in data and data["split_info"] is not None:
        # Session-based split - load session IDs
        session_ids, _, idx_to_session = load_pfc_session_ids(num_trials=pfc.shape[0])

    # Create datasets
    if use_sessions and session_ids is not None:
        train_dataset = MultiSessionSlidingWindowPFCDataset(
            pfc, ca1, train_idx, session_ids, trial_types,
            window_size=window_size, stride=stride,
            zscore_per_window=zscore_per_window,
            idx_to_session=idx_to_session,
        )
        val_dataset = MultiSessionSlidingWindowPFCDataset(
            pfc, ca1, val_idx, session_ids, trial_types,
            window_size=window_size, stride=val_stride,
            zscore_per_window=zscore_per_window,
            idx_to_session=idx_to_session,
        )
        test_dataset = MultiSessionSlidingWindowPFCDataset(
            pfc, ca1, test_idx, session_ids, trial_types,
            window_size=window_size, stride=val_stride,
            zscore_per_window=zscore_per_window,
            idx_to_session=idx_to_session,
        )
    else:
        train_dataset = SlidingWindowPFCDataset(
            pfc, ca1, train_idx, trial_types,
            window_size=window_size, stride=stride,
            zscore_per_window=zscore_per_window,
            session_ids=session_ids,
        )
        val_dataset = SlidingWindowPFCDataset(
            pfc, ca1, val_idx, trial_types,
            window_size=window_size, stride=val_stride,
            zscore_per_window=zscore_per_window,
            session_ids=session_ids,
        )
        test_dataset = SlidingWindowPFCDataset(
            pfc, ca1, test_idx, trial_types,
            window_size=window_size, stride=val_stride,
            zscore_per_window=zscore_per_window,
            session_ids=session_ids,
        )

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset, seed=42) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=42) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=42) if distributed else None

    # For small datasets, adjust batch size to ensure at least 1 batch per GPU
    train_batch_size = batch_size
    train_drop_last = True  # Default: drop incomplete batches for training stability
    if distributed:
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        train_per_gpu = len(train_dataset) // world_size
        if train_per_gpu < batch_size:
            if train_per_gpu >= batch_size // 2:
                train_batch_size = max(1, train_per_gpu)
            else:
                train_drop_last = False

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle when using sampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=train_drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    _print_primary(f"\nPFC Sliding Window DataLoaders created:")
    _print_primary(f"  Window size: {window_size} samples")
    _print_primary(f"  Train stride: {stride}, Val/Test stride: {val_stride}")
    _print_primary(f"  Train: {len(train_dataset)} windows")
    _print_primary(f"  Val: {len(val_dataset)} windows")
    _print_primary(f"  Test: {len(test_dataset)} windows")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "window_size": window_size,
        "stride": stride,
        "val_stride": val_stride,
        "n_channels_source": pfc.shape[1],
        "n_channels_target": ca1.shape[1],
    }


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


# =============================================================================
# PCx1 Continuous LFP Dataset (1kHz)
# =============================================================================

# Paths for PCx1 continuous dataset
PCX1_CONTINUOUS_PATH = _DATA_DIR / "PCx1" / "extracted" / "continuous_1khz"
PCX1_SAMPLING_RATE = 1000  # Hz
PCX1_N_CHANNELS = 32  # per region (OB and PCx)


def list_pcx1_sessions(path: Path = PCX1_CONTINUOUS_PATH) -> List[str]:
    """List available PCx1 continuous sessions.

    Returns:
        List of session names (e.g., ['141208-1', '141208-2', ...])
    """
    if not path.exists():
        raise FileNotFoundError(f"PCx1 continuous data path not found: {path}")

    sessions = sorted([
        d.name for d in path.iterdir()
        if d.is_dir() and (d / 'OB.npy').exists()
    ])
    return sessions


def load_pcx1_session_metadata(
    session: str,
    path: Path = PCX1_CONTINUOUS_PATH
) -> Dict[str, Any]:
    """Load metadata for a PCx1 session.

    Args:
        session: Session name (e.g., '141208-1')
        path: Base path to continuous_1khz folder

    Returns:
        Dictionary containing session metadata
    """
    session_path = path / session
    metadata_path = session_path / 'metadata.json'

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found for session {session}: {metadata_path}")

    with open(metadata_path) as f:
        return json.load(f)


def load_pcx1_session(
    session: str,
    path: Path = PCX1_CONTINUOUS_PATH,
    load_resp: bool = True,
    load_trials: bool = True,
    zscore: bool = False,
) -> Dict[str, Any]:
    """Load a single PCx1 session's continuous data.

    Args:
        session: Session name (e.g., '141208-1')
        path: Base path to continuous_1khz folder
        load_resp: Whether to load respiration signal
        load_trials: Whether to load trial info CSVs
        zscore: Whether to z-score normalize the data (per channel)

    Returns:
        Dictionary containing:
            - ob: OB signals [32, n_samples]
            - pcx: PCx signals [32, n_samples]
            - resp: Respiration signal [n_samples] (if load_resp=True)
            - breath_times: Array of breath timestamps (if exists)
            - trials: DataFrame of valid trials (if load_trials=True)
            - metadata: Session metadata dict
            - session: Session name
            - sampling_rate: 1000 Hz
    """
    session_path = path / session

    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_path}")

    # Load neural data
    ob = np.load(session_path / 'OB.npy').astype(np.float32)
    pcx = np.load(session_path / 'PCx.npy').astype(np.float32)

    # Optional z-score normalization (per channel)
    if zscore:
        ob = (ob - ob.mean(axis=1, keepdims=True)) / (ob.std(axis=1, keepdims=True) + 1e-8)
        pcx = (pcx - pcx.mean(axis=1, keepdims=True)) / (pcx.std(axis=1, keepdims=True) + 1e-8)

    result = {
        'ob': ob,
        'pcx': pcx,
        'session': session,
        'sampling_rate': PCX1_SAMPLING_RATE,
    }

    # Load respiration
    if load_resp:
        resp_path = session_path / 'Resp.npy'
        if resp_path.exists():
            result['resp'] = np.load(resp_path).astype(np.float32)

        breath_path = session_path / 'breath_times.npy'
        if breath_path.exists():
            result['breath_times'] = np.load(breath_path)

    # Load trial info
    if load_trials:
        trials_path = session_path / 'valid_trials.csv'
        if trials_path.exists():
            result['trials'] = pd.read_csv(trials_path)

    # Load metadata
    metadata_path = session_path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            result['metadata'] = json.load(f)

    return result


def load_pcx1_all_sessions(
    path: Path = PCX1_CONTINUOUS_PATH,
    sessions: Optional[List[str]] = None,
    zscore: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Load all PCx1 sessions.

    Args:
        path: Base path to continuous_1khz folder
        sessions: List of session names to load (None = all)
        zscore: Whether to z-score normalize

    Returns:
        Dictionary mapping session name to session data
    """
    if sessions is None:
        sessions = list_pcx1_sessions(path)

    all_data = {}
    for session in sessions:
        _print_primary(f"Loading session {session}...")
        all_data[session] = load_pcx1_session(session, path, zscore=zscore)

    return all_data


class ContinuousLFPDataset(Dataset):
    """PyTorch Dataset for continuous LFP data with sliding window.

    Creates windows from continuous recordings for training neural translation models.

    Args:
        ob: OB signals [n_channels, n_samples] or [n_samples, n_channels]
        pcx: PCx signals [n_channels, n_samples] or [n_samples, n_channels]
        window_size: Window size in samples (default: 5000 = 5 seconds at 1kHz)
        stride: Stride between windows in samples (default: window_size // 2)
        zscore_per_window: Whether to z-score each window independently
        channels_first: If True, expect input as [n_channels, n_samples]
    """
    def __init__(
        self,
        ob: np.ndarray,
        pcx: np.ndarray,
        window_size: int = 5000,
        stride: Optional[int] = None,
        zscore_per_window: bool = False,
        channels_first: bool = True,
    ):
        # Ensure channels-first format: [n_channels, n_samples]
        if not channels_first:
            ob = ob.T
            pcx = pcx.T

        self.ob = ob.astype(np.float32)
        self.pcx = pcx.astype(np.float32)
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size // 2
        self.zscore_per_window = zscore_per_window

        # Calculate number of windows
        n_samples = self.ob.shape[1]
        self.n_windows = max(0, (n_samples - window_size) // self.stride + 1)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        start = idx * self.stride
        end = start + self.window_size

        ob_window = self.ob[:, start:end].copy()
        pcx_window = self.pcx[:, start:end].copy()

        if self.zscore_per_window:
            ob_window = (ob_window - ob_window.mean(axis=1, keepdims=True)) / \
                        (ob_window.std(axis=1, keepdims=True) + 1e-8)
            pcx_window = (pcx_window - pcx_window.mean(axis=1, keepdims=True)) / \
                         (pcx_window.std(axis=1, keepdims=True) + 1e-8)

        # Return dummy label 0 for compatibility with training loop
        return torch.from_numpy(ob_window), torch.from_numpy(pcx_window), 0


class MultiSessionContinuousDataset(Dataset):
    """PyTorch Dataset combining multiple sessions with sliding windows.

    Useful for training on data from multiple recording sessions.

    Args:
        sessions_data: List of session data dicts (from load_pcx1_session)
        window_size: Window size in samples
        stride: Stride between windows
        zscore_per_window: Whether to z-score each window
    """
    def __init__(
        self,
        sessions_data: List[Dict[str, Any]],
        window_size: int = 5000,
        stride: Optional[int] = None,
        zscore_per_window: bool = False,
    ):
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size // 2
        self.zscore_per_window = zscore_per_window

        # Build index mapping: global_idx -> (session_idx, local_window_idx)
        self.sessions = []
        self.window_mapping = []

        for sess_idx, sess_data in enumerate(sessions_data):
            ob = sess_data['ob'].astype(np.float32)
            pcx = sess_data['pcx'].astype(np.float32)
            n_samples = ob.shape[1]
            n_windows = max(0, (n_samples - window_size) // self.stride + 1)

            self.sessions.append({
                'ob': ob,
                'pcx': pcx,
                'name': sess_data.get('session', f'session_{sess_idx}'),
                'n_windows': n_windows,
            })

            for local_idx in range(n_windows):
                self.window_mapping.append((sess_idx, local_idx))

        _print_primary(f"MultiSessionContinuousDataset: {len(sessions_data)} sessions, "
              f"{len(self.window_mapping)} total windows")

    def __len__(self) -> int:
        return len(self.window_mapping)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sess_idx, local_idx = self.window_mapping[idx]
        sess = self.sessions[sess_idx]

        start = local_idx * self.stride
        end = start + self.window_size

        ob_window = sess['ob'][:, start:end].copy()
        pcx_window = sess['pcx'][:, start:end].copy()

        if self.zscore_per_window:
            ob_window = (ob_window - ob_window.mean(axis=1, keepdims=True)) / \
                        (ob_window.std(axis=1, keepdims=True) + 1e-8)
            pcx_window = (pcx_window - pcx_window.mean(axis=1, keepdims=True)) / \
                         (pcx_window.std(axis=1, keepdims=True) + 1e-8)

        return torch.from_numpy(ob_window), torch.from_numpy(pcx_window), sess_idx

    def get_session_name(self, sess_idx: int) -> str:
        """Get session name by index."""
        return self.sessions[sess_idx]['name']


def create_pcx1_dataloaders(
    train_sessions: List[str],
    val_sessions: List[str],
    test_sessions: Optional[List[str]] = None,
    window_size: int = 5000,
    stride: Optional[int] = None,
    val_stride: Optional[int] = None,
    batch_size: int = 32,
    zscore_per_window: bool = False,  # Model handles input normalization internally
    num_workers: int = 4,
    path: Path = PCX1_CONTINUOUS_PATH,
    separate_val_sessions: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    loso_mode: bool = False,
    loso_val_ratio: float = 0.3,
    seed: int = 42,
    distributed: bool = False,  # Use DistributedSampler for multi-GPU training
) -> Dict[str, Any]:
    """Create DataLoaders for PCx1 continuous data with session-based splits.

    Args:
        train_sessions: List of session names for training
        val_sessions: List of session names for validation (ignored if loso_mode=True)
        test_sessions: List of session names for testing (optional)
        window_size: Window size in samples (5000 = 5 seconds)
        stride: Stride between windows for training (default: window_size // 2)
        val_stride: Stride for validation (default: same as stride). Use larger for faster eval.
        batch_size: Batch size for DataLoader
        zscore_per_window: Whether to z-score each window
        num_workers: Number of DataLoader workers
        path: Base path to continuous_1khz folder
        separate_val_sessions: If True, also create per-session val loaders
        persistent_workers: Keep workers alive between epochs (faster)
        prefetch_factor: Number of batches to prefetch per worker
        loso_mode: If True, combine all train_sessions and do 70/30 data split (not session split)
        loso_val_ratio: Fraction of data for validation in LOSO mode (default 0.3 = 30%)
        seed: Random seed for LOSO data split

    Returns:
        Dictionary with 'train', 'val', and optionally 'test' and 'val_sessions' DataLoaders
    """
    # Default strides
    if stride is None:
        stride = window_size // 2
    if val_stride is None:
        val_stride = stride

    # persistent_workers requires num_workers > 0
    use_persistent = persistent_workers and num_workers > 0

    # Common DataLoader kwargs for speed
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    if loso_mode:
        # LOSO MODE: Split raw samples TEMPORALLY first, then create windows
        # This prevents data leakage from overlapping windows
        _print_primary(f"Loading {len(train_sessions)} training sessions for LOSO...")
        all_train_data = [load_pcx1_session(s, path) for s in train_sessions]

        # Split each session's raw samples temporally (70% train, 30% val)
        # This ensures NO overlap between train and val windows
        train_portions = []
        val_portions = []
        total_train_samples = 0
        total_val_samples = 0

        for sess_data in all_train_data:
            n_samples = sess_data['ob'].shape[1]
            split_point = int(n_samples * (1.0 - loso_val_ratio))

            # First 70% for training
            train_portions.append({
                'session': sess_data['session'],
                'ob': sess_data['ob'][:, :split_point],
                'pcx': sess_data['pcx'][:, :split_point],
            })
            total_train_samples += split_point

            # Last 30% for validation
            val_portions.append({
                'session': sess_data['session'],
                'ob': sess_data['ob'][:, split_point:],
                'pcx': sess_data['pcx'][:, split_point:],
            })
            total_val_samples += (n_samples - split_point)

        _print_primary(f"  Temporal split: {total_train_samples} train samples, {total_val_samples} val samples")

        # Now create windows from each portion (no overlap between train/val)
        train_dataset = MultiSessionContinuousDataset(
            train_portions, window_size, stride, zscore_per_window
        )
        val_dataset = MultiSessionContinuousDataset(
            val_portions, window_size, stride, zscore_per_window
        )

        _print_primary(f"  LOSO windows: {len(train_dataset)} train, {len(val_dataset)} val (NO overlap)")

        # Create samplers for distributed training
        train_sampler = DistributedSampler(train_dataset, seed=seed) if distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=seed) if distributed else None

        dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=(train_sampler is None),  # Only shuffle if not using sampler
                sampler=train_sampler,
                **loader_kwargs,
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=val_sampler,
                **loader_kwargs,
            ),
        }

        if distributed:
            _print_primary(f"  Using DistributedSampler for {dist.get_world_size()} GPUs")

        # No per-session val loaders in LOSO mode (data is mixed)

    else:
        # STANDARD MODE: Separate session-based splits
        _print_primary("Loading training sessions...")
        train_data = [load_pcx1_session(s, path) for s in train_sessions]
        _print_primary("Loading validation sessions...")
        val_data = [load_pcx1_session(s, path) for s in val_sessions]

        # Create datasets (val can use larger stride for faster eval)
        train_dataset = MultiSessionContinuousDataset(
            train_data, window_size, stride, zscore_per_window
        )
        val_dataset = MultiSessionContinuousDataset(
            val_data, window_size, val_stride, zscore_per_window
        )

        # Create samplers for distributed training
        train_sampler = DistributedSampler(train_dataset, seed=seed) if distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=seed) if distributed else None

        dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                **loader_kwargs,
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=val_sampler,
                **loader_kwargs,
            ),
        }

        if distributed:
            _print_primary(f"  Using DistributedSampler for {dist.get_world_size()} GPUs")

        # Create per-session validation loaders for separate evaluation
        if separate_val_sessions:
            val_sessions_loaders = {}
            for sess_data in val_data:
                sess_name = sess_data['session']
                sess_dataset = ContinuousLFPDataset(
                    ob=sess_data['ob'],
                    pcx=sess_data['pcx'],
                    window_size=window_size,
                    stride=val_stride,  # Use val_stride for faster eval
                    zscore_per_window=zscore_per_window,
                )
                val_sessions_loaders[sess_name] = DataLoader(
                    sess_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    **loader_kwargs,
                )
            dataloaders['val_sessions'] = val_sessions_loaders

    if test_sessions:
        _print_primary("Loading test sessions...")
        test_data = [load_pcx1_session(s, path) for s in test_sessions]
        test_dataset = MultiSessionContinuousDataset(
            test_data, window_size, val_stride, zscore_per_window  # Use val_stride for test too
        )
        test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=seed) if distributed else None
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler,
            **loader_kwargs,
        )

        # Create per-session test loaders for LOSO evaluation
        test_sessions_loaders = {}
        for sess_data in test_data:
            sess_name = sess_data['session']
            sess_dataset = ContinuousLFPDataset(
                ob=sess_data['ob'],
                pcx=sess_data['pcx'],
                window_size=window_size,
                stride=val_stride,
                zscore_per_window=zscore_per_window,
            )
            test_sessions_loaders[sess_name] = DataLoader(
                sess_dataset,
                batch_size=batch_size,
                shuffle=False,
                **loader_kwargs,
            )
        dataloaders['test_sessions'] = test_sessions_loaders

    return dataloaders


def get_pcx1_session_splits(
    seed: int = 42,
    n_val: int = 2,
    n_test: int = 2,
    path: Path = PCX1_CONTINUOUS_PATH,
) -> Tuple[List[str], List[str], List[str]]:
    """Get random session-based train/val/test splits for PCx1.

    Args:
        seed: Random seed for reproducibility
        n_val: Number of validation sessions
        n_test: Number of test sessions
        path: Base path to continuous_1khz folder

    Returns:
        Tuple of (train_sessions, val_sessions, test_sessions)
    """
    sessions = list_pcx1_sessions(path)
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)

    test_sessions = sessions[:n_test]
    val_sessions = sessions[n_test:n_test + n_val]
    train_sessions = sessions[n_test + n_val:]

    _print_primary(f"PCx1 Session Split (seed={seed}):")
    _print_primary(f"  Train: {train_sessions}")
    _print_primary(f"  Val:   {val_sessions}")
    _print_primary(f"  Test:  {test_sessions}")

    return train_sessions, val_sessions, test_sessions


# =============================================================================
# DANDI 000623 Movie Dataset (Human iEEG during movie watching)
# =============================================================================
# Reference: Keles et al., 2024, Scientific Data
# "Multimodal single-neuron, intracranial EEG, and fMRI brain responses
# during movie watching in human patients"

def check_dandi_dependencies() -> bool:
    """Check if DANDI/NWB dependencies are available."""
    try:
        import pynwb
        import h5py
        return True
    except ImportError:
        return False


def download_dandi_dataset(
    dandiset_id: str = DANDI_DANDISET_ID,
    output_dir: Path = DANDI_RAW_PATH,
    version: str = "draft",
) -> Path:
    """Download DANDI dataset using dandi-cli.

    Args:
        dandiset_id: DANDI dataset ID (default: "000623")
        output_dir: Directory to save downloaded files
        version: Dataset version ("draft" or specific version)

    Returns:
        Path to downloaded dataset directory
    """
    import subprocess

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use dandi CLI to download
    cmd = [
        "dandi", "download",
        f"https://dandiarchive.org/dandiset/{dandiset_id}/{version}",
        "-o", str(output_dir),
    ]

    _print_primary(f"Downloading DANDI dataset {dandiset_id}...")
    _print_primary(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        _print_primary(f"Download failed: {result.stderr}")
        raise RuntimeError(f"Failed to download DANDI dataset: {result.stderr}")

    _print_primary(f"Dataset downloaded to: {output_dir}")
    return output_dir / dandiset_id


def list_dandi_nwb_files(
    data_dir: Path = DANDI_RAW_PATH,
    dandiset_id: str = DANDI_DANDISET_ID,
) -> List[Path]:
    """List all NWB files in the DANDI dataset directory.

    Args:
        data_dir: Base directory containing the dataset
        dandiset_id: DANDI dataset ID

    Returns:
        List of paths to NWB files
    """
    dataset_path = data_dir / dandiset_id
    if not dataset_path.exists():
        dataset_path = data_dir  # Try direct path

    nwb_files = sorted(dataset_path.glob("**/*.nwb"))

    if not nwb_files:
        raise FileNotFoundError(f"No NWB files found in {dataset_path}")

    return nwb_files


def load_dandi_nwb_file(
    nwb_path: Path,
    load_lfp: bool = True,
    load_ieeg: bool = True,
    load_spikes: bool = False,
    load_behavior: bool = False,
) -> Dict[str, Any]:
    """Load data from a single DANDI 000623 NWB file.

    Args:
        nwb_path: Path to the NWB file
        load_lfp: Whether to load LFP data (microwires)
        load_ieeg: Whether to load iEEG data (macroelectrodes)
        load_spikes: Whether to load spike times
        load_behavior: Whether to load behavioral data (eye tracking, etc.)

    Returns:
        Dictionary containing loaded data with keys:
            - subject_id: Subject identifier
            - lfp: LFP data array [n_channels, n_samples] (if load_lfp=True)
            - lfp_electrodes: Electrode info for LFP channels
            - ieeg: iEEG data array [n_channels, n_samples] (if load_ieeg=True)
            - ieeg_electrodes: Electrode info for iEEG channels
            - spikes: Dict of unit_id -> spike_times (if load_spikes=True)
            - behavior: Behavioral data dict (if load_behavior=True)
            - sampling_rate: Sampling rate in Hz
            - metadata: Session/subject metadata
    """
    if not check_dandi_dependencies():
        raise ImportError("pynwb and h5py required. Install with: pip install pynwb h5py")

    from pynwb import NWBHDF5IO

    nwb_path = Path(nwb_path)
    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    result = {
        "file_path": str(nwb_path),
        "sampling_rate": DANDI_SAMPLING_RATE_HZ,
    }

    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()

        # Extract subject info
        result["subject_id"] = nwbfile.subject.subject_id if nwbfile.subject else nwb_path.stem
        result["metadata"] = {
            "session_id": nwbfile.session_id,
            "session_description": nwbfile.session_description,
            "experimenter": list(nwbfile.experimenter) if nwbfile.experimenter else [],
            "institution": nwbfile.institution,
        }

        # Load electrode information
        if nwbfile.electrodes is not None:
            electrodes_df = nwbfile.electrodes.to_dataframe()
            result["electrodes"] = electrodes_df

        # Load LFP data from ecephys processing module
        # DANDI 000623 uses LFP_micro (microwires) and LFP_macro (macroelectrodes)
        if load_lfp:
            try:
                if "ecephys" in nwbfile.processing:
                    ecephys = nwbfile.processing["ecephys"]
                    # Try LFP_micro first (microwire recordings)
                    for lfp_name in ["LFP_micro", "LFP", "lfp"]:
                        if lfp_name in ecephys.data_interfaces:
                            lfp_module = ecephys.data_interfaces[lfp_name]
                            for name, es in lfp_module.electrical_series.items():
                                lfp_data = es.data[:]  # [time, channels]
                                result["lfp"] = np.array(lfp_data, dtype=np.float32).T  # [channels, time]
                                result["lfp_rate"] = es.rate if hasattr(es, 'rate') and es.rate else DANDI_SAMPLING_RATE_HZ
                                if hasattr(es, 'electrodes') and es.electrodes is not None:
                                    result["lfp_electrodes"] = es.electrodes.to_dataframe()
                                break
                            break
            except Exception as e:
                _print_primary(f"Warning: Could not load LFP data: {e}")

        # Load iEEG/macro data (macroelectrodes)
        if load_ieeg:
            try:
                if "ecephys" in nwbfile.processing:
                    ecephys = nwbfile.processing["ecephys"]
                    # Try LFP_macro first (DANDI 000623 format)
                    for ieeg_name in ["LFP_macro", "iEEG", "ieeg"]:
                        if ieeg_name in ecephys.data_interfaces:
                            ieeg_module = ecephys.data_interfaces[ieeg_name]
                            for name, es in ieeg_module.electrical_series.items():
                                ieeg_data = es.data[:]  # [time, channels]
                                result["ieeg"] = np.array(ieeg_data, dtype=np.float32).T  # [channels, time]
                                result["ieeg_rate"] = es.rate if hasattr(es, 'rate') and es.rate else DANDI_SAMPLING_RATE_HZ
                                if hasattr(es, 'electrodes') and es.electrodes is not None:
                                    result["ieeg_electrodes"] = es.electrodes.to_dataframe()
                                break
                            break
            except Exception as e:
                _print_primary(f"Warning: Could not load iEEG data: {e}")

        # Load spike data
        if load_spikes:
            try:
                if nwbfile.units is not None:
                    units_df = nwbfile.units.to_dataframe()
                    result["spikes"] = {}
                    for idx, row in units_df.iterrows():
                        if "spike_times" in row:
                            result["spikes"][idx] = np.array(row["spike_times"])
                    result["units_metadata"] = units_df.drop(columns=["spike_times"], errors="ignore")
            except Exception as e:
                _print_primary(f"Warning: Could not load spike data: {e}")

        # Load behavioral data
        if load_behavior:
            try:
                if "behavior" in nwbfile.processing:
                    behavior = nwbfile.processing["behavior"]
                    result["behavior"] = {}
                    for name, ts in behavior.data_interfaces.items():
                        if hasattr(ts, 'data'):
                            result["behavior"][name] = {
                                "data": np.array(ts.data[:]),
                                "timestamps": np.array(ts.timestamps[:]) if hasattr(ts, 'timestamps') and ts.timestamps is not None else None,
                            }
            except Exception as e:
                _print_primary(f"Warning: Could not load behavioral data: {e}")

    return result


def get_electrodes_by_region(
    electrodes_df: pd.DataFrame,
    target_regions: Optional[List[str]] = None,
) -> Dict[str, List[int]]:
    """Group electrode indices by brain region.

    Args:
        electrodes_df: DataFrame with electrode information (must have 'location' column)
        target_regions: List of regions to extract (None = all regions)

    Returns:
        Dictionary mapping region name to list of electrode indices
    """
    if "location" not in electrodes_df.columns:
        # Try alternative column names
        location_col = None
        for col in ["brain_region", "region", "area", "group_name"]:
            if col in electrodes_df.columns:
                location_col = col
                break
        if location_col is None:
            raise ValueError("No location/region column found in electrodes DataFrame")
    else:
        location_col = "location"

    region_electrodes = {}

    for i, (idx, row) in enumerate(electrodes_df.iterrows()):
        region = str(row[location_col]).lower().strip()

        # Normalize region names (handle DANDI 000623 naming convention)
        # Regions like "Left amygdala", "Right hippocampus" -> "amygdala", "hippocampus"
        if "amygdala" in region or "amy" in region:
            normalized = "amygdala"
        elif "hippocampus" in region or "hpc" in region or "hipp" in region:
            normalized = "hippocampus"
        elif "vmpfc" in region or "vmfc" in region:
            normalized = "medial_frontal_cortex"
        elif "acc" in region:
            normalized = "acc"
        elif "presma" in region or "sma" in region:
            normalized = "presma"
        elif "frontal" in region or "mfc" in region or "pfc" in region:
            normalized = "medial_frontal_cortex"
        else:
            normalized = region

        if target_regions is None or normalized in target_regions:
            if normalized not in region_electrodes:
                region_electrodes[normalized] = []
            # Use sequential index (position in dataframe) not the original electrode ID
            region_electrodes[normalized].append(i)

    return region_electrodes


def extract_region_signals(
    data: np.ndarray,
    electrodes_df: pd.DataFrame,
    source_region: str,
    target_region: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract signals from source and target brain regions.

    Args:
        data: Full data array [n_channels, n_samples]
        electrodes_df: Electrode information DataFrame
        source_region: Source region name (e.g., "amygdala")
        target_region: Target region name (e.g., "hippocampus")

    Returns:
        Tuple of (source_signals, target_signals) arrays
    """
    region_indices = get_electrodes_by_region(electrodes_df)

    if source_region not in region_indices:
        raise ValueError(f"Source region '{source_region}' not found. Available: {list(region_indices.keys())}")
    if target_region not in region_indices:
        raise ValueError(f"Target region '{target_region}' not found. Available: {list(region_indices.keys())}")

    source_idx = region_indices[source_region]
    target_idx = region_indices[target_region]

    source_signals = data[source_idx, :]
    target_signals = data[target_idx, :]

    return source_signals, target_signals


def load_dandi_subject(
    subject_id: str,
    data_dir: Path = DANDI_RAW_PATH,
    source_region: str = "amygdala",
    target_region: str = "hippocampus",
    zscore: bool = True,
) -> Dict[str, Any]:
    """Load and preprocess data for a single DANDI subject.

    Args:
        subject_id: Subject ID (e.g., "sub-CS41")
        data_dir: Directory containing NWB files
        source_region: Source brain region for translation
        target_region: Target brain region for translation
        zscore: Whether to z-score normalize the data

    Returns:
        Dictionary containing preprocessed data
    """
    # Find NWB file for this subject
    nwb_files = list_dandi_nwb_files(data_dir)
    subject_file = None

    for f in nwb_files:
        if subject_id in f.stem or subject_id.replace("sub-", "") in f.stem:
            subject_file = f
            break

    if subject_file is None:
        raise FileNotFoundError(f"No NWB file found for subject {subject_id}")

    # Load the NWB file
    data = load_dandi_nwb_file(subject_file, load_lfp=True, load_ieeg=True)

    # Determine which data to use (prefer LFP, fallback to iEEG)
    if "lfp" in data and data["lfp"] is not None:
        neural_data = data["lfp"]
        electrodes = data.get("lfp_electrodes", data.get("electrodes"))
    elif "ieeg" in data and data["ieeg"] is not None:
        neural_data = data["ieeg"]
        electrodes = data.get("ieeg_electrodes", data.get("electrodes"))
    else:
        raise ValueError(f"No LFP or iEEG data found for subject {subject_id}")

    # Extract regions
    source_signals, target_signals = extract_region_signals(
        neural_data, electrodes, source_region, target_region
    )

    # Z-score normalization
    if zscore:
        source_signals = (source_signals - source_signals.mean(axis=1, keepdims=True)) / (
            source_signals.std(axis=1, keepdims=True) + 1e-8
        )
        target_signals = (target_signals - target_signals.mean(axis=1, keepdims=True)) / (
            target_signals.std(axis=1, keepdims=True) + 1e-8
        )

    return {
        "subject_id": subject_id,
        "source": source_signals.astype(np.float32),
        "target": target_signals.astype(np.float32),
        "source_region": source_region,
        "target_region": target_region,
        "sampling_rate": data["sampling_rate"],
        "n_source_channels": source_signals.shape[0],
        "n_target_channels": target_signals.shape[0],
        "n_samples": source_signals.shape[1],
        "metadata": data["metadata"],
    }


class DANDIMovieDataset(Dataset):
    """PyTorch Dataset for DANDI 000623 movie watching iEEG data.

    Provides sliding window segments for neural signal translation between
    brain regions during naturalistic movie watching.

    Args:
        subjects_data: List of subject data dicts from load_dandi_subject()
        window_size: Size of each window in samples (default: 5000 = 5s at 1kHz)
        stride: Stride between windows (default: 2500 = 2.5s)
        zscore_per_window: Whether to z-score each window independently
        verbose: Whether to print info messages
        n_source_channels: Fixed number of source channels (None = use min across subjects)
        n_target_channels: Fixed number of target channels (None = use min across subjects)
    """

    def __init__(
        self,
        subjects_data: List[Dict[str, Any]],
        window_size: int = 5000,
        stride: int = 2500,
        zscore_per_window: bool = False,
        verbose: bool = True,
        n_source_channels: Optional[int] = None,
        n_target_channels: Optional[int] = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.zscore_per_window = zscore_per_window

        # Determine channel counts - use minimum across subjects to ensure consistent shapes
        if n_source_channels is None:
            self.n_source_channels = min(s["n_source_channels"] for s in subjects_data) if subjects_data else 0
        else:
            self.n_source_channels = n_source_channels

        if n_target_channels is None:
            self.n_target_channels = min(s["n_target_channels"] for s in subjects_data) if subjects_data else 0
        else:
            self.n_target_channels = n_target_channels

        # Build index of all windows across subjects
        self.windows = []  # List of (subject_idx, start_sample)

        for subj_idx, subj_data in enumerate(subjects_data):
            n_samples = subj_data["n_samples"]
            n_windows = (n_samples - window_size) // stride + 1

            for w in range(n_windows):
                start = w * stride
                self.windows.append((subj_idx, start))

        self.subjects_data = subjects_data

        if verbose:
            _print_primary(f"DANDIMovieDataset: {len(subjects_data)} subjects, "
                  f"{len(self.windows)} windows, "
                  f"window_size={window_size}, stride={stride}, "
                  f"source_ch={self.n_source_channels}, target_ch={self.n_target_channels}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subj_idx, start = self.windows[idx]
        subj_data = self.subjects_data[subj_idx]

        end = start + self.window_size

        # Extract window and truncate to normalized channel counts
        # This ensures consistent tensor shapes across subjects
        source = subj_data["source"][:self.n_source_channels, start:end].copy()
        target = subj_data["target"][:self.n_target_channels, start:end].copy()

        # Optional per-window normalization
        if self.zscore_per_window:
            source = (source - source.mean(axis=1, keepdims=True)) / (
                source.std(axis=1, keepdims=True) + 1e-8
            )
            target = (target - target.mean(axis=1, keepdims=True)) / (
                target.std(axis=1, keepdims=True) + 1e-8
            )

        return {
            "source": torch.from_numpy(source).float(),
            "target": torch.from_numpy(target).float(),
            "subject_idx": torch.tensor(subj_idx),
            "start_sample": torch.tensor(start),
        }


def prepare_dandi_data(
    data_dir: Path = DANDI_RAW_PATH,
    source_region: str = "amygdala",
    target_region: str = "hippocampus",
    window_size: int = 5000,
    stride: int = 2500,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    zscore: bool = True,
    verbose: bool = True,
    min_channels: int = 12,
    # Explicit subject holdout for LOSO cross-validation
    val_subjects: Optional[List[str]] = None,
    test_subjects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Complete data preparation pipeline for DANDI 000623 dataset.

    Args:
        data_dir: Directory containing NWB files
        source_region: Source brain region for translation
        target_region: Target brain region for translation
        window_size: Window size in samples
        stride: Stride between windows
        train_ratio: Fraction of subjects for training (ignored if val_subjects provided)
        val_ratio: Fraction of subjects for validation (ignored if val_subjects provided)
        test_ratio: Fraction of subjects for testing (ignored if test_subjects provided)
        seed: Random seed for reproducibility
        zscore: Whether to z-score normalize the data
        verbose: Whether to print progress messages (set False for non-primary processes)
        min_channels: Minimum channels required in both source and target (subjects with fewer are excluded)
        val_subjects: Explicit list of subjects for validation (for LOSO). Overrides val_ratio.
        test_subjects: Explicit list of subjects for testing (for LOSO). Overrides test_ratio.

    Returns:
        Dictionary containing train/val/test datasets and metadata

    Note:
        For LOSO cross-validation, use val_subjects to specify the held-out subject.
        All other subjects will be used for training. Example:
            prepare_dandi_data(val_subjects=["sub-CS41"], test_subjects=[])
    """
    if verbose:
        _print_primary(f"Preparing DANDI 000623 dataset...")
        _print_primary(f"  Source region: {source_region}")
        _print_primary(f"  Target region: {target_region}")

    # Get available subjects
    nwb_files = list_dandi_nwb_files(data_dir)
    all_subject_ids = []

    for f in nwb_files:
        # Extract subject ID from filename
        stem = f.stem
        if "sub-" in stem:
            subj_id = stem.split("_")[0]  # Get sub-CSXX part
        else:
            subj_id = stem
        if subj_id not in all_subject_ids:
            all_subject_ids.append(subj_id)

    all_subject_ids = sorted(all_subject_ids)  # Sort for reproducibility

    if verbose:
        _print_primary(f"  Found {len(all_subject_ids)} subjects: {all_subject_ids}")

    # Determine train/val/test split
    if val_subjects is not None or test_subjects is not None:
        # Explicit subject holdout mode (for LOSO)
        val_subjects = val_subjects or []
        test_subjects = test_subjects or []

        # Validate that specified subjects exist
        for subj in val_subjects + test_subjects:
            if subj not in all_subject_ids:
                raise ValueError(
                    f"Subject '{subj}' not found in dataset. "
                    f"Available: {all_subject_ids}"
                )

        # Check for overlap between val and test
        overlap = set(val_subjects) & set(test_subjects)
        if overlap:
            raise ValueError(
                f"Overlap between val and test subjects: {overlap}. "
                f"This would cause data leakage!"
            )

        # Training = all subjects not in val or test
        holdout_subjects = set(val_subjects) | set(test_subjects)
        train_subjects = [s for s in all_subject_ids if s not in holdout_subjects]

        if verbose:
            _print_primary(f"  [LOSO MODE] Explicit subject holdout:")
            _print_primary(f"    Train subjects ({len(train_subjects)}): {train_subjects}")
            _print_primary(f"    Val subjects ({len(val_subjects)}): {val_subjects}")
            _print_primary(f"    Test subjects ({len(test_subjects)}): {test_subjects}")
    else:
        # Random split mode (default)
        rng = np.random.default_rng(seed)
        subject_ids_shuffled = all_subject_ids.copy()
        rng.shuffle(subject_ids_shuffled)

        n_train = int(len(subject_ids_shuffled) * train_ratio)
        n_val = int(len(subject_ids_shuffled) * val_ratio)

        train_subjects = subject_ids_shuffled[:n_train]
        val_subjects = subject_ids_shuffled[n_train:n_train + n_val]
        test_subjects = subject_ids_shuffled[n_train + n_val:]

        if verbose:
            _print_primary(f"  [Random Split] seed={seed}")
            _print_primary(f"    Train subjects ({len(train_subjects)}): {train_subjects}")
            _print_primary(f"    Val subjects ({len(val_subjects)}): {val_subjects}")
            _print_primary(f"    Test subjects ({len(test_subjects)}): {test_subjects}")

    # Load data for each split
    def load_subjects(subject_list, split_name=""):
        data_list = []
        for subj_id in subject_list:
            try:
                subj_data = load_dandi_subject(
                    subj_id, data_dir, source_region, target_region, zscore
                )
                # Filter out subjects with too few channels
                n_src = subj_data['n_source_channels']
                n_tgt = subj_data['n_target_channels']
                if n_src < min_channels or n_tgt < min_channels:
                    if verbose:
                        _print_primary(f"    Skipping {subj_id}: only {n_src} source / {n_tgt} target channels (min={min_channels})")
                    continue
                data_list.append(subj_data)
                if verbose:
                    _print_primary(f"    Loaded {subj_id}: source={n_src}ch, target={n_tgt}ch, "
                          f"{subj_data['n_samples']} samples")
            except Exception as e:
                if verbose:
                    _print_primary(f"    Warning: Could not load {subj_id}: {e}")
        return data_list

    if verbose:
        _print_primary("\nLoading training subjects...")
    train_data = load_subjects(train_subjects, "train")

    if verbose:
        _print_primary("\nLoading validation subjects...")
    val_data = load_subjects(val_subjects, "val")

    if verbose:
        _print_primary("\nLoading test subjects...")
    test_data = load_subjects(test_subjects, "test")

    # Compute minimum channel counts across ALL subjects for consistent batching
    all_data = train_data + val_data + test_data
    if not all_data:
        raise ValueError("No subjects could be loaded from DANDI dataset")

    min_source_channels = min(s["n_source_channels"] for s in all_data)
    min_target_channels = min(s["n_target_channels"] for s in all_data)

    if verbose:
        _print_primary(f"\nNormalized channel counts: source={min_source_channels}, target={min_target_channels}")

    # Create datasets with consistent channel counts
    # Special handling for LOSO mode: when test_subjects provided but no val_subjects,
    # create val set by TEMPORALLY splitting training data (not random window split)
    loso_mode_no_val = (test_subjects is not None and len(test_subjects) > 0 and
                        (val_subjects is None or len(val_subjects) == 0))

    if loso_mode_no_val and len(train_data) > 0:
        # LOSO mode: Split raw samples TEMPORALLY first, then create windows
        # This prevents data leakage from overlapping windows
        train_portions = []
        val_portions = []
        total_train_samples = 0
        total_val_samples = 0

        for subj_data in train_data:
            n_samples = subj_data['n_samples']
            split_point = int(n_samples * 0.7)  # 70% train, 30% val

            # First 70% for training
            train_portions.append({
                'subject_id': subj_data['subject_id'],
                'source': subj_data['source'][:, :split_point],
                'target': subj_data['target'][:, :split_point],
                'source_region': subj_data['source_region'],
                'target_region': subj_data['target_region'],
                'sampling_rate': subj_data['sampling_rate'],
                'n_source_channels': subj_data['n_source_channels'],
                'n_target_channels': subj_data['n_target_channels'],
                'n_samples': split_point,
            })
            total_train_samples += split_point

            # Last 30% for validation
            val_portions.append({
                'subject_id': subj_data['subject_id'],
                'source': subj_data['source'][:, split_point:],
                'target': subj_data['target'][:, split_point:],
                'source_region': subj_data['source_region'],
                'target_region': subj_data['target_region'],
                'sampling_rate': subj_data['sampling_rate'],
                'n_source_channels': subj_data['n_source_channels'],
                'n_target_channels': subj_data['n_target_channels'],
                'n_samples': n_samples - split_point,
            })
            total_val_samples += (n_samples - split_point)

        if verbose:
            _print_primary(f"  Temporal split: {total_train_samples} train samples, {total_val_samples} val samples")

        # Now create windows from each portion (no overlap between train/val)
        train_dataset = DANDIMovieDataset(
            train_portions, window_size, stride, verbose=False,
            n_source_channels=min_source_channels, n_target_channels=min_target_channels,
        )
        val_dataset = DANDIMovieDataset(
            val_portions, window_size, stride, verbose=False,
            n_source_channels=min_source_channels, n_target_channels=min_target_channels,
        )

        if verbose:
            _print_primary(f"DANDIMovieDataset [LOSO]: {len(train_data)} subjects, "
                  f"{len(train_dataset)} train + {len(val_dataset)} val windows (NO overlap), "
                  f"window_size={window_size}, stride={stride}, "
                  f"source_ch={min_source_channels}, target_ch={min_target_channels}")
    else:
        train_dataset = DANDIMovieDataset(
            train_data, window_size, stride, verbose=verbose,
            n_source_channels=min_source_channels, n_target_channels=min_target_channels,
        )
        val_dataset = DANDIMovieDataset(
            val_data, window_size, stride, verbose=verbose,
            n_source_channels=min_source_channels, n_target_channels=min_target_channels,
        )

    test_dataset = DANDIMovieDataset(
        test_data, window_size, stride, verbose=verbose,
        n_source_channels=min_source_channels, n_target_channels=min_target_channels,
    )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "source_region": source_region,
        "target_region": target_region,
        "sampling_rate": DANDI_SAMPLING_RATE_HZ,
        "window_size": window_size,
        "stride": stride,
        "dataset_type": DatasetType.DANDI_MOVIE,
        "n_source_channels": min_source_channels,
        "n_target_channels": min_target_channels,
    }


def create_dandi_dataloaders(
    data_dir: Path = DANDI_RAW_PATH,
    source_region: str = "amygdala",
    target_region: str = "hippocampus",
    window_size: int = 5000,
    stride: int = 2500,
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for DANDI 000623 dataset.

    Args:
        data_dir: Directory containing NWB files
        source_region: Source brain region
        target_region: Target brain region
        window_size: Window size in samples
        stride: Stride between windows
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        distributed: Whether to use distributed samplers
        seed: Random seed

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Prepare data
    data = prepare_dandi_data(
        data_dir=data_dir,
        source_region=source_region,
        target_region=target_region,
        window_size=window_size,
        stride=stride,
        seed=seed,
    )

    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]
    test_dataset = data["test_dataset"]

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset, seed=seed) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=seed) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=seed) if distributed else None

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
            **loader_kwargs,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            drop_last=False,
            **loader_kwargs,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
            **loader_kwargs,
        ),
    }

    _print_primary(f"\nDANDI DataLoaders created:")
    _print_primary(f"  Train: {len(train_dataset)} windows, {len(dataloaders['train'])} batches")
    _print_primary(f"  Val: {len(val_dataset)} windows, {len(dataloaders['val'])} batches")
    _print_primary(f"  Test: {len(test_dataset)} windows, {len(dataloaders['test'])} batches")

    return dataloaders


# =============================================================================
# Miller ECoG Library - Dataset and Loading Functions
# =============================================================================
# Inter-region cortical translation using subdural ECoG recordings.
# Electrodes are grouped by brain lobe (frontal, temporal, parietal, etc.)
# and paired as source -> target for neural signal translation.
#
# Data format (OSF/Neuromatch preprocessed NPZ files):
#   alldat = np.load('exp.npz', allow_pickle=True)['dat']
#   alldat.shape = (n_subjects, n_blocks)  — 2D object array
#   alldat[subj][block] -> dict with keys:
#     'V': float16 (n_samples, n_channels) — voltage data
#     'srate': int or [[int]] — sampling rate (format varies!)
#     'lobe': list of str — e.g. "Frontal Lobe", "Temporal Lobe"
#     'gyrus', 'Brodmann_Area', 'hemisphere', 'locs': anatomical info
#     (experiment-specific keys: 'dg', 'stim_id', 't_on', 't_off', etc.)
#
# IMPORTANT: Some experiments (fingerflex, joystick_track) store different
# patients as "blocks" of a single "subject" row. We detect this by checking
# if channel counts vary across blocks (different electrode grids = different
# patients). These blocks are unpacked into separate subjects automatically.


def _parse_ecog_lobe(raw_lobe: str) -> str:
    """Normalize a raw lobe annotation to a canonical lowercase name.

    The data contains strings like "Frontal Lobe", "Temporal Lobe",
    "Limbic Lobe", "Sub-lobar", "Anterior Lobe", etc.

    Returns:
        Canonical lobe name: frontal, temporal, parietal, occipital,
        limbic, sub-lobar, or the lowercased input if unrecognized.
    """
    s = raw_lobe.strip().lower()
    for known in ("frontal", "temporal", "parietal", "occipital", "limbic"):
        if known in s:
            return known
    if "sub-lobar" in s or "sub lobar" in s:
        return "sub-lobar"
    # "Anterior Lobe" (cerebellum) and other rare cases
    return s


def _parse_ecog_srate(dat: dict) -> int:
    """Extract sampling rate from a block dict, handling format variations.

    The srate field can be: int, float, np.int64, np.ndarray([[1000]]),
    or missing entirely.
    """
    sr = dat.get("srate", ECOG_SAMPLING_RATE_HZ)
    if isinstance(sr, np.ndarray):
        return int(sr.flat[0])
    return int(sr)


def _enumerate_ecog_recordings(
    alldat: np.ndarray,
) -> List[Tuple[int, int, str]]:
    """Enumerate all individual recordings (patient x block) in an experiment.

    Some experiments pack multiple patients as blocks of 1 "subject" row
    (detectable when channel counts differ across blocks). This function
    returns a flat list of (subject_row, block_col, recording_id) where
    recording_id is unique per patient.

    Returns:
        List of (row_idx, col_idx, recording_id) tuples
    """
    n_subj, n_blocks = alldat.shape
    recordings = []

    for si in range(n_subj):
        # Check if blocks within this subject have varying channel counts
        # (indicating they are actually different patients)
        block_channels = []
        for bi in range(n_blocks):
            dat = alldat[si][bi]
            if dat is not None and "V" in dat:
                block_channels.append(np.float32(dat["V"]).shape[1])

        blocks_are_patients = (
            len(set(block_channels)) > 1 and n_subj == 1
        )

        for bi in range(n_blocks):
            dat = alldat[si][bi]
            if dat is None:
                continue
            if "V" not in dat:
                continue

            if blocks_are_patients:
                rec_id = f"ecog_s{si:02d}_b{bi:02d}"
            elif n_subj > 1:
                rec_id = f"ecog_s{si:02d}"
            else:
                rec_id = f"ecog_s{si:02d}_b{bi:02d}"

            recordings.append((si, bi, rec_id))

    return recordings


def list_ecog_subjects(
    data_dir: Optional[Path] = None,
    experiment: str = "fingerflex",
) -> List[str]:
    """List available subject/recording identifiers for an ECoG experiment.

    For multi-subject experiments (faceshouses, motor_imagery, memory_nback),
    returns one ID per subject using block 0.
    For single-subject experiments where blocks are different patients
    (fingerflex, joystick_track), returns one ID per block.

    Args:
        data_dir: Directory containing .npz files (default: _ECOG_DATA_DIR)
        experiment: Experiment name

    Returns:
        List of recording identifiers like ["ecog_s00", "ecog_s01", ...] or
        ["ecog_s00_b00", "ecog_s00_b01", ...] for block-as-patient experiments.
    """
    data_dir = data_dir or _ECOG_DATA_DIR
    filepath = data_dir / f"{experiment}.npz"

    if not filepath.exists():
        raise FileNotFoundError(
            f"ECoG data file not found: {filepath}\n"
            f"Run: python scripts/download_ecog.py --experiments {experiment}"
        )

    alldat = np.load(filepath, allow_pickle=True)["dat"]
    recordings = _enumerate_ecog_recordings(alldat)

    # For multi-subject experiments, deduplicate to one entry per subject row
    # (use first block only). For block-as-patient, each block is unique.
    n_subj = alldat.shape[0]
    if n_subj > 1:
        seen = set()
        unique = []
        for si, bi, rec_id in recordings:
            if si not in seen:
                seen.add(si)
                unique.append(rec_id)
        return unique
    else:
        return [rec_id for _, _, rec_id in recordings]


def _get_ecog_region_channels(
    dat: dict,
    source_region: str,
    target_region: str,
) -> Tuple[List[int], List[int]]:
    """Get channel indices for source and target brain regions.

    Electrodes are grouped by their anatomical lobe annotation.
    Raw lobe strings (e.g. "Frontal Lobe") are normalized via _parse_ecog_lobe.

    Args:
        dat: Single block data dict from the .npz file
        source_region: Canonical lobe name (e.g., "frontal")
        target_region: Canonical lobe name (e.g., "temporal")

    Returns:
        Tuple of (source_channel_indices, target_channel_indices)
    """
    if "lobe" not in dat:
        return [], []

    V = np.float32(dat["V"])
    n_channels = V.shape[1]
    lobes = [_parse_ecog_lobe(str(l)) for l in dat["lobe"]]

    src = source_region.lower()
    tgt = target_region.lower()

    source_chs = [i for i, l in enumerate(lobes) if l == src and i < n_channels]
    target_chs = [i for i, l in enumerate(lobes) if l == tgt and i < n_channels]

    return source_chs, target_chs


def load_ecog_subject(
    subject_idx: int = 0,
    block_idx: int = 0,
    data_dir: Optional[Path] = None,
    experiment: str = "fingerflex",
    source_region: str = "frontal",
    target_region: str = "temporal",
    zscore: bool = True,
    min_channels: int = 4,
    _alldat: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Load a single recording's ECoG data for inter-region translation.

    Groups electrodes by brain lobe and creates source/target pairs.

    Args:
        subject_idx: Subject row index in the NPZ array
        block_idx: Block column index in the NPZ array
        data_dir: Directory containing .npz files
        experiment: Experiment name
        source_region: Source brain lobe (canonical name)
        target_region: Target brain lobe (canonical name)
        zscore: Whether to z-score normalize the data
        min_channels: Minimum channels required per region
        _alldat: Pre-loaded alldat array (avoids re-reading the file)

    Returns:
        Dict with keys: source, target, subject_id, n_source_channels,
        n_target_channels, n_samples, metadata. Returns None if the
        recording doesn't have enough channels in both regions.
    """
    if _alldat is None:
        data_dir = data_dir or _ECOG_DATA_DIR
        filepath = data_dir / f"{experiment}.npz"
        if not filepath.exists():
            raise FileNotFoundError(
                f"ECoG data file not found: {filepath}\n"
                f"Run: python scripts/download_ecog.py --experiments {experiment}"
            )
        _alldat = np.load(filepath, allow_pickle=True)["dat"]

    if subject_idx >= _alldat.shape[0]:
        return None
    if block_idx >= _alldat.shape[1]:
        return None

    dat = _alldat[subject_idx][block_idx]
    if dat is None or "V" not in dat:
        return None

    # Get voltage data (stored as float16, convert to float32)
    V = np.float32(dat["V"])  # (n_samples, n_channels)

    # Get region channel indices
    source_chs, target_chs = _get_ecog_region_channels(dat, source_region, target_region)

    if len(source_chs) < min_channels or len(target_chs) < min_channels:
        _print_primary(
            f"  S{subject_idx} B{block_idx}: insufficient channels "
            f"({source_region}={len(source_chs)}, {target_region}={len(target_chs)}), "
            f"need >= {min_channels} each, skipping"
        )
        return None

    # Extract source and target signals: (channels, samples)
    source = V[:, source_chs].T.copy()  # (n_source_channels, n_samples)
    target = V[:, target_chs].T.copy()  # (n_target_channels, n_samples)

    # Z-score per channel
    if zscore:
        src_mean = source.mean(axis=1, keepdims=True)
        src_std = source.std(axis=1, keepdims=True) + 1e-8
        source = (source - src_mean) / src_std

        tgt_mean = target.mean(axis=1, keepdims=True)
        tgt_std = target.std(axis=1, keepdims=True) + 1e-8
        target = (target - tgt_mean) / tgt_std

    subject_id = f"ecog_s{subject_idx:02d}" if _alldat.shape[0] > 1 else f"ecog_s{subject_idx:02d}_b{block_idx:02d}"
    srate = _parse_ecog_srate(dat)

    metadata = {
        "experiment": experiment,
        "subject_idx": subject_idx,
        "block_idx": block_idx,
        "source_region": source_region,
        "target_region": target_region,
        "n_source_channels_raw": len(source_chs),
        "n_target_channels_raw": len(target_chs),
        "srate": srate,
    }

    return {
        "source": source,
        "target": target,
        "subject_id": subject_id,
        "n_source_channels": len(source_chs),
        "n_target_channels": len(target_chs),
        "n_samples": V.shape[0],
        "metadata": metadata,
    }


class ECoGDataset(Dataset):
    """PyTorch Dataset for Miller ECoG Library inter-region translation.

    Provides sliding window segments from ECoG recordings for neural signal
    translation between brain regions (e.g., frontal -> temporal cortex).

    Args:
        subjects_data: List of subject data dicts from load_ecog_subject()
        window_size: Size of each window in samples (default: 5000 = 5s at 1kHz)
        stride: Stride between windows (default: 2500 = 2.5s)
        zscore_per_window: Whether to z-score each window independently
        verbose: Whether to print info messages
        n_source_channels: Fixed number of source channels (None = use min)
        n_target_channels: Fixed number of target channels (None = use min)
    """

    def __init__(
        self,
        subjects_data: List[Dict[str, Any]],
        window_size: int = 5000,
        stride: int = 2500,
        zscore_per_window: bool = False,
        verbose: bool = True,
        n_source_channels: Optional[int] = None,
        n_target_channels: Optional[int] = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.zscore_per_window = zscore_per_window

        # Determine channel counts - use minimum across subjects for consistent shapes
        if n_source_channels is None:
            self.n_source_channels = min(s["n_source_channels"] for s in subjects_data) if subjects_data else 0
        else:
            self.n_source_channels = n_source_channels

        if n_target_channels is None:
            self.n_target_channels = min(s["n_target_channels"] for s in subjects_data) if subjects_data else 0
        else:
            self.n_target_channels = n_target_channels

        # Build index of all windows across subjects
        self.windows = []  # List of (subject_idx, start_sample)

        for subj_idx, subj_data in enumerate(subjects_data):
            n_samples = subj_data["n_samples"]
            n_windows = max(0, (n_samples - window_size) // stride + 1)

            for w in range(n_windows):
                start = w * stride
                self.windows.append((subj_idx, start))

        self.subjects_data = subjects_data

        if verbose:
            _print_primary(
                f"ECoGDataset: {len(subjects_data)} recordings, "
                f"{len(self.windows)} windows, "
                f"window_size={window_size}, stride={stride}, "
                f"source_ch={self.n_source_channels}, target_ch={self.n_target_channels}"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subj_idx, start = self.windows[idx]
        subj_data = self.subjects_data[subj_idx]

        end = start + self.window_size

        # Extract window and truncate to normalized channel counts
        source = subj_data["source"][:self.n_source_channels, start:end].copy()
        target = subj_data["target"][:self.n_target_channels, start:end].copy()

        # Optional per-window normalization
        if self.zscore_per_window:
            source = (source - source.mean(axis=1, keepdims=True)) / (
                source.std(axis=1, keepdims=True) + 1e-8
            )
            target = (target - target.mean(axis=1, keepdims=True)) / (
                target.std(axis=1, keepdims=True) + 1e-8
            )

        return {
            "source": torch.from_numpy(source).float(),
            "target": torch.from_numpy(target).float(),
            "subject_idx": torch.tensor(subj_idx),
            "start_sample": torch.tensor(start),
        }


def prepare_ecog_data(
    data_dir: Optional[Path] = None,
    experiment: str = "fingerflex",
    source_region: str = "frontal",
    target_region: str = "temporal",
    window_size: int = 5000,
    stride: int = 2500,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    zscore: bool = True,
    verbose: bool = True,
    min_channels: int = 4,
    val_subjects: Optional[List[str]] = None,
    test_subjects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Complete data preparation pipeline for Miller ECoG Library.

    Loads ECoG data, groups electrodes by brain lobe, and creates
    train/val/test datasets for inter-region neural signal translation.

    Handles the two data layout patterns in the NPZ files:
    - Multi-subject: alldat.shape = (N, B) where N > 1 subjects, B blocks each.
      Each subject uses block 0 (longest/primary recording).
    - Single-subject with block-as-patient: alldat.shape = (1, B) where blocks
      have different channel counts (= different patients). Each block becomes
      a separate "subject" for LOSO.

    Args:
        data_dir: Directory containing .npz files
        experiment: Experiment name (fingerflex, faceshouses, etc.)
        source_region: Source brain lobe for translation
        target_region: Target brain lobe for translation
        window_size: Window size in samples
        stride: Stride between windows
        train_ratio: Fraction of subjects for training
        val_ratio: Fraction of subjects for validation
        seed: Random seed
        zscore: Whether to z-score normalize
        verbose: Whether to print progress
        min_channels: Minimum channels required per region
        val_subjects: Explicit subjects for validation (LOSO mode)
        test_subjects: Explicit subjects for testing (LOSO mode)

    Returns:
        Dictionary with train/val/test datasets and metadata
    """
    data_dir = data_dir or _ECOG_DATA_DIR

    if verbose:
        _print_primary(f"Preparing ECoG Library dataset...")
        _print_primary(f"  Experiment: {experiment}")
        _print_primary(f"  Data directory: {data_dir}")
        _print_primary(f"  Source region: {source_region}")
        _print_primary(f"  Target region: {target_region}")

    # Load NPZ file once
    filepath = data_dir / f"{experiment}.npz"
    if not filepath.exists():
        raise FileNotFoundError(
            f"ECoG data file not found: {filepath}\n"
            f"Run: python scripts/download_ecog.py --experiments {experiment}"
        )

    alldat = np.load(filepath, allow_pickle=True)["dat"]
    recordings = _enumerate_ecog_recordings(alldat)

    if verbose:
        _print_primary(f"  NPZ shape: {alldat.shape} ({len(recordings)} recordings)")

    # Load each recording
    subjects_data = []
    valid_subject_ids = []

    for si, bi, rec_id in recordings:
        subj_data = load_ecog_subject(
            subject_idx=si,
            block_idx=bi,
            experiment=experiment,
            source_region=source_region,
            target_region=target_region,
            zscore=zscore,
            min_channels=min_channels,
            _alldat=alldat,
        )
        if subj_data is not None:
            # Override subject_id with the recording_id from enumeration
            subj_data["subject_id"] = rec_id
            subjects_data.append(subj_data)
            valid_subject_ids.append(rec_id)

    if verbose:
        _print_primary(f"  Valid recordings: {len(valid_subject_ids)}/{len(recordings)}")
        _print_primary(f"  IDs: {valid_subject_ids}")

    if len(valid_subject_ids) < 3:
        raise ValueError(
            f"Need at least 3 valid recordings for train/val/test split, "
            f"but only {len(valid_subject_ids)} have >= {min_channels} "
            f"channels in both {source_region} and {target_region}. "
            f"Try different region pairs or a different experiment."
        )

    # Determine train/val/test split
    if val_subjects is not None or test_subjects is not None:
        # Explicit subject holdout (LOSO mode)
        val_subjects = val_subjects or []
        test_subjects = test_subjects or []

        for subj in val_subjects + test_subjects:
            if subj not in valid_subject_ids:
                raise ValueError(
                    f"Subject '{subj}' not found in valid recordings. "
                    f"Available: {valid_subject_ids}"
                )

        holdout = set(val_subjects) | set(test_subjects)
        train_subject_ids = [s for s in valid_subject_ids if s not in holdout]
        val_subject_ids = val_subjects
        test_subject_ids = test_subjects

        if verbose:
            _print_primary(f"  [LOSO MODE] Explicit subject holdout:")
            _print_primary(f"    Train: {train_subject_ids}")
            _print_primary(f"    Val: {val_subject_ids}")
            _print_primary(f"    Test: {test_subject_ids}")
    else:
        # Random split
        rng = np.random.default_rng(seed)
        indices = list(range(len(valid_subject_ids)))
        rng.shuffle(indices)

        n_train = max(1, int(len(indices) * train_ratio))
        n_val = max(1, int(len(indices) * val_ratio))

        train_subject_ids = [valid_subject_ids[i] for i in indices[:n_train]]
        val_subject_ids = [valid_subject_ids[i] for i in indices[n_train:n_train + n_val]]
        test_subject_ids = [valid_subject_ids[i] for i in indices[n_train + n_val:]]

        if not test_subject_ids and len(val_subject_ids) > 1:
            test_subject_ids = [val_subject_ids.pop()]

        if verbose:
            _print_primary(f"  Random split:")
            _print_primary(f"    Train ({len(train_subject_ids)}): {train_subject_ids}")
            _print_primary(f"    Val ({len(val_subject_ids)}): {val_subject_ids}")
            _print_primary(f"    Test ({len(test_subject_ids)}): {test_subject_ids}")

    # Build subject data lists for each split
    subj_id_to_data = {s["subject_id"]: s for s in subjects_data}
    train_data = [subj_id_to_data[sid] for sid in train_subject_ids]
    val_data = [subj_id_to_data[sid] for sid in val_subject_ids]
    test_data = [subj_id_to_data[sid] for sid in test_subject_ids]

    # Determine normalized channel counts across ALL valid recordings
    all_n_src = [s["n_source_channels"] for s in subjects_data]
    all_n_tgt = [s["n_target_channels"] for s in subjects_data]
    n_source_channels = min(all_n_src)
    n_target_channels = min(all_n_tgt)

    if verbose:
        _print_primary(f"  Normalized channels: source={n_source_channels}, target={n_target_channels}")

    # Create datasets
    # Special handling for LOSO mode: when test_subjects provided but no val_subjects,
    # create val set by TEMPORALLY splitting training data (not random window split)
    loso_mode_no_val = (test_subjects is not None and len(test_subjects) > 0 and
                        (val_subjects is None or len(val_subjects) == 0))

    if loso_mode_no_val and len(train_data) > 0:
        # LOSO mode: Split raw samples TEMPORALLY first, then create windows
        # This prevents data leakage from overlapping windows
        train_portions = []
        val_portions = []
        total_train_samples = 0
        total_val_samples = 0

        for subj_data in train_data:
            n_samples = subj_data['n_samples']
            split_point = int(n_samples * 0.7)  # 70% train, 30% val

            train_portions.append({
                'subject_id': subj_data['subject_id'],
                'source': subj_data['source'][:, :split_point],
                'target': subj_data['target'][:, :split_point],
                'n_source_channels': subj_data['n_source_channels'],
                'n_target_channels': subj_data['n_target_channels'],
                'n_samples': split_point,
                'metadata': subj_data['metadata'],
            })
            total_train_samples += split_point

            val_portions.append({
                'subject_id': subj_data['subject_id'],
                'source': subj_data['source'][:, split_point:],
                'target': subj_data['target'][:, split_point:],
                'n_source_channels': subj_data['n_source_channels'],
                'n_target_channels': subj_data['n_target_channels'],
                'n_samples': n_samples - split_point,
                'metadata': subj_data['metadata'],
            })
            total_val_samples += (n_samples - split_point)

        if verbose:
            _print_primary(f"  Temporal split: {total_train_samples} train samples, {total_val_samples} val samples")

        train_dataset = ECoGDataset(
            train_portions, window_size=window_size, stride=stride,
            n_source_channels=n_source_channels, n_target_channels=n_target_channels,
            verbose=verbose,
        )
        val_dataset = ECoGDataset(
            val_portions, window_size=window_size, stride=window_size,
            n_source_channels=n_source_channels, n_target_channels=n_target_channels,
            verbose=verbose,
        )

        if verbose:
            _print_primary(f"ECoGDataset [LOSO]: {len(train_data)} subjects, "
                  f"{len(train_dataset)} train + {len(val_dataset)} val windows (NO overlap), "
                  f"window_size={window_size}, stride={stride}, "
                  f"source_ch={n_source_channels}, target_ch={n_target_channels}")
    else:
        train_dataset = ECoGDataset(
            train_data, window_size=window_size, stride=stride,
            n_source_channels=n_source_channels, n_target_channels=n_target_channels,
            verbose=verbose,
        )
        val_dataset = ECoGDataset(
            val_data, window_size=window_size, stride=window_size,
            n_source_channels=n_source_channels, n_target_channels=n_target_channels,
            verbose=verbose,
        )

    test_dataset = ECoGDataset(
        test_data, window_size=window_size, stride=window_size,
        n_source_channels=n_source_channels, n_target_channels=n_target_channels,
        verbose=verbose,
    )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "n_source_channels": n_source_channels,
        "n_target_channels": n_target_channels,
        "all_subjects": valid_subject_ids,
        "train_subjects": train_subject_ids,
        "val_subjects": val_subject_ids,
        "test_subjects": test_subject_ids,
        "experiment": experiment,
        "source_region": source_region,
        "target_region": target_region,
        "sampling_rate": _parse_ecog_srate(alldat[0, 0]) if alldat.size > 0 else ECOG_SAMPLING_RATE_HZ,
    }


# =============================================================================
# Boran MTL Working Memory Dataset
# =============================================================================
# Preprocessed format: one NPZ per subject (downsampled from 2kHz to 1kHz)
# Created by: python scripts/preprocess_boran.py
# NPZ keys: ieeg (n_channels, total_samples), probes, probe_n_contacts,
#            soz_probes, n_sessions, n_trials, sampling_rate (1000)
# Probe-to-region mapping assigns channels to hippocampus/entorhinal_cortex/amygdala

import re as _re


def _boran_parse_probe_region(prefix: str) -> str:
    """Parse a probe abbreviation to canonical region name."""
    prefix = prefix.upper().strip()
    if prefix in BORAN_PROBE_REGION_MAP:
        return BORAN_PROBE_REGION_MAP[prefix]
    for pfx in sorted(BORAN_PROBE_REGION_MAP.keys(), key=len, reverse=True):
        if prefix.startswith(pfx):
            return BORAN_PROBE_REGION_MAP[pfx]
    return "unknown"


def _boran_get_region_channels(
    probes: List[str],
    probe_n_contacts: np.ndarray,
    region: str,
    soz_probes: Optional[List[str]] = None,
) -> Tuple[List[int], List[str]]:
    """Get channel indices for a given canonical region from preprocessed probe info."""
    indices = []
    used_probes = []
    ch_offset = 0

    for i, probe in enumerate(probes):
        n_contacts = int(probe_n_contacts[i])
        probe_region = _boran_parse_probe_region(probe)

        if soz_probes and probe in soz_probes:
            ch_offset += n_contacts
            continue

        if probe_region == region:
            indices.extend(range(ch_offset, ch_offset + n_contacts))
            used_probes.append(probe)

        ch_offset += n_contacts

    return indices, used_probes


def load_boran_subject(
    subject_id: str,
    data_dir: Optional[Path] = None,
    source_region: str = "hippocampus",
    target_region: str = "entorhinal_cortex",
    zscore: bool = True,
    min_channels: int = 4,
    exclude_soz: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load preprocessed Boran MTL subject data from NPZ.

    Reads preprocessed NPZ file (downsampled to 1kHz), extracts source/target
    channels by region, and z-scores.

    Args:
        subject_id: Subject identifier (e.g., "S01", "S02")
        data_dir: Directory containing preprocessed NPZ files
        source_region: Source MTL region (hippocampus, entorhinal_cortex, amygdala)
        target_region: Target MTL region
        zscore: Whether to z-score normalize per channel
        min_channels: Minimum channels required per region
        exclude_soz: Whether to exclude Seizure Onset Zone probes

    Returns:
        Dict with keys: source, target, subject_id, n_source_channels,
        n_target_channels, n_samples, metadata. Returns None if subject
        doesn't have enough channels in both regions.
    """
    data_dir = data_dir or _BORAN_DATA_DIR
    npz_path = data_dir / f"{subject_id}.npz"

    if not npz_path.exists():
        _print_primary(f"  {subject_id}: NPZ file not found: {npz_path}")
        return None

    # Load preprocessed data
    npz = np.load(npz_path, allow_pickle=False)
    ieeg = npz["ieeg"]  # (n_channels, total_samples) at 1kHz
    probes = list(npz["probes"])
    probe_n_contacts = npz["probe_n_contacts"]
    soz_probes_arr = npz["soz_probes"]
    soz_probes = list(soz_probes_arr) if len(soz_probes_arr) > 0 else []
    n_sessions = int(npz["n_sessions"])
    n_trials = int(npz["n_trials"])

    # Get channel indices for source and target regions
    source_chs, source_probes = _boran_get_region_channels(
        probes, probe_n_contacts, source_region, soz_probes if exclude_soz else None
    )
    target_chs, target_probes = _boran_get_region_channels(
        probes, probe_n_contacts, target_region, soz_probes if exclude_soz else None
    )

    if len(source_chs) < min_channels or len(target_chs) < min_channels:
        _print_primary(
            f"  {subject_id}: insufficient channels "
            f"({source_region}={len(source_chs)}, {target_region}={len(target_chs)}), "
            f"need >= {min_channels} each, skipping"
        )
        return None

    # Extract source and target signals
    source = ieeg[source_chs, :].astype(np.float32)
    target = ieeg[target_chs, :].astype(np.float32)

    # Z-score per channel
    if zscore:
        src_mean = source.mean(axis=1, keepdims=True)
        src_std = source.std(axis=1, keepdims=True) + 1e-8
        source = (source - src_mean) / src_std

        tgt_mean = target.mean(axis=1, keepdims=True)
        tgt_std = target.std(axis=1, keepdims=True) + 1e-8
        target = (target - tgt_mean) / tgt_std

    n_samples = source.shape[1]

    metadata = {
        "subject_id": subject_id,
        "source_region": source_region,
        "target_region": target_region,
        "source_probes": source_probes,
        "target_probes": target_probes,
        "n_sessions": n_sessions,
        "n_trials": n_trials,
        "soz_excluded": exclude_soz,
        "soz_probes": soz_probes,
        "srate": BORAN_SAMPLING_RATE_HZ,
    }

    return {
        "source": source,
        "target": target,
        "subject_id": subject_id,
        "n_source_channels": len(source_chs),
        "n_target_channels": len(target_chs),
        "n_samples": n_samples,
        "metadata": metadata,
    }


def list_boran_subjects(
    data_dir: Optional[Path] = None,
) -> List[str]:
    """List available subject IDs in the Boran MTL dataset.

    Discovers subjects by scanning for preprocessed NPZ files (S01.npz, S02.npz, ...).

    Args:
        data_dir: Directory containing preprocessed NPZ files (default: _BORAN_DATA_DIR)

    Returns:
        Sorted list of subject IDs like ["S01", "S02", ...]
    """
    data_dir = data_dir or _BORAN_DATA_DIR

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Boran MTL data directory not found: {data_dir}\n"
            f"Run: python scripts/preprocess_boran.py"
        )

    npz_files = sorted(data_dir.glob("S*.npz"))
    subjects = [f.stem for f in npz_files]

    return sorted(subjects)


def validate_boran_subject(
    subject_id: str,
    data_dir: Optional[Path] = None,
    source_region: str = "hippocampus",
    target_region: str = "entorhinal_cortex",
    min_channels: int = 4,
    exclude_soz: bool = False,
) -> bool:
    """Check if a Boran subject has enough channels without loading signal data.

    Reads only probe metadata from NPZ to determine channel counts.

    Returns:
        True if subject has >= min_channels in both source and target regions.
    """
    data_dir = data_dir or _BORAN_DATA_DIR
    npz_path = data_dir / f"{subject_id}.npz"

    if not npz_path.exists():
        return False

    try:
        npz = np.load(npz_path, allow_pickle=False)
        probes = list(npz["probes"])
        probe_n_contacts = npz["probe_n_contacts"]
        soz_probes_arr = npz["soz_probes"]
        soz_probes = list(soz_probes_arr) if len(soz_probes_arr) > 0 else []

        source_chs, _ = _boran_get_region_channels(
            probes, probe_n_contacts, source_region, soz_probes if exclude_soz else None
        )
        target_chs, _ = _boran_get_region_channels(
            probes, probe_n_contacts, target_region, soz_probes if exclude_soz else None
        )

        return len(source_chs) >= min_channels and len(target_chs) >= min_channels

    except Exception:
        return False


def prepare_boran_data(
    data_dir: Optional[Path] = None,
    source_region: str = "hippocampus",
    target_region: str = "entorhinal_cortex",
    window_size: int = 10000,
    stride: int = 5000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    zscore: bool = True,
    verbose: bool = True,
    min_channels: int = 4,
    exclude_soz: bool = False,
    val_subjects: Optional[List[str]] = None,
    test_subjects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Complete data preparation pipeline for Boran MTL dataset.

    Loads depth electrode iEEG data, groups by MTL region, and creates
    train/val/test datasets for inter-region neural signal translation.

    Args:
        data_dir: Directory containing H5 files
        source_region: Source MTL region (hippocampus, entorhinal_cortex, amygdala)
        target_region: Target MTL region
        window_size: Window size in samples (default: 10000 = 5s at 2kHz)
        stride: Stride between windows (default: 5000 = 2.5s)
        train_ratio: Fraction of subjects for training (random split)
        val_ratio: Fraction of subjects for validation (random split)
        seed: Random seed
        zscore: Whether to z-score normalize
        verbose: Whether to print progress
        min_channels: Minimum channels required per region
        exclude_soz: Whether to exclude Seizure Onset Zone probes
        val_subjects: Explicit subjects for validation (LOSO mode)
        test_subjects: Explicit subjects for testing (LOSO mode)

    Returns:
        Dictionary with train/val/test datasets and metadata
    """
    data_dir = data_dir or _BORAN_DATA_DIR

    if verbose:
        _print_primary(f"Preparing Boran MTL dataset...")
        _print_primary(f"  Data directory: {data_dir}")
        _print_primary(f"  Source region: {source_region}")
        _print_primary(f"  Target region: {target_region}")
        _print_primary(f"  Exclude SOZ: {exclude_soz}")

    # Discover subjects
    all_subject_ids = list_boran_subjects(data_dir)

    if verbose:
        _print_primary(f"  Found {len(all_subject_ids)} subjects: {all_subject_ids}")

    # Load each subject
    subjects_data = []
    valid_subject_ids = []

    for subj_id in all_subject_ids:
        subj_data = load_boran_subject(
            subject_id=subj_id,
            data_dir=data_dir,
            source_region=source_region,
            target_region=target_region,
            zscore=zscore,
            min_channels=min_channels,
            exclude_soz=exclude_soz,
        )
        if subj_data is not None:
            subjects_data.append(subj_data)
            valid_subject_ids.append(subj_id)
            if verbose:
                meta = subj_data["metadata"]
                _print_primary(
                    f"  {subj_id}: {subj_data['n_source_channels']} src ch, "
                    f"{subj_data['n_target_channels']} tgt ch, "
                    f"{meta['n_sessions']} sessions, {meta['n_trials']} trials, "
                    f"{subj_data['n_samples']} samples"
                )

    if verbose:
        _print_primary(f"  Valid subjects: {len(valid_subject_ids)}/{len(all_subject_ids)}")

    if len(valid_subject_ids) < 3:
        raise ValueError(
            f"Need at least 3 valid subjects for train/val/test split, "
            f"but only {len(valid_subject_ids)} have >= {min_channels} "
            f"channels in both {source_region} and {target_region}."
        )

    # Determine train/val/test split
    if val_subjects is not None or test_subjects is not None:
        # Explicit subject holdout (LOSO mode)
        val_subjects = val_subjects or []
        test_subjects = test_subjects or []

        for subj in val_subjects + test_subjects:
            if subj not in valid_subject_ids:
                raise ValueError(
                    f"Subject '{subj}' not found in valid subjects. "
                    f"Available: {valid_subject_ids}"
                )

        holdout = set(val_subjects) | set(test_subjects)
        train_subject_ids = [s for s in valid_subject_ids if s not in holdout]
        val_subject_ids = val_subjects
        test_subject_ids = test_subjects

        if verbose:
            _print_primary(f"  [LOSO MODE] Explicit subject holdout:")
            _print_primary(f"    Train: {train_subject_ids}")
            _print_primary(f"    Val: {val_subject_ids}")
            _print_primary(f"    Test: {test_subject_ids}")
    else:
        # Random split
        rng = np.random.default_rng(seed)
        indices = list(range(len(valid_subject_ids)))
        rng.shuffle(indices)

        n_train = max(1, int(len(indices) * train_ratio))
        n_val = max(1, int(len(indices) * val_ratio))

        train_subject_ids = [valid_subject_ids[i] for i in indices[:n_train]]
        val_subject_ids = [valid_subject_ids[i] for i in indices[n_train:n_train + n_val]]
        test_subject_ids = [valid_subject_ids[i] for i in indices[n_train + n_val:]]

        if not test_subject_ids and len(val_subject_ids) > 1:
            test_subject_ids = [val_subject_ids.pop()]

        if verbose:
            _print_primary(f"  Random split:")
            _print_primary(f"    Train ({len(train_subject_ids)}): {train_subject_ids}")
            _print_primary(f"    Val ({len(val_subject_ids)}): {val_subject_ids}")
            _print_primary(f"    Test ({len(test_subject_ids)}): {test_subject_ids}")

    # Build subject data lists for each split
    subj_id_to_data = {s["subject_id"]: s for s in subjects_data}
    train_data = [subj_id_to_data[sid] for sid in train_subject_ids]
    val_data = [subj_id_to_data[sid] for sid in val_subject_ids]
    test_data = [subj_id_to_data[sid] for sid in test_subject_ids]

    # Determine normalized channel counts across ALL valid subjects
    all_n_src = [s["n_source_channels"] for s in subjects_data]
    all_n_tgt = [s["n_target_channels"] for s in subjects_data]
    n_source_channels = min(all_n_src)
    n_target_channels = min(all_n_tgt)

    if verbose:
        _print_primary(f"  Normalized channels: source={n_source_channels}, target={n_target_channels}")

    # Create datasets — reuse ECoGDataset (same sliding window pattern)
    # LOSO mode: when test_subjects provided but no val_subjects,
    # create val set by temporally splitting training data
    loso_mode_no_val = (test_subjects is not None and len(test_subjects) > 0 and
                        (val_subjects is None or len(val_subjects) == 0))

    if loso_mode_no_val and len(train_data) > 0:
        # Temporal split to prevent data leakage from overlapping windows
        train_portions = []
        val_portions = []

        for subj_data in train_data:
            n_samples = subj_data['n_samples']
            split_point = int(n_samples * 0.7)

            train_portions.append({
                'subject_id': subj_data['subject_id'],
                'source': subj_data['source'][:, :split_point],
                'target': subj_data['target'][:, :split_point],
                'n_source_channels': subj_data['n_source_channels'],
                'n_target_channels': subj_data['n_target_channels'],
                'n_samples': split_point,
                'metadata': subj_data['metadata'],
            })
            val_portions.append({
                'subject_id': subj_data['subject_id'],
                'source': subj_data['source'][:, split_point:],
                'target': subj_data['target'][:, split_point:],
                'n_source_channels': subj_data['n_source_channels'],
                'n_target_channels': subj_data['n_target_channels'],
                'n_samples': n_samples - split_point,
                'metadata': subj_data['metadata'],
            })

        train_dataset = ECoGDataset(
            train_portions, window_size=window_size, stride=stride,
            n_source_channels=n_source_channels, n_target_channels=n_target_channels,
            verbose=verbose,
        )
        val_dataset = ECoGDataset(
            val_portions, window_size=window_size, stride=window_size,
            n_source_channels=n_source_channels, n_target_channels=n_target_channels,
            verbose=verbose,
        )
    else:
        train_dataset = ECoGDataset(
            train_data, window_size=window_size, stride=stride,
            n_source_channels=n_source_channels, n_target_channels=n_target_channels,
            verbose=verbose,
        )
        val_dataset = ECoGDataset(
            val_data, window_size=window_size, stride=window_size,
            n_source_channels=n_source_channels, n_target_channels=n_target_channels,
            verbose=verbose,
        )

    test_dataset = ECoGDataset(
        test_data, window_size=window_size, stride=window_size,
        n_source_channels=n_source_channels, n_target_channels=n_target_channels,
        verbose=verbose,
    )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "n_source_channels": n_source_channels,
        "n_target_channels": n_target_channels,
        "all_subjects": valid_subject_ids,
        "train_subjects": train_subject_ids,
        "val_subjects": val_subject_ids,
        "test_subjects": test_subject_ids,
        "source_region": source_region,
        "target_region": target_region,
        "sampling_rate": BORAN_SAMPLING_RATE_HZ,
    }
