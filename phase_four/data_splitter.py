"""
Data Splitting Utilities for Phase 4
=====================================

Implements session-based data splitting for generalization studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import SplitMode


# =============================================================================
# Session Splitter
# =============================================================================

@dataclass
class SessionSplit:
    """Result of a session-based split.

    Attributes:
        train_indices: Indices for training samples
        val_indices: Indices for validation samples
        test_indices: Indices for test samples
        train_sessions: Session IDs in training set
        test_sessions: Session IDs in test set
    """

    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    train_sessions: List[int]
    test_sessions: List[int]

    @property
    def n_train(self) -> int:
        return len(self.train_indices)

    @property
    def n_val(self) -> int:
        return len(self.val_indices)

    @property
    def n_test(self) -> int:
        return len(self.test_indices)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "train_sessions": self.train_sessions,
            "test_sessions": self.test_sessions,
        }


class SessionSplitter:
    """Session-based data splitting for generalization studies.

    Supports both intra-session (random split within sessions) and
    inter-session (hold out entire sessions) splitting modes.

    Args:
        session_ids: Array of session IDs for each sample
        val_ratio: Validation set ratio (from training data)
        test_ratio: Test set ratio
        random_state: Random seed

    Example:
        >>> splitter = SessionSplitter(session_ids, random_state=42)
        >>> split = splitter.split(mode=SplitMode.INTER_SESSION)
        >>> train_data = data[split.train_indices]
    """

    def __init__(
        self,
        session_ids: np.ndarray,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_state: int = 42,
    ):
        self.session_ids = np.asarray(session_ids)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        # Get unique sessions
        self.unique_sessions = np.unique(self.session_ids)
        self.n_sessions = len(self.unique_sessions)
        self.n_samples = len(session_ids)

        # Build session to indices mapping
        self.session_to_indices: Dict[int, List[int]] = {}
        for idx, session in enumerate(self.session_ids):
            if session not in self.session_to_indices:
                self.session_to_indices[session] = []
            self.session_to_indices[session].append(idx)

    def split(self, mode: SplitMode) -> SessionSplit:
        """Create train/val/test split.

        Args:
            mode: INTRA_SESSION or INTER_SESSION

        Returns:
            SessionSplit with indices and session info
        """
        if mode == SplitMode.INTRA_SESSION:
            return self._intra_session_split()
        else:
            return self._inter_session_split()

    def _intra_session_split(self) -> SessionSplit:
        """Random split within each session."""
        rng = np.random.RandomState(self.random_state)

        train_indices = []
        val_indices = []
        test_indices = []

        for session in self.unique_sessions:
            indices = np.array(self.session_to_indices[session])
            rng.shuffle(indices)

            n = len(indices)
            n_test = max(1, int(n * self.test_ratio))
            n_val = max(1, int(n * self.val_ratio))
            n_train = n - n_test - n_val

            train_indices.extend(indices[:n_train].tolist())
            val_indices.extend(indices[n_train:n_train + n_val].tolist())
            test_indices.extend(indices[n_train + n_val:].tolist())

        return SessionSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_sessions=self.unique_sessions.tolist(),
            test_sessions=self.unique_sessions.tolist(),  # Same sessions
        )

    def _inter_session_split(self) -> SessionSplit:
        """Hold out entire sessions for test."""
        rng = np.random.RandomState(self.random_state)

        # Shuffle sessions
        sessions = self.unique_sessions.copy()
        rng.shuffle(sessions)

        # Split sessions
        n_test_sessions = max(1, int(len(sessions) * self.test_ratio))
        test_sessions = sessions[:n_test_sessions]
        train_val_sessions = sessions[n_test_sessions:]

        # Further split train sessions for validation
        n_val_sessions = max(1, int(len(train_val_sessions) * self.val_ratio))
        val_sessions = train_val_sessions[:n_val_sessions]
        train_sessions = train_val_sessions[n_val_sessions:]

        # Get indices
        train_indices = []
        for session in train_sessions:
            train_indices.extend(self.session_to_indices[session])

        val_indices = []
        for session in val_sessions:
            val_indices.extend(self.session_to_indices[session])

        test_indices = []
        for session in test_sessions:
            test_indices.extend(self.session_to_indices[session])

        return SessionSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_sessions=train_sessions.tolist(),
            test_sessions=test_sessions.tolist(),
        )

    def leave_one_session_out(self, test_session: int) -> SessionSplit:
        """Leave-one-session-out cross-validation.

        Args:
            test_session: Session ID to hold out for test

        Returns:
            SessionSplit with specified session as test
        """
        rng = np.random.RandomState(self.random_state)

        train_sessions = [s for s in self.unique_sessions if s != test_session]
        rng.shuffle(train_sessions)

        # Take one session for validation
        if len(train_sessions) > 1:
            val_sessions = [train_sessions[0]]
            train_sessions = train_sessions[1:]
        else:
            val_sessions = []

        train_indices = []
        for session in train_sessions:
            train_indices.extend(self.session_to_indices[session])

        val_indices = []
        for session in val_sessions:
            val_indices.extend(self.session_to_indices[session])

        test_indices = self.session_to_indices[test_session]

        return SessionSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_sessions=train_sessions,
            test_sessions=[test_session],
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_session_splits(
    session_ids: np.ndarray,
    mode: SplitMode,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> SessionSplit:
    """Create session-based split.

    Args:
        session_ids: Array of session IDs for each sample
        mode: INTRA_SESSION or INTER_SESSION
        val_ratio: Validation ratio
        test_ratio: Test ratio
        random_state: Random seed

    Returns:
        SessionSplit object
    """
    splitter = SessionSplitter(
        session_ids=session_ids,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    return splitter.split(mode)


def create_intra_session_splits(
    session_ids: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> SessionSplit:
    """Create intra-session (random) split."""
    return create_session_splits(
        session_ids=session_ids,
        mode=SplitMode.INTRA_SESSION,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )


def create_inter_session_splits(
    session_ids: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> SessionSplit:
    """Create inter-session (hold-out) split."""
    return create_session_splits(
        session_ids=session_ids,
        mode=SplitMode.INTER_SESSION,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )


# =============================================================================
# PyTorch Integration
# =============================================================================

if HAS_TORCH:

    def create_dataloaders_from_split(
        dataset: Dataset,
        split: SessionSplit,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders from a SessionSplit.

        Args:
            dataset: PyTorch Dataset
            split: SessionSplit object
            batch_size: Batch size
            num_workers: Number of workers

        Returns:
            train_loader, val_loader, test_loader
        """
        train_dataset = Subset(dataset, split.train_indices)
        val_dataset = Subset(dataset, split.val_indices)
        test_dataset = Subset(dataset, split.test_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader


# =============================================================================
# Analysis Utilities
# =============================================================================

def compute_session_statistics(
    session_ids: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Compute per-session performance statistics.

    Args:
        session_ids: Session ID for each sample
        predictions: Model predictions [N, ...]
        targets: Ground truth [N, ...]

    Returns:
        Dictionary mapping session ID to statistics
    """
    unique_sessions = np.unique(session_ids)
    stats = {}

    for session in unique_sessions:
        mask = session_ids == session
        pred_session = predictions[mask].flatten()
        target_session = targets[mask].flatten()

        # R²
        ss_res = np.sum((target_session - pred_session) ** 2)
        ss_tot = np.sum((target_session - target_session.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        # Correlation
        corr = np.corrcoef(pred_session, target_session)[0, 1]

        # MSE
        mse = np.mean((pred_session - target_session) ** 2)

        stats[int(session)] = {
            "r2": float(r2),
            "correlation": float(corr) if not np.isnan(corr) else 0.0,
            "mse": float(mse),
            "n_samples": int(mask.sum()),
        }

    return stats


def identify_easy_hard_sessions(
    session_stats: Dict[int, Dict[str, float]],
    threshold_percentile: float = 25.0,
) -> Tuple[List[int], List[int]]:
    """Identify easy and hard sessions based on R².

    Args:
        session_stats: Per-session statistics
        threshold_percentile: Percentile for easy/hard threshold

    Returns:
        easy_sessions, hard_sessions
    """
    r2_values = [stats["r2"] for stats in session_stats.values()]
    low_threshold = np.percentile(r2_values, threshold_percentile)
    high_threshold = np.percentile(r2_values, 100 - threshold_percentile)

    easy_sessions = [
        session for session, stats in session_stats.items()
        if stats["r2"] >= high_threshold
    ]
    hard_sessions = [
        session for session, stats in session_stats.items()
        if stats["r2"] <= low_threshold
    ]

    return easy_sessions, hard_sessions
