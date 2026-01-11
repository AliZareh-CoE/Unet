"""
Phase 4: Inter vs Intra Session Generalization Study
=====================================================

This module implements cross-session and within-session generalization
experiments across all three datasets.

Experiments:
    - Intra-session: Random train/test split within each session
    - Inter-session: Train on some sessions, test on held-out sessions

Datasets:
    - Olfactory (OB → PCx)
    - PFC (PFC → CA1)
    - DANDI (AMY → HPC, human)

Usage:
    python -m phase_four.runner --epochs 60

Author: Neural Signal Translation Team
"""

__version__ = "1.0.0"

from .config import (
    Phase4Config,
    DatasetConfig,
    SplitMode,
    DATASET_CONFIGS,
)
from .runner import run_phase4, Phase4Result
from .data_splitter import (
    SessionSplitter,
    create_session_splits,
    create_intra_session_splits,
    create_inter_session_splits,
)
from .visualization import Phase4Visualizer

__all__ = [
    "Phase4Config",
    "DatasetConfig",
    "SplitMode",
    "DATASET_CONFIGS",
    "run_phase4",
    "Phase4Result",
    "SessionSplitter",
    "create_session_splits",
    "create_intra_session_splits",
    "create_inter_session_splits",
    "Phase4Visualizer",
]
