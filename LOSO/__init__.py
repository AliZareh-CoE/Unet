"""LOSO: Leave-One-Subject-Out Cross-Validation Runner.

This module implements Leave-One-Session-Out cross-validation for
neural signal translation models. It systematically holds out each
recording session (or subject) for testing while training on all others.

Supports three datasets:
- olfactory: OB→PCx translation (session-based LOSO)
- pfc_hpc: PFC→CA1 translation (session-based LOSO)
- dandi_movie: Human iEEG movie watching (subject-based LOSO)

Usage:
    # Olfactory LOSO (default)
    python -m LOSO.runner --dataset olfactory --output-dir results/loso_olfactory

    # PFC/HPC LOSO
    python -m LOSO.runner --dataset pfc_hpc --output-dir results/loso_pfc

    # DANDI LOSO (Leave-One-Subject-Out)
    python -m LOSO.runner --dataset dandi_movie --output-dir results/loso_dandi \\
        --dandi-source-region amygdala --dandi-target-region hippocampus
"""

from .config import (
    LOSOConfig,
    LOSOFoldResult,
    LOSOResult,
    DatasetConfig,
    DATASET_CONFIGS,
    get_dataset_config,
)
from .runner import run_loso, run_single_fold, get_all_sessions, validate_sessions

__all__ = [
    # Configuration
    "LOSOConfig",
    "LOSOFoldResult",
    "LOSOResult",
    "DatasetConfig",
    "DATASET_CONFIGS",
    "get_dataset_config",
    # Runner functions
    "run_loso",
    "run_single_fold",
    "get_all_sessions",
    "validate_sessions",
]
