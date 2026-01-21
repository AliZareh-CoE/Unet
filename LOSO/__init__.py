"""LOSO: Leave-One-Subject-Out Cross-Validation Runner.

This module implements Leave-One-Session-Out cross-validation for
neural signal translation models. It systematically holds out each
recording session for testing while training on all others.
"""

from .config import LOSOConfig, LOSOResult
from .runner import run_loso, run_single_fold

__all__ = [
    "LOSOConfig",
    "LOSOResult",
    "run_loso",
    "run_single_fold",
]
