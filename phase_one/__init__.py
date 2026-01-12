"""
Phase 1: Classical Baselines for Neural Signal Translation
==========================================================

This module implements the classical baselines study for the Nature Methods paper.
It establishes the performance floor that neural network methods must beat.

Classical Methods (7 total):
    1. Wiener Filter - Single-channel optimal linear filter
    2. Wiener MIMO - Multi-input multi-output Wiener filter
    3. Ridge Regression - L2-regularized linear mapping
    4. Ridge Temporal - Ridge with temporal context windows
    5. Ridge CV - Cross-validated regularization selection
    6. VAR - Vector Autoregressive model
    7. VAR Exogenous - VAR with exogenous input signals

Usage:
    python -m phase_one.runner --dataset olfactory

Author: Neural Signal Translation Team
"""

__version__ = "1.0.0"

from .config import Phase1Config, NEURAL_BANDS
from .metrics import MetricsCalculator, Phase1Metrics
from .runner import run_phase1, Phase1Result

__all__ = [
    "Phase1Config",
    "NEURAL_BANDS",
    "MetricsCalculator",
    "Phase1Metrics",
    "run_phase1",
    "Phase1Result",
]
