"""
Analysis modules for HPO results.

This package provides tools for analyzing the results of the dual-track
HPO experiments and generating Nature Methods publication figures.
"""

from .phase1_screening import analyze_phase1, rank_approaches
from .phase2_deep_dive import analyze_phase2, compare_best_configs
from .confidence_intervals import bootstrap_ci, paired_bootstrap_test

__all__ = [
    "analyze_phase1",
    "rank_approaches",
    "analyze_phase2",
    "compare_best_configs",
    "bootstrap_ci",
    "paired_bootstrap_test",
]
