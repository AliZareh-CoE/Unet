"""
Study 7: Component Ablation Analysis
=====================================

Systematic ablation study to justify each architectural design choice.
Essential for Nature Methods publication.
"""

from .run_study7 import (
    BASELINE_CONFIG,
    ABLATION_VARIANTS,
    ABLATION_CATEGORIES,
    run_single_ablation,
    run_category_ablation,
    run_full_ablation,
)

__all__ = [
    "BASELINE_CONFIG",
    "ABLATION_VARIANTS",
    "ABLATION_CATEGORIES",
    "run_single_ablation",
    "run_category_ablation",
    "run_full_ablation",
]
