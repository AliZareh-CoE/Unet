"""
Study 6: Final Validation
=========================

Multi-seed validation with cross-validation and negative controls.
Top 5 configs x 10 seeds x cross-validation.
"""

from __future__ import annotations

from .multi_seed_runner import (
    SeedResult,
    MultiSeedResult,
    MultiSeedRunner,
    run_bootstrap_analysis,
    seed_everything,
    generate_seeds,
)
from .statistical_tests import (
    TestResult,
    StatisticalComparison,
    paired_t_test,
    wilcoxon_test,
    permutation_test,
    bonferroni_correction,
    benjamini_hochberg,
)
from .negative_controls import (
    NegativeControlResult,
    NegativeControlRunner,
    summarize_negative_controls,
)

__all__ = [
    # Multi-seed runner
    "SeedResult",
    "MultiSeedResult",
    "MultiSeedRunner",
    "run_bootstrap_analysis",
    "seed_everything",
    "generate_seeds",
    # Statistical tests
    "TestResult",
    "StatisticalComparison",
    "paired_t_test",
    "wilcoxon_test",
    "permutation_test",
    "bonferroni_correction",
    "benjamini_hochberg",
    # Negative controls
    "NegativeControlResult",
    "NegativeControlRunner",
    "summarize_negative_controls",
]
