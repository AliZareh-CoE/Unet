"""
Phase 3: Ablation Study for CondUNet Architecture
==================================================

Implements 3-fold × 3-seed cross-validation with proper train/val/test separation.
"""

from .runner import (
    AblationConfig,
    get_ablation_configs,
    run_3fold_ablation_study,
    run_ablation_experiment,
    main,
)

__all__ = [
    "AblationConfig",
    "get_ablation_configs",
    "run_3fold_ablation_study",
    "run_ablation_experiment",
    "main",
]
