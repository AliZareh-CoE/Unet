"""
Phase 3: CondUNet Ablation Studies
==================================

This module implements ablation studies for the CondUNet architecture
to justify design decisions and identify the optimal configuration.

Ablation Studies (6 total, 18 runs):
    1. Attention: none, basic, cross_freq, cross_freq_v2
    2. Conditioning: none, odor, spectro_temporal
    3. Loss function: L1, Huber, Wavelet, Combined
    4. Capacity: 32, 64, 128 base channels
    5. Bidirectional: unidirectional, bidirectional
    6. Augmentation: none, full

Usage:
    python -m phase_three.runner --dataset olfactory --epochs 60

Author: Neural Signal Translation Team
"""

__version__ = "1.0.0"

from .config import (
    Phase3Config,
    AblationConfig,
    ABLATION_STUDIES,
    ATTENTION_VARIANTS,
    CONDITIONING_VARIANTS,
    LOSS_VARIANTS,
    CAPACITY_VARIANTS,
)
from .runner import run_phase3, Phase3Result
from .visualization import Phase3Visualizer, create_ablation_table

__all__ = [
    "Phase3Config",
    "AblationConfig",
    "ABLATION_STUDIES",
    "ATTENTION_VARIANTS",
    "CONDITIONING_VARIANTS",
    "LOSS_VARIANTS",
    "CAPACITY_VARIANTS",
    "run_phase3",
    "Phase3Result",
    "Phase3Visualizer",
    "create_ablation_table",
]
