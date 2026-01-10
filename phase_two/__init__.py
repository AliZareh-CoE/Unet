"""
Phase 2: Architecture Screening for Neural Signal Translation
=============================================================

This module implements the neural architecture screening study.
Compares 8 different architectures on the olfactory dataset.

Architectures (8 total):
    1. Linear - Simple linear baseline
    2. SimpleCNN - Basic convolutional network
    3. WaveNet1D - Dilated causal convolutions
    4. FNet1D - FFT-based mixing (no attention)
    5. ViT1D - Vision Transformer adapted for 1D
    6. Performer1D - Linear attention transformer
    7. Mamba1D - State-space model (S4/Mamba)
    8. CondUNet - Conditional U-Net with skip connections

Usage:
    python -m phase_two.runner --dataset olfactory --epochs 60

Author: Neural Signal Translation Team
"""

__version__ = "1.0.0"

from .config import Phase2Config, ARCHITECTURE_LIST
from .runner import run_phase2, Phase2Result

__all__ = [
    "Phase2Config",
    "ARCHITECTURE_LIST",
    "run_phase2",
    "Phase2Result",
]
