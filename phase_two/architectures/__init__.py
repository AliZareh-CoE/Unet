"""
Neural Architectures for Phase 2
================================

5 architectures for neural signal translation:

1. Linear - Simple linear mapping baseline
2. SimpleCNN - Basic convolutional network
3. WaveNet1D - Dilated causal convolutions (van den Oord et al.)
4. ViT1D - Vision Transformer for 1D signals (Dosovitskiy et al.)
5. CondUNet - Conditional U-Net with skip connections

All architectures follow the same interface:
    - Input: [batch, channels_in, time]
    - Output: [batch, channels_out, time]
"""

from .linear import LinearBaseline
from .simplecnn import SimpleCNN
from .wavenet import WaveNet1D
from .vit import ViT1D
from .condunet import NeuroGate
from .registry import ARCHITECTURE_REGISTRY, create_architecture, list_architectures

__all__ = [
    "LinearBaseline",
    "SimpleCNN",
    "WaveNet1D",
    "ViT1D",
    "NeuroGate",
    "ARCHITECTURE_REGISTRY",
    "create_architecture",
    "list_architectures",
]
