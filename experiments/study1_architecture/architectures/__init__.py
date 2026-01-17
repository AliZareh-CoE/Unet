"""
Architecture implementations for Study 1.

All architectures for neural signal translation comparison:
- Linear baselines (floor performance)
- SimpleCNN (minimal deep learning)
- WaveNet (dilated causal convolutions)
- FNet (FFT-based mixing)
- ViT (transformer with attention)
"""

from __future__ import annotations

from .linear_baseline import LinearBaseline, LinearChannelMixer, create_linear_baseline
from .simple_cnn import SimpleCNN, SimpleCNNWithPooling, create_simple_cnn
from .wavenet_1d import WaveNet1D, WaveNetBidirectional, create_wavenet
from .fnet_1d import FNet1D, FNetWithConv, create_fnet
from .vit_1d import ViT1D, ViT1DWithConvStem, create_vit

__all__ = [
    # Linear baselines
    "LinearBaseline",
    "LinearChannelMixer",
    "create_linear_baseline",
    # Simple CNN
    "SimpleCNN",
    "SimpleCNNWithPooling",
    "create_simple_cnn",
    # WaveNet
    "WaveNet1D",
    "WaveNetBidirectional",
    "create_wavenet",
    # FNet
    "FNet1D",
    "FNetWithConv",
    "create_fnet",
    # ViT
    "ViT1D",
    "ViT1DWithConvStem",
    "create_vit",
    # Registry
    "ARCHITECTURE_REGISTRY",
    "create_architecture",
]

# Architecture registry for easy access
ARCHITECTURE_REGISTRY = {
    "linear": create_linear_baseline,
    "cnn": create_simple_cnn,
    "wavenet": create_wavenet,
    "fnet": create_fnet,
    "vit": create_vit,
}


def create_architecture(name: str, variant: str = "standard", **kwargs):
    """Create architecture by name.

    Args:
        name: Architecture name (linear, cnn, wavenet, fnet, vit)
        variant: Architecture variant
        **kwargs: Additional arguments

    Returns:
        Model instance
    """
    if name not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"Unknown architecture: {name}. Available: {list(ARCHITECTURE_REGISTRY.keys())}")

    factory = ARCHITECTURE_REGISTRY[name]

    # Handle variant naming differences
    if name == "linear":
        variant_map = {"standard": "simple", "simple": "simple", "shared": "shared", "mixer": "mixer"}
        variant = variant_map.get(variant, variant)

    return factory(variant, **kwargs)
