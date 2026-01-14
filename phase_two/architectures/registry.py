"""
Architecture Registry for Phase 2
=================================

Central registry for creating neural network architectures.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch.nn as nn

from .linear import LinearBaseline
from .simplecnn import SimpleCNN
from .wavenet import WaveNet1D
from .vit import ViT1D
from .condunet import CondUNet1D


# =============================================================================
# Architecture Registry
# =============================================================================

ARCHITECTURE_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "linear": LinearBaseline,
    "simplecnn": SimpleCNN,
    "wavenet": WaveNet1D,
    "vit": ViT1D,
    "condunet": CondUNet1D,
}


# =============================================================================
# Architecture Descriptions
# =============================================================================

ARCHITECTURE_DESCRIPTIONS: Dict[str, str] = {
    "linear": "Simple linear mapping (MLP per time point)",
    "simplecnn": "Basic convolutional neural network",
    "wavenet": "Dilated causal convolutions (WaveNet-style)",
    "vit": "Vision Transformer adapted for 1D signals",
    "condunet": "Conditional U-Net with skip connections",
}


# =============================================================================
# Default Configurations
# =============================================================================

ARCHITECTURE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "linear": {
        "hidden_dim": 256,
        "use_temporal": False,
    },
    "simplecnn": {
        "channels": [64, 128, 256, 128, 64],
        "kernel_size": 7,
    },
    "wavenet": {
        "residual_channels": 64,
        "skip_channels": 128,
        "n_layers": 8,
        "kernel_size": 3,
    },
    "vit": {
        "embed_dim": 256,
        "n_heads": 8,
        "n_layers": 4,
        "patch_size": 50,
        "dropout": 0.1,
    },
    "condunet": {
        "base_channels": 64,
        "n_downsample": 3,
        "attention_type": "cross_freq_v2",
        "use_film": True,
        "dropout": 0.1,
    },
}


# =============================================================================
# Factory Functions
# =============================================================================

def create_architecture(
    name: str,
    in_channels: int = 32,
    out_channels: int = 32,
    time_steps: int = 5000,
    **kwargs,
) -> nn.Module:
    """Create a neural network architecture by name.

    Args:
        name: Architecture name
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_steps: Number of time steps (for architectures that need it)
        **kwargs: Override default hyperparameters

    Returns:
        Instantiated model

    Raises:
        ValueError: If architecture name is unknown

    Example:
        >>> model = create_architecture("wavenet", in_channels=32, out_channels=32)
        >>> model = create_architecture("vit", embed_dim=512, n_layers=6)
    """
    if name not in ARCHITECTURE_REGISTRY:
        available = list(ARCHITECTURE_REGISTRY.keys())
        raise ValueError(f"Unknown architecture: '{name}'. Available: {available}")

    # Get defaults and merge with kwargs
    params = ARCHITECTURE_DEFAULTS.get(name, {}).copy()
    params.update(kwargs)

    # Add common parameters
    params["in_channels"] = in_channels
    params["out_channels"] = out_channels

    # Add time_steps for architectures that need it
    if name == "vit":
        params["time_steps"] = time_steps

    return ARCHITECTURE_REGISTRY[name](**params)


def list_architectures() -> List[str]:
    """List all available architectures.

    Returns:
        List of architecture names
    """
    return list(ARCHITECTURE_REGISTRY.keys())


def get_architecture_info(name: str) -> Dict[str, Any]:
    """Get information about an architecture.

    Args:
        name: Architecture name

    Returns:
        Dictionary with description and default parameters
    """
    if name not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"Unknown architecture: '{name}'")

    return {
        "name": name,
        "description": ARCHITECTURE_DESCRIPTIONS.get(name, ""),
        "default_params": ARCHITECTURE_DEFAULTS.get(name, {}),
    }


def print_architectures_summary() -> None:
    """Print formatted summary of all architectures."""
    print("\n" + "=" * 70)
    print("PHASE 2: Neural Architectures for Signal Translation")
    print("=" * 70)

    print("\n{:<12} {:<58}".format("Name", "Description"))
    print("-" * 70)

    for name in ARCHITECTURE_REGISTRY.keys():
        desc = ARCHITECTURE_DESCRIPTIONS.get(name, "")
        if len(desc) > 56:
            desc = desc[:53] + "..."
        print(f"{name:<12} {desc:<58}")

    print("-" * 70)
    print(f"Total: {len(ARCHITECTURE_REGISTRY)} architectures")
    print()
