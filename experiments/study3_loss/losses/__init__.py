"""
Loss Functions for Study 3: Loss Function Ablation
===================================================

All loss functions for neural signal translation comparison:
- Huber losses (robust to outliers)
- Probabilistic losses (uncertainty estimation)
- Multi-scale FFT losses (spectral fidelity)
"""

from __future__ import annotations

from .huber_loss import (
    HuberLoss,
    QuantileHuberLoss,
    ChannelWeightedHuberLoss,
    create_huber_loss,
)
from .probabilistic_losses import (
    GaussianNLLLoss,
    LaplacianNLLLoss,
    BetaNLLLoss,
    MixtureDensityLoss,
    QuantileLoss,
    create_probabilistic_loss,
)
from .multi_scale_fft import (
    SpectralLoss,
    MultiScaleSpectralLoss,
    STFTLoss,
    MultiResolutionSTFTLoss,
    BandSpecificLoss,
    CombinedSpectralTimeLoss,
    create_spectral_loss,
)

__all__ = [
    # Huber losses
    "HuberLoss",
    "QuantileHuberLoss",
    "ChannelWeightedHuberLoss",
    "create_huber_loss",
    # Probabilistic losses
    "GaussianNLLLoss",
    "LaplacianNLLLoss",
    "BetaNLLLoss",
    "MixtureDensityLoss",
    "QuantileLoss",
    "create_probabilistic_loss",
    # Spectral losses
    "SpectralLoss",
    "MultiScaleSpectralLoss",
    "STFTLoss",
    "MultiResolutionSTFTLoss",
    "BandSpecificLoss",
    "CombinedSpectralTimeLoss",
    "create_spectral_loss",
    # Registry
    "LOSS_REGISTRY",
    "create_loss",
]

# Loss registry for easy access
LOSS_REGISTRY = {
    # Standard losses
    "l1": lambda **kw: __import__("torch.nn", fromlist=["L1Loss"]).L1Loss(),
    "mse": lambda **kw: __import__("torch.nn", fromlist=["MSELoss"]).MSELoss(),
    # Huber variants
    "huber": lambda **kw: create_huber_loss("standard", **kw),
    "quantile_huber": lambda **kw: create_huber_loss("quantile", **kw),
    "channel_huber": lambda **kw: create_huber_loss("channel_weighted", **kw),
    # Probabilistic losses
    "gaussian_nll": lambda **kw: create_probabilistic_loss("gaussian_nll", **kw),
    "laplacian_nll": lambda **kw: create_probabilistic_loss("laplacian_nll", **kw),
    "beta_nll": lambda **kw: create_probabilistic_loss("beta_nll", **kw),
    "mdn": lambda **kw: create_probabilistic_loss("mdn", **kw),
    "quantile": lambda **kw: create_probabilistic_loss("quantile", **kw),
    # Spectral losses
    "spectral": lambda **kw: create_spectral_loss("basic", **kw),
    "multi_scale_spectral": lambda **kw: create_spectral_loss("multi_scale", **kw),
    "stft": lambda **kw: create_spectral_loss("stft", **kw),
    "multi_res_stft": lambda **kw: create_spectral_loss("multi_res_stft", **kw),
    "band_specific": lambda **kw: create_spectral_loss("band_specific", **kw),
    "combined": lambda **kw: create_spectral_loss("combined", **kw),
}


def create_loss(name: str, **kwargs):
    """Create loss by name.

    Args:
        name: Loss name (see LOSS_REGISTRY keys)
        **kwargs: Additional arguments for the loss

    Returns:
        Loss module instance
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")

    return LOSS_REGISTRY[name](**kwargs)
