"""
Loss Functions for Study 3: Loss Function Ablation
===================================================

All loss functions for neural signal translation comparison:
- Huber losses (robust to outliers)
- Probabilistic losses (uncertainty estimation)
- Multi-scale FFT losses (spectral fidelity)
- Literature-standard losses (perceptual, adversarial, correlation, etc.)
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
from .literature_losses import (
    LogCoshLoss,
    PearsonCorrelationLoss,
    ConcordanceCorrelationLoss,
    TemporalSSIMLoss,
    ContrastiveLoss,
    AdversarialLoss,
    FocalMSELoss,
    SlicedWassersteinLoss,
    NeuralPerceptualLoss,
    MultiObjectiveLoss,
    create_literature_loss,
)
from .neural_probabilistic_losses import (
    GaussianNLLProbLoss,
    LaplacianNLLProbLoss,
    RayleighProbLoss,
    VonMisesProbLoss,
    KLDivergenceProbLoss,
    CauchyNLLProbLoss,
    StudentTNLLProbLoss,
    GumbelProbLoss,
    GammaProbLoss,
    LogNormalProbLoss,
    GaussianMixtureProbLoss,
    NEURAL_PROB_LOSS_REGISTRY,
    create_neural_prob_loss,
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
    # Literature losses
    "LogCoshLoss",
    "PearsonCorrelationLoss",
    "ConcordanceCorrelationLoss",
    "TemporalSSIMLoss",
    "ContrastiveLoss",
    "AdversarialLoss",
    "FocalMSELoss",
    "SlicedWassersteinLoss",
    "NeuralPerceptualLoss",
    "MultiObjectiveLoss",
    "create_literature_loss",
    # Neural probabilistic losses
    "GaussianNLLProbLoss",
    "LaplacianNLLProbLoss",
    "RayleighProbLoss",
    "VonMisesProbLoss",
    "KLDivergenceProbLoss",
    "CauchyNLLProbLoss",
    "StudentTNLLProbLoss",
    "GumbelProbLoss",
    "GammaProbLoss",
    "LogNormalProbLoss",
    "GaussianMixtureProbLoss",
    "NEURAL_PROB_LOSS_REGISTRY",
    "create_neural_prob_loss",
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
    # Literature-standard losses
    "log_cosh": lambda **kw: create_literature_loss("log_cosh", **kw),
    "pearson": lambda **kw: create_literature_loss("pearson", **kw),
    "correlation": lambda **kw: create_literature_loss("pearson", **kw),  # Alias
    "concordance": lambda **kw: create_literature_loss("concordance", **kw),
    "ccc": lambda **kw: create_literature_loss("ccc", **kw),  # Alias
    "ssim": lambda **kw: create_literature_loss("ssim", **kw),
    "contrastive": lambda **kw: create_literature_loss("contrastive", **kw),
    "adversarial": lambda **kw: create_literature_loss("adversarial", **kw),
    "gan": lambda **kw: create_literature_loss("gan", **kw),  # Alias
    "focal": lambda **kw: create_literature_loss("focal", **kw),
    "focal_mse": lambda **kw: create_literature_loss("focal_mse", **kw),
    "wasserstein": lambda **kw: create_literature_loss("wasserstein", **kw),
    "swd": lambda **kw: create_literature_loss("swd", **kw),  # Alias
    "perceptual": lambda **kw: create_literature_loss("perceptual", **kw),
    # Neural probabilistic losses (for tier 2.5)
    "prob_gaussian_nll": lambda **kw: create_neural_prob_loss("gaussian_nll", **kw),
    "prob_laplacian_nll": lambda **kw: create_neural_prob_loss("laplacian_nll", **kw),
    "prob_rayleigh": lambda **kw: create_neural_prob_loss("rayleigh", **kw),
    "prob_von_mises": lambda **kw: create_neural_prob_loss("von_mises", **kw),
    "prob_kl_divergence": lambda **kw: create_neural_prob_loss("kl_divergence", **kw),
    "prob_cauchy_nll": lambda **kw: create_neural_prob_loss("cauchy_nll", **kw),
    "prob_student_t_nll": lambda **kw: create_neural_prob_loss("student_t_nll", **kw),
    "prob_gumbel": lambda **kw: create_neural_prob_loss("gumbel", **kw),
    "prob_gamma": lambda **kw: create_neural_prob_loss("gamma", **kw),
    "prob_log_normal": lambda **kw: create_neural_prob_loss("log_normal", **kw),
    "prob_mixture": lambda **kw: create_neural_prob_loss("mixture", **kw),
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
