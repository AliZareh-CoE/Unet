"""
Neural Signal-Specific Probabilistic Losses
============================================

Probability distribution-based losses designed specifically for neural signals
(EEG, LFP, etc.). These losses leverage known statistical properties of neural
oscillations to improve translation quality.

Key distributions modeled:
- Gaussian NLL: Standard negative log-likelihood for general signal modeling
- Laplacian NLL: Robust to outliers (common in neural spikes/artifacts)
- Rayleigh: Signal envelope (amplitude) distribution for oscillatory signals
- Von Mises: Phase distribution for circular statistics of neural oscillations
- KL Divergence: Distribution matching between predicted and target signals
- Cauchy NLL: Heavy-tailed distribution for signals with frequent large deviations
- Student-t NLL: Controllable heavy tails via degrees of freedom parameter

References:
- PyTorch GaussianNLLLoss: https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
- Circular statistics for neural phase: https://www.jneurosci.org/content/36/3/860
- KL Divergence: https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLLProbLoss(nn.Module):
    """Gaussian negative log-likelihood loss for neural signals.

    Models the output as samples from a Gaussian distribution.
    NLL = 0.5 * (log(2π) + log(σ²) + (y - μ)²/σ²)

    This is equivalent to MSE when σ² = 1, but allows the model to learn
    heteroscedastic uncertainty.

    Args:
        weight: Loss weight multiplier
        eps: Small constant for numerical stability
    """

    def __init__(self, weight: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps
        # Learnable log-variance (initialized to log(1) = 0)
        self.log_var = nn.Parameter(torch.zeros(1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        var = torch.exp(self.log_var).clamp(min=self.eps)
        # Gaussian NLL (dropping constant log(2π)/2 term)
        nll = 0.5 * (torch.log(var) + (target - pred) ** 2 / var)
        return self.weight * nll.mean()


class LaplacianNLLProbLoss(nn.Module):
    """Laplacian negative log-likelihood loss.

    More robust to outliers than Gaussian. The Laplacian distribution has
    heavier tails, making it suitable for neural signals with occasional
    large amplitude events (spikes, artifacts).

    NLL = log(2b) + |y - μ|/b

    Args:
        weight: Loss weight multiplier
        eps: Small constant for numerical stability
    """

    def __init__(self, weight: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps
        # Learnable log-scale (b parameter)
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.log_scale).clamp(min=self.eps)
        # Laplacian NLL
        nll = torch.log(2 * scale) + torch.abs(target - pred) / scale
        return self.weight * nll.mean()


class RayleighProbLoss(nn.Module):
    """Rayleigh distribution loss for signal envelope (amplitude).

    The envelope of narrow-band neural oscillations approximately follows
    a Rayleigh distribution. This loss encourages predicted signal envelopes
    to match the statistical properties of target envelopes.

    Rayleigh PDF: f(x; σ) = (x/σ²) * exp(-x²/(2σ²))
    NLL = -log(x/σ²) + x²/(2σ²)  for x > 0

    Args:
        weight: Loss weight multiplier
        eps: Small constant for numerical stability
    """

    def __init__(self, weight: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute signal envelopes (absolute values as proxy for Hilbert envelope)
        pred_env = torch.abs(pred) + self.eps
        target_env = torch.abs(target) + self.eps

        # Estimate Rayleigh σ² from target envelope using MLE
        # For Rayleigh: E[X²] = 2σ², so σ² = E[X²]/2
        sigma_sq = torch.mean(target_env ** 2, dim=-1, keepdim=True) / 2 + self.eps

        # Rayleigh NLL for predicted envelope
        # NLL = -log(x) + log(σ²) + x²/(2σ²)
        nll = -torch.log(pred_env) + torch.log(sigma_sq) + pred_env ** 2 / (2 * sigma_sq)

        return self.weight * nll.mean()


class VonMisesProbLoss(nn.Module):
    """Von Mises distribution loss for phase relationships.

    Neural oscillations have important phase relationships that follow
    circular statistics. The von Mises distribution is the circular
    analog of the Gaussian distribution.

    Von Mises PDF: f(θ; μ, κ) = exp(κ*cos(θ-μ)) / (2π*I₀(κ))
    Loss: 1 - cos(θ_pred - θ_target)  (simplified von Mises-like loss)

    This encourages phase alignment between predicted and target signals.

    Args:
        weight: Loss weight multiplier
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute instantaneous phase using gradient as proxy
        # (True Hilbert transform requires FFT which may be slow)
        pred_diff = F.pad(pred[..., 1:] - pred[..., :-1], (0, 1))
        target_diff = F.pad(target[..., 1:] - target[..., :-1], (0, 1))

        # Instantaneous phase via atan2
        pred_phase = torch.atan2(pred_diff, pred + 1e-8)
        target_phase = torch.atan2(target_diff, target + 1e-8)

        # Phase difference
        phase_diff = pred_phase - target_phase

        # Von Mises-inspired loss: 1 - cos(phase_diff)
        # This is 0 when phases align, 2 when they're opposite
        loss = 1.0 - torch.cos(phase_diff)

        return self.weight * loss.mean()


class KLDivergenceProbLoss(nn.Module):
    """KL Divergence loss for distribution matching.

    Measures how much the predicted signal distribution differs from
    the target signal distribution. Uses histogram-based estimation.

    KL(P||Q) = Σ P(x) * log(P(x)/Q(x))

    Args:
        weight: Loss weight multiplier
        n_bins: Number of histogram bins for distribution estimation
        eps: Small constant for numerical stability
    """

    def __init__(self, weight: float = 0.1, n_bins: int = 64, eps: float = 1e-8):
        super().__init__()
        self.weight = weight
        self.n_bins = n_bins
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B = pred.shape[0]

        # Flatten to [B, -1]
        pred_flat = pred.reshape(B, -1)
        target_flat = target.reshape(B, -1)

        # Compute range from target
        vmin = target_flat.min(dim=-1, keepdim=True)[0]
        vmax = target_flat.max(dim=-1, keepdim=True)[0]

        # Soft histogram via kernel density estimation (differentiable)
        # Use Gaussian kernel centered at bin centers
        bin_edges = torch.linspace(0, 1, self.n_bins + 1, device=pred.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        # Normalize values to [0, 1]
        pred_norm = (pred_flat - vmin) / (vmax - vmin + self.eps)
        target_norm = (target_flat - vmin) / (vmax - vmin + self.eps)

        # Soft histogram: sum of Gaussian kernels
        # [B, N, 1] - [1, 1, n_bins] -> [B, N, n_bins]
        pred_hist = torch.exp(-((pred_norm.unsqueeze(-1) - bin_centers.view(1, 1, -1)) ** 2) / (2 * bin_width ** 2))
        target_hist = torch.exp(-((target_norm.unsqueeze(-1) - bin_centers.view(1, 1, -1)) ** 2) / (2 * bin_width ** 2))

        # Normalize to probability distributions
        pred_prob = pred_hist.sum(dim=1) / (pred_hist.sum(dim=1).sum(dim=-1, keepdim=True) + self.eps)
        target_prob = target_hist.sum(dim=1) / (target_hist.sum(dim=1).sum(dim=-1, keepdim=True) + self.eps)

        # KL divergence: P * log(P/Q)
        kl = target_prob * (torch.log(target_prob + self.eps) - torch.log(pred_prob + self.eps))

        return self.weight * kl.sum(dim=-1).mean()


class CauchyNLLProbLoss(nn.Module):
    """Cauchy negative log-likelihood loss.

    The Cauchy distribution has very heavy tails, making it extremely robust
    to outliers. Useful for neural signals with frequent large amplitude events.

    Cauchy PDF: f(x; x₀, γ) = 1 / (πγ * (1 + ((x-x₀)/γ)²))
    NLL = log(πγ) + log(1 + ((y-μ)/γ)²)

    Args:
        weight: Loss weight multiplier
        eps: Small constant for numerical stability
    """

    def __init__(self, weight: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps
        # Learnable log-scale (γ parameter)
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gamma = torch.exp(self.log_scale).clamp(min=self.eps)
        # Cauchy NLL (dropping constant log(π) term)
        z = (target - pred) / gamma
        nll = torch.log(gamma) + torch.log(1 + z ** 2)
        return self.weight * nll.mean()


class StudentTNLLProbLoss(nn.Module):
    """Student's t negative log-likelihood loss.

    Student's t distribution provides controllable heavy tails via the
    degrees of freedom (df) parameter. As df → ∞, it becomes Gaussian.
    Lower df means heavier tails.

    Args:
        weight: Loss weight multiplier
        df: Degrees of freedom (lower = heavier tails)
        eps: Small constant for numerical stability
    """

    def __init__(self, weight: float = 0.1, df: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.df = df
        self.eps = eps
        # Learnable log-scale
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.log_scale).clamp(min=self.eps)
        df = self.df

        # Student's t NLL (up to constants)
        z = (target - pred) / scale
        nll = torch.log(scale) + 0.5 * (df + 1) * torch.log(1 + z ** 2 / df)

        return self.weight * nll.mean()


class GumbelProbLoss(nn.Module):
    """Gumbel distribution loss for extreme values (peaks/troughs).

    Peak values in neural signals follow extreme value distributions.
    The Gumbel distribution models the maximum of many samples.

    Args:
        weight: Loss weight multiplier
        percentile: Focus on top percentile of values (default 95)
        eps: Small constant for numerical stability
    """

    def __init__(self, weight: float = 0.1, percentile: float = 95.0, eps: float = 1e-8):
        super().__init__()
        self.weight = weight
        self.percentile = percentile
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten to [B, -1]
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        # Get top percentile values
        k = max(int((1 - self.percentile / 100) * pred_flat.shape[-1]), 1)

        pred_peaks, _ = torch.topk(pred_flat, k, dim=-1)
        target_peaks, _ = torch.topk(target_flat, k, dim=-1)

        # Estimate Gumbel parameters from target peaks
        # Method of moments: μ = mean - 0.5772*β, β = std*√6/π
        target_mean = target_peaks.mean(dim=-1, keepdim=True)
        target_std = target_peaks.std(dim=-1, keepdim=True) + self.eps
        beta = target_std * math.sqrt(6) / math.pi
        mu = target_mean - 0.5772 * beta  # Euler-Mascheroni constant

        # Gumbel NLL: z + exp(-z) + log(β)
        z = (pred_peaks - mu) / (beta + self.eps)
        nll = z + torch.exp(-z) + torch.log(beta + self.eps)

        return self.weight * nll.mean()


class GammaProbLoss(nn.Module):
    """Gamma distribution loss for positive-valued signals.

    Power and energy of neural signals are positive-valued and often
    follow gamma-like distributions.

    Args:
        weight: Loss weight multiplier
        eps: Small constant for numerical stability
    """

    def __init__(self, weight: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure positive values
        pred_pos = F.softplus(pred) + self.eps
        target_pos = torch.abs(target) + self.eps

        # Estimate gamma parameters from target using method of moments
        # shape (k) = mean²/var, rate (θ) = var/mean
        target_mean = target_pos.mean(dim=-1, keepdim=True)
        target_var = target_pos.var(dim=-1, keepdim=True) + self.eps

        shape = target_mean ** 2 / target_var
        rate = target_mean / target_var

        # Simplified gamma-inspired loss (avoid lgamma for efficiency)
        # Match first two moments + add shape-aware penalty
        pred_mean = pred_pos.mean(dim=-1, keepdim=True)
        pred_var = pred_pos.var(dim=-1, keepdim=True) + self.eps

        # Moment matching loss
        mean_loss = (pred_mean - target_mean) ** 2 / target_mean ** 2
        var_loss = (pred_var - target_var) ** 2 / target_var ** 2

        # Rate-weighted error (encourages matching the scale)
        rate_loss = rate * torch.abs(pred_pos - target_pos)

        loss = mean_loss + var_loss + 0.1 * rate_loss.mean(dim=-1, keepdim=True)

        return self.weight * loss.mean()


class LogNormalProbLoss(nn.Module):
    """Log-normal distribution loss for multiplicative processes.

    Many neural processes involve multiplicative noise, which leads to
    log-normal distributions. This loss works in log-space for better
    stability.

    Args:
        weight: Loss weight multiplier
        eps: Small constant for numerical stability
    """

    def __init__(self, weight: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Transform to log space (handle negatives via abs)
        pred_log = torch.log(torch.abs(pred) + self.eps)
        target_log = torch.log(torch.abs(target) + self.eps)

        # Estimate log-normal parameters from target
        mu = target_log.mean(dim=-1, keepdim=True)
        sigma = target_log.std(dim=-1, keepdim=True) + self.eps

        # Log-normal NLL in log space = Gaussian NLL on log values
        # Plus the Jacobian term: log(x)
        nll = pred_log + ((pred_log - mu) ** 2) / (2 * sigma ** 2)

        return self.weight * nll.mean()


class GaussianMixtureProbLoss(nn.Module):
    """Gaussian Mixture Model loss for multi-modal distributions.

    Neural signals often have multi-modal amplitude distributions
    (e.g., separate modes for different brain states).

    Args:
        weight: Loss weight multiplier
        n_components: Number of mixture components
        eps: Small constant for numerical stability
    """

    def __init__(self, weight: float = 0.1, n_components: int = 3, eps: float = 1e-8):
        super().__init__()
        self.weight = weight
        self.n_components = n_components
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, T = target.shape

        # Flatten for distribution estimation
        target_flat = target.reshape(B, -1)
        pred_flat = pred.reshape(B, -1)

        # Estimate mixture centers from target using quantiles
        quantiles = torch.linspace(0.15, 0.85, self.n_components, device=target.device)
        centers = torch.quantile(target_flat, quantiles, dim=-1).T  # [B, K]

        # Estimate shared variance
        sigma = target_flat.std(dim=-1, keepdim=True) / self.n_components + self.eps

        # Compute log-probabilities for each component
        # pred_flat: [B, N], centers: [B, K] -> dists: [B, N, K]
        dists = (pred_flat.unsqueeze(-1) - centers.unsqueeze(1)) ** 2  # [B, N, K]

        # Gaussian log-prob for each component
        log_probs = -dists / (2 * sigma.unsqueeze(-1) ** 2) - torch.log(sigma.unsqueeze(-1) + self.eps)

        # Log-sum-exp for mixture (equal mixing weights)
        log_prob_mixture = torch.logsumexp(log_probs, dim=-1) - math.log(self.n_components)

        # Negative log-likelihood
        nll = -log_prob_mixture

        return self.weight * nll.mean()


# Registry of neural probabilistic losses
NEURAL_PROB_LOSS_REGISTRY = {
    "gaussian_nll": GaussianNLLProbLoss,
    "laplacian_nll": LaplacianNLLProbLoss,
    "rayleigh": RayleighProbLoss,
    "von_mises": VonMisesProbLoss,
    "kl_divergence": KLDivergenceProbLoss,
    "cauchy_nll": CauchyNLLProbLoss,
    "student_t_nll": StudentTNLLProbLoss,
    "gumbel": GumbelProbLoss,
    "gamma": GammaProbLoss,
    "log_normal": LogNormalProbLoss,
    "mixture": GaussianMixtureProbLoss,
}


def create_neural_prob_loss(name: str, weight: float = 0.1, **kwargs) -> nn.Module:
    """Factory function for neural probabilistic losses.

    Args:
        name: Loss name (see NEURAL_PROB_LOSS_REGISTRY)
        weight: Loss weight multiplier
        **kwargs: Additional arguments for specific losses

    Returns:
        Loss module instance
    """
    if name not in NEURAL_PROB_LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(NEURAL_PROB_LOSS_REGISTRY.keys())}")

    return NEURAL_PROB_LOSS_REGISTRY[name](weight=weight, **kwargs)
