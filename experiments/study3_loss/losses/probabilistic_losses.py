"""
Probabilistic Losses for Neural Signal Translation
==================================================

Losses that treat model output as parameters of a probability distribution,
enabling uncertainty estimation and better handling of noise.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLLLoss(nn.Module):
    """Gaussian negative log-likelihood loss.

    Models output as mean of Gaussian distribution.
    Can optionally predict variance for uncertainty estimation.

    Args:
        learn_variance: If True, expects model to output (mean, log_var)
        min_variance: Minimum variance for numerical stability
        weight: Loss weight
    """

    def __init__(
        self,
        learn_variance: bool = False,
        min_variance: float = 1e-6,
        weight: float = 1.0,
    ):
        super().__init__()
        self.learn_variance = learn_variance
        self.min_variance = min_variance
        self.weight = weight

        # Always create log_var as fallback (used when learn_variance=False or log_var not provided)
        self.log_var = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        log_var: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Gaussian NLL loss.

        Args:
            pred: Predicted mean [B, C, T]
            target: Target signal [B, C, T]
            log_var: Optional log variance [B, C, T] or [B, C, 1]

        Returns:
            NLL loss
        """
        if self.learn_variance and log_var is not None:
            var = torch.exp(log_var).clamp(min=self.min_variance)
        else:
            var = torch.exp(self.log_var).clamp(min=self.min_variance)

        # NLL = 0.5 * (log(var) + (x - mu)^2 / var)
        diff_sq = (target - pred) ** 2
        nll = 0.5 * (torch.log(var) + diff_sq / var)

        return self.weight * nll.mean()


class LaplacianNLLLoss(nn.Module):
    """Laplacian negative log-likelihood loss.

    More robust to outliers than Gaussian NLL.

    Args:
        learn_scale: If True, expects model to output (mean, log_scale)
        min_scale: Minimum scale for numerical stability
        weight: Loss weight
    """

    def __init__(
        self,
        learn_scale: bool = False,
        min_scale: float = 1e-6,
        weight: float = 1.0,
    ):
        super().__init__()
        self.learn_scale = learn_scale
        self.min_scale = min_scale
        self.weight = weight

        if not learn_scale:
            self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        log_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Laplacian NLL loss.

        Args:
            pred: Predicted location [B, C, T]
            target: Target signal [B, C, T]
            log_scale: Optional log scale

        Returns:
            NLL loss
        """
        if self.learn_scale and log_scale is not None:
            scale = torch.exp(log_scale).clamp(min=self.min_scale)
        else:
            scale = torch.exp(self.log_scale).clamp(min=self.min_scale)

        # NLL = log(2*scale) + |x - mu| / scale
        nll = torch.log(2 * scale) + (target - pred).abs() / scale

        return self.weight * nll.mean()


class BetaNLLLoss(nn.Module):
    """Beta-NLL loss with learned uncertainty weighting.

    Combines NLL with a learned beta parameter that controls
    the trade-off between fitting the mean and uncertainty.

    Reference: Seitzer et al. "On the Pitfalls of Heteroscedastic Uncertainty Estimation" (2022)

    Args:
        beta: Initial beta value (0 = standard NLL, 1 = MSE-like)
        learn_beta: If True, beta is learned
        weight: Loss weight
    """

    def __init__(
        self,
        beta: float = 0.5,
        learn_beta: bool = True,
        weight: float = 1.0,
    ):
        super().__init__()
        self.weight = weight

        if learn_beta:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('beta', torch.tensor(beta))

        self.log_var = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Beta-NLL loss."""
        var = torch.exp(self.log_var).clamp(min=1e-6)
        diff_sq = (target - pred) ** 2

        # Beta-NLL: combines variance term with MSE
        beta = torch.sigmoid(self.beta)  # Keep in (0, 1)
        loss = (1 - beta) * 0.5 * torch.log(var) + beta * diff_sq / (2 * var)

        return self.weight * loss.mean()


class MixtureDensityLoss(nn.Module):
    """Mixture Density Network loss.

    Models output as mixture of Gaussians for multi-modal distributions.

    Args:
        num_components: Number of mixture components
        weight: Loss weight
    """

    def __init__(
        self,
        num_components: int = 3,
        weight: float = 1.0,
    ):
        super().__init__()
        self.num_components = num_components
        self.weight = weight

    def forward(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MDN loss.

        Args:
            pi: Mixing coefficients [B, K, C, T] (should sum to 1 over K)
            mu: Component means [B, K, C, T]
            sigma: Component stds [B, K, C, T]
            target: Target [B, C, T]

        Returns:
            NLL loss
        """
        # Expand target for K components
        target = target.unsqueeze(1)  # [B, 1, C, T]

        # Compute Gaussian log-likelihood for each component
        var = sigma ** 2 + 1e-6
        log_prob = -0.5 * (torch.log(2 * math.pi * var) + (target - mu) ** 2 / var)

        # Log-sum-exp for mixture
        log_pi = torch.log(pi + 1e-10)
        log_prob_weighted = log_pi + log_prob
        log_prob_mixture = torch.logsumexp(log_prob_weighted, dim=1)  # [B, C, T]

        return self.weight * (-log_prob_mixture.mean())


class QuantileLoss(nn.Module):
    """Quantile regression loss for uncertainty estimation.

    Predicts multiple quantiles of the output distribution.

    Args:
        quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
        weight: Loss weight
    """

    def __init__(
        self,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        weight: float = 1.0,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute quantile loss.

        Args:
            pred: Predicted quantiles [B, Q, C, T] where Q = len(quantiles)
            target: Target [B, C, T]

        Returns:
            Quantile loss
        """
        target = target.unsqueeze(1)  # [B, 1, C, T]
        errors = target - pred  # [B, Q, C, T]

        total_loss = 0
        for i, q in enumerate(self.quantiles):
            # Pinball loss
            loss_q = torch.max(q * errors[:, i], (q - 1) * errors[:, i])
            total_loss = total_loss + loss_q.mean()

        return self.weight * total_loss / len(self.quantiles)


def create_probabilistic_loss(
    variant: str = "gaussian_nll",
    **kwargs,
) -> nn.Module:
    """Factory function for probabilistic losses.

    Args:
        variant: "gaussian_nll", "laplacian_nll", "beta_nll", "mdn", "quantile"
        **kwargs: Additional arguments

    Returns:
        Loss module
    """
    if variant == "gaussian_nll":
        return GaussianNLLLoss(**kwargs)
    elif variant == "laplacian_nll":
        return LaplacianNLLLoss(**kwargs)
    elif variant == "beta_nll":
        return BetaNLLLoss(**kwargs)
    elif variant == "mdn":
        return MixtureDensityLoss(**kwargs)
    elif variant == "quantile":
        return QuantileLoss(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
