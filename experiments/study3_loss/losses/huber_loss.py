"""
Huber Loss for Neural Signal Translation
========================================

Huber loss (smooth L1) is more robust to outliers than MSE
while still encouraging fast convergence near zero error.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    """Huber loss for neural signal translation.

    Combines benefits of MSE (quadratic near 0) and MAE (linear for outliers).
    transition point controlled by delta parameter.

    L(x) = 0.5 * x^2 if |x| < delta
         = delta * (|x| - 0.5 * delta) otherwise

    Args:
        delta: Transition point between quadratic and linear regions
        reduction: 'mean', 'sum', or 'none'
        weight: Optional loss weight
    """

    def __init__(
        self,
        delta: float = 1.0,
        reduction: str = "mean",
        weight: float = 1.0,
    ):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Huber loss.

        Args:
            pred: Predicted signal [B, C, T]
            target: Target signal [B, C, T]

        Returns:
            Loss value
        """
        loss = F.huber_loss(pred, target, reduction=self.reduction, delta=self.delta)
        return self.weight * loss


class QuantileHuberLoss(nn.Module):
    """Quantile Huber loss for asymmetric penalties.

    Combines Huber loss with quantile regression for asymmetric
    penalties on over/under-prediction.

    Args:
        quantile: Target quantile (0.5 = symmetric, <0.5 penalizes over-prediction)
        delta: Huber delta parameter
        reduction: Reduction method
    """

    def __init__(
        self,
        quantile: float = 0.5,
        delta: float = 1.0,
        reduction: str = "mean",
        weight: float = 1.0,
    ):
        super().__init__()
        self.quantile = quantile
        self.delta = delta
        self.reduction = reduction
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute quantile Huber loss."""
        errors = target - pred
        abs_errors = errors.abs()

        # Huber component
        huber = torch.where(
            abs_errors <= self.delta,
            0.5 * errors ** 2,
            self.delta * (abs_errors - 0.5 * self.delta)
        )

        # Quantile weighting
        quantile_weights = torch.where(
            errors >= 0,
            self.quantile,
            1 - self.quantile
        )

        loss = quantile_weights * huber

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.weight * loss


class ChannelWeightedHuberLoss(nn.Module):
    """Huber loss with per-channel weights.

    Allows different emphasis on different channels.

    Args:
        num_channels: Number of channels
        delta: Huber delta
        learnable_weights: If True, channel weights are learned
    """

    def __init__(
        self,
        num_channels: int = 32,
        delta: float = 1.0,
        learnable_weights: bool = False,
        weight: float = 1.0,
    ):
        super().__init__()
        self.delta = delta
        self.weight = weight

        if learnable_weights:
            self.channel_weights = nn.Parameter(torch.ones(num_channels))
        else:
            self.register_buffer('channel_weights', torch.ones(num_channels))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute channel-weighted Huber loss."""
        B, C, T = pred.shape

        # Compute per-channel loss
        errors = pred - target
        abs_errors = errors.abs()

        huber = torch.where(
            abs_errors <= self.delta,
            0.5 * errors ** 2,
            self.delta * (abs_errors - 0.5 * self.delta)
        )

        # Average over time, weight by channel, average over batch
        channel_loss = huber.mean(dim=-1)  # [B, C]

        # Apply channel weights (softmax for stability if learned)
        weights = F.softmax(self.channel_weights, dim=0) if hasattr(self, 'channel_weights') else self.channel_weights
        weighted_loss = (channel_loss * weights.unsqueeze(0)).sum(dim=-1).mean()

        return self.weight * weighted_loss


def create_huber_loss(
    variant: str = "standard",
    **kwargs,
) -> nn.Module:
    """Factory function for Huber loss variants.

    Args:
        variant: "standard", "quantile", or "channel_weighted"
        **kwargs: Additional arguments

    Returns:
        Huber loss module
    """
    if variant == "standard":
        return HuberLoss(**kwargs)
    elif variant == "quantile":
        return QuantileHuberLoss(**kwargs)
    elif variant == "channel_weighted":
        return ChannelWeightedHuberLoss(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
