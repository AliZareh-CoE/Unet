"""
Loss Functions for Phase 3 Ablations
=====================================

Provides different loss function variants for ablation study.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


class L1Loss(nn.Module):
    """Simple L1 (MAE) loss."""

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss = F.l1_loss(pred, target)
        return loss, {"l1": loss}


class HuberLoss(nn.Module):
    """Huber loss (smooth L1).

    Less sensitive to outliers than L1, good for noisy neural signals.
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss = F.smooth_l1_loss(pred, target, beta=self.delta)
        return loss, {"huber": loss}


class WaveletLossWrapper(nn.Module):
    """Wrapper for WaveletLoss from models.py.

    Multi-scale frequency alignment using continuous wavelet transform.
    """

    def __init__(self):
        super().__init__()
        try:
            from models import WaveletLoss
            self.wavelet_loss = WaveletLoss()
            self._available = True
        except ImportError:
            self._available = False

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self._available:
            # Fallback to L1 if wavelet loss not available
            loss = F.l1_loss(pred, target)
            return loss, {"wavelet": loss}

        loss = self.wavelet_loss(pred, target)
        return loss, {"wavelet": loss}


class CombinedLoss(nn.Module):
    """Combined loss with Huber + Wavelet components.

    Args:
        huber_weight: Weight for Huber loss component
        wavelet_weight: Weight for Wavelet loss component
        huber_delta: Delta parameter for Huber loss
    """

    def __init__(
        self,
        huber_weight: float = 1.0,
        wavelet_weight: float = 0.5,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.huber_weight = huber_weight
        self.wavelet_weight = wavelet_weight
        self.huber_delta = huber_delta

        # Wavelet loss
        try:
            from models import WaveletLoss
            self.wavelet_loss = WaveletLoss()
            self._has_wavelet = True
        except ImportError:
            self._has_wavelet = False

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Huber loss
        huber = F.smooth_l1_loss(pred, target, beta=self.huber_delta)

        # Wavelet loss
        if self._has_wavelet:
            wavelet = self.wavelet_loss(pred, target)
        else:
            wavelet = F.l1_loss(pred, target)

        # Combine
        total = self.huber_weight * huber + self.wavelet_weight * wavelet

        return total, {
            "total": total,
            "huber": huber,
            "wavelet": wavelet,
        }


def build_loss(
    loss_type: str,
    loss_weights: Optional[Dict[str, float]] = None,
) -> nn.Module:
    """Build loss function based on configuration.

    Args:
        loss_type: Type of loss ('l1', 'huber', 'wavelet', 'combined')
        loss_weights: Weights for combined loss components

    Returns:
        Loss module
    """
    if loss_type == "l1":
        return L1Loss()

    elif loss_type == "huber":
        return HuberLoss()

    elif loss_type == "wavelet":
        return WaveletLossWrapper()

    elif loss_type == "combined":
        weights = loss_weights or {"huber": 1.0, "wavelet": 0.5}
        return CombinedLoss(
            huber_weight=weights.get("huber", 1.0),
            wavelet_weight=weights.get("wavelet", 0.5),
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
