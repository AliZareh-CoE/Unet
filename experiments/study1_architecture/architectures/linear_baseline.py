"""
Linear Baseline Architecture
============================

Simple linear regression baseline for neural signal translation.
This establishes the floor performance that all deep learning methods must beat.

Key characteristics:
- Per-channel linear mapping
- Optional odor conditioning via bias
- Minimal parameters (~2M for 32 channels)
- No temporal modeling (treats each time point independently)
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBaseline(nn.Module):
    """Linear baseline for neural signal translation.

    Maps input channels to output channels with optional odor conditioning.
    This is the simplest possible model - deep learning must beat this.

    Architecture:
        - Per-channel linear mapping (no cross-channel interaction)
        - OR shared linear projection across all channels
        - Optional odor-conditioned bias
        - Residual connection

    Args:
        in_channels: Number of input channels (default: 32 for OB)
        out_channels: Number of output channels (default: 32 for PCx)
        num_classes: Number of odor classes for conditioning (default: 7)
        use_conditioning: Whether to use odor conditioning (default: True)
        shared_weights: If True, use same weights for all channels (default: False)
        use_residual: If True, add input to output (default: True)
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        num_classes: int = 7,
        use_conditioning: bool = True,
        shared_weights: bool = False,
        use_residual: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.use_conditioning = use_conditioning
        self.shared_weights = shared_weights
        self.use_residual = use_residual and (in_channels == out_channels)

        # Main linear transformation
        if shared_weights:
            # Single set of weights applied to all channels
            self.linear = nn.Linear(1, 1, bias=True)
            # Additional channel mixing layer
            self.channel_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            # Per-channel linear mapping using 1x1 convolution
            self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
            self.channel_proj = None

        # Odor conditioning (adds per-class bias to each output channel)
        if use_conditioning:
            self.odor_embed = nn.Embedding(num_classes, out_channels)
            # Initialize with small values
            nn.init.normal_(self.odor_embed.weight, std=0.02)
        else:
            self.odor_embed = None

        # Residual projection if channels don't match
        if not self.use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

        # Initialize weights for residual-friendly behavior
        self._init_weights()

    def _init_weights(self):
        """Initialize weights so output ≈ input at start."""
        if self.shared_weights:
            nn.init.ones_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            nn.init.eye_(self.channel_proj.weight[:, :, 0])
            nn.init.zeros_(self.channel_proj.bias)
        else:
            # Initialize as identity if possible
            if self.in_channels == self.out_channels:
                nn.init.eye_(self.linear.weight[:, :, 0])
            else:
                nn.init.kaiming_normal_(self.linear.weight, mode='fan_out')
            nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            ob: Input signal [B, C, T]
            odor_ids: Odor class indices [B] for conditioning

        Returns:
            Predicted signal [B, C_out, T]
        """
        # Validate input shape
        if ob.dim() != 3:
            raise ValueError(f"Expected 3D input [B, C, T], got shape {ob.shape}")

        B, C, T = ob.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")

        # Main linear transformation
        if self.shared_weights:
            # Apply same linear transform to each channel independently
            # Reshape: [B, C, T] -> [B*C, T, 1] -> apply linear -> [B, C, T]
            x = ob.permute(0, 2, 1).reshape(B * T, C, 1)  # [B*T, C, 1]
            x = x.permute(0, 2, 1)  # [B*T, 1, C]
            x = self.linear(x)  # [B*T, 1, C]
            x = x.permute(0, 2, 1).reshape(B, T, C).permute(0, 2, 1)  # [B, C, T]
            x = self.channel_proj(x)  # [B, C_out, T]
        else:
            x = self.linear(ob)  # [B, C_out, T]

        # Apply odor conditioning as additive bias
        if self.odor_embed is not None and odor_ids is not None:
            odor_bias = self.odor_embed(odor_ids)  # [B, C_out]
            x = x + odor_bias.unsqueeze(-1)  # [B, C_out, T] + [B, C_out, 1]

        # Residual connection
        if self.use_residual:
            out = ob + x
        elif self.residual_proj is not None:
            out = self.residual_proj(ob) + x
        else:
            out = x

        return out

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters.

        Returns:
            Dictionary with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        breakdown = {
            'total': total,
            'trainable': trainable,
            'linear': sum(p.numel() for p in self.linear.parameters()),
        }

        if self.channel_proj is not None:
            breakdown['channel_proj'] = sum(p.numel() for p in self.channel_proj.parameters())

        if self.odor_embed is not None:
            breakdown['odor_embed'] = self.odor_embed.weight.numel()

        if self.residual_proj is not None:
            breakdown['residual_proj'] = sum(p.numel() for p in self.residual_proj.parameters())

        return breakdown

    def get_layer_names(self) -> list:
        """Get names of all layers for compatibility with analysis tools.

        Returns:
            List of layer names
        """
        names = ['linear']
        if self.channel_proj is not None:
            names.append('channel_proj')
        if self.odor_embed is not None:
            names.append('odor_embed')
        return names


class LinearChannelMixer(nn.Module):
    """Linear baseline with cross-channel interaction.

    Slightly more complex than LinearBaseline - allows learning
    cross-channel correlations through a mixing matrix.

    Architecture:
        - Input projection
        - Channel mixing matrix
        - Output projection
        - Optional odor conditioning
        - Residual connection

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        hidden_channels: Hidden dimension for mixing (default: same as in_channels)
        num_classes: Number of odor classes
        use_conditioning: Whether to use odor conditioning
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        hidden_channels: Optional[int] = None,
        num_classes: int = 7,
        use_conditioning: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels or in_channels
        self.use_residual = (in_channels == out_channels)

        # Two-layer linear transform with channel mixing
        self.input_proj = nn.Conv1d(in_channels, self.hidden_channels, kernel_size=1)
        self.mixing = nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=1)
        self.output_proj = nn.Conv1d(self.hidden_channels, out_channels, kernel_size=1)

        # Odor conditioning
        if use_conditioning:
            self.odor_embed = nn.Embedding(num_classes, out_channels)
            nn.init.normal_(self.odor_embed.weight, std=0.02)
        else:
            self.odor_embed = None

        # Initialize for residual-friendly behavior
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_normal_(self.input_proj.weight, mode='fan_out')
        nn.init.zeros_(self.input_proj.bias)
        nn.init.kaiming_normal_(self.mixing.weight, mode='fan_out')
        nn.init.zeros_(self.mixing.bias)
        # Initialize output to produce small values
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            ob: Input signal [B, C, T]
            odor_ids: Odor class indices [B] for conditioning

        Returns:
            Predicted signal [B, C_out, T]
        """
        if ob.dim() != 3:
            raise ValueError(f"Expected 3D input [B, C, T], got shape {ob.shape}")

        # Linear transforms with channel mixing
        x = self.input_proj(ob)
        x = self.mixing(x)
        delta = self.output_proj(x)

        # Apply odor conditioning
        if self.odor_embed is not None and odor_ids is not None:
            odor_bias = self.odor_embed(odor_ids)
            delta = delta + odor_bias.unsqueeze(-1)

        # Residual connection
        if self.use_residual:
            out = ob + delta
        else:
            out = delta

        return out

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        return {
            'total': total,
            'trainable': total,
            'input_proj': sum(p.numel() for p in self.input_proj.parameters()),
            'mixing': sum(p.numel() for p in self.mixing.parameters()),
            'output_proj': sum(p.numel() for p in self.output_proj.parameters()),
            'odor_embed': self.odor_embed.weight.numel() if self.odor_embed else 0,
        }


def create_linear_baseline(
    variant: str = "simple",
    **kwargs,
) -> nn.Module:
    """Factory function to create linear baseline variants.

    Args:
        variant: "simple" (per-channel), "shared" (shared weights), or "mixer" (with channel mixing)
        **kwargs: Additional arguments passed to the model

    Returns:
        Linear baseline model
    """
    if variant == "simple":
        return LinearBaseline(shared_weights=False, **kwargs)
    elif variant == "shared":
        return LinearBaseline(shared_weights=True, **kwargs)
    elif variant == "mixer":
        return LinearChannelMixer(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Available: simple, shared, mixer")
