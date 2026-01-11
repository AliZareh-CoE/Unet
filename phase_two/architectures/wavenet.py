"""
WaveNet1D for Neural Signal Translation
=======================================

Dilated causal convolutions inspired by WaveNet (van den Oord et al., 2016).
Adapted for signal-to-signal translation (non-autoregressive).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvBlock(nn.Module):
    """Dilated convolution block with gated activation.

    Uses gated activation: tanh(W_f * x) ⊙ σ(W_g * x)
    """

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()

        self.dilation = dilation

        # Dilated convolution (doubled for gating)
        self.conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,  # For gating
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
        )

        # 1x1 convolutions for residual and skip
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input [B, C, T]

        Returns:
            residual: For next layer [B, C, T]
            skip: For skip connection [B, skip_channels, T]
        """
        # Dilated conv with gated activation
        h = self.conv(x)
        h_filter, h_gate = h.chunk(2, dim=1)
        h = torch.tanh(h_filter) * torch.sigmoid(h_gate)

        # Residual and skip outputs
        residual = self.res_conv(h) + x
        skip = self.skip_conv(h)

        return residual, skip


class WaveNet1D(nn.Module):
    """WaveNet-style architecture for signal translation.

    Non-autoregressive version using dilated convolutions with
    exponentially increasing dilation factors.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        residual_channels: Channels in residual path
        skip_channels: Channels in skip connections
        n_layers: Number of dilated conv layers
        kernel_size: Convolution kernel size
        max_dilation: Maximum dilation factor

    Input: [B, C_in, T]
    Output: [B, C_out, T]
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        residual_channels: int = 64,
        skip_channels: int = 128,
        n_layers: int = 8,
        kernel_size: int = 3,
        max_dilation: int = 512,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        # Input projection
        self.input_conv = nn.Conv1d(in_channels, residual_channels, kernel_size=1)

        # Dilated conv blocks with exponential dilation
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = min(2 ** i, max_dilation)
            self.blocks.append(DilatedConvBlock(
                residual_channels=residual_channels,
                skip_channels=skip_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ))

        # Output network
        self.output_net = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(skip_channels, out_channels, kernel_size=1),
        )

        # Residual connection
        self.use_residual = in_channels == out_channels

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C_in, T]

        Returns:
            Output tensor [B, C_out, T]
        """
        identity = x

        # Input projection
        x = self.input_conv(x)

        # Accumulate skip connections
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip

        # Output
        out = self.output_net(skip_sum)

        # Residual
        if self.use_residual:
            out = out + identity

        return out

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def receptive_field(self) -> int:
        """Calculate receptive field size."""
        rf = 1
        for i in range(self.n_layers):
            dilation = 2 ** i
            rf += (3 - 1) * dilation  # kernel_size=3
        return rf
