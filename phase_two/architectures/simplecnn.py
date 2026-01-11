"""
Simple CNN for Neural Signal Translation
=========================================

Basic convolutional neural network baseline.
Uses standard conv-bn-relu blocks without fancy components.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic conv-bn-relu block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int = 3,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SimpleCNN(nn.Module):
    """Simple CNN for signal translation.

    Architecture:
        - Input projection
        - Stack of conv-bn-relu blocks
        - Output projection

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        channels: List of channel dimensions for each block
        kernel_size: Convolution kernel size

    Input: [B, C_in, T]
    Output: [B, C_out, T]
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        channels: List[int] = [64, 128, 256, 128, 64],
        kernel_size: int = 7,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, channels[0], kernel_size=1)

        # Conv blocks
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(ConvBlock(
                channels[i], channels[i + 1],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ))
        self.blocks = nn.Sequential(*blocks)

        # Output projection
        self.output_proj = nn.Conv1d(channels[-1], out_channels, kernel_size=1)

        # Residual connection (if in/out channels match)
        self.use_residual = in_channels == out_channels

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C_in, T]

        Returns:
            Output tensor [B, C_out, T]
        """
        identity = x

        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_proj(x)

        # Residual connection
        if self.use_residual:
            x = x + identity

        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
