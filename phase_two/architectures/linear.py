"""
Linear Baseline for Neural Signal Translation
==============================================

Simple linear mapping as a neural network baseline.
Serves as the simplest possible learned approach.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class LinearBaseline(nn.Module):
    """Linear baseline: simple learned linear mapping.

    Maps input channels to output channels at each time point.
    Can optionally include a hidden layer.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_steps: Number of time steps (for optional temporal mixing)
        hidden_dim: Hidden dimension (None = direct mapping)
        use_temporal: If True, also mix across time

    Input: [B, C_in, T]
    Output: [B, C_out, T]
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        time_steps: int = 5000,
        hidden_dim: Optional[int] = 256,
        use_temporal: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_steps = time_steps
        self.hidden_dim = hidden_dim
        self.use_temporal = use_temporal

        if hidden_dim is not None:
            # Two-layer MLP per time step
            self.net = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_channels),
            )
        else:
            # Direct linear mapping
            self.net = nn.Linear(in_channels, out_channels)

        if use_temporal:
            # Additional temporal mixing
            self.temporal_mix = nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel_size=11, padding=5, groups=out_channels),
                nn.GELU(),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C_in, T]

        Returns:
            Output tensor [B, C_out, T]
        """
        B, C, T = x.shape

        # Transpose to [B, T, C] for linear layer
        x = x.transpose(1, 2)  # [B, T, C_in]

        # Apply MLP
        x = self.net(x)  # [B, T, C_out]

        # Transpose back to [B, C, T]
        x = x.transpose(1, 2)  # [B, C_out, T]

        # Optional temporal mixing
        if self.use_temporal:
            x = self.temporal_mix(x)

        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
