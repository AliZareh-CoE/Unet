"""
CondUNet1D Wrapper for Phase 2
==============================

Wrapper around the main CondUNet1D implementation from models.py.
Provides consistent interface with other Phase 2 architectures.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


class CondUNet1D(nn.Module):
    """Conditional U-Net for 1D signal translation.

    Wraps the full CondUNet1D implementation from models.py,
    providing a consistent interface with other architectures.

    Features:
        - U-Net encoder-decoder with skip connections
        - FiLM conditioning for adaptive modulation
        - Cross-frequency attention mechanisms
        - Residual learning (predicts delta from input)

    Args:
        in_channels: Input channels
        out_channels: Output channels
        base_channels: Base channel dimension
        n_downsample: Number of downsampling stages
        attention_type: Type of attention ('none', 'basic', 'cross_freq', 'cross_freq_v2')
        use_film: Whether to use FiLM conditioning
        dropout: Dropout rate

    Input: [B, C_in, T]
    Output: [B, C_out, T]
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        base_channels: int = 64,
        n_downsample: int = 3,
        attention_type: str = "cross_freq_v2",
        use_film: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Try to import full implementation
        try:
            from models import CondUNet1D as FullCondUNet
            self.model = FullCondUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels,
                n_downsample=n_downsample,
                attention_type=attention_type,
            )
            self._using_full = True
        except ImportError:
            # Fallback to simplified version
            self.model = SimplifiedCondUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels,
                n_downsample=n_downsample,
                dropout=dropout,
            )
            self._using_full = False

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C_in, T]
            condition: Optional conditioning tensor

        Returns:
            Output tensor [B, C_out, T]
        """
        if self._using_full and condition is not None:
            return self.model(x, condition)
        return self.model(x)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimplifiedCondUNet(nn.Module):
    """Simplified U-Net fallback if main model unavailable.

    Basic encoder-decoder with skip connections.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        base_channels: int = 64,
        n_downsample: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_downsample = n_downsample

        # Input projection
        self.input_conv = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3)

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = base_channels
        for i in range(n_downsample):
            out_ch = ch * 2
            self.encoders.append(self._make_block(ch, out_ch, dropout))
            self.pools.append(nn.MaxPool1d(2))
            ch = out_ch

        # Bottleneck
        self.bottleneck = self._make_block(ch, ch, dropout)

        # Decoder
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(n_downsample):
            in_ch = ch + ch  # Skip connection
            out_ch = ch // 2
            self.upsamples.append(nn.ConvTranspose1d(ch, ch, kernel_size=2, stride=2))
            self.decoders.append(self._make_block(in_ch, out_ch, dropout))
            ch = out_ch

        # Output
        self.output_conv = nn.Conv1d(ch, out_channels, kernel_size=1)

        # Residual connection
        self.use_residual = in_channels == out_channels

    def _make_block(self, in_ch: int, out_ch: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Input
        x = self.input_conv(x)

        # Encoder with skip connections
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            # Handle size mismatch
            if x.shape[-1] != skip.shape[-1]:
                x = nn.functional.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # Output
        x = self.output_conv(x)

        # Residual
        if self.use_residual:
            x = x + identity

        return x
