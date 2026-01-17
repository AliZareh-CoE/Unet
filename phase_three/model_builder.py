"""
Model Builder for Phase 3 Ablations
====================================

Constructs CondUNet variants with ablation-specific configurations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from .config import AblationConfig


def build_condunet(
    config: AblationConfig,
    in_channels: int = 32,
    out_channels: int = 32,
) -> nn.Module:
    """Build CondUNet model with ablation-specific settings.

    Args:
        config: Ablation configuration
        in_channels: Input channels
        out_channels: Output channels

    Returns:
        Configured CondUNet model
    """
    try:
        from models import CondUNet1D

        model = CondUNet1D(
            in_channels=in_channels,
            out_channels=out_channels,
            base=config.base_channels,
            n_odors=7,  # Default for olfactory
            emb_dim=128,
            dropout=config.dropout,
            use_attention=config.attention_type != "none",
            attention_type=config.attention_type if config.attention_type != "none" else "basic",
            norm_type=config.norm_type,
            cond_mode=config.cond_mode,
            n_downsample=config.n_downsample,
            conv_type=config.conv_type,
            use_output_scaling=True,
            # Session conditioning - statistics-based approach
            # No session IDs needed - computes statistics from input signal
            use_session_stats=config.use_session_stats,
            session_emb_dim=config.session_emb_dim,
            session_use_spectral=config.session_use_spectral,
        )
        return model

    except ImportError:
        # Fallback to simplified version
        return SimplifiedCondUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=config.base_channels,
            n_downsample=config.n_downsample,
            use_attention=config.attention_type != "none",
        )


class SimplifiedCondUNet(nn.Module):
    """Simplified CondUNet fallback.

    Used when the full model from models.py is not available.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        base_channels: int = 64,
        n_downsample: int = 2,
        use_attention: bool = True,
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

        # Bottleneck with optional attention
        bottleneck_layers = [self._make_block(ch, ch, dropout)]
        if use_attention:
            bottleneck_layers.append(SimpleSelfAttention(ch))
        self.bottleneck = nn.Sequential(*bottleneck_layers)

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
            nn.InstanceNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [B, C, T]
            odor_ids: Optional odor conditioning (ignored in simplified version)

        Returns:
            Output [B, C, T]
        """
        identity = x

        # Normalize input
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)

        # Input projection
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

        # Residual connection
        if self.use_residual:
            x = x + identity

        return x


class SimpleSelfAttention(nn.Module):
    """Simple self-attention for bottleneck."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.key = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        q = self.query(x).permute(0, 2, 1)  # [B, T, C//r]
        k = self.key(x)  # [B, C//r, T]
        v = self.value(x).permute(0, 2, 1)  # [B, T, C]

        attn = torch.bmm(q, k)  # [B, T, T]
        attn = torch.softmax(attn / (C ** 0.5), dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1)  # [B, C, T]

        return x + self.gamma * out


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
