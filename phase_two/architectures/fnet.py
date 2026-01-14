"""
FNet1D for Neural Signal Translation
=====================================

Fourier-based token mixing (Lee-Thorp et al., 2021).
Replaces self-attention with FFT - O(N log N) complexity.

Reference: "FNet: Mixing Tokens with Fourier Transforms"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.fft


class FourierMixing(nn.Module):
    """Fourier-based mixing layer.

    Applies 2D FFT to mix information across both
    sequence (time) and hidden dimensions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier mixing.

        Args:
            x: Input [B, T, D]

        Returns:
            Mixed tensor [B, T, D]
        """
        # 2D FFT over sequence and hidden dimensions
        # Take only real part for simplicity
        # NOTE: FFT doesn't support BFloat16, so cast to float32 for FSDP compatibility
        orig_dtype = x.dtype
        if x.dtype == torch.bfloat16:
            x = x.float()
        result = torch.fft.fft2(x, dim=(-2, -1)).real
        return result.to(orig_dtype)


class FNetBlock(nn.Module):
    """FNet encoder block.

    Architecture:
        - Fourier mixing + residual + LayerNorm
        - MLP + residual + LayerNorm
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        mlp_dim = mlp_dim or hidden_dim * 4

        # Fourier mixing
        self.fourier = FourierMixing()
        self.norm1 = nn.LayerNorm(hidden_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [B, T, D]

        Returns:
            Output [B, T, D]
        """
        # Fourier mixing + residual
        x = x + self.fourier(self.norm1(x))

        # MLP + residual
        x = x + self.mlp(self.norm2(x))

        return x


class FNet1D(nn.Module):
    """FNet for 1D signal translation.

    Replaces attention with FFT-based mixing for efficient
    global information exchange.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        hidden_dim: Hidden dimension
        n_layers: Number of FNet blocks
        dropout: Dropout rate

    Input: [B, C_in, T]
    Output: [B, C_out, T]
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        # Input embedding: project channels to hidden dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # FNet blocks
        self.blocks = nn.ModuleList([
            FNetBlock(hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_channels)

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Residual
        self.use_residual = in_channels == out_channels

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
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

        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        # Input projection
        x = self.input_proj(x)  # [B, T, hidden_dim]

        # FNet blocks
        for block in self.blocks:
            x = block(x)

        # Final norm and output projection
        x = self.final_norm(x)
        x = self.output_proj(x)  # [B, T, C_out]

        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        # Residual
        if self.use_residual:
            x = x + identity

        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
