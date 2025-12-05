"""
FNet 1D Architecture
====================

FFT-based architecture for neural signal translation.
Replaces self-attention with Fourier transforms for efficient global mixing.

Key characteristics:
- FFT-based token mixing (O(T log T) instead of O(T²) attention)
- No learned parameters in mixing layer
- Fast inference
- Natural frequency domain processing for neural signals
- ~50M parameters

Reference:
    Lee-Thorp et al. "FNet: Mixing Tokens with Fourier Transforms" (2021)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierMixing1D(nn.Module):
    """Fourier mixing layer for 1D sequences.

    Replaces attention with FFT along temporal dimension.
    This provides global mixing with O(T log T) complexity.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier mixing.

        Args:
            x: Input [B, C, T]

        Returns:
            Mixed output [B, C, T]
        """
        # FFT along time dimension, take real part
        # This mixes information across all time steps
        fft_out = torch.fft.fft(x, dim=-1)
        return fft_out.real


class FourierMixing2D(nn.Module):
    """Fourier mixing along both channel and time dimensions."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D Fourier mixing.

        Args:
            x: Input [B, C, T]

        Returns:
            Mixed output [B, C, T]
        """
        # FFT along both channel and time dimensions
        fft_out = torch.fft.fft2(x, dim=(-2, -1))
        return fft_out.real


class FNetBlock(nn.Module):
    """FNet block with Fourier mixing and feedforward.

    Structure:
        input -> LayerNorm -> Fourier Mixing -> + -> LayerNorm -> FFN -> + -> output
              |                               |                         |
              |-------------------------------^                         |
              |--------------------------------------------------------|

    Args:
        channels: Channel dimension
        ffn_ratio: Feedforward expansion ratio (default: 4)
        dropout: Dropout probability
        mixing_type: "1d" or "2d" Fourier mixing
    """

    def __init__(
        self,
        channels: int,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        mixing_type: str = "1d",
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        if mixing_type == "1d":
            self.mixing = FourierMixing1D()
        else:
            self.mixing = FourierMixing2D()

        # Feedforward network
        hidden_dim = channels * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [B, C, T]

        Returns:
            Output [B, C, T]
        """
        # Transpose for LayerNorm (expects last dim = channels)
        # x: [B, C, T] -> [B, T, C]
        x_t = x.transpose(1, 2)

        # Fourier mixing sublayer
        h = self.norm1(x_t).transpose(1, 2)  # [B, C, T]
        h = self.mixing(h).transpose(1, 2)  # [B, T, C]
        x_t = x_t + h

        # FFN sublayer
        h = self.norm2(x_t)
        h = self.ffn(h)
        x_t = x_t + h

        # Transpose back: [B, T, C] -> [B, C, T]
        return x_t.transpose(1, 2)


class FNet1D(nn.Module):
    """FNet for neural signal translation.

    A transformer-like architecture that replaces attention with FFT.
    Particularly suited for neural signals where frequency information is important.

    Architecture:
        - Patch embedding (optional)
        - Stacked FNet blocks
        - Output projection
        - Residual connection

    Args:
        in_channels: Input channels (default: 32)
        out_channels: Output channels (default: 32)
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of FNet blocks (default: 6)
        ffn_ratio: FFN expansion ratio (default: 4)
        patch_size: Patch size for embedding (default: 1, no patching)
        num_classes: Number of odor classes (default: 7)
        condition_dim: Conditioning dimension (default: 64)
        use_conditioning: Whether to use conditioning (default: True)
        dropout: Dropout probability (default: 0.1)
        mixing_type: "1d" or "2d" Fourier mixing (default: "1d")
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 6,
        ffn_ratio: int = 4,
        patch_size: int = 1,
        num_classes: int = 7,
        condition_dim: int = 64,
        use_conditioning: bool = True,
        dropout: float = 0.1,
        mixing_type: str = "1d",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.use_residual = (in_channels == out_channels)

        # Conditioning
        if use_conditioning:
            self.embed = nn.Embedding(num_classes, condition_dim)
            nn.init.normal_(self.embed.weight, std=0.02)
            self.cond_proj = nn.Linear(condition_dim, hidden_dim)
        else:
            self.embed = None
            self.cond_proj = None

        # Input embedding
        if patch_size > 1:
            self.input_proj = nn.Conv1d(
                in_channels, hidden_dim,
                kernel_size=patch_size, stride=patch_size,
            )
            self.output_unproj = nn.ConvTranspose1d(
                hidden_dim, hidden_dim,
                kernel_size=patch_size, stride=patch_size,
            )
        else:
            self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
            self.output_unproj = None

        # Stacked FNet blocks
        self.blocks = nn.ModuleList([
            FNetBlock(hidden_dim, ffn_ratio, dropout, mixing_type)
            for _ in range(num_layers)
        ])

        # Final norm and output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)

        # Residual projection
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

        # Initialize output small
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
            odor_ids: Odor class indices [B]

        Returns:
            Predicted signal [B, C_out, T]
        """
        if ob.dim() != 3:
            raise ValueError(f"Expected 3D input [B, C, T], got shape {ob.shape}")

        B, C, T = ob.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")

        # Input embedding
        x = self.input_proj(ob)  # [B, hidden_dim, T'] where T' = T / patch_size

        # Add conditioning
        if self.embed is not None and self.cond_proj is not None and odor_ids is not None:
            cond = self.embed(odor_ids)  # [B, cond_dim]
            cond = self.cond_proj(cond)  # [B, hidden_dim]
            x = x + cond.unsqueeze(-1)  # Broadcast over time

        # Apply FNet blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = x.transpose(1, 2)  # [B, T', hidden_dim]
        x = self.final_norm(x)
        x = x.transpose(1, 2)  # [B, hidden_dim, T']

        # Upsample if patched
        if self.output_unproj is not None:
            x = self.output_unproj(x)
            # Handle size mismatch
            if x.shape[-1] != T:
                x = F.interpolate(x, size=T, mode='linear', align_corners=False)

        # Output projection
        delta = self.output_proj(x)

        # Residual connection
        if self.use_residual:
            out = ob + delta
        elif self.residual_proj is not None:
            out = self.residual_proj(ob) + delta
        else:
            out = delta

        return out

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'input_proj': sum(p.numel() for p in self.input_proj.parameters()),
            'blocks': sum(p.numel() for block in self.blocks for p in block.parameters()),
            'output_proj': sum(p.numel() for p in self.output_proj.parameters()),
            'embed': self.embed.weight.numel() if self.embed else 0,
        }

    def get_layer_names(self) -> List[str]:
        """Get layer names."""
        names = ['input_proj']
        names.extend([f'block_{i}' for i in range(self.num_layers)])
        names.extend(['final_norm', 'output_proj'])
        return names


class FNetWithConv(nn.Module):
    """FNet with convolutional stem and head.

    Combines FNet global mixing with local convolutions for better
    local feature extraction.

    Args:
        Same as FNet1D plus conv_layers for local processing
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 4,
        conv_layers: int = 2,
        kernel_size: int = 7,
        num_classes: int = 7,
        use_conditioning: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = (in_channels == out_channels)

        # Conditioning
        if use_conditioning:
            self.embed = nn.Embedding(num_classes, 64)
            nn.init.normal_(self.embed.weight, std=0.02)
            self.cond_proj = nn.Linear(64, hidden_dim)
        else:
            self.embed = None
            self.cond_proj = None

        # Convolutional stem
        self.stem = nn.ModuleList()
        ch = in_channels
        for i in range(conv_layers):
            out_ch = hidden_dim if i == conv_layers - 1 else min(hidden_dim, ch * 2)
            self.stem.append(nn.Sequential(
                nn.Conv1d(ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
            ))
            ch = out_ch

        # FNet blocks
        self.blocks = nn.ModuleList([
            FNetBlock(hidden_dim, ffn_ratio=4, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Convolutional head
        self.head = nn.ModuleList()
        for i in range(conv_layers):
            in_ch = hidden_dim if i == 0 else min(hidden_dim, hidden_dim // (2 ** i))
            out_ch = out_channels if i == conv_layers - 1 else hidden_dim // (2 ** (i + 1))
            self.head.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(min(8, out_ch), out_ch) if i < conv_layers - 1 else nn.Identity(),
                nn.GELU() if i < conv_layers - 1 else nn.Identity(),
            ))

        # Residual
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

        # Initialize last layer small
        nn.init.zeros_(self.head[-1][0].weight)
        nn.init.zeros_(self.head[-1][0].bias)

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        if ob.dim() != 3:
            raise ValueError(f"Expected 3D input [B, C, T], got shape {ob.shape}")

        B, C, T = ob.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")

        # Stem
        x = ob
        for conv in self.stem:
            x = conv(x)

        # Add conditioning
        if self.embed is not None and self.cond_proj is not None and odor_ids is not None:
            cond = self.embed(odor_ids)
            cond = self.cond_proj(cond)
            x = x + cond.unsqueeze(-1)

        # FNet blocks
        for block in self.blocks:
            x = block(x)

        # Head
        for conv in self.head:
            x = conv(x)

        delta = x

        # Residual
        if self.use_residual:
            out = ob + delta
        elif self.residual_proj is not None:
            out = self.residual_proj(ob) + delta
        else:
            out = delta

        return out

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters."""
        return {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def create_fnet(
    variant: str = "standard",
    **kwargs,
) -> nn.Module:
    """Factory function for FNet variants.

    Args:
        variant: "standard", "conv", or "2d"
        **kwargs: Arguments for model

    Returns:
        FNet model
    """
    if variant == "standard":
        return FNet1D(**kwargs)
    elif variant == "conv":
        return FNetWithConv(**kwargs)
    elif variant == "2d":
        return FNet1D(mixing_type="2d", **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Available: standard, conv, 2d")
