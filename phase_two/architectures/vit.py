"""
Vision Transformer (ViT) for 1D Signals
=======================================

Adapted Vision Transformer for 1D signal translation.
Based on Dosovitskiy et al., 2020 "An Image is Worth 16x16 Words".

Treats signal segments as "patches" and applies standard transformer.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [B, N, D]

        Returns:
            Output [B, N, D]
        """
        B, N, D = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):
    """Standard transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT1D(nn.Module):
    """Vision Transformer adapted for 1D signal translation.

    Splits input into patches, applies transformer, then reconstructs.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        time_steps: Expected input time length
        embed_dim: Transformer embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        patch_size: Size of each patch (time steps)
        dropout: Dropout rate

    Input: [B, C_in, T]
    Output: [B, C_out, T]
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        time_steps: int = 5000,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        patch_size: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Number of patches
        self.n_patches = time_steps // patch_size

        # Patch embedding: [patch_size * in_channels] -> embed_dim
        patch_dim = patch_size * in_channels
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Patch reconstruction: embed_dim -> [patch_size * out_channels]
        self.head = nn.Linear(embed_dim, patch_size * out_channels)

        # Residual
        self.use_residual = in_channels == out_channels

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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
        B, C, T = x.shape
        identity = x

        # Handle variable length inputs
        n_patches = T // self.patch_size
        T_used = n_patches * self.patch_size

        # Truncate if necessary
        x = x[:, :, :T_used]

        # Create patches: [B, C, T] -> [B, n_patches, patch_size * C]
        x = x.reshape(B, C, n_patches, self.patch_size)
        x = x.permute(0, 2, 3, 1)  # [B, n_patches, patch_size, C]
        x = x.reshape(B, n_patches, -1)  # [B, n_patches, patch_size * C]

        # Patch embedding
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]

        # Add positional embeddings (interpolate if needed)
        if n_patches != self.n_patches:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=n_patches,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos = self.pos_embed

        x = x + pos

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Reconstruct patches: [B, n_patches, embed_dim] -> [B, C_out, T]
        x = self.head(x)  # [B, n_patches, patch_size * C_out]
        x = x.reshape(B, n_patches, self.patch_size, self.out_channels)
        x = x.permute(0, 3, 1, 2)  # [B, C_out, n_patches, patch_size]
        x = x.reshape(B, self.out_channels, -1)  # [B, C_out, T_used]

        # Pad back to original length if needed
        if T_used < T:
            x = F.pad(x, (0, T - T_used), mode='replicate')

        # Residual
        if self.use_residual:
            x = x + identity

        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
