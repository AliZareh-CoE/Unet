"""
Vision Transformer 1D Architecture
==================================

Transformer architecture adapted for 1D neural signal translation.
Based on ViT but for time series instead of image patches.

Key characteristics:
- Patch-based tokenization of time series
- Standard transformer encoder with self-attention
- Position embeddings
- Optional FiLM conditioning
- ~50-100M parameters

Reference:
    Dosovitskiy et al. "An Image is Worth 16x16 Words" (2020)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding1D(nn.Module):
    """Embed 1D signal patches.

    Converts signal [B, C, T] into patches [B, N, D] where N = T / patch_size.

    Args:
        in_channels: Input channels
        embed_dim: Embedding dimension
        patch_size: Size of each patch
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Linear projection of flattened patches
        self.proj = nn.Conv1d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed patches.

        Args:
            x: Input [B, C, T]

        Returns:
            Patch embeddings [B, N, D] where N = T / patch_size
        """
        x = self.proj(x)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        return x


class PositionalEmbedding1D(nn.Module):
    """Learnable positional embeddings for 1D sequences."""

    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings.

        Args:
            x: Input [B, N, D]

        Returns:
            Output with position info [B, N, D]
        """
        seq_len = x.shape[1]
        return x + self.pos_embed[:, :seq_len]


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, not learned)."""

    def __init__(self, embed_dim: int, max_len: int = 8192):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, D]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        return x + self.pe[:, :x.shape[1]]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Attention dropout
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
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

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):
    """Standard transformer encoder block.

    Structure:
        x -> LayerNorm -> MHSA -> + -> LayerNorm -> FFN -> + -> out
           |                   |    |                   |
           |-------------------^    |-------------------^

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ffn_ratio: FFN expansion ratio
        dropout: Dropout probability
        condition_dim: Conditioning dimension (0 to disable)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        condition_dim: int = 0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = embed_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Conditioning
        if condition_dim > 0:
            self.gamma = nn.Linear(condition_dim, embed_dim)
            self.beta = nn.Linear(condition_dim, embed_dim)
            nn.init.ones_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)
        else:
            self.gamma = None
            self.beta = None

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [B, N, D]
            condition: Conditioning [B, condition_dim]

        Returns:
            Output [B, N, D]
        """
        # Self-attention
        x = x + self.attn(self.norm1(x))

        # FFN
        x = x + self.ffn(self.norm2(x))

        # Apply conditioning (FiLM-like)
        if self.gamma is not None and self.beta is not None and condition is not None:
            gamma = self.gamma(condition).unsqueeze(1)  # [B, 1, D]
            beta = self.beta(condition).unsqueeze(1)
            x = gamma * x + beta

        return x


class ViT1D(nn.Module):
    """Vision Transformer adapted for 1D signals.

    Tokenizes time series into patches and processes with transformer.

    Args:
        in_channels: Input channels (default: 32)
        out_channels: Output channels (default: 32)
        patch_size: Patch size (default: 32)
        embed_dim: Embedding dimension (default: 256)
        num_layers: Number of transformer blocks (default: 6)
        num_heads: Attention heads (default: 8)
        ffn_ratio: FFN expansion ratio (default: 4)
        num_classes: Number of odor classes (default: 7)
        condition_dim: Conditioning dimension (default: 64)
        use_conditioning: Whether to use conditioning (default: True)
        dropout: Dropout probability (default: 0.1)
        pos_embed_type: "learned" or "sinusoidal" (default: "learned")
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        patch_size: int = 32,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        num_classes: int = 7,
        condition_dim: int = 64,
        use_conditioning: bool = True,
        dropout: float = 0.1,
        pos_embed_type: str = "learned",
        max_seq_len: int = 512,  # Max number of patches
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_residual = (in_channels == out_channels)

        # Conditioning
        if use_conditioning:
            self.embed = nn.Embedding(num_classes, condition_dim)
            nn.init.normal_(self.embed.weight, std=0.02)
            cond_dim = condition_dim
        else:
            self.embed = None
            cond_dim = 0

        # Patch embedding
        self.patch_embed = PatchEmbedding1D(in_channels, embed_dim, patch_size)

        # Position embedding
        if pos_embed_type == "learned":
            self.pos_embed = PositionalEmbedding1D(max_seq_len, embed_dim)
        else:
            self.pos_embed = SinusoidalPositionalEncoding(embed_dim, max_seq_len)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_ratio, dropout, cond_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output projection: unpatchify
        self.unpatchify = nn.ConvTranspose1d(
            embed_dim, out_channels,
            kernel_size=patch_size, stride=patch_size,
        )

        # Residual projection
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

        # Initialize output small
        nn.init.zeros_(self.unpatchify.weight)
        nn.init.zeros_(self.unpatchify.bias)

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

        # Get conditioning
        if self.embed is not None and odor_ids is not None:
            condition = self.embed(odor_ids)
        else:
            condition = None

        # Patch embedding
        x = self.patch_embed(ob)  # [B, N, D]

        # Add position embedding
        x = self.pos_embed(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, condition)

        x = self.norm(x)

        # Unpatchify: [B, N, D] -> [B, D, N] -> [B, C_out, T']
        x = x.transpose(1, 2)
        delta = self.unpatchify(x)

        # Handle size mismatch
        if delta.shape[-1] != T:
            delta = F.interpolate(delta, size=T, mode='linear', align_corners=False)

        # Residual connection
        if self.use_residual:
            out = ob + delta
        elif self.residual_proj is not None:
            out = self.residual_proj(ob) + delta
        else:
            out = delta

        return out

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'patch_embed': sum(p.numel() for p in self.patch_embed.parameters()),
            'blocks': sum(p.numel() for block in self.blocks for p in block.parameters()),
            'unpatchify': sum(p.numel() for p in self.unpatchify.parameters()),
            'embed': self.embed.weight.numel() if self.embed else 0,
        }

    def get_layer_names(self) -> List[str]:
        """Get layer names."""
        names = ['patch_embed', 'pos_embed']
        names.extend([f'block_{i}' for i in range(self.num_layers)])
        names.extend(['norm', 'unpatchify'])
        return names


class ViT1DWithConvStem(nn.Module):
    """ViT with convolutional stem for better local features.

    Adds convolutional layers before transformer for better
    initial feature extraction.

    Args:
        Same as ViT1D plus stem_channels
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        stem_channels: int = 64,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
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
            self.cond_proj = nn.Linear(64, embed_dim)
        else:
            self.embed = None
            self.cond_proj = None

        # Conv stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=7, padding=3),
            nn.GroupNorm(8, stem_channels),
            nn.GELU(),
            nn.Conv1d(stem_channels, stem_channels, kernel_size=7, padding=3),
            nn.GroupNorm(8, stem_channels),
            nn.GELU(),
        )

        # Patch embed from stem output
        self.patch_embed = PatchEmbedding1D(stem_channels, embed_dim, patch_size)
        self.pos_embed = SinusoidalPositionalEncoding(embed_dim)

        # Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Output head
        self.unpatchify = nn.ConvTranspose1d(
            embed_dim, stem_channels,
            kernel_size=patch_size, stride=patch_size,
        )
        self.head = nn.Sequential(
            nn.Conv1d(stem_channels, stem_channels, kernel_size=7, padding=3),
            nn.GroupNorm(8, stem_channels),
            nn.GELU(),
            nn.Conv1d(stem_channels, out_channels, kernel_size=1),
        )

        # Residual
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

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

        # Conditioning
        if self.embed is not None and self.cond_proj is not None and odor_ids is not None:
            condition = self.embed(odor_ids)
            cond_vec = self.cond_proj(condition)
        else:
            cond_vec = None

        # Conv stem
        x = self.stem(ob)

        # Patch + position
        x = self.patch_embed(x)
        x = self.pos_embed(x)

        # Add conditioning
        if cond_vec is not None:
            x = x + cond_vec.unsqueeze(1)

        # Transformer
        for block in self.blocks:
            x = block(x, None)  # Already added conditioning
        x = self.norm(x)

        # Decode
        x = x.transpose(1, 2)
        x = self.unpatchify(x)
        if x.shape[-1] != T:
            x = F.interpolate(x, size=T, mode='linear', align_corners=False)
        delta = self.head(x)

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


def create_vit(
    variant: str = "standard",
    **kwargs,
) -> nn.Module:
    """Factory function for ViT variants.

    Args:
        variant: "standard" or "conv_stem"
        **kwargs: Model arguments

    Returns:
        ViT model
    """
    if variant == "standard":
        return ViT1D(**kwargs)
    elif variant == "conv_stem":
        return ViT1DWithConvStem(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Available: standard, conv_stem")
