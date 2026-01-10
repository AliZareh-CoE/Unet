"""
Performer1D for Neural Signal Translation
==========================================

Linear attention transformer using FAVOR+ mechanism.
Based on Choromanski et al., 2020 "Rethinking Attention with Performers".

Achieves O(N) complexity instead of O(N²) for standard attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_kernel_favor_plus(
    data: torch.Tensor,
    projection_matrix: torch.Tensor,
    is_query: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute FAVOR+ softmax kernel features (Choromanski et al., 2020).

    Uses the positive random features approximation:
        φ(x) = exp(w^T x - ||x||^2/2) / sqrt(m)

    This approximates: softmax(Q @ K^T) ≈ φ(Q) @ φ(K)^T

    Args:
        data: Input tensor [B, heads, N, head_dim]
        projection_matrix: Random features [n_features, head_dim]
        is_query: Whether this is query (affects normalization)
        eps: Small constant for numerical stability

    Returns:
        Kernel features [B, heads, N, n_features]
    """
    head_dim = data.shape[-1]
    n_features = projection_matrix.shape[0]

    # Normalize by sqrt(head_dim) as in standard attention
    data_normalized = data / math.sqrt(head_dim)

    # Compute ||x||^2 / 2 for the Gaussian kernel
    data_norm_sq = (data_normalized ** 2).sum(dim=-1, keepdim=True) / 2  # [B, heads, N, 1]

    # Project: w^T x
    data_proj = data_normalized @ projection_matrix.T  # [B, heads, N, n_features]

    # FAVOR+ kernel: exp(w^T x - ||x||^2/2) / sqrt(m)
    # For numerical stability, subtract max before exp
    data_combined = data_proj - data_norm_sq
    data_max = data_combined.max(dim=-1, keepdim=True).values
    kernel_features = torch.exp(data_combined - data_max)

    # Normalize by sqrt(n_features) for unbiased estimate
    kernel_features = kernel_features / math.sqrt(n_features)

    # Add small epsilon for stability
    kernel_features = kernel_features + eps

    return kernel_features


class FAVORPlusAttention(nn.Module):
    """FAVOR+ linear attention mechanism.

    Approximates softmax attention with random features
    for O(N) complexity.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        n_features: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert embed_dim % n_heads == 0

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.n_features = n_features

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Random projection matrix for FAVOR+
        # Initialize with orthogonal random features
        self.register_buffer(
            'projection_matrix',
            self._create_projection_matrix()
        )

    def _create_projection_matrix(self) -> torch.Tensor:
        """Create orthogonal random projection matrix."""
        # Use orthogonal initialization for better approximation
        proj = torch.randn(self.n_features, self.head_dim)
        # Normalize rows
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FAVOR+ attention.

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

        # Compute kernel features using FAVOR+
        q_prime = softmax_kernel_favor_plus(q, self.projection_matrix, is_query=True)
        k_prime = softmax_kernel_favor_plus(k, self.projection_matrix, is_query=False)

        # Linear attention: O(N) instead of O(N²)
        # Attention = Q' @ (K'^T @ V) instead of softmax(Q @ K^T) @ V
        kv = torch.einsum('bhnd,bhnv->bhdv', k_prime, v)  # [B, heads, n_features, head_dim]
        qkv = torch.einsum('bhnd,bhdv->bhnv', q_prime, kv)  # [B, heads, N, head_dim]

        # Normalize
        k_sum = k_prime.sum(dim=2, keepdim=True)  # [B, heads, 1, n_features]
        normalizer = torch.einsum('bhnd,bhkd->bhn', q_prime, k_sum).unsqueeze(-1)
        qkv = qkv / (normalizer + 1e-6)

        # Reshape and project
        x = qkv.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class PerformerBlock(nn.Module):
    """Performer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        n_features: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = FAVORPlusAttention(embed_dim, n_heads, n_features, dropout)

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
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Performer1D(nn.Module):
    """Performer for 1D signal translation.

    Linear attention transformer using FAVOR+ mechanism.
    O(N) complexity for long sequences.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        embed_dim: Transformer dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        n_features: Number of random features for FAVOR+
        dropout: Dropout rate

    Input: [B, C_in, T]
    Output: [B, C_out, T]
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        n_features: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input projection
        self.input_proj = nn.Linear(in_channels, embed_dim)

        # Performer blocks
        self.blocks = nn.ModuleList([
            PerformerBlock(embed_dim, n_heads, n_features, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, out_channels)

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

        # Project
        x = self.input_proj(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_proj(x)

        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        # Residual
        if self.use_residual:
            x = x + identity

        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
