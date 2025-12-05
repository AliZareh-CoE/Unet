"""
Performer 1D Architecture
=========================

Linear attention transformer for neural signal translation.
Uses FAVOR+ mechanism for O(n) attention complexity.

Key characteristics:
- Linear attention with random feature approximation
- O(T) complexity instead of O(T²)
- Suitable for long sequences
- ~50-100M parameters

Reference:
    Choromanski et al. "Rethinking Attention with Performers" (2020)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_kernel_transformation(
    data: torch.Tensor,
    is_query: bool,
    projection_matrix: torch.Tensor,
    numerical_stabilizer: float = 1e-6,
) -> torch.Tensor:
    """Apply FAVOR+ softmax kernel transformation.

    Args:
        data: Input tensor [B, H, N, D]
        is_query: Whether this is query (True) or key (False)
        projection_matrix: Random projection matrix [D, M]
        numerical_stabilizer: Small constant for numerical stability

    Returns:
        Transformed features [B, H, N, M]
    """
    # data: [B, H, N, D]
    # projection_matrix: [D, M]

    # Project to random features
    data_proj = torch.matmul(data, projection_matrix)  # [B, H, N, M]

    # Apply exponential approximation
    # For softmax attention, we use: exp(qk/sqrt(d)) ≈ φ(q)^T φ(k)
    # where φ(x) = exp(x - ||x||²/2) / sqrt(M)

    # Compute norm
    data_normalizer = data.norm(dim=-1, keepdim=True) ** 2 / 2  # [B, H, N, 1]

    # Ratio for normalization
    ratio = 1.0 / math.sqrt(projection_matrix.shape[1])

    # Apply transformation
    data_prime = ratio * torch.exp(
        data_proj - data_normalizer + numerical_stabilizer
    )

    return data_prime


class FAVORPlusAttention(nn.Module):
    """FAVOR+ linear attention mechanism.

    Approximates softmax attention with linear complexity using
    random feature maps.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_features: Number of random features (default: embed_dim // 2)
        dropout: Dropout probability
        ortho_features: Use orthogonal random features
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_features: Optional[int] = None,
        dropout: float = 0.0,
        ortho_features: bool = True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features or self.head_dim

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Random projection matrix (orthogonal for better approximation)
        self.ortho_features = ortho_features
        self.register_buffer('projection_matrix', self._create_projection())

    def _create_projection(self) -> torch.Tensor:
        """Create random projection matrix."""
        if self.ortho_features:
            # Orthogonal random features
            # Create multiple orthogonal blocks if needed
            num_blocks = self.num_features // self.head_dim + 1
            blocks = []
            for _ in range(num_blocks):
                q, _ = torch.linalg.qr(torch.randn(self.head_dim, self.head_dim))
                blocks.append(q)
            proj = torch.cat(blocks, dim=1)[:, :self.num_features]
        else:
            # IID Gaussian
            proj = torch.randn(self.head_dim, self.num_features) / math.sqrt(self.head_dim)

        return proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FAVOR+ attention.

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

        # Apply kernel transformation
        q_prime = softmax_kernel_transformation(
            q, is_query=True, projection_matrix=self.projection_matrix
        )  # [B, H, N, M]
        k_prime = softmax_kernel_transformation(
            k, is_query=False, projection_matrix=self.projection_matrix
        )  # [B, H, N, M]

        # Linear attention: O(NM) instead of O(N²)
        # Attention = Q' @ (K'^T @ V) instead of softmax(Q @ K^T) @ V
        kv = torch.einsum('bhnd,bhnm->bhdm', v, k_prime)  # [B, H, head_dim, M]
        qkv_out = torch.einsum('bhnm,bhdm->bhnd', q_prime, kv)  # [B, H, N, head_dim]

        # Normalize
        k_sum = k_prime.sum(dim=2)  # [B, H, M]
        normalizer = torch.einsum('bhnm,bhm->bhn', q_prime, k_sum) + 1e-6  # [B, H, N]
        qkv_out = qkv_out / normalizer.unsqueeze(-1)

        # Reshape and project
        x = qkv_out.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class PerformerBlock(nn.Module):
    """Performer transformer block.

    Uses FAVOR+ linear attention instead of standard softmax attention.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_features: Random features for attention
        ffn_ratio: FFN expansion ratio
        dropout: Dropout probability
        condition_dim: Conditioning dimension
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_features: Optional[int] = None,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        condition_dim: int = 0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = FAVORPlusAttention(embed_dim, num_heads, num_features, dropout)
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
        """Forward pass."""
        # Attention
        x = x + self.attn(self.norm1(x))

        # FFN
        x = x + self.ffn(self.norm2(x))

        # Conditioning
        if self.gamma is not None and condition is not None:
            gamma = self.gamma(condition).unsqueeze(1)
            beta = self.beta(condition).unsqueeze(1)
            x = gamma * x + beta

        return x


class Performer1D(nn.Module):
    """Performer for neural signal translation.

    Linear-complexity transformer suitable for long neural recordings.

    Args:
        in_channels: Input channels (default: 32)
        out_channels: Output channels (default: 32)
        patch_size: Patch size (default: 16)
        embed_dim: Embedding dimension (default: 256)
        num_layers: Number of blocks (default: 6)
        num_heads: Attention heads (default: 8)
        num_features: Random features (default: embed_dim)
        ffn_ratio: FFN expansion ratio (default: 4)
        num_classes: Odor classes (default: 7)
        condition_dim: Conditioning dimension (default: 64)
        use_conditioning: Use conditioning (default: True)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_features: Optional[int] = None,
        ffn_ratio: int = 4,
        num_classes: int = 7,
        condition_dim: int = 64,
        use_conditioning: bool = True,
        dropout: float = 0.1,
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
        self.patch_embed = nn.Conv1d(in_channels, embed_dim, patch_size, stride=patch_size)

        # Position embedding
        self.max_seq_len = 512
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Performer blocks
        self.blocks = nn.ModuleList([
            PerformerBlock(embed_dim, num_heads, num_features, ffn_ratio, dropout, cond_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output
        self.unpatchify = nn.ConvTranspose1d(
            embed_dim, out_channels, patch_size, stride=patch_size
        )

        # Residual
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

        nn.init.zeros_(self.unpatchify.weight)
        nn.init.zeros_(self.unpatchify.bias)

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
        if self.embed is not None and odor_ids is not None:
            condition = self.embed(odor_ids)
        else:
            condition = None

        # Patch embedding
        x = self.patch_embed(ob)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        N = x.shape[1]

        # Position embedding
        x = x + self.pos_embed[:, :N]

        # Performer blocks
        for block in self.blocks:
            x = block(x, condition)

        x = self.norm(x)

        # Unpatchify
        x = x.transpose(1, 2)  # [B, D, N]
        delta = self.unpatchify(x)

        # Handle size mismatch
        if delta.shape[-1] != T:
            delta = F.interpolate(delta, size=T, mode='linear', align_corners=False)

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
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'patch_embed': sum(p.numel() for p in self.patch_embed.parameters()),
            'pos_embed': self.pos_embed.numel(),
            'blocks': sum(p.numel() for block in self.blocks for p in block.parameters()),
            'unpatchify': sum(p.numel() for p in self.unpatchify.parameters()),
        }

    def get_layer_names(self) -> List[str]:
        """Get layer names."""
        names = ['patch_embed', 'pos_embed']
        names.extend([f'block_{i}' for i in range(self.num_layers)])
        names.extend(['norm', 'unpatchify'])
        return names


def create_performer(
    variant: str = "standard",
    **kwargs,
) -> nn.Module:
    """Factory function for Performer variants.

    Args:
        variant: "standard" or "fast" (fewer features)
        **kwargs: Model arguments

    Returns:
        Performer model
    """
    if variant == "standard":
        return Performer1D(**kwargs)
    elif variant == "fast":
        # Use fewer random features for speed
        kwargs.setdefault('num_features', kwargs.get('embed_dim', 256) // 4)
        return Performer1D(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Available: standard, fast")
