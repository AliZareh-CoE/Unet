"""
Vector-Quantized VAE (VQ-VAE) Discrete Codes
============================================

Learn a discrete codebook from OB signals using Vector Quantization.
The discrete bottleneck forces the model to discover meaningful
neural states beyond continuous representations.

VQ-VAE process:
1. Encoder: OB signal -> continuous latent z
2. Quantization: z -> nearest codebook entry z_q
3. Straight-through estimator for gradients
4. Use z_q as conditioning for the main model

The discrete codes may correspond to interpretable neural states
(e.g., respiratory phases, odor-evoked patterns).

Search parameters:
- codebook_size: Number of discrete codes (K)
- code_dim: Dimension of each code
- commitment_cost: VQ commitment loss weight
- vq_loss_weight: Overall VQ auxiliary loss weight
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA codebook updates.

    Maps continuous vectors to nearest codebook entries with
    straight-through gradient estimator.

    Args:
        codebook_size: Number of codebook entries (K)
        code_dim: Dimension of each code
        commitment_cost: Weight for commitment loss
        ema_decay: Decay for exponential moving average updates
        use_ema: Whether to use EMA for codebook updates
    """

    def __init__(
        self,
        codebook_size: int = 512,
        code_dim: int = 64,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        use_ema: bool = True,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.use_ema = use_ema

        # Codebook
        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.codebook.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

        if use_ema:
            # EMA cluster counts and sums
            self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
            self.register_buffer("ema_w", self.codebook.weight.data.clone())

    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input vectors.

        Args:
            z: Input vectors [B, code_dim] or [B, T, code_dim]

        Returns:
            (z_q, vq_loss, indices): Quantized vectors, loss, codebook indices
        """
        # Handle both 2D and 3D inputs
        input_shape = z.shape
        if len(input_shape) == 3:
            B, T, D = z.shape
            z_flat = z.reshape(-1, D)  # [B*T, code_dim]
        else:
            z_flat = z

        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z . e
        distances = (
            (z_flat ** 2).sum(dim=-1, keepdim=True)
            + (self.codebook.weight ** 2).sum(dim=-1)
            - 2 * torch.matmul(z_flat, self.codebook.weight.t())
        )

        # Find nearest codebook entry
        indices = distances.argmin(dim=-1)  # [B*T] or [B]

        # Quantize
        z_q = self.codebook(indices)

        # Compute losses
        if self.training:
            if self.use_ema:
                # EMA update
                self._ema_update(z_flat, indices)
                # Only commitment loss (codebook updated via EMA)
                commitment_loss = F.mse_loss(z_flat, z_q.detach())
                vq_loss = self.commitment_cost * commitment_loss
            else:
                # Standard VQ loss
                codebook_loss = F.mse_loss(z_q, z_flat.detach())
                commitment_loss = F.mse_loss(z_flat, z_q.detach())
                vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        else:
            vq_loss = torch.tensor(0.0, device=z.device)

        # Straight-through estimator
        z_q = z_flat + (z_q - z_flat).detach()

        # Reshape back
        if len(input_shape) == 3:
            z_q = z_q.reshape(B, T, D)
            indices = indices.reshape(B, T)

        return z_q, vq_loss, indices

    def _ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor) -> None:
        """Update codebook using exponential moving average."""
        with torch.no_grad():
            # Count assignments
            encodings = F.one_hot(indices, self.codebook_size).float()  # [N, K]
            cluster_size = encodings.sum(dim=0)  # [K]

            # Sum of assigned vectors
            cluster_sum = encodings.t() @ z_flat  # [K, code_dim]

            # EMA update
            self.ema_cluster_size.data.mul_(self.ema_decay).add_(
                cluster_size, alpha=1 - self.ema_decay
            )
            self.ema_w.data.mul_(self.ema_decay).add_(
                cluster_sum, alpha=1 - self.ema_decay
            )

            # Update codebook
            n = self.ema_cluster_size.sum()
            cluster_size_stable = (
                (self.ema_cluster_size + 1e-5)
                / (n + self.codebook_size * 1e-5) * n
            )
            self.codebook.weight.data.copy_(
                self.ema_w / cluster_size_stable.unsqueeze(1)
            )

    def get_codebook_usage(self) -> Dict[str, float]:
        """Get codebook utilization statistics."""
        with torch.no_grad():
            counts = self.ema_cluster_size if self.use_ema else None
            if counts is not None:
                used = (counts > 0).sum().item()
                return {
                    "codebook_usage": used / self.codebook_size,
                    "codes_used": used,
                    "codes_total": self.codebook_size,
                }
            return {}


class VQVAEConditioner(nn.Module):
    """VQ-VAE based conditioning module for CondUNet1D.

    Encodes OB signal to discrete codes via vector quantization,
    then uses these codes as conditioning.

    Args:
        in_channels: Number of input channels
        hidden_dim: Encoder hidden dimension
        code_dim: Codebook code dimension
        emb_dim: Final embedding dimension (for FiLM)
        codebook_size: Number of codebook entries
        commitment_cost: VQ commitment loss weight
        n_odors: Number of odor classes
        use_odor_residual: Add odor embedding as residual
    """

    def __init__(
        self,
        in_channels: int = 32,
        hidden_dim: int = 256,
        code_dim: int = 64,
        emb_dim: int = 128,
        codebook_size: int = 512,
        commitment_cost: float = 0.25,
        n_odors: int = 7,
        use_odor_residual: bool = True,
    ):
        super().__init__()

        self.use_odor_residual = use_odor_residual

        # Encoder: OB signal -> continuous latent
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=15, stride=5, padding=7),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=3, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, code_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )

        # Global pooling to get single code per sample
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Vector quantizer
        self.vq = VectorQuantizer(
            codebook_size=codebook_size,
            code_dim=code_dim,
            commitment_cost=commitment_cost,
        )

        # Project to embedding dimension
        self.proj = nn.Linear(code_dim, emb_dim)

        # Odor embedding (optional residual)
        if use_odor_residual:
            self.odor_embed = nn.Embedding(n_odors, emb_dim)
            self.gate = nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.Sigmoid(),
            )
        else:
            self.odor_embed = None

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Compute VQ-VAE conditioning embedding.

        Args:
            ob: OB signal [B, C, T]
            odor_ids: Odor class indices [B]

        Returns:
            (embedding, vq_loss, info): [B, emb_dim], scalar, dict
        """
        # Encode
        z = self.encoder(ob)  # [B, code_dim, T']
        z = self.pool(z).squeeze(-1)  # [B, code_dim]

        # Quantize
        z_q, vq_loss, indices = self.vq(z)

        # Project to embedding dimension
        emb = self.proj(z_q)  # [B, emb_dim]

        # Add odor residual if enabled
        if self.use_odor_residual and odor_ids is not None:
            odor_emb = self.odor_embed(odor_ids)  # [B, emb_dim]
            # Gated combination
            combined = torch.cat([emb, odor_emb], dim=-1)
            gate = self.gate(combined)
            emb = gate * emb + (1 - gate) * odor_emb

        info = {
            "indices": indices,
            "codebook_usage": self.vq.get_codebook_usage(),
        }

        return emb, vq_loss, info

    def get_code_for_odor(self, odor_ids: torch.Tensor) -> torch.Tensor:
        """Get most common codes for each odor (for analysis)."""
        # This would be populated during training
        return None


class VQVAEConditionerWrapper(nn.Module):
    """Wrapper that returns (embedding, loss) tuple for compatibility."""

    def __init__(self, conditioner: VQVAEConditioner, vq_loss_weight: float = 0.1):
        super().__init__()
        self.conditioner = conditioner
        self.vq_loss_weight = vq_loss_weight

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb, vq_loss, _ = self.conditioner(ob, odor_ids)
        return emb, vq_loss * self.vq_loss_weight


def create_vqvae_conditioner(
    config: Dict,
    in_channels: int = 32,
    emb_dim: int = 128,
    n_odors: int = 7,
) -> VQVAEConditionerWrapper:
    """Factory function to create VQ-VAE conditioner from config.

    Args:
        config: Configuration dictionary
        in_channels: Number of input channels
        emb_dim: Embedding dimension
        n_odors: Number of odor classes

    Returns:
        Configured VQVAEConditioner wrapped for compatibility
    """
    conditioner = VQVAEConditioner(
        in_channels=in_channels,
        hidden_dim=config.get("vq_hidden_dim", 256),
        code_dim=config.get("vq_code_dim", 64),
        emb_dim=emb_dim,
        codebook_size=config.get("vq_codebook_size", 512),
        commitment_cost=config.get("vq_commitment_cost", 0.25),
        n_odors=n_odors,
        use_odor_residual=config.get("vq_use_odor_residual", True),
    )

    return VQVAEConditionerWrapper(
        conditioner,
        vq_loss_weight=config.get("vq_loss_weight", 0.1),
    )
