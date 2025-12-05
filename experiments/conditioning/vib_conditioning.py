"""
Variational Information Bottleneck (VIB) Conditioning
=====================================================

Compress OB signal to minimal sufficient statistics using a variational
information bottleneck. The β parameter controls the compression level.

VIB principle:
- Learn encoder q(z|OB) that compresses OB into latent z
- Loss = reconstruction + β * KL(q(z|OB) || p(z))
- Higher β = more compression = only task-relevant information survives

This discovers what information from OB is actually needed for
accurate PCx prediction, filtering out noise and irrelevant details.

Search parameters:
- vib_latent_dim: Latent dimension
- vib_beta: KL divergence weight (compression strength)
- vib_encoder_type: "conv" or "transformer"
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Convolutional encoder for VIB.

    Args:
        in_channels: Number of input channels
        hidden_dim: Hidden dimension
        out_dim: Output dimension (latent_dim * 2 for mean and logvar)
    """

    def __init__(
        self,
        in_channels: int = 32,
        hidden_dim: int = 256,
        out_dim: int = 128,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=15, stride=5, padding=7),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=3, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


class TransformerEncoder(nn.Module):
    """Transformer encoder for VIB.

    Args:
        in_channels: Number of input channels
        hidden_dim: Hidden/embedding dimension
        out_dim: Output dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
    """

    def __init__(
        self,
        in_channels: int = 32,
        hidden_dim: int = 256,
        out_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        max_len: int = 512,
    ):
        super().__init__()

        # Patch embedding (downsample time)
        self.patch_embed = nn.Conv1d(
            in_channels, hidden_dim,
            kernel_size=16, stride=16, padding=0
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x).transpose(1, 2)  # [B, T', hidden_dim]
        T = x.shape[1]

        # Add positional encoding
        x = x + self.pos_embed[:, :T, :]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Transformer
        x = self.transformer(x)

        # Use CLS token output
        cls_out = x[:, 0, :]

        return self.fc(cls_out)


class VIBConditioner(nn.Module):
    """Variational Information Bottleneck conditioning module.

    Compresses OB signal through a stochastic bottleneck, regularized
    by KL divergence to encourage compression.

    Args:
        in_channels: Number of input channels
        latent_dim: Dimension of latent z
        emb_dim: Final embedding dimension
        hidden_dim: Encoder hidden dimension
        beta: KL divergence weight (compression strength)
        encoder_type: "conv" or "transformer"
        n_odors: Number of odor classes
        use_odor: Whether to include odor information
    """

    def __init__(
        self,
        in_channels: int = 32,
        latent_dim: int = 64,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        beta: float = 0.01,
        encoder_type: str = "conv",
        n_odors: int = 7,
        use_odor: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.use_odor = use_odor

        # Encoder (outputs mean and logvar)
        encoder_out_dim = latent_dim * 2  # Mean and logvar

        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                out_dim=encoder_out_dim,
            )
        else:
            self.encoder = ConvEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                out_dim=encoder_out_dim,
            )

        # Projection to embedding dimension
        proj_in_dim = latent_dim
        if use_odor:
            self.odor_embed = nn.Embedding(n_odors, latent_dim)
            proj_in_dim += latent_dim

        self.proj = nn.Sequential(
            nn.Linear(proj_in_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input signal [B, C, T]

        Returns:
            (mu, logvar): Mean and log-variance of q(z|x)
        """
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick for sampling.

        Args:
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]

        Returns:
            Sampled z [B, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Use mean at inference
            return mu

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence KL(q(z|x) || p(z)).

        Assumes p(z) = N(0, I).

        Args:
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]

        Returns:
            KL divergence (scalar)
        """
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute VIB conditioning.

        Args:
            ob: OB signal [B, C, T]
            odor_ids: Odor class indices [B]

        Returns:
            (embedding, kl_loss): [B, emb_dim], scalar
        """
        # Encode to distribution parameters
        mu, logvar = self.encode(ob)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Compute KL divergence
        kl_loss = self.kl_divergence(mu, logvar)

        # Build conditioning embedding
        if self.use_odor and odor_ids is not None:
            odor_emb = self.odor_embed(odor_ids)
            combined = torch.cat([z, odor_emb], dim=-1)
        else:
            combined = z

        emb = self.proj(combined)

        return emb, self.beta * kl_loss

    def get_latent_stats(self, ob: torch.Tensor) -> Dict[str, float]:
        """Get statistics about the latent distribution."""
        with torch.no_grad():
            mu, logvar = self.encode(ob)
            return {
                "mu_mean": mu.mean().item(),
                "mu_std": mu.std().item(),
                "logvar_mean": logvar.mean().item(),
                "logvar_std": logvar.std().item(),
                "kl": self.kl_divergence(mu, logvar).item(),
            }


def create_vib_conditioner(
    config: Dict,
    in_channels: int = 32,
    emb_dim: int = 128,
    n_odors: int = 7,
) -> VIBConditioner:
    """Factory function to create VIB conditioner from config.

    Args:
        config: Configuration dictionary
        in_channels: Number of input channels
        emb_dim: Embedding dimension
        n_odors: Number of odor classes

    Returns:
        Configured VIBConditioner
    """
    return VIBConditioner(
        in_channels=in_channels,
        latent_dim=config.get("vib_latent_dim", 64),
        emb_dim=emb_dim,
        hidden_dim=config.get("vib_hidden_dim", 256),
        beta=config.get("vib_beta", 0.01),
        encoder_type=config.get("vib_encoder_type", "conv"),
        n_odors=n_odors,
        use_odor=config.get("use_odor", True),
    )
