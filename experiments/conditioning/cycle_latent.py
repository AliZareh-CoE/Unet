"""
Cycle-Consistent Latent Discovery
=================================

Bidirectional latent encoders with cycle consistency loss to discover
shared representation between OB and PCx.

Key insight: The optimal conditioning should capture information that
is useful for BOTH directions:
- OB -> PCx translation
- PCx -> OB translation

By enforcing that latents extracted from OB and PCx are mutually
consistent, we discover the "shared neural state" that both regions
encode.

Search parameters:
- latent_dim: Dimension of shared latent space
- cycle_loss_weight: Weight for cycle consistency loss
- encoder_depth: Number of encoder layers
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentEncoder(nn.Module):
    """Encoder network for extracting latent representation.

    Args:
        in_channels: Number of input channels
        latent_dim: Output latent dimension
        hidden_dim: Hidden layer dimension
        n_layers: Number of conv layers
    """

    def __init__(
        self,
        in_channels: int = 32,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()

        layers = []
        current_channels = in_channels

        for i in range(n_layers):
            out_channels = hidden_dim if i < n_layers - 1 else hidden_dim
            stride = 4 if i == 0 else 2
            layers.extend([
                nn.Conv1d(
                    current_channels,
                    out_channels,
                    kernel_size=7,
                    stride=stride,
                    padding=3,
                ),
                nn.GELU(),
                nn.BatchNorm1d(out_channels),
            ])
            current_channels = out_channels

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent.

        Args:
            x: Input signal [B, C, T]

        Returns:
            Latent [B, latent_dim]
        """
        h = self.encoder(x)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


class LatentPredictor(nn.Module):
    """MLP to predict one latent from another.

    Used for cycle consistency: z_ob -> z_pcx_pred and z_pcx -> z_ob_pred

    Args:
        latent_dim: Latent dimension
        hidden_dim: Hidden layer dimension
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CycleLatentConditioner(nn.Module):
    """Cycle-consistent latent conditioning module.

    Learns latent encoders for both OB and PCx, with a consistency
    loss ensuring the latents are mutually predictable.

    Args:
        in_channels_ob: Number of OB input channels
        in_channels_pcx: Number of PCx input channels
        latent_dim: Shared latent dimension
        emb_dim: Final embedding dimension
        hidden_dim: Hidden dimension for encoders
        encoder_depth: Number of encoder layers
        cycle_loss_weight: Weight for cycle consistency loss
        n_odors: Number of odor classes
        use_odor: Whether to include odor information
    """

    def __init__(
        self,
        in_channels_ob: int = 32,
        in_channels_pcx: int = 32,
        latent_dim: int = 64,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        encoder_depth: int = 3,
        cycle_loss_weight: float = 0.5,
        n_odors: int = 7,
        use_odor: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.cycle_loss_weight = cycle_loss_weight
        self.use_odor = use_odor

        # OB encoder
        self.ob_encoder = LatentEncoder(
            in_channels=in_channels_ob,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=encoder_depth,
        )

        # PCx encoder (used during training for cycle consistency)
        self.pcx_encoder = LatentEncoder(
            in_channels=in_channels_pcx,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=encoder_depth,
        )

        # Cross-prediction heads for cycle consistency
        self.ob_to_pcx_predictor = LatentPredictor(latent_dim, hidden_dim)
        self.pcx_to_ob_predictor = LatentPredictor(latent_dim, hidden_dim)

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

    def compute_cycle_loss(
        self,
        z_ob: torch.Tensor,
        z_pcx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bidirectional cycle consistency loss.

        Args:
            z_ob: OB latent [B, latent_dim]
            z_pcx: PCx latent [B, latent_dim]

        Returns:
            Cycle consistency loss
        """
        # OB -> PCx prediction
        z_pcx_pred = self.ob_to_pcx_predictor(z_ob)
        loss_ob_to_pcx = F.mse_loss(z_pcx_pred, z_pcx.detach())

        # PCx -> OB prediction
        z_ob_pred = self.pcx_to_ob_predictor(z_pcx)
        loss_pcx_to_ob = F.mse_loss(z_ob_pred, z_ob.detach())

        # Total cycle loss
        return (loss_ob_to_pcx + loss_pcx_to_ob) / 2

    def forward(
        self,
        ob: torch.Tensor,
        pcx: Optional[torch.Tensor] = None,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cycle-consistent conditioning.

        Args:
            ob: OB signal [B, C, T]
            pcx: PCx signal [B, C, T] (optional, for cycle loss)
            odor_ids: Odor class indices [B]

        Returns:
            (embedding, cycle_loss): [B, emb_dim], scalar
        """
        # Encode OB
        z_ob = self.ob_encoder(ob)

        # Compute cycle loss if PCx provided (training)
        if pcx is not None and self.training:
            z_pcx = self.pcx_encoder(pcx)
            cycle_loss = self.compute_cycle_loss(z_ob, z_pcx)
        else:
            cycle_loss = torch.tensor(0.0, device=ob.device)

        # Build conditioning embedding
        if self.use_odor and odor_ids is not None:
            odor_emb = self.odor_embed(odor_ids)
            combined = torch.cat([z_ob, odor_emb], dim=-1)
        else:
            combined = z_ob

        emb = self.proj(combined)

        return emb, cycle_loss * self.cycle_loss_weight

    def encode_ob(self, ob: torch.Tensor) -> torch.Tensor:
        """Encode OB signal only (for inference)."""
        return self.ob_encoder(ob)

    def encode_pcx(self, pcx: torch.Tensor) -> torch.Tensor:
        """Encode PCx signal only (for analysis)."""
        return self.pcx_encoder(pcx)


class CycleLatentConditionerWrapper(nn.Module):
    """Wrapper that handles PCx signal passing during training.

    During training, we need to pass both OB and PCx to compute cycle loss.
    This wrapper stores PCx and provides it when forward is called.
    """

    def __init__(self, conditioner: CycleLatentConditioner):
        super().__init__()
        self.conditioner = conditioner
        self._pcx = None

    def set_pcx(self, pcx: torch.Tensor) -> None:
        """Store PCx signal for cycle loss computation."""
        self._pcx = pcx

    def clear_pcx(self) -> None:
        """Clear stored PCx signal."""
        self._pcx = None

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conditioner(ob, self._pcx, odor_ids)


def create_cycle_latent_conditioner(
    config: Dict,
    in_channels_ob: int = 32,
    in_channels_pcx: int = 32,
    emb_dim: int = 128,
    n_odors: int = 7,
) -> CycleLatentConditionerWrapper:
    """Factory function to create cycle-consistent latent conditioner.

    Args:
        config: Configuration dictionary
        in_channels_ob: OB input channels
        in_channels_pcx: PCx input channels
        emb_dim: Embedding dimension
        n_odors: Number of odor classes

    Returns:
        Wrapped CycleLatentConditioner
    """
    conditioner = CycleLatentConditioner(
        in_channels_ob=in_channels_ob,
        in_channels_pcx=in_channels_pcx,
        latent_dim=config.get("latent_dim", 64),
        emb_dim=emb_dim,
        hidden_dim=config.get("hidden_dim", 128),
        encoder_depth=config.get("encoder_depth", 3),
        cycle_loss_weight=config.get("cycle_loss_weight", 0.5),
        n_odors=n_odors,
        use_odor=config.get("use_odor", True),
    )

    return CycleLatentConditionerWrapper(conditioner)
