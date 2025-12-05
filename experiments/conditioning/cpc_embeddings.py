"""
Contrastive Predictive Coding (CPC) Embeddings
==============================================

Learn temporal context embeddings from OB signals using Contrastive
Predictive Coding, replacing or augmenting the static odor embedding.

CPC learns representations by predicting future timesteps:
1. Encoder: OB signal -> local features z_t
2. Context network: z_1...z_t -> context c_t
3. Prediction heads: c_t -> z_{t+k} for k steps ahead
4. Contrastive loss: distinguish true future from negatives

The learned context embedding captures temporal patterns that are
predictive of future signal content - potentially more informative
than discrete odor labels.

Search parameters:
- hidden_dim: Encoder hidden dimension
- context_dim: Context/embedding dimension
- n_steps_ahead: Number of future steps to predict
- cpc_loss_weight: Weight for CPC auxiliary loss
- use_odor_concat: Whether to concatenate with odor embedding
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPCEncoder(nn.Module):
    """Contrastive Predictive Coding encoder for OB signal conditioning.

    Learns to predict future timesteps from current context, producing
    a conditioning embedding that captures temporal patterns.

    Args:
        in_channels: Number of input channels (OB channels)
        hidden_dim: Hidden dimension for encoder
        context_dim: Output context/embedding dimension
        n_steps_ahead: Number of future steps to predict (K)
        n_negatives: Number of negative samples for contrastive loss
        sample_rate: Sampling rate for stride calculation
    """

    def __init__(
        self,
        in_channels: int = 32,
        hidden_dim: int = 256,
        context_dim: int = 128,
        n_steps_ahead: int = 12,
        n_negatives: int = 10,
        sample_rate: float = 1000.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.n_steps_ahead = n_steps_ahead
        self.n_negatives = n_negatives

        # Encoder: Extract local features z_t from raw signal
        # Strided convolutions to downsample time dimension
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=10, stride=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # Context network: Aggregate z_t into context c_t
        self.context_gru = nn.GRU(
            hidden_dim,
            context_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Prediction heads: W_k for k-step ahead prediction
        # c_t -> prediction of z_{t+k}
        self.predictors = nn.ModuleList([
            nn.Linear(context_dim, hidden_dim)
            for _ in range(n_steps_ahead)
        ])

        # Temperature for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode(self, ob: torch.Tensor) -> torch.Tensor:
        """Encode OB signal to local features.

        Args:
            ob: OB signal [B, C, T]

        Returns:
            Local features z [B, T', hidden_dim]
        """
        z = self.encoder(ob)  # [B, hidden_dim, T']
        z = z.transpose(1, 2)  # [B, T', hidden_dim]
        return z

    def get_context(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute context from local features.

        Args:
            z: Local features [B, T', hidden_dim]

        Returns:
            (context_sequence, final_context): [B, T', context_dim], [B, context_dim]
        """
        c, _ = self.context_gru(z)  # [B, T', context_dim]
        final_c = c[:, -1, :]  # [B, context_dim] - final context
        return c, final_c

    def compute_cpc_loss(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss.

        For each context c_t, predict z_{t+k} and distinguish from negatives.

        Args:
            z: Local features [B, T', hidden_dim]
            c: Context sequence [B, T', context_dim]

        Returns:
            CPC loss (scalar)
        """
        B, T_enc, _ = z.shape
        device = z.device

        total_loss = torch.tensor(0.0, device=device)
        n_predictions = 0

        for k in range(1, self.n_steps_ahead + 1):
            # Get predictions and targets
            # Predict z_{t+k} from c_t
            max_t = T_enc - k
            if max_t <= 0:
                continue

            # Context at times 0..max_t-1
            c_t = c[:, :max_t, :]  # [B, max_t, context_dim]

            # Predicted z_{t+k}
            pred_z = self.predictors[k-1](c_t)  # [B, max_t, hidden_dim]

            # True z_{t+k}
            true_z = z[:, k:k+max_t, :]  # [B, max_t, hidden_dim]

            # Negative samples: shuffle batch dimension
            # For each (c_t, z_{t+k}) pair, negatives are z from other batches
            batch_indices = torch.arange(B, device=device)

            # Compute similarity scores
            # Positive: pred_z[b, t] . true_z[b, t]
            # Negative: pred_z[b, t] . true_z[b', t] for b' != b

            # Reshape for batch computation
            pred_flat = pred_z.reshape(B * max_t, -1)  # [B*max_t, hidden_dim]
            true_flat = true_z.reshape(B * max_t, -1)  # [B*max_t, hidden_dim]

            # Compute all pairwise similarities within each time step
            # For efficiency, compute positive scores directly
            pos_scores = (pred_flat * true_flat).sum(dim=-1) / self.temperature  # [B*max_t]

            # For negatives, we sample from same time step but different batch
            # Simple approach: use all other batch elements as negatives
            neg_scores_list = []
            for t in range(max_t):
                pred_t = pred_z[:, t, :]  # [B, hidden_dim]
                true_t = true_z[:, t, :]  # [B, hidden_dim]

                # All pairwise similarities at time t
                sim_matrix = torch.mm(pred_t, true_t.t()) / self.temperature  # [B, B]

                neg_scores_list.append(sim_matrix)

            # Stack and compute InfoNCE loss
            # pos_scores: diagonal elements
            # neg_scores: off-diagonal elements
            for t, sim_matrix in enumerate(neg_scores_list):
                # InfoNCE: -log(exp(pos) / sum(exp(all)))
                # = -pos + log(sum(exp(all)))
                log_sum_exp = torch.logsumexp(sim_matrix, dim=1)  # [B]
                pos = sim_matrix.diag()  # [B]
                loss_t = -pos + log_sum_exp
                total_loss = total_loss + loss_t.mean()
                n_predictions += 1

        if n_predictions > 0:
            total_loss = total_loss / n_predictions

        return total_loss

    def forward(
        self,
        ob: torch.Tensor,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract CPC embedding from OB signal.

        Args:
            ob: OB signal [B, C, T]
            return_loss: Whether to compute and return CPC loss

        Returns:
            (embedding, cpc_loss): [B, context_dim], scalar
        """
        # Encode to local features
        z = self.encode(ob)

        # Get context
        c_seq, final_context = self.get_context(z)

        # Compute CPC loss if training
        if return_loss and self.training:
            cpc_loss = self.compute_cpc_loss(z, c_seq)
        else:
            cpc_loss = torch.tensor(0.0, device=ob.device)

        return final_context, cpc_loss


class CPCConditioner(nn.Module):
    """CPC-based conditioning module for CondUNet1D.

    Wraps CPCEncoder and optionally concatenates with odor embedding.

    Args:
        in_channels: Number of input channels
        hidden_dim: CPC encoder hidden dimension
        context_dim: Context dimension (must be compatible with FiLM)
        emb_dim: Final embedding dimension (for FiLM)
        n_odors: Number of odor classes
        use_odor_concat: Whether to concatenate with odor embedding
        cpc_loss_weight: Weight for CPC auxiliary loss
        **cpc_kwargs: Additional CPC encoder arguments
    """

    def __init__(
        self,
        in_channels: int = 32,
        hidden_dim: int = 256,
        context_dim: int = 128,
        emb_dim: int = 128,
        n_odors: int = 7,
        use_odor_concat: bool = True,
        cpc_loss_weight: float = 0.1,
        **cpc_kwargs,
    ):
        super().__init__()

        self.use_odor_concat = use_odor_concat
        self.cpc_loss_weight = cpc_loss_weight

        # CPC encoder
        self.cpc = CPCEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            **cpc_kwargs,
        )

        # Odor embedding (if using concat)
        if use_odor_concat:
            self.odor_embed = nn.Embedding(n_odors, context_dim)
            # Project concatenated embedding to final dimension
            self.proj = nn.Linear(context_dim * 2, emb_dim)
        else:
            self.odor_embed = None
            self.proj = nn.Linear(context_dim, emb_dim)

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute conditioning embedding.

        Args:
            ob: OB signal [B, C, T]
            odor_ids: Odor class indices [B]

        Returns:
            (embedding, cpc_loss): [B, emb_dim], scalar
        """
        # Get CPC context
        cpc_emb, cpc_loss = self.cpc(ob, return_loss=self.training)

        # Combine with odor if enabled
        if self.use_odor_concat and odor_ids is not None:
            odor_emb = self.odor_embed(odor_ids)  # [B, context_dim]
            combined = torch.cat([cpc_emb, odor_emb], dim=-1)  # [B, context_dim*2]
        else:
            combined = cpc_emb

        # Project to final embedding dimension
        emb = self.proj(combined)

        return emb, cpc_loss * self.cpc_loss_weight


def create_cpc_conditioner(
    config: Dict,
    in_channels: int = 32,
    emb_dim: int = 128,
    n_odors: int = 7,
) -> CPCConditioner:
    """Factory function to create CPC conditioner from config.

    Args:
        config: Configuration dictionary
        in_channels: Number of input channels
        emb_dim: Embedding dimension
        n_odors: Number of odor classes

    Returns:
        Configured CPCConditioner
    """
    return CPCConditioner(
        in_channels=in_channels,
        hidden_dim=config.get("cpc_hidden_dim", 256),
        context_dim=config.get("cpc_context_dim", 128),
        emb_dim=emb_dim,
        n_odors=n_odors,
        use_odor_concat=config.get("cpc_use_odor_concat", True),
        cpc_loss_weight=config.get("cpc_loss_weight", 0.1),
        n_steps_ahead=config.get("cpc_n_steps_ahead", 12),
        n_negatives=config.get("cpc_n_negatives", 10),
    )
