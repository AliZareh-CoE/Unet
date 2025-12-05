"""
Track A Objective Function
==========================

Objective function for conditioning experiments that maximizes R².
"""

from __future__ import annotations

import gc
import json
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    optuna = None
    TrialPruned = Exception

# Import conditioners
from .cpc_embeddings import create_cpc_conditioner
from .vqvae_codes import create_vqvae_conditioner
from .freq_disentangled import create_freq_disentangled_conditioner
from .cycle_latent import create_cycle_latent_conditioner
from .vib_conditioning import create_vib_conditioner
from .hp_spaces import get_hp_space

# Import from main codebase
import sys
sys.path.insert(0, str(__file__).rsplit("/experiments", 1)[0])

from models import CondUNet1D, build_wavelet_loss, pearson_batch
from data import prepare_data, create_dataloaders, crop_to_target_torch, SAMPLING_RATE_HZ
from experiments.common.metrics import compute_metrics, explained_variance


def suggest_hyperparameters(
    trial: "optuna.Trial",
    hp_space: Dict[str, Any],
) -> Dict[str, Any]:
    """Suggest hyperparameters from search space.

    Args:
        trial: Optuna trial
        hp_space: Search space dictionary

    Returns:
        Configuration dictionary
    """
    config = {}

    for hp_name, hp_config in hp_space.items():
        hp_type = hp_config["type"]

        if hp_type == "fixed":
            config[hp_name] = hp_config["value"]
        elif hp_type == "categorical":
            config[hp_name] = trial.suggest_categorical(hp_name, hp_config["choices"])
        elif hp_type == "int":
            config[hp_name] = trial.suggest_int(
                hp_name, hp_config["low"], hp_config["high"]
            )
        elif hp_type == "float":
            log_scale = hp_config.get("log", False)
            config[hp_name] = trial.suggest_float(
                hp_name, hp_config["low"], hp_config["high"], log=log_scale
            )

    return config


def build_conditioner(
    config: Dict[str, Any],
    in_channels: int = 32,
    emb_dim: int = 128,
    n_odors: int = 7,
) -> Optional[nn.Module]:
    """Build conditioner based on configuration.

    Args:
        config: Configuration with conditioning_type
        in_channels: Input channels
        emb_dim: Embedding dimension
        n_odors: Number of odor classes

    Returns:
        Conditioner module or None for baseline
    """
    cond_type = config.get("conditioning_type", "baseline")

    if cond_type == "baseline":
        return None  # Use default odor embedding

    elif cond_type == "cpc":
        return create_cpc_conditioner(config, in_channels, emb_dim, n_odors)

    elif cond_type == "vqvae":
        return create_vqvae_conditioner(config, in_channels, emb_dim, n_odors)

    elif cond_type == "freq_disentangled":
        return create_freq_disentangled_conditioner(
            config, in_channels, emb_dim, n_odors, SAMPLING_RATE_HZ
        )

    elif cond_type == "cycle_latent":
        return create_cycle_latent_conditioner(
            config, in_channels, in_channels, emb_dim, n_odors
        )

    elif cond_type == "vib":
        return create_vib_conditioner(config, in_channels, emb_dim, n_odors)

    else:
        raise ValueError(f"Unknown conditioning type: {cond_type}")


class ConditionedModel(nn.Module):
    """Wrapper that combines custom conditioner with CondUNet1D.

    Args:
        model: Base CondUNet1D model
        conditioner: Custom conditioning module
    """

    def __init__(
        self,
        model: CondUNet1D,
        conditioner: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.conditioner = conditioner

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: torch.Tensor,
        pcx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with conditioning.

        Args:
            ob: OB signal [B, C, T]
            odor_ids: Odor indices [B]
            pcx: PCx signal for cycle-latent (optional)

        Returns:
            (prediction, aux_loss): Model output and auxiliary loss
        """
        if self.conditioner is None:
            # Baseline: use model's built-in embedding
            return self.model(ob, odor_ids), torch.tensor(0.0, device=ob.device)

        # Get conditioning embedding from custom conditioner
        # Handle cycle-latent specially
        if hasattr(self.conditioner, 'set_pcx') and pcx is not None:
            self.conditioner.set_pcx(pcx)

        emb, aux_loss = self.conditioner(ob, odor_ids)

        # Inject embedding into model (bypass built-in embedding)
        # We'll use the embedding directly in forward
        return self._forward_with_embedding(ob, emb), aux_loss

    def _forward_with_embedding(
        self,
        ob: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with custom embedding.

        This requires modifying how the model uses conditioning.
        For simplicity, we inject the embedding where the model expects it.
        """
        # The model's forward expects odor_ids, but we want to use our embedding
        # We can achieve this by temporarily replacing the embed layer

        # Save original embed
        original_embed = self.model.embed

        # Create a dummy embed that returns our embedding
        class DummyEmbed(nn.Module):
            def __init__(self, emb):
                super().__init__()
                self._emb = emb

            def forward(self, x):
                return self._emb

        self.model.embed = DummyEmbed(emb)

        # Forward pass (pass dummy odor_ids since embed is bypassed)
        output = self.model(ob, torch.zeros(ob.shape[0], dtype=torch.long, device=ob.device))

        # Restore original embed
        self.model.embed = original_embed

        return output


def track_a_objective(
    trial: "optuna.Trial",
    approach: str,
    data_loaders,
    device: torch.device,
    num_epochs: int = 80,
    early_stop_patience: int = 10,
    trial_seed: int = 42,
) -> float:
    """Track A objective: Maximize R² via conditioning.

    Args:
        trial: Optuna trial
        approach: Conditioning approach name
        data_loaders: Dict[str, DataLoader] with 'train', 'val', 'test' keys
        device: Target device
        num_epochs: Maximum epochs
        early_stop_patience: Early stopping patience
        trial_seed: Random seed

    Returns:
        Best validation R²
    """
    # Handle both dict and tuple formats
    if isinstance(data_loaders, dict):
        train_loader = data_loaders["train"]
        val_loader = data_loaders["val"]
    else:
        train_loader, val_loader = data_loaders

    # Get search space and suggest hyperparameters
    hp_space = get_hp_space(approach)
    config = suggest_hyperparameters(trial, hp_space)

    # Set seed
    torch.manual_seed(trial_seed)

    # Build model
    model = CondUNet1D(
        in_channels=config.get("in_channels", 32),
        out_channels=config.get("out_channels", 32),
        base=config.get("base_channels", 64),
        n_odors=7,
        emb_dim=128,
        dropout=config.get("dropout", 0.0),
        use_attention=True,
        attention_type="cross_freq_v2",
        norm_type="instance",
        cond_mode="cross_attn_gated",
        use_spectral_shift=False,
        n_downsample=config.get("n_downsample", 4),
        conv_type="modern",
        use_se=True,
    ).to(device)

    # Build conditioner
    conditioner = build_conditioner(config, in_channels=32, emb_dim=128, n_odors=7)
    if conditioner is not None:
        conditioner = conditioner.to(device)

    # Create conditioned model
    conditioned_model = ConditionedModel(model, conditioner)

    # Reverse model (for bidirectional training)
    reverse_model = CondUNet1D(
        in_channels=config.get("in_channels", 32),
        out_channels=config.get("out_channels", 32),
        base=config.get("base_channels", 64),
        n_odors=7,
        emb_dim=128,
        dropout=config.get("dropout", 0.0),
        use_attention=True,
        attention_type="cross_freq_v2",
        norm_type="instance",
        cond_mode="cross_attn_gated",
        use_spectral_shift=False,
        n_downsample=config.get("n_downsample", 4),
        conv_type="modern",
        use_se=True,
    ).to(device)

    # Optimizer
    params = list(conditioned_model.parameters()) + list(reverse_model.parameters())
    optimizer = AdamW(
        params,
        lr=config.get("learning_rate", 1e-3),
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    # Loss functions
    wavelet_loss_fn = build_wavelet_loss(
        wavelet="morlet",
        omega0=5.0,
        use_complex_morlet=False,
    ).to(device)

    # Training loop
    best_r2 = -float("inf")
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        # Training
        conditioned_model.train()
        reverse_model.train()

        train_loss = 0.0
        n_batches = 0

        for ob, pcx, odor in train_loader:
            ob = ob.to(device)
            pcx = pcx.to(device)
            odor = odor.to(device)

            optimizer.zero_grad()

            # Forward: OB -> PCx
            pred, aux_loss = conditioned_model(ob, odor, pcx)
            pred_c = crop_to_target_torch(pred)
            pcx_c = crop_to_target_torch(pcx)

            loss_l1 = F.l1_loss(pred_c, pcx_c)
            loss_wavelet = wavelet_loss_fn(pred_c, pcx_c)

            # Reverse: PCx -> OB
            pred_rev = reverse_model(pcx, odor)
            pred_rev_c = crop_to_target_torch(pred_rev)
            ob_c = crop_to_target_torch(ob)

            loss_l1_rev = F.l1_loss(pred_rev_c, ob_c)
            loss_wavelet_rev = wavelet_loss_fn(pred_rev_c, ob_c)

            # Total loss
            loss = (
                config.get("weight_l1", 1.0) * (loss_l1 + loss_l1_rev)
                + config.get("weight_wavelet", 1.0) * (loss_wavelet + loss_wavelet_rev)
                + aux_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # Validation
        conditioned_model.eval()
        val_r2s = []

        with torch.no_grad():
            for ob, pcx, odor in val_loader:
                ob = ob.to(device)
                pcx = pcx.to(device)
                odor = odor.to(device)

                pred, _ = conditioned_model(ob, odor)
                pred_c = crop_to_target_torch(pred)
                pcx_c = crop_to_target_torch(pcx)

                r2 = explained_variance(pred_c, pcx_c)
                val_r2s.append(r2.item())

        val_r2 = sum(val_r2s) / len(val_r2s)

        # Report to Optuna
        trial.report(val_r2, epoch)

        # Pruning
        if trial.should_prune():
            raise TrialPruned()

        # Track best
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            break

    # Record trial info
    elapsed = time.time() - start_time
    trial.set_user_attr("best_r2", best_r2)
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("training_time_seconds", elapsed)
    trial.set_user_attr("approach", approach)
    trial.set_user_attr("full_config", json.dumps(config, default=str))

    # Cleanup
    del conditioned_model, reverse_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return best_r2
