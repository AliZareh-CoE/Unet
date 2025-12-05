"""
Track B Objective Function
==========================

Objective function for SpectralShift experiments.
Maximizes: PSD improvement - 10 * max(0, R² degradation)

The U-Net model is FROZEN - only SpectralShift parameters are trained.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
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

# Import spectral modules
from .phase_preserving import create_phase_preserving_module
from .adaptive_gated import create_adaptive_gated_module
from .wavelet_correction import create_wavelet_correction_module
from .spectral_loss import create_spectral_loss_module, MultiScaleSpectralLoss
from .iterative_refinement import create_iterative_refinement_module
from .hp_spaces import get_hp_space

# Import from main codebase
import sys
sys.path.insert(0, str(__file__).rsplit("/experiments", 1)[0])

from models import CondUNet1D, AdaptiveSpectralShift
from data import prepare_data, create_dataloaders, crop_to_target_torch, SAMPLING_RATE_HZ
from experiments.common.metrics import compute_metrics, compute_psd_error_db, explained_variance


def suggest_hyperparameters(
    trial: "optuna.Trial",
    hp_space: Dict[str, Any],
) -> Dict[str, Any]:
    """Suggest hyperparameters from search space."""
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


def load_frozen_stage1(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    """Load and freeze Stage 1 model from checkpoint.

    Args:
        checkpoint_path: Path to Stage 1 checkpoint
        device: Target device

    Returns:
        (model, reverse_model): Frozen forward and reverse models
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create models
    config = checkpoint.get("config", {})

    model = CondUNet1D(
        in_channels=config.get("in_channels", 32),
        out_channels=config.get("out_channels", 32),
        base=config.get("base_channels", 64),
        n_odors=7,
        emb_dim=128,
        dropout=0.0,  # No dropout at inference
        use_attention=config.get("use_attention", True),
        attention_type=config.get("attention_type", "cross_freq_v2"),
        norm_type=config.get("norm_type", "instance"),
        cond_mode=config.get("cond_mode", "cross_attn_gated"),
        use_spectral_shift=False,
        n_downsample=config.get("n_downsample", 4),
        conv_type=config.get("conv_type", "modern"),
        use_se=config.get("use_se", True),
    ).to(device)

    reverse_model = CondUNet1D(
        in_channels=config.get("in_channels", 32),
        out_channels=config.get("out_channels", 32),
        base=config.get("base_channels", 64),
        n_odors=7,
        emb_dim=128,
        dropout=0.0,
        use_attention=config.get("use_attention", True),
        attention_type=config.get("attention_type", "cross_freq_v2"),
        norm_type=config.get("norm_type", "instance"),
        cond_mode=config.get("cond_mode", "cross_attn_gated"),
        use_spectral_shift=False,
        n_downsample=config.get("n_downsample", 4),
        conv_type=config.get("conv_type", "modern"),
        use_se=config.get("use_se", True),
    ).to(device)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    if "reverse_model_state_dict" in checkpoint:
        reverse_model.load_state_dict(checkpoint["reverse_model_state_dict"])

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    for param in reverse_model.parameters():
        param.requires_grad = False

    model.eval()
    reverse_model.eval()

    return model, reverse_model


def build_spectral_module(
    config: Dict[str, Any],
    n_channels: int = 32,
    sample_rate: float = 1000.0,
) -> Tuple[nn.Module, Optional[nn.Module]]:
    """Build spectral correction module based on configuration.

    Args:
        config: Configuration with spectral_method
        n_channels: Number of channels
        sample_rate: Sampling rate

    Returns:
        (spectral_module, loss_fn): Module and optional loss function
    """
    method = config.get("spectral_method", "baseline")

    if method == "baseline":
        # Use existing AdaptiveSpectralShift
        module = AdaptiveSpectralShift(
            n_channels=n_channels,
            n_odors=7,
            emb_dim=128,
            hidden_dim=config.get("spectral_shift_hidden_dim", 128),
            sample_rate=sample_rate,
            max_shift_db=config.get("spectral_shift_max_db", 12.0),
            per_channel=config.get("spectral_shift_per_channel", False),
        )
        return module, None

    elif method == "phase_preserving":
        return create_phase_preserving_module(config, n_channels, sample_rate), None

    elif method == "adaptive_gated":
        return create_adaptive_gated_module(config, n_channels, sample_rate), None

    elif method == "wavelet":
        return create_wavelet_correction_module(config, n_channels, sample_rate), None

    elif method == "spectral_loss":
        return create_spectral_loss_module(config, sample_rate)

    elif method == "iterative":
        return create_iterative_refinement_module(config, n_channels, sample_rate), None

    else:
        raise ValueError(f"Unknown spectral method: {method}")


def compute_stage1_metrics(
    model: nn.Module,
    val_loader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute Stage 1 metrics (R², PSD error) without any correction.

    Args:
        model: Stage 1 model
        val_loader: Validation data loader
        device: Target device

    Returns:
        Dictionary with metrics
    """
    model.eval()

    all_r2s = []
    all_psd_errs = []

    with torch.no_grad():
        for ob, pcx, odor in val_loader:
            ob = ob.to(device)
            pcx = pcx.to(device)
            odor = odor.to(device)

            pred = model(ob, odor)
            pred_c = crop_to_target_torch(pred)
            pcx_c = crop_to_target_torch(pcx)

            r2 = explained_variance(pred_c, pcx_c)
            all_r2s.append(r2.item())

            psd_err = compute_psd_error_db(pred_c, pcx_c, sample_rate=SAMPLING_RATE_HZ)
            all_psd_errs.append(psd_err["overall"])

    return {
        "r2": sum(all_r2s) / len(all_r2s),
        "psd_err_db": sum(all_psd_errs) / len(all_psd_errs),
    }


def track_b_objective(
    trial: "optuna.Trial",
    approach: str,
    data_loaders: Tuple,
    device: torch.device,
    stage1_checkpoint: str,
    stage1_metrics: Optional[Dict[str, float]] = None,
    num_epochs: int = 30,
    early_stop_patience: int = 8,
    trial_seed: int = 42,
) -> float:
    """Track B objective: Maximize PSD improvement without R² degradation.

    Score = PSD_improvement_db - 10 * max(0, R²_degradation)

    Args:
        trial: Optuna trial
        approach: Spectral correction approach name
        data_loaders: (train_loader, val_loader) tuple
        device: Target device
        stage1_checkpoint: Path to frozen Stage 1 checkpoint
        stage1_metrics: Pre-computed Stage 1 metrics (optional)
        num_epochs: Maximum epochs
        early_stop_patience: Early stopping patience
        trial_seed: Random seed

    Returns:
        Composite score
    """
    # Handle both dict and tuple formats for data_loaders
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

    # Load frozen Stage 1 model
    model, reverse_model = load_frozen_stage1(stage1_checkpoint, device)

    # Get Stage 1 baseline metrics if not provided
    if stage1_metrics is None:
        stage1_metrics = compute_stage1_metrics(model, val_loader, device)

    stage1_r2 = stage1_metrics["r2"]
    stage1_psd_err = stage1_metrics["psd_err_db"]

    # Build spectral module
    spectral_fwd, spectral_loss_fn = build_spectral_module(
        config, n_channels=32, sample_rate=SAMPLING_RATE_HZ
    )
    spectral_fwd = spectral_fwd.to(device)

    spectral_rev, _ = build_spectral_module(
        config, n_channels=32, sample_rate=SAMPLING_RATE_HZ
    )
    spectral_rev = spectral_rev.to(device)

    # Optimizer (only spectral parameters)
    params = list(spectral_fwd.parameters()) + list(spectral_rev.parameters())
    optimizer = AdamW(
        params,
        lr=config.get("spectral_lr", 1e-3),
        weight_decay=1e-4,
    )

    # Learning rate scheduler
    lr_decay = config.get("spectral_lr_decay", 0.95)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    # Training loop
    best_score = -float("inf")
    best_r2 = stage1_r2
    best_psd_err = stage1_psd_err
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        # Training (only spectral module)
        spectral_fwd.train()
        spectral_rev.train()

        train_loss = 0.0
        n_batches = 0

        for ob, pcx, odor in train_loader:
            ob = ob.to(device)
            pcx = pcx.to(device)
            odor = odor.to(device)

            optimizer.zero_grad()

            # Forward pass with frozen model
            with torch.no_grad():
                pred_raw = model(ob, odor)
                pred_rev_raw = reverse_model(pcx, odor)

            # Apply spectral correction
            pred = spectral_fwd(pred_raw, pcx, odor)
            pred_rev = spectral_rev(pred_rev_raw, ob, odor)

            pred_c = crop_to_target_torch(pred)
            pcx_c = crop_to_target_torch(pcx)
            pred_rev_c = crop_to_target_torch(pred_rev)
            ob_c = crop_to_target_torch(ob)

            # Loss: PSD matching + some R² preservation
            # PSD loss
            psd_loss_fwd = compute_psd_error_db(pred_c, pcx_c)["overall"]
            psd_loss_rev = compute_psd_error_db(pred_rev_c, ob_c)["overall"]

            # R² preservation loss (penalize deviation from Stage 1)
            r2_fwd = explained_variance(pred_c, pcx_c)
            r2_rev = explained_variance(pred_rev_c, ob_c)

            # Total loss
            loss = (psd_loss_fwd + psd_loss_rev) / 2

            # Add spectral loss if using that approach
            if spectral_loss_fn is not None:
                spec_loss, _ = spectral_loss_fn(pred_c, pcx_c)
                loss = loss + config.get("spectral_loss_weight", 1.0) * spec_loss

            # Convert to tensor if needed
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=device, requires_grad=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            train_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            n_batches += 1

        train_loss /= n_batches
        scheduler.step()

        # Validation
        spectral_fwd.eval()
        spectral_rev.eval()

        val_r2s = []
        val_psd_errs = []

        with torch.no_grad():
            for ob, pcx, odor in val_loader:
                ob = ob.to(device)
                pcx = pcx.to(device)
                odor = odor.to(device)

                # Forward with frozen model + spectral
                pred_raw = model(ob, odor)
                pred = spectral_fwd(pred_raw, pcx, odor)
                pred_c = crop_to_target_torch(pred)
                pcx_c = crop_to_target_torch(pcx)

                r2 = explained_variance(pred_c, pcx_c)
                val_r2s.append(r2.item())

                psd_err = compute_psd_error_db(pred_c, pcx_c)["overall"]
                val_psd_errs.append(psd_err)

        val_r2 = sum(val_r2s) / len(val_r2s)
        val_psd_err = sum(val_psd_errs) / len(val_psd_errs)

        # Compute composite score
        psd_improvement = stage1_psd_err - val_psd_err  # Positive = better
        r2_degradation = stage1_r2 - val_r2  # Positive = worse
        score = psd_improvement - 10 * max(0, r2_degradation)

        # Report to Optuna
        trial.report(score, epoch)

        # Pruning
        if trial.should_prune():
            raise TrialPruned()

        # Track best
        if score > best_score:
            best_score = score
            best_r2 = val_r2
            best_psd_err = val_psd_err
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            break

    # Record trial info
    elapsed = time.time() - start_time
    trial.set_user_attr("best_score", best_score)
    trial.set_user_attr("best_r2", best_r2)
    trial.set_user_attr("best_psd_err_db", best_psd_err)
    trial.set_user_attr("psd_improvement_db", stage1_psd_err - best_psd_err)
    trial.set_user_attr("r2_change", best_r2 - stage1_r2)
    trial.set_user_attr("stage1_r2", stage1_r2)
    trial.set_user_attr("stage1_psd_err_db", stage1_psd_err)
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("training_time_seconds", elapsed)
    trial.set_user_attr("approach", approach)
    trial.set_user_attr("full_config", json.dumps(config, default=str))

    # Cleanup
    del spectral_fwd, spectral_rev, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return best_score
