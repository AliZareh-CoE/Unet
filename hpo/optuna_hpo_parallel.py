#!/usr/bin/env python3
"""
Parallel HPO Worker for Single-GPU Trials (Nature Methods Edition)
===================================================================

This script runs independent Optuna trials on a single GPU with:
- Full reproducibility (comprehensive config/seed saving)
- Cycle consistency loss (matches train.py exactly)
- Checkpoint saving for top trials
- Dry-run mode for pre-flight testing
- Export functions for publication

Usage:
    # Pre-flight test (ALWAYS RUN FIRST!)
    python optuna_hpo_parallel.py --dry-run --gpu-id 0

    # Single GPU worker
    CUDA_VISIBLE_DEVICES=0 python optuna_hpo_parallel.py --gpu-id 0 --n-trials 32

    # Export results after completion
    python optuna_hpo_parallel.py --export --study-name my_study

Or use the launcher script:
    bash run_parallel_hpo.sh --dry-run
    bash run_parallel_hpo.sh --save-checkpoints --export
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.exceptions import TrialPruned

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Import from existing modules
from models import (
    NeuroGate,
    build_wavelet_loss,
    pearson_batch,
    explained_variance_torch,
)
from data import (
    prepare_data,
    create_dataloaders,
    crop_to_target_torch,
    SAMPLING_RATE_HZ,
)

# Import probabilistic losses
try:
    from efficient_hpo_strategy import (
        CombinedProbabilisticLoss,
        RayleighEnvelopeLoss,
        VonMisesPhaseLoss,
        ChiSquaredPowerLoss,
        GumbelPeakLoss,
    )
    PROB_LOSSES_AVAILABLE = True
except ImportError:
    PROB_LOSSES_AVAILABLE = False
    print("Warning: Probabilistic losses not available")


# =============================================================================
# Configuration
# =============================================================================

STUDY_DIR = Path("artifacts/optuna_parallel")
CHECKPOINT_DIR = STUDY_DIR / "checkpoints"
RESULTS_DIR = STUDY_DIR / "results"

# Create directories
for d in [STUDY_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Fixed parameters (match train.py EXACTLY)
FIXED_CONFIG = {
    "in_channels": 32,
    "out_channels": 32,
    "sampling_rate": SAMPLING_RATE_HZ,
    "base_channels": 64,
    "conv_type": "modern",
    "conv_dilations": (1, 4, 16, 32),
    "use_wavelet_loss": True,
    "wavelet_family": "morlet",
    "use_complex_morlet": False,
    "use_two_stage": False,
    "use_spectral_shift": False,

    # Match train.py architecture
    "use_attention": True,
    "attention_type": "cross_freq_v2",
    "use_se": True,
    "norm_type": "instance",
    "cond_mode": "cross_attn_gated",
    "use_bidirectional": True,

    # CRITICAL: Cycle consistency (was missing before!)
    "cycle_lambda": 1.0,
}

# Focused search space
HP_SPACE_FOCUSED = {
    # CRITICAL importance
    "learning_rate": {"type": "float", "low": 5e-5, "high": 5e-3, "log": True},

    # HIGH importance
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},

    # MEDIUM importance
    "weight_l1": {"type": "float", "low": 0.5, "high": 10.0, "log": True},
    "weight_wavelet": {"type": "float", "low": 0.5, "high": 10.0, "log": True},
    "dropout": {"type": "float", "low": 0.0, "high": 0.3},
    "n_downsample": {"type": "categorical", "choices": [3, 4, 5]},
    "conv_kernel_size": {"type": "categorical", "choices": [5, 7, 9]},

    # LOW importance - FIXED
    "beta1": {"type": "fixed", "value": 0.9},
    "beta2": {"type": "fixed", "value": 0.999},
    "grad_clip": {"type": "fixed", "value": 5.0},  # Match train.py (was 1.0!)
    "wavelet_omega0": {"type": "fixed", "value": 5.0},

    # PROBABILISTIC LOSSES
    "use_prob_losses": {"type": "categorical", "choices": [True, False]},
    "weight_prob_total": {
        "type": "float", "low": 0.1, "high": 2.0, "log": True,
        "condition": "use_prob_losses"
    },
}


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Use deterministic mode for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_trial_seed(base_seed: int, trial_number: int) -> int:
    """Generate deterministic seed for specific trial."""
    seed_str = f"{base_seed}_trial_{trial_number}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**31)


# =============================================================================
# Hyperparameter Suggestion
# =============================================================================

def suggest_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest hyperparameters from focused search space."""
    config = FIXED_CONFIG.copy()

    for hp_name, hp_config in HP_SPACE_FOCUSED.items():
        hp_type = hp_config["type"]

        # Check condition
        if "condition" in hp_config:
            cond_param = hp_config["condition"]
            if not config.get(cond_param, False):
                continue

        if hp_type == "fixed":
            config[hp_name] = hp_config["value"]
        elif hp_type == "categorical":
            config[hp_name] = trial.suggest_categorical(hp_name, hp_config["choices"])
        elif hp_type == "int":
            config[hp_name] = trial.suggest_int(hp_name, hp_config["low"], hp_config["high"])
        elif hp_type == "float":
            log_scale = hp_config.get("log", False)
            config[hp_name] = trial.suggest_float(
                hp_name, hp_config["low"], hp_config["high"], log=log_scale
            )

    return config


def config_to_json_safe(config: Dict[str, Any]) -> str:
    """Convert config to JSON, handling non-serializable types."""
    safe_config = {}
    for k, v in config.items():
        if isinstance(v, (tuple, list)):
            safe_config[k] = list(v)
        elif isinstance(v, (int, float, str, bool, type(None))):
            safe_config[k] = v
        else:
            safe_config[k] = str(v)
    return json.dumps(safe_config)


# =============================================================================
# Model Building
# =============================================================================

def build_model_with_checkpointing(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Build model with gradient checkpointing for memory efficiency."""
    model = NeuroGate(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        base_channels=config["base_channels"],
        n_downsample=config.get("n_downsample", 4),
        conv_type=config.get("conv_type", "modern"),
        conv_kernel_size=config.get("conv_kernel_size", 7),
        conv_dilations=config.get("conv_dilations", (1, 4, 16, 32)),
        use_attention=config.get("use_attention", True),
        attention_type=config.get("attention_type", "cross_freq_v2"),
        use_se=config.get("use_se", True),
        norm_type=config.get("norm_type", "instance"),
        cond_mode=config.get("cond_mode", "cross_attn_gated"),
        dropout=config.get("dropout", 0.0),
        n_odors=7,
    )

    model = model.to(device)

    # Enable gradient checkpointing if available
    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
    else:
        for module in model.modules():
            if hasattr(module, 'use_checkpoint'):
                module.use_checkpoint = True

    return model


# =============================================================================
# Training (Aligned with train.py)
# =============================================================================

def train_epoch_single_gpu(
    model: nn.Module,
    reverse_model: Optional[nn.Module],
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict[str, Any],
    wavelet_loss: Optional[nn.Module] = None,
    prob_loss: Optional[nn.Module] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Train one epoch on single GPU.

    NOW MATCHES train.py EXACTLY:
    - Includes cycle consistency loss
    - Same gradient clipping (5.0)
    - Same loss weighting
    """
    model.train()
    if reverse_model is not None:
        reverse_model.train()

    total_loss = 0.0
    loss_components = {"l1_fwd": 0.0, "wavelet_fwd": 0.0, "l1_rev": 0.0, "wavelet_rev": 0.0,
                       "cycle_ob": 0.0, "cycle_pcx": 0.0, "prob": 0.0}

    for batch in loader:
        # Handle both 3-tuple (legacy) and 4-tuple (with session_ids) formats
        if len(batch) == 4:
            ob, pcx, odor, _ = batch  # Discard session_ids in HPO
        else:
            ob, pcx, odor = batch
        ob = ob.to(device, non_blocking=True)
        pcx = pcx.to(device, non_blocking=True)
        odor = odor.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Forward: OB → PCx
            pred_raw = model(ob, odor)
            pred_c = crop_to_target_torch(pred_raw)
            pcx_c = crop_to_target_torch(pcx)
            ob_c = crop_to_target_torch(ob)

            # L1 loss (forward)
            l1_fwd = config["weight_l1"] * F.l1_loss(pred_c, pcx_c)
            loss = l1_fwd
            loss_components["l1_fwd"] += l1_fwd.item()

            # Wavelet loss (forward)
            if wavelet_loss is not None:
                wav_fwd = config["weight_wavelet"] * wavelet_loss(pred_c, pcx_c)
                loss = loss + wav_fwd
                loss_components["wavelet_fwd"] += wav_fwd.item()

            # Probabilistic loss
            if prob_loss is not None and config.get("use_prob_losses", False):
                prob_loss_val, _ = prob_loss(pred_c, pcx_c)
                prob_weighted = config.get("weight_prob_total", 0.5) * prob_loss_val
                loss = loss + prob_weighted
                loss_components["prob"] += prob_weighted.item()

            # Bidirectional training with CYCLE CONSISTENCY (critical for correlation!)
            if reverse_model is not None:
                # Reverse: PCx → OB
                pred_rev = reverse_model(pcx, odor)
                pred_rev_c = crop_to_target_torch(pred_rev)

                # L1 loss (reverse)
                l1_rev = config["weight_l1"] * F.l1_loss(pred_rev_c, ob_c)
                loss = loss + l1_rev
                loss_components["l1_rev"] += l1_rev.item()

                # Wavelet loss (reverse)
                if wavelet_loss is not None:
                    wav_rev = config["weight_wavelet"] * wavelet_loss(pred_rev_c, ob_c)
                    loss = loss + wav_rev
                    loss_components["wavelet_rev"] += wav_rev.item()

                # CYCLE CONSISTENCY (this was MISSING before!)
                # OB → PCx → OB
                cycle_ob = reverse_model(pred_raw, odor)
                cycle_ob_c = crop_to_target_torch(cycle_ob)
                cycle_loss_ob = config.get("cycle_lambda", 1.0) * F.l1_loss(cycle_ob_c, ob_c)
                loss = loss + cycle_loss_ob
                loss_components["cycle_ob"] += cycle_loss_ob.item()

                # PCx → OB → PCx
                cycle_pcx = model(pred_rev, odor)
                cycle_pcx_c = crop_to_target_torch(cycle_pcx)
                cycle_loss_pcx = config.get("cycle_lambda", 1.0) * F.l1_loss(cycle_pcx_c, pcx_c)
                loss = loss + cycle_loss_pcx
                loss_components["cycle_pcx"] += cycle_loss_pcx.item()

        # Backward with scaler
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Match train.py: grad_clip = 5.0 (was 1.0!)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 5.0))
            if reverse_model is not None:
                torch.nn.utils.clip_grad_norm_(reverse_model.parameters(), config.get("grad_clip", 5.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 5.0))
            if reverse_model is not None:
                torch.nn.utils.clip_grad_norm_(reverse_model.parameters(), config.get("grad_clip", 5.0))
            optimizer.step()

        total_loss += loss.item()

    n_batches = len(loader)
    avg_components = {k: v / n_batches for k, v in loss_components.items()}

    return total_loss / n_batches, avg_components


@torch.inference_mode()
def evaluate_single_gpu(
    model: nn.Module,
    reverse_model: Optional[nn.Module],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate model on single GPU."""
    model.eval()
    if reverse_model is not None:
        reverse_model.eval()

    corr_list, r2_list = [], []
    corr_list_rev, r2_list_rev = [], []
    mae_list, mae_list_rev = [], []

    for batch in loader:
        # Handle both 3-tuple (legacy) and 4-tuple (with session_ids) formats
        if len(batch) == 4:
            ob, pcx, odor, _ = batch  # Discard session_ids in HPO
        else:
            ob, pcx, odor = batch
        ob = ob.to(device)
        pcx = pcx.to(device)
        odor = odor.to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            pred = model(ob, odor)
            pred_c = crop_to_target_torch(pred).float()
            pcx_c = crop_to_target_torch(pcx).float()
            ob_c = crop_to_target_torch(ob).float()

            corr_list.append(pearson_batch(pred_c, pcx_c).item())
            r2_list.append(explained_variance_torch(pred_c, pcx_c).item())
            mae_list.append(F.l1_loss(pred_c, pcx_c).item())

            if reverse_model is not None:
                pred_rev = reverse_model(pcx, odor)
                pred_rev_c = crop_to_target_torch(pred_rev).float()
                corr_list_rev.append(pearson_batch(pred_rev_c, ob_c).item())
                r2_list_rev.append(explained_variance_torch(pred_rev_c, ob_c).item())
                mae_list_rev.append(F.l1_loss(pred_rev_c, ob_c).item())

    results = {
        "corr": float(np.mean(corr_list)),
        "r2": float(np.mean(r2_list)),
        "mae": float(np.mean(mae_list)),
    }

    if corr_list_rev:
        results["corr_rev"] = float(np.mean(corr_list_rev))
        results["r2_rev"] = float(np.mean(r2_list_rev))
        results["mae_rev"] = float(np.mean(mae_list_rev))
        # Combined score (average of forward and reverse)
        results["score"] = (results["corr"] + results["r2"] +
                          results["corr_rev"] + results["r2_rev"]) / 4
    else:
        results["score"] = (results["corr"] + results["r2"]) / 2

    return results


# =============================================================================
# Checkpoint Management
# =============================================================================

def should_save_checkpoint(
    trial: optuna.Trial,
    study: optuna.Study,
    current_score: float,
    top_percent: float = 10.0,
) -> bool:
    """
    Determine if this trial should have its checkpoint saved.

    Saves checkpoint if trial is in top N% of completed trials.
    """
    completed_trials = [t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]

    if len(completed_trials) < 5:
        # Save all early trials
        return True

    scores = sorted([t.value for t in completed_trials], reverse=True)
    threshold_idx = max(1, int(len(scores) * top_percent / 100))
    threshold = scores[min(threshold_idx, len(scores) - 1)]

    return current_score >= threshold


def save_trial_checkpoint(
    trial: optuna.Trial,
    model: nn.Module,
    reverse_model: Optional[nn.Module],
    config: Dict[str, Any],
    trial_seed: int,
    val_metrics: Dict[str, float],
    best_epoch: int,
) -> str:
    """Save model checkpoint for a trial."""
    checkpoint_path = CHECKPOINT_DIR / f"trial_{trial.number}.pt"

    checkpoint = {
        "trial_number": trial.number,
        "model_state_dict": model.state_dict(),
        "config": config,
        "seed": trial_seed,
        "metrics": val_metrics,
        "best_epoch": best_epoch,
        "timestamp": datetime.now().isoformat(),
    }

    if reverse_model is not None:
        checkpoint["reverse_model_state_dict"] = reverse_model.state_dict()

    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


# =============================================================================
# Main Objective Function
# =============================================================================

def objective(
    trial: optuna.Trial,
    data: Dict[str, Any],
    device: torch.device,
    study: optuna.Study,
    num_epochs: int = 80,
    early_stop_patience: int = 10,
    base_seed: int = 42,
    save_checkpoints: bool = True,
) -> float:
    """
    Single-GPU trial objective with comprehensive saving.

    This function now:
    1. Uses cycle consistency (matches train.py)
    2. Saves 15+ user attributes for reproducibility
    3. Saves checkpoints for top 10% trials
    4. Records full training curves
    """
    start_time = time.time()

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    # Set deterministic seed for this trial
    trial_seed = get_trial_seed(base_seed, trial.number)
    seed_everything(trial_seed)

    # Suggest hyperparameters
    config = suggest_hyperparameters(trial)

    # CRITICAL: Save full config immediately for reproducibility
    trial.set_user_attr("full_config", config_to_json_safe(config))
    trial.set_user_attr("trial_seed", trial_seed)
    trial.set_user_attr("start_time", datetime.now().isoformat())

    print(f"\n[Trial {trial.number}] Config: lr={config['learning_rate']:.2e}, "
          f"batch={config['batch_size']}, dropout={config['dropout']:.2f}")

    try:
        # Create dataloaders
        loaders = create_dataloaders(
            data,
            batch_size=config["batch_size"],
            num_workers=4,
            distributed=False,
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]

        # Build models
        model = build_model_with_checkpointing(config, device)
        reverse_model = None
        if config.get("use_bidirectional", True):
            reverse_model = build_model_with_checkpointing(config, device)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        if reverse_model:
            n_params += sum(p.numel() for p in reverse_model.parameters())
        trial.set_user_attr("n_params_million", round(n_params / 1e6, 2))
        print(f"  Total parameters: {n_params / 1e6:.1f}M")

        # Losses
        wavelet_loss = None
        if config.get("use_wavelet_loss", True):
            wavelet_loss = build_wavelet_loss(
                config["wavelet_family"],
                use_complex_morlet=config.get("use_complex_morlet", False),
                omega0=config.get("wavelet_omega0", 5.0),
            ).to(device)

        prob_loss = None
        if config.get("use_prob_losses", False) and PROB_LOSSES_AVAILABLE:
            prob_loss = CombinedProbabilisticLoss(
                use_rayleigh=True,
                use_von_mises=True,
                use_chi_squared=True,
                use_gumbel=True,
            ).to(device)

        # Optimizer
        params = list(model.parameters())
        if reverse_model:
            params += list(reverse_model.parameters())

        optimizer = AdamW(
            params,
            lr=config["learning_rate"],
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
            weight_decay=1e-4,
        )

        # Mixed precision scaler
        scaler = torch.amp.GradScaler('cuda')

        # Training tracking
        best_score = 0.0
        best_epoch = 0
        best_val_metrics = {}
        patience_counter = 0

        # Training curves (for Nature Methods plots)
        loss_history = []
        corr_history = []
        r2_history = []

        # Track GPU memory
        peak_memory_gb = 0.0

        for epoch in range(num_epochs):
            train_loss, loss_components = train_epoch_single_gpu(
                model, reverse_model, train_loader, optimizer,
                device, config, wavelet_loss, prob_loss, scaler
            )

            val_metrics = evaluate_single_gpu(
                model, reverse_model, val_loader, device, config
            )

            score = val_metrics["score"]

            # Track training curves
            loss_history.append(train_loss)
            corr_history.append(val_metrics["corr"])
            r2_history.append(val_metrics["r2"])

            # Track peak memory
            if torch.cuda.is_available():
                current_memory = torch.cuda.max_memory_allocated(device) / 1e9
                peak_memory_gb = max(peak_memory_gb, current_memory)

            # Report to Optuna for ASHA pruning
            trial.report(score, epoch)

            # Check if should prune
            if trial.should_prune():
                print(f"  [Epoch {epoch}] Pruned! Score: {score:.4f}")
                # Save partial results before pruning
                trial.set_user_attr("pruned_at_epoch", epoch)
                trial.set_user_attr("pruned_score", score)
                raise optuna.TrialPruned()

            # Early stopping
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_val_metrics = val_metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"  [Epoch {epoch}] Early stop. Best: {best_score:.4f}")
                    break

            if epoch % 10 == 0:
                print(f"  [Epoch {epoch}] loss={train_loss:.4f}, "
                      f"corr={val_metrics['corr']:.4f}, r2={val_metrics['r2']:.4f}")

        # =================================================================
        # COMPREHENSIVE SAVING (15+ attributes for Nature Methods)
        # =================================================================

        elapsed_time = time.time() - start_time

        # Core metrics
        trial.set_user_attr("best_corr", round(best_val_metrics.get("corr", 0), 4))
        trial.set_user_attr("best_r2", round(best_val_metrics.get("r2", 0), 4))
        trial.set_user_attr("best_corr_rev", round(best_val_metrics.get("corr_rev", 0), 4))
        trial.set_user_attr("best_r2_rev", round(best_val_metrics.get("r2_rev", 0), 4))
        trial.set_user_attr("best_mae", round(best_val_metrics.get("mae", 0), 4))
        trial.set_user_attr("best_mae_rev", round(best_val_metrics.get("mae_rev", 0), 4))

        # Training dynamics
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("epochs_run", epoch + 1)
        trial.set_user_attr("training_time_seconds", round(elapsed_time, 1))
        trial.set_user_attr("final_train_loss", round(loss_history[-1] if loss_history else 0, 4))

        # Resource usage
        trial.set_user_attr("gpu_memory_peak_gb", round(peak_memory_gb, 2))

        # Training curves (JSON for later analysis)
        trial.set_user_attr("loss_history", json.dumps([round(x, 4) for x in loss_history]))
        trial.set_user_attr("corr_history", json.dumps([round(x, 4) for x in corr_history]))
        trial.set_user_attr("r2_history", json.dumps([round(x, 4) for x in r2_history]))

        # Completion timestamp
        trial.set_user_attr("end_time", datetime.now().isoformat())

        # =================================================================
        # CHECKPOINT SAVING (Top 10% trials)
        # =================================================================

        if save_checkpoints and should_save_checkpoint(trial, study, best_score, top_percent=10):
            checkpoint_path = save_trial_checkpoint(
                trial, model, reverse_model, config, trial_seed, best_val_metrics, best_epoch
            )
            trial.set_user_attr("checkpoint_path", checkpoint_path)
            print(f"  [Checkpoint saved: {checkpoint_path}]")

        print(f"[Trial {trial.number}] DONE. Score: {best_score:.4f}, "
              f"Corr: {best_val_metrics.get('corr', 0):.4f}, Time: {elapsed_time:.0f}s")

        return best_score

    except torch.cuda.OutOfMemoryError:
        print(f"[Trial {trial.number}] OOM! Reducing batch size might help.")
        trial.set_user_attr("error", "OOM")
        gc.collect()
        torch.cuda.empty_cache()
        raise optuna.TrialPruned()

    except Exception as e:
        print(f"[Trial {trial.number}] Error: {e}")
        trial.set_user_attr("error", str(e))
        traceback.print_exc()
        raise optuna.TrialPruned()

    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# Export Functions (Nature Methods Publication)
# =============================================================================

def export_trials_csv(study: optuna.Study, output_path: Path) -> None:
    """Export all trials to CSV for analysis."""
    rows = []

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        row = {
            "trial_number": trial.number,
            "score": trial.value,
            "state": str(trial.state),
        }

        # Add hyperparameters
        for k, v in trial.params.items():
            row[f"param_{k}"] = v

        # Add user attributes
        for k, v in trial.user_attrs.items():
            if k not in ["loss_history", "corr_history", "r2_history", "full_config"]:
                row[k] = v

        rows.append(row)

    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Exported {len(rows)} trials to {output_path}")


def export_best_configs(study: optuna.Study, output_path: Path, n: int = 10) -> None:
    """Export top N configs as JSON for reproduction."""
    completed = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:n]

    configs = []
    for trial in top_trials:
        config_json = trial.user_attrs.get("full_config", "{}")
        try:
            config = json.loads(config_json)
        except:
            config = {}

        configs.append({
            "trial_number": trial.number,
            "score": trial.value,
            "corr": trial.user_attrs.get("best_corr", 0),
            "r2": trial.user_attrs.get("best_r2", 0),
            "seed": trial.user_attrs.get("trial_seed", 0),
            "config": config,
            "checkpoint_path": trial.user_attrs.get("checkpoint_path", None),
        })

    with open(output_path, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"Exported top {len(configs)} configs to {output_path}")


def export_training_curves(study: optuna.Study, output_path: Path) -> None:
    """Export training curves for all trials."""
    rows = []

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        loss_hist = json.loads(trial.user_attrs.get("loss_history", "[]"))
        corr_hist = json.loads(trial.user_attrs.get("corr_history", "[]"))
        r2_hist = json.loads(trial.user_attrs.get("r2_history", "[]"))

        for epoch, (loss, corr, r2) in enumerate(zip(loss_hist, corr_hist, r2_hist)):
            rows.append({
                "trial_number": trial.number,
                "epoch": epoch,
                "loss": loss,
                "corr": corr,
                "r2": r2,
            })

    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["trial_number", "epoch", "loss", "corr", "r2"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Exported training curves to {output_path}")


def export_importance(study: optuna.Study, output_path: Path) -> None:
    """Export hyperparameter importance scores."""
    try:
        importance = optuna.importance.get_param_importances(study)
        with open(output_path, "w") as f:
            json.dump(importance, f, indent=2)
        print(f"Exported importance to {output_path}")
    except Exception as e:
        print(f"Could not compute importance: {e}")


def export_study_results(study_name: str, storage: str, output_dir: Path) -> None:
    """Export all results for Nature Methods publication."""
    output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.load_study(study_name=study_name, storage=storage)

    print(f"\nExporting study '{study_name}' ({len(study.trials)} trials)...")

    # Export everything
    export_trials_csv(study, output_dir / "hpo_trials.csv")
    export_best_configs(study, output_dir / "hpo_best_configs.json", n=10)
    export_training_curves(study, output_dir / "hpo_training_curves.csv")
    export_importance(study, output_dir / "hpo_importance.json")

    # Summary
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    summary = {
        "study_name": study_name,
        "total_trials": len(study.trials),
        "completed": len(completed),
        "pruned": len(pruned),
        "best_score": study.best_value if study.best_trial else None,
        "best_trial": study.best_trial.number if study.best_trial else None,
        "best_corr": study.best_trial.user_attrs.get("best_corr") if study.best_trial else None,
        "best_r2": study.best_trial.user_attrs.get("best_r2") if study.best_trial else None,
        "export_time": datetime.now().isoformat(),
    }

    with open(output_dir / "hpo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExport complete! Files in: {output_dir}")
    print(f"  - hpo_trials.csv")
    print(f"  - hpo_best_configs.json")
    print(f"  - hpo_training_curves.csv")
    print(f"  - hpo_importance.json")
    print(f"  - hpo_summary.json")


# =============================================================================
# Dry-Run Mode (Pre-Flight Testing)
# =============================================================================

def run_dry_run(gpu_id: int, seed: int = 42) -> bool:
    """
    Run comprehensive pre-flight test before full HPO sweep.

    Tests:
    1. Data loading
    2. Model building
    3. Forward/backward pass
    4. Checkpoint saving
    5. JSON serialization
    6. User attributes

    Returns True if all tests pass.
    """
    print("\n" + "="*60)
    print("[DRY-RUN] Testing HPO pipeline...")
    print("="*60 + "\n")

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    all_passed = True
    results = {}

    # Test 1: Data loading
    print("[1/6] Testing data loading...")
    try:
        seed_everything(seed)
        data = prepare_data(seed=seed)
        loaders = create_dataloaders(data, batch_size=16, num_workers=2, distributed=False)
        n_train = len(data['train_idx'])
        n_val = len(data['val_idx'])
        print(f"  OK (train={n_train}, val={n_val} samples)")
        results["data_loading"] = {"status": "OK", "train": n_train, "val": n_val}
    except Exception as e:
        print(f"  FAILED: {e}")
        results["data_loading"] = {"status": "FAILED", "error": str(e)}
        all_passed = False
        return False

    # Test 2: Model building
    print("[2/6] Testing model build...")
    try:
        config = FIXED_CONFIG.copy()
        config["learning_rate"] = 1e-4
        config["batch_size"] = 16
        config["weight_l1"] = 1.0
        config["weight_wavelet"] = 1.0
        config["dropout"] = 0.1
        config["n_downsample"] = 4
        config["conv_kernel_size"] = 7

        model = build_model_with_checkpointing(config, device)
        reverse_model = build_model_with_checkpointing(config, device)

        n_params = sum(p.numel() for p in model.parameters())
        n_params += sum(p.numel() for p in reverse_model.parameters())

        gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
        print(f"  OK ({n_params/1e6:.1f}M params, {gpu_mem:.1f}GB GPU memory)")
        results["model_build"] = {"status": "OK", "params_million": round(n_params/1e6, 1), "gpu_gb": round(gpu_mem, 1)}
    except Exception as e:
        print(f"  FAILED: {e}")
        results["model_build"] = {"status": "FAILED", "error": str(e)}
        all_passed = False
        return False

    # Test 3: Forward pass
    print("[3/6] Testing forward pass...")
    try:
        ob, pcx, odor = next(iter(loaders["train"]))
        ob = ob.to(device)
        pcx = pcx.to(device)
        odor = odor.to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            pred = model(ob, odor)
            pred_rev = reverse_model(pcx, odor)

        print(f"  OK (input: {ob.shape}, output: {pred.shape})")
        results["forward_pass"] = {"status": "OK", "input_shape": list(ob.shape), "output_shape": list(pred.shape)}
    except Exception as e:
        print(f"  FAILED: {e}")
        results["forward_pass"] = {"status": "FAILED", "error": str(e)}
        all_passed = False
        return False

    # Test 4: Backward pass + training
    print("[4/6] Testing backward pass (3 epochs)...")
    try:
        wavelet_loss = build_wavelet_loss("morlet", use_complex_morlet=False, omega0=5.0).to(device)

        params = list(model.parameters()) + list(reverse_model.parameters())
        optimizer = AdamW(params, lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(3):
            train_loss, _ = train_epoch_single_gpu(
                model, reverse_model, loaders["train"], optimizer,
                device, config, wavelet_loss, None, scaler
            )
            val_metrics = evaluate_single_gpu(model, reverse_model, loaders["val"], device, config)
            print(f"    Epoch {epoch}: loss={train_loss:.4f}, corr={val_metrics['corr']:.4f}")

        print(f"  OK")
        results["backward_pass"] = {"status": "OK", "final_loss": round(train_loss, 4), "final_corr": round(val_metrics['corr'], 4)}
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        results["backward_pass"] = {"status": "FAILED", "error": str(e)}
        all_passed = False
        return False

    # Test 5: Checkpoint saving
    print("[5/6] Testing checkpoint save...")
    try:
        test_checkpoint_path = STUDY_DIR / "dry_run_checkpoint.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "reverse_model_state_dict": reverse_model.state_dict(),
            "config": config,
            "seed": seed,
        }
        torch.save(checkpoint, test_checkpoint_path)

        # Verify can load
        loaded = torch.load(test_checkpoint_path, weights_only=False)
        assert "model_state_dict" in loaded

        print(f"  OK ({test_checkpoint_path})")
        results["checkpoint_save"] = {"status": "OK", "path": str(test_checkpoint_path)}
    except Exception as e:
        print(f"  FAILED: {e}")
        results["checkpoint_save"] = {"status": "FAILED", "error": str(e)}
        all_passed = False

    # Test 6: JSON serialization
    print("[6/6] Testing JSON serialization...")
    try:
        config_json = config_to_json_safe(config)
        parsed = json.loads(config_json)
        assert "learning_rate" in parsed

        # Test full attribute set
        test_attrs = {
            "full_config": config_json,
            "trial_seed": seed,
            "best_corr": 0.75,
            "best_r2": 0.60,
            "loss_history": json.dumps([1.0, 0.9, 0.8]),
        }

        print(f"  OK ({len(test_attrs)} attributes)")
        results["json_serialize"] = {"status": "OK", "n_attrs": len(test_attrs)}
    except Exception as e:
        print(f"  FAILED: {e}")
        results["json_serialize"] = {"status": "FAILED", "error": str(e)}
        all_passed = False

    # Save dry-run report
    report_path = STUDY_DIR / "dry_run_report.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "gpu_id": gpu_id,
        "seed": seed,
        "all_passed": all_passed,
        "results": results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*60)
    if all_passed:
        print("[DRY-RUN] All systems GO! Ready for full HPO sweep.")
    else:
        print("[DRY-RUN] FAILED - Fix issues before running full sweep!")
    print(f"Report saved: {report_path}")
    print("="*60 + "\n")

    return all_passed


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parallel HPO Worker (Nature Methods Edition)")

    # Basic options
    parser.add_argument("--study-name", type=str, default="olfactory_parallel_hpo")
    parser.add_argument("--storage", type=str,
                       default="sqlite:///artifacts/optuna_parallel/study.db")
    parser.add_argument("--n-trials", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # New options
    parser.add_argument("--dry-run", action="store_true",
                       help="Run pre-flight test (ALWAYS RUN FIRST!)")
    parser.add_argument("--save-checkpoints", action="store_true", default=True,
                       help="Save checkpoints for top 10%% trials")
    parser.add_argument("--no-checkpoints", action="store_true",
                       help="Disable checkpoint saving")
    parser.add_argument("--export", action="store_true",
                       help="Export results and exit (no training)")
    parser.add_argument("--export-dir", type=str, default=None,
                       help="Directory for exports (default: artifacts/optuna_parallel/results)")

    args = parser.parse_args()

    # Handle dry-run mode
    if args.dry_run:
        success = run_dry_run(args.gpu_id, args.seed)
        sys.exit(0 if success else 1)

    # Handle export mode
    if args.export:
        export_dir = Path(args.export_dir) if args.export_dir else RESULTS_DIR
        export_study_results(args.study_name, args.storage, export_dir)
        sys.exit(0)

    # Normal HPO mode
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Worker on GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")

    # Load or create study with ASHA pruner
    try:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=args.storage,
        )
        print(f"Loaded existing study with {len(study.trials)} trials")
    except:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=5,
                max_resource=args.epochs,
                reduction_factor=3,
            ),
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=20,
                multivariate=True,
            ),
        )
        print("Created new study")

    # Load data
    print("Loading data...")
    data = prepare_data(seed=args.seed)
    print(f"  Train: {len(data['train_idx'])}, Val: {len(data['val_idx'])}")

    # Determine checkpoint saving
    save_checkpoints = args.save_checkpoints and not args.no_checkpoints

    # Run trials
    def wrapped_objective(trial):
        return objective(
            trial, data, device, study,
            num_epochs=args.epochs,
            early_stop_patience=args.early_stop,
            base_seed=args.seed,
            save_checkpoints=save_checkpoints,
        )

    study.optimize(
        wrapped_objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    # Print summary
    print("\n" + "="*60)
    print(f"Worker GPU {args.gpu_id} Complete!")
    print(f"Total trials: {len(study.trials)}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  Completed: {len(completed)}, Pruned: {len(pruned)}")

    if study.best_trial:
        print(f"\nBest trial: #{study.best_trial.number}")
        print(f"  Score: {study.best_trial.value:.4f}")
        print(f"  Corr: {study.best_trial.user_attrs.get('best_corr', 'N/A')}")
        print(f"  R²: {study.best_trial.user_attrs.get('best_r2', 'N/A')}")
        print("  Params:")
        for k, v in study.best_trial.params.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4e}")
            else:
                print(f"    {k}: {v}")

    print("="*60)

    # Auto-export results
    print("\nAuto-exporting results...")
    export_study_results(args.study_name, args.storage, RESULTS_DIR)


if __name__ == "__main__":
    main()
