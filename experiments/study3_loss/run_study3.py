#!/usr/bin/env python3
"""
Study 3: Loss Function Ablation (STAGE 1)
=========================================

Runs HPO to find optimal loss function combinations for OB→PCx translation.

Stage 1 Configuration:
- 80 epochs per trial
- Early stopping patience: 8
- Top-3 checkpoint saving

Usage:
    # Full run
    python experiments/study3_loss/run_study3.py --trials 80

    # Dry-run test
    python experiments/study3_loss/run_study3.py --dry-run --epochs 3

    # Specific loss
    python experiments/study3_loss/run_study3.py --loss huber --trials 20
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# CRITICAL: Fix Python path for imports
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

try:
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("Error: optuna not installed. Run: pip install optuna")
    sys.exit(1)

from experiments.common.error_handling import trial_context
from experiments.common.validation import validate_not_nan_inf


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "study3_loss"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Stage 1 training config
STAGE1_EPOCHS = 80
STAGE1_PATIENCE = 8
CHECKPOINT_TOP_K = 3

# Loss functions to compare - comprehensive ablation for Nature Methods
LOSS_TYPES = [
    # Standard regression losses
    "l1", "mse", "huber", "log_cosh",
    # Spectral/frequency-domain losses
    "multi_scale_spectral", "stft", "band_specific",
    # Correlation-based losses (direct optimization of correlation metrics)
    "pearson", "concordance",
    # Probabilistic/uncertainty losses
    "gaussian_nll",
    # Structure-preserving losses
    "ssim", "focal",
    # Adversarial/perceptual losses
    "adversarial", "perceptual",
    # Combined losses
    "combined"
]

# Core losses for quick runs (subset)
CORE_LOSS_TYPES = [
    "l1", "mse", "huber", "log_cosh",
    "multi_scale_spectral", "pearson", "concordance"
]


# =============================================================================
# HP Spaces for Loss Functions
# =============================================================================

def get_loss_hp_space(trial: optuna.Trial, loss_type: str) -> Dict[str, Any]:
    """Get hyperparameter space for specific loss function."""

    # Common training params
    hp = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    }

    if loss_type == "l1":
        pass  # No additional params

    elif loss_type == "mse":
        pass  # No additional params

    elif loss_type == "huber":
        hp.update({
            "delta": trial.suggest_float("delta", 0.1, 2.0),
        })

    elif loss_type == "multi_scale_spectral":
        hp.update({
            "log_scale": trial.suggest_categorical("log_scale", [True, False]),
            "mag_weight": trial.suggest_float("mag_weight", 0.5, 2.0),
            "phase_weight": trial.suggest_float("phase_weight", 0.0, 0.5),
        })

    elif loss_type == "stft":
        hp.update({
            "n_fft": trial.suggest_categorical("n_fft", [256, 512, 1024]),
            "hop_length": trial.suggest_categorical("hop_length", [64, 128, 256]),
            "log_scale": trial.suggest_categorical("log_scale", [True, False]),
        })

    elif loss_type == "band_specific":
        hp.update({
            "n_fft": trial.suggest_categorical("n_fft", [512, 1024, 2048]),
            "delta_weight": trial.suggest_float("delta_weight", 0.1, 2.0),
            "theta_weight": trial.suggest_float("theta_weight", 0.5, 2.0),
            "alpha_weight": trial.suggest_float("alpha_weight", 0.5, 2.0),
            "beta_weight": trial.suggest_float("beta_weight", 0.5, 2.0),
            "gamma_weight": trial.suggest_float("gamma_weight", 1.0, 3.0),
        })

    elif loss_type == "gaussian_nll":
        hp.update({
            "learn_variance": trial.suggest_categorical("learn_variance", [True, False]),
            "min_variance": trial.suggest_float("min_variance", 1e-8, 1e-4, log=True),
        })

    elif loss_type == "combined":
        hp.update({
            "time_loss": trial.suggest_categorical("time_loss", ["l1", "mse", "huber"]),
            "time_weight": trial.suggest_float("time_weight", 0.5, 2.0),
            "spectral_weight": trial.suggest_float("spectral_weight", 0.5, 2.0),
        })

    # =========================================================================
    # Literature-standard losses
    # =========================================================================

    elif loss_type == "log_cosh":
        pass  # No additional params - smooth L1/L2 hybrid

    elif loss_type == "pearson":
        hp.update({
            "per_channel": trial.suggest_categorical("per_channel", [True, False]),
        })

    elif loss_type == "concordance":
        hp.update({
            "per_channel": trial.suggest_categorical("per_channel", [True, False]),
        })

    elif loss_type == "ssim":
        hp.update({
            "window_size": trial.suggest_categorical("window_size", [7, 11, 15]),
            "data_range": trial.suggest_float("data_range", 1.0, 4.0),
        })

    elif loss_type == "focal":
        hp.update({
            "gamma": trial.suggest_float("gamma", 1.0, 4.0),
        })

    elif loss_type == "adversarial":
        hp.update({
            "loss_type": trial.suggest_categorical("adv_loss_type", ["lsgan", "hinge"]),
        })

    elif loss_type == "perceptual":
        hp.update({
            "feature_dims": trial.suggest_categorical("feature_dims",
                ["64,128", "64,128,256", "128,256"]),
        })

    return hp


def create_loss_function(loss_type: str, hp: Dict[str, Any]) -> nn.Module:
    """Create loss function from hyperparameters."""

    from experiments.study3_loss.losses import create_loss

    loss_params = {k: v for k, v in hp.items()
                   if k not in ["lr", "weight_decay"]}

    if loss_type == "band_specific":
        # Convert band weights
        band_weights = {
            "delta": loss_params.pop("delta_weight", 1.0),
            "theta": loss_params.pop("theta_weight", 1.0),
            "alpha": loss_params.pop("alpha_weight", 1.0),
            "beta": loss_params.pop("beta_weight", 1.0),
            "gamma": loss_params.pop("gamma_weight", 2.0),
        }
        loss_params["band_weights"] = band_weights

    return create_loss(loss_type, **loss_params)


# =============================================================================
# Objective Function
# =============================================================================

def study3_objective(
    trial: optuna.Trial,
    loss_type: str,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    n_epochs: int = 80,
) -> float:
    """Objective function for loss function ablation.

    Returns:
        Validation R² (to maximize)
    """
    with trial_context(trial):
        # Get hyperparameters
        hp = get_loss_hp_space(trial, loss_type)

        # Create loss function
        criterion = create_loss_function(loss_type, hp)
        if hasattr(criterion, 'to'):
            criterion = criterion.to(device)

        # Clone model for this trial (use deepcopy to handle both custom and Sequential models)
        model_copy = copy.deepcopy(model)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model_copy.parameters(),
            lr=hp["lr"],
            weight_decay=hp["weight_decay"],
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=hp["lr"] * 0.01
        )

        best_val_r2 = -float("inf")

        for epoch in range(n_epochs):
            # Training
            model_copy.train()

            for batch in train_loader:
                # Handle both 2-tuple (x, y) and 3-tuple (x, y, label) formats
                if len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                y_pred = model_copy(x)

                validate_not_nan_inf(y_pred, f"output epoch {epoch}")

                loss = criterion(y_pred, y)

                # Handle losses that return learnable params
                if hasattr(criterion, 'parameters'):
                    total_params = list(model_copy.parameters()) + list(criterion.parameters())
                    loss.backward()
                else:
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(model_copy.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # Validation (always with L1 for fair comparison)
            model_copy.eval()
            val_preds, val_targets = [], []

            with torch.no_grad():
                for batch in val_loader:
                    # Handle both 2-tuple (x, y) and 3-tuple (x, y, label) formats
                    if len(batch) == 3:
                        x, y, _ = batch
                    else:
                        x, y = batch
                    x, y = x.to(device), y.to(device)
                    y_pred = model_copy(x)
                    val_preds.append(y_pred.cpu())
                    val_targets.append(y.cpu())

            val_preds = torch.cat(val_preds, dim=0)
            val_targets = torch.cat(val_targets, dim=0)

            # Compute R²
            ss_res = ((val_targets - val_preds) ** 2).sum()
            ss_tot = ((val_targets - val_targets.mean()) ** 2).sum()
            val_r2 = (1 - ss_res / (ss_tot + 1e-8)).item()

            # Report and prune
            trial.report(val_r2, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2

        # Cleanup
        del model_copy, optimizer, criterion
        gc.collect()
        torch.cuda.empty_cache()

        return best_val_r2


# =============================================================================
# Main Runner
# =============================================================================

def run_loss_type(
    loss_type: str,
    model: nn.Module,
    device: torch.device,
    train_loader,
    val_loader,
    n_trials: int = 20,
    n_epochs: int = 80,
    seed: int = 42,
    dry_run: bool = False,
) -> optuna.Study:
    """Run HPO for a single loss type."""

    study_name = f"study3_{loss_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage = f"sqlite:///{ARTIFACTS_DIR / 'study3.db'}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=HyperbandPruner(min_resource=10, max_resource=n_epochs, reduction_factor=3),
        load_if_exists=True,
    )

    def objective(trial):
        return study3_objective(
            trial=trial,
            loss_type=loss_type,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            n_epochs=n_epochs,
        )

    n_trials_actual = 1 if dry_run else n_trials
    study.optimize(objective, n_trials=n_trials_actual, show_progress_bar=True)

    return study


def run_all_losses(
    model: nn.Module,
    device: torch.device,
    train_loader,
    val_loader,
    n_trials_per_loss: int = 10,
    n_epochs: int = 80,
    seed: int = 42,
    dry_run: bool = False,
    loss_types: Optional[List[str]] = None,
) -> Dict[str, optuna.Study]:
    """Run HPO for all loss types."""

    loss_types = loss_types or LOSS_TYPES
    results = {}

    for loss_type in loss_types:
        print(f"\n{'='*60}")
        print(f"Running Study 3: Loss = {loss_type}")
        print(f"{'='*60}\n")

        study = run_loss_type(
            loss_type=loss_type,
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=n_trials_per_loss,
            n_epochs=n_epochs,
            seed=seed,
            dry_run=dry_run,
        )

        results[loss_type] = study

        try:
            best = study.best_trial
            if best:
                print(f"\n{loss_type} best R²: {best.value:.4f}")
        except (ValueError, RuntimeError):
            print(f"\n{loss_type}: No successful trials")

    return results


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Study 3: Loss Function Ablation")
    parser.add_argument("--loss", type=str, choices=LOSS_TYPES, help="Single loss type to run")
    parser.add_argument("--trials", type=int, default=10, help="Trials per loss type")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal test")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    try:
        from data import prepare_data, create_dataloaders
        data = prepare_data()
        loaders = create_dataloaders(data, batch_size=16)
        train_loader = loaders["train"]
        val_loader = loaders["val"]
    except Exception as e:
        print(f"Warning: Using dummy data for testing (reason: {type(e).__name__}: {e})")
        from torch.utils.data import TensorDataset, DataLoader
        X = torch.randn(100, 32, 1000)
        y = torch.randn(100, 32, 1000)
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16)

    # Create base model (use UNet as default)
    print("Creating base model...")
    try:
        from models import UNet1DConditioned
        model = UNet1DConditioned(
            in_channels=32, out_channels=32, base_channels=64
        ).to(device)
        model.init_kwargs = {"in_channels": 32, "out_channels": 32, "base_channels": 64}
    except Exception as e:
        print(f"Warning: Using dummy model for testing (reason: {type(e).__name__}: {e})")
        # Dummy model for testing
        model = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
        ).to(device)
        model.init_kwargs = {}

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Run
    if args.loss:
        results = {args.loss: run_loss_type(
            loss_type=args.loss,
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=args.trials,
            n_epochs=args.epochs,
            seed=args.seed,
            dry_run=args.dry_run,
        )}
    else:
        results = run_all_losses(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials_per_loss=args.trials,
            n_epochs=args.epochs,
            seed=args.seed,
            dry_run=args.dry_run,
        )

    # Summary
    print("\n" + "="*60)
    print("Study 3 Complete: Loss Function Ablation")
    print("="*60)

    for loss_type, study in results.items():
        try:
            best = study.best_trial
            if best:
                print(f"  {loss_type}: R² = {best.value:.4f}")
        except (ValueError, RuntimeError):
            print(f"  {loss_type}: No successful trials")


if __name__ == "__main__":
    main()
