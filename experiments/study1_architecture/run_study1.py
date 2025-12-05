#!/usr/bin/env python3
"""
Study 1: Architecture Comparison (STAGE 1)
==========================================

Runs HPO to compare 8 architectures for OB→PCx neural signal translation.

Stage 1 Configuration:
- 80 epochs per trial
- Early stopping patience: 8
- Top-3 checkpoint saving per architecture

Usage:
    # Full run
    python experiments/study1_architecture/run_study1.py --trials 100

    # Dry-run test (3 epochs, 1 trial)
    python experiments/study1_architecture/run_study1.py --dry-run --epochs 3

    # Specific architecture
    python experiments/study1_architecture/run_study1.py --arch unet --trials 20
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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

from experiments.common.error_handling import trial_context, OOMError
from experiments.common.validation import validate_tensor_shape, validate_not_nan_inf
from experiments.common.checkpointing import TrialCheckpoint, BestModelSaver, TopKCheckpointSaver


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "study1_architecture"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Stage 1 training config
STAGE1_EPOCHS = 80
STAGE1_PATIENCE = 8
CHECKPOINT_TOP_K = 3

# Architecture registry
ARCHITECTURES = ["linear", "cnn", "unet", "wavenet", "fnet", "vit", "performer", "mamba"]


# =============================================================================
# HP Spaces
# =============================================================================

def get_architecture_hp_space(trial: optuna.Trial, arch: str) -> Dict[str, Any]:
    """Get hyperparameter space for specific architecture."""

    # Common parameters
    hp = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
    }

    if arch == "linear":
        hp.update({
            "variant": trial.suggest_categorical("variant", ["simple", "shared", "mixer"]),
        })

    elif arch == "cnn":
        hp.update({
            "variant": trial.suggest_categorical("variant", ["basic", "pooling"]),
            "n_layers": trial.suggest_int("n_layers", 3, 8),
            "hidden_channels": trial.suggest_categorical("hidden_channels", [32, 64, 128]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "use_film": trial.suggest_categorical("use_film", [True, False]),
        })

    elif arch == "unet":
        hp.update({
            "base_channels": trial.suggest_categorical("base_channels", [32, 64, 128]),
            "n_downsample": trial.suggest_int("n_downsample", 2, 4),
            "attention_type": trial.suggest_categorical("attention_type", ["none", "self", "cross"]),
        })

    elif arch == "wavenet":
        hp.update({
            "hidden_channels": trial.suggest_categorical("hidden_channels", [32, 64, 128]),
            "n_layers": trial.suggest_int("n_layers", 6, 12),
            "dilation_base": trial.suggest_categorical("dilation_base", [2, 3]),
            "kernel_size": trial.suggest_categorical("kernel_size", [2, 3]),
            "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        })

    elif arch == "fnet":
        hp.update({
            "embed_dim": trial.suggest_categorical("embed_dim", [128, 256, 512]),
            "num_layers": trial.suggest_int("num_layers", 4, 8),
            "use_conv": trial.suggest_categorical("use_conv", [True, False]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        })

    elif arch == "vit":
        hp.update({
            "patch_size": trial.suggest_categorical("patch_size", [16, 32, 64]),
            "embed_dim": trial.suggest_categorical("embed_dim", [128, 256, 512]),
            "num_layers": trial.suggest_int("num_layers", 4, 8),
            "num_heads": trial.suggest_categorical("num_heads", [4, 8]),
            "use_conv_stem": trial.suggest_categorical("use_conv_stem", [True, False]),
        })

    elif arch == "performer":
        hp.update({
            "embed_dim": trial.suggest_categorical("embed_dim", [128, 256, 512]),
            "num_layers": trial.suggest_int("num_layers", 4, 8),
            "num_heads": trial.suggest_categorical("num_heads", [4, 8]),
            "num_features": trial.suggest_categorical("num_features", [64, 128, 256]),
        })

    elif arch == "mamba":
        hp.update({
            "d_model": trial.suggest_categorical("d_model", [128, 256]),
            "d_state": trial.suggest_categorical("d_state", [16, 32, 64]),
            "num_layers": trial.suggest_int("num_layers", 4, 12),
            "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        })

    return hp


# =============================================================================
# Objective Function
# =============================================================================

def study1_objective(
    trial: optuna.Trial,
    arch: str,
    train_loader,
    val_loader,
    device: torch.device,
    n_epochs: int = 80,
    checkpoint_dir: Optional[Path] = None,
) -> float:
    """Objective function for architecture comparison.

    Args:
        trial: Optuna trial
        arch: Architecture name
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Target device
        n_epochs: Number of training epochs
        checkpoint_dir: Directory for checkpoints

    Returns:
        Validation R² (to maximize)
    """
    with trial_context(trial):
        # Get hyperparameters
        hp = get_architecture_hp_space(trial, arch)

        # Create model
        from experiments.study1_architecture.architectures import create_architecture

        model = create_architecture(
            name=arch,
            variant=hp.get("variant", "standard"),
            in_channels=train_loader.dataset[0][0].shape[0],
            out_channels=train_loader.dataset[0][1].shape[0],
            **{k: v for k, v in hp.items() if k not in ["lr", "weight_decay", "batch_size", "variant"]}
        )
        model = model.to(device)

        # Log parameter count
        n_params = sum(p.numel() for p in model.parameters())
        trial.set_user_attr("n_params", n_params)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hp["lr"],
            weight_decay=hp["weight_decay"],
        )

        # Loss
        criterion = nn.L1Loss()

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=hp["lr"] * 0.01
        )

        # Checkpointing
        if checkpoint_dir:
            checkpointer = TrialCheckpoint(checkpoint_dir / f"trial_{trial.number}")
            start_epoch = checkpointer.get_latest_epoch()
            if start_epoch > 0:
                checkpointer.load(model, optimizer, scheduler)
        else:
            start_epoch = 0
            checkpointer = None

        best_val_r2 = -float("inf")

        for epoch in range(start_epoch, n_epochs):
            # Training
            model.train()
            train_loss = 0.0

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                y_pred = model(x)

                # Validate outputs
                validate_not_nan_inf(y_pred, f"output epoch {epoch}")

                loss = criterion(y_pred, y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            # Validation
            model.eval()
            val_preds, val_targets = [], []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    val_preds.append(y_pred.cpu())
                    val_targets.append(y.cpu())

            val_preds = torch.cat(val_preds, dim=0)
            val_targets = torch.cat(val_targets, dim=0)

            # Compute R²
            ss_res = ((val_targets - val_preds) ** 2).sum()
            ss_tot = ((val_targets - val_targets.mean()) ** 2).sum()
            val_r2 = 1 - ss_res / (ss_tot + 1e-8)
            val_r2 = val_r2.item()

            # Report and prune
            trial.report(val_r2, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            # Track best
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2

            # Checkpoint
            if checkpointer and (epoch + 1) % 10 == 0:
                checkpointer.save(model, optimizer, scheduler, epoch + 1, {"val_r2": val_r2})

        # Cleanup
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

        return best_val_r2


# =============================================================================
# Main Runner
# =============================================================================

def run_architecture(
    arch: str,
    device: torch.device,
    train_loader,
    val_loader,
    n_trials: int = 20,
    n_epochs: int = 80,
    seed: int = 42,
    dry_run: bool = False,
) -> optuna.Study:
    """Run HPO for a single architecture."""

    # Create study
    study_name = f"study1_{arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage = f"sqlite:///{ARTIFACTS_DIR / 'study1.db'}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=HyperbandPruner(
            min_resource=10,
            max_resource=n_epochs,
            reduction_factor=3,
        ),
        load_if_exists=True,
    )

    # Checkpoint directory
    checkpoint_dir = ARTIFACTS_DIR / "checkpoints" / arch
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Run optimization
    def objective(trial):
        return study1_objective(
            trial=trial,
            arch=arch,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            n_epochs=n_epochs,
            checkpoint_dir=checkpoint_dir if not dry_run else None,
        )

    n_trials_actual = 1 if dry_run else n_trials
    study.optimize(objective, n_trials=n_trials_actual, show_progress_bar=True)

    return study


def run_all_architectures(
    device: torch.device,
    train_loader,
    val_loader,
    n_trials_per_arch: int = 15,
    n_epochs: int = 80,
    seed: int = 42,
    dry_run: bool = False,
    architectures: Optional[List[str]] = None,
) -> Dict[str, optuna.Study]:
    """Run HPO for all architectures."""

    architectures = architectures or ARCHITECTURES
    results = {}

    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"Running Study 1: Architecture = {arch}")
        print(f"{'='*60}\n")

        study = run_architecture(
            arch=arch,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=n_trials_per_arch,
            n_epochs=n_epochs,
            seed=seed,
            dry_run=dry_run,
        )

        results[arch] = study

        # Log best result
        try:
            if study.best_trial:
                print(f"\n{arch} best R²: {study.best_trial.value:.4f}")
                print(f"Best params: {study.best_trial.params}")
        except ValueError:
            print(f"\n{arch}: No successful trials")

    return results


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Study 1: Architecture Comparison (Stage 1)")
    parser.add_argument("--arch", type=str, choices=ARCHITECTURES, help="Single architecture to run")
    parser.add_argument("--trials", type=int, default=15, help="Trials per architecture")
    parser.add_argument("--epochs", type=int, default=STAGE1_EPOCHS, help="Training epochs (default: 80)")
    parser.add_argument("--patience", type=int, default=STAGE1_PATIENCE, help="Early stopping patience (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal test")
    parser.add_argument("--export", action="store_true", help="Export results only")

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.export:
        # Export existing results
        from experiments.common.reporting import export_study_results
        storage = f"sqlite:///{ARTIFACTS_DIR / 'study1.db'}"
        summaries = optuna.get_all_study_summaries(storage)
        for s in summaries:
            study = optuna.load_study(study_name=s.study_name, storage=storage)
            export_study_results(study, ARTIFACTS_DIR / f"{s.study_name}_results.json")
        return

    # Load data (placeholder - adapt to actual data loading)
    print("Loading data...")
    try:
        from data import prepare_data, create_dataloaders
        data = prepare_data()
        loaders = create_dataloaders(data, batch_size=16, num_workers=4)
        train_loader = loaders["train"]
        val_loader = loaders["val"]
    except Exception as e:
        print(f"Warning: Using dummy data loaders for testing (reason: {type(e).__name__}: {e})")
        # Create dummy data for testing
        from torch.utils.data import TensorDataset, DataLoader
        X = torch.randn(100, 32, 1000)
        y = torch.randn(100, 32, 1000)
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Run
    if args.arch:
        results = {args.arch: run_architecture(
            arch=args.arch,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=args.trials,
            n_epochs=args.epochs,
            seed=args.seed,
            dry_run=args.dry_run,
        )}
    else:
        results = run_all_architectures(
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials_per_arch=args.trials,
            n_epochs=args.epochs,
            seed=args.seed,
            dry_run=args.dry_run,
        )

    # Summary
    print("\n" + "="*60)
    print("Study 1 Complete: Architecture Comparison")
    print("="*60)

    for arch, study in results.items():
        try:
            if study.best_trial:
                print(f"  {arch}: R² = {study.best_trial.value:.4f}")
        except ValueError:
            print(f"  {arch}: No successful trials")


if __name__ == "__main__":
    main()
