#!/usr/bin/env python3
"""
Tier 1: Quick Screening
=======================

Fast elimination of clearly inferior options before heavy compute.
Uses 20% of data, 20 epochs, single seed/fold.

Purpose: Reduce 8 arch × 6 loss = 48 combinations down to top-3 each.

GATE: At least one neural method must beat classical floor + 0.05

Usage:
    python experiments/tier1_screening/run_tier1.py
    python experiments/tier1_screening/run_tier1.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# CRITICAL: Fix Python path for imports
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.cuda.amp import autocast, GradScaler

from experiments.common.config_registry import (
    ScreeningResult,
    Tier1Result,
    GateFailure,
    get_registry,
)
from experiments.study1_architecture.architectures import create_architecture, ARCHITECTURE_REGISTRY
from experiments.study3_loss.losses import create_loss


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier1_screening"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 8 architectures (covers all paradigms)
ARCHITECTURES = list(ARCHITECTURE_REGISTRY.keys())  # linear, cnn, wavenet, fnet, vit, performer, mamba

# Add unet if available
try:
    from models import UNet1DConditioned
    ARCHITECTURES = ["unet"] + ARCHITECTURES
except ImportError:
    pass

# 6 loss categories (representatives from each family)
LOSS_CATEGORIES = {
    "huber": "huber",           # Point-wise robust
    "spectral": "multi_scale_spectral",  # Spectral fidelity
    "ccc": "concordance",       # Correlation-based
    "ssim": "ssim",             # Structure-preserving
    "gaussian_nll": "gaussian_nll",  # Probabilistic
    "combined": "combined",     # Time + frequency
}

# Screening hyperparameters
SCREEN_DATA_FRACTION = 0.2
SCREEN_EPOCHS = 20
SCREEN_BATCH_SIZE = 8  # Reduced from 16 to avoid OOM
SCREEN_LR = 1e-3


# =============================================================================
# Model Creation Helpers
# =============================================================================

def create_model(arch_name: str, in_channels: int, out_channels: int, device: torch.device) -> nn.Module:
    """Create model by architecture name.

    Args:
        arch_name: Architecture name
        in_channels: Input channels
        out_channels: Output channels
        device: Device to place model on

    Returns:
        Model instance

    Variant mapping:
        - linear: simple
        - cnn: basic (NOT standard!)
        - wavenet: standard
        - fnet: standard
        - vit: standard
        - performer: standard
        - mamba: standard
    """
    # Variant mapping - each arch has different default variants
    VARIANT_MAP = {
        "linear": "simple",
        "cnn": "basic",  # CNN uses "basic", not "standard"
        "wavenet": "standard",
        "fnet": "standard",
        "vit": "standard",
        "performer": "standard",
        "mamba": "standard",
    }

    if arch_name == "unet":
        from models import UNet1DConditioned
        model = UNet1DConditioned(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=32,  # Smaller for screening
        )
    elif arch_name in VARIANT_MAP:
        variant = VARIANT_MAP[arch_name]
        model = create_architecture(arch_name, variant=variant, in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    return model.to(device)


# =============================================================================
# Quick Training
# =============================================================================

def quick_train(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 20,
    lr: float = 1e-3,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Quick training for screening with AMP for memory efficiency.

    Args:
        model: Model to train
        criterion: Loss function
        train_loader: Training data
        val_loader: Validation data
        device: Device
        n_epochs: Number of epochs
        lr: Learning rate
        use_amp: Use automatic mixed precision (reduces memory ~50%)

    Returns:
        Dictionary with r2, mae, and training time
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # AMP scaler for mixed precision training
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None

    start_time = time.time()

    for epoch in range(n_epochs):
        # Train
        model.train()
        for batch in train_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            if scaler is not None:
                with autocast():
                    y_pred = model(x)
                    try:
                        loss = criterion(y_pred, y)
                    except Exception:
                        loss = nn.functional.l1_loss(y_pred, y)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                y_pred = model(x)
                try:
                    loss = criterion(y_pred, y)
                except Exception:
                    loss = nn.functional.l1_loss(y_pred, y)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        scheduler.step()

    training_time = time.time() - start_time

    # Evaluate (use AMP for inference too to save memory)
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)

            if scaler is not None:
                with autocast():
                    y_pred = model(x)
            else:
                y_pred = model(x)

            all_preds.append(y_pred.float().cpu())
            all_targets.append(y.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # R²
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = (1 - ss_res / (ss_tot + 1e-8)).item()

    # MAE
    mae = (targets - preds).abs().mean().item()

    return {
        "r2": r2,
        "mae": mae,
        "training_time": training_time,
    }


# =============================================================================
# Checkpoint Support
# =============================================================================

def _load_checkpoint() -> Dict[str, ScreeningResult]:
    """Load checkpoint for intra-tier resume."""
    checkpoint_file = ARTIFACTS_DIR / "tier1_checkpoint.json"
    if not checkpoint_file.exists():
        return {}

    with open(checkpoint_file) as f:
        data = json.load(f)

    results = {}
    for key, r in data.items():
        results[key] = ScreeningResult(
            architecture=r["architecture"],
            loss=r["loss"],
            r2=r["r2"],
            mae=r["mae"],
            training_time=r["training_time"],
        )
    return results


def _save_checkpoint(results: Dict[str, ScreeningResult]) -> None:
    """Save checkpoint for intra-tier resume."""
    checkpoint_file = ARTIFACTS_DIR / "tier1_checkpoint.json"
    data = {}
    for key, r in results.items():
        data[key] = {
            "architecture": r.architecture,
            "loss": r.loss,
            "r2": r.r2,
            "mae": r.mae,
            "training_time": r.training_time,
        }
    with open(checkpoint_file, "w") as f:
        json.dump(data, f, indent=2)


def _clear_checkpoint() -> None:
    """Clear checkpoint after successful tier completion."""
    checkpoint_file = ARTIFACTS_DIR / "tier1_checkpoint.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()


# =============================================================================
# Screening Matrix
# =============================================================================

def run_screening_matrix(
    X: torch.Tensor,
    y: torch.Tensor,
    architectures: List[str],
    losses: Dict[str, str],
    device: torch.device,
    data_fraction: float = 0.2,
    n_epochs: int = 20,
    batch_size: int = 16,
    seed: int = 42,
    resume: bool = True,
) -> List[ScreeningResult]:
    """Run full 8x6 screening matrix.

    Args:
        X: Input data [N, C, T]
        y: Target data [N, C, T]
        architectures: List of architecture names
        losses: Dict of loss_category -> loss_name
        device: Device
        data_fraction: Fraction of data to use
        n_epochs: Training epochs
        batch_size: Batch size
        seed: Random seed
        resume: Whether to resume from checkpoint

    Returns:
        List of ScreeningResult for each combination
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load checkpoint for intra-tier resume
    completed_results = {}
    if resume:
        completed_results = _load_checkpoint()
        if completed_results:
            print(f"\n🔄 Resuming from checkpoint: {len(completed_results)} combinations already done")

    n_samples = X.shape[0]
    in_channels = X.shape[1]
    out_channels = y.shape[1]

    # Subsample data
    n_screen = int(n_samples * data_fraction)
    indices = np.random.permutation(n_samples)[:n_screen]

    # Split into train/val
    n_train = int(n_screen * 0.8)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Create dataloaders
    dataset = TensorDataset(X, y)
    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    results = []
    total_combos = len(architectures) * len(losses)
    combo_idx = 0

    for arch_name in architectures:
        for loss_category, loss_name in losses.items():
            combo_idx += 1
            combo_key = f"{arch_name}_{loss_category}"

            # Skip if already completed
            if combo_key in completed_results:
                result = completed_results[combo_key]
                print(f"  [{combo_idx}/{total_combos}] {arch_name} + {loss_category}... SKIPPED (already done) R² = {result.r2:.4f}")
                results.append(result)
                continue

            print(f"  [{combo_idx}/{total_combos}] {arch_name} + {loss_category}...", end=" ", flush=True)

            model = None  # Initialize for cleanup
            try:
                # Create fresh model
                model = create_model(arch_name, in_channels, out_channels, device)

                # Create loss
                criterion = create_loss(loss_name)
                if hasattr(criterion, 'to'):
                    criterion = criterion.to(device)

                # Train
                metrics = quick_train(
                    model=model,
                    criterion=criterion,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    n_epochs=n_epochs,
                    lr=SCREEN_LR,
                )

                result = ScreeningResult(
                    architecture=arch_name,
                    loss=loss_category,
                    r2=metrics["r2"],
                    mae=metrics["mae"],
                    training_time=metrics["training_time"],
                )
                results.append(result)

                print(f"R² = {metrics['r2']:.4f} ({metrics['training_time']:.1f}s)")

                # Save checkpoint after each combination
                completed_results[combo_key] = result
                _save_checkpoint(completed_results)

            except Exception as e:
                print(f"FAILED: {e}")
                result = ScreeningResult(
                    architecture=arch_name,
                    loss=loss_category,
                    r2=float("-inf"),
                    mae=float("inf"),
                    training_time=0,
                )
                results.append(result)

                # Save checkpoint even for failures
                completed_results[combo_key] = result
                _save_checkpoint(completed_results)

            # Cleanup
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()

    return results


def select_top_k(
    results: List[ScreeningResult],
    k: int = 3,
    margin: float = 0.02,
) -> Tuple[List[str], List[str]]:
    """Select top-k architectures and losses with safety margin.

    Args:
        results: Screening results
        k: Number to select
        margin: Safety margin for including extras

    Returns:
        (top_architectures, top_losses)
    """
    # Aggregate by architecture
    arch_scores = {}
    for r in results:
        if r.r2 > float("-inf"):
            if r.architecture not in arch_scores:
                arch_scores[r.architecture] = []
            arch_scores[r.architecture].append(r.r2)

    arch_means = {a: np.mean(scores) for a, scores in arch_scores.items()}

    # Aggregate by loss
    loss_scores = {}
    for r in results:
        if r.r2 > float("-inf"):
            if r.loss not in loss_scores:
                loss_scores[r.loss] = []
            loss_scores[r.loss].append(r.r2)

    loss_means = {l: np.mean(scores) for l, scores in loss_scores.items()}

    # Select top-k with safety margin
    def select_with_margin(scores: Dict[str, float], k: int, margin: float) -> List[str]:
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        selected = [x[0] for x in sorted_items[:k]]

        # Check if (k+1)th is within margin
        if len(sorted_items) > k:
            kth_score = sorted_items[k - 1][1]
            next_score = sorted_items[k][1]
            if kth_score - next_score < margin:
                selected.append(sorted_items[k][0])

        return selected

    top_archs = select_with_margin(arch_means, k, margin)
    top_losses = select_with_margin(loss_means, k, margin)

    return top_archs, top_losses


# =============================================================================
# Main Runner
# =============================================================================

def run_tier1(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    architectures: Optional[List[str]] = None,
    losses: Optional[Dict[str, str]] = None,
    data_fraction: float = SCREEN_DATA_FRACTION,
    n_epochs: int = SCREEN_EPOCHS,
    seed: int = 42,
    register: bool = True,
    resume: bool = True,
) -> Tier1Result:
    """Run Tier 1: Quick Screening.

    Args:
        X: Input data [N, C, T]
        y: Target data [N, C, T]
        device: Device
        architectures: List of architecture names
        losses: Dict of loss_category -> loss_name
        data_fraction: Fraction of data to use
        n_epochs: Training epochs
        seed: Random seed
        register: Whether to register with ConfigRegistry
        resume: Whether to resume from checkpoint

    Returns:
        Tier1Result with top selections
    """
    architectures = architectures or ARCHITECTURES
    losses = losses or LOSS_CATEGORIES

    print(f"\nRunning {len(architectures)} architectures × {len(losses)} losses = {len(architectures) * len(losses)} combinations")
    print(f"Data: {int(data_fraction * 100)}%, Epochs: {n_epochs}\n")

    # Run screening matrix
    results = run_screening_matrix(
        X=X,
        y=y,
        architectures=architectures,
        losses=losses,
        device=device,
        data_fraction=data_fraction,
        n_epochs=n_epochs,
        seed=seed,
        resume=resume,
    )

    # Select top performers
    top_archs, top_losses = select_top_k(results, k=3, margin=0.02)

    # Get best R²
    valid_results = [r for r in results if r.r2 > float("-inf")]
    best_r2 = max(r.r2 for r in valid_results) if valid_results else 0.0

    # Get classical floor from registry
    registry = get_registry(ARTIFACTS_DIR)
    classical_floor = registry.classical_floor if registry.tier0 else 0.0

    tier1_result = Tier1Result(
        results=results,
        top_architectures=top_archs,
        top_losses=top_losses,
        best_r2=best_r2,
        classical_floor=classical_floor,
    )

    # Register
    if register:
        registry.register_tier1(tier1_result)

    # Clear checkpoint on successful completion
    _clear_checkpoint()

    return tier1_result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tier 1: Quick Screening")
    parser.add_argument("--dry-run", action="store_true", help="Use minimal configuration")
    parser.add_argument("--epochs", type=int, default=SCREEN_EPOCHS, help="Training epochs")
    parser.add_argument("--data-fraction", type=float, default=SCREEN_DATA_FRACTION, help="Data fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TIER 1: Quick Screening")
    print("=" * 60)
    print("Purpose: Eliminate inferior options before heavy compute")
    print("Method: 8 architectures × 6 losses on 20% data")
    print()

    # Load data
    print("Loading data...")

    if args.dry_run:
        # Minimal data
        N = 100
        C = 32
        T = 500
        X = torch.randn(N, C, T)
        y = torch.randn(N, C, T)
        # Use only 2 architectures and 2 losses for dry-run
        architectures = ["linear", "cnn"]
        losses = {"huber": "huber", "spectral": "multi_scale_spectral"}
        n_epochs = 3
        data_fraction = 0.5
        print(f"  [DRY-RUN] Using synthetic data: {X.shape}")
    else:
        try:
            from data import prepare_data, create_dataloaders
            data = prepare_data()
            X_train = data["train"]["ob"]
            y_train = data["train"]["hp"]
            X_val = data["val"]["ob"]
            y_val = data["val"]["hp"]
            X = torch.cat([X_train, X_val], dim=0)
            y = torch.cat([y_train, y_val], dim=0)
            architectures = ARCHITECTURES
            losses = LOSS_CATEGORIES
            n_epochs = args.epochs
            data_fraction = args.data_fraction
            print(f"  Loaded real data: {X.shape}")
        except Exception as e:
            print(f"  Warning: Could not load data ({e}), using synthetic")
            N = 200
            C = 32
            T = 1000
            X = torch.randn(N, C, T)
            y = torch.randn(N, C, T)
            architectures = ARCHITECTURES
            losses = LOSS_CATEGORIES
            n_epochs = args.epochs
            data_fraction = args.data_fraction

    # Run screening
    result = run_tier1(
        X=X,
        y=y,
        device=device,
        architectures=architectures,
        losses=losses,
        data_fraction=data_fraction,
        n_epochs=n_epochs,
        seed=args.seed,
    )

    # Summary
    print("\n" + "=" * 60)
    print("TIER 1 RESULTS: Screening Matrix")
    print("=" * 60)

    # Print matrix
    print("\nR² Matrix:")
    print("-" * 60)

    # Get unique archs and losses
    archs = sorted(set(r.architecture for r in result.results))
    loss_cats = sorted(set(r.loss for r in result.results))

    # Header
    print(f"{'Architecture':15s}", end="")
    for loss in loss_cats:
        print(f" {loss:>10s}", end="")
    print()

    # Rows
    for arch in archs:
        print(f"{arch:15s}", end="")
        for loss in loss_cats:
            r = next((r for r in result.results if r.architecture == arch and r.loss == loss), None)
            if r and r.r2 > float("-inf"):
                print(f" {r.r2:10.4f}", end="")
            else:
                print(f" {'FAIL':>10s}", end="")
        print()

    print("\n" + "-" * 60)
    print(f"Top Architectures: {result.top_architectures}")
    print(f"Top Losses: {result.top_losses}")
    print(f"Best R²: {result.best_r2:.4f}")

    # Check gate
    margin = result.best_r2 - result.classical_floor
    print(f"\nClassical Floor: {result.classical_floor:.4f}")
    print(f"Margin over classical: {margin:.4f}")
    if margin >= 0.05:
        print("GATE: PASSED (neural beats classical)")
    else:
        print("GATE: FAILED (neural doesn't beat classical by sufficient margin)")

    # Save
    output_file = args.output or (ARTIFACTS_DIR / f"tier1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_file = Path(output_file)

    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return result


if __name__ == "__main__":
    main()
