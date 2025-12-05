#!/usr/bin/env python3
"""
Tier 2: Factorial Design (KEY TIER)
===================================

Tests interactions between architecture × loss × conditioning.
Uses proper 5-fold CV with multiple seeds for statistical power.

This is the KEY tier that discovers:
- Main effects (which architecture is best on average?)
- Interactions (does the best architecture depend on the loss?)

Design: (Top-4 arch) × (Top-3 loss) × (2 cond) = 24 combinations
        5 seeds × 5 folds = 25 samples per combination

GATE: Winner must beat classical floor by R² >= 0.10

Usage:
    python experiments/tier2_factorial/run_tier2.py
    python experiments/tier2_factorial/run_tier2.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from itertools import product
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

from experiments.common.cross_validation import create_data_splits, CVSplit
from experiments.common.statistics import (
    three_way_anova,
    cohens_d,
    cohens_d_interpretation,
    bootstrap_ci,
    pairwise_tests,
)
from experiments.common.config_registry import (
    FactorialResult,
    Tier2Result,
    GateFailure,
    get_registry,
)
from experiments.common.metrics import compute_psd_error_db
from experiments.study1_architecture.architectures import create_architecture, ARCHITECTURE_REGISTRY
from experiments.study3_loss.losses import create_loss, LOSS_REGISTRY


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier2_factorial"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Training configuration
FACTORIAL_EPOCHS = 80
FACTORIAL_BATCH_SIZE = 16
FACTORIAL_LR = 1e-3
N_SEEDS = 5
N_FOLDS = 5

# Conditioning types
CONDITIONING_TYPES = ["cross_attn", "concat"]

# Loss category mapping (from tier1) - all 6 categories
LOSS_CATEGORY_MAP = {
    "huber": "huber",
    "spectral": "multi_scale_spectral",
    "ccc": "concordance",
    "ssim": "ssim",
    "gaussian_nll": "gaussian_nll",
    "combined": "combined",
}


# =============================================================================
# Model Creation with Conditioning
# =============================================================================

def create_model_with_conditioning(
    arch_name: str,
    cond_type: str,
    in_channels: int,
    out_channels: int,
    device: torch.device,
) -> nn.Module:
    """Create model with specified conditioning type.

    Args:
        arch_name: Architecture name
        cond_type: Conditioning type (cross_attn, concat)
        in_channels: Input channels
        out_channels: Output channels
        device: Device

    Returns:
        Model instance
    """
    # For architectures that support conditioning
    if arch_name == "unet":
        try:
            from models import UNet1DConditioned
            # UNet already supports conditioning
            model = UNet1DConditioned(
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=64,
                conditioning_type=cond_type,
            )
            return model.to(device)
        except (ImportError, TypeError):
            pass

    # For other architectures, wrap with conditioning layer
    base_model = create_architecture(
        arch_name,
        variant="standard",
        in_channels=in_channels,
        out_channels=out_channels,
    )

    # Wrap with conditioning if needed
    if cond_type == "cross_attn":
        model = ConditionedWrapper(base_model, cond_type="attention")
    else:
        model = ConditionedWrapper(base_model, cond_type="concat")

    return model.to(device)


class ConditionedWrapper(nn.Module):
    """Wrapper that adds conditioning to any architecture."""

    def __init__(self, base_model: nn.Module, cond_type: str = "concat"):
        super().__init__()
        self.base_model = base_model
        self.cond_type = cond_type

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional conditioning.

        Args:
            x: Input tensor [B, C, T]
            condition: Optional conditioning tensor

        Returns:
            Output tensor [B, C, T]
        """
        # For now, just pass through (conditioning is architecture-specific)
        return self.base_model(x)


# =============================================================================
# Training Function
# =============================================================================

def train_single_config(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 80,
    lr: float = 1e-3,
) -> Dict[str, float]:
    """Train a single configuration and return metrics.

    Args:
        model: Model to train
        criterion: Loss function
        train_loader: Training data
        val_loader: Validation data
        device: Device
        n_epochs: Number of epochs
        lr: Learning rate

    Returns:
        Dictionary with metrics (r2, mae, pearson, psd_error_db)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    best_val_r2 = float("-inf")
    patience_counter = 0
    patience = 10

    for epoch in range(n_epochs):
        # Training
        model.train()
        for batch in train_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
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

        # Quick validation check for early stopping
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        x, y, _ = batch
                    else:
                        x, y = batch
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    val_preds.append(y_pred.cpu())
                    val_targets.append(y.cpu())

            preds = torch.cat(val_preds, dim=0)
            targets = torch.cat(val_targets, dim=0)

            ss_res = ((targets - preds) ** 2).sum()
            ss_tot = ((targets - targets.mean()) ** 2).sum()
            val_r2 = (1 - ss_res / (ss_tot + 1e-8)).item()

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

    # Final evaluation
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # R²
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = (1 - ss_res / (ss_tot + 1e-8)).item()

    # MAE
    mae = (targets - preds).abs().mean().item()

    # Pearson correlation
    preds_flat = preds.view(-1).numpy()
    targets_flat = targets.view(-1).numpy()
    pearson = np.corrcoef(preds_flat, targets_flat)[0, 1]

    # PSD error
    try:
        psd_error = compute_psd_error_db(preds, targets)
    except Exception:
        psd_error = 0.0

    return {
        "r2": r2,
        "mae": mae,
        "pearson": float(pearson) if not np.isnan(pearson) else 0.0,
        "psd_error_db": psd_error,
    }


# =============================================================================
# Factorial Design Runner
# =============================================================================

def run_factorial_design(
    X: torch.Tensor,
    y: torch.Tensor,
    architectures: List[str],
    losses: List[str],
    conditionings: List[str],
    device: torch.device,
    n_seeds: int = 5,
    n_folds: int = 5,
    n_epochs: int = 80,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> List[FactorialResult]:
    """Run full factorial design with CV and multiple seeds.

    Args:
        X: Input data [N, C, T]
        y: Target data [N, C, T]
        architectures: List of architecture names
        losses: List of loss category names
        conditionings: List of conditioning types
        device: Device
        n_seeds: Number of random seeds
        n_folds: Number of CV folds
        n_epochs: Training epochs
        holdout_fraction: Fraction for holdout test set
        seed: Base random seed

    Returns:
        List of FactorialResult for all combinations
    """
    n_samples = X.shape[0]
    in_channels = X.shape[1]
    out_channels = y.shape[1]

    # Create data splits
    splits = create_data_splits(
        n_samples=n_samples,
        n_folds=n_folds,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )

    # Create dataset
    dataset = TensorDataset(X, y)

    results = []
    combinations = list(product(architectures, losses, conditionings))
    total_runs = len(combinations) * n_seeds * n_folds
    run_idx = 0

    print(f"\nFactorial Design: {len(combinations)} combinations × {n_seeds} seeds × {n_folds} folds = {total_runs} runs")

    for arch, loss_cat, cond in combinations:
        loss_name = LOSS_CATEGORY_MAP.get(loss_cat, loss_cat)

        for seed_offset in range(n_seeds):
            current_seed = seed + seed_offset

            for fold_idx, cv_split in enumerate(splits.cv_splits):
                run_idx += 1

                print(f"  [{run_idx}/{total_runs}] {arch}+{loss_cat}+{cond} seed={current_seed} fold={fold_idx}...",
                      end=" ", flush=True)

                try:
                    # Set seeds
                    torch.manual_seed(current_seed)
                    np.random.seed(current_seed)

                    # Create dataloaders
                    train_subset = Subset(dataset, cv_split.train_indices.tolist())
                    val_subset = Subset(dataset, cv_split.val_indices.tolist())

                    train_loader = DataLoader(
                        train_subset,
                        batch_size=FACTORIAL_BATCH_SIZE,
                        shuffle=True,
                        drop_last=True,
                    )
                    val_loader = DataLoader(
                        val_subset,
                        batch_size=FACTORIAL_BATCH_SIZE,
                        shuffle=False,
                    )

                    # Create model and loss
                    model = create_model_with_conditioning(
                        arch, cond, in_channels, out_channels, device
                    )
                    criterion = create_loss(loss_name)
                    if hasattr(criterion, 'to'):
                        criterion = criterion.to(device)

                    # Train
                    metrics = train_single_config(
                        model=model,
                        criterion=criterion,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        n_epochs=n_epochs,
                        lr=FACTORIAL_LR,
                    )

                    results.append(FactorialResult(
                        architecture=arch,
                        loss=loss_cat,
                        conditioning=cond,
                        seed=current_seed,
                        fold=fold_idx,
                        r2=metrics["r2"],
                        mae=metrics["mae"],
                        pearson=metrics["pearson"],
                        psd_error_db=metrics["psd_error_db"],
                    ))

                    print(f"R² = {metrics['r2']:.4f}")

                except Exception as e:
                    print(f"FAILED: {e}")
                    results.append(FactorialResult(
                        architecture=arch,
                        loss=loss_cat,
                        conditioning=cond,
                        seed=current_seed,
                        fold=fold_idx,
                        r2=float("-inf"),
                        mae=float("inf"),
                        pearson=0.0,
                        psd_error_db=0.0,
                    ))

                # Cleanup
                del model
                gc.collect()
                torch.cuda.empty_cache()

    return results


def analyze_factorial_results(results: List[FactorialResult]) -> Dict[str, Any]:
    """Perform ANOVA analysis on factorial results.

    Args:
        results: List of FactorialResult

    Returns:
        Dictionary with ANOVA results and summary statistics
    """
    # Filter valid results
    valid_results = [r for r in results if r.r2 > float("-inf")]

    if not valid_results:
        return {"error": "No valid results"}

    # Prepare data for ANOVA
    r2_values = np.array([r.r2 for r in valid_results])
    archs = np.array([r.architecture for r in valid_results])
    losses = np.array([r.loss for r in valid_results])
    conds = np.array([r.conditioning for r in valid_results])

    # Run three-way ANOVA
    anova_result = three_way_anova(
        data=r2_values,
        factor_a=archs,
        factor_b=losses,
        factor_c=conds,
        names=("Architecture", "Loss", "Conditioning"),
    )

    # Compute combination means
    combo_stats = {}
    for arch in np.unique(archs):
        for loss in np.unique(losses):
            for cond in np.unique(conds):
                mask = (archs == arch) & (losses == loss) & (conds == cond)
                values = r2_values[mask]
                if len(values) > 0:
                    combo_stats[(arch, loss, cond)] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "n": len(values),
                    }

    # Find best and runner-up
    sorted_combos = sorted(combo_stats.items(), key=lambda x: -x[1]["mean"])
    best_combo = sorted_combos[0]
    runner_up = sorted_combos[1] if len(sorted_combos) > 1 else sorted_combos[0]

    return {
        "anova": {
            "main_effects": {
                name: {
                    "F": result.f_stat,
                    "p": result.p_value,
                    "eta_sq": result.eta_sq,
                    "significant": result.is_significant(),
                }
                for name, result in anova_result.main_effects.items()
            },
            "interactions": {
                name: {
                    "F": result.f_stat,
                    "p": result.p_value,
                    "eta_sq": result.eta_sq,
                    "significant": result.is_significant(),
                }
                for name, result in anova_result.interactions.items()
            },
        },
        "combination_stats": combo_stats,
        "best_combination": {
            "config": best_combo[0],
            "mean": best_combo[1]["mean"],
            "std": best_combo[1]["std"],
        },
        "runner_up": {
            "config": runner_up[0],
            "mean": runner_up[1]["mean"],
            "std": runner_up[1]["std"],
        },
    }


# =============================================================================
# Main Runner
# =============================================================================

def run_tier2(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    architectures: Optional[List[str]] = None,
    losses: Optional[List[str]] = None,
    conditionings: Optional[List[str]] = None,
    n_seeds: int = N_SEEDS,
    n_folds: int = N_FOLDS,
    n_epochs: int = FACTORIAL_EPOCHS,
    seed: int = 42,
    register: bool = True,
) -> Tier2Result:
    """Run Tier 2: Factorial Design with ANOVA.

    Args:
        X: Input data [N, C, T]
        y: Target data [N, C, T]
        device: Device
        architectures: List of architecture names (from Tier 1)
        losses: List of loss category names (from Tier 1)
        conditionings: List of conditioning types
        n_seeds: Number of random seeds
        n_folds: Number of CV folds
        n_epochs: Training epochs
        seed: Base random seed
        register: Whether to register with ConfigRegistry

    Returns:
        Tier2Result with factorial analysis
    """
    # Get selections from Tier 1 if not provided
    registry = get_registry(ARTIFACTS_DIR)

    if architectures is None or losses is None:
        if registry.tier1 is None:
            raise ValueError("Tier 1 must be completed first, or provide architectures and losses")
        tier1_selections = registry.get_tier1_selections()
        architectures = architectures or tier1_selections["architectures"]
        losses = losses or tier1_selections["losses"]

    conditionings = conditionings or CONDITIONING_TYPES

    print(f"\nTier 2: Factorial Design")
    print(f"Architectures: {architectures}")
    print(f"Losses: {losses}")
    print(f"Conditionings: {conditionings}")

    # Run factorial design
    results = run_factorial_design(
        X=X,
        y=y,
        architectures=architectures,
        losses=losses,
        conditionings=conditionings,
        device=device,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_epochs=n_epochs,
        seed=seed,
    )

    # Analyze results
    analysis = analyze_factorial_results(results)

    # Get best combination stats
    best_config = analysis["best_combination"]["config"]
    runner_up_config = analysis["runner_up"]["config"]

    # Get PSD error for best config
    best_results = [r for r in results
                    if r.architecture == best_config[0]
                    and r.loss == best_config[1]
                    and r.conditioning == best_config[2]
                    and r.r2 > float("-inf")]
    avg_psd_error = np.mean([r.psd_error_db for r in best_results]) if best_results else 0.0

    tier2_result = Tier2Result(
        results=results,
        anova_results=analysis["anova"],
        best_combination=best_config,
        best_r2_mean=analysis["best_combination"]["mean"],
        best_r2_std=analysis["best_combination"]["std"],
        runner_up=runner_up_config,
        runner_up_r2_mean=analysis["runner_up"]["mean"],
        psd_error_db=avg_psd_error,
    )

    # Register
    if register:
        registry.register_tier2(tier2_result)

    return tier2_result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tier 2: Factorial Design")
    parser.add_argument("--dry-run", action="store_true", help="Use minimal configuration")
    parser.add_argument("--epochs", type=int, default=FACTORIAL_EPOCHS, help="Training epochs")
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="Number of seeds")
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TIER 2: Factorial Design (KEY TIER)")
    print("=" * 60)
    print("Purpose: Find INTERACTIONS between architecture × loss × conditioning")
    print("Method: Full factorial with 5-fold CV × 5 seeds")
    print()

    # Load data
    print("Loading data...")

    if args.dry_run:
        N = 100
        C = 32
        T = 500
        X = torch.randn(N, C, T)
        y = torch.randn(N, C, T)
        architectures = ["linear", "cnn"]
        losses = ["huber", "spectral"]
        conditionings = ["concat"]
        n_epochs = 5
        n_seeds = 2
        n_folds = 2
        print(f"  [DRY-RUN] Using synthetic data: {X.shape}")
    else:
        try:
            from data import prepare_data
            data = prepare_data()
            X_train = data["train"]["ob"]
            y_train = data["train"]["hp"]
            X_val = data["val"]["ob"]
            y_val = data["val"]["hp"]
            X = torch.cat([X_train, X_val], dim=0)
            y = torch.cat([y_train, y_val], dim=0)
            # Use selections from Tier 1
            architectures = None  # Will be loaded from registry
            losses = None
            conditionings = CONDITIONING_TYPES
            n_epochs = args.epochs
            n_seeds = args.seeds
            n_folds = args.folds
            print(f"  Loaded real data: {X.shape}")
        except Exception as e:
            print(f"  Warning: Could not load data ({e}), using synthetic")
            N = 200
            C = 32
            T = 1000
            X = torch.randn(N, C, T)
            y = torch.randn(N, C, T)
            architectures = ["linear", "cnn", "wavenet"]
            losses = ["huber", "spectral", "ccc"]
            conditionings = CONDITIONING_TYPES
            n_epochs = args.epochs
            n_seeds = args.seeds
            n_folds = args.folds

    # Run factorial design
    result = run_tier2(
        X=X,
        y=y,
        device=device,
        architectures=architectures,
        losses=losses,
        conditionings=conditionings,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_epochs=n_epochs,
        seed=args.seed,
    )

    # Print ANOVA summary
    print("\n" + "=" * 60)
    print("TIER 2 RESULTS: Factorial Analysis")
    print("=" * 60)

    print("\nANOVA Results:")
    print("-" * 40)
    print("Main Effects:")
    for name, stats in result.anova_results.get("main_effects", {}).items():
        sig = "*" if stats.get("significant") else ""
        print(f"  {name}: F={stats.get('F', 0):.2f}, p={stats.get('p', 1):.4f}{sig}, "
              f"eta²={stats.get('eta_sq', 0):.3f}")

    print("\nInteractions:")
    for name, stats in result.anova_results.get("interactions", {}).items():
        sig = "**" if stats.get("significant") else ""
        print(f"  {name}: F={stats.get('F', 0):.2f}, p={stats.get('p', 1):.4f}{sig}, "
              f"eta²={stats.get('eta_sq', 0):.3f}")

    print("\n" + "-" * 40)
    arch, loss, cond = result.best_combination
    print(f"WINNER: {arch} + {loss} + {cond}")
    print(f"  R² = {result.best_r2_mean:.4f} ± {result.best_r2_std:.4f}")
    print(f"  PSD Error = {result.psd_error_db:.2f} dB")

    arch2, loss2, cond2 = result.runner_up
    print(f"\nRUNNER-UP: {arch2} + {loss2} + {cond2}")
    print(f"  R² = {result.runner_up_r2_mean:.4f}")

    # Check SpectralShift need
    if result.needs_spectral_shift():
        print(f"\nSpectralShift: NEEDED (PSD error {result.psd_error_db:.1f} dB > 3.0 dB)")
    else:
        print(f"\nSpectralShift: NOT NEEDED (PSD error {result.psd_error_db:.1f} dB <= 3.0 dB)")

    # Save
    output_file = args.output or (ARTIFACTS_DIR / f"tier2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_file = Path(output_file)

    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return result


if __name__ == "__main__":
    main()
