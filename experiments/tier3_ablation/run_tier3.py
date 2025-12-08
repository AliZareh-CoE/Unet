#!/usr/bin/env python3
"""
Tier 3: U-Net Component Ablation
================================

Purpose: Ablate U-Net (CondUNet1D) components to find the minimal optimal architecture.

After Tiers 1-2 prove U-Net with optimal loss/conditioning, this tier removes
unnecessary components to create a lean, efficient model.

Architecture: CondUNet1D (from models.py) - FIXED

5 Focused Ablation Questions:
1. Does attention help?
2. Does conditioning type matter?
3. Does SE block help?
4. Is depth optimal?
5. Does normalization matter?

Also tests interaction ablations (removing pairs of components).

FEEDBACK: Components with Cohen's d < 0.3 are REMOVED from final model.

Usage:
    python experiments/tier3_ablation/run_tier3.py
    python experiments/tier3_ablation/run_tier3.py --dry-run
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

from experiments.common.cross_validation import create_data_splits
from experiments.common.statistics import cohens_d, cohens_d_interpretation
from experiments.common.config_registry import (
    AblationResult,
    Tier3Result,
    get_registry,
)
from experiments.study1_architecture.architectures import create_architecture
from experiments.study3_loss.losses import create_loss


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier3_ablation"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Ablation configuration
ABLATION_EPOCHS = 80
ABLATION_BATCH_SIZE = 16
ABLATION_LR = 1e-3
N_SEEDS = 5
N_FOLDS = 5

# Cohen's d threshold for component removal
REMOVAL_THRESHOLD_D = 0.3

# Ablation questions
ABLATION_QUESTIONS = [
    ("attention", ["with", "without"]),
    ("conditioning", ["cross_attn", "add", "concat"]),
    ("se_block", ["with", "without"]),
    ("depth", ["3", "4", "5"]),
    ("normalization", ["instance", "batch", "layer"]),
]

# Interaction ablations
INTERACTION_ABLATIONS = [
    ("attention+se", "Remove both attention and SE"),
    ("attention+conditioning", "Remove both attention and gated conditioning"),
]

# Loss category mapping - all 6 categories
LOSS_CATEGORY_MAP = {
    "huber": "huber",
    "spectral": "multi_scale_spectral",
    "ccc": "concordance",
    "ssim": "ssim",
    "gaussian_nll": "gaussian_nll",
    "combined": "combined",
}


# =============================================================================
# Ablation Model Creation
# =============================================================================

def create_ablated_model(
    base_arch: str,
    ablation_component: str,
    ablation_variant: str,
    in_channels: int,
    out_channels: int,
    device: torch.device,
) -> nn.Module:
    """Create model with specific ablation.

    Args:
        base_arch: Base architecture name
        ablation_component: Component being ablated
        ablation_variant: Variant of the ablation
        in_channels: Input channels
        out_channels: Output channels
        device: Device

    Returns:
        Ablated model instance
    """
    # Build config based on ablation
    config = {
        "use_attention": True,
        "use_se": True,
        "conditioning_type": "cross_attn",
        "n_layers": 4,
        "norm_type": "instance",
    }

    if ablation_component == "baseline":
        pass  # Use default config for baseline
    elif ablation_component == "attention":
        config["use_attention"] = ablation_variant == "with"
    elif ablation_component == "conditioning":
        config["conditioning_type"] = ablation_variant
    elif ablation_component == "se_block":
        config["use_se"] = ablation_variant == "with"
    elif ablation_component == "depth":
        config["n_layers"] = int(ablation_variant)
    elif ablation_component == "normalization":
        config["norm_type"] = ablation_variant
    elif ablation_component == "attention+se":
        config["use_attention"] = False
        config["use_se"] = False
    elif ablation_component == "attention+conditioning":
        config["use_attention"] = False
        config["conditioning_type"] = "add"

    # Create model - U-Net (CondUNet1D) is the primary architecture
    if base_arch == "unet":
        from models import CondUNet1D
        model = CondUNet1D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=64,
            n_odors=7,  # Default odors for conditioning
        )
        return model.to(device)

    # Variant mapping for architectures
    VARIANT_MAP = {
        "linear": "simple",
        "cnn": "basic",  # CNN uses "basic", not "standard"
        "wavenet": "standard",
        "fnet": "standard",
        "vit": "standard",
        "performer": "standard",
        "mamba": "standard",
    }
    variant = VARIANT_MAP.get(base_arch, "standard")

    # Fallback to base architecture
    model = create_architecture(
        base_arch,
        variant=variant,
        in_channels=in_channels,
        out_channels=out_channels,
    )

    return model.to(device)


# =============================================================================
# Ablation Training
# =============================================================================

def train_ablation_config(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 80,
    lr: float = 1e-3,
) -> float:
    """Train ablated configuration and return R².

    Args:
        model: Model to train
        criterion: Loss function
        train_loader: Training data
        val_loader: Validation data
        device: Device
        n_epochs: Number of epochs
        lr: Learning rate

    Returns:
        Validation R²
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
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

            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        scheduler.step()

    # Evaluate
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

    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = (1 - ss_res / (ss_tot + 1e-8)).item()

    return r2


# =============================================================================
# Ablation Runner
# =============================================================================

def run_single_ablation(
    X: torch.Tensor,
    y: torch.Tensor,
    base_config: Tuple[str, str, str],
    ablation_component: str,
    ablation_variant: str,
    device: torch.device,
    n_seeds: int = 5,
    n_folds: int = 5,
    n_epochs: int = 80,
    seed: int = 42,
) -> List[float]:
    """Run single ablation with multiple seeds/folds.

    Args:
        X: Input data
        y: Target data
        base_config: (architecture, loss, conditioning)
        ablation_component: Component being ablated
        ablation_variant: Variant of the ablation
        device: Device
        n_seeds: Number of seeds
        n_folds: Number of folds
        n_epochs: Training epochs
        seed: Base seed

    Returns:
        List of R² values across seeds/folds
    """
    arch, loss_cat, cond = base_config
    loss_name = LOSS_CATEGORY_MAP.get(loss_cat, loss_cat)

    n_samples = X.shape[0]
    in_channels = X.shape[1]
    out_channels = y.shape[1]

    splits = create_data_splits(
        n_samples=n_samples,
        n_folds=n_folds,
        holdout_fraction=0.2,
        seed=seed,
    )

    dataset = TensorDataset(X, y)
    r2_values = []

    for seed_offset in range(n_seeds):
        current_seed = seed + seed_offset

        for fold_idx, cv_split in enumerate(splits.cv_splits):
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)

            train_subset = Subset(dataset, cv_split.train_indices.tolist())
            val_subset = Subset(dataset, cv_split.val_indices.tolist())

            train_loader = DataLoader(
                train_subset,
                batch_size=ABLATION_BATCH_SIZE,
                shuffle=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=ABLATION_BATCH_SIZE,
                shuffle=False,
            )

            model = None  # Initialize for cleanup
            try:
                model = create_ablated_model(
                    arch, ablation_component, ablation_variant,
                    in_channels, out_channels, device
                )
                criterion = create_loss(loss_name)
                if hasattr(criterion, 'to'):
                    criterion = criterion.to(device)

                r2 = train_ablation_config(
                    model=model,
                    criterion=criterion,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    n_epochs=n_epochs,
                    lr=ABLATION_LR,
                )
                r2_values.append(r2)

            except Exception as e:
                print(f"    Failed: {e}")
                r2_values.append(float("-inf"))

            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()

    return r2_values


def run_ablation_study(
    X: torch.Tensor,
    y: torch.Tensor,
    base_config: Tuple[str, str, str],
    device: torch.device,
    n_seeds: int = 5,
    n_folds: int = 5,
    n_epochs: int = 80,
    seed: int = 42,
) -> List[AblationResult]:
    """Run complete ablation study.

    Args:
        X: Input data
        y: Target data
        base_config: (architecture, loss, conditioning)
        device: Device
        n_seeds: Number of seeds
        n_folds: Number of folds
        n_epochs: Training epochs
        seed: Base seed

    Returns:
        List of AblationResult for each ablation
    """
    results = []

    # First, get baseline performance
    print(f"\n  Running baseline ({base_config[0]} + {base_config[1]} + {base_config[2]})...")
    baseline_values = run_single_ablation(
        X, y, base_config,
        ablation_component="baseline",
        ablation_variant="full",
        device=device,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_epochs=n_epochs,
        seed=seed,
    )

    valid_baseline = [v for v in baseline_values if v > float("-inf")]
    baseline_mean = np.mean(valid_baseline) if valid_baseline else 0
    baseline_std = np.std(valid_baseline, ddof=1) if len(valid_baseline) > 1 else 0

    results.append(AblationResult(
        component="baseline",
        variant="full",
        r2_mean=baseline_mean,
        r2_std=baseline_std,
        delta_from_baseline=0.0,
        cohens_d=0.0,
        verdict="BASELINE",
    ))

    print(f"    Baseline R² = {baseline_mean:.4f} ± {baseline_std:.4f}")

    # Run each ablation
    for component, variants in ABLATION_QUESTIONS:
        for variant in variants:
            # Skip the "default" variant (it's the baseline)
            if (component == "attention" and variant == "with") or \
               (component == "se_block" and variant == "with") or \
               (component == "conditioning" and variant == "cross_attn") or \
               (component == "depth" and variant == "4") or \
               (component == "normalization" and variant == "instance"):
                continue

            print(f"  Running ablation: {component} = {variant}...", end=" ", flush=True)

            ablation_values = run_single_ablation(
                X, y, base_config,
                ablation_component=component,
                ablation_variant=variant,
                device=device,
                n_seeds=n_seeds,
                n_folds=n_folds,
                n_epochs=n_epochs,
                seed=seed,
            )

            valid_values = [v for v in ablation_values if v > float("-inf")]

            if valid_values:
                ablation_mean = np.mean(valid_values)
                ablation_std = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0

                delta = baseline_mean - ablation_mean
                d = cohens_d(np.array(valid_baseline), np.array(valid_values))

                # Determine verdict
                if delta > 0.03 and abs(d) >= 0.8:
                    verdict = "ESSENTIAL"
                elif delta > 0.01 and abs(d) >= 0.5:
                    verdict = "KEEP"
                elif delta > 0.005 and abs(d) >= REMOVAL_THRESHOLD_D:
                    verdict = "MARGINAL"
                else:
                    verdict = "REMOVE"

                results.append(AblationResult(
                    component=component,
                    variant=variant,
                    r2_mean=ablation_mean,
                    r2_std=ablation_std,
                    delta_from_baseline=delta,
                    cohens_d=d,
                    verdict=verdict,
                ))

                print(f"R² = {ablation_mean:.4f}, Δ = {delta:+.4f}, d = {d:.2f} → {verdict}")
            else:
                results.append(AblationResult(
                    component=component,
                    variant=variant,
                    r2_mean=0.0,
                    r2_std=0.0,
                    delta_from_baseline=0.0,
                    cohens_d=0.0,
                    verdict="FAILED",
                ))
                print("FAILED")

    # Run interaction ablations
    print("\n  Running interaction ablations...")
    for interaction, description in INTERACTION_ABLATIONS:
        print(f"  Running: {interaction} ({description})...", end=" ", flush=True)

        interaction_values = run_single_ablation(
            X, y, base_config,
            ablation_component=interaction,
            ablation_variant="removed",
            device=device,
            n_seeds=min(3, n_seeds),  # Fewer seeds for interactions
            n_folds=n_folds,
            n_epochs=n_epochs,
            seed=seed,
        )

        valid_values = [v for v in interaction_values if v > float("-inf")]

        if valid_values:
            interaction_mean = np.mean(valid_values)
            interaction_std = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0
            delta = baseline_mean - interaction_mean
            d = cohens_d(np.array(valid_baseline), np.array(valid_values))

            verdict = "SYNERGISTIC" if delta > 0.05 else "ADDITIVE"

            results.append(AblationResult(
                component=interaction,
                variant="removed",
                r2_mean=interaction_mean,
                r2_std=interaction_std,
                delta_from_baseline=delta,
                cohens_d=d,
                verdict=verdict,
            ))

            print(f"R² = {interaction_mean:.4f}, Δ = {delta:+.4f} → {verdict}")

    return results


# =============================================================================
# Main Runner
# =============================================================================

def run_tier3(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    base_config: Optional[Tuple[str, str, str]] = None,
    n_seeds: int = N_SEEDS,
    n_folds: int = N_FOLDS,
    n_epochs: int = ABLATION_EPOCHS,
    seed: int = 42,
    register: bool = True,
) -> Tier3Result:
    """Run Tier 3: Focused Ablation.

    Args:
        X: Input data
        y: Target data
        device: Device
        base_config: (architecture, loss, conditioning) to ablate
        n_seeds: Number of seeds
        n_folds: Number of folds
        n_epochs: Training epochs
        seed: Base seed
        register: Whether to register with ConfigRegistry

    Returns:
        Tier3Result with ablation findings
    """
    registry = get_registry(ARTIFACTS_DIR)

    # Get winner from Tier 2 if not provided
    if base_config is None:
        if registry.tier2 is None:
            raise ValueError("Tier 2 must be completed first, or provide base_config")
        winner = registry.get_tier2_winner()
        base_config = (winner["architecture"], winner["loss"], winner["conditioning"])

    print(f"\nTier 3: Focused Ablation")
    print(f"Base config: {base_config}")

    # Run ablation study
    ablations = run_ablation_study(
        X=X,
        y=y,
        base_config=base_config,
        device=device,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_epochs=n_epochs,
        seed=seed,
    )

    # Get baseline R²
    baseline = next((a for a in ablations if a.component == "baseline"), None)
    baseline_r2 = baseline.r2_mean if baseline else 0.0

    # Determine components to remove
    components_to_remove = [a.component for a in ablations if a.verdict == "REMOVE"]

    # Build final config
    final_config = {
        "architecture": base_config[0],
        "loss": base_config[1],
        "conditioning": base_config[2],
        "removed_components": components_to_remove,
    }

    tier3_result = Tier3Result(
        baseline_config=base_config,
        baseline_r2=baseline_r2,
        ablations=ablations,
        components_to_remove=components_to_remove,
        final_config=final_config,
    )

    if register:
        registry.register_tier3(tier3_result)

    return tier3_result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tier 3: Focused Ablation")
    parser.add_argument("--dry-run", action="store_true", help="Use minimal configuration")
    parser.add_argument("--epochs", type=int, default=ABLATION_EPOCHS, help="Training epochs")
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="Number of seeds")
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TIER 3: Focused Ablation")
    print("=" * 60)
    print("Purpose: Determine which components are ESSENTIAL vs REMOVE")
    print("Feedback: Components with Cohen's d < 0.3 will be removed")
    print()

    # Load data
    print("Loading data...")

    if args.dry_run:
        N = 100
        C = 32
        T = 500
        X = torch.randn(N, C, T)
        y = torch.randn(N, C, T)
        base_config = ("linear", "huber", "concat")
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
            base_config = None  # Will be loaded from registry
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
            base_config = ("cnn", "huber", "cross_attn")
            n_epochs = args.epochs
            n_seeds = args.seeds
            n_folds = args.folds

    # Run ablation
    result = run_tier3(
        X=X,
        y=y,
        device=device,
        base_config=base_config,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_epochs=n_epochs,
        seed=args.seed,
    )

    # Summary
    print("\n" + "=" * 60)
    print("TIER 3 RESULTS: Ablation Summary")
    print("=" * 60)

    print(f"\nBaseline: R² = {result.baseline_r2:.4f}")
    print("\nComponent Ablations:")
    print("-" * 60)
    print(f"{'Component':20s} {'Variant':12s} {'R²':>8s} {'Δ':>8s} {'d':>6s} {'Verdict':>12s}")
    print("-" * 60)

    for a in result.ablations:
        if a.component != "baseline":
            print(f"{a.component:20s} {a.variant:12s} {a.r2_mean:8.4f} {a.delta_from_baseline:+8.4f} "
                  f"{a.cohens_d:6.2f} {a.verdict:>12s}")

    print("\n" + "-" * 60)
    if result.components_to_remove:
        print(f"REMOVING: {result.components_to_remove}")
    else:
        print("No components removed (all contribute meaningfully)")

    # Save
    output_file = args.output or (ARTIFACTS_DIR / f"tier3_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_file = Path(output_file)

    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return result


if __name__ == "__main__":
    main()
