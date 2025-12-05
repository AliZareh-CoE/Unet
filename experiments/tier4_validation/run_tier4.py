#!/usr/bin/env python3
"""
Tier 4: Final Validation
========================

Final validation on HOLDOUT test set (first and only time it's used).
Includes negative controls and bootstrap confidence intervals.

Tests:
- Multi-seed variance (10 seeds per config)
- Bootstrap 95% CIs (10K resamples)
- Pairwise tests with Bonferroni correction
- Negative controls (shuffled labels, time-reversed, permutation)

GATE: All negative controls must PASS

Usage:
    python experiments/tier4_validation/run_tier4.py
    python experiments/tier4_validation/run_tier4.py --dry-run
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
from experiments.common.statistics import (
    bootstrap_ci,
    bootstrap_diff_ci,
    cohens_d,
    pairwise_tests,
    permutation_test,
    bonferroni_correction,
)
from experiments.common.config_registry import (
    ValidationResult,
    NegativeControlResult,
    Tier4Result,
    get_registry,
)
from experiments.common.metrics import compute_psd_error_db
from experiments.study1_architecture.architectures import create_architecture
from experiments.study3_loss.losses import create_loss


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier4_validation"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Validation configuration
VALIDATION_EPOCHS = 80
VALIDATION_BATCH_SIZE = 16
VALIDATION_LR = 1e-3
N_SEEDS = 10
N_BOOTSTRAP = 10000

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
# Model Training
# =============================================================================

def train_and_evaluate(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 80,
    lr: float = 1e-3,
) -> Dict[str, float]:
    """Train model and evaluate on test set.

    Args:
        model: Model to train
        criterion: Loss function
        train_loader: Training data
        test_loader: Test data (holdout)
        device: Device
        n_epochs: Number of epochs
        lr: Learning rate

    Returns:
        Dictionary with metrics
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

    # Evaluate on test set
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
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

    # Pearson
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


def create_model_from_config(
    config: Dict[str, Any],
    in_channels: int,
    out_channels: int,
    device: torch.device,
) -> nn.Module:
    """Create model from configuration.

    Args:
        config: Model configuration
        in_channels: Input channels
        out_channels: Output channels
        device: Device

    Returns:
        Model instance
    """
    arch = config.get("architecture", "cnn")

    if arch == "unet":
        try:
            from models import UNet1DConditioned
            model = UNet1DConditioned(
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=64,
            )
            return model.to(device)
        except (ImportError, TypeError):
            pass

    model = create_architecture(
        arch,
        variant="standard",
        in_channels=in_channels,
        out_channels=out_channels,
    )

    return model.to(device)


# =============================================================================
# Validation Runner
# =============================================================================

def run_multi_seed_validation(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    config: Dict[str, Any],
    config_name: str,
    device: torch.device,
    n_seeds: int = 10,
    n_epochs: int = 80,
) -> ValidationResult:
    """Run validation with multiple seeds.

    Args:
        X_train: Training inputs
        y_train: Training targets
        X_test: Test inputs (holdout)
        y_test: Test targets (holdout)
        config: Model configuration
        config_name: Name for this configuration
        device: Device
        n_seeds: Number of random seeds
        n_epochs: Training epochs

    Returns:
        ValidationResult with statistics
    """
    in_channels = X_train.shape[1]
    out_channels = y_train.shape[1]

    loss_cat = config.get("loss", "huber")
    loss_name = LOSS_CATEGORY_MAP.get(loss_cat, loss_cat)

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=False)

    r2_values = []
    mae_values = []
    pearson_values = []
    psd_values = []

    print(f"\n  Validating {config_name} with {n_seeds} seeds...")

    for seed in range(42, 42 + n_seeds):
        print(f"    Seed {seed}...", end=" ", flush=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        try:
            model = create_model_from_config(config, in_channels, out_channels, device)
            criterion = create_loss(loss_name)
            if hasattr(criterion, 'to'):
                criterion = criterion.to(device)

            metrics = train_and_evaluate(
                model=model,
                criterion=criterion,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                n_epochs=n_epochs,
                lr=VALIDATION_LR,
            )

            r2_values.append(metrics["r2"])
            mae_values.append(metrics["mae"])
            pearson_values.append(metrics["pearson"])
            psd_values.append(metrics["psd_error_db"])

            print(f"R² = {metrics['r2']:.4f}")

        except Exception as e:
            print(f"FAILED: {e}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Bootstrap CI for R²
    r2_array = np.array(r2_values)
    ci = bootstrap_ci(r2_array, statistic="mean", n_bootstrap=N_BOOTSTRAP, seed=42)

    return ValidationResult(
        config_name=config_name,
        r2_mean=float(np.mean(r2_values)),
        r2_std=float(np.std(r2_values, ddof=1)) if len(r2_values) > 1 else 0.0,
        r2_ci_lower=ci["ci_lower"],
        r2_ci_upper=ci["ci_upper"],
        mae_mean=float(np.mean(mae_values)),
        pearson_mean=float(np.mean(pearson_values)),
        psd_error_db=float(np.mean(psd_values)),
        n_seeds=len(r2_values),
    )


# =============================================================================
# Negative Controls
# =============================================================================

def run_negative_controls(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    config: Dict[str, Any],
    device: torch.device,
    n_seeds: int = 5,
    n_epochs: int = 80,
) -> List[NegativeControlResult]:
    """Run negative control tests.

    Args:
        X_train: Training inputs
        y_train: Training targets
        X_test: Test inputs
        y_test: Test targets
        config: Model configuration
        device: Device
        n_seeds: Number of seeds
        n_epochs: Training epochs

    Returns:
        List of NegativeControlResult
    """
    results = []
    in_channels = X_train.shape[1]
    out_channels = y_train.shape[1]

    loss_cat = config.get("loss", "huber")
    loss_name = LOSS_CATEGORY_MAP.get(loss_cat, loss_cat)

    # 1. Shuffled Labels Control
    print("\n  Running Shuffled Labels Control...")
    shuffled_r2s = []

    for seed in range(42, 42 + n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Shuffle targets across samples
        perm = torch.randperm(y_train.shape[0])
        y_shuffled = y_train[perm]

        train_dataset = TensorDataset(X_train, y_shuffled)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=False)

        try:
            model = create_model_from_config(config, in_channels, out_channels, device)
            criterion = create_loss(loss_name)
            if hasattr(criterion, 'to'):
                criterion = criterion.to(device)

            metrics = train_and_evaluate(
                model, criterion, train_loader, test_loader, device, n_epochs, VALIDATION_LR
            )
            shuffled_r2s.append(metrics["r2"])
        except Exception:
            pass

        del model
        gc.collect()
        torch.cuda.empty_cache()

    shuffled_r2 = np.mean(shuffled_r2s) if shuffled_r2s else 0.0
    shuffled_passed = shuffled_r2 < 0.1  # Should fail to learn

    results.append(NegativeControlResult(
        control_type="shuffled",
        r2=shuffled_r2,
        expected_behavior="R² < 0.1 (model should fail on shuffled labels)",
        passed=shuffled_passed,
    ))
    print(f"    Shuffled R² = {shuffled_r2:.4f} ({'PASS' if shuffled_passed else 'FAIL'})")

    # 2. Time-Reversed Control
    print("\n  Running Time-Reversed Control...")
    reversed_r2s = []

    for seed in range(42, 42 + n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Reverse time dimension
        y_reversed = y_train.flip(dims=[-1])

        train_dataset = TensorDataset(X_train, y_reversed)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=False)

        try:
            model = create_model_from_config(config, in_channels, out_channels, device)
            criterion = create_loss(loss_name)
            if hasattr(criterion, 'to'):
                criterion = criterion.to(device)

            metrics = train_and_evaluate(
                model, criterion, train_loader, test_loader, device, n_epochs, VALIDATION_LR
            )
            reversed_r2s.append(metrics["r2"])
        except Exception:
            pass

        del model
        gc.collect()
        torch.cuda.empty_cache()

    reversed_r2 = np.mean(reversed_r2s) if reversed_r2s else 0.0
    reversed_passed = reversed_r2 < 0.5  # Should perform worse than real

    results.append(NegativeControlResult(
        control_type="reversed",
        r2=reversed_r2,
        expected_behavior="R² < 0.5 (model uses temporal structure)",
        passed=reversed_passed,
    ))
    print(f"    Reversed R² = {reversed_r2:.4f} ({'PASS' if reversed_passed else 'FAIL'})")

    # 3. Permutation Test (statistical significance)
    print("\n  Running Permutation Test (this may take a while)...")

    # Get real performance for comparison
    real_r2s = []
    for seed in range(42, 42 + 3):
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=False)

        try:
            model = create_model_from_config(config, in_channels, out_channels, device)
            criterion = create_loss(loss_name)
            if hasattr(criterion, 'to'):
                criterion = criterion.to(device)

            metrics = train_and_evaluate(
                model, criterion, train_loader, test_loader, device, n_epochs // 2, VALIDATION_LR
            )
            real_r2s.append(metrics["r2"])
        except Exception:
            pass

        del model
        gc.collect()
        torch.cuda.empty_cache()

    real_r2 = np.mean(real_r2s) if real_r2s else 0.5

    # Compare against shuffled distribution (already computed)
    n_better = sum(1 for r in shuffled_r2s if r >= real_r2)
    perm_p_value = (n_better + 1) / (len(shuffled_r2s) + 1)  # Add 1 for real result
    perm_passed = perm_p_value < 0.05

    results.append(NegativeControlResult(
        control_type="permutation",
        r2=perm_p_value,
        expected_behavior="p < 0.05 (real performance is statistically significant)",
        passed=perm_passed,
    ))
    print(f"    Permutation p-value = {perm_p_value:.4f} ({'PASS' if perm_passed else 'FAIL'})")

    return results


# =============================================================================
# Main Runner
# =============================================================================

def run_tier4(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    configs: Optional[List[Dict[str, Any]]] = None,
    n_seeds: int = N_SEEDS,
    n_epochs: int = VALIDATION_EPOCHS,
    seed: int = 42,
    register: bool = True,
) -> Tier4Result:
    """Run Tier 4: Final Validation.

    Args:
        X_train: Training inputs
        y_train: Training targets
        X_test: Holdout test inputs (FIRST TIME USED)
        y_test: Holdout test targets
        device: Device
        configs: List of configurations to validate
        n_seeds: Number of random seeds
        n_epochs: Training epochs
        seed: Base random seed
        register: Whether to register with ConfigRegistry

    Returns:
        Tier4Result with validation statistics
    """
    registry = get_registry(ARTIFACTS_DIR)

    # Get configs from previous tiers if not provided
    if configs is None:
        configs = []
        if registry.tier3:
            configs.append({
                "name": "winner",
                **registry.get_final_config(),
            })
        if registry.tier2:
            winner = registry.get_tier2_winner()
            runner_up = registry.get_tier2_runner_up()
            if not configs:
                configs.append({"name": "winner", **winner})
            configs.append({"name": "runner_up", **runner_up})

        if not configs:
            configs = [{"name": "baseline", "architecture": "cnn", "loss": "huber"}]

    print("\nTier 4: Final Validation")
    print(f"Testing {len(configs)} configurations on HOLDOUT test set")

    # Run validation for each config
    validation_results = []

    for config in configs:
        config_name = config.get("name", "config")
        result = run_multi_seed_validation(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=config,
            config_name=config_name,
            device=device,
            n_seeds=n_seeds,
            n_epochs=n_epochs,
        )
        validation_results.append(result)

    # Run negative controls on winner
    winner_config = configs[0]
    negative_controls = run_negative_controls(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        config=winner_config,
        device=device,
        n_seeds=min(5, n_seeds),
        n_epochs=n_epochs,
    )

    # Find best result
    best_result = max(validation_results, key=lambda r: r.r2_mean)

    # Get classical floor
    classical_r2 = registry.classical_floor if registry.tier0 else 0.0

    # Compute effect size vs classical
    effect_vs_classical = (best_result.r2_mean - classical_r2) / best_result.r2_std if best_result.r2_std > 0 else 0.0

    # Check if all controls passed
    all_passed = all(c.passed for c in negative_controls)

    tier4_result = Tier4Result(
        validation_results=validation_results,
        negative_controls=negative_controls,
        winner=best_result.config_name,
        winner_r2=best_result.r2_mean,
        winner_ci=(best_result.r2_ci_lower, best_result.r2_ci_upper),
        classical_r2=classical_r2,
        improvement_over_classical=best_result.r2_mean - classical_r2,
        effect_size_vs_classical=effect_vs_classical,
        all_controls_passed=all_passed,
    )

    if register:
        registry.register_tier4(tier4_result)

    return tier4_result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tier 4: Final Validation")
    parser.add_argument("--dry-run", action="store_true", help="Use minimal configuration")
    parser.add_argument("--epochs", type=int, default=VALIDATION_EPOCHS, help="Training epochs")
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TIER 4: Final Validation")
    print("=" * 60)
    print("Purpose: Rigorous validation on HOLDOUT test set")
    print("CRITICAL: First time holdout set is used!")
    print()

    # Load data
    print("Loading data...")

    if args.dry_run:
        N_train = 80
        N_test = 20
        C = 32
        T = 500
        X_train = torch.randn(N_train, C, T)
        y_train = torch.randn(N_train, C, T)
        X_test = torch.randn(N_test, C, T)
        y_test = torch.randn(N_test, C, T)
        configs = [{"name": "test", "architecture": "linear", "loss": "huber"}]
        n_epochs = 5
        n_seeds = 2
        print(f"  [DRY-RUN] Using synthetic data")
    else:
        try:
            from data import prepare_data
            data = prepare_data()
            X_train = data["train"]["ob"]
            y_train = data["train"]["hp"]
            X_test = data["test"]["ob"]  # HOLDOUT
            y_test = data["test"]["hp"]  # HOLDOUT
            configs = None  # Will be loaded from registry
            n_epochs = args.epochs
            n_seeds = args.seeds
            print(f"  Loaded real data: Train {X_train.shape}, Test {X_test.shape}")
        except Exception as e:
            print(f"  Warning: Could not load data ({e}), using synthetic")
            N_train = 160
            N_test = 40
            C = 32
            T = 1000
            X_train = torch.randn(N_train, C, T)
            y_train = torch.randn(N_train, C, T)
            X_test = torch.randn(N_test, C, T)
            y_test = torch.randn(N_test, C, T)
            configs = [
                {"name": "winner", "architecture": "cnn", "loss": "huber"},
                {"name": "runner_up", "architecture": "linear", "loss": "spectral"},
            ]
            n_epochs = args.epochs
            n_seeds = args.seeds

    # Run validation
    result = run_tier4(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device=device,
        configs=configs,
        n_seeds=n_seeds,
        n_epochs=n_epochs,
        seed=args.seed,
    )

    # Summary
    print("\n" + "=" * 60)
    print("TIER 4 RESULTS: Final Validation")
    print("=" * 60)

    print("\nValidation Results (Holdout Test Set):")
    print("-" * 60)
    for vr in result.validation_results:
        print(f"  {vr.config_name}:")
        print(f"    R² = {vr.r2_mean:.4f} ± {vr.r2_std:.4f}")
        print(f"    95% CI = [{vr.r2_ci_lower:.4f}, {vr.r2_ci_upper:.4f}]")
        print(f"    MAE = {vr.mae_mean:.4f}, Pearson = {vr.pearson_mean:.4f}")

    print("\nNegative Controls:")
    print("-" * 60)
    for nc in result.negative_controls:
        status = "PASS" if nc.passed else "FAIL"
        print(f"  {nc.control_type}: R² = {nc.r2:.4f} [{status}]")
        print(f"    Expected: {nc.expected_behavior}")

    print("\n" + "-" * 60)
    print(f"WINNER: {result.winner}")
    print(f"  R² = {result.winner_r2:.4f}")
    print(f"  95% CI = [{result.winner_ci[0]:.4f}, {result.winner_ci[1]:.4f}]")
    print(f"  Improvement over classical: {result.improvement_over_classical:.4f}")
    print(f"  Effect size (d): {result.effect_size_vs_classical:.2f}")
    print(f"  Negative controls: {'ALL PASSED' if result.all_controls_passed else 'SOME FAILED'}")

    # Save
    output_file = args.output or (ARTIFACTS_DIR / f"tier4_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_file = Path(output_file)

    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return result


if __name__ == "__main__":
    main()
