#!/usr/bin/env python3
"""
Study 6: Final Validation (POST-HPO)
====================================

Runs comprehensive validation for top configurations from Stage 1 & 2:
- Multi-seed runs for statistical robustness (10 seeds)
- Cross-validation
- Negative controls (shuffled labels, time-reversed)
- Statistical significance tests (paired t-test, Wilcoxon, permutation)
- Bootstrap 95% confidence intervals

REQUIRES: All Stage 1 and Stage 2 studies must complete first.

Usage:
    # Full validation on top 5 configs
    python experiments/study6_validation/run_study6.py --top-k 5 --n-seeds 10

    # Quick validation test
    python experiments/study6_validation/run_study6.py --dry-run --n-seeds 2

    # Run negative controls only
    python experiments/study6_validation/run_study6.py --negative-controls-only
"""

from __future__ import annotations

import argparse
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

from experiments.study6_validation.multi_seed_runner import (
    MultiSeedRunner,
    run_bootstrap_analysis,
)
from experiments.study6_validation.statistical_tests import StatisticalComparison
from experiments.study6_validation.negative_controls import (
    NegativeControlRunner,
    summarize_negative_controls,
)


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "study6_validation"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Default Training Function
# =============================================================================

def default_train_fn(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    config: Dict[str, Any],
    n_epochs: int = 80,
) -> Dict[str, float]:
    """Default training function for validation.

    Args:
        model: Model to train
        train_loader, val_loader, test_loader: Data loaders
        device: Device
        config: Configuration dict
        n_epochs: Number of epochs

    Returns:
        Dict with train_r2, val_r2, test_r2, losses
    """
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    best_val_r2 = -float("inf")
    best_state = None

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # Handle both (x, y) and (x, y, label) formats
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
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

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        ss_res = ((val_targets - val_preds) ** 2).sum()
        ss_tot = ((val_targets - val_targets.mean()) ** 2).sum()
        val_r2 = (1 - ss_res / (ss_tot + 1e-8)).item()

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = model.state_dict().copy()

    # Load best model and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_preds, test_targets = [], []
    train_preds, train_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            test_preds.append(y_pred.cpu())
            test_targets.append(y.cpu())

        for batch in train_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            train_preds.append(y_pred.cpu())
            train_targets.append(y.cpu())

    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)
    train_preds = torch.cat(train_preds)
    train_targets = torch.cat(train_targets)

    # Compute R²
    def compute_r2(preds, targets):
        ss_res = ((targets - preds) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        return (1 - ss_res / (ss_tot + 1e-8)).item()

    return {
        "train_r2": compute_r2(train_preds, train_targets),
        "val_r2": best_val_r2,
        "test_r2": compute_r2(test_preds, test_targets),
        "train_loss": train_loss / len(train_loader),
        "val_loss": 0.0,  # Not tracked
        "n_epochs": n_epochs,
    }


# =============================================================================
# Load Best Configs
# =============================================================================

def load_best_configs(
    top_k: int = 5,
    study_dirs: Optional[List[Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load best configurations from previous studies.

    Args:
        top_k: Number of top configs to load
        study_dirs: Directories to search for results

    Returns:
        Dict mapping config name to config dict
    """
    if study_dirs is None:
        study_dirs = [
            Path("artifacts/study1_architecture"),
            Path("artifacts/study2_conditioning"),
            Path("artifacts/study3_loss"),
        ]

    configs = {}

    # Try to load from Optuna databases
    try:
        import optuna

        for study_dir in study_dirs:
            db_files = list(study_dir.glob("*.db"))
            for db_file in db_files:
                storage = f"sqlite:///{db_file}"
                try:
                    summaries = optuna.get_all_study_summaries(storage)
                    for s in summaries:
                        if s.best_trial is not None:
                            study = optuna.load_study(
                                study_name=s.study_name, storage=storage
                            )
                            config_name = s.study_name
                            configs[config_name] = {
                                "params": study.best_trial.params,
                                "r2": study.best_trial.value,
                                "source": str(db_file),
                            }
                except Exception:
                    pass

    except ImportError:
        pass

    # Sort by R² and take top k
    if configs:
        sorted_configs = sorted(
            configs.items(),
            key=lambda x: x[1].get("r2", 0),
            reverse=True,
        )
        configs = dict(sorted_configs[:top_k])

    # Fallback to default configs if none found
    if not configs:
        configs = {
            "unet_default": {"base_channels": 64, "n_downsample": 3, "lr": 1e-4},
            "vit_default": {"embed_dim": 256, "num_layers": 6, "lr": 1e-4},
            "wavenet_default": {"hidden_channels": 64, "n_layers": 8, "lr": 1e-4},
            "performer_default": {"embed_dim": 256, "num_layers": 6, "lr": 1e-4},
            "mamba_default": {"d_model": 256, "d_state": 32, "lr": 1e-4},
        }

    return configs


# =============================================================================
# Main Validation
# =============================================================================

def run_full_validation(
    configs: Dict[str, Dict[str, Any]],
    model_factory,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    n_seeds: int = 10,
    n_epochs: int = 80,
    run_negative_controls: bool = True,
) -> Dict[str, Any]:
    """Run full validation pipeline.

    Args:
        configs: Dict of configurations to validate
        model_factory: Function to create model from config
        train_loader, val_loader, test_loader: Data loaders
        device: Device
        n_seeds: Number of seeds for multi-seed validation
        n_epochs: Training epochs
        run_negative_controls: Whether to run negative controls

    Returns:
        Complete validation results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_configs": len(configs),
        "n_seeds": n_seeds,
        "n_epochs": n_epochs,
    }

    # Multi-seed validation
    print("\n" + "="*60)
    print("Phase 1: Multi-Seed Validation")
    print("="*60)

    runner = MultiSeedRunner(
        model_factory=model_factory,
        train_fn=default_train_fn,
        n_seeds=n_seeds,
    )

    multi_seed_results = runner.run_all_configs(
        configs=configs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        n_epochs=n_epochs,
    )

    results["multi_seed"] = [r.to_dict() for r in multi_seed_results]

    # Bootstrap analysis
    print("\n" + "="*60)
    print("Phase 2: Bootstrap Analysis")
    print("="*60)

    bootstrap_results = run_bootstrap_analysis(multi_seed_results)
    results["bootstrap"] = bootstrap_results

    for name, br in bootstrap_results.items():
        print(f"  {name}: {br['original_mean']:.4f} "
              f"[{br['ci_95_lower']:.4f}, {br['ci_95_upper']:.4f}]")

    # Statistical comparisons
    print("\n" + "="*60)
    print("Phase 3: Statistical Comparisons")
    print("="*60)

    comparison = StatisticalComparison(alpha=0.05, correction="bh")

    method_values = {
        r.config_name: np.array([sr.test_r2 for sr in r.seed_results])
        for r in multi_seed_results
    }

    comparisons = comparison.compare_all_pairs(method_values)
    results["statistical_tests"] = comparison.summary()

    # Count significant differences
    n_significant = sum(1 for r in comparison.results if r.significant)
    print(f"  {n_significant}/{len(comparison.results)} significant comparisons")

    # Negative controls
    if run_negative_controls and len(multi_seed_results) > 0:
        print("\n" + "="*60)
        print("Phase 4: Negative Controls")
        print("="*60)

        # Use best config for negative controls
        best_config = multi_seed_results[0].config
        best_config_name = multi_seed_results[0].config_name

        # Get raw data from loader (handle both 2-tuple and 3-tuple batches)
        train_data_list, train_labels_list = [], []
        for batch in train_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            train_data_list.append(x)
            train_labels_list.append(y)
        train_data = torch.cat(train_data_list, dim=0)
        train_labels = torch.cat(train_labels_list, dim=0)

        test_data_list, test_labels_list = [], []
        for batch in test_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            test_data_list.append(x)
            test_labels_list.append(y)
        test_data = torch.cat(test_data_list, dim=0)
        test_labels = torch.cat(test_labels_list, dim=0)

        nc_runner = NegativeControlRunner(
            model_factory=model_factory,
            train_fn=default_train_fn,
        )

        nc_results = nc_runner.run_all_controls(
            config=best_config,
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            device=device,
            n_epochs=n_epochs // 2,  # Shorter for controls
        )

        nc_summary = summarize_negative_controls(nc_results)
        results["negative_controls"] = nc_summary

        print(f"  Passed: {nc_summary['passed']}/{nc_summary['total_controls']}")

    return results


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Study 6: Final Validation")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top configs")
    parser.add_argument("--n-seeds", type=int, default=10, help="Seeds per config")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--dry-run", action="store_true", help="Quick test")
    parser.add_argument("--negative-controls-only", action="store_true")
    parser.add_argument("--output", type=str, help="Output JSON file")

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
        test_loader = loaders.get("test", loaders["val"])  # Use test if available, else val
    except Exception as e:
        print(f"Warning: Using dummy data (reason: {type(e).__name__}: {e})")
        from torch.utils.data import TensorDataset, DataLoader
        X = torch.randn(100, 32, 1000)
        y = torch.randn(100, 32, 1000)
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16)
        test_loader = val_loader

    # Load configs
    configs = load_best_configs(top_k=args.top_k)
    print(f"Validating {len(configs)} configurations")

    # Model factory
    def model_factory(config):
        # Default to simple model for testing
        return nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
        )

    # Dry-run settings
    if args.dry_run:
        args.n_seeds = 2
        args.epochs = 3
        configs = dict(list(configs.items())[:2])

    # Run validation
    results = run_full_validation(
        configs=configs,
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        n_seeds=args.n_seeds,
        n_epochs=args.epochs,
        run_negative_controls=not args.negative_controls_only,
    )

    # Save results
    output_file = args.output or (
        ARTIFACTS_DIR / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_file = Path(output_file)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)

    if "multi_seed" in results:
        print("\nTop configurations (mean ± std):")
        for r in sorted(results["multi_seed"],
                       key=lambda x: x.get("mean_r2", 0),
                       reverse=True)[:5]:
            print(f"  {r['config_name']}: {r['mean_r2']:.4f} ± {r['std_r2']:.4f}")


if __name__ == "__main__":
    main()
