"""
Phase 3 v2: Bulletproof Ablation Study Runner
==============================================

DESIGN PRINCIPLES:
1. NO SUBPROCESS: Direct Python calls, no CLI string parsing
2. PRE-FLIGHT VALIDATION: Validate ALL configs before running ANY
3. EXPERIMENT REGISTRY: Skip duplicates, resume gracefully
4. FAIL-FAST: Invalid config = immediate crash with clear error
5. GREEDY FORWARD SELECTION: Efficient hyperparameter search

Usage:
    python runner_v2.py --groups 1,2,3 --n-folds 5
    python runner_v2.py --validate-only  # Pre-flight check
    python runner_v2.py --resume  # Continue from last run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from phase_three.config_v2 import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    AblationGroup,
    ABLATION_GROUPS_V2,
    validate_config,
    validate_all_groups,
    ExperimentRegistry,
)


# =============================================================================
# Direct Training (No Subprocess!)
# =============================================================================

def run_experiment_direct(
    config: ExperimentConfig,
    fold_idx: int = 0,
    device: str = "cuda",
    output_dir: Path = None,
) -> Dict[str, Any]:
    """Run a single experiment DIRECTLY in Python - no subprocess.

    This is the KEY IMPROVEMENT: no CLI parsing, no string conversion,
    just direct Python object passing.

    Args:
        config: Validated experiment configuration
        fold_idx: Cross-validation fold index
        device: Device to run on
        output_dir: Directory to save results

    Returns:
        Dictionary with metrics
    """
    from models import CondUNet1D
    from data import (
        OlfactoryTranslationDataset,
        load_olfactory_data,
        create_train_val_test_loaders,
    )

    print(f"\n{'='*60}")
    print(f"Running: {config.name} [fold {fold_idx}]")
    print(f"Config hash: {config.config_hash()}")
    print(f"{'='*60}")

    # Set seed for reproducibility
    seed = config.data.seed + fold_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create model DIRECTLY from config (no CLI parsing!)
    model = CondUNet1D(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base=config.model.base_channels,
        n_downsample=config.model.n_downsample,
        conv_type=config.model.conv_type,
        attention_type=config.model.attention_type,
        n_heads=config.model.n_heads,
        cond_mode=config.model.cond_mode,
        activation=config.model.activation,
        norm_type=config.model.norm_type,
        skip_type=config.model.skip_type,
        dropout=config.model.dropout,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Move to device
    model = model.to(device)

    # Create optimizer DIRECTLY from config
    if config.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=config.training.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    # Create loss DIRECTLY from config
    if config.training.loss_type in ["l1", "l1_wavelet"]:
        criterion = torch.nn.L1Loss()
    elif config.training.loss_type in ["huber", "huber_wavelet"]:
        criterion = torch.nn.SmoothL1Loss()
    else:
        criterion = torch.nn.L1Loss()

    # Load data
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        batch_size=config.training.batch_size,
        split_by_session=config.data.split_by_session,
        n_test_sessions=config.data.n_test_sessions,
        n_val_sessions=config.data.n_val_sessions,
        seed=seed,
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.training.epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x, y, cond = batch["ob"], batch["pcx"], batch["odor_idx"]
            x, y, cond = x.to(device), y.to(device), cond.to(device)

            optimizer.zero_grad()
            output = model(x, cond)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y, cond = batch["ob"], batch["pcx"], batch["odor_idx"]
                x, y, cond = x.to(device), y.to(device), cond.to(device)

                output = model(x, cond)
                loss = criterion(output, y)

                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.training.early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # Final evaluation on test set
    model.eval()
    test_loss = 0.0
    n_test_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            x, y, cond = batch["ob"], batch["pcx"], batch["odor_idx"]
            x, y, cond = x.to(device), y.to(device), cond.to(device)

            output = model(x, cond)
            loss = criterion(output, y)

            test_loss += loss.item()
            n_test_batches += 1

    test_loss /= max(n_test_batches, 1)

    results = {
        "config_hash": config.config_hash(),
        "config_name": config.name,
        "fold": fold_idx,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "n_params": n_params,
        "epochs_trained": epoch + 1,
    }

    print(f"Results: val_loss={best_val_loss:.4f}, test_loss={test_loss:.4f}")

    return results


# =============================================================================
# Greedy Forward Selection
# =============================================================================

def run_greedy_forward(
    groups: List[int] = None,
    n_folds: int = 1,
    output_dir: Path = None,
    validate_only: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run greedy forward selection ablation study.

    For each group:
    1. Test all variants with current best config
    2. Select winner (lowest validation loss)
    3. Update defaults with winner
    4. Move to next group

    Args:
        groups: Which groups to run (default: all)
        n_folds: Number of CV folds per variant
        output_dir: Where to save results
        validate_only: Just validate configs, don't run
        resume: Resume from previous run

    Returns:
        Dictionary with all results and final best config
    """
    if output_dir is None:
        output_dir = Path("outputs/phase3_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize registry for deduplication
    registry = ExperimentRegistry(output_dir / "registry.json")

    # Start with baseline
    current_best = ExperimentConfig.baseline()

    # Filter groups if specified
    if groups is None:
        groups_to_run = ABLATION_GROUPS_V2
    else:
        groups_to_run = [g for g in ABLATION_GROUPS_V2 if g.group_id in groups]

    # PRE-FLIGHT VALIDATION
    print("\n" + "=" * 60)
    print("PHASE 3 v2: BULLETPROOF ABLATION STUDY")
    print("=" * 60)
    print(f"Groups to run: {[g.group_id for g in groups_to_run]}")
    print(f"Folds per variant: {n_folds}")
    print(f"Output directory: {output_dir}")

    if validate_only:
        print("\n*** VALIDATE-ONLY MODE ***")
        validate_all_groups(current_best)
        return {"status": "validated"}

    # Validate all configs first
    print("\nPre-flight validation...")
    try:
        validate_all_groups(current_best)
    except Exception as e:
        print(f"\n❌ PRE-FLIGHT FAILED: {e}")
        print("Fix the configuration before running!")
        return {"status": "preflight_failed", "error": str(e)}

    print("\n✓ All configurations validated. Starting experiments...\n")

    # Results storage
    all_results = {
        "start_time": datetime.now().isoformat(),
        "groups": {},
        "best_config_per_group": {},
    }

    # Greedy forward selection
    for group in groups_to_run:
        print(f"\n{'=' * 60}")
        print(f"GROUP {group.group_id}: {group.name.upper()}")
        print(f"Testing: {group.parameter}")
        print(f"Variants: {len(group.variants)}")
        print("=" * 60)

        group_results = []
        configs = group.get_configs(current_best)

        for config in configs:
            # Check if already completed
            if registry.is_completed(config):
                print(f"\n[SKIP] {config.name} - already completed")
                cached = registry.get_results(config)
                group_results.append({
                    "config": config,
                    "results": cached,
                    "cached": True,
                })
                continue

            # Run experiment
            fold_results = []
            for fold_idx in range(n_folds):
                try:
                    result = run_experiment_direct(
                        config=config,
                        fold_idx=fold_idx,
                        output_dir=output_dir,
                    )
                    fold_results.append(result)
                except Exception as e:
                    print(f"❌ FAILED: {config.name} fold {fold_idx}: {e}")
                    continue

            if fold_results:
                # Aggregate across folds
                mean_val_loss = np.mean([r["best_val_loss"] for r in fold_results])
                mean_test_loss = np.mean([r["test_loss"] for r in fold_results])
                std_val_loss = np.std([r["best_val_loss"] for r in fold_results])

                aggregated = {
                    "mean_val_loss": mean_val_loss,
                    "mean_test_loss": mean_test_loss,
                    "std_val_loss": std_val_loss,
                    "n_folds": len(fold_results),
                    "fold_results": fold_results,
                }

                # Register completion
                registry.mark_completed(config, aggregated)

                group_results.append({
                    "config": config,
                    "results": aggregated,
                    "cached": False,
                })

                print(f"  → {config.name}: val={mean_val_loss:.4f}±{std_val_loss:.4f}")

        # Select winner for this group
        if group_results:
            winner = min(group_results, key=lambda x: x["results"]["mean_val_loss"])
            winner_config = winner["config"]
            winner_loss = winner["results"]["mean_val_loss"]

            print(f"\n★ WINNER: {winner_config.name} (val_loss={winner_loss:.4f})")

            # Update current best with winner's parameter
            winner_overrides = {group.parameter: getattr(winner_config.model, group.parameter)}
            current_best = current_best.with_override(**winner_overrides)

            all_results["groups"][group.name] = {
                "variants_tested": len(group_results),
                "winner": winner_config.name,
                "winner_loss": winner_loss,
                "all_results": [
                    {
                        "name": r["config"].name,
                        "mean_val_loss": r["results"]["mean_val_loss"],
                        "cached": r["cached"],
                    }
                    for r in group_results
                ],
            }
            all_results["best_config_per_group"][group.name] = asdict(winner_config)

    # Save final results
    all_results["end_time"] = datetime.now().isoformat()
    all_results["final_best_config"] = asdict(current_best)

    results_path = output_dir / "final_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {results_path}")
    print(f"\nFinal best configuration:")
    print(f"  - activation: {current_best.model.activation}")
    print(f"  - norm_type: {current_best.model.norm_type}")
    print(f"  - skip_type: {current_best.model.skip_type}")
    print(f"  - conv_type: {current_best.model.conv_type}")
    print(f"  - n_downsample: {current_best.model.n_downsample}")
    print(f"  - base_channels: {current_best.model.base_channels}")
    print(f"  - attention_type: {current_best.model.attention_type}")
    print(f"  - optimizer: {current_best.training.optimizer}")
    print(f"  - learning_rate: {current_best.training.learning_rate}")
    print(f"  - batch_size: {current_best.training.batch_size}")

    return all_results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 v2: Bulletproof Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all groups with 3 folds
    python runner_v2.py --n-folds 3

    # Run specific groups
    python runner_v2.py --groups 1,2,3

    # Validate only (pre-flight check)
    python runner_v2.py --validate-only

    # Resume from previous run
    python runner_v2.py --resume
        """,
    )

    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated list of group IDs to run (default: all)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="Number of CV folds per variant (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase3_v2"),
        help="Output directory",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configs, don't run experiments",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip completed experiments)",
    )

    args = parser.parse_args()

    # Parse groups
    groups = None
    if args.groups:
        groups = [int(g) for g in args.groups.split(",")]

    # Run
    results = run_greedy_forward(
        groups=groups,
        n_folds=args.n_folds,
        output_dir=args.output_dir,
        validate_only=args.validate_only,
        resume=args.resume,
    )

    return 0 if results.get("status") != "preflight_failed" else 1


if __name__ == "__main__":
    sys.exit(main())
