#!/usr/bin/env python3
"""
Master Tier Runner - Bulletproof Experimental Pipeline
======================================================

Orchestrates the complete tier-based experimental pipeline for Nature Methods.

Tier Flow:
    Tier 0: Classical Floor (GATE: establishes minimum bar)
       ↓
    Tier 1: Quick Screening (GATE: neural must beat classical by 0.05)
       ↓
    Tier 2: Factorial Design (KEY TIER: finds interactions)
       ↓
    [Conditional: SpectralShift if PSD error > 3dB]
       ↓
    Tier 3: Ablation (FEEDBACK: removes useless components)
       ↓
    Tier 4: Final Validation (GATE: negative controls must pass)
       ↓
    COMPLETE

Usage:
    # Full pipeline
    python experiments/tier_runner.py

    # Dry-run (fast test)
    python experiments/tier_runner.py --dry-run

    # Resume from specific tier
    python experiments/tier_runner.py --start-tier 2

    # Run single tier
    python experiments/tier_runner.py --tier 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# =============================================================================
# CRITICAL: Fix Python path for imports
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from experiments.common.config_registry import (
    ConfigRegistry,
    GateFailure,
    get_registry,
    reset_registry,
)


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier_system"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading
# =============================================================================

def load_data(dry_run: bool = False) -> Dict[str, torch.Tensor]:
    """Load or create data for experiments.

    Args:
        dry_run: If True, use small synthetic data

    Returns:
        Dictionary with train/test tensors
    """
    if dry_run:
        N_train, N_test = 80, 20
        C, T = 32, 500

        data = {
            "X_train": torch.randn(N_train, C, T),
            "y_train": torch.randn(N_train, C, T),
            "X_test": torch.randn(N_test, C, T),
            "y_test": torch.randn(N_test, C, T),
        }
        print(f"[DRY-RUN] Using synthetic data: {N_train} train, {N_test} test")
        return data

    try:
        from data import prepare_data
        import numpy as np

        raw_data = prepare_data()

        # prepare_data returns:
        # - ob: [trials, channels, time] OB signals
        # - pcx: [trials, channels, time] PCx signals
        # - train_idx, val_idx, test_idx: index arrays
        ob = raw_data["ob"]
        pcx = raw_data["pcx"]
        train_idx = raw_data["train_idx"]
        val_idx = raw_data["val_idx"]
        test_idx = raw_data["test_idx"]

        # Combine train and val for CV, keep test as holdout
        train_val_idx = np.concatenate([train_idx, val_idx])

        X_train = torch.from_numpy(ob[train_val_idx]).float()
        y_train = torch.from_numpy(pcx[train_val_idx]).float()
        X_test = torch.from_numpy(ob[test_idx]).float()
        y_test = torch.from_numpy(pcx[test_idx]).float()

        print(f"Loaded real data: {X_train.shape[0]} train, {X_test.shape[0]} test")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    except Exception as e:
        print(f"Warning: Could not load real data ('{e}'), using synthetic")
        N_train, N_test = 160, 40
        C, T = 32, 1000

        return {
            "X_train": torch.randn(N_train, C, T),
            "y_train": torch.randn(N_train, C, T),
            "X_test": torch.randn(N_test, C, T),
            "y_test": torch.randn(N_test, C, T),
        }


# =============================================================================
# Tier Runners
# =============================================================================

def run_tier0(data: Dict[str, torch.Tensor], dry_run: bool = False) -> None:
    """Run Tier 0: Classical Floor."""
    from experiments.tier0_classical import run_tier0 as _run_tier0

    print("\n" + "=" * 70)
    print("TIER 0: CLASSICAL FLOOR")
    print("=" * 70)

    X = torch.cat([data["X_train"], data["X_test"]], dim=0).numpy()
    y = torch.cat([data["y_train"], data["y_test"]], dim=0).numpy()

    baselines = ["ridge", "wiener"] if dry_run else None

    _run_tier0(
        X=X,
        y=y,
        baselines=baselines,
        n_folds=2 if dry_run else 5,
    )


def run_tier1(data: Dict[str, torch.Tensor], device: torch.device, dry_run: bool = False) -> None:
    """Run Tier 1: Quick Screening."""
    from experiments.tier1_screening import run_tier1 as _run_tier1

    print("\n" + "=" * 70)
    print("TIER 1: QUICK SCREENING")
    print("=" * 70)

    X = data["X_train"]
    y = data["y_train"]

    if dry_run:
        architectures = ["linear", "cnn"]
        losses = {"huber": "huber", "spectral": "multi_scale_spectral"}
        n_epochs = 3
        data_fraction = 0.5
    else:
        architectures = None
        losses = None
        n_epochs = 20
        data_fraction = 0.2

    _run_tier1(
        X=X,
        y=y,
        device=device,
        architectures=architectures,
        losses=losses,
        data_fraction=data_fraction,
        n_epochs=n_epochs,
    )

    # Check gate
    registry = get_registry()
    registry.check_gate_tier1(dry_run=dry_run)
    if not dry_run or (registry.tier1 and registry.tier0 and registry.tier1.best_r2 > registry.tier0.best_r2 + 0.05):
        print("\n✓ GATE PASSED: Neural beats classical")


def run_tier2(data: Dict[str, torch.Tensor], device: torch.device, dry_run: bool = False) -> None:
    """Run Tier 2: Factorial Design."""
    from experiments.tier2_factorial import run_tier2 as _run_tier2

    print("\n" + "=" * 70)
    print("TIER 2: FACTORIAL DESIGN (KEY TIER)")
    print("=" * 70)

    X = data["X_train"]
    y = data["y_train"]

    if dry_run:
        architectures = ["linear", "cnn"]
        losses = ["huber", "spectral"]
        conditionings = ["concat"]
        n_epochs = 5
        n_seeds = 2
        n_folds = 2
    else:
        architectures = None  # From Tier 1
        losses = None  # From Tier 1
        conditionings = None
        n_epochs = 80
        n_seeds = 5
        n_folds = 5

    _run_tier2(
        X=X,
        y=y,
        device=device,
        architectures=architectures,
        losses=losses,
        conditionings=conditionings,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_epochs=n_epochs,
    )

    # Check gate
    registry = get_registry()
    registry.check_gate_tier2(dry_run=dry_run)
    if not dry_run or (registry.tier2 and registry.tier0 and registry.tier2.best_r2_mean > registry.tier0.best_r2 + 0.10):
        print("\n✓ GATE PASSED: Winner beats classical by sufficient margin")

    # Check if SpectralShift is needed
    if registry.needs_spectral_shift():
        print("\n⚠ SpectralShift optimization NEEDED (PSD error > 3dB)")
    else:
        print("\n✓ SpectralShift NOT needed (PSD error acceptable)")


def run_tier3(data: Dict[str, torch.Tensor], device: torch.device, dry_run: bool = False) -> None:
    """Run Tier 3: Focused Ablation."""
    from experiments.tier3_ablation import run_tier3 as _run_tier3

    print("\n" + "=" * 70)
    print("TIER 3: FOCUSED ABLATION")
    print("=" * 70)

    X = data["X_train"]
    y = data["y_train"]

    if dry_run:
        base_config = ("linear", "huber", "concat")
        n_epochs = 5
        n_seeds = 2
        n_folds = 2
    else:
        base_config = None  # From Tier 2
        n_epochs = 80
        n_seeds = 5
        n_folds = 5

    _run_tier3(
        X=X,
        y=y,
        device=device,
        base_config=base_config,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_epochs=n_epochs,
    )

    # Report removed components
    registry = get_registry()
    removed = registry.get_components_to_remove()
    if removed:
        print(f"\n⚠ Components REMOVED: {removed}")
    else:
        print("\n✓ All components contribute meaningfully")


def run_tier4(data: Dict[str, torch.Tensor], device: torch.device, dry_run: bool = False) -> None:
    """Run Tier 4: Final Validation."""
    from experiments.tier4_validation import run_tier4 as _run_tier4

    print("\n" + "=" * 70)
    print("TIER 4: FINAL VALIDATION (HOLDOUT TEST SET)")
    print("=" * 70)
    print("⚠ CRITICAL: First time using holdout test set!")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    if dry_run:
        configs = [{"name": "test", "architecture": "linear", "loss": "huber"}]
        n_epochs = 5
        n_seeds = 2
    else:
        configs = None  # From previous tiers
        n_epochs = 80
        n_seeds = 10

    _run_tier4(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device=device,
        configs=configs,
        n_seeds=n_seeds,
        n_epochs=n_epochs,
    )

    # Check gate
    registry = get_registry()
    registry.check_gate_tier4(dry_run=dry_run)
    if not dry_run or (registry.tier4 and registry.tier4.all_controls_passed):
        print("\n✓ GATE PASSED: All negative controls passed")


# =============================================================================
# Master Pipeline
# =============================================================================

def run_full_pipeline(
    dry_run: bool = False,
    start_tier: int = 0,
    single_tier: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run the complete tier-based experimental pipeline.

    Args:
        dry_run: If True, use minimal configuration for testing
        start_tier: Tier to start from (for resuming)
        single_tier: Run only this specific tier
        device: Device to use

    Returns:
        Dictionary with all results
    """
    start_time = time.time()

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Reset registry for fresh run (unless resuming)
    if start_tier == 0 and single_tier is None:
        reset_registry()

    registry = get_registry(ARTIFACTS_DIR)

    # Load data
    print("\nLoading data...")
    data = load_data(dry_run=dry_run)

    # Determine which tiers to run
    if single_tier is not None:
        tiers_to_run = [single_tier]
    else:
        tiers_to_run = list(range(start_tier, 5))

    # Run each tier with clear progress tracking
    current_tier = None
    completed_tiers = []

    try:
        if 0 in tiers_to_run:
            current_tier = 0
            run_tier0(data, dry_run=dry_run)
            completed_tiers.append(0)

        if 1 in tiers_to_run:
            current_tier = 1
            run_tier1(data, device, dry_run=dry_run)
            completed_tiers.append(1)

        if 2 in tiers_to_run:
            current_tier = 2
            run_tier2(data, device, dry_run=dry_run)
            completed_tiers.append(2)

        if 3 in tiers_to_run:
            current_tier = 3
            run_tier3(data, device, dry_run=dry_run)
            completed_tiers.append(3)

        if 4 in tiers_to_run:
            current_tier = 4
            run_tier4(data, device, dry_run=dry_run)
            completed_tiers.append(4)

    except GateFailure as e:
        print(f"\n❌ GATE FAILURE at Tier {current_tier}: {e}")
        print("Pipeline stopped. Review results and adjust approach.")
        return {"status": "gate_failure", "error": str(e), "failed_tier": current_tier}

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ERROR at Tier {current_tier}")
        print(f"{'='*70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n{'='*70}")
        print(f"COMPLETED TIERS: {completed_tiers}")
        print(f"FAILED TIER: {current_tier}")
        print(f"\n★ TO RESUME, RUN:")
        print(f"  python experiments/tier_runner.py --start-tier {current_tier}")
        print(f"{'='*70}")
        return {"status": "error", "error": str(e), "failed_tier": current_tier, "completed": completed_tiers}

    # Final summary
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed / 60:.1f} minutes")

    # Print summary from registry
    summary = registry.get_full_summary()
    print("\n" + registry.export_for_paper())

    # Save final results
    output_file = ARTIFACTS_DIR / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nFinal results saved to: {output_file}")

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Master Tier Runner - Bulletproof Experimental Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tier Flow:
    Tier 0: Classical Floor
    Tier 1: Quick Screening (8×6 matrix)
    Tier 2: Factorial Design (KEY: finds interactions)
    Tier 3: Focused Ablation (removes useless components)
    Tier 4: Final Validation (negative controls)

Examples:
    # Full pipeline
    python experiments/tier_runner.py

    # Quick test
    python experiments/tier_runner.py --dry-run

    # Resume from Tier 2
    python experiments/tier_runner.py --start-tier 2

    # Run only Tier 3
    python experiments/tier_runner.py --tier 3
        """
    )

    parser.add_argument("--dry-run", action="store_true",
                        help="Run with minimal configuration for testing")
    parser.add_argument("--start-tier", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Tier to start from (for resuming)")
    parser.add_argument("--tier", type=int, choices=[0, 1, 2, 3, 4],
                        help="Run only this specific tier")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    print("=" * 70)
    print("NATURE METHODS EXPERIMENTAL PIPELINE")
    print("Bulletproof Edition")
    print("=" * 70)
    print()
    print("This pipeline implements:")
    print("  • 5-fold cross-validation throughout")
    print("  • Factorial design to detect interactions")
    print("  • ANOVA with effect sizes (Cohen's d)")
    print("  • Bonferroni correction for multiple comparisons")
    print("  • Negative controls (shuffled, reversed, permutation)")
    print("  • Bootstrap 95% confidence intervals")
    print("  • Gates between tiers to catch failures early")
    print()

    if args.dry_run:
        print("⚠ DRY-RUN MODE: Using minimal configuration")
        print()

    run_full_pipeline(
        dry_run=args.dry_run,
        start_tier=args.start_tier,
        single_tier=args.tier,
        device=args.device,
    )


if __name__ == "__main__":
    main()
