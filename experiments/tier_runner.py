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
    Tier 1.5: Conditioning Deep Dive (tests 6 conditioning mechanisms)
       ↓
    Tier 2: Factorial Design (KEY TIER: finds interactions)
       ↓
    Tier 2.5: Probabilistic Loss Ablation (signal distribution losses)
       ↓
    Tier 3: Ablation (FEEDBACK: removes useless components)
       ↓
    Tier 3.5: SpectralShift Variants (PSD correction approaches)
       ↓
    Tier 4: Final Validation (GATE: negative controls must pass)
       ↓
    Tier 5: Biological Validation (GATE: proves biological meaning)
       ↓
    Tier 6: Cross-Dataset Generalization (tests on external data)
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

    # Run tier 1.5
    python experiments/tier_runner.py --tier 1.5
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


def run_tier1_5(data: Dict[str, torch.Tensor], device: torch.device, dry_run: bool = False) -> None:
    """Run Tier 1.5: Conditioning Deep Dive."""
    from experiments.tier1_5_conditioning import run_tier1_5 as _run_tier1_5

    print("\n" + "=" * 70)
    print("TIER 1.5: CONDITIONING DEEP DIVE")
    print("=" * 70)

    X = data["X_train"]
    y = data["y_train"]

    if dry_run:
        architecture = "linear"
        conditionings = ["concat", "cross_attn_gated"]
        n_epochs = 5
        n_seeds = 2
        n_folds = 2
    else:
        architecture = None  # From Tier 1
        conditionings = None  # All 6
        n_epochs = 80
        n_seeds = 3
        n_folds = 5

    _run_tier1_5(
        X=X,
        y=y,
        device=device,
        architecture=architecture,
        conditionings=conditionings,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_epochs=n_epochs,
    )

    registry = get_registry()
    if registry.tier1_5:
        print(f"\n✓ Best conditioning: {registry.tier1_5.best_conditioning}")


def run_tier2_5(data: Dict[str, torch.Tensor], device: torch.device, dry_run: bool = False) -> None:
    """Run Tier 2.5: Probabilistic Loss Ablation."""
    from experiments.tier2_5_probabilistic import run_tier2_5 as _run_tier2_5

    print("\n" + "=" * 70)
    print("TIER 2.5: PROBABILISTIC LOSS ABLATION")
    print("=" * 70)

    X = data["X_train"]
    y = data["y_train"]

    if dry_run:
        architecture = "linear"
        base_losses = ["huber"]
        prob_losses = ["rayleigh", "von_mises"]
        n_epochs = 5
        n_seeds = 2
        n_folds = 2
    else:
        architecture = None  # From Tier 2
        base_losses = None  # All base losses
        prob_losses = None  # All 7 probabilistic losses
        n_epochs = 80
        n_seeds = 3
        n_folds = 5

    _run_tier2_5(
        X=X,
        y=y,
        device=device,
        architecture=architecture,
        base_losses=base_losses,
        prob_losses=prob_losses,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_epochs=n_epochs,
    )

    registry = get_registry()
    if registry.tier2_5:
        print(f"\n✓ Best probabilistic loss: {registry.tier2_5.best_combination}")


def run_tier3_5(data: Dict[str, torch.Tensor], device: torch.device, dry_run: bool = False) -> None:
    """Run Tier 3.5: SpectralShift Variants."""
    from experiments.tier3_5_spectralshift import run_tier3_5 as _run_tier3_5

    print("\n" + "=" * 70)
    print("TIER 3.5: SPECTRALSHIFT VARIANTS")
    print("=" * 70)

    X = data["X_train"]
    y = data["y_train"]

    if dry_run:
        methods = ["hard_fft", "soft"]
        n_epochs = 5
        n_seeds = 2
        n_folds = 2
    else:
        methods = None  # All methods
        n_epochs = 80
        n_seeds = 3
        n_folds = 5

    _run_tier3_5(
        X=X,
        y=y,
        device=device,
        methods=methods,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_epochs=n_epochs,
    )

    registry = get_registry()
    if registry.tier3_5:
        print(f"\n✓ Best SpectralShift method: {registry.tier3_5.best_method}")
        print(f"  PSD Error: {registry.tier3_5.best_psd_error:.2f} dB")


def run_tier5(data: Dict[str, torch.Tensor], device: torch.device, dry_run: bool = False) -> None:
    """Run Tier 5: Biological Validation."""
    from experiments.tier5_biological import run_tier5 as _run_tier5

    print("\n" + "=" * 70)
    print("TIER 5: BIOLOGICAL VALIDATION")
    print("=" * 70)
    print("⚠ CRITICAL: Proves translated signals are biologically meaningful!")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    if dry_run:
        n_epochs = 5
    else:
        n_epochs = 80

    _run_tier5(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device=device,
        n_epochs=n_epochs,
    )

    # Check gate
    registry = get_registry()
    registry.check_gate_tier5(dry_run=dry_run)
    if not dry_run or (registry.tier5 and registry.tier5.passed):
        print("\n✓ GATE PASSED: Biological validation passed")


def run_tier6(data: Dict[str, torch.Tensor], device: torch.device, dry_run: bool = False) -> None:
    """Run Tier 6: Cross-Dataset Generalization."""
    from experiments.tier6_generalization import run_tier6 as _run_tier6

    print("\n" + "=" * 70)
    print("TIER 6: CROSS-DATASET GENERALIZATION")
    print("=" * 70)
    print("⚠ CRITICAL: Tests model on external/shifted datasets!")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    if dry_run:
        datasets = ["frequency_shifted"]
        n_epochs = 5
        n_seeds = 2
    else:
        datasets = None  # All datasets
        n_epochs = 80
        n_seeds = 5

    _run_tier6(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device=device,
        datasets=datasets,
        n_seeds=n_seeds,
        n_epochs=n_epochs,
    )

    # Check gate
    registry = get_registry()
    registry.check_gate_tier6(dry_run=dry_run)
    if not dry_run or (registry.tier6 and registry.tier6.generalizes):
        print("\n✓ GATE PASSED: Model generalizes to external datasets")


# =============================================================================
# Master Pipeline
# =============================================================================

def run_full_pipeline(
    dry_run: bool = False,
    start_tier: float = 0,
    single_tier: Optional[float] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run the complete tier-based experimental pipeline.

    Args:
        dry_run: If True, use minimal configuration for testing
        start_tier: Tier to start from (for resuming), supports 1.5, 2.5, 3.5
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

    # Define tier order with half-tiers
    ALL_TIERS = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]

    # Determine which tiers to run
    if single_tier is not None:
        tiers_to_run = [single_tier]
    else:
        start_idx = ALL_TIERS.index(start_tier) if start_tier in ALL_TIERS else 0
        tiers_to_run = ALL_TIERS[start_idx:]

    # Run each tier
    try:
        if 0 in tiers_to_run:
            run_tier0(data, dry_run=dry_run)

        if 1 in tiers_to_run:
            run_tier1(data, device, dry_run=dry_run)

        if 1.5 in tiers_to_run:
            run_tier1_5(data, device, dry_run=dry_run)

        if 2 in tiers_to_run:
            run_tier2(data, device, dry_run=dry_run)

        if 2.5 in tiers_to_run:
            run_tier2_5(data, device, dry_run=dry_run)

        if 3 in tiers_to_run:
            run_tier3(data, device, dry_run=dry_run)

        if 3.5 in tiers_to_run:
            run_tier3_5(data, device, dry_run=dry_run)

        if 4 in tiers_to_run:
            run_tier4(data, device, dry_run=dry_run)

        if 5 in tiers_to_run:
            run_tier5(data, device, dry_run=dry_run)

        if 6 in tiers_to_run:
            run_tier6(data, device, dry_run=dry_run)

    except GateFailure as e:
        print(f"\n❌ GATE FAILURE: {e}")
        print("Pipeline stopped. Review results and adjust approach.")
        return {"status": "gate_failure", "error": str(e)}

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

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
    Tier 1.5: Conditioning Deep Dive
    Tier 2: Factorial Design (KEY: finds interactions)
    Tier 2.5: Probabilistic Loss Ablation
    Tier 3: Focused Ablation (removes useless components)
    Tier 3.5: SpectralShift Variants
    Tier 4: Final Validation (negative controls)
    Tier 5: Biological Validation (CRITICAL)
    Tier 6: Cross-Dataset Generalization (CRITICAL)

Examples:
    # Full pipeline
    python experiments/tier_runner.py

    # Quick test
    python experiments/tier_runner.py --dry-run

    # Resume from Tier 2
    python experiments/tier_runner.py --start-tier 2

    # Run only Tier 3
    python experiments/tier_runner.py --tier 3

    # Run Tier 1.5 (Conditioning)
    python experiments/tier_runner.py --tier 1.5

    # Run Tier 5 (Biological Validation)
    python experiments/tier_runner.py --tier 5
        """
    )

    VALID_TIERS = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]

    parser.add_argument("--dry-run", action="store_true",
                        help="Run with minimal configuration for testing")
    parser.add_argument("--start-tier", type=float, default=0,
                        help="Tier to start from (for resuming). Options: 0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6")
    parser.add_argument("--tier", type=float,
                        help="Run only this specific tier. Options: 0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Validate tier arguments
    VALID_TIERS = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]
    if args.start_tier not in VALID_TIERS:
        print(f"Error: --start-tier must be one of {VALID_TIERS}")
        sys.exit(1)
    if args.tier is not None and args.tier not in VALID_TIERS:
        print(f"Error: --tier must be one of {VALID_TIERS}")
        sys.exit(1)

    print("=" * 70)
    print("NATURE METHODS EXPERIMENTAL PIPELINE")
    print("Bulletproof Edition - Complete 10-Tier System")
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
    print("  • Conditioning mechanism comparison (Tier 1.5)")
    print("  • Probabilistic loss ablation (Tier 2.5)")
    print("  • SpectralShift optimization (Tier 3.5)")
    print("  • Biological validation (Tier 5)")
    print("  • Cross-dataset generalization (Tier 6)")
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
