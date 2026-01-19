"""
Phase 3 v2: Bulletproof Ablation Study Runner
==============================================

DESIGN PRINCIPLES:
1. TYPE-SAFE CONFIG: Validated before training starts
2. USES EXISTING train.py: No duplicate training loop - consistency!
3. PRE-FLIGHT VALIDATION: Validate ALL configs before running ANY
4. EXPERIMENT REGISTRY: Skip duplicates, resume gracefully
5. FAIL-FAST: Invalid config = immediate crash with clear error

The key insight: We keep the subprocess call to train.py (for FSDP, logging,
checkpointing, etc.) but we VALIDATE the config BEFORE spawning, and we
use type-safe configs that generate correct CLI args.

Usage:
    python runner_v2.py --groups 1,2,3 --n-folds 5
    python runner_v2.py --validate-only  # Pre-flight check
    python runner_v2.py --resume  # Continue from last run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

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
# Config to CLI Args (Type-Safe Generation)
# =============================================================================

def config_to_cli_args(config: ExperimentConfig, fold_idx: int, output_file: Path) -> List[str]:
    """Convert validated ExperimentConfig to CLI args for train.py.

    This is the ONLY place where CLI args are generated.
    Since config is type-safe and validated, the CLI args are guaranteed correct.

    Args:
        config: Validated experiment configuration
        fold_idx: Cross-validation fold index
        output_file: Path to save results JSON

    Returns:
        List of CLI arguments for train.py
    """
    args = [
        # Architecture (from ModelConfig)
        "--arch", "condunet",
        "--base-channels", str(config.model.base_channels),
        "--n-downsample", str(config.model.n_downsample),
        "--conv-type", config.model.conv_type,
        "--attention-type", config.model.attention_type,
        "--n-heads", str(config.model.n_heads),
        "--cond-mode", config.model.cond_mode,
        "--activation", config.model.activation,
        "--norm-type", config.model.norm_type,
        "--skip-type", config.model.skip_type,
        "--dropout", str(config.model.dropout),

        # Training (from TrainingConfig)
        "--optimizer", config.training.optimizer,
        "--lr", str(config.training.learning_rate),
        "--weight-decay", str(config.training.weight_decay),
        "--lr-schedule", config.training.lr_schedule,
        "--epochs", str(config.training.epochs),
        "--batch-size", str(config.training.batch_size),
        "--loss", config.training.loss_type,

        # Data (from DataConfig)
        "--seed", str(config.data.seed + fold_idx),
        "--fold", str(fold_idx),

        # Output
        "--output-results-file", str(output_file),
        "--no-plots",  # Skip plots for ablation speed
    ]

    # Augmentation
    if config.training.aug_strength == "none":
        args.append("--no-aug")
    else:
        args.extend(["--aug-strength", config.training.aug_strength])

    # Bidirectional
    if not config.training.bidirectional:
        args.append("--no-bidirectional")

    # Session splits
    if config.data.split_by_session:
        args.append("--split-by-session")
        args.extend(["--n-test-sessions", str(config.data.n_test_sessions)])
        args.extend(["--n-val-sessions", str(config.data.n_val_sessions)])

    return args


def run_experiment_via_train_py(
    config: ExperimentConfig,
    fold_idx: int = 0,
    output_dir: Path = None,
    use_fsdp: bool = True,
    fsdp_strategy: str = "full",
) -> Dict[str, Any]:
    """Run experiment using existing train.py - CONSISTENT training loop.

    This calls train.py as a subprocess but with TYPE-SAFE, VALIDATED config.
    We get all the benefits of train.py (FSDP, logging, checkpointing, etc.)
    without the risk of misconfigured CLI args.

    Args:
        config: Validated experiment configuration
        fold_idx: Cross-validation fold index
        output_dir: Directory to save results
        use_fsdp: Enable FSDP distributed training
        fsdp_strategy: FSDP sharding strategy

    Returns:
        Dictionary with metrics from train.py
    """
    if output_dir is None:
        output_dir = Path("outputs/phase3_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file for results
    output_file = output_dir / f"{config.config_hash()}_fold{fold_idx}_results.json"

    print(f"\n{'='*60}")
    print(f"Running: {config.name} [fold {fold_idx}]")
    print(f"Config hash: {config.config_hash()}")
    print(f"{'='*60}")

    # Generate CLI args from validated config
    cli_args = config_to_cli_args(config, fold_idx, output_file)

    # Build command
    if use_fsdp:
        # Use torchrun for distributed training
        import torch
        nproc = torch.cuda.device_count() if torch.cuda.is_available() else 1
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            "--master_port", str(29500 + fold_idx),  # Unique port per fold
            str(PROJECT_ROOT / "train.py"),
            "--fsdp",
            "--fsdp-strategy", fsdp_strategy,
        ] + cli_args
    else:
        cmd = ["python", str(PROJECT_ROOT / "train.py")] + cli_args

    # Log the command (for debugging)
    print(f"Command: {' '.join(cmd[:10])}...")  # First 10 args

    # Run train.py
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,  # Let output flow to console
            text=True,
            timeout=3600 * 4,  # 4 hour timeout
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"❌ train.py failed with return code {result.returncode}")
            return {
                "config_hash": config.config_hash(),
                "config_name": config.name,
                "fold": fold_idx,
                "status": "failed",
                "return_code": result.returncode,
                "elapsed_seconds": elapsed,
            }

    except subprocess.TimeoutExpired:
        print(f"❌ train.py timed out after 4 hours")
        return {
            "config_hash": config.config_hash(),
            "config_name": config.name,
            "fold": fold_idx,
            "status": "timeout",
            "elapsed_seconds": 3600 * 4,
        }
    except Exception as e:
        print(f"❌ Exception running train.py: {e}")
        return {
            "config_hash": config.config_hash(),
            "config_name": config.name,
            "fold": fold_idx,
            "status": "error",
            "error": str(e),
        }

    # Read results from output file
    if output_file.exists():
        with open(output_file) as f:
            results = json.load(f)
        results["config_hash"] = config.config_hash()
        results["config_name"] = config.name
        results["fold"] = fold_idx
        results["status"] = "success"
        results["elapsed_seconds"] = elapsed
        print(f"✓ Completed in {elapsed:.1f}s: val_loss={results.get('best_val_loss', 'N/A')}")
        return results
    else:
        print(f"⚠ No results file found at {output_file}")
        return {
            "config_hash": config.config_hash(),
            "config_name": config.name,
            "fold": fold_idx,
            "status": "no_results_file",
            "elapsed_seconds": elapsed,
        }


# =============================================================================
# Greedy Forward Selection
# =============================================================================

def run_greedy_forward(
    groups: List[int] = None,
    n_folds: int = 1,
    output_dir: Path = None,
    validate_only: bool = False,
    resume: bool = False,
    use_fsdp: bool = True,
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

            # Run experiment using existing train.py (CONSISTENT!)
            fold_results = []
            for fold_idx in range(n_folds):
                try:
                    result = run_experiment_via_train_py(
                        config=config,
                        fold_idx=fold_idx,
                        output_dir=output_dir,
                        use_fsdp=use_fsdp,
                    )
                    if result.get("status") == "success":
                        fold_results.append(result)
                    else:
                        print(f"⚠ {config.name} fold {fold_idx}: {result.get('status')}")
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
    parser.add_argument(
        "--fsdp",
        action="store_true",
        default=True,
        help="Enable FSDP distributed training (default: True)",
    )
    parser.add_argument(
        "--no-fsdp",
        action="store_true",
        help="Disable FSDP, run on single GPU",
    )

    args = parser.parse_args()

    # Parse groups
    groups = None
    if args.groups:
        groups = [int(g) for g in args.groups.split(",")]

    # FSDP setting
    use_fsdp = args.fsdp and not args.no_fsdp

    # Run
    results = run_greedy_forward(
        groups=groups,
        n_folds=args.n_folds,
        output_dir=args.output_dir,
        validate_only=args.validate_only,
        resume=args.resume,
        use_fsdp=use_fsdp,
    )

    return 0 if results.get("status") != "preflight_failed" else 1


if __name__ == "__main__":
    sys.exit(main())
