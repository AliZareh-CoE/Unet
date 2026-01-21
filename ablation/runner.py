"""
Ablation Study Runner
=====================

Runs ablation experiments using LOSO cross-validation.
Each variant is evaluated across all sessions for robust comparison.

Usage:
    python -m ablation.runner                    # Run all ablations
    python -m ablation.runner --group conv_type  # Run specific group
    python -m ablation.runner --dry-run          # Show what would run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ABLATION_GROUPS, BASELINE_CONFIG, AblationConfig


def print_header(text: str, char: str = "=") -> None:
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(f" {text}")
    print(f"{char * 60}\n")


def build_variant_config(
    baseline: Dict[str, Any],
    group: Dict[str, Any],
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    """Build config for a specific variant."""
    config = baseline.copy()
    config[group["parameter"]] = variant["value"]
    return config


def run_single_variant(
    config: Dict[str, Any],
    group_name: str,
    variant_name: str,
    output_dir: Path,
    ablation_config: AblationConfig,
) -> Dict[str, Any]:
    """Run LOSO evaluation for a single variant."""
    from LOSO.config import LOSOConfig
    from LOSO.runner import run_loso

    variant_dir = output_dir / group_name / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Build LOSO config from ablation config
    loso_config = LOSOConfig(
        dataset=config.get("dataset", "olfactory"),
        output_dir=variant_dir,
        epochs=ablation_config.epochs,
        batch_size=ablation_config.batch_size,
        learning_rate=ablation_config.learning_rate,
        seed=42,
        arch=config.get("arch", "condunet"),
        base_channels=config.get("base_channels", 128),
        n_downsample=config.get("n_downsample", 2),
        attention_type=config.get("attention_type", "none"),
        cond_mode=config.get("cond_mode", "cross_attn_gated"),
        conv_type=config.get("conv_type", "modern"),
        activation=config.get("activation", "relu"),
        skip_type=config.get("skip_type", "add"),
        n_heads=config.get("n_heads", 4),
        conditioning=config.get("conditioning", "spectro_temporal"),
        optimizer=config.get("optimizer", "adamw"),
        lr_schedule=config.get("lr_schedule", "step"),
        weight_decay=config.get("weight_decay", 0.01),
        dropout=config.get("dropout", 0.0),
        use_session_stats=config.get("use_session_stats", False),
        session_use_spectral=config.get("session_use_spectral", False),
        use_adaptive_scaling=config.get("use_adaptive_scaling", True),
        use_bidirectional=config.get("use_bidirectional", True),
        use_fsdp=True,
        fsdp_strategy="full",
        resume=ablation_config.resume,
        verbose=ablation_config.verbose,
        generate_plots=False,
    )

    print(f"  Config: {group_name}={config[ABLATION_GROUPS[[g['name'] for g in ABLATION_GROUPS].index(group_name)]['parameter']]}")

    result = run_loso(loso_config, folds_to_run=ablation_config.loso_folds)

    # Save result
    result_dict = {
        "group": group_name,
        "variant": variant_name,
        "mean_r2": result.mean_r2,
        "std_r2": result.std_r2,
        "fold_r2s": result.fold_r2s,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    with open(variant_dir / "result.json", "w") as f:
        json.dump(result_dict, f, indent=2, default=str)

    return result_dict


def run_ablation_group(
    group: Dict[str, Any],
    baseline: Dict[str, Any],
    output_dir: Path,
    ablation_config: AblationConfig,
) -> List[Dict[str, Any]]:
    """Run all variants in an ablation group."""
    print_header(f"Ablation Group: {group['name']}", "-")
    print(f"Description: {group['description']}")
    print(f"Parameter: {group['parameter']}")
    print(f"Variants: {len(group['variants'])}")

    results = []
    for variant in group["variants"]:
        print(f"\n  Running variant: {variant['name']} ({variant['desc']})")

        config = build_variant_config(baseline, group, variant)
        result = run_single_variant(
            config=config,
            group_name=group["name"],
            variant_name=variant["name"],
            output_dir=output_dir,
            ablation_config=ablation_config,
        )
        results.append(result)

        print(f"  Result: R² = {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")

    return results


def run_ablation(config: Optional[AblationConfig] = None) -> Dict[str, Any]:
    """Run complete ablation study."""
    if config is None:
        config = AblationConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print_header("CondUNet Ablation Study")
    print(f"Output: {config.output_dir}")
    print(f"Groups: {len(ABLATION_GROUPS)}")
    print(f"Total variants: {sum(len(g['variants']) for g in ABLATION_GROUPS)}")

    # Determine which groups to run
    groups = ABLATION_GROUPS
    if config.groups_to_run:
        groups = [g for g in ABLATION_GROUPS if g["name"] in config.groups_to_run]
        print(f"Running subset: {[g['name'] for g in groups]}")

    all_results = {}

    for group in groups:
        results = run_ablation_group(
            group=group,
            baseline=BASELINE_CONFIG,
            output_dir=config.output_dir,
            ablation_config=config,
        )
        all_results[group["name"]] = results

    # Print summary
    print_header("Ablation Results Summary")

    for group_name, results in all_results.items():
        print(f"\n{group_name}:")
        for r in sorted(results, key=lambda x: x["mean_r2"], reverse=True):
            marker = " *" if r["mean_r2"] == max(x["mean_r2"] for x in results) else ""
            print(f"  {r['variant']:20s}: R² = {r['mean_r2']:.4f} ± {r['std_r2']:.4f}{marker}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "baseline_config": BASELINE_CONFIG,
        "results": all_results,
    }

    with open(config.output_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {config.output_dir / 'ablation_summary.json'}")

    return all_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run CondUNet ablation study")

    parser.add_argument("--output-dir", type=str, default="artifacts/ablation",
                        help="Output directory for results")
    parser.add_argument("--group", type=str, nargs="+", default=None,
                        help="Specific groups to run (default: all)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Specific LOSO folds to run")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from checkpoint")
    parser.add_argument("--quiet", action="store_true",
                        help="Less verbose output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.dry_run:
        print_header("Ablation Study (DRY RUN)")
        print("Would run the following experiments:\n")

        groups = ABLATION_GROUPS
        if args.group:
            groups = [g for g in ABLATION_GROUPS if g["name"] in args.group]

        for group in groups:
            print(f"{group['name']} ({group['description']}):")
            for variant in group["variants"]:
                print(f"  - {variant['name']}: {variant['desc']}")
            print()

        total = sum(len(g["variants"]) for g in groups)
        print(f"Total: {total} experiments × LOSO folds")
        return 0

    config = AblationConfig(
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        loso_folds=args.folds,
        resume=not args.no_resume,
        verbose=not args.quiet,
        groups_to_run=args.group,
    )

    run_ablation(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
