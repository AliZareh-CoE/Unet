"""
Ablation Study Runner
=====================

Runs ablation experiments with held-out session validation.
Fast comparison of architectural variants on a single split.

Usage:
    python -m ablation.runner                    # Run all ablations
    python -m ablation.runner --group conv_type  # Run specific group
    python -m ablation.runner --dry-run          # Show what would run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    """Run training for a single variant with held-out sessions."""

    variant_dir = output_dir / group_name / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    results_file = variant_dir / "results.json"

    # Build command - use torchrun for FSDP
    if ablation_config.use_fsdp:
        cmd = [
            "torchrun",
            f"--nproc_per_node={ablation_config.n_gpus}",
            "train.py",
        ]
    else:
        cmd = ["python", "train.py"]

    cmd.extend([
        "--dataset", config.get("dataset", "olfactory"),
        "--epochs", str(ablation_config.epochs),
        "--batch-size", str(ablation_config.batch_size),
        "--lr", str(ablation_config.learning_rate),
        "--seed", str(ablation_config.seed),
        "--arch", config.get("arch", "condunet"),
        "--base-channels", str(config.get("base_channels", 128)),
        "--n-downsample", str(config.get("n_downsample", 2)),
        "--conv-type", config.get("conv_type", "modern"),
        "--attention-type", config.get("attention_type", "none"),
        "--activation", config.get("activation", "relu"),
        "--skip-type", config.get("skip_type", "add"),
        "--n-heads", str(config.get("n_heads", 4)),
        "--conditioning", config.get("conditioning", "spectro_temporal"),
        "--optimizer", config.get("optimizer", "adamw"),
        "--lr-schedule", config.get("lr_schedule", "step"),
        "--weight-decay", str(config.get("weight_decay", 0.01)),
        # Held-out session evaluation
        "--split-by-session",
        "--n-val-sessions", str(ablation_config.n_val_sessions),
        "--no-test-set",
        "--force-recreate-splits",
        # Output
        "--output-results-file", str(results_file),
        "--no-plots",
    ])

    # Conditional flags
    if config.get("cond_mode") == "none":
        cmd.extend(["--cond-mode", "none"])
    else:
        cmd.extend(["--cond-mode", config.get("cond_mode", "cross_attn_gated")])

    if config.get("use_adaptive_scaling"):
        cmd.append("--use-adaptive-scaling")

    if not config.get("use_bidirectional", True):
        cmd.append("--no-bidirectional")

    if ablation_config.use_fsdp:
        cmd.append("--fsdp")

    # Run training
    print(f"    Command: {' '.join(cmd[:10])}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not ablation_config.verbose,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )

        if result.returncode != 0:
            print(f"    ERROR: Training failed")
            if not ablation_config.verbose:
                print(result.stderr[-1000:] if result.stderr else "No stderr")
            return {"error": "Training failed", "variant": variant_name}

    except Exception as e:
        print(f"    ERROR: {e}")
        return {"error": str(e), "variant": variant_name}

    # Load results
    if results_file.exists():
        with open(results_file) as f:
            train_results = json.load(f)
    else:
        train_results = {}

    result_dict = {
        "group": group_name,
        "variant": variant_name,
        "best_r2": train_results.get("best_val_r2", 0.0),
        "best_mae": train_results.get("best_val_mae", 0.0),
        "best_epoch": train_results.get("best_epoch", 0),
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save result
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
    print_header(f"Ablation: {group['name']}", "-")
    print(f"  {group['description']}")
    print(f"  Parameter: {group['parameter']}")
    print(f"  Variants: {len(group['variants'])}")

    results = []
    for i, variant in enumerate(group["variants"]):
        print(f"\n  [{i+1}/{len(group['variants'])}] {variant['name']}: {variant['desc']}")

        config = build_variant_config(baseline, group, variant)
        result = run_single_variant(
            config=config,
            group_name=group["name"],
            variant_name=variant["name"],
            output_dir=output_dir,
            ablation_config=ablation_config,
        )
        results.append(result)

        if "error" not in result:
            print(f"    R² = {result['best_r2']:.4f}, MAE = {result['best_mae']:.4f}")

    return results


def run_ablation(config: Optional[AblationConfig] = None) -> Dict[str, Any]:
    """Run complete ablation study."""
    if config is None:
        config = AblationConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print_header("CondUNet Ablation Study")
    print(f"Output: {config.output_dir}")
    print(f"Validation: {config.n_val_sessions} held-out sessions")
    print(f"Epochs: {config.epochs}")

    # Determine which groups to run
    groups = ABLATION_GROUPS
    if config.groups_to_run:
        groups = [g for g in ABLATION_GROUPS if g["name"] in config.groups_to_run]

    print(f"Groups: {[g['name'] for g in groups]}")
    print(f"Total runs: {sum(len(g['variants']) for g in groups)}")

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
    print_header("Results Summary")

    for group_name, results in all_results.items():
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            print(f"\n{group_name}: ALL FAILED")
            continue

        print(f"\n{group_name}:")
        best_r2 = max(r["best_r2"] for r in valid_results)
        for r in sorted(valid_results, key=lambda x: x["best_r2"], reverse=True):
            marker = " <-- best" if r["best_r2"] == best_r2 else ""
            print(f"  {r['variant']:20s}: R² = {r['best_r2']:.4f}{marker}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_val_sessions": config.n_val_sessions,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
        },
        "baseline": BASELINE_CONFIG,
        "results": all_results,
    }

    with open(config.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved: {config.output_dir / 'summary.json'}")

    return all_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run CondUNet ablation study")

    parser.add_argument("--output-dir", type=str, default="artifacts/ablation",
                        help="Output directory")
    parser.add_argument("--group", type=str, nargs="+", default=None,
                        help="Specific groups to run")
    parser.add_argument("--n-val-sessions", type=int, default=3,
                        help="Number of held-out validation sessions (default: 3)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--fsdp", action="store_true",
                        help="Use FSDP distributed training")
    parser.add_argument("--n-gpus", type=int, default=8,
                        help="Number of GPUs for FSDP (default: 8)")
    parser.add_argument("--quiet", action="store_true",
                        help="Less verbose output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.dry_run:
        print_header("Ablation Study (DRY RUN)")

        groups = ABLATION_GROUPS
        if args.group:
            groups = [g for g in ABLATION_GROUPS if g["name"] in args.group]

        for group in groups:
            print(f"{group['name']}:")
            for v in group["variants"]:
                print(f"  - {v['name']}: {v['desc']}")
            print()

        total = sum(len(g["variants"]) for g in groups)
        print(f"Total: {total} training runs")
        print(f"Validation: {args.n_val_sessions} held-out sessions")
        return 0

    config = AblationConfig(
        output_dir=Path(args.output_dir),
        n_val_sessions=args.n_val_sessions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        use_fsdp=args.fsdp,
        n_gpus=args.n_gpus,
        verbose=not args.quiet,
        groups_to_run=args.group,
    )

    run_ablation(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
