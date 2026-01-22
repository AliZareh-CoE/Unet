"""
Ablation Study Runner
=====================

Runs baseline once, then tests removing each component.
Total: 8 runs (1 baseline + 5 ablations + 2 depth variants)

Usage:
    python -m ablation.runner                    # Run all
    python -m ablation.runner --group conv_type  # Run specific ablation
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
    print(f"\n{char * 60}")
    print(f" {text}")
    print(f"{char * 60}\n")


def run_training(
    config: Dict[str, Any],
    name: str,
    output_dir: Path,
    ablation_config: AblationConfig,
) -> Dict[str, Any]:
    """Run a single training experiment."""

    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_file = run_dir / "results.json"

    # Build command
    if ablation_config.use_fsdp:
        cmd = ["torchrun", f"--nproc_per_node={ablation_config.n_gpus}", "train.py"]
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
        "--cond-mode", config.get("cond_mode", "cross_attn_gated"),
        "--optimizer", config.get("optimizer", "adamw"),
        "--lr-schedule", config.get("lr_schedule", "step"),
        "--weight-decay", str(config.get("weight_decay", 0.01)),
        "--split-by-session",
        "--n-val-sessions", str(ablation_config.n_val_sessions),
        "--no-test-set",
        "--force-recreate-splits",
        "--output-results-file", str(results_file),
        "--no-plots",
    ])

    if config.get("use_adaptive_scaling"):
        cmd.append("--use-adaptive-scaling")

    if not config.get("use_bidirectional", True):
        cmd.append("--no-bidirectional")

    if ablation_config.use_fsdp:
        cmd.append("--fsdp")

    print(f"  Running: {name}")
    if ablation_config.verbose:
        print(f"  Command: {' '.join(cmd[:8])}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not ablation_config.verbose,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        if result.returncode != 0:
            print(f"  ERROR: Training failed")
            return {"name": name, "error": "Training failed"}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"name": name, "error": str(e)}

    # Load results
    if results_file.exists():
        with open(results_file) as f:
            train_results = json.load(f)
    else:
        train_results = {}

    result_dict = {
        "name": name,
        "best_r2": train_results.get("best_val_r2", 0.0),
        "best_mae": train_results.get("best_val_mae", 0.0),
        "best_epoch": train_results.get("best_epoch", 0),
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    with open(run_dir / "result.json", "w") as f:
        json.dump(result_dict, f, indent=2, default=str)

    print(f"  Result: R² = {result_dict['best_r2']:.4f}, MAE = {result_dict['best_mae']:.4f}")
    return result_dict


def run_ablation(config: Optional[AblationConfig] = None) -> Dict[str, Any]:
    """Run complete ablation study."""
    if config is None:
        config = AblationConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print_header("CondUNet Ablation Study")
    print(f"Output: {config.output_dir}")
    print(f"Validation: {config.n_val_sessions} held-out sessions")

    results = {"baseline": None, "ablations": {}}

    # 1. Run baseline (full model)
    print_header("Baseline (Full Model)", "-")
    baseline_result = run_training(
        config=BASELINE_CONFIG.copy(),
        name="baseline",
        output_dir=config.output_dir,
        ablation_config=config,
    )
    results["baseline"] = baseline_result
    baseline_r2 = baseline_result.get("best_r2", 0.0)

    # 2. Run ablations (remove each component)
    groups = ABLATION_GROUPS
    if config.groups_to_run:
        groups = [g for g in ABLATION_GROUPS if g["name"] in config.groups_to_run]

    for group in groups:
        print_header(f"Ablation: {group['name']}", "-")
        print(f"  {group['description']}")

        # Get variants to test
        if "variants" in group:
            variants = group["variants"]
        else:
            variants = [group["variant"]]

        group_results = []
        for variant in variants:
            # Build config with this ablation
            ablation_config_dict = BASELINE_CONFIG.copy()
            ablation_config_dict[group["parameter"]] = variant["value"]

            result = run_training(
                config=ablation_config_dict,
                name=f"{group['name']}_{variant['name']}",
                output_dir=config.output_dir,
                ablation_config=config,
            )
            result["variant"] = variant["name"]
            result["description"] = variant["desc"]
            group_results.append(result)

        results["ablations"][group["name"]] = group_results

    # Print summary
    print_header("Results Summary")
    print(f"{'Experiment':<30} {'R²':>10} {'Δ R²':>10}")
    print("-" * 52)

    print(f"{'Baseline (full model)':<30} {baseline_r2:>10.4f} {'--':>10}")

    for group_name, group_results in results["ablations"].items():
        for r in group_results:
            if "error" in r:
                print(f"{r['name']:<30} {'FAILED':>10} {'--':>10}")
            else:
                delta = r["best_r2"] - baseline_r2
                sign = "+" if delta >= 0 else ""
                print(f"{r['name']:<30} {r['best_r2']:>10.4f} {sign}{delta:>9.4f}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_val_sessions": config.n_val_sessions,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "seed": config.seed,
        },
        "baseline_config": BASELINE_CONFIG,
        "results": results,
    }

    with open(config.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved: {config.output_dir / 'summary.json'}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CondUNet ablation study")
    parser.add_argument("--output-dir", type=str, default="artifacts/ablation")
    parser.add_argument("--group", type=str, nargs="+", default=None,
                        help="Specific ablation groups to run")
    parser.add_argument("--n-val-sessions", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        print_header("Ablation Study (DRY RUN)")
        print("1. baseline (full model)")

        groups = ABLATION_GROUPS
        if args.group:
            groups = [g for g in ABLATION_GROUPS if g["name"] in args.group]

        run_num = 2
        for group in groups:
            variants = group.get("variants", [group.get("variant")])
            for v in variants:
                print(f"{run_num}. {group['name']}_{v['name']}: {v['desc']}")
                run_num += 1

        print(f"\nTotal: {run_num - 1} training runs")
        return 0

    ablation_cfg = AblationConfig(
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

    run_ablation(ablation_cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
