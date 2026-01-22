"""
Ablation Study Runner
=====================

Runs ablation with held-out test sessions and multiple seeds.

Setup:
- 3 sessions held out as TEST (never touched)
- Remaining 6 sessions: 70% train, 30% val
- 3 random seeds for variance estimation

Usage:
    python -m phase_three.runner --fsdp
    python -m phase_three.runner --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .config import ABLATION_GROUPS, BASELINE_CONFIG, AblationConfig


def print_header(text: str, char: str = "=") -> None:
    print(f"\n{char * 60}")
    print(f" {text}")
    print(f"{char * 60}\n")


def get_all_sessions(dataset: str = "olfactory") -> List[str]:
    """Get all available sessions for a dataset."""
    if dataset == "olfactory":
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data import load_session_ids, ODOR_CSV_PATH
        _, session_to_idx, _ = load_session_ids(ODOR_CSV_PATH)
        return sorted(session_to_idx.keys())
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def create_splits(
    sessions: List[str],
    n_test: int = 3,
    val_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split sessions into train/val/test.

    Args:
        sessions: All available sessions
        n_test: Number of sessions to hold out for test
        val_ratio: Fraction of remaining sessions for validation
        seed: Random seed for reproducibility

    Returns:
        (train_sessions, val_sessions, test_sessions)
    """
    rng = np.random.default_rng(seed)
    shuffled = sessions.copy()
    rng.shuffle(shuffled)

    # Hold out test sessions (always the same regardless of seed)
    # Use a fixed seed for test selection to ensure consistency
    test_rng = np.random.default_rng(0)
    test_order = sessions.copy()
    test_rng.shuffle(test_order)
    test_sessions = test_order[:n_test]

    # Remaining sessions for train/val
    remaining = [s for s in shuffled if s not in test_sessions]

    # Split remaining into train/val
    n_val = max(1, int(len(remaining) * val_ratio))
    val_sessions = remaining[:n_val]
    train_sessions = remaining[n_val:]

    return train_sessions, val_sessions, test_sessions


def run_training(
    config: Dict[str, Any],
    name: str,
    seed: int,
    train_sessions: List[str],
    val_sessions: List[str],
    output_dir: Path,
    ablation_config: AblationConfig,
) -> Dict[str, Any]:
    """Run a single training run with specified seed."""

    run_dir = output_dir / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_file = run_dir / "results.json"

    # Skip if already completed
    if results_file.exists():
        with open(results_file) as f:
            existing = json.load(f)
        if existing.get("completed_successfully"):
            print(f"    seed {seed}: R² = {existing.get('best_val_r2', 0):.4f} (cached)")
            return existing

    # Build base command
    if ablation_config.use_fsdp:
        cmd = ["torchrun", f"--nproc_per_node={ablation_config.n_gpus}", "train.py"]
    else:
        cmd = ["python", "train.py"]

    cmd.extend([
        "--dataset", config.get("dataset", "olfactory"),
        "--epochs", str(ablation_config.epochs),
        "--batch-size", str(ablation_config.batch_size),
        "--lr", str(ablation_config.learning_rate),
        "--seed", str(seed),
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
        "--no-test-set",  # We handle test separately
        "--output-results-file", str(results_file),
        "--no-plots",
        "--no-early-stop",
        "--force-recreate-splits",
        "--val-sessions", *val_sessions,
    ])

    if config.get("use_adaptive_scaling"):
        cmd.append("--use-adaptive-scaling")

    if not config.get("use_bidirectional", False):
        cmd.append("--no-bidirectional")

    if ablation_config.use_fsdp:
        cmd.append("--fsdp")

    if ablation_config.verbose:
        print(f"    Seed {seed}: train={train_sessions}, val={val_sessions}")

    # Set NCCL environment variables for stability with FSDP
    env = os.environ.copy()
    env["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1800"
    env["NCCL_TIMEOUT"] = "1800"
    env["NCCL_DEBUG"] = "WARN"

    try:
        result = subprocess.run(
            cmd,
            capture_output=not ablation_config.verbose,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env=env,
        )
        if result.returncode != 0:
            print(f"    seed {seed}: FAILED")
            if not ablation_config.verbose and result.stderr:
                print(f"    stderr: {result.stderr[-500:]}")
            return {"seed": seed, "error": "Training failed"}
    except Exception as e:
        print(f"    seed {seed}: ERROR - {e}")
        return {"seed": seed, "error": str(e)}

    # Load results
    if results_file.exists():
        with open(results_file) as f:
            train_results = json.load(f)
        r2 = train_results.get("best_val_r2", 0.0)
        print(f"    seed {seed}: R² = {r2:.4f}")
        return train_results
    else:
        return {"seed": seed, "error": "No results file"}


def run_experiment(
    config: Dict[str, Any],
    name: str,
    train_sessions: List[str],
    val_sessions: List[str],
    ablation_config: AblationConfig,
) -> Dict[str, Any]:
    """Run experiment with multiple seeds."""

    print(f"\n  {name}")

    r2_values = []
    mae_values = []
    seed_results = []

    for seed in ablation_config.seeds:
        result = run_training(
            config=config,
            name=name,
            seed=seed,
            train_sessions=train_sessions,
            val_sessions=val_sessions,
            output_dir=ablation_config.output_dir,
            ablation_config=ablation_config,
        )
        seed_results.append(result)

        if "error" not in result:
            r2_values.append(result.get("best_val_r2", 0.0))
            mae_values.append(result.get("best_val_mae", 0.0))

    # Compute mean ± std
    if r2_values:
        mean_r2 = np.mean(r2_values)
        std_r2 = np.std(r2_values)
        mean_mae = np.mean(mae_values)
        std_mae = np.std(mae_values)
    else:
        mean_r2 = std_r2 = mean_mae = std_mae = 0.0

    print(f"  -> Mean R² = {mean_r2:.4f} ± {std_r2:.4f}")

    return {
        "name": name,
        "config": config,
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "r2_values": r2_values,
        "seed_results": seed_results,
    }


def run_ablation(config: Optional[AblationConfig] = None) -> Dict[str, Any]:
    """Run complete ablation study."""
    if config is None:
        config = AblationConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print_header("CondUNet Ablation Study")
    print(f"Output: {config.output_dir}")
    print(f"Seeds: {config.seeds}")

    # Get all sessions and create splits
    all_sessions = get_all_sessions(config.dataset)
    train_sessions, val_sessions, test_sessions = create_splits(
        all_sessions,
        n_test=config.n_test_sessions,
        val_ratio=config.val_ratio,
        seed=42,  # Fixed seed for split consistency
    )

    print(f"\nAll sessions ({len(all_sessions)}): {all_sessions}")
    print(f"\nData Split:")
    print(f"  Test (held out):  {test_sessions}")
    print(f"  Train:            {train_sessions}")
    print(f"  Val:              {val_sessions}")

    results = {}

    # 1. Run baseline
    print_header("Baseline (Full Model)", "-")
    baseline_result = run_experiment(
        config=BASELINE_CONFIG.copy(),
        name="baseline",
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        ablation_config=config,
    )
    results["baseline"] = baseline_result

    # 2. Run ablations
    groups = ABLATION_GROUPS
    if config.groups_to_run:
        groups = [g for g in ABLATION_GROUPS if g["name"] in config.groups_to_run]

    for group in groups:
        print_header(f"Ablation: {group['name']}", "-")
        print(f"  {group['description']}")

        variants = group.get("variants", [group.get("variant")])

        for variant in variants:
            ablation_cfg = BASELINE_CONFIG.copy()

            # Support both single parameter and multi-parameter ablations
            if "parameters" in group:
                ablation_cfg.update(group["parameters"])
            else:
                ablation_cfg[group["parameter"]] = variant["value"]

            result = run_experiment(
                config=ablation_cfg,
                name=f"{group['name']}_{variant['name']}",
                train_sessions=train_sessions,
                val_sessions=val_sessions,
                ablation_config=config,
            )
            results[f"{group['name']}_{variant['name']}"] = result

    # Print summary
    print_header("Results Summary")
    print(f"{'Experiment':<25} {'R² (mean±std)':>18} {'Δ R²':>10}")
    print("-" * 55)

    baseline_r2 = results["baseline"]["mean_r2"]
    print(f"{'baseline':<25} {baseline_r2:>7.4f} ± {results['baseline']['std_r2']:<7.4f} {'--':>10}")

    for name, r in results.items():
        if name == "baseline":
            continue
        delta = r["mean_r2"] - baseline_r2
        sign = "+" if delta >= 0 else ""
        print(f"{name:<25} {r['mean_r2']:>7.4f} ± {r['std_r2']:<7.4f} {sign}{delta:>9.4f}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_test_sessions": config.n_test_sessions,
            "val_ratio": config.val_ratio,
            "seeds": config.seeds,
            "epochs": config.epochs,
        },
        "splits": {
            "all_sessions": all_sessions,
            "test_sessions": test_sessions,
            "train_sessions": train_sessions,
            "val_sessions": val_sessions,
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
    parser.add_argument("--output-dir", type=str, default="results/ablation")
    parser.add_argument("--group", type=str, nargs="+", default=None)
    parser.add_argument("--n-test-sessions", type=int, default=3, help="Sessions held out for test")
    parser.add_argument("--val-ratio", type=float, default=0.3, help="Fraction of remaining for validation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        print_header("Ablation Study (DRY RUN)")

        # Show sessions and splits
        all_sessions = get_all_sessions("olfactory")
        train_sessions, val_sessions, test_sessions = create_splits(
            all_sessions,
            n_test=args.n_test_sessions,
            val_ratio=args.val_ratio,
            seed=42,
        )

        print(f"All sessions ({len(all_sessions)}): {all_sessions}")
        print(f"\nData Split:")
        print(f"  Test (held out):  {test_sessions}")
        print(f"  Train:            {train_sessions}")
        print(f"  Val:              {val_sessions}")
        print(f"\nSeeds: {args.seeds}")

        groups = ABLATION_GROUPS
        if args.group:
            groups = [g for g in ABLATION_GROUPS if g["name"] in args.group]

        experiments = ["baseline"]
        for g in groups:
            variants = g.get("variants", [g.get("variant")])
            for v in variants:
                experiments.append(f"{g['name']}_{v['name']}")

        print(f"\nExperiments:")
        for i, exp in enumerate(experiments, 1):
            print(f"  {i}. {exp}")

        print(f"\n{len(experiments)} experiments × {len(args.seeds)} seeds = {len(experiments) * len(args.seeds)} total runs")
        return 0

    ablation_cfg = AblationConfig(
        output_dir=Path(args.output_dir),
        n_test_sessions=args.n_test_sessions,
        val_ratio=args.val_ratio,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_fsdp=args.fsdp,
        n_gpus=args.n_gpus,
        verbose=not args.quiet,
        groups_to_run=args.group,
    )

    run_ablation(ablation_cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
