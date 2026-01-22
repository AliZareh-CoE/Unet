"""
Ablation Study Runner
=====================

Runs ablation with k-fold CV for robust component selection.

Setup:
- All sessions split into k folds
- Each fold uses different sessions for validation
- Proper cross-validation: every session is in validation exactly once

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
from typing import Any, Dict, List, Optional
import numpy as np

from .config import ABLATION_GROUPS, BASELINE_CONFIG, AblationConfig


def print_header(text: str, char: str = "=") -> None:
    print(f"\n{char * 60}")
    print(f" {text}")
    print(f"{char * 60}\n")


def get_all_sessions(dataset: str = "olfactory") -> List[str]:
    """Get all available sessions for a dataset."""
    if dataset == "olfactory":
        # Import from project
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data import load_session_ids, ODOR_CSV_PATH
        _, session_to_idx, _ = load_session_ids(ODOR_CSV_PATH)
        return sorted(session_to_idx.keys())
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def create_cv_folds(sessions: List[str], n_folds: int, seed: int = 42) -> List[List[str]]:
    """Split sessions into k folds for cross-validation.

    Returns:
        List of k lists, where each inner list contains the validation sessions for that fold.
    """
    rng = np.random.default_rng(seed)
    shuffled = sessions.copy()
    rng.shuffle(shuffled)

    # Split into n_folds groups
    folds = []
    fold_size = len(shuffled) // n_folds
    remainder = len(shuffled) % n_folds

    start = 0
    for i in range(n_folds):
        # Distribute remainder across first few folds
        size = fold_size + (1 if i < remainder else 0)
        folds.append(shuffled[start:start + size])
        start += size

    return folds


def run_training(
    config: Dict[str, Any],
    name: str,
    fold: int,
    val_sessions: List[str],
    output_dir: Path,
    ablation_config: AblationConfig,
) -> Dict[str, Any]:
    """Run a single training fold."""

    run_dir = output_dir / name / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_file = run_dir / "results.json"

    # Skip if already completed
    if results_file.exists():
        with open(results_file) as f:
            existing = json.load(f)
        if existing.get("completed_successfully"):
            print(f"    fold {fold}: R² = {existing.get('best_val_r2', 0):.4f} (cached)")
            return existing

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
        "--seed", str(42 + fold),  # Different seed per fold
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
        "--no-test-set",
        "--output-results-file", str(results_file),
        "--no-plots",
        "--no-early-stop",
        "--fold", str(fold),
        "--force-recreate-splits",  # Force new splits with our val sessions
        "--val-sessions", *val_sessions,  # Explicitly specify validation sessions
    ])

    if config.get("use_adaptive_scaling"):
        cmd.append("--use-adaptive-scaling")

    if not config.get("use_bidirectional", False):
        cmd.append("--no-bidirectional")

    if ablation_config.use_fsdp:
        cmd.append("--fsdp")

    if ablation_config.verbose:
        print(f"    Fold {fold} val sessions: {val_sessions}")

    # Set NCCL environment variables for stability
    env = os.environ.copy()
    env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1800"  # 30 minutes
    env["NCCL_TIMEOUT"] = "1800"

    try:
        result = subprocess.run(
            cmd,
            capture_output=not ablation_config.verbose,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env=env,
        )
        if result.returncode != 0:
            print(f"    fold {fold}: FAILED")
            if not ablation_config.verbose and result.stderr:
                print(f"    stderr: {result.stderr[-500:]}")
            return {"fold": fold, "error": "Training failed", "val_sessions": val_sessions}
    except Exception as e:
        print(f"    fold {fold}: ERROR - {e}")
        return {"fold": fold, "error": str(e), "val_sessions": val_sessions}

    # Load results
    if results_file.exists():
        with open(results_file) as f:
            train_results = json.load(f)
        r2 = train_results.get("best_val_r2", 0.0)
        print(f"    fold {fold}: R² = {r2:.4f}")
        return train_results
    else:
        return {"fold": fold, "error": "No results file", "val_sessions": val_sessions}


def run_experiment(
    config: Dict[str, Any],
    name: str,
    cv_folds: List[List[str]],
    ablation_config: AblationConfig,
) -> Dict[str, Any]:
    """Run experiment with k-fold CV."""

    print(f"\n  {name}")

    r2_values = []
    mae_values = []
    fold_results = []

    for fold_idx, val_sessions in enumerate(cv_folds):
        result = run_training(
            config=config,
            name=name,
            fold=fold_idx,
            val_sessions=val_sessions,
            output_dir=ablation_config.output_dir,
            ablation_config=ablation_config,
        )
        fold_results.append(result)

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
        "fold_results": fold_results,
    }


def run_ablation(config: Optional[AblationConfig] = None) -> Dict[str, Any]:
    """Run complete ablation study."""
    if config is None:
        config = AblationConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print_header("CondUNet Ablation Study")
    print(f"Output: {config.output_dir}")
    print(f"Folds: {config.n_folds}-fold CV")

    # Get all sessions and create CV folds
    all_sessions = get_all_sessions(config.dataset)
    cv_folds = create_cv_folds(all_sessions, config.n_folds, seed=42)

    print(f"\nAll sessions ({len(all_sessions)}): {all_sessions}")
    print(f"\nCV Folds:")
    for i, fold_sessions in enumerate(cv_folds):
        print(f"  Fold {i} (val): {fold_sessions}")

    results = {}

    # 1. Run baseline
    print_header("Baseline (Full Model)", "-")
    baseline_result = run_experiment(
        config=BASELINE_CONFIG.copy(),
        name="baseline",
        cv_folds=cv_folds,
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
            ablation_cfg[group["parameter"]] = variant["value"]

            result = run_experiment(
                config=ablation_cfg,
                name=f"{group['name']}_{variant['name']}",
                cv_folds=cv_folds,
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
            "n_folds": config.n_folds,
            "epochs": config.epochs,
        },
        "all_sessions": all_sessions,
        "cv_folds": cv_folds,
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
    parser.add_argument("--n-folds", type=int, default=5, help="k-fold CV (sessions split into k folds)")
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

        # Show sessions and folds
        all_sessions = get_all_sessions("olfactory")
        cv_folds = create_cv_folds(all_sessions, args.n_folds, seed=42)

        print(f"All sessions ({len(all_sessions)}): {all_sessions}")
        print(f"\nCV Folds:")
        for i, fold_sessions in enumerate(cv_folds):
            print(f"  Fold {i} (val): {fold_sessions}")

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

        print(f"\n{len(experiments)} experiments × {args.n_folds} folds = {len(experiments) * args.n_folds} total runs")
        return 0

    ablation_cfg = AblationConfig(
        output_dir=Path(args.output_dir),
        n_folds=args.n_folds,
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
