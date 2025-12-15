#!/usr/bin/env python3
"""
Validate Optuna Trial with Full train.py Code Path
===================================================

This script loads a trial configuration from an Optuna study and runs
full training using train.py to validate that the HPO results are reproducible.

This addresses the 78% → 73% reproducibility gap by ensuring HPO results
can be replicated with the full training pipeline.

Usage:
    # Validate specific trial
    python validate_optuna_trial.py --trial-number 42 --study-name my_study

    # Validate top 5 trials
    python validate_optuna_trial.py --top 5 --study-name my_study

    # Validate from checkpoint
    python validate_optuna_trial.py --checkpoint artifacts/optuna_parallel/checkpoints/trial_42.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
import torch


# =============================================================================
# Configuration
# =============================================================================

STUDY_DIR = Path("artifacts/optuna_parallel")
RESULTS_DIR = STUDY_DIR / "validation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Trial Loading
# =============================================================================

def load_trial_config(study_name: str, storage: str, trial_number: int) -> Dict[str, Any]:
    """Load configuration from a specific trial."""
    study = optuna.load_study(study_name=study_name, storage=storage)

    trial = None
    for t in study.trials:
        if t.number == trial_number:
            trial = t
            break

    if trial is None:
        raise ValueError(f"Trial {trial_number} not found in study {study_name}")

    # Get full config from user attributes
    config_json = trial.user_attrs.get("full_config", "{}")
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError:
        config = {}

    # Add trial metadata
    config["_trial_number"] = trial.number
    config["_trial_score"] = trial.value
    config["_trial_seed"] = trial.user_attrs.get("trial_seed", 42)
    config["_best_corr"] = trial.user_attrs.get("best_corr", 0)
    config["_best_r2"] = trial.user_attrs.get("best_r2", 0)

    return config


def load_config_from_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load configuration from a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config", {})
    config["_trial_number"] = checkpoint.get("trial_number", -1)
    config["_trial_seed"] = checkpoint.get("seed", 42)
    config["_checkpoint_path"] = str(checkpoint_path)

    if "metrics" in checkpoint:
        config["_best_corr"] = checkpoint["metrics"].get("corr", 0)
        config["_best_r2"] = checkpoint["metrics"].get("r2", 0)

    return config


def get_top_trials(study_name: str, storage: str, n: int = 5) -> List[Dict[str, Any]]:
    """Get configurations for top N trials."""
    study = optuna.load_study(study_name=study_name, storage=storage)

    completed = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:n]

    configs = []
    for trial in top_trials:
        config_json = trial.user_attrs.get("full_config", "{}")
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError:
            config = {}

        config["_trial_number"] = trial.number
        config["_trial_score"] = trial.value
        config["_trial_seed"] = trial.user_attrs.get("trial_seed", 42)
        config["_best_corr"] = trial.user_attrs.get("best_corr", 0)
        config["_best_r2"] = trial.user_attrs.get("best_r2", 0)
        configs.append(config)

    return configs


# =============================================================================
# Validation with train.py
# =============================================================================

def build_train_command(config: Dict[str, Any], epochs: int = 80, seed: int = 42) -> List[str]:
    """Build command line arguments for train.py based on config."""
    cmd = ["python", "train.py"]

    # Basic training params
    cmd.extend(["--epochs", str(epochs)])
    cmd.extend(["--seed", str(seed)])

    # Batch size
    if "batch_size" in config:
        cmd.extend(["--batch-size", str(config["batch_size"])])

    # Learning rate
    if "learning_rate" in config:
        cmd.extend(["--lr", str(config["learning_rate"])])

    # Model architecture
    if "base_channels" in config:
        cmd.extend(["--base-channels", str(config["base_channels"])])

    # Conditioning mode
    if "cond_mode" in config:
        cmd.extend(["--cond-mode", str(config["cond_mode"])])

    # Note: Many HPO params aren't exposed via train.py CLI
    # For full control, we'd need to modify train.py or use a config file

    return cmd


def run_validation(
    config: Dict[str, Any],
    epochs: int = 80,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run validation using train.py with the given config.

    Note: This runs train.py as a subprocess and parses the output.
    For production use, consider importing train functions directly.
    """
    trial_num = config.get("_trial_number", -1)
    trial_seed = config.get("_trial_seed", 42)
    hpo_corr = config.get("_best_corr", 0)
    hpo_r2 = config.get("_best_r2", 0)
    hpo_score = config.get("_trial_score", 0)

    print(f"\n{'='*60}")
    print(f"Validating Trial #{trial_num}")
    print(f"{'='*60}")
    print(f"HPO Results: score={hpo_score:.4f}, corr={hpo_corr:.4f}, r²={hpo_r2:.4f}")
    print(f"Seed: {trial_seed}")
    print(f"Config: {json.dumps({k: v for k, v in config.items() if not k.startswith('_')}, indent=2)}")

    if dry_run:
        print("\n[DRY-RUN] Would run train.py with above config")
        return {
            "trial_number": trial_num,
            "hpo_score": hpo_score,
            "hpo_corr": hpo_corr,
            "hpo_r2": hpo_r2,
            "validated": False,
            "dry_run": True,
        }

    # Build and run train.py command
    cmd = build_train_command(config, epochs=epochs, seed=trial_seed)
    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 4,  # 4 hour timeout
        )

        output = result.stdout + result.stderr

        # Parse results from train.py output
        validated_corr = None
        validated_r2 = None

        for line in output.split("\n"):
            if "RESULT_CORR=" in line:
                validated_corr = float(line.split("=")[1])
            if "RESULT_R2=" in line:
                validated_r2 = float(line.split("=")[1])

        if validated_corr is not None:
            corr_diff = validated_corr - hpo_corr
            r2_diff = validated_r2 - hpo_r2 if validated_r2 else None

            print(f"\n{'='*60}")
            print(f"Validation Results for Trial #{trial_num}")
            print(f"{'='*60}")
            print(f"HPO Corr:      {hpo_corr:.4f}")
            print(f"Validated Corr: {validated_corr:.4f}")
            print(f"Difference:     {corr_diff:+.4f}")
            if r2_diff is not None:
                print(f"HPO R²:        {hpo_r2:.4f}")
                print(f"Validated R²:   {validated_r2:.4f}")
                print(f"Difference:     {r2_diff:+.4f}")

            return {
                "trial_number": trial_num,
                "hpo_score": hpo_score,
                "hpo_corr": hpo_corr,
                "hpo_r2": hpo_r2,
                "validated_corr": validated_corr,
                "validated_r2": validated_r2,
                "corr_diff": corr_diff,
                "r2_diff": r2_diff,
                "validated": True,
                "success": True,
            }
        else:
            print(f"\nFailed to parse results from train.py output")
            return {
                "trial_number": trial_num,
                "hpo_score": hpo_score,
                "hpo_corr": hpo_corr,
                "validated": False,
                "success": False,
                "error": "Could not parse train.py output",
                "output": output[-2000:],  # Last 2000 chars
            }

    except subprocess.TimeoutExpired:
        print(f"\nValidation timed out for trial {trial_num}")
        return {
            "trial_number": trial_num,
            "validated": False,
            "success": False,
            "error": "Timeout",
        }
    except Exception as e:
        print(f"\nValidation failed for trial {trial_num}: {e}")
        return {
            "trial_number": trial_num,
            "validated": False,
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Report Generation
# =============================================================================

def generate_validation_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate a validation report comparing HPO vs train.py results."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_trials_validated": len(results),
        "n_successful": sum(1 for r in results if r.get("success", False)),
        "results": results,
    }

    # Compute summary statistics
    successful = [r for r in results if r.get("success", False)]
    if successful:
        corr_diffs = [r["corr_diff"] for r in successful if "corr_diff" in r]
        if corr_diffs:
            report["mean_corr_diff"] = sum(corr_diffs) / len(corr_diffs)
            report["max_corr_diff"] = max(corr_diffs)
            report["min_corr_diff"] = min(corr_diffs)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nValidation report saved to: {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Trials validated: {len(results)}")
    print(f"Successful: {report.get('n_successful', 0)}")
    if "mean_corr_diff" in report:
        print(f"Mean correlation difference: {report['mean_corr_diff']:+.4f}")
        print(f"Range: [{report['min_corr_diff']:+.4f}, {report['max_corr_diff']:+.4f}]")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate Optuna trials with full train.py"
    )

    # Trial selection
    parser.add_argument("--trial-number", type=int, default=None,
                       help="Specific trial number to validate")
    parser.add_argument("--top", type=int, default=None,
                       help="Validate top N trials")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint file to validate")

    # Study info
    parser.add_argument("--study-name", type=str, default="olfactory_parallel_hpo",
                       help="Optuna study name")
    parser.add_argument("--storage", type=str,
                       default="sqlite:///artifacts/optuna_parallel/study.db",
                       help="Optuna storage URL")

    # Validation options
    parser.add_argument("--epochs", type=int, default=80,
                       help="Number of epochs for validation training")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show config without running training")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for validation report")

    args = parser.parse_args()

    # Determine which trials to validate
    configs = []

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        configs.append(load_config_from_checkpoint(checkpoint_path))

    elif args.trial_number is not None:
        configs.append(load_trial_config(args.study_name, args.storage, args.trial_number))

    elif args.top:
        configs = get_top_trials(args.study_name, args.storage, args.top)

    else:
        # Default: validate top 5
        print("No trial specified, validating top 5 trials...")
        configs = get_top_trials(args.study_name, args.storage, 5)

    if not configs:
        print("No trials to validate!")
        sys.exit(1)

    print(f"\nWill validate {len(configs)} trial(s)")

    # Run validation
    results = []
    for config in configs:
        result = run_validation(config, epochs=args.epochs, dry_run=args.dry_run)
        results.append(result)

    # Generate report
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    generate_validation_report(results, output_path)


if __name__ == "__main__":
    main()
