#!/usr/bin/env python3
"""
Combine Phase 3 JSON Results
============================
Combines all Phase 3 result JSON files into a single comprehensive summary.

Handles:
1. Individual training results from train.py (g{group_id}_{variant}_results.json)
2. Combined phase3_results.json from the runner
3. Greedy forward selection results with 17 unified groups

The script extracts group_id and variant from file names, aggregates results,
and identifies the winner (best R²) for each ablation group.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from phase_three.config import ABLATION_GROUPS, GREEDY_DEFAULTS


# Build lookup from group_id to group info
GROUP_INFO = {g["group_id"]: g for g in ABLATION_GROUPS}
# Build lookup from list index to group info (for proper ordering)
GROUP_BY_INDEX = {i: g for i, g in enumerate(ABLATION_GROUPS)}


def get_group_info(group_id: int) -> Dict[str, Any]:
    """Get group info from ABLATION_GROUPS config."""
    return GROUP_INFO.get(group_id, {"name": f"group_{group_id}", "parameter": "unknown"})


def get_group_name(group_id: int) -> str:
    """Get group name from ABLATION_GROUPS config."""
    return get_group_info(group_id).get("name", f"group_{group_id}")


def get_group_parameter(group_id: int) -> str:
    """Get the config parameter name for a group."""
    return get_group_info(group_id).get("parameter", "unknown")


def get_variant_value(group_id: int, variant_name: str) -> Any:
    """Get the actual config value for a variant name."""
    group = GROUP_INFO.get(group_id)
    if not group:
        return variant_name

    for v in group.get("variants", []):
        if v.get("name") == variant_name:
            return v.get("value")

    # Fallback: return variant name
    return variant_name


def parse_result_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Parse group_id and variant from result filename.

    Expected formats:
    - g{group_id}_{variant}_results.json (e.g., g1_batch_norm_results.json)
    - g{group_id}_{variant}_results-checkpoint.json (checkpoint files)
    - {study}_{variant}_fold{fold}_results.json (e.g., normalization_batch_norm_fold0_results.json)
    - baseline_full_fold{fold}_results.json (e.g., baseline_full_fold0_results.json)
    """
    # Format: g{group_id}_{variant}_results.json or g{group_id}_{variant}_results-checkpoint.json
    match = re.match(r'g(\d+)_(.+)_results(?:-checkpoint)?\.json', filename)
    if match:
        return {
            "group_id": int(match.group(1)),
            "variant": match.group(2),
            "fold": None,
            "is_checkpoint": "-checkpoint" in filename,
            "stage": 1,
        }

    # Format: s2g{group_id}_{variant}_results.json (Stage 2 greedy - deprecated but handle)
    match = re.match(r's2g(\d+)_(.+)_results(?:-checkpoint)?\.json', filename)
    if match:
        return {
            "group_id": int(match.group(1)),
            "variant": match.group(2),
            "fold": None,
            "is_checkpoint": "-checkpoint" in filename,
            "stage": 2,
        }

    # Format: {study}_{variant}_fold{fold}_results.json
    match = re.match(r'(.+)_(.+)_fold(\d+)_results\.json', filename)
    if match:
        study = match.group(1)
        variant = match.group(2)
        fold = int(match.group(3))

        # Look up group_id from study name
        group_id = None
        for group in ABLATION_GROUPS:
            if group["name"] == study:
                group_id = group["group_id"]
                break

        return {
            "group_id": group_id,
            "study": study,
            "variant": variant,
            "fold": fold
        }

    # Format: baseline_full_fold{fold}_results.json
    match = re.match(r'baseline_full_fold(\d+)_results\.json', filename)
    if match:
        return {
            "group_id": 0,
            "study": "baseline",
            "variant": "baseline",
            "fold": int(match.group(1))
        }

    return None


def combine_results(results_dir: Path = Path("results/phase3")):
    """Combine all Phase 3 JSON results into a single summary."""

    results_dir = Path(results_dir)

    # Find all JSON files (including in subdirectories like train_results/)
    json_files = sorted(results_dir.glob("**/*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir} or subdirectories")
        return None

    print(f"Found {len(json_files)} JSON files:")
    for f in json_files:
        print(f"  - {f.relative_to(results_dir)}")

    # Combined result structure
    combined = {
        "generated_at": datetime.now().isoformat(),
        "source_files": [str(f.relative_to(results_dir)) for f in json_files],
        "group_results": {},  # group_id -> {variants: {variant: {r2, mae, corr, ...}}, winner: ...}
        "optimal_config": dict(GREEDY_DEFAULTS),  # Start with defaults
        "winner_summary": [],
        "raw_results": [],  # All individual results
    }

    # Temporary storage for aggregating by group
    group_variants: Dict[int, Dict[str, List[Dict]]] = {}  # group_id -> variant -> list of fold results

    # Load and process all results
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Skip the combined results file if we're re-running
            if json_file.name == "phase3_combined_results.json":
                continue

            # Check if this is a Phase3Result (from runner) or individual training result
            if "aggregated" in data and "optimal_config" in data:
                # This is a Phase3Result - use it directly for optimal_config
                combined["optimal_config"].update(data.get("optimal_config", {}))

                # Extract individual results
                for result in data.get("results", []):
                    combined["raw_results"].append(result)

                continue

            # Parse filename to get group and variant info
            parsed = parse_result_filename(json_file.name)

            if parsed is None:
                print(f"  Warning: Could not parse filename {json_file.name}")
                continue

            group_id = parsed.get("group_id")
            variant = parsed.get("variant")
            fold = parsed.get("fold")
            is_checkpoint = parsed.get("is_checkpoint", False)

            if group_id is None:
                print(f"  Warning: Unknown group for {json_file.name}")
                continue

            # Extract metrics from individual training result
            result_entry = {
                "file": str(json_file.relative_to(results_dir)),
                "group_id": group_id,
                "variant": variant,
                "fold": fold,
                "is_checkpoint": is_checkpoint,
                "best_val_r2": data.get("best_val_r2", 0),
                "best_val_mae": data.get("best_val_mae", 0),
                "best_val_corr": data.get("best_val_corr", 0),
                "best_val_loss": data.get("best_val_loss", 0),
                "best_epoch": data.get("best_epoch", 0),
                "epochs_trained": data.get("epochs_trained", 0),
                "n_parameters": data.get("n_parameters", 0),
                "completed_successfully": data.get("completed_successfully", False),
            }

            combined["raw_results"].append(result_entry)

            # Aggregate by group and variant
            # Use (group_id, variant) as key, prefer final results over checkpoints
            if group_id not in group_variants:
                group_variants[group_id] = {}
            if variant not in group_variants[group_id]:
                group_variants[group_id][variant] = []

            # Check if we already have a result for this variant
            existing = group_variants[group_id][variant]
            if existing:
                # If existing is checkpoint and new is final, replace
                if existing[0].get("is_checkpoint", False) and not is_checkpoint:
                    group_variants[group_id][variant] = [result_entry]
                # If existing is final and new is checkpoint, skip
                elif not existing[0].get("is_checkpoint", False) and is_checkpoint:
                    continue
                else:
                    # Both same type (both final or both checkpoint), append for fold averaging
                    group_variants[group_id][variant].append(result_entry)
            else:
                group_variants[group_id][variant].append(result_entry)

        except Exception as e:
            print(f"  Warning: Could not load {json_file.name}: {e}")

    # Build group results with statistics
    for group_id in sorted(group_variants.keys()):
        group_info = get_group_info(group_id)
        group_name = group_info.get("name", f"group_{group_id}") if group_id > 0 else "baseline"
        parameter = group_info.get("parameter", "unknown")

        variants_data = {}
        best_variant = None
        best_r2 = -float('inf')

        for variant, fold_results in group_variants[group_id].items():
            # Calculate mean metrics across folds
            n_folds = len(fold_results)
            mean_r2 = sum(r["best_val_r2"] for r in fold_results) / n_folds
            mean_mae = sum(r["best_val_mae"] for r in fold_results) / n_folds
            mean_corr = sum(r["best_val_corr"] for r in fold_results) / n_folds
            mean_loss = sum(r["best_val_loss"] for r in fold_results) / n_folds

            # Track if all folds completed successfully
            all_completed = all(r.get("completed_successfully", False) for r in fold_results)

            variants_data[variant] = {
                "mean_r2": mean_r2,
                "mean_mae": mean_mae,
                "mean_corr": mean_corr,
                "mean_loss": mean_loss,
                "n_folds": n_folds,
                "all_completed": all_completed,
                "fold_r2s": [r["best_val_r2"] for r in fold_results],
                "value": get_variant_value(group_id, variant),
            }

            # Track best variant
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_variant = variant

        combined["group_results"][str(group_id)] = {
            "group_name": group_name,
            "parameter": parameter,
            "variants": variants_data,
            "winner": best_variant,
            "winner_r2": best_r2,
            "winner_value": get_variant_value(group_id, best_variant) if best_variant else None,
        }

        # Update optimal config with the actual parameter value
        if group_id > 0 and best_variant:
            winner_value = get_variant_value(group_id, best_variant)
            combined["optimal_config"][parameter] = winner_value

    # Calculate improvements (relative to first completed group)
    sorted_group_ids = sorted(group_variants.keys())
    prev_r2 = None

    for group_id in sorted_group_ids:
        group_data = combined["group_results"][str(group_id)]
        current_r2 = group_data["winner_r2"]

        if prev_r2 is not None:
            improvement = current_r2 - prev_r2
            group_data["improvement"] = improvement
        else:
            group_data["improvement"] = 0

        prev_r2 = current_r2

    # Build winner summary table (ordered by list index, not group_id)
    # First, map group_ids to their list index for proper ordering
    group_id_to_index = {g["group_id"]: i for i, g in enumerate(ABLATION_GROUPS)}

    for group_id in sorted(group_variants.keys(), key=lambda gid: group_id_to_index.get(gid, 999)):
        group_data = combined["group_results"][str(group_id)]
        combined["winner_summary"].append({
            "group_id": group_id,
            "index": group_id_to_index.get(group_id, -1),
            "name": group_data["group_name"],
            "parameter": group_data["parameter"],
            "winner": group_data["winner"],
            "winner_value": group_data.get("winner_value"),
            "r2": group_data["winner_r2"],
            "improvement": group_data.get("improvement", 0),
        })

    # Save combined results
    output_path = results_dir / "phase3_combined_results.json"
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved combined results to {output_path}")

    # Print summary table
    print_summary(combined)

    return combined


def print_summary(combined: Dict[str, Any]):
    """Print a formatted summary of the combined results."""
    print("\n" + "=" * 100)
    print("PHASE 3 GREEDY FORWARD SELECTION - COMBINED RESULTS")
    print("=" * 100)

    # Count completed groups
    n_completed = len(combined["winner_summary"])
    n_total = len(ABLATION_GROUPS)
    print(f"\nProgress: {n_completed}/{n_total} groups completed")

    # Print winner summary table
    print(f"\n{'#':<3} {'Group':<20} {'Parameter':<18} {'Winner':<22} {'Value':<12} {'R²':<10} {'Δ R²':<10}")
    print("-" * 100)

    first_r2 = None
    final_r2 = None

    for i, row in enumerate(combined["winner_summary"], 1):
        if first_r2 is None:
            first_r2 = row["r2"]
        final_r2 = row["r2"]

        # Format improvement
        imp_str = f"{row['improvement']:+.4f}" if row['improvement'] != 0 else "-"

        # Format value (handle different types)
        value = row.get("winner_value", "")
        if isinstance(value, bool):
            value_str = "Yes" if value else "No"
        elif isinstance(value, float):
            value_str = f"{value:.0e}" if value < 0.01 else f"{value}"
        else:
            value_str = str(value)[:12]

        print(f"{i:<3} {row['name']:<20} {row['parameter']:<18} {row['winner']:<22} "
              f"{value_str:<12} {row['r2']:<10.4f} {imp_str:<10}")

    print("-" * 100)

    if first_r2 and final_r2:
        total_improvement = final_r2 - first_r2
        print(f"Cumulative: {first_r2:.4f} → {final_r2:.4f} (Δ = {total_improvement:+.4f})")

    # Print remaining groups
    completed_ids = {row["group_id"] for row in combined["winner_summary"]}
    remaining = [(i, g) for i, g in enumerate(ABLATION_GROUPS) if g["group_id"] not in completed_ids]

    if remaining:
        print(f"\n{'='*100}")
        print(f"REMAINING GROUPS ({len(remaining)})")
        print("=" * 100)
        for idx, group in remaining:
            print(f"  {idx+1}. {group['name']} ({group['parameter']})")

    # Print variant details for each group
    print("\n" + "=" * 100)
    print("DETAILED VARIANT RESULTS BY GROUP")
    print("=" * 100)

    for row in combined["winner_summary"]:
        group_id_str = str(row["group_id"])
        group_data = combined["group_results"][group_id_str]

        print(f"\n[{row['name']}] parameter: {row['parameter']}")
        print("-" * 60)

        # Sort variants by R²
        variants = sorted(
            group_data["variants"].items(),
            key=lambda x: x[1]["mean_r2"],
            reverse=True
        )

        for variant, stats in variants:
            marker = " ★ WINNER" if variant == group_data["winner"] else ""
            value = stats.get("value", "")
            print(f"  {variant:<25} R²={stats['mean_r2']:.4f} "
                  f"(MAE={stats['mean_mae']:.4f}, corr={stats['mean_corr']:.4f}) "
                  f"[{value}]{marker}")

    # Print optimal configuration
    print("\n" + "=" * 100)
    print("OPTIMAL CONFIGURATION (for train.py)")
    print("=" * 100)

    # Group by category for readability
    arch_params = ["base_channels", "norm_type", "skip_type", "activation", "conv_type",
                   "n_downsample", "attention_type", "n_heads", "cond_mode", "dropout"]
    train_params = ["optimizer", "lr_schedule", "weight_decay", "loss_type", "batch_size",
                    "bidirectional", "cycle_lambda"]
    session_params = ["use_adaptive_scaling", "use_session_stats", "session_use_spectral"]
    aug_params = ["use_augmentation", "aug_strength"]

    def print_param_group(name: str, params: List[str]):
        print(f"\n  # {name}")
        for p in params:
            if p in combined["optimal_config"]:
                v = combined["optimal_config"][p]
                print(f"  {p}: {v}")

    print_param_group("Architecture", arch_params)
    print_param_group("Training", train_params)
    print_param_group("Session Adaptation", session_params)
    print_param_group("Augmentation", aug_params)

    # Print CLI command
    print("\n" + "=" * 100)
    print("CLI COMMAND (copy-paste ready)")
    print("=" * 100)

    cli_args = []
    cfg = combined["optimal_config"]

    # Map config keys to CLI args
    cli_mapping = {
        "base_channels": "--base-channels",
        "norm_type": "--norm-type",
        "skip_type": "--skip-type",
        "activation": "--activation",
        "conv_type": "--conv-type",
        "n_downsample": "--n-downsample",
        "attention_type": "--attention-type",
        "n_heads": "--n-heads",
        "cond_mode": "--cond-mode",
        "dropout": "--dropout",
        "optimizer": "--optimizer",
        "lr_schedule": "--lr-schedule",
        "weight_decay": "--weight-decay",
        "loss_type": "--loss",
        "batch_size": "--batch-size",
    }

    for key, flag in cli_mapping.items():
        if key in cfg and cfg[key] is not None:
            cli_args.append(f"{flag} {cfg[key]}")

    # Boolean flags
    if cfg.get("use_adaptive_scaling"):
        cli_args.append("--use-adaptive-scaling")
    if cfg.get("bidirectional") is False:
        cli_args.append("--no-bidirectional")
    if cfg.get("use_augmentation") is False:
        cli_args.append("--no-aug")
    elif cfg.get("aug_strength"):
        cli_args.append(f"--aug-strength {cfg['aug_strength']}")

    print(f"\npython train.py --arch condunet \\")
    for i, arg in enumerate(cli_args):
        if i < len(cli_args) - 1:
            print(f"    {arg} \\")
        else:
            print(f"    {arg}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine Phase 3 JSON results")
    parser.add_argument(
        "--dir", "-d",
        type=Path,
        default=Path("results/phase3"),
        help="Directory containing Phase 3 results (default: results/phase3)"
    )
    args = parser.parse_args()

    combine_results(args.dir)
