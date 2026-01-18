#!/usr/bin/env python3
"""
Combine Phase 3 JSON Results
============================
Combines all Phase 3 result JSON files into a single comprehensive summary.

Handles two formats:
1. Individual training results from train.py (g{group_id}_{variant}_results.json)
2. Combined phase3_results.json from the runner

The script extracts group_id and variant from file names, aggregates results,
and identifies the winner (best R²) for each ablation group.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from phase_three.config import ABLATION_GROUPS


def get_group_name(group_id: int) -> str:
    """Get group name from ABLATION_GROUPS config."""
    for group in ABLATION_GROUPS:
        if group["group_id"] == group_id:
            return group["study"]
    return f"group_{group_id}"


def parse_result_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Parse group_id and variant from result filename.

    Expected formats:
    - g{group_id}_{variant}_results.json (e.g., g1_batch_norm_results.json)
    - {study}_{variant}_fold{fold}_results.json (e.g., normalization_batch_norm_fold0_results.json)
    - baseline_full_fold{fold}_results.json (e.g., baseline_full_fold0_results.json)
    """
    # Format: g{group_id}_{variant}_results.json
    match = re.match(r'g(\d+)_(.+)_results\.json', filename)
    if match:
        return {
            "group_id": int(match.group(1)),
            "variant": match.group(2),
            "fold": None
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
            if group["study"] == study:
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
        "optimal_config": {},
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

            if group_id is None:
                print(f"  Warning: Unknown group for {json_file.name}")
                continue

            # Extract metrics from individual training result
            result_entry = {
                "file": str(json_file.relative_to(results_dir)),
                "group_id": group_id,
                "variant": variant,
                "fold": fold,
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
            if group_id not in group_variants:
                group_variants[group_id] = {}
            if variant not in group_variants[group_id]:
                group_variants[group_id][variant] = []
            group_variants[group_id][variant].append(result_entry)

        except Exception as e:
            print(f"  Warning: Could not load {json_file.name}: {e}")

    # Build group results with statistics
    for group_id in sorted(group_variants.keys()):
        group_name = get_group_name(group_id) if group_id > 0 else "baseline"

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
            }

            # Track best variant
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_variant = variant

        combined["group_results"][str(group_id)] = {
            "group_name": group_name,
            "variants": variants_data,
            "winner": best_variant,
            "winner_r2": best_r2,
        }

        # Update optimal config
        if group_id > 0 and best_variant:
            combined["optimal_config"][group_name] = best_variant

    # Calculate improvements (relative to baseline or previous group)
    baseline_r2 = combined["group_results"].get("0", {}).get("winner_r2", None)
    prev_r2 = baseline_r2

    for group_id in sorted(group_variants.keys()):
        group_data = combined["group_results"][str(group_id)]
        current_r2 = group_data["winner_r2"]

        if prev_r2 is not None:
            improvement = current_r2 - prev_r2
            group_data["improvement"] = improvement
        else:
            group_data["improvement"] = 0

        prev_r2 = current_r2

    # Build winner summary table
    for group_id in sorted(group_variants.keys()):
        group_data = combined["group_results"][str(group_id)]
        combined["winner_summary"].append({
            "group": group_id,
            "name": group_data["group_name"],
            "winner": group_data["winner"],
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
    print("\n" + "=" * 80)
    print("PHASE 3 COMBINED RESULTS SUMMARY")
    print("=" * 80)

    # Print winner summary table
    print(f"\n{'Group':<6} {'Name':<22} {'Winner':<20} {'R²':<10} {'Δ R²':<10}")
    print("-" * 80)

    baseline_r2 = None
    final_r2 = None

    for row in combined["winner_summary"]:
        if baseline_r2 is None:
            baseline_r2 = row["r2"]
        final_r2 = row["r2"]

        # Format improvement
        imp_str = f"{row['improvement']:+.4f}" if row['improvement'] != 0 else "-"

        print(f"{row['group']:<6} {row['name']:<22} {row['winner']:<20} "
              f"{row['r2']:<10.4f} {imp_str:<10}")

    print("-" * 80)

    if baseline_r2 and final_r2:
        total_improvement = final_r2 - baseline_r2
        print(f"Total: {baseline_r2:.4f} → {final_r2:.4f} (Δ = {total_improvement:+.4f})")

    # Print variant details for each group
    print("\n" + "=" * 80)
    print("DETAILED VARIANT RESULTS")
    print("=" * 80)

    for group_id_str in sorted(combined["group_results"].keys(), key=lambda x: int(x)):
        group_data = combined["group_results"][group_id_str]
        print(f"\n[Group {group_id_str}] {group_data['group_name']}")
        print("-" * 50)

        # Sort variants by R²
        variants = sorted(
            group_data["variants"].items(),
            key=lambda x: x[1]["mean_r2"],
            reverse=True
        )

        for variant, stats in variants:
            marker = " ★ WINNER" if variant == group_data["winner"] else ""
            print(f"  {variant:<25} R²={stats['mean_r2']:.4f} "
                  f"(MAE={stats['mean_mae']:.4f}, corr={stats['mean_corr']:.4f}){marker}")

    # Print optimal configuration
    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATION")
    print("=" * 80)

    for key, value in sorted(combined["optimal_config"].items()):
        print(f"  {key}: {value}")


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
