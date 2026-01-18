#!/usr/bin/env python3
"""
Combine Phase 3 JSON Results
============================
Combines all Phase 3 result JSON files into a single comprehensive summary.
"""

import json
from pathlib import Path
from datetime import datetime


def combine_results(results_dir: Path = Path("results/phase3")):
    """Combine all Phase 3 JSON results into a single summary."""

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
        "group_results": {},
        "optimal_config": {},
        "winner_summary": [],
    }

    # Load and merge all results
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Extract group results
            if "group_results" in data:
                for group_id, group_data in data["group_results"].items():
                    if group_id not in combined["group_results"]:
                        combined["group_results"][group_id] = group_data
                    else:
                        # Merge if exists (keep better R²)
                        existing = combined["group_results"][group_id]
                        if group_data.get("winner_r2", 0) > existing.get("winner_r2", 0):
                            combined["group_results"][group_id] = group_data

            # Extract optimal config (use latest)
            if "optimal_config" in data:
                combined["optimal_config"].update(data["optimal_config"])

            # Extract final config
            if "final_config" in data:
                combined["optimal_config"].update(data["final_config"])

        except Exception as e:
            print(f"  Warning: Could not load {json_file.name}: {e}")

    # Build winner summary table
    for group_id in sorted(combined["group_results"].keys(), key=lambda x: int(x)):
        group_data = combined["group_results"][group_id]
        combined["winner_summary"].append({
            "group": int(group_id),
            "name": group_data.get("group_name", "unknown"),
            "winner": group_data.get("winner", "unknown"),
            "r2": group_data.get("winner_r2", 0),
            "improvement": group_data.get("improvement", 0),
        })

    # Save combined results
    output_path = results_dir / "phase3_combined_results.json"
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved combined results to {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("PHASE 3 COMBINED RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Group':<6} {'Name':<22} {'Winner':<20} {'R²':<10} {'Δ':<10}")
    print("-" * 70)

    total_improvement = 0
    baseline_r2 = None
    final_r2 = None

    for row in combined["winner_summary"]:
        if baseline_r2 is None:
            baseline_r2 = row["r2"]
        final_r2 = row["r2"]
        total_improvement += row.get("improvement", 0)

        print(f"{row['group']:<6} {row['name']:<22} {row['winner']:<20} "
              f"{row['r2']:<10.4f} {row['improvement']:+.4f}")

    print("-" * 70)
    if baseline_r2 and final_r2:
        print(f"Total: {baseline_r2:.4f} → {final_r2:.4f} (Δ = {final_r2 - baseline_r2:+.4f})")

    # Print optimal configuration
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATION")
    print("=" * 70)

    for key, value in sorted(combined["optimal_config"].items()):
        print(f"  {key}: {value}")

    return combined


if __name__ == "__main__":
    combine_results()
