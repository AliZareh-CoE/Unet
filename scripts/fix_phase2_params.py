#!/usr/bin/env python3
"""
Fix parameter counts in Phase 2 results by recalculating from actual models.

Usage:
    python scripts/fix_phase2_params.py /path/to/results/phase2/
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_phase2_params.py <phase2_dir>")
        sys.exit(1)

    phase2_dir = Path(sys.argv[1])

    # Import architectures
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from phase_two.architectures import create_architecture

    # Calculate correct parameter counts
    print("\nCalculating correct parameter counts...")
    print("=" * 50)

    correct_params = {}
    archs = ["linear", "simplecnn", "wavenet", "vit", "condunet"]

    for arch in archs:
        model = create_architecture(arch, in_channels=32, out_channels=32, time_steps=5000)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        correct_params[arch] = n_params
        print(f"  {arch:<12}: {n_params:>12,}")

    print("=" * 50)

    # Update train_results JSON files
    train_results_dir = phase2_dir / "train_results"
    if train_results_dir.exists():
        print(f"\nUpdating files in {train_results_dir}...")
        for json_file in train_results_dir.glob("*_results.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)

            arch = data.get("architecture")
            if arch in correct_params:
                old_params = data.get("n_parameters", 0)
                new_params = correct_params[arch]
                if old_params != new_params:
                    data["n_parameters"] = new_params
                    with open(json_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"  Fixed {json_file.name}: {old_params:,} → {new_params:,}")

    # Update main results JSON
    results_files = list(phase2_dir.glob("phase2_results_*.json"))
    for results_file in results_files:
        print(f"\nUpdating {results_file.name}...")
        with open(results_file, 'r') as f:
            data = json.load(f)

        # Fix individual results
        for result in data.get("results", []):
            arch = result.get("architecture")
            if arch in correct_params:
                result["n_parameters"] = correct_params[arch]

        # Save updated file
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Updated {len(data.get('results', []))} entries")

    print("\n✓ Parameter counts fixed!")
    print("\nCorrect counts:")
    for arch, params in correct_params.items():
        print(f"  {arch:<12}: {params:>12,}")


if __name__ == "__main__":
    main()
