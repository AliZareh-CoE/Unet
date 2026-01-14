#!/usr/bin/env python3
"""
Reconstruct Phase 2 results from individual fold JSON files.

Usage:
    python scripts/reconstruct_phase2_results.py /path/to/phase2/train_results/

This script:
1. Reads all *_fold*_results.json files
2. Aggregates metrics per architecture
3. Reports missing folds
4. Outputs a proper phase2_results.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from datetime import datetime


def main():
    if len(sys.argv) < 2:
        print("Usage: python reconstruct_phase2_results.py <train_results_dir>")
        print("Example: python reconstruct_phase2_results.py /path/to/phase2/train_results/")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"ERROR: Directory not found: {results_dir}")
        sys.exit(1)

    # Expected architectures and folds
    EXPECTED_ARCHS = ["condunet", "wavenet", "vit", "fnet", "simplecnn", "linear"]
    N_FOLDS = 5

    # Collect all results
    arch_results = defaultdict(list)
    found_files = list(results_dir.glob("*_fold*_results.json"))

    print(f"\n{'='*60}")
    print("Phase 2 Results Reconstruction")
    print(f"{'='*60}")
    print(f"Directory: {results_dir}")
    print(f"Found {len(found_files)} result files\n")

    # Parse each file
    for fpath in sorted(found_files):
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)

            arch = data.get("architecture", fpath.stem.split("_fold")[0])
            fold = data.get("fold", int(fpath.stem.split("fold")[1].split("_")[0]))

            arch_results[arch].append({
                "fold": fold,
                "best_val_r2": data.get("best_val_r2", 0.0),
                "best_val_mae": data.get("best_val_mae", 0.0),
                "best_epoch": data.get("best_epoch", 0),
                "n_parameters": data.get("n_parameters", 0),
                "epochs_trained": data.get("epochs_trained", 0),
                "train_losses": data.get("train_losses", []),
                "val_losses": data.get("val_losses", []),
                "val_r2s": data.get("val_r2s", []),
                "total_time": data.get("total_time", 0.0),
                "file": str(fpath.name),
            })
        except Exception as e:
            print(f"ERROR reading {fpath.name}: {e}")

    # Report status
    print("Architecture Status:")
    print("-" * 60)

    all_results = []
    aggregated = {}

    for arch in EXPECTED_ARCHS:
        results = arch_results.get(arch, [])
        folds_found = sorted([r["fold"] for r in results])
        folds_missing = [i for i in range(N_FOLDS) if i not in folds_found]

        status = "✓ COMPLETE" if len(results) == N_FOLDS else f"✗ MISSING folds: {folds_missing}"
        print(f"  {arch:12} : {len(results)}/{N_FOLDS} folds  {status}")

        if results:
            r2s = [r["best_val_r2"] for r in results]
            params = [r["n_parameters"] for r in results if r["n_parameters"] > 0]

            aggregated[arch] = {
                "r2_mean": float(np.mean(r2s)),
                "r2_std": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
                "r2_min": float(np.min(r2s)),
                "r2_max": float(np.max(r2s)),
                "n_folds": len(r2s),
                "fold_r2s": r2s,
                "n_parameters": params[0] if params else 0,
            }

            # Add individual fold results
            for r in results:
                all_results.append({
                    "architecture": arch,
                    "fold": r["fold"],
                    "best_val_r2": r["best_val_r2"],
                    "best_val_mae": r["best_val_mae"],
                    "best_epoch": r["best_epoch"],
                    "train_losses": r["train_losses"],
                    "val_losses": r["val_losses"],
                    "val_r2s": r["val_r2s"],
                    "total_time": r["total_time"],
                    "epochs_trained": r["epochs_trained"],
                    "n_parameters": r["n_parameters"],
                    "completed_successfully": True,
                })

    print("-" * 60)

    # Find best
    if aggregated:
        best_arch = max(aggregated.keys(), key=lambda k: aggregated[k]["r2_mean"])
        best_r2 = aggregated[best_arch]["r2_mean"]

        # Assume classical baseline R² (from Phase 1)
        classical_r2 = 0.35  # Default, adjust if you have the actual value
        gate_passed = best_r2 >= (classical_r2 + 0.10)

        print(f"\nBest Architecture: {best_arch} (R² = {best_r2:.4f})")
        print(f"Gate Threshold: {classical_r2 + 0.10:.4f} (classical R² + 0.10)")
        print(f"Gate Passed: {gate_passed}")

        # Results table
        print(f"\n{'='*60}")
        print("Results Summary (sorted by R²)")
        print(f"{'='*60}")
        print(f"{'Arch':<12} {'R² Mean':>10} {'± Std':>10} {'Folds':>8} {'Params':>12}")
        print("-" * 60)

        sorted_archs = sorted(aggregated.keys(), key=lambda k: aggregated[k]["r2_mean"], reverse=True)
        for arch in sorted_archs:
            stats = aggregated[arch]
            params_str = f"{stats['n_parameters']/1e6:.2f}M" if stats['n_parameters'] > 0 else "N/A"
            print(f"{arch:<12} {stats['r2_mean']:>10.4f} {stats['r2_std']:>10.4f} {stats['n_folds']:>8} {params_str:>12}")

        print("-" * 60)

        # Save reconstructed results
        output = {
            "results": all_results,
            "aggregated": aggregated,
            "best_architecture": best_arch,
            "best_r2": best_r2,
            "gate_passed": gate_passed,
            "classical_baseline_r2": classical_r2,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "config": {
                "n_folds": N_FOLDS,
                "reconstructed": True,
                "source_dir": str(results_dir),
            }
        }

        output_path = results_dir.parent / f"phase2_results_reconstructed_{output['timestamp']}.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nReconstructed results saved to:")
        print(f"  {output_path}")

        # Show what needs to be re-run
        missing_runs = []
        for arch in EXPECTED_ARCHS:
            results = arch_results.get(arch, [])
            folds_found = [r["fold"] for r in results]
            for fold in range(N_FOLDS):
                if fold not in folds_found:
                    missing_runs.append((arch, fold))

        if missing_runs:
            print(f"\n{'='*60}")
            print("MISSING RUNS (need to re-run these):")
            print(f"{'='*60}")
            for arch, fold in missing_runs:
                print(f"  python -m phase_two.runner --arch {arch} --fold {fold}")
            print(f"\nTotal missing: {len(missing_runs)} runs")
        else:
            print("\n✓ All runs complete!")

    else:
        print("\nERROR: No valid results found!")
        sys.exit(1)


if __name__ == "__main__":
    main()
