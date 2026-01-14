#!/usr/bin/env python3
"""
Re-run only missing Phase 2 folds using existing fold indices.

Usage:
    python scripts/rerun_missing_folds.py /path/to/phase2/ [--fsdp]

This uses your existing fold_indices/*.pkl files to ensure
the same train/val splits as your completed runs.
"""

import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python rerun_missing_folds.py <phase2_dir> [--fsdp]")
        sys.exit(1)

    phase2_dir = Path(sys.argv[1])
    use_fsdp = "--fsdp" in sys.argv

    fold_indices_dir = phase2_dir / "fold_indices"
    train_results_dir = phase2_dir / "train_results"

    if not fold_indices_dir.exists():
        print(f"ERROR: fold_indices directory not found: {fold_indices_dir}")
        sys.exit(1)

    # Find what fold indices exist
    available_indices = {}
    for pkl in fold_indices_dir.glob("*_fold*_indices.pkl"):
        parts = pkl.stem.replace("_indices", "").split("_fold")
        arch = parts[0]
        fold = int(parts[1])
        if arch not in available_indices:
            available_indices[arch] = []
        available_indices[arch].append((fold, pkl))

    # Find what results already exist
    existing_results = set()
    if train_results_dir.exists():
        for json_file in train_results_dir.glob("*_fold*_results.json"):
            parts = json_file.stem.replace("_results", "").split("_fold")
            arch = parts[0]
            fold = int(parts[1])
            existing_results.add((arch, fold))

    # Determine what's missing
    missing = []
    for arch, folds in available_indices.items():
        for fold, pkl_path in folds:
            if (arch, fold) not in existing_results:
                missing.append((arch, fold, pkl_path))

    print(f"\n{'='*60}")
    print("Phase 2 Missing Folds Analysis")
    print(f"{'='*60}")
    print(f"Fold indices dir: {fold_indices_dir}")
    print(f"Train results dir: {train_results_dir}")
    print(f"\nArchitectures with indices: {list(available_indices.keys())}")
    print(f"Existing results: {len(existing_results)}")
    print(f"Missing runs: {len(missing)}")

    if not missing:
        print("\n✓ All folds complete! Nothing to re-run.")
        return

    print(f"\nMissing runs to execute:")
    print("-" * 60)
    for arch, fold, pkl_path in sorted(missing):
        print(f"  {arch} fold {fold}: {pkl_path.name}")

    print(f"\n{'='*60}")
    response = input("Run these missing folds now? [y/N]: ").strip().lower()

    if response != 'y':
        print("Aborted.")
        return

    # Run each missing fold
    train_results_dir.mkdir(parents=True, exist_ok=True)

    for arch, fold, pkl_path in sorted(missing):
        print(f"\n{'='*60}")
        print(f"Running: {arch} fold {fold}")
        print(f"{'='*60}")

        output_file = train_results_dir / f"{arch}_fold{fold}_results.json"

        cmd = [
            sys.executable, "-m", "phase_two.runner",
            "--arch", arch,
            "--use-train-py",
            "--epochs", "60",
        ]

        if use_fsdp:
            cmd.append("--fsdp")

        # The runner will use existing fold indices automatically
        print(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Completed: {arch} fold {fold}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {arch} fold {fold} (exit code {e.returncode})")
            continue

    print(f"\n{'='*60}")
    print("Re-run complete!")


if __name__ == "__main__":
    main()
