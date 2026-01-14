#!/usr/bin/env python3
"""
Create phase2_checkpoint.pkl from existing train_results/*.json files.

This allows --resume to recognize already-completed runs.

Usage:
    python scripts/create_checkpoint_from_results.py /path/to/phase2/
"""

import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class TrainingResult:
    """Training result for a single architecture/fold."""
    architecture: str
    fold: int
    best_val_r2: float
    best_val_mae: float
    best_epoch: int
    train_losses: List[float]
    val_losses: List[float]
    val_r2s: List[float]
    total_time: float
    epochs_trained: int
    n_parameters: int
    nan_detected: bool = False
    nan_recovery_count: int = 0
    early_stopped: bool = False
    error_message: Optional[str] = None
    peak_gpu_memory_mb: Optional[float] = None
    completed_successfully: bool = True
    dsp_metrics: Optional[Dict[str, float]] = None


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_checkpoint_from_results.py <phase2_dir>")
        print("Example: python create_checkpoint_from_results.py results/phase2/")
        sys.exit(1)

    phase2_dir = Path(sys.argv[1])
    train_results_dir = phase2_dir / "train_results"

    if not train_results_dir.exists():
        print(f"ERROR: train_results directory not found: {train_results_dir}")
        sys.exit(1)

    # Load all existing results
    all_results = []
    arch_r2s = defaultdict(list)
    completed_runs = set()

    json_files = list(train_results_dir.glob("*_fold*_results.json"))
    print(f"\nFound {len(json_files)} result files in {train_results_dir}")

    for fpath in sorted(json_files):
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)

            arch = data.get("architecture", fpath.stem.split("_fold")[0])
            fold = data.get("fold", int(fpath.stem.split("fold")[1].split("_")[0]))

            result = TrainingResult(
                architecture=arch,
                fold=fold,
                best_val_r2=data.get("best_val_r2", 0.0),
                best_val_mae=data.get("best_val_mae", 0.0),
                best_epoch=data.get("best_epoch", 0),
                train_losses=data.get("train_losses", []),
                val_losses=data.get("val_losses", []),
                val_r2s=data.get("val_r2s", []),
                total_time=data.get("total_time", 0.0),
                epochs_trained=data.get("epochs_trained", 0),
                n_parameters=data.get("n_parameters", 0),
                completed_successfully=data.get("completed_successfully", True),
            )

            all_results.append(result)
            arch_r2s[arch].append(result.best_val_r2)
            completed_runs.add((arch, fold))

            print(f"  Loaded: {arch} fold {fold} (R²={result.best_val_r2:.4f})")

        except Exception as e:
            print(f"  ERROR reading {fpath.name}: {e}")

    if not all_results:
        print("\nERROR: No valid results found!")
        sys.exit(1)

    # Create checkpoint
    checkpoint = {
        "all_results": all_results,
        "arch_r2s": dict(arch_r2s),
        "all_models": {},  # Models not loaded, just tracking progress
        "completed_runs": list(completed_runs),
    }

    # Save checkpoint
    checkpoint_path = phase2_dir / "phase2_checkpoint.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"\n{'='*60}")
    print(f"Checkpoint created: {checkpoint_path}")
    print(f"{'='*60}")
    print(f"Completed runs: {len(completed_runs)}")

    # Show what's in the checkpoint
    archs_found = set(arch for arch, _ in completed_runs)
    for arch in sorted(archs_found):
        folds = sorted([f for a, f in completed_runs if a == arch])
        r2s = arch_r2s[arch]
        print(f"  {arch}: folds {folds}, mean R²={sum(r2s)/len(r2s):.4f}")

    # Show what's missing
    expected_archs = ["linear", "simplecnn", "wavenet", "fnet", "vit", "condunet"]
    n_folds = 5
    missing = []
    for arch in expected_archs:
        for fold in range(n_folds):
            if (arch, fold) not in completed_runs:
                missing.append((arch, fold))

    if missing:
        print(f"\nMISSING RUNS ({len(missing)}):")
        for arch, fold in missing:
            print(f"  {arch} fold {fold}")
        print(f"\nNow run: python -m phase_two.runner --use-train-py --fsdp --resume")
        print("It will skip completed runs and only train the missing ones.")
    else:
        print("\n✓ All runs complete!")


if __name__ == "__main__":
    main()
