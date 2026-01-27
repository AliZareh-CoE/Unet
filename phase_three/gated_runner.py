#!/usr/bin/env python3
"""
GatedTranslator Runner for Phase 3
==================================

Trains the GatedTranslator which learns when brain regions are communicating
vs when translation is just fitting noise.

Architecture:
    y_pred[t] = gate[t] * y_translated[t] + (1 - gate[t]) * y_baseline[t]

Training dynamics:
    - When regions communicate: translation accurate → gate ON
    - When regions don't: translation is noise → gate OFF
    - Sparsity penalty encourages gate OFF by default

Uses train.py as subprocess (same pattern as other Phase 3 runners).
Supports FSDP and BFloat16 via torchrun.

Usage:
    # Single GPU
    python -m phase_three.gated_runner

    # Multi-GPU with FSDP and bf16
    python -m phase_three.gated_runner --fsdp

    # DANDI dataset
    python -m phase_three.gated_runner --dataset dandi

    # Custom sparsity settings
    python -m phase_three.gated_runner --sparsity-target 0.2 --sparsity-weight 0.05

    # 3-fold cross-validation
    python -m phase_three.gated_runner --n-folds 3 --n-seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# JSON Encoder
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class GatedFoldResult:
    """Result from a single fold."""
    fold_idx: int
    seed: int
    seed_idx: int

    # Metrics
    train_r2: float = 0.0
    val_r2: float = 0.0
    test_r2: float = 0.0

    train_corr: float = 0.0
    val_corr: float = 0.0
    test_corr: float = 0.0

    # Gate statistics
    gate_mean: float = 0.0

    # Timing
    train_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_idx": self.fold_idx,
            "seed": self.seed,
            "seed_idx": self.seed_idx,
            "train_r2": self.train_r2,
            "val_r2": self.val_r2,
            "test_r2": self.test_r2,
            "train_corr": self.train_corr,
            "val_corr": self.val_corr,
            "test_corr": self.test_corr,
            "gate_mean": self.gate_mean,
            "train_time": self.train_time,
        }


@dataclass
class GatedResult:
    """Aggregated results across all folds."""
    fold_results: List[GatedFoldResult] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    mean_test_r2: float = 0.0
    std_test_r2: float = 0.0
    mean_val_r2: float = 0.0
    std_val_r2: float = 0.0
    mean_gate: float = 0.0

    def compute_statistics(self):
        if not self.fold_results:
            return
        test_r2s = [r.test_r2 for r in self.fold_results]
        val_r2s = [r.val_r2 for r in self.fold_results]
        gate_means = [r.gate_mean for r in self.fold_results]

        self.mean_test_r2 = float(np.mean(test_r2s))
        self.std_test_r2 = float(np.std(test_r2s))
        self.mean_val_r2 = float(np.mean(val_r2s))
        self.std_val_r2 = float(np.std(val_r2s))
        self.mean_gate = float(np.mean(gate_means))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_test_r2": self.mean_test_r2,
            "std_test_r2": self.std_test_r2,
            "mean_val_r2": self.mean_val_r2,
            "std_val_r2": self.std_val_r2,
            "mean_gate": self.mean_gate,
            "config": self.config,
            "fold_results": [r.to_dict() for r in self.fold_results],
        }


# =============================================================================
# 3-Fold Session Splits (Same as Phase 3 runner)
# =============================================================================

def get_3fold_session_splits(all_sessions: List[str]) -> List[Dict[str, Any]]:
    """Create 3-fold CV splits where each fold holds out ~3 sessions."""
    n_sessions = len(all_sessions)
    if n_sessions < 6:
        raise ValueError(f"Need at least 6 sessions for 3-fold CV, got {n_sessions}")

    sessions_per_fold = n_sessions // 3
    remainder = n_sessions % 3

    splits = []
    start_idx = 0

    for fold_idx in range(3):
        n_test = sessions_per_fold + (1 if fold_idx < remainder else 0)
        end_idx = start_idx + n_test

        test_sessions = all_sessions[start_idx:end_idx]
        train_sessions = all_sessions[:start_idx] + all_sessions[end_idx:]

        splits.append({
            "fold_idx": fold_idx,
            "test_sessions": test_sessions,
            "train_sessions": train_sessions,
        })
        start_idx = end_idx

    return splits


# =============================================================================
# Training via subprocess
# =============================================================================

def run_training(
    fold_idx: int,
    seed: int,
    seed_idx: int,
    config: Dict[str, Any],
    output_dir: Path,
    use_fsdp: bool = False,
    verbose: bool = True,
) -> Optional[GatedFoldResult]:
    """Run training for a single fold by calling train.py."""
    import torch

    start_time = time.time()

    # Build command
    if use_fsdp:
        nproc = torch.cuda.device_count() if torch.cuda.is_available() else 1
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            str(PROJECT_ROOT / "train.py"),
        ]
    else:
        cmd = [sys.executable, str(PROJECT_ROOT / "train.py")]

    # Checkpoint path
    checkpoint_dir = output_dir / f"fold{fold_idx}_seed{seed_idx}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"
    results_file = checkpoint_dir / "results.json"

    # Base arguments
    cmd.extend([
        "--dataset", config["dataset"],
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--seed", str(seed),
        # Model architecture (use LOSO-optimized defaults)
        "--arch", "condunet",
        "--base-channels", str(config.get("base_channels", 64)),
        "--n-downsample", str(config.get("n_downsample", 2)),
        "--attention-type", config.get("attention_type", "none"),
        "--cond-mode", config.get("cond_mode", "cross_attn_gated"),
        "--conv-type", config.get("conv_type", "modern"),
        "--activation", config.get("activation", "gelu"),
        "--skip-type", config.get("skip_type", "add"),
        "--conditioning", config.get("conditioning", "spectro_temporal"),
        # Optimizer
        "--optimizer", config.get("optimizer", "adamw"),
        "--lr-schedule", config.get("lr_schedule", "cosine_warmup"),
        # Gated options
        "--gated",
        "--gated-sparsity-weight", str(config["sparsity_weight"]),
        "--gated-sparsity-target", str(config["sparsity_target"]),
        "--gated-gate-channels", str(config.get("gate_channels", 64)),
        "--gated-baseline-channels", str(config.get("baseline_channels", 64)),
        # Output
        "--checkpoint-prefix", str(checkpoint_path.stem),
        "--output-results-file", str(results_file),
        # Disable plots for speed
        "--no-plots",
        # Quiet mode
        "--quiet",
        # Disable bidirectional (not supported with gated)
        "--no-bidirectional",
    ])

    # DANDI-specific arguments
    if config["dataset"] == "dandi":
        cmd.extend([
            "--dandi-source-region", config.get("dandi_source_region", "amygdala"),
            "--dandi-target-region", config.get("dandi_target_region", "hippocampus"),
            "--dandi-window-size", str(config.get("dandi_window_size", 5000)),
            "--dandi-stride-ratio", str(config.get("dandi_stride_ratio", 0.5)),
        ])

    # FSDP options
    if use_fsdp:
        cmd.extend(["--fsdp"])

    # Fold-specific test sessions (if using session splits)
    if config.get("test_sessions"):
        cmd.extend(["--test-sessions"] + config["test_sessions"])

    # Environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    # Run training
    try:
        if verbose:
            print(f"\n  Running: {' '.join(cmd[:5])}...")
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
            stdout_lines = []
            for line in process.stdout:
                stdout_lines.append(line)
                # Filter to show only key lines
                if any(x in line for x in ["Epoch", "R²", "loss", "gate", "Best", "Final", "Error", "error", "Exception", "Traceback"]):
                    print(f"    {line.rstrip()}")
            process.wait()
            stderr_output = process.stderr.read()
            if process.returncode != 0:
                print(f"  STDERR: {stderr_output[:2000]}")
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr=stderr_output)
        else:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed for fold {fold_idx}")
        print(f"  Command: {' '.join(cmd)}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"  stderr: {e.stderr[:2000]}")
        return None

    elapsed = time.time() - start_time

    # Load results
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)

        return GatedFoldResult(
            fold_idx=fold_idx,
            seed=seed,
            seed_idx=seed_idx,
            train_r2=results.get("train_r2", 0.0),
            val_r2=results.get("val_r2", 0.0),
            test_r2=results.get("test_r2", 0.0),
            train_corr=results.get("train_corr", 0.0),
            val_corr=results.get("val_corr", 0.0),
            test_corr=results.get("test_corr", 0.0),
            gate_mean=results.get("gate_mean", 0.0),
            train_time=elapsed,
        )
    else:
        print(f"  Warning: Results file not found: {results_file}")
        return GatedFoldResult(
            fold_idx=fold_idx,
            seed=seed,
            seed_idx=seed_idx,
            train_time=elapsed,
        )


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GatedTranslator Phase 3 Runner")

    # Dataset
    parser.add_argument("--dataset", type=str, default="olfactory",
                        choices=["olfactory", "pfc", "dandi"])

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)

    # Gating parameters
    parser.add_argument("--sparsity-weight", type=float, default=0.01,
                        help="Weight for gate sparsity penalty")
    parser.add_argument("--sparsity-target", type=float, default=0.3,
                        help="Target gate activation rate (0-1)")
    parser.add_argument("--gate-channels", type=int, default=64,
                        help="Hidden channels for gate network")
    parser.add_argument("--baseline-channels", type=int, default=64,
                        help="Hidden channels for baseline predictor")

    # Model architecture (LOSO-optimized defaults)
    parser.add_argument("--n-downsample", type=int, default=2)
    parser.add_argument("--attention-type", type=str, default="none")
    parser.add_argument("--cond-mode", type=str, default="cross_attn_gated")
    parser.add_argument("--conv-type", type=str, default="modern")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--skip-type", type=str, default="add")
    parser.add_argument("--conditioning", type=str, default="spectro_temporal")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr-schedule", type=str, default="cosine_warmup")

    # DANDI-specific
    parser.add_argument("--dandi-source-region", type=str, default="amygdala")
    parser.add_argument("--dandi-target-region", type=str, default="hippocampus")
    parser.add_argument("--dandi-window-size", type=int, default=5000)
    parser.add_argument("--dandi-stride-ratio", type=float, default=0.5)

    # Cross-validation
    parser.add_argument("--n-folds", type=int, default=3,
                        help="Number of CV folds")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of seeds per fold")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run specific fold only")

    # Distributed
    parser.add_argument("--fsdp", action="store_true",
                        help="Use FSDP with torchrun")

    # Output
    parser.add_argument("--output-dir", type=str, default="artifacts/phase3_gated")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    verbose = args.verbose and not args.quiet

    print("=" * 70)
    print("Phase 3: GatedTranslator Evaluation")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Sparsity target: {args.sparsity_target}, weight: {args.sparsity_weight}")
    print(f"FSDP: {args.fsdp}")
    print(f"Folds: {args.n_folds}, Seeds per fold: {args.n_seeds}")
    print("=" * 70)

    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "base_channels": args.base_channels,
        # Gating
        "sparsity_weight": args.sparsity_weight,
        "sparsity_target": args.sparsity_target,
        "gate_channels": args.gate_channels,
        "baseline_channels": args.baseline_channels,
        # Model architecture
        "n_downsample": args.n_downsample,
        "attention_type": args.attention_type,
        "cond_mode": args.cond_mode,
        "conv_type": args.conv_type,
        "activation": args.activation,
        "skip_type": args.skip_type,
        "conditioning": args.conditioning,
        "optimizer": args.optimizer,
        "lr_schedule": args.lr_schedule,
        # DANDI-specific
        "dandi_source_region": args.dandi_source_region,
        "dandi_target_region": args.dandi_target_region,
        "dandi_window_size": args.dandi_window_size,
        "dandi_stride_ratio": args.dandi_stride_ratio,
    }

    if args.dry_run:
        print("\n[DRY RUN] Would train with config:")
        print(json.dumps(config, indent=2))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate seeds
    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]

    # Run folds
    result = GatedResult(config=config)
    folds = [args.fold] if args.fold is not None else range(args.n_folds)

    for fold_idx in folds:
        for seed_idx, seed in enumerate(seeds):
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx + 1}/{args.n_folds}, Seed {seed_idx + 1}/{args.n_seeds} (seed={seed})")
            print(f"{'='*60}")

            fold_result = run_training(
                fold_idx=fold_idx,
                seed=seed,
                seed_idx=seed_idx,
                config=config,
                output_dir=output_dir,
                use_fsdp=args.fsdp,
                verbose=verbose,
            )

            if fold_result is not None:
                result.fold_results.append(fold_result)
                print(f"  Test R²: {fold_result.test_r2:.4f}")
                print(f"  Gate mean: {fold_result.gate_mean:.3f}")
            else:
                print(f"  FAILED")

    # Compute statistics
    result.compute_statistics()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"gated_results_{args.dataset}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Mean test R²: {result.mean_test_r2:.4f} ± {result.std_test_r2:.4f}")
    print(f"Mean val R²:  {result.mean_val_r2:.4f} ± {result.std_val_r2:.4f}")
    print(f"Mean gate:    {result.mean_gate:.3f}")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
