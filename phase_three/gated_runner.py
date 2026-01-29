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

Uses train.py as subprocess (same pattern as LOSO runner).
Supports FSDP and BFloat16 via torchrun.

Usage:
    # Single GPU - olfactory
    python -m phase_three.gated_runner

    # Multi-GPU with FSDP - DANDI
    python -m phase_three.gated_runner --dataset dandi --fsdp

    # Custom sparsity
    python -m phase_three.gated_runner --sparsity-target 0.2 --sparsity-weight 0.05
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

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
# Configuration (following LOSO pattern)
# =============================================================================

@dataclass
class GatedConfig:
    """Configuration for GatedTranslator training."""

    # Dataset
    dataset: str = "olfactory"  # olfactory, pfc, pcx1, dandi
    output_dir: Path = field(default_factory=lambda: Path("artifacts/phase3_gated"))

    # Training
    epochs: int = 50
    batch_size: int = 64  # Optimized: 64
    learning_rate: float = 1e-4
    seed: int = 42
    n_seeds: int = 1

    # Model architecture (LOSO-optimized defaults)
    arch: str = "condunet"
    base_channels: int = 256  # Optimized: 256
    n_downsample: int = 2
    attention_type: str = "none"
    cond_mode: str = "cross_attn_gated"
    conv_type: str = "modern"
    activation: str = "gelu"
    skip_type: str = "add"
    n_heads: int = 4
    conditioning: str = "spectro_temporal"

    # Optimizer
    optimizer: str = "adamw"
    lr_schedule: str = "cosine_warmup"
    weight_decay: float = 0.0
    dropout: float = 0.0

    # Gated translator specific
    sparsity_weight: float = 0.01
    sparsity_target: float = 0.3
    gate_channels: int = 64
    baseline_channels: int = 64

    # FSDP
    use_fsdp: bool = False
    fsdp_strategy: str = "full"

    # Execution
    verbose: bool = True
    generate_plots: bool = False
    n_folds: int = 3
    folds_to_run: Optional[List[int]] = None

    # DANDI-specific
    dandi_source_region: str = "amygdala"
    dandi_target_region: str = "hippocampus"
    dandi_window_size: int = 5000
    dandi_stride_ratio: float = 0.5

    # PCx1-specific
    pcx1_window_size: int = 5000
    pcx1_stride_ratio: float = 0.5

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result


# =============================================================================
# Result Dataclasses (following LOSO pattern)
# =============================================================================

@dataclass
class GatedFoldResult:
    """Result from a single fold."""
    fold_idx: int
    seed: int
    seed_idx: int

    # Metrics (from train.py results)
    test_r2: float = 0.0
    test_corr: float = 0.0
    val_r2: float = 0.0
    val_corr: float = 0.0
    train_loss: float = 0.0

    # Gate statistics
    gate_mean: float = 0.0

    # Timing
    total_time: float = 0.0

    # Config snapshot
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
        test_r2s = [r.test_r2 for r in self.fold_results if r.test_r2 is not None]
        val_r2s = [r.val_r2 for r in self.fold_results if r.val_r2 is not None]
        gate_means = [r.gate_mean for r in self.fold_results if r.gate_mean is not None]

        if test_r2s:
            self.mean_test_r2 = float(np.mean(test_r2s))
            self.std_test_r2 = float(np.std(test_r2s))
        if val_r2s:
            self.mean_val_r2 = float(np.mean(val_r2s))
            self.std_val_r2 = float(np.std(val_r2s))
        if gate_means:
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
# Single Fold Training (following LOSO pattern)
# =============================================================================

def run_single_fold(
    fold_idx: int,
    config: GatedConfig,
    output_results_file: Path,
    seed_idx: int = 0,
) -> Optional[GatedFoldResult]:
    """Run a single fold by calling train.py.

    Args:
        fold_idx: Fold index (0-indexed)
        config: Gated configuration
        output_results_file: Path to save train.py results
        seed_idx: Seed index within this fold

    Returns:
        GatedFoldResult or None on failure
    """
    print(f"\n{'='*70}")
    print(f"GATED FOLD {fold_idx + 1}/{config.n_folds}")
    print(f"{'='*70}")
    print(f"Dataset: {config.dataset}")
    print(f"Sparsity target: {config.sparsity_target}, weight: {config.sparsity_weight}")
    print()

    # Build train.py command
    if config.use_fsdp:
        nproc = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # Use unique port per fold to avoid "address already in use" errors
        master_port = 29500 + fold_idx
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            f"--master_port={master_port}",
            str(PROJECT_ROOT / "train.py"),
        ]
    else:
        cmd = [sys.executable, str(PROJECT_ROOT / "train.py")]

    # Seed formula ensures unique seeds across all folds and seed runs
    run_seed = config.seed + fold_idx * config.n_seeds + seed_idx

    # Base arguments
    cmd.extend([
        "--arch", config.arch,
        "--epochs", str(config.epochs),
        "--batch-size", str(config.batch_size),
        "--lr", str(config.learning_rate),
        "--seed", str(run_seed),
        "--output-results-file", str(output_results_file),
        "--fold", str(fold_idx),
    ])

    # Skip plots for speed
    if not config.generate_plots:
        cmd.append("--no-plots")

    # Dataset
    if config.dataset != "olfactory":
        cmd.extend(["--dataset", config.dataset])

    # Dataset-specific arguments
    if config.dataset == "dandi":
        cmd.extend([
            "--dandi-source-region", config.dandi_source_region,
            "--dandi-target-region", config.dandi_target_region,
            "--dandi-window-size", str(config.dandi_window_size),
            "--dandi-stride-ratio", str(config.dandi_stride_ratio),
        ])
    elif config.dataset == "pcx1":
        cmd.extend([
            "--pcx1-window-size", str(config.pcx1_window_size),
            "--pcx1-stride-ratio", str(config.pcx1_stride_ratio),
        ])

    # Model architecture arguments
    if config.base_channels:
        cmd.extend(["--base-channels", str(config.base_channels)])
    if config.n_downsample:
        cmd.extend(["--n-downsample", str(config.n_downsample)])
    if config.attention_type:
        cmd.extend(["--attention-type", config.attention_type])
    if config.cond_mode:
        cmd.extend(["--cond-mode", config.cond_mode])
    if config.conv_type:
        cmd.extend(["--conv-type", config.conv_type])
    if config.activation:
        cmd.extend(["--activation", config.activation])
    if config.skip_type:
        cmd.extend(["--skip-type", config.skip_type])
    if config.n_heads is not None:
        cmd.extend(["--n-heads", str(config.n_heads)])
    if config.conditioning and config.conditioning.lower() != "none":
        cmd.extend(["--conditioning", config.conditioning])

    # Training arguments
    if config.optimizer:
        cmd.extend(["--optimizer", config.optimizer])
    if config.lr_schedule:
        cmd.extend(["--lr-schedule", config.lr_schedule])
    if config.weight_decay > 0:
        cmd.extend(["--weight-decay", str(config.weight_decay)])
    if config.dropout > 0:
        cmd.extend(["--dropout", str(config.dropout)])

    # Disable bidirectional (not supported with gated)
    cmd.append("--no-bidirectional")

    # Gated translator arguments
    cmd.extend([
        "--gated",
        "--gated-sparsity-weight", str(config.sparsity_weight),
        "--gated-sparsity-target", str(config.sparsity_target),
        "--gated-gate-channels", str(config.gate_channels),
        "--gated-baseline-channels", str(config.baseline_channels),
    ])

    # FSDP
    if config.use_fsdp:
        cmd.extend(["--fsdp", "--fsdp-strategy", config.fsdp_strategy])

    # Set NCCL environment variables for stability with FSDP
    env = os.environ.copy()
    env["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1800"
    env["NCCL_TIMEOUT"] = "1800"
    env["NCCL_DEBUG"] = "WARN"

    # Run subprocess
    start_time = time.time()
    try:
        if config.verbose:
            # Stream output in real-time
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            for line in process.stdout:
                print(line, end='', flush=True)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
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
        print(f"ERROR: train.py failed for fold {fold_idx}")
        print(f"  Return code: {e.returncode}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"  stderr: {e.stderr[:2000]}")
        return None

    elapsed = time.time() - start_time

    # Load results from JSON file
    if output_results_file.exists():
        with open(output_results_file, 'r') as f:
            results = json.load(f)

        # Extract metrics (same keys train.py writes)
        test_r2 = results.get("test_avg_r2", 0.0)
        test_corr = results.get("test_avg_corr", 0.0)
        val_r2 = results.get("val_avg_r2", 0.0)
        val_corr = results.get("val_avg_corr", 0.0)
        train_loss = results.get("final_train_loss", 0.0)
        gate_mean = results.get("gate_mean", 0.0)

        fold_result = GatedFoldResult(
            fold_idx=fold_idx,
            seed=run_seed,
            seed_idx=seed_idx,
            test_r2=test_r2 if test_r2 is not None else 0.0,
            test_corr=test_corr if test_corr is not None else 0.0,
            val_r2=val_r2 if val_r2 is not None else 0.0,
            val_corr=val_corr if val_corr is not None else 0.0,
            train_loss=train_loss if train_loss is not None else 0.0,
            gate_mean=gate_mean if gate_mean is not None else 0.0,
            total_time=elapsed,
            config=config.to_dict(),
        )

        print(f"\nFold {fold_idx} completed:")
        print(f"  Test R²: {fold_result.test_r2:.4f}")
        print(f"  Val R²: {fold_result.val_r2:.4f}")
        print(f"  Gate mean: {fold_result.gate_mean:.3f}")
        print(f"  Time: {elapsed/60:.1f} minutes")

        return fold_result
    else:
        print(f"WARNING: Results file not found: {output_results_file}")
        return None


# =============================================================================
# Main Runner (following LOSO pattern)
# =============================================================================

def run_gated(config: GatedConfig) -> GatedResult:
    """Run gated translator training with cross-validation.

    Args:
        config: GatedConfig with all parameters

    Returns:
        GatedResult with aggregated results
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    result = GatedResult(config=config.to_dict())

    # Determine which folds to run
    folds = config.folds_to_run if config.folds_to_run else list(range(config.n_folds))

    for fold_idx in folds:
        for seed_idx in range(config.n_seeds):
            # Output path
            fold_dir = config.output_dir / f"fold{fold_idx}_seed{seed_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            output_results_file = fold_dir / "results.json"

            fold_result = run_single_fold(
                fold_idx=fold_idx,
                config=config,
                output_results_file=output_results_file,
                seed_idx=seed_idx,
            )

            if fold_result is not None:
                result.fold_results.append(fold_result)

    # Compute statistics
    result.compute_statistics()

    return result


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GatedTranslator Phase 3 Runner")

    # Dataset
    parser.add_argument("--dataset", type=str, default="olfactory",
                        choices=["olfactory", "pfc", "pcx1", "dandi"])
    parser.add_argument("--output-dir", type=str, default="artifacts/phase3_gated")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)  # Optimized
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1)

    # Model architecture
    parser.add_argument("--base-channels", type=int, default=256)  # Optimized
    parser.add_argument("--n-downsample", type=int, default=2)
    parser.add_argument("--attention-type", type=str, default="none")
    parser.add_argument("--cond-mode", type=str, default="cross_attn_gated")
    parser.add_argument("--conv-type", type=str, default="modern")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--skip-type", type=str, default="add")
    parser.add_argument("--conditioning", type=str, default="spectro_temporal")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr-schedule", type=str, default="cosine_warmup")

    # Gating parameters
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--sparsity-target", type=float, default=0.3)
    parser.add_argument("--gate-channels", type=int, default=64)
    parser.add_argument("--baseline-channels", type=int, default=64)

    # FSDP
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--fsdp-strategy", type=str, default="full")

    # Cross-validation
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--folds", type=int, nargs="+", default=None)

    # DANDI-specific
    parser.add_argument("--dandi-source-region", type=str, default="amygdala")
    parser.add_argument("--dandi-target-region", type=str, default="hippocampus")
    parser.add_argument("--dandi-window-size", type=int, default=5000)
    parser.add_argument("--dandi-stride-ratio", type=float, default=0.5)

    # PCx1-specific
    parser.add_argument("--pcx1-window-size", type=int, default=5000)
    parser.add_argument("--pcx1-stride-ratio", type=float, default=0.5)

    # Execution
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    config = GatedConfig(
        dataset=args.dataset,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        n_seeds=args.n_seeds,
        base_channels=args.base_channels,
        n_downsample=args.n_downsample,
        attention_type=args.attention_type,
        cond_mode=args.cond_mode,
        conv_type=args.conv_type,
        activation=args.activation,
        skip_type=args.skip_type,
        conditioning=args.conditioning,
        optimizer=args.optimizer,
        lr_schedule=args.lr_schedule,
        sparsity_weight=args.sparsity_weight,
        sparsity_target=args.sparsity_target,
        gate_channels=args.gate_channels,
        baseline_channels=args.baseline_channels,
        use_fsdp=args.fsdp,
        fsdp_strategy=args.fsdp_strategy,
        n_folds=args.n_folds,
        folds_to_run=args.folds,
        dandi_source_region=args.dandi_source_region,
        dandi_target_region=args.dandi_target_region,
        dandi_window_size=args.dandi_window_size,
        dandi_stride_ratio=args.dandi_stride_ratio,
        pcx1_window_size=args.pcx1_window_size,
        pcx1_stride_ratio=args.pcx1_stride_ratio,
        verbose=args.verbose and not args.quiet,
    )

    print("=" * 70)
    print("Phase 3: GatedTranslator Evaluation")
    print("=" * 70)
    print(f"Dataset: {config.dataset}")
    print(f"Epochs: {config.epochs}")
    print(f"Sparsity target: {config.sparsity_target}, weight: {config.sparsity_weight}")
    print(f"FSDP: {config.use_fsdp}")
    print(f"Folds: {config.n_folds}, Seeds: {config.n_seeds}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would train with config:")
        print(json.dumps(config.to_dict(), indent=2))
        return

    # Run
    result = run_gated(config)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = config.output_dir / f"gated_results_{config.dataset}_{timestamp}.json"

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
