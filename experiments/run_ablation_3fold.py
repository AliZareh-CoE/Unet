#!/usr/bin/env python3
"""
3-Fold Cross-Validation Ablation Study
=======================================

Implements a proper 3-fold cross-validation where:
- 9 sessions are divided into 3 groups of 3
- Each fold holds out a different group of 3 sessions for testing
- All 9 sessions get evaluated as test data across the 3 folds

This gives us:
- More robust performance estimates (mean ± std across folds)
- Test performance for EVERY session
- Better understanding of model generalization

Baseline: depth_deep (n_downsample=4) since it showed +0.0143 improvement

Ablation configurations to test:
1. Baseline (depth_deep with all bells and whistles)
2. conv_type: standard vs modern
3. conditioning: none vs spectro_temporal
4. adaptive_scaling: off vs on
5. attention_type: none vs cross_freq_v2
6. skip_type: add vs concat
7. bidirectional: off vs on
8. dropout: 0.0 vs 0.1
9. depth: compare with depth_medium (n_downsample=3)
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str

    # Model architecture
    n_downsample: int = 4  # depth_deep as default
    conv_type: str = "modern"
    attention_type: str = "cross_freq_v2"
    cond_mode: str = "cross_attn_gated"
    skip_type: str = "add"

    # Training
    use_adaptive_scaling: bool = True
    use_bidirectional: bool = False
    dropout: float = 0.0
    conditioning: str = "spectro_temporal"

    # Fixed parameters
    base_channels: int = 128
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adamw"
    lr_schedule: str = "cosine_warmup"
    activation: str = "gelu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "n_downsample": self.n_downsample,
            "conv_type": self.conv_type,
            "attention_type": self.attention_type,
            "cond_mode": self.cond_mode,
            "skip_type": self.skip_type,
            "use_adaptive_scaling": self.use_adaptive_scaling,
            "use_bidirectional": self.use_bidirectional,
            "dropout": self.dropout,
            "conditioning": self.conditioning,
            "base_channels": self.base_channels,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }


# =============================================================================
# Define Ablation Configurations
# =============================================================================

def get_ablation_configs() -> Dict[str, AblationConfig]:
    """Define all ablation configurations to test.

    Baseline: depth_deep with modern convs, cross_freq_v2 attention,
              adaptive scaling, spectro_temporal conditioning.
    """
    configs = {}

    # =========================================================================
    # BASELINE (depth_deep - best performing from previous study)
    # =========================================================================
    configs["baseline"] = AblationConfig(
        name="baseline",
        description="depth_deep baseline with all best components",
        n_downsample=4,  # depth_deep
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
        use_bidirectional=False,
    )

    # =========================================================================
    # CONVOLUTION TYPE ABLATION
    # =========================================================================
    configs["conv_type_standard"] = AblationConfig(
        name="conv_type_standard",
        description="Standard convolutions instead of modern (dilated depthwise separable)",
        n_downsample=4,
        conv_type="standard",  # Changed
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    )

    # =========================================================================
    # CONDITIONING ABLATION
    # =========================================================================
    configs["conditioning_none"] = AblationConfig(
        name="conditioning_none",
        description="No conditioning (cond_mode=none bypasses conditioning entirely)",
        n_downsample=4,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",  # Source doesn't matter when cond_mode=none
        cond_mode="none",  # This bypasses conditioning in forward pass
    )

    # =========================================================================
    # ADAPTIVE SCALING ABLATION
    # =========================================================================
    configs["adaptive_scaling_off"] = AblationConfig(
        name="adaptive_scaling_off",
        description="Disable adaptive output scaling",
        n_downsample=4,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=False,  # Changed
        conditioning="spectro_temporal",
    )

    # =========================================================================
    # DEPTH ABLATIONS (compare with baseline depth_deep)
    # =========================================================================
    configs["depth_medium"] = AblationConfig(
        name="depth_medium",
        description="Medium depth (n_downsample=3) vs deep",
        n_downsample=3,  # depth_medium
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    )

    configs["depth_shallow"] = AblationConfig(
        name="depth_shallow",
        description="Shallow depth (n_downsample=2) - original default",
        n_downsample=2,  # depth_shallow (original default)
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    )

    # =========================================================================
    # ATTENTION ABLATIONS
    # =========================================================================
    configs["attention_none"] = AblationConfig(
        name="attention_none",
        description="No attention mechanism",
        n_downsample=4,
        conv_type="modern",
        attention_type="none",  # Changed
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    )

    configs["attention_basic"] = AblationConfig(
        name="attention_basic",
        description="Basic attention instead of cross_freq_v2",
        n_downsample=4,
        conv_type="modern",
        attention_type="basic",  # Changed
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    )

    # =========================================================================
    # SKIP CONNECTION ABLATION
    # =========================================================================
    configs["skip_type_concat"] = AblationConfig(
        name="skip_type_concat",
        description="Concatenation skip connections instead of addition",
        n_downsample=4,
        conv_type="modern",
        attention_type="cross_freq_v2",
        skip_type="concat",  # Changed
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    )

    # =========================================================================
    # BIDIRECTIONAL TRAINING ABLATION
    # =========================================================================
    configs["bidirectional_on"] = AblationConfig(
        name="bidirectional_on",
        description="Enable bidirectional training (OB->PCx and PCx->OB)",
        n_downsample=4,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
        use_bidirectional=True,  # Changed
    )

    # =========================================================================
    # DROPOUT ABLATION
    # =========================================================================
    configs["dropout_01"] = AblationConfig(
        name="dropout_01",
        description="Add 10% dropout for regularization",
        n_downsample=4,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
        dropout=0.1,  # Changed
    )

    configs["dropout_02"] = AblationConfig(
        name="dropout_02",
        description="Add 20% dropout for regularization",
        n_downsample=4,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
        dropout=0.2,  # Changed
    )

    # NOTE: cond_mode only supports "none" and "cross_attn_gated" in the model
    # The conditioning_none ablation tests disabling conditioning entirely
    # No additional cond_mode ablations needed since only 2 modes exist

    return configs


# =============================================================================
# 3-Fold Session Splits
# =============================================================================

def get_3fold_session_splits(all_sessions: List[str]) -> List[Dict[str, List[str]]]:
    """Create 3-fold CV splits where each fold holds out 3 sessions.

    For 9 sessions [s0, s1, s2, s3, s4, s5, s6, s7, s8]:
    - Fold 0: test=[s0,s1,s2], train=[s3,s4,s5,s6,s7,s8]
    - Fold 1: test=[s3,s4,s5], train=[s0,s1,s2,s6,s7,s8]
    - Fold 2: test=[s6,s7,s8], train=[s0,s1,s2,s3,s4,s5]

    This ensures every session is tested exactly once.
    """
    n_sessions = len(all_sessions)
    if n_sessions < 6:
        raise ValueError(f"Need at least 6 sessions for 3-fold CV, got {n_sessions}")

    # Calculate sessions per fold (handle non-divisible cases)
    sessions_per_fold = n_sessions // 3
    remainder = n_sessions % 3

    splits = []
    start_idx = 0

    for fold_idx in range(3):
        # Handle remainder by adding extra session to first folds
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
# Result Classes
# =============================================================================

@dataclass
class FoldResult:
    """Result from a single fold of ablation experiment."""
    fold_idx: int
    test_sessions: List[str]
    train_sessions: List[str]
    val_r2: float
    val_loss: float
    epochs_trained: int
    total_time: float
    per_session_r2: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_idx": self.fold_idx,
            "test_sessions": self.test_sessions,
            "train_sessions": self.train_sessions,
            "val_r2": self.val_r2,
            "val_loss": self.val_loss,
            "epochs_trained": self.epochs_trained,
            "total_time": self.total_time,
            "per_session_r2": self.per_session_r2,
        }


@dataclass
class AblationResult:
    """Aggregated results for one ablation configuration."""
    config: AblationConfig
    fold_results: List[FoldResult]

    # Statistics (computed after all folds)
    mean_r2: float = 0.0
    std_r2: float = 0.0
    all_session_r2s: Dict[str, float] = field(default_factory=dict)

    def compute_statistics(self) -> None:
        """Compute aggregate statistics from fold results."""
        if not self.fold_results:
            return

        r2_values = [r.val_r2 for r in self.fold_results]
        self.mean_r2 = float(np.mean(r2_values))
        self.std_r2 = float(np.std(r2_values))

        # Aggregate per-session R2s across folds
        for fold in self.fold_results:
            for session, r2 in fold.per_session_r2.items():
                self.all_session_r2s[session] = r2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "fold_results": [f.to_dict() for f in self.fold_results],
            "mean_r2": self.mean_r2,
            "std_r2": self.std_r2,
            "all_session_r2s": self.all_session_r2s,
        }


# =============================================================================
# Training Functions
# =============================================================================

def get_all_sessions(dataset: str = "olfactory") -> List[str]:
    """Get list of all session names for a dataset."""
    if dataset == "olfactory":
        from data import list_pcx1_sessions, PCX1_CONTINUOUS_PATH
        return list_pcx1_sessions(PCX1_CONTINUOUS_PATH)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def run_single_fold(
    ablation_config: AblationConfig,
    fold_split: Dict[str, Any],
    output_dir: Path,
    seed: int = 42,
    verbose: bool = True,
    use_fsdp: bool = False,
    dry_run: bool = False,
) -> Optional[FoldResult]:
    """Run training for a single fold.

    Args:
        ablation_config: Configuration for this ablation
        fold_split: Dict with fold_idx, test_sessions, train_sessions
        output_dir: Directory to save results
        seed: Random seed
        verbose: Print training output
        use_fsdp: Use FSDP for multi-GPU training
        dry_run: If True, print commands without running them

    Returns:
        FoldResult or None on failure
    """
    import torch

    fold_idx = fold_split["fold_idx"]
    test_sessions = fold_split["test_sessions"]
    train_sessions = fold_split["train_sessions"]

    print(f"\n{'='*70}")
    print(f"  {ablation_config.name} | Fold {fold_idx + 1}/3")
    print(f"{'='*70}")
    print(f"  Test sessions:  {test_sessions}")
    print(f"  Train sessions: {train_sessions}")
    print()

    # Output file for results
    results_file = output_dir / f"{ablation_config.name}_fold{fold_idx}_results.json"

    # Build train.py command
    if use_fsdp:
        nproc = torch.cuda.device_count() if torch.cuda.is_available() else 1
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            str(PROJECT_ROOT / "train.py"),
        ]
    else:
        cmd = [sys.executable, str(PROJECT_ROOT / "train.py")]

    # Base arguments
    cmd.extend([
        "--arch", "condunet",
        "--dataset", "olfactory",
        "--epochs", str(ablation_config.epochs),
        "--batch-size", str(ablation_config.batch_size),
        "--lr", str(ablation_config.learning_rate),
        "--seed", str(seed + fold_idx),
        "--output-results-file", str(results_file),
        "--fold", str(fold_idx),
        "--no-plots",
    ])

    # Session-based splitting
    cmd.append("--split-by-session")
    cmd.append("--force-recreate-splits")
    cmd.extend(["--val-sessions"] + test_sessions)  # Hold out test sessions
    cmd.append("--no-test-set")
    cmd.append("--no-early-stop")  # Train full epochs

    # Model architecture arguments
    cmd.extend(["--base-channels", str(ablation_config.base_channels)])
    cmd.extend(["--n-downsample", str(ablation_config.n_downsample)])
    cmd.extend(["--attention-type", ablation_config.attention_type])
    cmd.extend(["--conv-type", ablation_config.conv_type])
    cmd.extend(["--skip-type", ablation_config.skip_type])
    cmd.extend(["--activation", ablation_config.activation])

    # Conditioning
    # Always pass cond_mode explicitly to ensure ablation is applied
    cmd.extend(["--cond-mode", ablation_config.cond_mode])

    # Only pass conditioning source if cond_mode is not "none"
    # (when cond_mode=none, conditioning is bypassed regardless of source)
    if ablation_config.cond_mode != "none":
        cmd.extend(["--conditioning", ablation_config.conditioning])

    # Training options
    cmd.extend(["--optimizer", ablation_config.optimizer])
    cmd.extend(["--lr-schedule", ablation_config.lr_schedule])

    if ablation_config.dropout > 0:
        cmd.extend(["--dropout", str(ablation_config.dropout)])

    if ablation_config.use_adaptive_scaling:
        cmd.append("--use-adaptive-scaling")

    if not ablation_config.use_bidirectional:
        cmd.append("--no-bidirectional")

    if use_fsdp:
        cmd.extend(["--fsdp", "--fsdp-strategy", "grad_op"])

    # Set environment variables
    env = os.environ.copy()
    env["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1800"

    # DRY RUN: Print command and return fake result
    if dry_run:
        print(f"\n  [DRY RUN] Command that would be executed:")
        print(f"  {' '.join(cmd)}")
        print()
        # Return fake result for dry run
        return FoldResult(
            fold_idx=fold_idx,
            test_sessions=test_sessions,
            train_sessions=train_sessions,
            val_r2=0.0,
            val_loss=0.0,
            epochs_trained=0,
            total_time=0.0,
        )

    # Run training
    start_time = time.time()
    try:
        if verbose:
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
        print(f"ERROR: Training failed for {ablation_config.name} fold {fold_idx}")
        print(f"  Command: {' '.join(cmd)}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"  stderr: {e.stderr[:1000]}...")
        return None

    elapsed = time.time() - start_time

    # Load results
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)

        fold_result = FoldResult(
            fold_idx=fold_idx,
            test_sessions=test_sessions,
            train_sessions=train_sessions,
            val_r2=results.get("best_val_r2", results.get("val_r2", 0.0)),
            val_loss=results.get("best_val_loss", results.get("val_loss", float('inf'))),
            epochs_trained=results.get("epochs_trained", ablation_config.epochs),
            total_time=elapsed,
            per_session_r2=results.get("per_session_r2", {}),
        )

        print(f"\n  Fold {fold_idx} completed:")
        print(f"    Val R2: {fold_result.val_r2:.4f}")
        print(f"    Time: {elapsed/60:.1f} minutes")

        return fold_result
    else:
        print(f"WARNING: Results file not found: {results_file}")
        return None


def run_ablation_experiment(
    ablation_config: AblationConfig,
    fold_splits: List[Dict[str, Any]],
    output_dir: Path,
    seed: int = 42,
    verbose: bool = True,
    use_fsdp: bool = False,
    folds_to_run: Optional[List[int]] = None,
    dry_run: bool = False,
) -> AblationResult:
    """Run all folds for an ablation configuration.

    Args:
        ablation_config: Configuration for this ablation
        fold_splits: List of fold split dicts
        output_dir: Directory to save results
        seed: Random seed
        verbose: Print training output
        use_fsdp: Use FSDP for multi-GPU training
        folds_to_run: Optional list of specific fold indices to run
        dry_run: If True, print commands without running them

    Returns:
        AblationResult with all fold results
    """
    print(f"\n{'#'*70}")
    print(f"# ABLATION: {ablation_config.name}")
    print(f"# {ablation_config.description}")
    print(f"{'#'*70}")

    ablation_dir = output_dir / ablation_config.name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for split in fold_splits:
        fold_idx = split["fold_idx"]

        # Skip if not in folds_to_run
        if folds_to_run is not None and fold_idx not in folds_to_run:
            continue

        result = run_single_fold(
            ablation_config=ablation_config,
            fold_split=split,
            output_dir=ablation_dir,
            seed=seed,
            verbose=verbose,
            use_fsdp=use_fsdp,
            dry_run=dry_run,
        )

        if result is not None:
            fold_results.append(result)

    # Create ablation result
    ablation_result = AblationResult(
        config=ablation_config,
        fold_results=fold_results,
    )
    ablation_result.compute_statistics()

    # Save ablation result
    result_file = ablation_dir / "ablation_result.json"
    with open(result_file, 'w') as f:
        json.dump(ablation_result.to_dict(), f, indent=2)

    return ablation_result


# =============================================================================
# Main Runner
# =============================================================================

def run_3fold_ablation_study(
    output_dir: Path,
    ablations_to_run: Optional[List[str]] = None,
    folds_to_run: Optional[List[int]] = None,
    seed: int = 42,
    verbose: bool = True,
    use_fsdp: bool = False,
    dry_run: bool = False,
) -> Dict[str, AblationResult]:
    """Run the complete 3-fold ablation study.

    Args:
        output_dir: Directory to save all results
        ablations_to_run: Optional list of ablation names to run (default: all)
        folds_to_run: Optional list of fold indices to run (default: all 3)
        seed: Random seed
        verbose: Print training output
        use_fsdp: Use FSDP for multi-GPU training
        dry_run: If True, print commands without running them

    Returns:
        Dict mapping ablation name to AblationResult
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / f"ablation_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print("=" * 70)
    print("3-FOLD CROSS-VALIDATION ABLATION STUDY")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Log file: {log_file}")
    print()

    # Get all sessions
    all_sessions = get_all_sessions("olfactory")
    print(f"Found {len(all_sessions)} sessions: {all_sessions}")

    # Create 3-fold splits
    fold_splits = get_3fold_session_splits(all_sessions)
    print(f"\n3-Fold Session Splits:")
    for split in fold_splits:
        print(f"  Fold {split['fold_idx']}: test={split['test_sessions']}")
    print()

    # Get ablation configurations
    all_configs = get_ablation_configs()

    if ablations_to_run is not None:
        configs_to_run = {k: v for k, v in all_configs.items() if k in ablations_to_run}
    else:
        configs_to_run = all_configs

    print(f"Ablations to run: {list(configs_to_run.keys())}")
    print()

    # Run ablations
    results = {}
    start_time = time.time()

    for ablation_name, config in configs_to_run.items():
        result = run_ablation_experiment(
            ablation_config=config,
            fold_splits=fold_splits,
            output_dir=output_dir,
            seed=seed,
            verbose=verbose,
            use_fsdp=use_fsdp,
            folds_to_run=folds_to_run,
            dry_run=dry_run,
        )
        results[ablation_name] = result

    total_time = time.time() - start_time

    # Print summary
    print_summary(results, total_time)

    # Save summary
    save_summary(results, output_dir)

    return results


def print_summary(results: Dict[str, AblationResult], total_time: float) -> None:
    """Print summary of ablation study results."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 70)

    # Find baseline for comparison
    baseline_r2 = results.get("baseline", AblationResult(
        config=AblationConfig(name="baseline", description=""),
        fold_results=[]
    )).mean_r2

    # Sort by mean R2
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].mean_r2,
        reverse=True
    )

    print(f"\n{'Ablation':<25} {'R2 (mean ± std)':<20} {'Delta':>10}")
    print("-" * 60)

    for name, result in sorted_results:
        r2_str = f"{result.mean_r2:.4f} +/- {result.std_r2:.4f}"
        delta = result.mean_r2 - baseline_r2 if name != "baseline" else 0.0
        delta_str = f"{delta:+.4f}" if name != "baseline" else "--"
        sign = "+" if delta > 0 else "" if delta == 0 else ""

        print(f"{name:<25} {r2_str:<20} {sign}{delta_str:>10}")

    print("-" * 60)
    print(f"\nTotal time: {total_time/3600:.1f} hours")
    print("=" * 70)


def save_summary(results: Dict[str, AblationResult], output_dir: Path) -> None:
    """Save summary of results to JSON."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "results": {name: result.to_dict() for name, result in results.items()},
        "summary_table": [],
    }

    # Find baseline
    baseline_r2 = results.get("baseline", AblationResult(
        config=AblationConfig(name="baseline", description=""),
        fold_results=[]
    )).mean_r2

    # Build summary table
    for name, result in sorted(results.items(), key=lambda x: x[1].mean_r2, reverse=True):
        delta = result.mean_r2 - baseline_r2 if name != "baseline" else 0.0
        summary["summary_table"].append({
            "name": name,
            "mean_r2": result.mean_r2,
            "std_r2": result.std_r2,
            "delta_vs_baseline": delta,
            "n_folds": len(result.fold_results),
        })

    # Save
    summary_file = output_dir / "ablation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="3-Fold Cross-Validation Ablation Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation_3fold",
        help="Output directory for results",
    )

    parser.add_argument(
        "--ablations",
        type=str,
        nargs="+",
        default=None,
        help="Specific ablations to run (default: all)",
    )

    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific fold indices to run (default: all 3)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Use FSDP for multi-GPU training",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (don't show training logs)",
    )

    parser.add_argument(
        "--list-ablations",
        action="store_true",
        help="List all available ablation configurations and exit",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be run without actually running them",
    )

    args = parser.parse_args()

    # List ablations and exit
    if args.list_ablations:
        configs = get_ablation_configs()
        print("\nAvailable ablation configurations:")
        print("-" * 60)
        for name, config in configs.items():
            print(f"  {name:<25} - {config.description}")
        print()
        return 0

    # Run ablation study
    run_3fold_ablation_study(
        output_dir=Path(args.output_dir),
        ablations_to_run=args.ablations,
        folds_to_run=args.folds,
        seed=args.seed,
        verbose=not args.quiet,
        use_fsdp=args.fsdp,
        dry_run=args.dry_run,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
