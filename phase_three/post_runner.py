#!/usr/bin/env python3
"""
POST Phase 3: Adaptive Scaling Mode Exploration
================================================

Systematic exploration of SessionAdaptiveScalingV3 transformation modes
to find the best approach for session generalization.

MODES TO EXPLORE:
=================

1. **scalar** (baseline): Per-channel scale + bias
   - Same as original FiLM, 2*C parameters
   - Fast, simple, but limited expressiveness

2. **band_wise**: Per-channel, per-frequency-band scale + bias
   - Different γ, β for delta, theta, alpha, beta, gamma
   - Can correct session-specific spectral tilts
   - 2*C*B parameters (B=5 bands)

3. **spectral**: Full spectral transfer function per channel
   - Learn frequency-dependent scaling in Fourier domain
   - Can model complex frequency response differences
   - Uses bottleneck architecture to control parameters

4. **harmonic**: Model harmonic relationships
   - Fundamental + harmonics scaled coherently
   - Neural oscillations often have harmonics at 2f, 3f, ...
   - Learns harmonic coupling structure

CROSS-CHANNEL ATTENTION:
========================
Each mode can optionally include cross-channel attention, allowing
channels to influence each other's scaling. This is useful when:
- Channels are spatially related (neighboring electrodes)
- Session differences affect channels non-uniformly

EVALUATION:
===========
For each configuration:
- 3-fold cross-validation with held-out sessions
- Primary metric: Test R² (true generalization)
- Secondary metrics: PSD error, PSD difference, spectral correlation

USAGE:
======
    # Run all adaptive scaling experiments
    python phase_three/post_runner.py --output-dir results/adaptive_scaling

    # Run specific modes only
    python phase_three/post_runner.py --modes scalar band_wise

    # Dry run to see commands
    python phase_three/post_runner.py --dry-run

    # Use FSDP for multi-GPU
    python phase_three/post_runner.py --fsdp
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
# Custom JSON Encoder for NumPy types
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
# Configuration
# =============================================================================

@dataclass
class AdaptiveScalingConfig:
    """Configuration for adaptive scaling experiment."""
    name: str
    description: str

    # Adaptive scaling mode
    adaptive_scaling_mode: str = "scalar"  # scalar, band_wise, spectral, harmonic
    adaptive_scaling_cross_channel: bool = False

    # Fixed architecture (use best from Phase 3)
    n_downsample: int = 4
    base_channels: int = 128
    conv_type: str = "modern"
    attention_type: str = "cross_freq_v2"
    cond_mode: str = "cross_attn_gated"
    skip_type: str = "add"

    # Training
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 1e-3
    optimizer: str = "adamw"
    lr_schedule: str = "cosine_warmup"
    activation: str = "gelu"
    conditioning: str = "spectro_temporal"

    # Noise augmentation (same as Phase 3)
    use_noise_augmentation: bool = True
    noise_gaussian_std: float = 0.1
    noise_pink: bool = True
    noise_pink_std: float = 0.05
    noise_channel_dropout: float = 0.05
    noise_temporal_dropout: float = 0.02
    noise_prob: float = 0.5

    # PSD validation
    compute_psd_validation: bool = True  # Always compute PSD during validation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "adaptive_scaling_mode": self.adaptive_scaling_mode,
            "adaptive_scaling_cross_channel": self.adaptive_scaling_cross_channel,
            "n_downsample": self.n_downsample,
            "base_channels": self.base_channels,
            "conv_type": self.conv_type,
            "attention_type": self.attention_type,
            "cond_mode": self.cond_mode,
            "skip_type": self.skip_type,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "conditioning": self.conditioning,
            "compute_psd_validation": self.compute_psd_validation,
        }


# =============================================================================
# Experiment Configurations
# =============================================================================

def get_adaptive_scaling_configs(
    modes: Optional[List[str]] = None,
    include_cross_channel: bool = True,
    include_no_scaling: bool = True,
) -> Dict[str, AdaptiveScalingConfig]:
    """Get configurations for adaptive scaling exploration.

    Args:
        modes: List of modes to test. If None, test all modes.
        include_cross_channel: If True, include cross-channel variants.
        include_no_scaling: If True, include baseline without adaptive scaling.

    Returns:
        Dictionary of config_name -> AdaptiveScalingConfig
    """
    all_modes = ["scalar", "band_wise", "spectral", "harmonic"]

    if modes is None:
        modes = all_modes
    else:
        # Validate modes
        invalid = set(modes) - set(all_modes)
        if invalid:
            raise ValueError(f"Invalid modes: {invalid}. Valid: {all_modes}")

    configs_list = []

    # Baseline: No adaptive scaling
    if include_no_scaling:
        configs_list.append(AdaptiveScalingConfig(
            name="no_adaptive_scaling",
            description="Baseline without adaptive scaling",
            adaptive_scaling_mode="scalar",  # Doesn't matter, won't be used
        ))

    # Test each mode
    for mode in modes:
        # Without cross-channel
        configs_list.append(AdaptiveScalingConfig(
            name=f"adaptive_{mode}",
            description=f"Adaptive scaling: {mode} mode",
            adaptive_scaling_mode=mode,
            adaptive_scaling_cross_channel=False,
        ))

        # With cross-channel (optional)
        if include_cross_channel:
            configs_list.append(AdaptiveScalingConfig(
                name=f"adaptive_{mode}_cross_ch",
                description=f"Adaptive scaling: {mode} mode + cross-channel attention",
                adaptive_scaling_mode=mode,
                adaptive_scaling_cross_channel=True,
            ))

    return {c.name: c for c in configs_list}


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class FoldResult:
    """Result from a single fold."""
    fold_idx: int
    seed: int = 42
    seed_idx: int = 0
    test_sessions: List[str] = field(default_factory=list)
    train_sessions: List[str] = field(default_factory=list)

    # Validation metrics
    val_r2: float = 0.0
    val_loss: float = 0.0
    val_corr: float = 0.0
    val_psd_diff_db: float = 0.0
    val_psd_err_db: float = 0.0

    # Test metrics (PRIMARY)
    test_r2: float = 0.0
    test_corr: float = 0.0
    test_psd_diff_db: float = 0.0
    test_psd_err_db: float = 0.0

    # PSD diagnostics
    psd_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Per-session results
    per_session_test_results: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Training metadata
    epochs_trained: int = 0
    total_time: float = 0.0
    n_parameters: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_idx": self.fold_idx,
            "seed": self.seed,
            "seed_idx": self.seed_idx,
            "test_sessions": self.test_sessions,
            "train_sessions": self.train_sessions,
            "val_r2": self.val_r2,
            "val_loss": self.val_loss,
            "val_corr": self.val_corr,
            "val_psd_diff_db": self.val_psd_diff_db,
            "val_psd_err_db": self.val_psd_err_db,
            "test_r2": self.test_r2,
            "test_corr": self.test_corr,
            "test_psd_diff_db": self.test_psd_diff_db,
            "test_psd_err_db": self.test_psd_err_db,
            "psd_diagnostics": self.psd_diagnostics,
            "per_session_test_results": self.per_session_test_results,
            "epochs_trained": self.epochs_trained,
            "total_time": self.total_time,
            "n_parameters": self.n_parameters,
        }


@dataclass
class ExperimentResult:
    """Aggregated results for an experiment across all folds."""
    config: AdaptiveScalingConfig
    fold_results: List[FoldResult] = field(default_factory=list)

    # Aggregated R² statistics
    mean_r2: float = 0.0
    std_r2: float = 0.0

    # Aggregated PSD statistics
    mean_psd_diff_db: float = 0.0
    std_psd_diff_db: float = 0.0
    mean_psd_err_db: float = 0.0
    std_psd_err_db: float = 0.0

    # Aggregated correlation
    mean_corr: float = 0.0
    std_corr: float = 0.0

    # Training metadata
    total_time: float = 0.0
    n_parameters: int = 0

    def compute_statistics(self) -> None:
        """Compute aggregate statistics from fold results."""
        if not self.fold_results:
            return

        # R² statistics
        r2_values = [r.test_r2 for r in self.fold_results]
        self.mean_r2 = float(np.mean(r2_values))
        self.std_r2 = float(np.std(r2_values, ddof=1)) if len(r2_values) > 1 else 0.0

        # Correlation statistics
        corr_values = [r.test_corr for r in self.fold_results]
        self.mean_corr = float(np.mean(corr_values))
        self.std_corr = float(np.std(corr_values, ddof=1)) if len(corr_values) > 1 else 0.0

        # PSD statistics
        psd_diff_values = [r.test_psd_diff_db for r in self.fold_results if r.test_psd_diff_db != 0.0]
        if psd_diff_values:
            self.mean_psd_diff_db = float(np.mean(psd_diff_values))
            self.std_psd_diff_db = float(np.std(psd_diff_values, ddof=1)) if len(psd_diff_values) > 1 else 0.0

        psd_err_values = [r.test_psd_err_db for r in self.fold_results if r.test_psd_err_db != 0.0]
        if psd_err_values:
            self.mean_psd_err_db = float(np.mean(psd_err_values))
            self.std_psd_err_db = float(np.std(psd_err_values, ddof=1)) if len(psd_err_values) > 1 else 0.0

        # Training metadata
        self.total_time = sum(r.total_time for r in self.fold_results)
        self.n_parameters = self.fold_results[0].n_parameters if self.fold_results else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "fold_results": [f.to_dict() for f in self.fold_results],
            "mean_r2": self.mean_r2,
            "std_r2": self.std_r2,
            "mean_corr": self.mean_corr,
            "std_corr": self.std_corr,
            "mean_psd_diff_db": self.mean_psd_diff_db,
            "std_psd_diff_db": self.std_psd_diff_db,
            "mean_psd_err_db": self.mean_psd_err_db,
            "std_psd_err_db": self.std_psd_err_db,
            "total_time": self.total_time,
            "n_parameters": self.n_parameters,
        }


# =============================================================================
# Session Splits
# =============================================================================

def get_all_sessions(dataset: str = "olfactory") -> List[str]:
    """Get list of all session names."""
    if dataset == "olfactory":
        from data import list_pcx1_sessions, PCX1_CONTINUOUS_PATH
        return list_pcx1_sessions(PCX1_CONTINUOUS_PATH)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_3fold_session_splits(all_sessions: List[str]) -> List[Dict[str, Any]]:
    """Create 3-fold CV splits where each fold holds out sessions for test."""
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
# Training Functions
# =============================================================================

def run_single_fold(
    config: AdaptiveScalingConfig,
    fold_split: Dict[str, Any],
    output_dir: Path,
    seed: int = 42,
    seed_idx: int = 0,
    verbose: bool = True,
    use_fsdp: bool = False,
    fsdp_strategy: str = "grad_op",
    dry_run: bool = False,
    is_no_scaling: bool = False,
) -> Optional[FoldResult]:
    """Run training for a single fold."""
    import torch

    fold_idx = fold_split["fold_idx"]
    test_sessions = fold_split["test_sessions"]
    train_sessions = fold_split["train_sessions"]

    print(f"\n{'='*70}")
    print(f"  {config.name} | Fold {fold_idx + 1}/3")
    print(f"{'='*70}")
    print(f"  Mode: {config.adaptive_scaling_mode}")
    print(f"  Cross-channel: {config.adaptive_scaling_cross_channel}")
    print(f"  Test sessions:  {test_sessions}")
    print()

    # Output file for results
    results_file = output_dir / f"{config.name}_fold{fold_idx}_seed{seed_idx}_results.json"

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

    # Checkpoint path
    checkpoint_prefix = f"{config.name}_fold{fold_idx}_seed{seed_idx}"

    # Base arguments
    cmd.extend([
        "--arch", "condunet",
        "--dataset", "olfactory",
        "--epochs", str(config.epochs),
        "--batch-size", str(config.batch_size),
        "--lr", str(config.learning_rate),
        "--seed", str(seed),
        "--output-results-file", str(results_file),
        "--checkpoint-prefix", checkpoint_prefix,
        "--fold", str(fold_idx),
        "--no-plots",
        "--quiet",
    ])

    # Data split (exclude test sessions)
    cmd.append("--force-recreate-splits")
    cmd.extend(["--exclude-sessions"] + test_sessions)
    cmd.append("--no-test-set")
    cmd.append("--no-early-stop")

    # Model architecture
    cmd.extend(["--base-channels", str(config.base_channels)])
    cmd.extend(["--n-downsample", str(config.n_downsample)])
    cmd.extend(["--attention-type", config.attention_type])
    cmd.extend(["--conv-type", config.conv_type])
    cmd.extend(["--skip-type", config.skip_type])
    cmd.extend(["--activation", config.activation])
    cmd.extend(["--cond-mode", config.cond_mode])
    cmd.extend(["--conditioning", config.conditioning])

    # Training options
    cmd.extend(["--optimizer", config.optimizer])
    cmd.extend(["--lr-schedule", config.lr_schedule])

    # Adaptive scaling
    if not is_no_scaling:
        cmd.append("--use-adaptive-scaling")
        cmd.extend(["--adaptive-scaling-mode", config.adaptive_scaling_mode])
        if config.adaptive_scaling_cross_channel:
            cmd.append("--adaptive-scaling-cross-channel")

    # PSD validation (always enabled for this study)
    if config.compute_psd_validation:
        cmd.append("--compute-psd-validation")

    # Noise augmentation
    if config.use_noise_augmentation:
        cmd.append("--use-noise-augmentation")
        cmd.extend(["--noise-gaussian-std", str(config.noise_gaussian_std)])
        if config.noise_pink:
            cmd.append("--noise-pink")
        cmd.extend(["--noise-pink-std", str(config.noise_pink_std)])
        cmd.extend(["--noise-channel-dropout", str(config.noise_channel_dropout)])
        cmd.extend(["--noise-temporal-dropout", str(config.noise_temporal_dropout)])
        cmd.extend(["--noise-prob", str(config.noise_prob)])

    if use_fsdp:
        cmd.extend(["--fsdp", "--fsdp-strategy", fsdp_strategy])

    # Environment
    env = os.environ.copy()
    env["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1800"

    # Dry run
    if dry_run:
        print(f"\n  [DRY RUN] Command:")
        print(f"  {' '.join(cmd)}")
        return FoldResult(
            fold_idx=fold_idx,
            seed=seed,
            seed_idx=seed_idx,
            test_sessions=test_sessions,
            train_sessions=train_sessions,
        )

    # Run training
    start_time = time.time()

    try:
        if verbose:
            result = subprocess.run(cmd, env=env, check=True)
        else:
            result = subprocess.run(
                cmd, env=env, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )

        elapsed = time.time() - start_time

        # Parse results
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)

            fold_result = FoldResult(
                fold_idx=fold_idx,
                seed=seed,
                seed_idx=seed_idx,
                test_sessions=test_sessions,
                train_sessions=train_sessions,
                val_r2=results.get("val_r2", results.get("r2", 0.0)),
                val_loss=results.get("val_loss", results.get("loss", 0.0)),
                val_corr=results.get("val_corr", results.get("corr", 0.0)),
                val_psd_diff_db=results.get("psd_diff_db", 0.0),
                val_psd_err_db=results.get("psd_err_db", 0.0),
                test_r2=results.get("test_r2", 0.0),
                test_corr=results.get("test_corr", 0.0),
                test_psd_diff_db=results.get("test_psd_diff_db", 0.0),
                test_psd_err_db=results.get("test_psd_err_db", 0.0),
                psd_diagnostics=results.get("psd_diagnostics", {}),
                epochs_trained=results.get("epochs_trained", config.epochs),
                total_time=elapsed,
                n_parameters=results.get("n_parameters", 0),
            )

            print(f"\n  Fold {fold_idx + 1} Results:")
            print(f"    Val R²: {fold_result.val_r2:.4f}")
            print(f"    Val PSD Diff: {fold_result.val_psd_diff_db:+.2f} dB")
            if fold_result.psd_diagnostics:
                diag = fold_result.psd_diagnostics
                print(f"    PSD Diagnostics:")
                print(f"      STD ratio: {diag.get('std_ratio', 'N/A')}")
                print(f"      Spectral corr: {diag.get('spectral_correlation', 'N/A')}")
            print(f"    Time: {elapsed/60:.1f} min")

            return fold_result
        else:
            print(f"  Warning: Results file not found: {results_file}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"  Error running fold {fold_idx}: {e}")
        return None
    except Exception as e:
        print(f"  Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_experiment(
    config: AdaptiveScalingConfig,
    fold_splits: List[Dict[str, Any]],
    output_dir: Path,
    seed: int = 42,
    verbose: bool = True,
    use_fsdp: bool = False,
    fsdp_strategy: str = "grad_op",
    dry_run: bool = False,
) -> ExperimentResult:
    """Run a full experiment with all folds."""
    is_no_scaling = config.name == "no_adaptive_scaling"

    result = ExperimentResult(config=config)

    for fold_split in fold_splits:
        fold_result = run_single_fold(
            config=config,
            fold_split=fold_split,
            output_dir=output_dir,
            seed=seed,
            verbose=verbose,
            use_fsdp=use_fsdp,
            fsdp_strategy=fsdp_strategy,
            dry_run=dry_run,
            is_no_scaling=is_no_scaling,
        )

        if fold_result is not None:
            result.fold_results.append(fold_result)

    result.compute_statistics()
    return result


# =============================================================================
# Results Summary
# =============================================================================

def print_summary(results: Dict[str, ExperimentResult]) -> None:
    """Print a summary table of all results."""
    print("\n" + "="*90)
    print("ADAPTIVE SCALING EXPLORATION RESULTS")
    print("="*90)

    # Sort by mean R²
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].mean_r2,
        reverse=True
    )

    # Table header
    print(f"\n{'Config':<35} {'R² (mean±std)':<18} {'PSD Diff (dB)':<15} {'PSD Err (dB)':<15} {'Time':<10}")
    print("-"*90)

    for name, result in sorted_results:
        r2_str = f"{result.mean_r2:.4f}±{result.std_r2:.4f}"
        psd_diff_str = f"{result.mean_psd_diff_db:+.2f}±{result.std_psd_diff_db:.2f}"
        psd_err_str = f"{result.mean_psd_err_db:.2f}±{result.std_psd_err_db:.2f}"
        time_str = f"{result.total_time/60:.1f}m"

        print(f"{name:<35} {r2_str:<18} {psd_diff_str:<15} {psd_err_str:<15} {time_str:<10}")

    print("-"*90)

    # Best result
    best_name, best_result = sorted_results[0]
    print(f"\nBest configuration: {best_name}")
    print(f"  R²: {best_result.mean_r2:.4f} ± {best_result.std_r2:.4f}")
    print(f"  PSD Diff: {best_result.mean_psd_diff_db:+.2f} ± {best_result.std_psd_diff_db:.2f} dB")
    print(f"  PSD Err: {best_result.mean_psd_err_db:.2f} ± {best_result.std_psd_err_db:.2f} dB")


def save_results(results: Dict[str, ExperimentResult], output_dir: Path) -> None:
    """Save all results to JSON."""
    output_file = output_dir / "adaptive_scaling_results.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "results": {name: r.to_dict() for name, r in results.items()},
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to: {output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Explore adaptive scaling modes for session generalization"
    )

    parser.add_argument(
        "--output-dir", type=str, default="results/adaptive_scaling",
        help="Output directory for results"
    )
    parser.add_argument(
        "--modes", type=str, nargs="+", default=None,
        choices=["scalar", "band_wise", "spectral", "harmonic"],
        help="Modes to test (default: all)"
    )
    parser.add_argument(
        "--no-cross-channel", action="store_true",
        help="Skip cross-channel variants"
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="Skip no-adaptive-scaling baseline"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--fsdp", action="store_true",
        help="Use FSDP for multi-GPU training"
    )
    parser.add_argument(
        "--fsdp-strategy", type=str, default="grad_op",
        help="FSDP sharding strategy"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without running"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show training output"
    )
    parser.add_argument(
        "--dataset", type=str, default="olfactory",
        help="Dataset to use"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ADAPTIVE SCALING MODE EXPLORATION")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Modes: {args.modes or 'all'}")
    print(f"Cross-channel: {not args.no_cross_channel}")
    print(f"Baseline: {not args.no_baseline}")
    print(f"Seed: {args.seed}")
    print(f"FSDP: {args.fsdp}")
    print("="*70 + "\n")

    # Get session splits
    all_sessions = get_all_sessions(args.dataset)
    fold_splits = get_3fold_session_splits(all_sessions)

    print(f"Sessions: {len(all_sessions)}")
    print(f"Folds: {len(fold_splits)}")
    for split in fold_splits:
        print(f"  Fold {split['fold_idx']}: test={split['test_sessions']}")

    # Get configurations
    configs = get_adaptive_scaling_configs(
        modes=args.modes,
        include_cross_channel=not args.no_cross_channel,
        include_no_scaling=not args.no_baseline,
    )

    print(f"\nConfigurations to test: {len(configs)}")
    for name in configs:
        print(f"  - {name}")

    # Run experiments
    all_results = {}

    for name, config in configs.items():
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT: {name}")
        print(f"{'#'*70}")

        result = run_experiment(
            config=config,
            fold_splits=fold_splits,
            output_dir=output_dir,
            seed=args.seed,
            verbose=args.verbose,
            use_fsdp=args.fsdp,
            fsdp_strategy=args.fsdp_strategy,
            dry_run=args.dry_run,
        )

        all_results[name] = result

        # Save intermediate results
        if not args.dry_run:
            save_results(all_results, output_dir)

    # Print summary
    if not args.dry_run:
        print_summary(all_results)
        save_results(all_results, output_dir)


if __name__ == "__main__":
    main()
