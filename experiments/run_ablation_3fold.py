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

    # =========================================================================
    # WIDTH ABLATIONS (base_channels)
    # =========================================================================
    configs["width_narrow"] = AblationConfig(
        name="width_narrow",
        description="Narrow network (base_channels=64) vs default 128",
        n_downsample=4,
        base_channels=64,  # Half the default
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    )

    configs["width_wide"] = AblationConfig(
        name="width_wide",
        description="Wide network (base_channels=256) vs default 128",
        n_downsample=4,
        base_channels=256,  # Double the default
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    )

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
# Result Classes - COMPREHENSIVE
# =============================================================================

@dataclass
class FoldResult:
    """Result from a single fold of ablation experiment.

    Captures ALL metrics from train.py output for comprehensive analysis.
    """
    fold_idx: int
    test_sessions: List[str]
    train_sessions: List[str]

    # Primary metrics
    val_r2: float
    val_loss: float
    val_corr: float = 0.0
    val_mae: float = 0.0

    # Training metadata
    epochs_trained: int = 0
    total_time: float = 0.0
    n_parameters: int = 0
    best_epoch: int = 0

    # Training curves (full history)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_r2s: List[float] = field(default_factory=list)
    val_corrs: List[float] = field(default_factory=list)

    # Per-session metrics (CRITICAL for session-level analysis)
    per_session_r2: Dict[str, float] = field(default_factory=dict)
    per_session_corr: Dict[str, float] = field(default_factory=dict)
    per_session_loss: Dict[str, float] = field(default_factory=dict)

    # Test set metrics (if available)
    test_avg_r2: Optional[float] = None
    test_avg_corr: Optional[float] = None
    per_session_test_results: List[Dict] = field(default_factory=list)

    # Raw results dict (keep everything)
    raw_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_idx": self.fold_idx,
            "test_sessions": self.test_sessions,
            "train_sessions": self.train_sessions,
            # Primary metrics
            "val_r2": self.val_r2,
            "val_loss": self.val_loss,
            "val_corr": self.val_corr,
            "val_mae": self.val_mae,
            # Training metadata
            "epochs_trained": self.epochs_trained,
            "total_time": self.total_time,
            "n_parameters": self.n_parameters,
            "best_epoch": self.best_epoch,
            # Training curves
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_r2s": self.val_r2s,
            "val_corrs": self.val_corrs,
            # Per-session metrics
            "per_session_r2": self.per_session_r2,
            "per_session_corr": self.per_session_corr,
            "per_session_loss": self.per_session_loss,
            # Test metrics
            "test_avg_r2": self.test_avg_r2,
            "test_avg_corr": self.test_avg_corr,
            "per_session_test_results": self.per_session_test_results,
        }


@dataclass
class AblationResult:
    """Aggregated results for one ablation configuration.

    Includes comprehensive statistics and per-session analysis.
    """
    config: AblationConfig
    fold_results: List[FoldResult]

    # Primary statistics (computed after all folds)
    mean_r2: float = 0.0
    std_r2: float = 0.0
    sem_r2: float = 0.0
    median_r2: float = 0.0
    min_r2: float = 0.0
    max_r2: float = 0.0

    # Confidence intervals
    ci_lower_r2: float = 0.0
    ci_upper_r2: float = 0.0

    # Correlation statistics
    mean_corr: float = 0.0
    std_corr: float = 0.0

    # Loss statistics
    mean_loss: float = 0.0
    std_loss: float = 0.0

    # Per-session aggregated statistics
    all_session_r2s: Dict[str, float] = field(default_factory=dict)
    all_session_corrs: Dict[str, float] = field(default_factory=dict)
    session_r2_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Training metadata
    mean_epochs: float = 0.0
    mean_time: float = 0.0
    total_time: float = 0.0
    n_parameters: int = 0

    # Raw fold values for statistical tests
    fold_r2_values: List[float] = field(default_factory=list)
    fold_corr_values: List[float] = field(default_factory=list)
    fold_loss_values: List[float] = field(default_factory=list)

    def compute_statistics(self) -> None:
        """Compute comprehensive aggregate statistics from fold results."""
        if not self.fold_results:
            return

        # Extract fold-level values
        self.fold_r2_values = [r.val_r2 for r in self.fold_results]
        self.fold_corr_values = [r.val_corr for r in self.fold_results]
        self.fold_loss_values = [r.val_loss for r in self.fold_results]

        r2_arr = np.array(self.fold_r2_values)
        corr_arr = np.array(self.fold_corr_values)
        loss_arr = np.array(self.fold_loss_values)

        # R² statistics
        self.mean_r2 = float(np.mean(r2_arr))
        self.std_r2 = float(np.std(r2_arr, ddof=1)) if len(r2_arr) > 1 else 0.0
        self.sem_r2 = float(self.std_r2 / np.sqrt(len(r2_arr))) if len(r2_arr) > 1 else 0.0
        self.median_r2 = float(np.median(r2_arr))
        self.min_r2 = float(np.min(r2_arr))
        self.max_r2 = float(np.max(r2_arr))

        # Confidence interval for R² (95%)
        if len(r2_arr) >= 2:
            from scipy import stats as scipy_stats
            t_crit = scipy_stats.t.ppf(0.975, len(r2_arr) - 1)
            margin = t_crit * self.sem_r2
            self.ci_lower_r2 = self.mean_r2 - margin
            self.ci_upper_r2 = self.mean_r2 + margin

        # Correlation statistics
        self.mean_corr = float(np.mean(corr_arr))
        self.std_corr = float(np.std(corr_arr, ddof=1)) if len(corr_arr) > 1 else 0.0

        # Loss statistics
        self.mean_loss = float(np.mean(loss_arr))
        self.std_loss = float(np.std(loss_arr, ddof=1)) if len(loss_arr) > 1 else 0.0

        # Training metadata
        self.mean_epochs = float(np.mean([r.epochs_trained for r in self.fold_results]))
        self.mean_time = float(np.mean([r.total_time for r in self.fold_results]))
        self.total_time = sum(r.total_time for r in self.fold_results)
        self.n_parameters = self.fold_results[0].n_parameters if self.fold_results else 0

        # Aggregate per-session R2s across folds
        session_r2_lists: Dict[str, List[float]] = {}
        session_corr_lists: Dict[str, List[float]] = {}

        for fold in self.fold_results:
            for session, r2 in fold.per_session_r2.items():
                self.all_session_r2s[session] = r2
                if session not in session_r2_lists:
                    session_r2_lists[session] = []
                session_r2_lists[session].append(r2)

            for session, corr in fold.per_session_corr.items():
                self.all_session_corrs[session] = corr
                if session not in session_corr_lists:
                    session_corr_lists[session] = []
                session_corr_lists[session].append(corr)

        # Compute per-session statistics (if session appears in multiple folds, which shouldn't happen in our design)
        for session, r2_list in session_r2_lists.items():
            self.session_r2_stats[session] = {
                "r2": float(np.mean(r2_list)),
                "n_folds": len(r2_list),
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "fold_results": [f.to_dict() for f in self.fold_results],
            # Primary R² statistics
            "mean_r2": self.mean_r2,
            "std_r2": self.std_r2,
            "sem_r2": self.sem_r2,
            "median_r2": self.median_r2,
            "min_r2": self.min_r2,
            "max_r2": self.max_r2,
            "ci_lower_r2": self.ci_lower_r2,
            "ci_upper_r2": self.ci_upper_r2,
            # Correlation statistics
            "mean_corr": self.mean_corr,
            "std_corr": self.std_corr,
            # Loss statistics
            "mean_loss": self.mean_loss,
            "std_loss": self.std_loss,
            # Per-session data
            "all_session_r2s": self.all_session_r2s,
            "all_session_corrs": self.all_session_corrs,
            "session_r2_stats": self.session_r2_stats,
            # Training metadata
            "mean_epochs": self.mean_epochs,
            "mean_time": self.mean_time,
            "total_time": self.total_time,
            "n_parameters": self.n_parameters,
            # Raw values for statistical tests
            "fold_r2_values": self.fold_r2_values,
            "fold_corr_values": self.fold_corr_values,
            "fold_loss_values": self.fold_loss_values,
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

    # Load results - CAPTURE EVERYTHING
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Create comprehensive FoldResult with ALL metrics
        fold_result = FoldResult(
            fold_idx=fold_idx,
            test_sessions=test_sessions,
            train_sessions=train_sessions,
            # Primary metrics
            val_r2=results.get("best_val_r2", results.get("val_r2", 0.0)),
            val_loss=results.get("best_val_loss", results.get("val_loss", float('inf'))),
            val_corr=results.get("best_val_corr", 0.0),
            val_mae=results.get("best_val_mae", 0.0),
            # Training metadata
            epochs_trained=results.get("epochs_trained", ablation_config.epochs),
            total_time=elapsed,
            n_parameters=results.get("n_parameters", 0),
            best_epoch=results.get("best_epoch", 0),
            # Training curves (full history)
            train_losses=results.get("train_losses", []),
            val_losses=results.get("val_losses", []),
            val_r2s=results.get("val_r2s", []),
            val_corrs=results.get("val_corrs", []),
            # Per-session metrics
            per_session_r2=results.get("per_session_r2", {}),
            per_session_corr=results.get("per_session_corr", {}),
            per_session_loss=results.get("per_session_loss", {}),
            # Test metrics (if available)
            test_avg_r2=results.get("test_avg_r2"),
            test_avg_corr=results.get("test_avg_corr"),
            per_session_test_results=results.get("per_session_test_results", []),
            # Keep raw results for reference
            raw_results=results,
        )

        print(f"\n  Fold {fold_idx} completed:")
        print(f"    Val R2: {fold_result.val_r2:.4f}")
        print(f"    Val Corr: {fold_result.val_corr:.4f}")
        print(f"    Val Loss: {fold_result.val_loss:.4f}")
        print(f"    Best Epoch: {fold_result.best_epoch}")
        print(f"    Parameters: {fold_result.n_parameters:,}")
        print(f"    Time: {elapsed/60:.1f} minutes")
        if fold_result.per_session_r2:
            print(f"    Per-session R2: {fold_result.per_session_r2}")

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


def compute_statistical_comparisons(
    results: Dict[str, AblationResult],
    baseline_name: str = "baseline",
) -> Dict[str, Dict[str, Any]]:
    """Compute comprehensive statistical comparisons vs baseline.

    Returns dict with effect sizes, p-values, CIs for each ablation.
    """
    from scipy import stats as scipy_stats

    if baseline_name not in results:
        return {}

    baseline = results[baseline_name]
    baseline_r2s = np.array(baseline.fold_r2_values)

    if len(baseline_r2s) < 2:
        return {}

    comparisons = {}

    for name, result in results.items():
        if name == baseline_name:
            continue

        ablation_r2s = np.array(result.fold_r2_values)

        if len(ablation_r2s) < 2:
            continue

        # Basic statistics
        mean_diff = result.mean_r2 - baseline.mean_r2
        diff = ablation_r2s - baseline_r2s  # For paired tests

        # Cohen's d (paired)
        if np.std(diff, ddof=1) > 0:
            cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        else:
            cohens_d = 0.0

        # Hedges' g (bias-corrected for small samples)
        n = len(diff)
        correction = 1 - (3 / (4 * n - 1)) if n > 1 else 1.0
        hedges_g = cohens_d * correction

        # Effect size interpretation
        d_abs = abs(cohens_d)
        if d_abs < 0.2:
            effect_interp = "negligible"
        elif d_abs < 0.5:
            effect_interp = "small"
        elif d_abs < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"

        # Paired t-test
        try:
            t_stat, t_pvalue = scipy_stats.ttest_rel(ablation_r2s, baseline_r2s)
        except Exception:
            t_stat, t_pvalue = 0.0, 1.0

        # Wilcoxon signed-rank test (non-parametric)
        try:
            diff_nonzero = diff[diff != 0]
            if len(diff_nonzero) >= 2:
                w_stat, w_pvalue = scipy_stats.wilcoxon(diff_nonzero)
            else:
                w_stat, w_pvalue = 0.0, 1.0
        except Exception:
            w_stat, w_pvalue = 0.0, 1.0

        # Confidence interval for mean difference
        if len(diff) >= 2:
            se = scipy_stats.sem(diff)
            t_crit = scipy_stats.t.ppf(0.975, len(diff) - 1)
            ci_lower = np.mean(diff) - t_crit * se
            ci_upper = np.mean(diff) + t_crit * se
        else:
            ci_lower, ci_upper = mean_diff, mean_diff

        # Normality test on differences
        try:
            _, norm_p = scipy_stats.shapiro(diff)
            normality_ok = norm_p > 0.05
        except Exception:
            norm_p, normality_ok = 1.0, True

        # Significance markers
        if t_pvalue < 0.001:
            sig_marker = "***"
        elif t_pvalue < 0.01:
            sig_marker = "**"
        elif t_pvalue < 0.05:
            sig_marker = "*"
        else:
            sig_marker = "ns"

        comparisons[name] = {
            "mean_diff": float(mean_diff),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "cohens_d": float(cohens_d),
            "hedges_g": float(hedges_g),
            "effect_interpretation": effect_interp,
            "t_statistic": float(t_stat),
            "t_pvalue": float(t_pvalue),
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_pvalue": float(w_pvalue),
            "normality_pvalue": float(norm_p),
            "normality_ok": normality_ok,
            "recommended_test": "t-test" if normality_ok else "Wilcoxon",
            "recommended_pvalue": float(t_pvalue) if normality_ok else float(w_pvalue),
            "significance_marker": sig_marker,
            "significant_005": t_pvalue < 0.05,
            "significant_001": t_pvalue < 0.01,
        }

    # Apply multiple comparison correction (Holm-Bonferroni)
    if comparisons:
        p_values = [(name, c["t_pvalue"]) for name, c in comparisons.items()]
        sorted_pvals = sorted(p_values, key=lambda x: x[1])
        n_comparisons = len(p_values)

        for i, (name, p) in enumerate(sorted_pvals):
            threshold = 0.05 / (n_comparisons - i)
            comparisons[name]["holm_significant"] = p < threshold
            comparisons[name]["holm_threshold"] = threshold

    return comparisons


def print_summary(results: Dict[str, AblationResult], total_time: float) -> None:
    """Print comprehensive summary of ablation study results with statistics."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 80)

    # Find baseline
    baseline_result = results.get("baseline")
    baseline_r2 = baseline_result.mean_r2 if baseline_result else 0.0

    # Compute statistical comparisons
    comparisons = compute_statistical_comparisons(results, "baseline")

    # Sort by mean R2
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].mean_r2,
        reverse=True
    )

    # Print main results table
    print(f"\n{'Ablation':<22} {'R2 (mean±std)':<18} {'95% CI':<18} {'Delta':>8} {'d':>6} {'p':>8} {'Sig':>4}")
    print("-" * 90)

    for name, result in sorted_results:
        r2_str = f"{result.mean_r2:.4f}±{result.std_r2:.4f}"
        ci_str = f"[{result.ci_lower_r2:.4f},{result.ci_upper_r2:.4f}]"

        if name == "baseline":
            delta_str = "--"
            d_str = "--"
            p_str = "--"
            sig_str = "--"
        else:
            delta = result.mean_r2 - baseline_r2
            delta_str = f"{delta:+.4f}"

            if name in comparisons:
                comp = comparisons[name]
                d_str = f"{comp['cohens_d']:.2f}"
                p_str = f"{comp['recommended_pvalue']:.4f}"
                sig_str = comp['significance_marker']
            else:
                d_str = "--"
                p_str = "--"
                sig_str = "--"

        print(f"{name:<22} {r2_str:<18} {ci_str:<18} {delta_str:>8} {d_str:>6} {p_str:>8} {sig_str:>4}")

    print("-" * 90)

    # Print statistical summary
    if comparisons:
        print("\nSTATISTICAL ANALYSIS (vs baseline):")
        print("-" * 60)

        # Count significant results
        sig_005 = sum(1 for c in comparisons.values() if c['significant_005'])
        sig_001 = sum(1 for c in comparisons.values() if c['significant_001'])
        holm_sig = sum(1 for c in comparisons.values() if c.get('holm_significant', False))

        print(f"  Significant at p<0.05: {sig_005}/{len(comparisons)}")
        print(f"  Significant at p<0.01: {sig_001}/{len(comparisons)}")
        print(f"  Significant (Holm-corrected): {holm_sig}/{len(comparisons)}")

        # Effect size summary
        large_effects = [n for n, c in comparisons.items() if c['effect_interpretation'] == 'large']
        medium_effects = [n for n, c in comparisons.items() if c['effect_interpretation'] == 'medium']

        if large_effects:
            print(f"  Large effects (|d|≥0.8): {', '.join(large_effects)}")
        if medium_effects:
            print(f"  Medium effects (0.5≤|d|<0.8): {', '.join(medium_effects)}")

    # Print per-session summary
    print("\nPER-SESSION R² SUMMARY:")
    print("-" * 60)
    all_sessions = set()
    for result in results.values():
        all_sessions.update(result.all_session_r2s.keys())

    if all_sessions:
        for session in sorted(all_sessions):
            session_r2s = []
            for name, result in results.items():
                if session in result.all_session_r2s:
                    session_r2s.append((name, result.all_session_r2s[session]))
            if session_r2s:
                best = max(session_r2s, key=lambda x: x[1])
                worst = min(session_r2s, key=lambda x: x[1])
                print(f"  {session}: best={best[0]} ({best[1]:.4f}), worst={worst[0]} ({worst[1]:.4f})")

    # Print timing
    print(f"\nTotal time: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")
    print("=" * 80)


def save_summary(results: Dict[str, AblationResult], output_dir: Path) -> None:
    """Save comprehensive summary with all statistics to JSON files."""
    output_dir = Path(output_dir)

    # Compute statistical comparisons
    comparisons = compute_statistical_comparisons(results, "baseline")

    # Find baseline
    baseline_result = results.get("baseline")
    baseline_r2 = baseline_result.mean_r2 if baseline_result else 0.0

    # =========================================================================
    # 1. MAIN SUMMARY FILE (ablation_summary.json)
    # =========================================================================
    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_ablations": len(results),
            "n_folds": 3,
            "baseline": "baseline" if "baseline" in results else None,
        },
        "results": {name: result.to_dict() for name, result in results.items()},
        "statistical_comparisons": comparisons,
    }

    # Build summary table (sorted by R²)
    summary_table = []
    for name, result in sorted(results.items(), key=lambda x: x[1].mean_r2, reverse=True):
        entry = {
            "rank": len(summary_table) + 1,
            "name": name,
            "description": result.config.description,
            # R² statistics
            "mean_r2": result.mean_r2,
            "std_r2": result.std_r2,
            "sem_r2": result.sem_r2,
            "ci_lower_r2": result.ci_lower_r2,
            "ci_upper_r2": result.ci_upper_r2,
            "median_r2": result.median_r2,
            "min_r2": result.min_r2,
            "max_r2": result.max_r2,
            # Correlation statistics
            "mean_corr": result.mean_corr,
            "std_corr": result.std_corr,
            # Loss statistics
            "mean_loss": result.mean_loss,
            "std_loss": result.std_loss,
            # Comparison vs baseline
            "delta_vs_baseline": result.mean_r2 - baseline_r2 if name != "baseline" else 0.0,
            # Training info
            "n_folds": len(result.fold_results),
            "n_parameters": result.n_parameters,
            "mean_epochs": result.mean_epochs,
            "total_time_minutes": result.total_time / 60,
        }

        # Add statistical comparison if available
        if name in comparisons:
            comp = comparisons[name]
            entry["cohens_d"] = comp["cohens_d"]
            entry["hedges_g"] = comp["hedges_g"]
            entry["effect_size"] = comp["effect_interpretation"]
            entry["t_pvalue"] = comp["t_pvalue"]
            entry["wilcoxon_pvalue"] = comp["wilcoxon_pvalue"]
            entry["significance"] = comp["significance_marker"]
            entry["holm_significant"] = comp.get("holm_significant", False)

        summary_table.append(entry)

    summary["summary_table"] = summary_table

    # Save main summary
    summary_file = output_dir / "ablation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nMain summary saved to: {summary_file}")

    # =========================================================================
    # 2. STATISTICAL ANALYSIS FILE (statistical_analysis.json)
    # =========================================================================
    stats_analysis = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "baseline": "baseline",
            "alpha": 0.05,
            "correction_method": "Holm-Bonferroni",
        },
        "comparisons": comparisons,
        "effect_size_thresholds": {
            "negligible": "|d| < 0.2",
            "small": "0.2 ≤ |d| < 0.5",
            "medium": "0.5 ≤ |d| < 0.8",
            "large": "|d| ≥ 0.8",
        },
        "significance_markers": {
            "***": "p < 0.001",
            "**": "p < 0.01",
            "*": "p < 0.05",
            "ns": "not significant",
        },
    }

    stats_file = output_dir / "statistical_analysis.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_analysis, f, indent=2)
    print(f"Statistical analysis saved to: {stats_file}")

    # =========================================================================
    # 3. PER-SESSION RESULTS FILE (per_session_results.json)
    # =========================================================================
    per_session = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
        },
        "sessions": {},
    }

    # Collect all sessions
    all_sessions = set()
    for result in results.values():
        all_sessions.update(result.all_session_r2s.keys())

    for session in sorted(all_sessions):
        session_data = {"ablations": {}}
        for name, result in results.items():
            if session in result.all_session_r2s:
                session_data["ablations"][name] = {
                    "r2": result.all_session_r2s[session],
                    "corr": result.all_session_corrs.get(session, 0.0),
                }
        if session_data["ablations"]:
            r2_values = [v["r2"] for v in session_data["ablations"].values()]
            session_data["mean_r2"] = float(np.mean(r2_values))
            session_data["std_r2"] = float(np.std(r2_values))
            session_data["best_ablation"] = max(
                session_data["ablations"].items(),
                key=lambda x: x[1]["r2"]
            )[0]
            session_data["worst_ablation"] = min(
                session_data["ablations"].items(),
                key=lambda x: x[1]["r2"]
            )[0]
        per_session["sessions"][session] = session_data

    session_file = output_dir / "per_session_results.json"
    with open(session_file, 'w') as f:
        json.dump(per_session, f, indent=2)
    print(f"Per-session results saved to: {session_file}")

    # =========================================================================
    # 4. TRAINING CURVES FILE (training_curves.json)
    # =========================================================================
    training_curves = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
        },
        "ablations": {},
    }

    for name, result in results.items():
        curves = {
            "folds": [],
            "mean_final_loss": result.mean_loss,
            "mean_final_r2": result.mean_r2,
        }
        for fold in result.fold_results:
            curves["folds"].append({
                "fold_idx": fold.fold_idx,
                "train_losses": fold.train_losses,
                "val_losses": fold.val_losses,
                "val_r2s": fold.val_r2s,
                "val_corrs": fold.val_corrs,
                "best_epoch": fold.best_epoch,
                "epochs_trained": fold.epochs_trained,
            })
        training_curves["ablations"][name] = curves

    curves_file = output_dir / "training_curves.json"
    with open(curves_file, 'w') as f:
        json.dump(training_curves, f, indent=2)
    print(f"Training curves saved to: {curves_file}")

    # =========================================================================
    # 5. PUBLICATION-READY TABLE (results_table.csv)
    # =========================================================================
    try:
        import csv
        csv_file = output_dir / "results_table.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Rank", "Ablation", "R² (mean±std)", "95% CI",
                "Δ vs baseline", "Cohen's d", "Effect Size",
                "p-value", "Significance", "Holm Significant"
            ])
            for entry in summary_table:
                r2_str = f"{entry['mean_r2']:.4f}±{entry['std_r2']:.4f}"
                ci_str = f"[{entry['ci_lower_r2']:.4f}, {entry['ci_upper_r2']:.4f}]"
                delta_str = f"{entry['delta_vs_baseline']:+.4f}" if entry['name'] != 'baseline' else "--"
                d_str = f"{entry.get('cohens_d', 0):.3f}" if entry['name'] != 'baseline' else "--"
                effect_str = entry.get('effect_size', '--') if entry['name'] != 'baseline' else "--"
                p_str = f"{entry.get('t_pvalue', 1):.4f}" if entry['name'] != 'baseline' else "--"
                sig_str = entry.get('significance', '--') if entry['name'] != 'baseline' else "--"
                holm_str = "Yes" if entry.get('holm_significant', False) else "No"

                writer.writerow([
                    entry['rank'], entry['name'], r2_str, ci_str,
                    delta_str, d_str, effect_str, p_str, sig_str, holm_str
                ])
        print(f"Results table saved to: {csv_file}")
    except Exception as e:
        print(f"Warning: Could not save CSV table: {e}")


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
