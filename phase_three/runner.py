#!/usr/bin/env python3
"""
3-Fold Cross-Validation Ablation Study
=======================================

Implements a proper 3-fold cross-validation with NO DATA LEAKAGE:

For each fold:
- 3 sessions are HELD OUT for final TEST evaluation (never seen during training)
- Remaining 6 sessions are split 70/30 trial-wise into TRAIN/VAL
- Model selection is based on VAL performance (NOT test!)
- Final evaluation on held-out TEST sessions

Data flow per fold:
  9 sessions total
  ├── 3 sessions → TEST (held out, excluded during training)
  └── 6 sessions → 70/30 trial-wise split
      ├── ~70% trials → TRAIN
      └── ~30% trials → VAL (for model selection)

This gives us:
- NO data leakage (model never sees test during training OR selection)
- Robust performance estimates (mean ± std across 3 folds)
- Every session gets tested exactly once across all folds
- Proper separation of model selection (val) vs final evaluation (test)

Baseline: Original default (n_downsample=2) as the reference point

Ablation configurations (10 total, ordered logically):
1. baseline: Original default (n_downsample=2) with all components
2. depth_medium: n_downsample=3
3. depth_deep: n_downsample=4
4. width_narrow: base_channels=64
5. width_wide: base_channels=256
6. conv_type_standard: standard convolutions
7. attention_none: no attention
8. skip_type_concat: concatenation skip connections
9. conditioning_none: no conditioning (cond_mode=none)
10. adaptive_scaling_off: no adaptive scaling
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
    batch_size: int = 64  # Larger batch = fewer iterations per epoch
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
# Sweep State Management (Persistent Decisions)
# =============================================================================

@dataclass
class SweepState:
    """Persistent state for sweep decisions.

    Tracks which configurations have been eliminated or promoted.
    Saved to disk and loaded on subsequent runs.
    """
    current_baseline: str = "baseline"
    eliminated: List[str] = field(default_factory=list)
    upgrade_history: List[Dict[str, Any]] = field(default_factory=list)
    decision_log: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_baseline": self.current_baseline,
            "eliminated": self.eliminated,
            "upgrade_history": self.upgrade_history,
            "decision_log": self.decision_log,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SweepState":
        return cls(
            current_baseline=data.get("current_baseline", "baseline"),
            eliminated=data.get("eliminated", []),
            upgrade_history=data.get("upgrade_history", []),
            decision_log=data.get("decision_log", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


def load_sweep_state(output_dir: Path) -> SweepState:
    """Load sweep state from disk, or create new if doesn't exist."""
    state_file = output_dir / "sweep_state.json"
    if state_file.exists():
        with open(state_file, 'r') as f:
            data = json.load(f)
        state = SweepState.from_dict(data)
        print(f"\nLoaded sweep state from {state_file}")
        print(f"  Current baseline: {state.current_baseline}")
        print(f"  Eliminated: {state.eliminated}")
        return state
    else:
        state = SweepState(created_at=datetime.now().isoformat())
        print(f"\nNo existing sweep state found. Starting fresh.")
        return state


def save_sweep_state(state: SweepState, output_dir: Path) -> None:
    """Save sweep state to disk."""
    state.updated_at = datetime.now().isoformat()
    state_file = output_dir / "sweep_state.json"
    with open(state_file, 'w') as f:
        json.dump(state.to_dict(), f, indent=2, cls=NumpyEncoder)
    print(f"Sweep state saved to: {state_file}")


def apply_sweep_decisions(
    recommendations: Dict[str, Any],
    sweep_state: SweepState,
    auto_apply: bool = True,
) -> SweepState:
    """Apply sweep decisions based on recommendations.

    Args:
        recommendations: Output from compute_recommendations()
        sweep_state: Current sweep state
        auto_apply: If True, automatically apply decisions

    Returns:
        Updated sweep state
    """
    timestamp = datetime.now().isoformat()

    # Track eliminations
    for cand in recommendations.get("eliminate_candidates", []):
        name = cand["name"]
        if name not in sweep_state.eliminated:
            sweep_state.eliminated.append(name)
            sweep_state.decision_log.append({
                "timestamp": timestamp,
                "action": "ELIMINATE",
                "config": name,
                "delta": cand["delta"],
                "p_value": cand["p_value"],
                "reason": f"Significantly worse ({cand['delta']:.4f} R², p={cand['p_value']:.4f})",
            })
            print(f"  [SWEEP] Permanently ELIMINATED: {name}")

    # Track upgrades (pick the best one)
    if recommendations.get("upgrade_candidates"):
        best = max(recommendations["upgrade_candidates"], key=lambda x: x["delta"])
        new_baseline = best["name"]

        if new_baseline != sweep_state.current_baseline:
            old_baseline = sweep_state.current_baseline
            sweep_state.upgrade_history.append({
                "timestamp": timestamp,
                "from": old_baseline,
                "to": new_baseline,
                "delta": best["delta"],
                "p_value": best["p_value"],
            })
            sweep_state.decision_log.append({
                "timestamp": timestamp,
                "action": "UPGRADE",
                "config": new_baseline,
                "from_baseline": old_baseline,
                "delta": best["delta"],
                "p_value": best["p_value"],
                "reason": f"Significantly better (+{best['delta']:.4f} R², p={best['p_value']:.4f})",
            })
            sweep_state.current_baseline = new_baseline
            print(f"  [SWEEP] UPGRADED baseline: {old_baseline} -> {new_baseline}")

    return sweep_state


def filter_ablations_by_sweep_state(
    configs: Dict[str, AblationConfig],
    sweep_state: SweepState,
) -> Dict[str, AblationConfig]:
    """Filter out eliminated ablations from configs.

    Args:
        configs: All ablation configurations
        sweep_state: Current sweep state with eliminations

    Returns:
        Filtered configs (without eliminated ones)
    """
    filtered = {}
    skipped = []

    for name, config in configs.items():
        if name in sweep_state.eliminated:
            skipped.append(name)
        else:
            filtered[name] = config

    if skipped:
        print(f"\n[SWEEP] Skipping {len(skipped)} eliminated ablations: {skipped}")

    return filtered


# =============================================================================
# Define Ablation Configurations (ORDERED)
# =============================================================================

def get_ablation_configs() -> Dict[str, AblationConfig]:
    """Define all ablation configurations to test.

    Baseline: Original default (n_downsample=2) with modern convs, cross_freq_v2
              attention, adaptive scaling, spectro_temporal conditioning.

    ORDER MATTERS: Configs are ordered logically from fundamental architecture
    choices (depth, width) to component-level choices (conv, attention, etc.)

    Total: 11 ablations (baseline + 10 variants)
    """
    # Use list to preserve insertion order, then convert to dict
    configs_list = []

    # =========================================================================
    # 1. BASELINE (original default - n_downsample=2)
    # =========================================================================
    configs_list.append(AblationConfig(
        name="baseline",
        description="Original default baseline (n_downsample=2) with all components",
        n_downsample=2,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
        use_bidirectional=False,
    ))

    # =========================================================================
    # 2-3. DEPTH ABLATIONS (fundamental architecture choice)
    # =========================================================================
    configs_list.append(AblationConfig(
        name="depth_medium",
        description="Medium depth (n_downsample=3) vs baseline",
        n_downsample=3,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    ))

    configs_list.append(AblationConfig(
        name="depth_deep",
        description="Deep network (n_downsample=4) vs baseline",
        n_downsample=4,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    ))

    # =========================================================================
    # 4-5. WIDTH ABLATIONS (fundamental architecture choice)
    # =========================================================================
    configs_list.append(AblationConfig(
        name="width_narrow",
        description="Narrow network (base_channels=64) vs default 128",
        n_downsample=2,
        base_channels=64,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    ))

    configs_list.append(AblationConfig(
        name="width_wide",
        description="Wide network (base_channels=256) vs default 128",
        n_downsample=2,
        base_channels=256,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    ))

    # =========================================================================
    # 6. CONVOLUTION TYPE ABLATION
    # =========================================================================
    configs_list.append(AblationConfig(
        name="conv_type_standard",
        description="Standard convolutions instead of modern (dilated depthwise separable)",
        n_downsample=2,
        conv_type="standard",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    ))

    # =========================================================================
    # 7-8. ATTENTION ABLATIONS
    # =========================================================================
    configs_list.append(AblationConfig(
        name="attention_none",
        description="No attention mechanism",
        n_downsample=2,
        conv_type="modern",
        attention_type="none",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    ))

    # =========================================================================
    # 8. SKIP CONNECTION ABLATION
    # =========================================================================
    configs_list.append(AblationConfig(
        name="skip_type_concat",
        description="Concatenation skip connections instead of addition",
        n_downsample=2,
        conv_type="modern",
        attention_type="cross_freq_v2",
        skip_type="concat",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
    ))

    # =========================================================================
    # 10. CONDITIONING ABLATION
    # =========================================================================
    configs_list.append(AblationConfig(
        name="conditioning_none",
        description="No conditioning (cond_mode=none bypasses conditioning entirely)",
        n_downsample=2,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",  # Source doesn't matter when cond_mode=none
        cond_mode="none",
    ))

    # =========================================================================
    # 11. ADAPTIVE SCALING ABLATION
    # =========================================================================
    configs_list.append(AblationConfig(
        name="adaptive_scaling_off",
        description="Disable adaptive output scaling",
        n_downsample=2,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=False,
        conditioning="spectro_temporal",
    ))

    # Convert to ordered dict
    configs = {c.name: c for c in configs_list}
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

    IMPORTANT: Proper separation of val vs test metrics:
    - val_* metrics: From 70/30 trial-wise split on training sessions (model selection)
    - test_* metrics: From held-out test sessions (true generalization, NO leakage)

    The test_r2 is the PRIMARY metric for comparing ablations.
    """
    fold_idx: int
    test_sessions: List[str]
    train_sessions: List[str]

    # Validation metrics (from 70/30 split on training sessions - for model selection)
    val_r2: float
    val_loss: float
    val_corr: float = 0.0
    val_mae: float = 0.0

    # TEST metrics (from held-out sessions - PRIMARY metric for ablation comparison)
    test_r2: float = 0.0
    test_corr: float = 0.0
    test_mae: float = 0.0

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
            # Validation metrics (from 70/30 split - for model selection)
            "val_r2": self.val_r2,
            "val_loss": self.val_loss,
            "val_corr": self.val_corr,
            "val_mae": self.val_mae,
            # TEST metrics (from held-out sessions - PRIMARY for ablation comparison)
            "test_r2": self.test_r2,
            "test_corr": self.test_corr,
            "test_mae": self.test_mae,
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
            # Per-session metrics (from validation)
            "per_session_r2": self.per_session_r2,
            "per_session_corr": self.per_session_corr,
            "per_session_loss": self.per_session_loss,
            # Per-session TEST metrics (from held-out sessions)
            "test_avg_r2": self.test_avg_r2,
            "test_avg_corr": self.test_avg_corr,
            "per_session_test_results": self.per_session_test_results,
        }


@dataclass
class AblationResult:
    """Aggregated results for one ablation configuration.

    IMPORTANT: Uses TEST metrics (from held-out sessions) as PRIMARY metrics.
    Val metrics are only for reference (model selection during training).

    Includes comprehensive statistics and per-session analysis.
    """
    config: AblationConfig
    fold_results: List[FoldResult]

    # PRIMARY statistics - TEST metrics (from held-out sessions, NO leakage)
    mean_r2: float = 0.0  # Mean TEST R² across folds
    std_r2: float = 0.0
    sem_r2: float = 0.0
    median_r2: float = 0.0
    min_r2: float = 0.0
    max_r2: float = 0.0

    # Confidence intervals (for TEST R²)
    ci_lower_r2: float = 0.0
    ci_upper_r2: float = 0.0

    # TEST correlation statistics
    mean_corr: float = 0.0
    std_corr: float = 0.0

    # Validation statistics (for reference only - model selection)
    mean_val_r2: float = 0.0
    std_val_r2: float = 0.0

    # Loss statistics (from validation)
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
        """Compute comprehensive aggregate statistics from fold results.

        PRIMARY metrics are from TEST (held-out sessions) - NO data leakage.
        Val metrics are kept for reference (model selection during training).
        """
        if not self.fold_results:
            return

        # Extract fold-level TEST values (PRIMARY - for ablation comparison)
        self.fold_r2_values = [r.test_r2 for r in self.fold_results]
        self.fold_corr_values = [r.test_corr for r in self.fold_results]

        # Also track validation metrics (for reference)
        val_r2_values = [r.val_r2 for r in self.fold_results]
        self.fold_loss_values = [r.val_loss for r in self.fold_results]

        test_r2_arr = np.array(self.fold_r2_values)
        test_corr_arr = np.array(self.fold_corr_values)
        val_r2_arr = np.array(val_r2_values)
        loss_arr = np.array(self.fold_loss_values)

        # PRIMARY R² statistics (from TEST - held-out sessions)
        self.mean_r2 = float(np.mean(test_r2_arr))
        self.std_r2 = float(np.std(test_r2_arr, ddof=1)) if len(test_r2_arr) > 1 else 0.0
        self.sem_r2 = float(self.std_r2 / np.sqrt(len(test_r2_arr))) if len(test_r2_arr) > 1 else 0.0
        self.median_r2 = float(np.median(test_r2_arr))
        self.min_r2 = float(np.min(test_r2_arr))
        self.max_r2 = float(np.max(test_r2_arr))

        # Confidence interval for TEST R² (95%)
        if len(test_r2_arr) >= 2:
            from scipy import stats as scipy_stats
            t_crit = scipy_stats.t.ppf(0.975, len(test_r2_arr) - 1)
            margin = t_crit * self.sem_r2
            self.ci_lower_r2 = self.mean_r2 - margin
            self.ci_upper_r2 = self.mean_r2 + margin

        # TEST correlation statistics
        self.mean_corr = float(np.mean(test_corr_arr))
        self.std_corr = float(np.std(test_corr_arr, ddof=1)) if len(test_corr_arr) > 1 else 0.0

        # Validation statistics (for reference - model selection)
        self.mean_val_r2 = float(np.mean(val_r2_arr))
        self.std_val_r2 = float(np.std(val_r2_arr, ddof=1)) if len(val_r2_arr) > 1 else 0.0

        # Loss statistics (from validation)
        self.mean_loss = float(np.mean(loss_arr))
        self.std_loss = float(np.std(loss_arr, ddof=1)) if len(loss_arr) > 1 else 0.0

        # Training metadata
        self.mean_epochs = float(np.mean([r.epochs_trained for r in self.fold_results]))
        self.mean_time = float(np.mean([r.total_time for r in self.fold_results]))
        self.total_time = sum(r.total_time for r in self.fold_results)
        self.n_parameters = self.fold_results[0].n_parameters if self.fold_results else 0

        # Aggregate per-session TEST R²s across folds (from held-out test sessions)
        # Each session appears in exactly one fold's test set, so each session has one R² value
        session_test_r2_dict: Dict[str, float] = {}

        for fold in self.fold_results:
            # per_session_test_results contains TEST metrics (held-out sessions)
            if isinstance(fold.per_session_test_results, dict):
                for session, r2 in fold.per_session_test_results.items():
                    session_test_r2_dict[session] = r2
            elif isinstance(fold.per_session_test_results, list):
                # Handle list format if it exists
                for item in fold.per_session_test_results:
                    if isinstance(item, dict) and "session" in item:
                        session_test_r2_dict[item["session"]] = item.get("r2", 0.0)

        # Store per-session TEST R²s
        self.all_session_r2s = session_test_r2_dict
        self.all_session_corrs = {}  # Could add test correlations if needed

        # Per-session stats (each session tested once, so no aggregation needed)
        for session, r2 in session_test_r2_dict.items():
            self.session_r2_stats[session] = {
                "test_r2": r2,
                "n_folds": 1,  # Each session appears in exactly one fold's test set
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


def evaluate_on_test_sessions(
    checkpoint_path: Path,
    test_sessions: List[str],
    dataset: str = "olfactory",
) -> Dict[str, float]:
    """Evaluate a trained model on held-out test sessions.

    This is the TRUE generalization metric - model has NEVER seen these sessions.

    Args:
        checkpoint_path: Path to the best model checkpoint
        test_sessions: List of session names to evaluate on
        dataset: Dataset name

    Returns:
        Dict with test_r2, test_corr, test_mae, and per_session metrics
    """
    import torch
    from sklearn.metrics import r2_score, mean_absolute_error

    if not checkpoint_path.exists():
        print(f"    Warning: Checkpoint not found at {checkpoint_path}")
        return {"test_r2": 0.0, "test_corr": 0.0, "test_mae": 0.0, "per_session_test_r2": {}}

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Import model creation
        from models import CondUNet1D
        from data import prepare_data, load_session_ids, ODOR_CSV_PATH

        # Get model config from checkpoint
        config = checkpoint.get("config", {})

        # Create model with same architecture
        model = CondUNet1D(
            in_channels=config.get("in_channels", 32),
            out_channels=config.get("out_channels", 32),
            base=config.get("base_channels", 128),  # Note: CLI uses base_channels, model uses base
            n_odors=config.get("n_odors", 7),
            n_downsample=config.get("n_downsample", 2),
            conv_type=config.get("conv_type", "modern"),
            attention_type=config.get("attention_type", "none"),
            skip_type=config.get("skip_type", "add"),
            activation=config.get("activation", "gelu"),
            cond_mode=config.get("cond_mode", "cross_attn_gated"),
            use_adaptive_scaling=config.get("use_adaptive_scaling", True),
            n_heads=config.get("n_heads", 4),
        )

        # Load weights (handle different checkpoint formats)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            # Assume checkpoint is the state dict directly
            model.load_state_dict(checkpoint)
        model.eval()

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Load data for test sessions only
        data = prepare_data(
            split_by_session=False,
            force_recreate_splits=True,
        )

        X = data["ob"]
        y = data["pcx"]
        odors = data["odors"]

        # Get session IDs
        session_ids, session_to_idx, idx_to_session = load_session_ids(
            ODOR_CSV_PATH, num_trials=len(odors)
        )

        # Get indices for test sessions
        test_session_ids = set(session_to_idx[s] for s in test_sessions if s in session_to_idx)
        all_indices = np.arange(len(session_ids))
        test_idx = all_indices[np.isin(session_ids, list(test_session_ids))]

        if len(test_idx) == 0:
            print(f"    Warning: No test indices found for sessions {test_sessions}")
            return {"test_r2": 0.0, "test_corr": 0.0, "test_mae": 0.0, "per_session_test_r2": {}}

        X_test = torch.tensor(X[test_idx], dtype=torch.float32)
        y_test = torch.tensor(y[test_idx], dtype=torch.float32)
        odors_test = torch.tensor(odors[test_idx], dtype=torch.long)

        # Evaluate
        with torch.no_grad():
            X_test_dev = X_test.to(device)
            odors_test_dev = odors_test.to(device)
            y_pred = model(X_test_dev, odors_test_dev).cpu()

        # Compute overall metrics
        y_test_flat = y_test.numpy().flatten()
        y_pred_flat = y_pred.numpy().flatten()

        test_r2 = float(r2_score(y_test_flat, y_pred_flat))
        test_mae = float(mean_absolute_error(y_test_flat, y_pred_flat))
        test_corr = float(np.corrcoef(y_test_flat, y_pred_flat)[0, 1])

        # Compute per-session metrics
        per_session_test_r2 = {}
        test_session_ids_arr = session_ids[test_idx]

        for sess_name in test_sessions:
            if sess_name not in session_to_idx:
                continue
            sess_id = session_to_idx[sess_name]
            local_mask = test_session_ids_arr == sess_id
            if not np.any(local_mask):
                continue

            y_sess = y_test[local_mask].numpy().flatten()
            y_pred_sess = y_pred[local_mask].numpy().flatten()
            per_session_test_r2[sess_name] = float(r2_score(y_sess, y_pred_sess))

        return {
            "test_r2": test_r2,
            "test_corr": test_corr,
            "test_mae": test_mae,
            "per_session_test_r2": per_session_test_r2,
        }

    except Exception as e:
        print(f"    Warning: Test evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"test_r2": 0.0, "test_corr": 0.0, "test_mae": 0.0, "per_session_test_r2": {}}


def run_single_fold(
    ablation_config: AblationConfig,
    fold_split: Dict[str, Any],
    output_dir: Path,
    seed: int = 42,
    verbose: bool = True,
    use_fsdp: bool = False,
    fsdp_strategy: str = "grad_op",
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
        fsdp_strategy: FSDP sharding strategy
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

    # Checkpoint path - unique per config and fold to avoid overwriting
    checkpoint_prefix = f"{ablation_config.name}_fold{fold_idx}"
    checkpoint_path = output_dir / ablation_config.name / f"fold{fold_idx}" / "best_model.pt"

    # Base arguments
    cmd.extend([
        "--arch", "condunet",
        "--dataset", "olfactory",
        "--epochs", str(ablation_config.epochs),
        "--batch-size", str(ablation_config.batch_size),
        "--lr", str(ablation_config.learning_rate),
        "--seed", str(seed + fold_idx),
        "--output-results-file", str(results_file),
        "--checkpoint-prefix", checkpoint_prefix,  # Unique checkpoint per fold
        "--fold", str(fold_idx),
        "--no-plots",
    ])

    # CORRECT DATA SPLIT (no leakage):
    # - Test sessions are EXCLUDED entirely (never seen during training)
    # - Remaining sessions use 70/30 trial-wise split for train/val
    # - Model selection based on val (NOT test)
    cmd.append("--force-recreate-splits")
    cmd.extend(["--exclude-sessions"] + test_sessions)  # Exclude test sessions entirely
    # Default 70/30 trial-wise split on remaining sessions for train/val
    cmd.append("--no-test-set")  # No separate test during training (we evaluate separately)
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
        cmd.extend(["--fsdp", "--fsdp-strategy", fsdp_strategy])

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

        # Evaluate on held-out TEST sessions (TRUE generalization, NO leakage)
        print(f"\n  Evaluating on held-out test sessions: {test_sessions}")

        # Find the checkpoint - try several possible locations
        possible_checkpoint_paths = [
            checkpoint_path,  # Unique path we specified
            PROJECT_ROOT / "artifacts" / "checkpoints" / f"{checkpoint_prefix}_best_model.pt",
            PROJECT_ROOT / "artifacts" / "checkpoints" / "best_model.pt",
        ]

        actual_checkpoint_path = None
        for cp_path in possible_checkpoint_paths:
            if cp_path.exists():
                actual_checkpoint_path = cp_path
                break

        if actual_checkpoint_path:
            test_metrics = evaluate_on_test_sessions(
                checkpoint_path=actual_checkpoint_path,
                test_sessions=test_sessions,
                dataset="olfactory",
            )
            print(f"    TEST R2: {test_metrics['test_r2']:.4f} (PRIMARY METRIC)")
            print(f"    TEST Corr: {test_metrics['test_corr']:.4f}")
            if test_metrics.get('per_session_test_r2'):
                print(f"    Per-session TEST R2: {test_metrics['per_session_test_r2']}")
        else:
            print(f"    WARNING: Could not find checkpoint for test evaluation")
            test_metrics = {"test_r2": 0.0, "test_corr": 0.0, "test_mae": 0.0, "per_session_test_r2": {}}

        # Create comprehensive FoldResult with ALL metrics
        fold_result = FoldResult(
            fold_idx=fold_idx,
            test_sessions=test_sessions,
            train_sessions=train_sessions,
            # Validation metrics (from 70/30 split - for model selection)
            val_r2=results.get("best_val_r2", results.get("val_r2", 0.0)),
            val_loss=results.get("best_val_loss", results.get("val_loss", float('inf'))),
            val_corr=results.get("best_val_corr", 0.0),
            val_mae=results.get("best_val_mae", 0.0),
            # TEST metrics (from held-out sessions - PRIMARY for ablation comparison)
            test_r2=test_metrics["test_r2"],
            test_corr=test_metrics["test_corr"],
            test_mae=test_metrics["test_mae"],
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
            # Per-session metrics (from validation)
            per_session_r2=results.get("per_session_r2", {}),
            per_session_corr=results.get("per_session_corr", {}),
            per_session_loss=results.get("per_session_loss", {}),
            # Per-session TEST metrics
            test_avg_r2=test_metrics.get("test_r2"),
            test_avg_corr=test_metrics.get("test_corr"),
            per_session_test_results=test_metrics.get("per_session_test_r2", {}),
            # Keep raw results for reference
            raw_results=results,
        )

        print(f"\n  Fold {fold_idx} completed:")
        print(f"    Val R2: {fold_result.val_r2:.4f} (model selection)")
        print(f"    TEST R2: {fold_result.test_r2:.4f} (PRIMARY - true generalization)")
        print(f"    Val Corr: {fold_result.val_corr:.4f}")
        print(f"    Val Loss: {fold_result.val_loss:.4f}")
        print(f"    Best Epoch: {fold_result.best_epoch}")
        print(f"    Parameters: {fold_result.n_parameters:,}")
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
    fsdp_strategy: str = "grad_op",
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
        fsdp_strategy: FSDP sharding strategy
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
            fsdp_strategy=fsdp_strategy,
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
        json.dump(ablation_result.to_dict(), f, indent=2, cls=NumpyEncoder)

    return ablation_result


# =============================================================================
# Main Runner
# =============================================================================

def run_3fold_ablation_study(
    output_dir: Path,
    ablations_to_run: Optional[List[str]] = None,
    folds_to_run: Optional[List[int]] = None,
    seed: int = 42,
    epochs: Optional[int] = None,
    verbose: bool = True,
    use_fsdp: bool = False,
    fsdp_strategy: str = "grad_op",
    dry_run: bool = False,
    enable_sweep: bool = True,
) -> Dict[str, AblationResult]:
    """Run the complete 3-fold ablation study.

    Args:
        output_dir: Directory to save all results
        ablations_to_run: Optional list of ablation names to run (default: all)
        folds_to_run: Optional list of fold indices to run (default: all 3)
        seed: Random seed
        epochs: Number of training epochs (default: 80 from config)
        verbose: Print training output
        use_fsdp: Use FSDP for multi-GPU training
        dry_run: If True, print commands without running them
        enable_sweep: If True, apply sweep decisions (eliminate/upgrade permanently)

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
    print(f"Sweep mode: {'ENABLED' if enable_sweep else 'DISABLED'}")
    print()

    # Load sweep state (persistent decisions)
    sweep_state = load_sweep_state(output_dir) if enable_sweep else SweepState()

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

    # Filter out eliminated ablations (from previous sweep decisions)
    if enable_sweep:
        all_configs = filter_ablations_by_sweep_state(all_configs, sweep_state)

    if ablations_to_run is not None:
        configs_to_run = {k: v for k, v in all_configs.items() if k in ablations_to_run}
    else:
        configs_to_run = all_configs

    # Override epochs if specified
    if epochs is not None:
        for config in configs_to_run.values():
            config.epochs = epochs

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
            fsdp_strategy=fsdp_strategy,
            folds_to_run=folds_to_run,
            dry_run=dry_run,
        )
        results[ablation_name] = result

    total_time = time.time() - start_time

    # Print summary
    print_summary(results, total_time)

    # Save summary
    save_summary(results, output_dir)

    # Apply and save sweep decisions (if enabled and not dry run)
    if enable_sweep and not dry_run:
        # Compute comparisons and recommendations
        comparisons = compute_statistical_comparisons(results, "baseline")
        recommendations = compute_recommendations(results, comparisons, "baseline")

        # Apply sweep decisions (eliminate/upgrade)
        sweep_state = apply_sweep_decisions(recommendations, sweep_state)

        # Save updated sweep state
        save_sweep_state(sweep_state, output_dir)

        # Print sweep summary
        print("\n" + "=" * 60)
        print("SWEEP STATE UPDATED")
        print("=" * 60)
        print(f"  Current baseline: {sweep_state.current_baseline}")
        print(f"  Total eliminated: {len(sweep_state.eliminated)}")
        if sweep_state.eliminated:
            print(f"    {sweep_state.eliminated}")
        print(f"  Total upgrades: {len(sweep_state.upgrade_history)}")
        print("=" * 60)

    return results


def compute_statistical_comparisons(
    results: Dict[str, AblationResult],
    baseline_name: str = "baseline",
) -> Dict[str, Dict[str, Any]]:
    """Compute comprehensive statistical comparisons vs baseline.

    NOTE: Uses TEST R² (from held-out sessions) - NO data leakage.
    The fold_r2_values in AblationResult are populated from test_r2, not val_r2.

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


def compute_recommendations(
    results: Dict[str, AblationResult],
    comparisons: Dict[str, Dict[str, Any]],
    baseline_name: str = "baseline",
    significance_threshold: float = 0.05,
    equivalence_margin: float = 0.01,  # R² difference considered "equivalent"
) -> Dict[str, Any]:
    """Compute recommendations based on performance and computational cost.

    Decision logic:
    1. If ablation is SIGNIFICANTLY BETTER (p < threshold, positive delta): UPGRADE
    2. If ablation is SIGNIFICANTLY WORSE (p < threshold, negative delta): ELIMINATE
    3. If ablation is EQUIVALENT (not significant, |delta| < margin) AND cheaper: PREFER
    4. If ablation is EQUIVALENT but more expensive: KEEP BASELINE

    Args:
        results: Dict of ablation results
        comparisons: Statistical comparisons from compute_statistical_comparisons
        baseline_name: Name of baseline configuration
        significance_threshold: p-value threshold for significance
        equivalence_margin: R² difference below which configs are considered equivalent

    Returns:
        Dict with recommendations and reasoning
    """
    if baseline_name not in results:
        return {"error": "Baseline not found"}

    baseline = results[baseline_name]
    baseline_params = baseline.n_parameters
    baseline_time = baseline.mean_time

    recommendations = {
        "baseline": baseline_name,
        "baseline_r2": baseline.mean_r2,
        "baseline_params": baseline_params,
        "baseline_time_minutes": baseline_time / 60,
        "decisions": {},
        "upgrade_candidates": [],  # Significantly better
        "eliminate_candidates": [],  # Significantly worse
        "prefer_candidates": [],  # Equivalent but cheaper
        "keep_baseline_for": [],  # Equivalent but more expensive
        "recommended_config": baseline_name,
        "recommended_reason": "Default baseline",
    }

    for name, result in results.items():
        if name == baseline_name:
            continue

        if name not in comparisons:
            continue

        comp = comparisons[name]
        delta = comp["mean_diff"]
        p_value = comp["recommended_pvalue"]
        is_significant = p_value < significance_threshold

        # Computational cost comparison
        params = result.n_parameters
        time_minutes = result.mean_time / 60
        params_ratio = params / baseline_params if baseline_params > 0 else 1.0
        time_ratio = result.mean_time / baseline_time if baseline_time > 0 else 1.0

        is_cheaper = params_ratio < 0.9 or time_ratio < 0.9  # At least 10% cheaper
        is_more_expensive = params_ratio > 1.1 or time_ratio > 1.1

        decision = {
            "delta_r2": delta,
            "p_value": p_value,
            "is_significant": is_significant,
            "n_parameters": params,
            "params_ratio": params_ratio,
            "time_minutes": time_minutes,
            "time_ratio": time_ratio,
            "is_cheaper": is_cheaper,
            "is_more_expensive": is_more_expensive,
        }

        # Decision logic
        if is_significant and delta > 0:
            # SIGNIFICANTLY BETTER - UPGRADE!
            decision["action"] = "UPGRADE"
            decision["reason"] = f"Significantly better (+{delta:.4f} R², p={p_value:.4f})"
            recommendations["upgrade_candidates"].append({
                "name": name,
                "delta": delta,
                "p_value": p_value,
                "params": params,
            })

        elif is_significant and delta < 0:
            # SIGNIFICANTLY WORSE - ELIMINATE
            decision["action"] = "ELIMINATE"
            decision["reason"] = f"Significantly worse ({delta:.4f} R², p={p_value:.4f})"
            recommendations["eliminate_candidates"].append({
                "name": name,
                "delta": delta,
                "p_value": p_value,
            })

        elif not is_significant and abs(delta) < equivalence_margin:
            # EQUIVALENT performance
            if is_cheaper:
                decision["action"] = "PREFER"
                decision["reason"] = f"Equivalent performance ({delta:+.4f} R²) but {(1-params_ratio)*100:.0f}% fewer params"
                recommendations["prefer_candidates"].append({
                    "name": name,
                    "delta": delta,
                    "params_ratio": params_ratio,
                    "params": params,
                })
            else:
                decision["action"] = "KEEP_BASELINE"
                decision["reason"] = f"Equivalent performance ({delta:+.4f} R²) but not cheaper"
                recommendations["keep_baseline_for"].append(name)

        else:
            # Not significant but notable difference - needs more data
            decision["action"] = "INCONCLUSIVE"
            decision["reason"] = f"Not significant (p={p_value:.4f}) but delta={delta:+.4f}"

        recommendations["decisions"][name] = decision

    # Determine final recommendation
    # Priority: 1) Best significant upgrade, 2) Cheapest equivalent, 3) Baseline
    if recommendations["upgrade_candidates"]:
        # Pick the best significant upgrade
        best_upgrade = max(recommendations["upgrade_candidates"], key=lambda x: x["delta"])
        recommendations["recommended_config"] = best_upgrade["name"]
        recommendations["recommended_reason"] = (
            f"UPGRADE: {best_upgrade['name']} is significantly better "
            f"(+{best_upgrade['delta']:.4f} R², p={best_upgrade['p_value']:.4f})"
        )
    elif recommendations["prefer_candidates"]:
        # Pick the cheapest equivalent
        cheapest = min(recommendations["prefer_candidates"], key=lambda x: x["params"])
        recommendations["recommended_config"] = cheapest["name"]
        recommendations["recommended_reason"] = (
            f"PREFER: {cheapest['name']} has equivalent performance "
            f"({cheapest['delta']:+.4f} R²) with {(1-cheapest['params_ratio'])*100:.0f}% fewer parameters"
        )

    # Summary statistics
    recommendations["summary"] = {
        "n_upgrades": len(recommendations["upgrade_candidates"]),
        "n_eliminations": len(recommendations["eliminate_candidates"]),
        "n_equivalent_cheaper": len(recommendations["prefer_candidates"]),
        "n_equivalent_expensive": len(recommendations["keep_baseline_for"]),
    }

    return recommendations


def print_summary(results: Dict[str, AblationResult], total_time: float) -> None:
    """Print comprehensive summary of ablation study results with statistics.

    NOTE: All R² values shown are TEST R² from held-out sessions (NO data leakage).
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 80)
    print("NOTE: All R² values are from TEST (held-out sessions) - NO data leakage")
    print("      Model selection was based on validation (70/30 split), NOT test")
    print()

    # Find baseline
    baseline_result = results.get("baseline")
    baseline_r2 = baseline_result.mean_r2 if baseline_result else 0.0  # This is TEST R²

    # Compute statistical comparisons (uses TEST R²)
    comparisons = compute_statistical_comparisons(results, "baseline")

    # Sort by mean TEST R²
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].mean_r2,
        reverse=True
    )

    # Print main results table (TEST R²)
    print(f"{'Ablation':<22} {'TEST R² (mean±std)':<20} {'95% CI':<18} {'Delta':>8} {'d':>6} {'p':>8} {'Sig':>4}")
    print("-" * 92)

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

    # Print per-session TEST summary (each session tested in exactly one fold)
    print("\nPER-SESSION TEST R² SUMMARY (from held-out sessions):")
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

    # Compute and print recommendations
    recommendations = compute_recommendations(results, comparisons, "baseline")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Print upgrade candidates (significantly better)
    if recommendations["upgrade_candidates"]:
        print("\n[UPGRADE] Significantly BETTER than baseline:")
        for cand in sorted(recommendations["upgrade_candidates"], key=lambda x: -x["delta"]):
            print(f"  * {cand['name']}: +{cand['delta']:.4f} R² (p={cand['p_value']:.4f})")

    # Print eliminate candidates (significantly worse)
    if recommendations["eliminate_candidates"]:
        print("\n[ELIMINATE] Significantly WORSE than baseline:")
        for cand in sorted(recommendations["eliminate_candidates"], key=lambda x: x["delta"]):
            print(f"  * {cand['name']}: {cand['delta']:.4f} R² (p={cand['p_value']:.4f})")

    # Print prefer candidates (equivalent but cheaper)
    if recommendations["prefer_candidates"]:
        print("\n[PREFER] Equivalent performance but CHEAPER:")
        for cand in sorted(recommendations["prefer_candidates"], key=lambda x: x["params"]):
            savings = (1 - cand["params_ratio"]) * 100
            print(f"  * {cand['name']}: {cand['delta']:+.4f} R², {savings:.0f}% fewer params")

    # Print final recommendation
    print("\n" + "-" * 60)
    print(f"FINAL RECOMMENDATION: {recommendations['recommended_config']}")
    print(f"  Reason: {recommendations['recommended_reason']}")
    print("-" * 60)

    print("=" * 80)


def save_summary(results: Dict[str, AblationResult], output_dir: Path) -> None:
    """Save comprehensive summary with all statistics to JSON files.

    NOTE: All R² metrics are from TEST (held-out sessions) - NO data leakage.
    Val metrics are included for reference but TEST is the primary metric.
    """
    output_dir = Path(output_dir)

    # Compute statistical comparisons (uses TEST R² as primary metric)
    comparisons = compute_statistical_comparisons(results, "baseline")

    # Find baseline
    baseline_result = results.get("baseline")
    baseline_r2 = baseline_result.mean_r2 if baseline_result else 0.0  # This is TEST R²

    # =========================================================================
    # 1. MAIN SUMMARY FILE (ablation_summary.json)
    # =========================================================================
    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_ablations": len(results),
            "n_folds": 3,
            "baseline": "baseline" if "baseline" in results else None,
            "primary_metric": "test_r2",  # From held-out sessions, NO leakage
            "data_split_design": {
                "test": "3 sessions held out per fold (never seen during training)",
                "train_val": "Remaining 6 sessions split 70/30 trial-wise",
                "model_selection": "Based on val (70/30 split), NOT test",
                "ablation_comparison": "Based on test (held-out sessions)",
            },
        },
        "results": {name: result.to_dict() for name, result in results.items()},
        "statistical_comparisons": comparisons,
    }

    # Build summary table (sorted by TEST R² - primary metric)
    summary_table = []
    for name, result in sorted(results.items(), key=lambda x: x[1].mean_r2, reverse=True):
        entry = {
            "rank": len(summary_table) + 1,
            "name": name,
            "description": result.config.description,
            # TEST R² statistics (PRIMARY - from held-out sessions, NO leakage)
            "test_r2_mean": result.mean_r2,
            "test_r2_std": result.std_r2,
            "test_r2_sem": result.sem_r2,
            "test_r2_ci_lower": result.ci_lower_r2,
            "test_r2_ci_upper": result.ci_upper_r2,
            "test_r2_median": result.median_r2,
            "test_r2_min": result.min_r2,
            "test_r2_max": result.max_r2,
            # Validation R² (for reference - model selection)
            "val_r2_mean": result.mean_val_r2,
            "val_r2_std": result.std_val_r2,
            # TEST correlation statistics
            "test_corr_mean": result.mean_corr,
            "test_corr_std": result.std_corr,
            # Loss statistics (from validation)
            "val_loss_mean": result.mean_loss,
            "val_loss_std": result.std_loss,
            # Comparison vs baseline (TEST R²)
            "delta_test_r2_vs_baseline": result.mean_r2 - baseline_r2 if name != "baseline" else 0.0,
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
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
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
        json.dump(stats_analysis, f, indent=2, cls=NumpyEncoder)
    print(f"Statistical analysis saved to: {stats_file}")

    # =========================================================================
    # 3. PER-SESSION TEST RESULTS FILE (per_session_test_results.json)
    # These are TRUE TEST metrics from held-out sessions - NO data leakage
    # =========================================================================
    per_session = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "note": "TEST R² from held-out sessions (NO data leakage)",
            "data_split": "Each session tested in exactly one fold, model never saw it during training",
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
                    "test_r2": result.all_session_r2s[session],  # TEST R² (held-out)
                }
        if session_data["ablations"]:
            r2_values = [v["test_r2"] for v in session_data["ablations"].values()]
            session_data["mean_test_r2"] = float(np.mean(r2_values))
            session_data["std_test_r2"] = float(np.std(r2_values))
            session_data["best_ablation"] = max(
                session_data["ablations"].items(),
                key=lambda x: x[1]["test_r2"]
            )[0]
            session_data["worst_ablation"] = min(
                session_data["ablations"].items(),
                key=lambda x: x[1]["test_r2"]
            )[0]
        per_session["sessions"][session] = session_data

    session_file = output_dir / "per_session_test_results.json"
    with open(session_file, 'w') as f:
        json.dump(per_session, f, indent=2, cls=NumpyEncoder)
    print(f"Per-session TEST results saved to: {session_file}")

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
        json.dump(training_curves, f, indent=2, cls=NumpyEncoder)
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

    # =========================================================================
    # 6. RECOMMENDATIONS FILE (recommendations.json)
    # =========================================================================
    recommendations = compute_recommendations(results, comparisons, "baseline")

    recommendations_file = output_dir / "recommendations.json"
    with open(recommendations_file, 'w') as f:
        json.dump(recommendations, f, indent=2, cls=NumpyEncoder)
    print(f"Recommendations saved to: {recommendations_file}")

    # Also add recommendations to main summary
    summary["recommendations"] = recommendations
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)


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
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 80)",
    )

    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Use FSDP for multi-GPU training",
    )

    parser.add_argument(
        "--fsdp-strategy",
        type=str,
        default="grad_op",
        choices=["full", "grad_op", "no_shard"],
        help="FSDP sharding strategy (default: grad_op)",
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

    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Disable sweep mode (don't apply permanent eliminate/upgrade decisions)",
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
        epochs=args.epochs,
        verbose=not args.quiet,
        use_fsdp=args.fsdp,
        fsdp_strategy=args.fsdp_strategy,
        dry_run=args.dry_run,
        enable_sweep=not args.no_sweep,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
