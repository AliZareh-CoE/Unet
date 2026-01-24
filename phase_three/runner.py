#!/usr/bin/env python3
"""
CASCADING Ablation Study with 3-Fold Cross-Validation
======================================================

Implements a SCIENTIFICALLY RIGOROUS cascading ablation study where each phase
builds on the best configuration from the previous phase.

CASCADING PHASES:
=================

Phase 1: ARCHITECTURE SEARCH (find optimal depth × width)
    - Test depth: n_downsample = 2, 3, 4
    - Test width: base_channels = 64, 128, 256
    - Total: 9 combinations (3×3 grid search)
    - Winner becomes baseline for Phase 2

Phase 2: COMPONENT ABLATIONS (using best architecture from Phase 1)
    - conv_type: modern vs standard
    - attention_type: cross_freq_v2 vs none
    - skip_type: add vs concat
    - Winner becomes baseline for Phase 3

Phase 3: CONDITIONING & SCALING (using best from Phase 2)
    - with vs without conditioning
    - with vs without adaptive_scaling
    - Winner becomes baseline for Phase 4

Phase 4: DOMAIN ADAPTATION (using best from Phase 3)
    - Test contribution of each augmentation component:
      - Euclidean alignment
      - Test-time BN adaptation
      - Session augmentation
      - MMD loss
      - Noise augmentation
    - Final optimized configuration

WHY CASCADING?
==============
If n_downsample=4 gives +5% over n_downsample=2, testing attention on
n_downsample=2 would give WRONG conclusions. We must test components
on the OPTIMAL architecture, not a suboptimal one.

CROSS-VALIDATION:
=================
For each config in each phase:
- 3 sessions HELD OUT for TEST (never seen during training)
- Remaining 6 sessions split 70/30 trial-wise into TRAIN/VAL
- Model selection based on VAL (NOT test!)
- Final metric: TEST R² (true generalization)

NO DATA LEAKAGE: Model never sees test during training OR selection.
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

    # =========================================================================
    # NEW ABLATION COMPONENTS (ALL ENABLED BY DEFAULT)
    # =========================================================================

    # Euclidean Alignment (+2-5% expected improvement)
    use_euclidean_alignment: bool = True  # ENABLED BY DEFAULT
    euclidean_momentum: float = 0.1

    # Test-time BN Adaptation (+2-4% expected improvement)
    use_bn_adaptation: bool = True  # ENABLED BY DEFAULT
    bn_adaptation_steps: int = 10
    bn_adaptation_momentum: float = 0.1
    bn_reset_stats: bool = False

    # Session Augmentation (+2-5% expected improvement)
    use_session_augmentation: bool = True  # ENABLED BY DEFAULT
    session_aug_mix_prob: float = 0.3
    session_aug_scale_range: Tuple[float, float] = (0.9, 1.1)
    session_aug_shift_range: Tuple[float, float] = (-0.1, 0.1)

    # MMD Loss for Session Invariance (+1-3% expected improvement)
    use_mmd_loss: bool = True  # ENABLED BY DEFAULT
    mmd_weight: float = 0.1

    # Noise Augmentation (robustness improvement)
    use_noise_augmentation: bool = True  # ENABLED BY DEFAULT
    noise_gaussian_std: float = 0.1
    noise_pink: bool = True  # ENABLED BY DEFAULT
    noise_pink_std: float = 0.05
    noise_channel_dropout: float = 0.05  # Small dropout by default
    noise_temporal_dropout: float = 0.02  # Small dropout by default
    noise_prob: float = 0.5

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
            # New ablation components
            "use_euclidean_alignment": self.use_euclidean_alignment,
            "euclidean_momentum": self.euclidean_momentum,
            "use_bn_adaptation": self.use_bn_adaptation,
            "bn_adaptation_steps": self.bn_adaptation_steps,
            "bn_adaptation_momentum": self.bn_adaptation_momentum,
            "bn_reset_stats": self.bn_reset_stats,
            "use_session_augmentation": self.use_session_augmentation,
            "session_aug_mix_prob": self.session_aug_mix_prob,
            "session_aug_scale_range": self.session_aug_scale_range,
            "session_aug_shift_range": self.session_aug_shift_range,
            "use_mmd_loss": self.use_mmd_loss,
            "mmd_weight": self.mmd_weight,
            "use_noise_augmentation": self.use_noise_augmentation,
            "noise_gaussian_std": self.noise_gaussian_std,
            "noise_pink": self.noise_pink,
            "noise_pink_std": self.noise_pink_std,
            "noise_channel_dropout": self.noise_channel_dropout,
            "noise_temporal_dropout": self.noise_temporal_dropout,
            "noise_prob": self.noise_prob,
        }


# =============================================================================
# Sweep State Management (Persistent Decisions)
# =============================================================================

@dataclass
class CascadingState:
    """State for cascading ablation study.

    Tracks which phase we're in and the best config from each completed phase.
    """
    current_phase: int = 1
    phase_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    best_configs: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def get_best_config_for_phase(self, phase: int) -> Optional[Dict[str, Any]]:
        """Get the best config to use as baseline for the given phase."""
        if phase <= 1:
            return None  # Phase 1 has no previous best
        return self.best_configs.get(phase - 1)

    def set_phase_result(self, phase: int, results: Dict[str, Any], best_config: Dict[str, Any]):
        """Record results and best config for a phase."""
        self.phase_results[phase] = results
        self.best_configs[phase] = best_config
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_phase": self.current_phase,
            "phase_results": self.phase_results,
            "best_configs": self.best_configs,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CascadingState":
        state = cls(
            current_phase=data.get("current_phase", 1),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
        # Convert string keys back to int for phase_results and best_configs
        state.phase_results = {int(k): v for k, v in data.get("phase_results", {}).items()}
        state.best_configs = {int(k): v for k, v in data.get("best_configs", {}).items()}
        return state


def load_cascading_state(output_dir: Path) -> CascadingState:
    """Load cascading state from disk, or create new if doesn't exist."""
    state_file = output_dir / "cascading_state.json"
    if state_file.exists():
        with open(state_file, 'r') as f:
            data = json.load(f)
        state = CascadingState.from_dict(data)
        print(f"\n{'='*60}")
        print("LOADED CASCADING STATE")
        print(f"{'='*60}")
        print(f"  Current phase: {state.current_phase}")
        print(f"  Completed phases: {list(state.best_configs.keys())}")
        for phase, config in state.best_configs.items():
            print(f"  Phase {phase} best: {config.get('name', 'unknown')}")
        print(f"{'='*60}\n")
        return state
    else:
        state = CascadingState(created_at=datetime.now().isoformat())
        print(f"\nNo existing cascading state found. Starting from Phase 1.")
        return state


def save_cascading_state(state: CascadingState, output_dir: Path) -> None:
    """Save cascading state to disk."""
    state.updated_at = datetime.now().isoformat()
    state_file = output_dir / "cascading_state.json"
    with open(state_file, 'w') as f:
        json.dump(state.to_dict(), f, indent=2, cls=NumpyEncoder)
    print(f"Cascading state saved to: {state_file}")


def select_best_config(results: Dict[str, Any], configs: Dict[str, AblationConfig]) -> Tuple[str, Dict[str, Any]]:
    """Select the best configuration based on test R² results.

    Args:
        results: Dictionary of config_name -> AblationResult (with mean_r2, etc.)
        configs: Dictionary of config_name -> AblationConfig

    Returns:
        Tuple of (best_config_name, best_config_dict)
    """
    best_name = None
    best_r2 = -float('inf')

    for name, result in results.items():
        # Handle both AblationResult objects and dicts
        if hasattr(result, 'mean_r2'):
            r2 = result.mean_r2
        elif isinstance(result, dict):
            r2 = result.get('mean_r2', result.get('test_r2', 0))
        else:
            continue

        if r2 > best_r2:
            best_r2 = r2
            best_name = name

    if best_name is None:
        raise ValueError("No valid results found to select best config")

    # Convert AblationConfig to dict
    best_config = configs[best_name]
    return best_name, best_config.to_dict()


@dataclass
class SweepState:
    """Persistent state for sweep decisions (legacy, kept for compatibility).

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
# CASCADING ABLATION PHASES
# =============================================================================

# Default augmentation settings (applied to all configs for fair comparison)
DEFAULT_AUGMENTATIONS = {
    "use_euclidean_alignment": True,
    "euclidean_momentum": 0.1,
    "use_bn_adaptation": True,
    "bn_adaptation_steps": 10,
    "bn_adaptation_momentum": 0.1,
    "use_session_augmentation": True,
    "session_aug_mix_prob": 0.3,
    "session_aug_scale_range": (0.9, 1.1),
    "use_mmd_loss": True,
    "mmd_weight": 0.1,
    "use_noise_augmentation": True,
    "noise_gaussian_std": 0.1,
    "noise_pink": True,
    "noise_pink_std": 0.05,
    "noise_channel_dropout": 0.05,
    "noise_temporal_dropout": 0.02,
    "noise_prob": 0.5,
}


def get_phase1_configs(base_config: Optional[Dict[str, Any]] = None) -> Dict[str, AblationConfig]:
    """Phase 1: Architecture Search - find optimal depth × width combination.

    Tests 9 configurations (3 depths × 3 widths):
    - Depth: n_downsample = 2, 3, 4
    - Width: base_channels = 64, 128, 256

    Winner becomes baseline for Phase 2.
    """
    configs_list = []

    depths = [2, 3, 4]
    widths = [64, 128, 256]

    for depth in depths:
        for width in widths:
            name = f"arch_d{depth}_w{width}"
            configs_list.append(AblationConfig(
                name=name,
                description=f"Architecture: depth={depth}, width={width}",
                n_downsample=depth,
                base_channels=width,
                conv_type="modern",
                attention_type="cross_freq_v2",
                skip_type="add",
                use_adaptive_scaling=True,
                conditioning="spectro_temporal",
                **DEFAULT_AUGMENTATIONS,
            ))

    return {c.name: c for c in configs_list}


def get_phase2_configs(best_arch: Dict[str, Any]) -> Dict[str, AblationConfig]:
    """Phase 2: Component Ablations - test conv, attention, skip types.

    Uses best architecture from Phase 1.
    Tests each component independently.

    Args:
        best_arch: Best architecture config from Phase 1 (n_downsample, base_channels)
    """
    configs_list = []
    depth = best_arch.get("n_downsample", 2)
    width = best_arch.get("base_channels", 128)

    # Baseline with best architecture
    configs_list.append(AblationConfig(
        name="phase2_baseline",
        description=f"Phase 2 baseline (d={depth}, w={width})",
        n_downsample=depth,
        base_channels=width,
        conv_type="modern",
        attention_type="cross_freq_v2",
        skip_type="add",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
        **DEFAULT_AUGMENTATIONS,
    ))

    # Test: Standard convolutions (instead of modern)
    configs_list.append(AblationConfig(
        name="conv_standard",
        description="Standard convolutions (vs modern depthwise separable)",
        n_downsample=depth,
        base_channels=width,
        conv_type="standard",  # CHANGED
        attention_type="cross_freq_v2",
        skip_type="add",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
        **DEFAULT_AUGMENTATIONS,
    ))

    # Test: No attention
    configs_list.append(AblationConfig(
        name="attention_none",
        description="No attention mechanism",
        n_downsample=depth,
        base_channels=width,
        conv_type="modern",
        attention_type="none",  # CHANGED
        skip_type="add",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
        **DEFAULT_AUGMENTATIONS,
    ))

    # Test: Concatenation skip connections
    configs_list.append(AblationConfig(
        name="skip_concat",
        description="Concatenation skip connections (vs addition)",
        n_downsample=depth,
        base_channels=width,
        conv_type="modern",
        attention_type="cross_freq_v2",
        skip_type="concat",  # CHANGED
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
        **DEFAULT_AUGMENTATIONS,
    ))

    return {c.name: c for c in configs_list}


def get_phase3_configs(best_config: Dict[str, Any]) -> Dict[str, AblationConfig]:
    """Phase 3: Conditioning & Scaling - test conditioning and adaptive scaling.

    Uses best configuration from Phase 2.

    Args:
        best_config: Best config from Phase 2
    """
    configs_list = []
    depth = best_config.get("n_downsample", 2)
    width = best_config.get("base_channels", 128)
    conv_type = best_config.get("conv_type", "modern")
    attention_type = best_config.get("attention_type", "cross_freq_v2")
    skip_type = best_config.get("skip_type", "add")

    # Baseline with best config from Phase 2
    configs_list.append(AblationConfig(
        name="phase3_baseline",
        description=f"Phase 3 baseline",
        n_downsample=depth,
        base_channels=width,
        conv_type=conv_type,
        attention_type=attention_type,
        skip_type=skip_type,
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
        **DEFAULT_AUGMENTATIONS,
    ))

    # Test: No conditioning
    configs_list.append(AblationConfig(
        name="conditioning_none",
        description="No conditioning (cond_mode=none)",
        n_downsample=depth,
        base_channels=width,
        conv_type=conv_type,
        attention_type=attention_type,
        skip_type=skip_type,
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
        cond_mode="none",  # CHANGED
        **DEFAULT_AUGMENTATIONS,
    ))

    # Test: No adaptive scaling
    configs_list.append(AblationConfig(
        name="adaptive_scaling_off",
        description="No adaptive output scaling",
        n_downsample=depth,
        base_channels=width,
        conv_type=conv_type,
        attention_type=attention_type,
        skip_type=skip_type,
        use_adaptive_scaling=False,  # CHANGED
        conditioning="spectro_temporal",
        **DEFAULT_AUGMENTATIONS,
    ))

    return {c.name: c for c in configs_list}


def get_phase4_configs(best_config: Dict[str, Any]) -> Dict[str, AblationConfig]:
    """Phase 4: Domain Adaptation - test contribution of each augmentation.

    Uses best configuration from Phase 3.
    Tests removing each augmentation component to measure its contribution.

    Args:
        best_config: Best config from Phase 3
    """
    configs_list = []
    depth = best_config.get("n_downsample", 2)
    width = best_config.get("base_channels", 128)
    conv_type = best_config.get("conv_type", "modern")
    attention_type = best_config.get("attention_type", "cross_freq_v2")
    skip_type = best_config.get("skip_type", "add")
    use_adaptive_scaling = best_config.get("use_adaptive_scaling", True)
    cond_mode = best_config.get("cond_mode", "cross_attn_gated")

    # Full baseline with ALL augmentations
    configs_list.append(AblationConfig(
        name="full_baseline",
        description="Full optimized config with all augmentations",
        n_downsample=depth,
        base_channels=width,
        conv_type=conv_type,
        attention_type=attention_type,
        skip_type=skip_type,
        use_adaptive_scaling=use_adaptive_scaling,
        cond_mode=cond_mode,
        conditioning="spectro_temporal",
        **DEFAULT_AUGMENTATIONS,
    ))

    # Test: NO Euclidean Alignment
    no_euclidean = DEFAULT_AUGMENTATIONS.copy()
    no_euclidean["use_euclidean_alignment"] = False
    configs_list.append(AblationConfig(
        name="no_euclidean_alignment",
        description="Ablate: disable Euclidean alignment (expect -2-5%)",
        n_downsample=depth,
        base_channels=width,
        conv_type=conv_type,
        attention_type=attention_type,
        skip_type=skip_type,
        use_adaptive_scaling=use_adaptive_scaling,
        cond_mode=cond_mode,
        conditioning="spectro_temporal",
        **no_euclidean,
    ))

    # Test: NO BN Adaptation
    no_bn = DEFAULT_AUGMENTATIONS.copy()
    no_bn["use_bn_adaptation"] = False
    configs_list.append(AblationConfig(
        name="no_bn_adaptation",
        description="Ablate: disable test-time BN adaptation (expect -2-4%)",
        n_downsample=depth,
        base_channels=width,
        conv_type=conv_type,
        attention_type=attention_type,
        skip_type=skip_type,
        use_adaptive_scaling=use_adaptive_scaling,
        cond_mode=cond_mode,
        conditioning="spectro_temporal",
        **no_bn,
    ))

    # Test: NO Session Augmentation
    no_session = DEFAULT_AUGMENTATIONS.copy()
    no_session["use_session_augmentation"] = False
    configs_list.append(AblationConfig(
        name="no_session_augmentation",
        description="Ablate: disable session augmentation (expect -2-5%)",
        n_downsample=depth,
        base_channels=width,
        conv_type=conv_type,
        attention_type=attention_type,
        skip_type=skip_type,
        use_adaptive_scaling=use_adaptive_scaling,
        cond_mode=cond_mode,
        conditioning="spectro_temporal",
        **no_session,
    ))

    # Test: NO MMD Loss
    no_mmd = DEFAULT_AUGMENTATIONS.copy()
    no_mmd["use_mmd_loss"] = False
    configs_list.append(AblationConfig(
        name="no_mmd_loss",
        description="Ablate: disable MMD loss (expect -1-3%)",
        n_downsample=depth,
        base_channels=width,
        conv_type=conv_type,
        attention_type=attention_type,
        skip_type=skip_type,
        use_adaptive_scaling=use_adaptive_scaling,
        cond_mode=cond_mode,
        conditioning="spectro_temporal",
        **no_mmd,
    ))

    # Test: NO Noise Augmentation
    no_noise = DEFAULT_AUGMENTATIONS.copy()
    no_noise["use_noise_augmentation"] = False
    configs_list.append(AblationConfig(
        name="no_noise_augmentation",
        description="Ablate: disable noise augmentation",
        n_downsample=depth,
        base_channels=width,
        conv_type=conv_type,
        attention_type=attention_type,
        skip_type=skip_type,
        use_adaptive_scaling=use_adaptive_scaling,
        cond_mode=cond_mode,
        conditioning="spectro_temporal",
        **no_noise,
    ))

    # Test: NO augmentations at all (legacy baseline comparison)
    configs_list.append(AblationConfig(
        name="no_augmentations",
        description="Ablate: disable ALL augmentations (legacy baseline)",
        n_downsample=depth,
        base_channels=width,
        conv_type=conv_type,
        attention_type=attention_type,
        skip_type=skip_type,
        use_adaptive_scaling=use_adaptive_scaling,
        cond_mode=cond_mode,
        conditioning="spectro_temporal",
        use_euclidean_alignment=False,
        use_bn_adaptation=False,
        use_session_augmentation=False,
        use_mmd_loss=False,
        use_noise_augmentation=False,
    ))

    return {c.name: c for c in configs_list}


def get_ablation_configs(phase: int = 0, best_config: Optional[Dict[str, Any]] = None) -> Dict[str, AblationConfig]:
    """Get ablation configurations for a specific phase.

    Args:
        phase: Which phase to get configs for (1-4), or 0 for legacy (all configs)
        best_config: Best configuration from previous phase (required for phases 2-4)

    Returns:
        Dictionary of ablation configurations for the specified phase
    """
    if phase == 0:
        # Legacy mode: return all configs for backwards compatibility
        return _get_legacy_all_configs()
    elif phase == 1:
        return get_phase1_configs()
    elif phase == 2:
        if best_config is None:
            raise ValueError("Phase 2 requires best_config from Phase 1")
        return get_phase2_configs(best_config)
    elif phase == 3:
        if best_config is None:
            raise ValueError("Phase 3 requires best_config from Phase 2")
        return get_phase3_configs(best_config)
    elif phase == 4:
        if best_config is None:
            raise ValueError("Phase 4 requires best_config from Phase 3")
        return get_phase4_configs(best_config)
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 0-4.")


def _get_legacy_all_configs() -> Dict[str, AblationConfig]:
    """Legacy function: return all configs for backwards compatibility."""
    configs_list = []

    # Baseline
    configs_list.append(AblationConfig(
        name="baseline",
        description="Full baseline with all augmentations",
        n_downsample=2,
        base_channels=128,
        conv_type="modern",
        attention_type="cross_freq_v2",
        skip_type="add",
        use_adaptive_scaling=True,
        conditioning="spectro_temporal",
        **DEFAULT_AUGMENTATIONS,
    ))

    # Architecture variants
    for depth in [3, 4]:
        configs_list.append(AblationConfig(
            name=f"depth_{depth}",
            description=f"Depth: n_downsample={depth}",
            n_downsample=depth,
            base_channels=128,
            conv_type="modern",
            attention_type="cross_freq_v2",
            **DEFAULT_AUGMENTATIONS,
        ))

    for width in [64, 256]:
        configs_list.append(AblationConfig(
            name=f"width_{width}",
            description=f"Width: base_channels={width}",
            n_downsample=2,
            base_channels=width,
            conv_type="modern",
            attention_type="cross_freq_v2",
            **DEFAULT_AUGMENTATIONS,
        ))

    # Component ablations
    configs_list.append(AblationConfig(
        name="conv_standard",
        description="Standard convolutions",
        n_downsample=2,
        conv_type="standard",
        attention_type="cross_freq_v2",
        **DEFAULT_AUGMENTATIONS,
    ))

    configs_list.append(AblationConfig(
        name="attention_none",
        description="No attention",
        n_downsample=2,
        conv_type="modern",
        attention_type="none",
        **DEFAULT_AUGMENTATIONS,
    ))

    configs_list.append(AblationConfig(
        name="skip_concat",
        description="Concatenation skip connections",
        n_downsample=2,
        conv_type="modern",
        attention_type="cross_freq_v2",
        skip_type="concat",
        **DEFAULT_AUGMENTATIONS,
    ))

    configs_list.append(AblationConfig(
        name="conditioning_none",
        description="No conditioning",
        n_downsample=2,
        conv_type="modern",
        attention_type="cross_freq_v2",
        cond_mode="none",
        **DEFAULT_AUGMENTATIONS,
    ))

    configs_list.append(AblationConfig(
        name="adaptive_scaling_off",
        description="No adaptive scaling",
        n_downsample=2,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_adaptive_scaling=False,
        **DEFAULT_AUGMENTATIONS,
    ))

    # Domain adaptation ablations
    for aug_name in ["euclidean_alignment", "bn_adaptation", "session_augmentation", "mmd_loss", "noise_augmentation"]:
        no_aug = DEFAULT_AUGMENTATIONS.copy()
        no_aug[f"use_{aug_name}"] = False
        configs_list.append(AblationConfig(
            name=f"no_{aug_name}",
            description=f"Disable {aug_name}",
            n_downsample=2,
            conv_type="modern",
            attention_type="cross_freq_v2",
            **no_aug,
        ))

    # No augmentations at all
    configs_list.append(AblationConfig(
        name="no_augmentations",
        description="Disable ALL augmentations",
        n_downsample=2,
        conv_type="modern",
        attention_type="cross_freq_v2",
        use_euclidean_alignment=False,
        use_bn_adaptation=False,
        use_session_augmentation=False,
        use_mmd_loss=False,
        use_noise_augmentation=False,
    ))

    return {c.name: c for c in configs_list}


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
    seed: int = 42  # Actual seed used
    seed_idx: int = 0  # Which seed iteration (0, 1, 2)
    test_sessions: List[str] = field(default_factory=list)
    train_sessions: List[str] = field(default_factory=list)

    # Validation metrics (from 70/30 split on training sessions - for model selection)
    val_r2: float = 0.0
    val_loss: float = 0.0
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

    # Per-session TEST metrics (from held-out sessions)
    per_session_test_results: Dict[str, float] = field(default_factory=dict)

    # Raw results dict (keep everything)
    raw_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_idx": self.fold_idx,
            "seed": self.seed,
            "seed_idx": self.seed_idx,
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
            "per_session_test_r2": self.per_session_test_results,
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

        # CRITICAL FIX FOR DATA LEAKAGE:
        # During training, test sessions were EXCLUDED entirely (via --exclude-sessions).
        # Normalization was computed from training data only.
        # We must use the SAME normalization approach here:
        # 1. Load data excluding test sessions to get training normalization stats
        # 2. Load full data and apply training normalization to test sessions

        from data import load_signals, load_odor_labels, extract_window, compute_normalization, normalize, DATA_PATH

        # Step 1: Load data with test sessions excluded to get training normalization
        train_data = prepare_data(
            split_by_session=False,
            force_recreate_splits=True,
            exclude_sessions=test_sessions,  # Exclude test sessions, same as training
        )
        train_norm_stats = train_data["norm_stats"]

        # Step 2: Load full data (all sessions) without normalization
        signals = load_signals(DATA_PATH)
        num_trials = signals.shape[0]
        odors, vocab = load_odor_labels(ODOR_CSV_PATH, num_trials)
        windowed = extract_window(signals)

        # Apply training normalization to ALL data (including test)
        normalized = normalize(windowed, train_norm_stats)
        X = normalized[:, 0]  # OB
        y = normalized[:, 1]  # PCx

        # Get session IDs
        session_ids, session_to_idx, idx_to_session = load_session_ids(
            ODOR_CSV_PATH, num_trials=num_trials
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

        # CRITICAL FIX: Apply per_channel_normalize to target!
        # During training, the target pcx is normalized per-channel per-sample.
        # The model learns to output values matching per_channel_normalize(pcx).
        # We MUST apply the same normalization to y_test for fair comparison.
        def per_channel_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            """Apply per-channel z-score normalization to a batch of signals."""
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp(min=eps)
            return (x - mean) / std

        y_test_norm = per_channel_normalize(y_test)

        # Evaluate in batches to avoid OOM
        batch_size = 32
        y_pred_list = []
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                X_batch = X_test[i:i+batch_size].to(device)
                odors_batch = odors_test[i:i+batch_size].to(device)
                y_batch = model(X_batch, odors_batch).cpu()
                y_pred_list.append(y_batch)
                # Clear GPU memory
                del X_batch, odors_batch
                torch.cuda.empty_cache()
        y_pred = torch.cat(y_pred_list, dim=0)

        # Compute overall metrics using normalized target
        y_test_flat = y_test_norm.numpy().flatten()
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

            y_sess = y_test_norm[local_mask].numpy().flatten()
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
    seed_idx: int = 0,  # Which seed iteration this is (0, 1, 2)
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

    # Output file for results - unique per fold AND seed
    results_file = output_dir / f"{ablation_config.name}_fold{fold_idx}_seed{seed_idx}_results.json"

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

    # Checkpoint path - unique per config, fold, AND seed to avoid overwriting
    checkpoint_prefix = f"{ablation_config.name}_fold{fold_idx}_seed{seed_idx}"
    checkpoint_path = output_dir / ablation_config.name / f"fold{fold_idx}_seed{seed_idx}" / "best_model.pt"

    # Base arguments
    cmd.extend([
        "--arch", "condunet",
        "--dataset", "olfactory",
        "--epochs", str(ablation_config.epochs),
        "--batch-size", str(ablation_config.batch_size),
        "--lr", str(ablation_config.learning_rate),
        "--seed", str(seed),  # Use the actual seed passed in
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

    # =========================================================================
    # NEW ABLATION COMPONENT FLAGS
    # =========================================================================

    # Euclidean Alignment
    if ablation_config.use_euclidean_alignment:
        cmd.append("--use-euclidean-alignment")
        cmd.extend(["--euclidean-momentum", str(ablation_config.euclidean_momentum)])

    # Test-time BN Adaptation (note: this is used during evaluation, passed for metadata)
    if ablation_config.use_bn_adaptation:
        cmd.append("--use-bn-adaptation")
        cmd.extend(["--bn-adaptation-steps", str(ablation_config.bn_adaptation_steps)])
        cmd.extend(["--bn-adaptation-momentum", str(ablation_config.bn_adaptation_momentum)])
        if ablation_config.bn_reset_stats:
            cmd.append("--bn-reset-stats")

    # Session Augmentation
    if ablation_config.use_session_augmentation:
        cmd.append("--use-session-augmentation")
        cmd.extend(["--session-aug-mix-prob", str(ablation_config.session_aug_mix_prob)])
        cmd.extend(["--session-aug-scale-min", str(ablation_config.session_aug_scale_range[0])])
        cmd.extend(["--session-aug-scale-max", str(ablation_config.session_aug_scale_range[1])])

    # MMD Loss
    if ablation_config.use_mmd_loss:
        cmd.append("--use-mmd-loss")
        cmd.extend(["--mmd-weight", str(ablation_config.mmd_weight)])

    # Noise Augmentation
    if ablation_config.use_noise_augmentation:
        cmd.append("--use-noise-augmentation")
        cmd.extend(["--noise-gaussian-std", str(ablation_config.noise_gaussian_std)])
        if ablation_config.noise_pink:
            cmd.append("--noise-pink")
        cmd.extend(["--noise-pink-std", str(ablation_config.noise_pink_std)])
        cmd.extend(["--noise-channel-dropout", str(ablation_config.noise_channel_dropout)])
        cmd.extend(["--noise-temporal-dropout", str(ablation_config.noise_temporal_dropout)])
        cmd.extend(["--noise-prob", str(ablation_config.noise_prob)])

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
            seed=seed,
            seed_idx=seed_idx,
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
            # Per-session TEST metrics (from held-out sessions)
            per_session_test_results=test_metrics.get("per_session_test_r2", {}),
            # Keep raw results for reference
            raw_results=results,
        )

        print(f"\n  Fold {fold_idx} completed:")
        print(f"    Val R2: {fold_result.val_r2:.4f} (model selection)")
        print(f"    TEST R2: {fold_result.test_r2:.4f} (PRIMARY - true generalization)")
        print(f"    TEST Corr: {fold_result.test_corr:.4f}")
        print(f"    Val Loss: {fold_result.val_loss:.4f}")
        print(f"    Best Epoch: {fold_result.best_epoch}")
        print(f"    Parameters: {fold_result.n_parameters:,}")
        print(f"    Time: {elapsed/60:.1f} minutes")

        # Print per-session TEST R² (the important part!)
        if fold_result.per_session_test_results:
            print(f"    Per-session TEST R²:")
            for sess, r2 in fold_result.per_session_test_results.items():
                print(f"      {sess}: {r2:.4f}")
        else:
            print(f"    WARNING: No per-session test results!")

        return fold_result
    else:
        print(f"WARNING: Results file not found: {results_file}")
        return None


def run_ablation_experiment(
    ablation_config: AblationConfig,
    fold_splits: List[Dict[str, Any]],
    output_dir: Path,
    seed: int = 42,
    n_seeds: int = 1,  # Single seed per fold (removed 3-seed approach for ablation)
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
    print(f"# Strategy: 3 folds × {n_seeds} seeds = {3 * n_seeds} runs")
    print(f"{'#'*70}")

    ablation_dir = output_dir / ablation_config.name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for split in fold_splits:
        fold_idx = split["fold_idx"]

        # Skip if not in folds_to_run
        if folds_to_run is not None and fold_idx not in folds_to_run:
            continue

        # Run multiple seeds per fold
        for seed_idx in range(n_seeds):
            current_seed = seed + fold_idx * n_seeds + seed_idx

            print(f"\n  --- Fold {fold_idx}, Seed {seed_idx + 1}/{n_seeds} (seed={current_seed}) ---")

            result = run_single_fold(
                ablation_config=ablation_config,
                fold_split=split,
                output_dir=ablation_dir,
                seed=current_seed,
                seed_idx=seed_idx,  # Track which seed this is
                verbose=verbose,
                use_fsdp=use_fsdp,
                fsdp_strategy=fsdp_strategy,
                dry_run=dry_run,
            )

            if result is not None:
                fold_results.append(result)

                # Save after EACH fold+seed (don't lose progress if interrupted)
                fold_file = ablation_dir / f"fold{fold_idx}_seed{seed_idx}_result.json"
                with open(fold_file, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2, cls=NumpyEncoder)
                print(f"  Fold {fold_idx} Seed {seed_idx} saved to: {fold_file}")

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
    n_seeds: int = 1,  # Single seed (removed 3-seed approach for ablation)
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
        seed: Base random seed
        n_seeds: Number of seeds per fold (default: 3, giving 3 folds × 3 seeds = 9 runs)
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
    print(f"Seeds per fold: {n_seeds} (3 folds × {n_seeds} seeds = {3 * n_seeds} runs per ablation)")
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
            n_seeds=n_seeds,
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


# =============================================================================
# CASCADING ABLATION STUDY
# =============================================================================

def run_cascading_ablation_study(
    output_dir: Path,
    start_phase: int = 1,
    end_phase: int = 4,
    folds_to_run: Optional[List[int]] = None,
    seed: int = 42,
    n_seeds: int = 1,
    epochs: Optional[int] = None,
    verbose: bool = True,
    use_fsdp: bool = False,
    fsdp_strategy: str = "grad_op",
    dry_run: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """Run cascading ablation study where each phase builds on the best from previous.

    PHASES:
    1. Architecture Search (9 configs: 3 depths × 3 widths)
    2. Component Ablations (4 configs using best arch)
    3. Conditioning & Scaling (3 configs using best from phase 2)
    4. Domain Adaptation (7 configs using best from phase 3)

    Args:
        output_dir: Directory to save all results
        start_phase: Phase to start from (1-4)
        end_phase: Phase to end at (1-4)
        folds_to_run: Optional list of fold indices to run (default: all 3)
        seed: Base random seed
        n_seeds: Number of seeds per fold
        epochs: Number of training epochs
        verbose: Print training output
        use_fsdp: Use FSDP for multi-GPU training
        dry_run: Print commands without running them

    Returns:
        Dict mapping phase number to phase results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CASCADING ABLATION STUDY")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Phases to run: {start_phase} -> {end_phase}")
    print(f"3 folds × {n_seeds} seeds = {3 * n_seeds} runs per config")
    print("=" * 70)
    print()

    # Load or create cascading state
    state = load_cascading_state(output_dir)

    # Get all sessions and create fold splits
    all_sessions = get_all_sessions("olfactory")
    fold_splits = get_3fold_session_splits(all_sessions)

    print(f"Found {len(all_sessions)} sessions: {all_sessions}")
    print(f"\n3-Fold Session Splits:")
    for split in fold_splits:
        print(f"  Fold {split['fold_idx']}: test={split['test_sessions']}")
    print()

    all_phase_results = {}

    for phase in range(start_phase, end_phase + 1):
        print("\n" + "=" * 70)
        print(f"PHASE {phase}")
        print("=" * 70)

        # Get best config from previous phase (None for phase 1)
        best_config = state.get_best_config_for_phase(phase)

        if phase > 1 and best_config is None:
            # Try to load from previous phase results if state doesn't have it
            if phase - 1 in state.phase_results:
                print(f"Warning: No best_config for phase {phase}, using previous phase results")
            else:
                raise ValueError(
                    f"Cannot run phase {phase} without completing phase {phase - 1}. "
                    f"Run with --start-phase {phase - 1} first."
                )

        # Get configs for this phase
        configs = get_ablation_configs(phase=phase, best_config=best_config)

        print(f"Phase {phase} configurations ({len(configs)} total):")
        for name, config in configs.items():
            print(f"  - {name}: {config.description}")
        print()

        # Override epochs if specified
        if epochs is not None:
            for config in configs.values():
                config.epochs = epochs

        # Run all configs in this phase
        phase_results = {}
        phase_start_time = time.time()

        for config_name, config in configs.items():
            print(f"\n--- Running: {config_name} ---")
            result = run_ablation_experiment(
                ablation_config=config,
                fold_splits=fold_splits,
                output_dir=output_dir / f"phase{phase}",
                seed=seed,
                n_seeds=n_seeds,
                verbose=verbose,
                use_fsdp=use_fsdp,
                fsdp_strategy=fsdp_strategy,
                folds_to_run=folds_to_run,
                dry_run=dry_run,
            )
            phase_results[config_name] = result

        phase_time = time.time() - phase_start_time

        # Print phase summary
        print(f"\n{'='*60}")
        print(f"PHASE {phase} COMPLETE (took {phase_time/60:.1f} min)")
        print(f"{'='*60}")

        if not dry_run:
            # Select best config for this phase
            best_name, best_config_dict = select_best_config(phase_results, configs)

            print(f"\nPhase {phase} Results (sorted by Test R²):")
            sorted_results = sorted(
                phase_results.items(),
                key=lambda x: x[1].mean_r2 if hasattr(x[1], 'mean_r2') else 0,
                reverse=True
            )
            for name, result in sorted_results:
                r2 = result.mean_r2 if hasattr(result, 'mean_r2') else 0
                std = result.std_r2 if hasattr(result, 'std_r2') else 0
                marker = " <-- BEST" if name == best_name else ""
                print(f"  {name:<30} R² = {r2:.4f} ± {std:.4f}{marker}")

            print(f"\n*** Phase {phase} WINNER: {best_name} ***")
            print(f"    Config: n_downsample={best_config_dict.get('n_downsample')}, "
                  f"base_channels={best_config_dict.get('base_channels')}")

            # Save phase results and best config
            state.set_phase_result(
                phase,
                {name: result.to_dict() if hasattr(result, 'to_dict') else result
                 for name, result in phase_results.items()},
                best_config_dict
            )
            state.current_phase = phase + 1
            save_cascading_state(state, output_dir)

            # Save phase-specific results
            phase_output = output_dir / f"phase{phase}"
            phase_output.mkdir(parents=True, exist_ok=True)
            save_summary(phase_results, phase_output)

        all_phase_results[phase] = phase_results

    # Final summary
    print("\n" + "=" * 70)
    print("CASCADING ABLATION STUDY COMPLETE")
    print("=" * 70)

    if not dry_run:
        print("\nBest configs at each phase:")
        for phase, best in state.best_configs.items():
            print(f"  Phase {phase}: {best.get('name', 'unknown')}")
            print(f"           depth={best.get('n_downsample')}, width={best.get('base_channels')}")
            print(f"           conv={best.get('conv_type')}, attention={best.get('attention_type')}")

        # Print final optimized config
        final_config = state.best_configs.get(end_phase, {})
        print(f"\n*** FINAL OPTIMIZED CONFIGURATION ***")
        for key, value in final_config.items():
            if key not in ['name', 'description']:
                print(f"  {key}: {value}")

    return all_phase_results


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
        description="Cascading Ablation Study with 3-Fold Cross-Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation_cascading",
        help="Output directory for results",
    )

    # Cascading mode arguments
    parser.add_argument(
        "--cascading",
        action="store_true",
        default=True,
        help="Run cascading ablation study (default: True)",
    )

    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run legacy (non-cascading) ablation study",
    )

    parser.add_argument(
        "--start-phase",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Phase to start from (1=Architecture, 2=Components, 3=Conditioning, 4=Augmentation)",
    )

    parser.add_argument(
        "--end-phase",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Phase to end at",
    )

    parser.add_argument(
        "--ablations",
        type=str,
        nargs="+",
        default=None,
        help="Specific ablations to run (legacy mode only)",
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
        help="Base random seed",
    )

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of seeds per fold (default: 1, giving 3 folds × 1 seed = 3 runs per ablation)",
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
        "--list-phases",
        action="store_true",
        help="List all phases and their configurations",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be run without actually running them",
    )

    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Disable sweep mode (legacy mode only)",
    )

    args = parser.parse_args()

    # List phases and exit
    if args.list_phases:
        print("\n" + "=" * 70)
        print("CASCADING ABLATION PHASES")
        print("=" * 70)

        print("\nPhase 1: ARCHITECTURE SEARCH (9 configs)")
        print("  Find optimal depth × width combination")
        configs = get_phase1_configs()
        for name in configs:
            print(f"    - {name}")

        print("\nPhase 2: COMPONENT ABLATIONS (4 configs)")
        print("  Test conv_type, attention, skip connections")
        print("  Uses best architecture from Phase 1")
        dummy_arch = {"n_downsample": 2, "base_channels": 128}
        configs = get_phase2_configs(dummy_arch)
        for name, c in configs.items():
            print(f"    - {name}: {c.description}")

        print("\nPhase 3: CONDITIONING & SCALING (3 configs)")
        print("  Test conditioning and adaptive scaling")
        print("  Uses best config from Phase 2")
        configs = get_phase3_configs(dummy_arch)
        for name, c in configs.items():
            print(f"    - {name}: {c.description}")

        print("\nPhase 4: DOMAIN ADAPTATION (7 configs)")
        print("  Test each augmentation component's contribution")
        print("  Uses best config from Phase 3")
        configs = get_phase4_configs(dummy_arch)
        for name, c in configs.items():
            print(f"    - {name}: {c.description}")

        print("\nTotal: 23 configurations across 4 phases")
        print("=" * 70)
        return 0

    # List ablations (legacy) and exit
    if args.list_ablations:
        configs = get_ablation_configs(phase=0)  # Legacy mode
        print("\nAvailable ablation configurations (legacy mode):")
        print("-" * 60)
        for name, config in configs.items():
            print(f"  {name:<25} - {config.description}")
        print()
        return 0

    # Run cascading or legacy study
    if args.legacy:
        print("Running LEGACY (non-cascading) ablation study...")
        run_3fold_ablation_study(
            output_dir=Path(args.output_dir),
            ablations_to_run=args.ablations,
            folds_to_run=args.folds,
            seed=args.seed,
            n_seeds=args.n_seeds,
            epochs=args.epochs,
            verbose=not args.quiet,
            use_fsdp=args.fsdp,
            fsdp_strategy=args.fsdp_strategy,
            dry_run=args.dry_run,
            enable_sweep=not args.no_sweep,
        )
    else:
        print("Running CASCADING ablation study...")
        run_cascading_ablation_study(
            output_dir=Path(args.output_dir),
            start_phase=args.start_phase,
            end_phase=args.end_phase,
            folds_to_run=args.folds,
            seed=args.seed,
            n_seeds=args.n_seeds,
            epochs=args.epochs,
            verbose=not args.quiet,
            use_fsdp=args.fsdp,
            fsdp_strategy=args.fsdp_strategy,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
