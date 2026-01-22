"""
Ablation Study Configuration
============================

Proves each architectural component contributes to performance.
Uses 3 random seeds for robust mean ± std results.

Total: 6 experiments × 3 seeds = 18 runs
  1. baseline (full model)
  2. conv_type_standard (remove modern conv)
  3. conditioning_none (remove conditioning)
  4. adaptive_scaling_off (remove FiLM scaling)
  5. depth_medium (3 levels)
  6. depth_deep (4 levels)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Baseline Configuration (full model)
# =============================================================================

BASELINE_CONFIG: Dict[str, Any] = {
    "arch": "condunet",
    "conv_type": "modern",
    "base_channels": 128,
    "n_downsample": 2,
    "attention_type": "none",
    "n_heads": 4,
    "skip_type": "add",
    "activation": "relu",
    "cond_mode": "cross_attn_gated",
    "conditioning": "spectro_temporal",
    "use_adaptive_scaling": True,
    "use_session_stats": False,
    "use_bidirectional": False,
    "optimizer": "adamw",
    "lr_schedule": "step",
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "batch_size": 64,
    "epochs": 100,
    "dropout": 0.0,
    "dataset": "olfactory",
}

# =============================================================================
# Ablation Groups
# =============================================================================

ABLATION_GROUPS: List[Dict[str, Any]] = [
    {
        "name": "conv_type",
        "description": "Remove modern convolutions",
        "parameter": "conv_type",
        "variant": {"value": "standard", "name": "standard", "desc": "Standard Conv1d"},
    },
    {
        "name": "conditioning",
        "description": "Remove spectro-temporal conditioning",
        "parameter": "cond_mode",
        "variant": {"value": "none", "name": "none", "desc": "No conditioning"},
    },
    {
        "name": "adaptive_scaling",
        "description": "Remove session-adaptive scaling",
        "parameter": "use_adaptive_scaling",
        "variant": {"value": False, "name": "off", "desc": "No FiLM scaling"},
    },
    {
        "name": "depth",
        "description": "Test different network depths",
        "parameter": "n_downsample",
        "variants": [
            {"value": 3, "name": "medium", "desc": "3 levels"},
            {"value": 4, "name": "deep", "desc": "4 levels"},
        ],
    },
]


@dataclass
class AblationConfig:
    """Configuration for ablation study."""

    output_dir: Path = Path("artifacts/ablation")
    n_val_sessions: int = 3
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    seeds: List[int] = None  # Default: [42, 123, 456]
    use_fsdp: bool = False
    n_gpus: int = 8
    verbose: bool = True
    groups_to_run: Optional[List[str]] = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.seeds is None:
            self.seeds = [42, 123, 456]
