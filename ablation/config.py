"""
Ablation Study Configuration
============================

Clean, focused ablation to prove each architectural component contributes.
Uses held-out session validation for fast comparison.

Runs baseline once, then tests removing each component.
Total: 8 runs (1 baseline + 5 ablations + 2 depth variants)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Baseline Configuration (full model - our best setup)
# =============================================================================

BASELINE_CONFIG: Dict[str, Any] = {
    # Architecture
    "arch": "condunet",
    "conv_type": "modern",
    "base_channels": 128,
    "n_downsample": 2,
    "attention_type": "none",
    "n_heads": 4,
    "skip_type": "add",
    "activation": "relu",

    # Conditioning
    "cond_mode": "cross_attn_gated",
    "conditioning": "spectro_temporal",

    # Session adaptation
    "use_adaptive_scaling": True,
    "use_session_stats": False,

    # Training
    "use_bidirectional": True,
    "optimizer": "adamw",
    "lr_schedule": "step",
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "batch_size": 64,
    "epochs": 100,
    "dropout": 0.0,

    # Data
    "dataset": "olfactory",
}

# =============================================================================
# Ablation Groups - only non-baseline variants
# =============================================================================

ABLATION_GROUPS: List[Dict[str, Any]] = [
    {
        "name": "conv_type",
        "description": "Remove modern convolutions",
        "parameter": "conv_type",
        "baseline_value": "modern",
        "variant": {"value": "standard", "name": "standard", "desc": "Standard Conv1d (no SE, no dilation)"},
    },
    {
        "name": "conditioning",
        "description": "Remove spectro-temporal conditioning",
        "parameter": "cond_mode",
        "baseline_value": "cross_attn_gated",
        "variant": {"value": "none", "name": "none", "desc": "No conditioning"},
    },
    {
        "name": "adaptive_scaling",
        "description": "Remove session-adaptive scaling",
        "parameter": "use_adaptive_scaling",
        "baseline_value": True,
        "variant": {"value": False, "name": "off", "desc": "No session adaptation"},
    },
    {
        "name": "bidirectional",
        "description": "Remove bidirectional training",
        "parameter": "use_bidirectional",
        "baseline_value": True,
        "variant": {"value": False, "name": "unidirectional", "desc": "Forward only (no cycle loss)"},
    },
    {
        "name": "depth",
        "description": "Test different network depths",
        "parameter": "n_downsample",
        "baseline_value": 2,
        "variants": [
            {"value": 3, "name": "medium", "desc": "3 downsample levels"},
            {"value": 4, "name": "deep", "desc": "4 downsample levels"},
        ],
    },
]


@dataclass
class AblationConfig:
    """Configuration for ablation study."""

    output_dir: Path = Path("artifacts/ablation")

    # Validation
    n_val_sessions: int = 3

    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42

    # Execution
    use_fsdp: bool = False
    n_gpus: int = 8
    verbose: bool = True
    groups_to_run: Optional[List[str]] = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
