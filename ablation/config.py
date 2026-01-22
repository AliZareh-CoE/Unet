"""
Ablation Study Configuration
============================

Clean, focused ablation to prove each architectural component contributes.
Uses held-out session validation for fast comparison.

5 Core Ablation Groups (~11 runs total):
1. Convolution Type: standard vs modern
2. Conditioning Mode: none vs cross_attn_gated
3. Adaptive Scaling: off vs on
4. Bidirectional Training: off vs on
5. Network Depth: 2 vs 3 vs 4
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Baseline Configuration (current best setup)
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
# Ablation Groups
# =============================================================================

ABLATION_GROUPS: List[Dict[str, Any]] = [
    {
        "name": "conv_type",
        "description": "Convolution architecture",
        "parameter": "conv_type",
        "variants": [
            {"value": "standard", "name": "standard", "desc": "Standard Conv1d"},
            {"value": "modern", "name": "modern", "desc": "Dilated depthwise-sep + SE (ours)"},
        ],
    },
    {
        "name": "conditioning",
        "description": "Conditioning mechanism",
        "parameter": "cond_mode",
        "variants": [
            {"value": "none", "name": "none", "desc": "No conditioning"},
            {"value": "cross_attn_gated", "name": "gated", "desc": "Gated cross-attention (ours)"},
        ],
    },
    {
        "name": "adaptive_scaling",
        "description": "Session-adaptive output scaling",
        "parameter": "use_adaptive_scaling",
        "variants": [
            {"value": False, "name": "off", "desc": "No session adaptation"},
            {"value": True, "name": "on", "desc": "FiLM-style scaling (ours)"},
        ],
    },
    {
        "name": "bidirectional",
        "description": "Bidirectional training with cycle loss",
        "parameter": "use_bidirectional",
        "variants": [
            {"value": False, "name": "unidirectional", "desc": "Forward only"},
            {"value": True, "name": "bidirectional", "desc": "With cycle consistency (ours)"},
        ],
    },
    {
        "name": "depth",
        "description": "Network depth (downsample levels)",
        "parameter": "n_downsample",
        "variants": [
            {"value": 2, "name": "shallow", "desc": "2 levels (ours)"},
            {"value": 3, "name": "medium", "desc": "3 levels"},
            {"value": 4, "name": "deep", "desc": "4 levels"},
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
    verbose: bool = True
    groups_to_run: Optional[List[str]] = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
