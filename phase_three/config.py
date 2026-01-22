"""
Ablation Study Configuration
============================

Component selection with held-out test sessions.

Setup:
- 3 sessions held out as TEST (never touched)
- Remaining 6 sessions: 70% train, 30% val
- 3 random seeds for variance estimation
- 6 variants × 3 seeds = 18 runs total
"""

from dataclasses import dataclass, field
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
        "parameters": {"cond_mode": "none", "conditioning": "none"},  # Disable both
        "variant": {"name": "none", "desc": "No conditioning"},
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
    """Configuration for ablation with held-out test sessions."""

    output_dir: Path = Path("results/ablation")
    dataset: str = "olfactory"

    # Data split
    n_test_sessions: int = 3  # Held out, never touched
    val_ratio: float = 0.3  # 30% of remaining for validation

    # Seeds for variance estimation
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # Training
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 1e-3

    # Execution
    use_fsdp: bool = False
    n_gpus: int = 8
    verbose: bool = True
    groups_to_run: Optional[List[str]] = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
