"""
LOSO (Leave-One-Session-Out) Cross-Validation Configuration
============================================================

Methodology: Final evaluation with pre-specified hyperparameters.
This is acceptable for CV without separate test set because:
1. All hyperparameters pre-specified from ablation study (no tuning here)
2. Each fold uses one session as validation, rest as training
3. Reports mean ± std across all sessions for unbiased generalization estimate

Pipeline:
1. Ablation (phase_three) -> select best components with fixed hyperparams
2. LOSO (this) -> final evaluation with winning config, no further tuning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LOSOConfig:
    """Configuration for LOSO cross-validation.

    All hyperparameters should be pre-specified from ablation study.
    No tuning should be done based on LOSO results.
    """

    # Dataset
    dataset: str = "olfactory"
    output_dir: Path = field(default_factory=lambda: Path("results/loso"))

    # Training (pre-specified from ablation)
    epochs: int = 80  # Fixed, no early stopping
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42

    # Model (pre-specified from ablation)
    arch: str = "condunet"
    base_channels: int = 128
    n_downsample: int = 2
    attention_type: str = "none"
    cond_mode: str = "cross_attn_gated"
    conv_type: str = "modern"
    activation: str = "relu"
    skip_type: str = "add"
    n_heads: int = 4
    conditioning: str = "spectro_temporal"

    # Optimizer (pre-specified)
    optimizer: str = "adamw"
    lr_schedule: str = "step"
    weight_decay: float = 0.01
    dropout: float = 0.0

    # Session adaptation (pre-specified from ablation)
    use_session_stats: bool = False
    use_adaptive_scaling: bool = True
    session_use_spectral: bool = False
    use_bidirectional: bool = False

    # FSDP
    use_fsdp: bool = False
    fsdp_strategy: str = "full"

    # Execution
    resume: bool = True
    verbose: bool = True
    save_models: bool = True
    generate_plots: bool = False

    # Folds to run (None = all)
    folds_to_run: Optional[List[int]] = None

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
