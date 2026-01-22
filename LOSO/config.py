"""
Nested Cross-Validation for LOSO
================================

Gold standard methodology for limited subjects/sessions:

OUTER LOOP: LOSO (Leave-One-Session-Out)
  - Final unbiased evaluation
  - Each session held out once as test set

INNER LOOP: k-fold CV on remaining sessions
  - Component/hyperparameter selection
  - Runs ablation within each outer fold
  - Winner used to train final model for that fold

This avoids ANY data leakage because:
1. Test session never seen during inner loop selection
2. Different sessions may select different optimal configs (that's ok!)
3. Final metric is mean ± std across outer folds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LOSOConfig:
    """Configuration for nested LOSO cross-validation."""

    # Dataset
    dataset: str = "olfactory"
    output_dir: Path = field(default_factory=lambda: Path("results/loso"))

    # Training (fixed across all experiments)
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42

    # Model defaults (may be overridden by inner CV winner)
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

    # Optimizer (fixed)
    optimizer: str = "adamw"
    lr_schedule: str = "step"
    weight_decay: float = 0.01
    dropout: float = 0.0

    # Session adaptation
    use_session_stats: bool = False
    use_adaptive_scaling: bool = True
    session_use_spectral: bool = False
    use_bidirectional: bool = False

    # FSDP
    use_fsdp: bool = False
    fsdp_strategy: str = "full"

    # Nested CV settings
    inner_cv_folds: int = 3  # k-fold CV for inner loop
    run_inner_cv: bool = False  # Set True to run nested CV with ablation

    # Execution
    resume: bool = True
    verbose: bool = True
    save_models: bool = True
    generate_plots: bool = False
    folds_to_run: Optional[List[int]] = None

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class LOSOFoldResult:
    """Result from a single LOSO fold."""
    fold_idx: int
    session: str
    best_val_r2: float
    best_val_mae: float
    best_epoch: int
    train_time: float = 0.0


@dataclass
class LOSOResult:
    """Aggregated LOSO results."""
    fold_results: List[LOSOFoldResult]
    mean_r2: float
    std_r2: float
    mean_mae: float
    std_mae: float
    fold_r2s: List[float]
    sessions: List[str]
