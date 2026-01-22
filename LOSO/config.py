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

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result


@dataclass
class LOSOFoldResult:
    """Result from a single LOSO fold."""
    fold_idx: int
    test_session: str
    train_sessions: List[str]
    val_r2: float
    val_loss: float
    train_loss: float = 0.0
    per_session_r2: Dict[str, float] = field(default_factory=dict)
    per_session_loss: Dict[str, float] = field(default_factory=dict)
    epochs_trained: int = 0
    total_time: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)

    # Legacy aliases
    @property
    def session(self) -> str:
        return self.test_session

    @property
    def best_val_r2(self) -> float:
        return self.val_r2

    @property
    def best_val_mae(self) -> float:
        return self.val_loss

    @property
    def best_epoch(self) -> int:
        return self.epochs_trained

    @property
    def train_time(self) -> float:
        return self.total_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "fold_idx": self.fold_idx,
            "test_session": self.test_session,
            "train_sessions": self.train_sessions,
            "val_r2": self.val_r2,
            "val_loss": self.val_loss,
            "train_loss": self.train_loss,
            "per_session_r2": self.per_session_r2,
            "per_session_loss": self.per_session_loss,
            "epochs_trained": self.epochs_trained,
            "total_time": self.total_time,
            "config": self.config,
        }


@dataclass
class LOSOResult:
    """Aggregated LOSO results."""
    config: LOSOConfig
    fold_results: List[LOSOFoldResult]
    all_sessions: List[str]

    # Statistics (computed after all folds)
    mean_r2: float = 0.0
    std_r2: float = 0.0
    mean_loss: float = 0.0
    std_loss: float = 0.0
    fold_r2s: List[float] = field(default_factory=list)

    # Legacy aliases
    @property
    def mean_mae(self) -> float:
        return self.mean_loss

    @property
    def std_mae(self) -> float:
        return self.std_loss

    @property
    def sessions(self) -> List[str]:
        return self.all_sessions

    def compute_statistics(self) -> None:
        """Compute aggregate statistics from fold results."""
        if not self.fold_results:
            return

        r2_values = [r.val_r2 for r in self.fold_results]
        loss_values = [r.val_loss for r in self.fold_results]

        self.fold_r2s = r2_values
        self.mean_r2 = float(np.mean(r2_values))
        self.std_r2 = float(np.std(r2_values))
        self.mean_loss = float(np.mean(loss_values))
        self.std_loss = float(np.std(loss_values))

    def print_summary(self) -> None:
        """Print summary of LOSO results."""
        print("\n" + "=" * 70)
        print("LOSO CROSS-VALIDATION RESULTS")
        print("=" * 70)

        print(f"\n{'Fold':<6} {'Test Session':<20} {'R²':>10} {'Loss':>10}")
        print("-" * 50)

        for r in self.fold_results:
            print(f"{r.fold_idx:<6} {r.test_session:<20} {r.val_r2:>10.4f} {r.val_loss:>10.4f}")

        print("-" * 50)
        print(f"{'Mean':<6} {'':<20} {self.mean_r2:>10.4f} {self.mean_loss:>10.4f}")
        print(f"{'Std':<6} {'':<20} {self.std_r2:>10.4f} {self.std_loss:>10.4f}")
        print()
        print(f"Final: R² = {self.mean_r2:.4f} ± {self.std_r2:.4f}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "fold_results": [r.to_dict() for r in self.fold_results],
            "all_sessions": self.all_sessions,
            "mean_r2": self.mean_r2,
            "std_r2": self.std_r2,
            "mean_loss": self.mean_loss,
            "std_loss": self.std_loss,
            "fold_r2s": self.fold_r2s,
        }
