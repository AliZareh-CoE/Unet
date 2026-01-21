"""Configuration for LOSO (Leave-One-Subject-Out) Cross-Validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class LOSOConfig:
    """Configuration for LOSO cross-validation."""

    # Dataset
    dataset: str = "olfactory"
    output_dir: Path = field(default_factory=lambda: Path("artifacts/loso"))

    # Training
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42

    # Model
    arch: str = "condunet"
    base_channels: int = 128
    n_downsample: int = 2
    attention_type: str = "cross_freq_v2"
    cond_mode: str = "cross_attn_gated"
    conv_type: str = "modern"
    activation: str = "gelu"
    skip_type: str = "add"
    n_heads: int = 4
    conditioning: str = "spectro_temporal"

    # Optimizer
    optimizer: str = "adamw"
    lr_schedule: str = "cosine_warmup"
    weight_decay: float = 0.0
    dropout: float = 0.0

    # Session adaptation
    use_session_stats: bool = False
    use_adaptive_scaling: bool = False
    session_use_spectral: bool = False
    use_bidirectional: bool = False

    # FSDP
    use_fsdp: bool = False
    fsdp_strategy: str = "grad_op"

    # Execution
    resume: bool = True
    verbose: bool = True
    save_models: bool = True
    generate_plots: bool = False

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "output_dir": str(self.output_dir),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "arch": self.arch,
            "base_channels": self.base_channels,
            "n_downsample": self.n_downsample,
            "attention_type": self.attention_type,
            "cond_mode": self.cond_mode,
            "conv_type": self.conv_type,
            "activation": self.activation,
            "skip_type": self.skip_type,
            "n_heads": self.n_heads,
            "conditioning": self.conditioning,
            "optimizer": self.optimizer,
            "lr_schedule": self.lr_schedule,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "use_session_stats": self.use_session_stats,
            "use_adaptive_scaling": self.use_adaptive_scaling,
            "session_use_spectral": self.session_use_spectral,
            "use_bidirectional": self.use_bidirectional,
            "use_fsdp": self.use_fsdp,
            "fsdp_strategy": self.fsdp_strategy,
            "resume": self.resume,
            "verbose": self.verbose,
            "save_models": self.save_models,
            "generate_plots": self.generate_plots,
        }


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

    def to_dict(self) -> Dict[str, Any]:
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
    """Aggregated results from LOSO cross-validation."""

    config: LOSOConfig
    fold_results: List[LOSOFoldResult] = field(default_factory=list)

    mean_r2: float = 0.0
    std_r2: float = 0.0
    mean_loss: float = 0.0
    std_loss: float = 0.0

    all_sessions: List[str] = field(default_factory=list)
    total_time: float = 0.0
    n_folds: int = 0

    def compute_statistics(self):
        if not self.fold_results:
            return

        import numpy as np

        r2_scores = [f.val_r2 for f in self.fold_results]
        losses = [f.val_loss for f in self.fold_results]

        self.mean_r2 = float(np.mean(r2_scores))
        self.std_r2 = float(np.std(r2_scores))
        self.mean_loss = float(np.mean(losses))
        self.std_loss = float(np.std(losses))
        self.n_folds = len(self.fold_results)
        self.total_time = sum(f.total_time for f in self.fold_results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "fold_results": [f.to_dict() for f in self.fold_results],
            "mean_r2": self.mean_r2,
            "std_r2": self.std_r2,
            "mean_loss": self.mean_loss,
            "std_loss": self.std_loss,
            "all_sessions": self.all_sessions,
            "total_time": self.total_time,
            "n_folds": self.n_folds,
        }

    def print_summary(self):
        print("\n" + "=" * 70)
        print("LOSO CROSS-VALIDATION RESULTS")
        print("=" * 70)
        print(f"Dataset: {self.config.dataset}")
        print(f"Folds: {self.n_folds} | Sessions: {len(self.all_sessions)}")
        print(f"Time: {self.total_time / 3600:.2f}h")
        print(f"\nR²: {self.mean_r2:.4f} ± {self.std_r2:.4f}")
        print(f"Loss: {self.mean_loss:.4f} ± {self.std_loss:.4f}")
        print("\nPer-fold:")
        for r in self.fold_results:
            print(f"  {r.fold_idx}: {r.test_session:15s} R²={r.val_r2:.4f}")
        print("=" * 70)
