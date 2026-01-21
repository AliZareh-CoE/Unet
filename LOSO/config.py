"""Configuration for LOSO (Leave-One-Subject-Out) Cross-Validation.

Defines configuration dataclasses and default settings for LOSO evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LOSOConfig:
    """Configuration for LOSO cross-validation.

    Attributes:
        dataset: Dataset to use (olfactory, pfc_hpc, dandi_movie)
        output_dir: Directory to save results and checkpoints
        epochs: Number of training epochs per fold
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        seed: Random seed for reproducibility

        # Model configuration
        arch: Model architecture (default: condunet)
        base_channels: Base channel count for model
        n_downsample: Number of downsampling layers
        attention_type: Type of attention (none, basic, cross_freq_v2)
        cond_mode: Conditioning mode (none, cross_attn_gated, film, etc.)
        conv_type: Convolution type (standard, modern)

        # Training configuration
        optimizer: Optimizer to use (adamw, adam, sgd)
        lr_schedule: LR schedule (cosine, cosine_warmup, step, plateau, constant)
        weight_decay: L2 regularization weight
        dropout: Dropout rate
        loss_type: Loss function (l1, huber)

        # Session adaptation
        use_session_stats: Use statistics-based FiLM conditioning
        use_adaptive_scaling: Use adaptive output scaling

        # FSDP settings
        use_fsdp: Enable FSDP distributed training
        fsdp_strategy: FSDP sharding strategy

        # Execution settings
        resume: Resume from checkpoint
        verbose: Print detailed output
        save_models: Save model checkpoints for each fold
    """

    # Dataset
    dataset: str = "olfactory"
    output_dir: Path = field(default_factory=lambda: Path("artifacts/loso"))

    # Training hyperparameters
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42

    # Model architecture
    arch: str = "condunet"
    base_channels: int = 64
    n_downsample: int = 4
    attention_type: str = "cross_freq_v2"
    cond_mode: str = "cross_attn_gated"
    conv_type: str = "modern"
    norm_type: str = "batch"
    activation: str = "gelu"
    skip_type: str = "add"  # Skip connection type: add, concat
    n_heads: int = 4  # Number of attention heads
    conditioning: str = "spectro_temporal"  # Auto-conditioning: none, spectro_temporal, etc.

    # Training configuration
    optimizer: str = "adamw"
    lr_schedule: str = "cosine_warmup"
    weight_decay: float = 0.0
    dropout: float = 0.0
    loss_type: str = "l1"

    # Session adaptation (generalizes to new sessions)
    use_session_stats: bool = False
    use_adaptive_scaling: bool = False
    session_use_spectral: bool = False

    use_bidirectional: bool = True

    # FSDP distributed training
    use_fsdp: bool = False
    fsdp_strategy: str = "grad_op"

    # Execution settings
    resume: bool = True
    verbose: bool = True
    save_models: bool = True
    generate_plots: bool = False  # Skip plots by default for speed

    def __post_init__(self):
        """Convert output_dir to Path if string."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
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
            "norm_type": self.norm_type,
            "activation": self.activation,
            "skip_type": self.skip_type,
            "n_heads": self.n_heads,
            "conditioning": self.conditioning,
            "optimizer": self.optimizer,
            "lr_schedule": self.lr_schedule,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "loss_type": self.loss_type,
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
    """Result from a single LOSO fold (one held-out session).

    Attributes:
        fold_idx: Fold index (0-indexed)
        test_session: Name of the held-out session
        train_sessions: List of sessions used for training

        # Performance metrics
        val_r2: Best validation R² score
        val_loss: Best validation loss
        train_loss: Final training loss

        # Per-session metrics (if multiple val sessions)
        per_session_r2: Dict mapping session name to R²
        per_session_loss: Dict mapping session name to loss

        # Training metadata
        epochs_trained: Number of epochs trained
        total_time: Training time in seconds
        config: Configuration used for this fold
    """

    fold_idx: int
    test_session: str
    train_sessions: List[str]

    # Performance metrics
    val_r2: float
    val_loss: float
    train_loss: float = 0.0

    # Per-session metrics
    per_session_r2: Dict[str, float] = field(default_factory=dict)
    per_session_loss: Dict[str, float] = field(default_factory=dict)

    # Training metadata
    epochs_trained: int = 0
    total_time: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
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
    """Aggregated results from LOSO cross-validation.

    Attributes:
        config: LOSO configuration used
        fold_results: List of results from each fold

        # Aggregate statistics
        mean_r2: Mean R² across folds
        std_r2: Standard deviation of R² across folds
        mean_loss: Mean loss across folds
        std_loss: Standard deviation of loss across folds

        # All sessions
        all_sessions: List of all session names

        # Metadata
        total_time: Total time for all folds
        n_folds: Number of folds completed
    """

    config: LOSOConfig
    fold_results: List[LOSOFoldResult] = field(default_factory=list)

    # Aggregate statistics (computed after all folds)
    mean_r2: float = 0.0
    std_r2: float = 0.0
    mean_loss: float = 0.0
    std_loss: float = 0.0

    # Session information
    all_sessions: List[str] = field(default_factory=list)

    # Metadata
    total_time: float = 0.0
    n_folds: int = 0

    def compute_statistics(self):
        """Compute aggregate statistics from fold results."""
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
        """Convert result to dictionary."""
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
        """Print summary of LOSO results."""
        print("\n" + "=" * 70)
        print("LOSO CROSS-VALIDATION RESULTS")
        print("=" * 70)
        print(f"Dataset: {self.config.dataset}")
        print(f"Number of folds: {self.n_folds}")
        print(f"Total sessions: {len(self.all_sessions)}")
        print(f"Total time: {self.total_time / 3600:.2f} hours")
        print()
        print(f"Mean R²: {self.mean_r2:.4f} ± {self.std_r2:.4f}")
        print(f"Mean Loss: {self.mean_loss:.4f} ± {self.std_loss:.4f}")
        print()

        print("Per-fold results:")
        print("-" * 50)
        for result in self.fold_results:
            print(f"  Fold {result.fold_idx}: Test={result.test_session:15s} "
                  f"R²={result.val_r2:.4f}  Loss={result.val_loss:.4f}")
        print("=" * 70)
