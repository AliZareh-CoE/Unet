"""
Configuration for Phase 4: Inter vs Intra Session
==================================================

Defines configuration for generalization study across datasets and split modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch


# =============================================================================
# Split Modes
# =============================================================================

class SplitMode(Enum):
    """Data splitting mode for generalization study."""

    INTRA_SESSION = "intra"  # Random split within each session
    INTER_SESSION = "inter"  # Hold out entire sessions


# =============================================================================
# Dataset Configurations
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a single dataset.

    Attributes:
        name: Dataset identifier
        source_region: Source brain region
        target_region: Target brain region
        n_sessions: Expected number of sessions
        in_channels: Input channels
        out_channels: Output channels
        sample_rate: Sampling rate in Hz
    """

    name: str
    source_region: str
    target_region: str
    n_sessions: int
    in_channels: int = 32
    out_channels: int = 32
    sample_rate: float = 1000.0

    # DANDI-specific
    dandi_source_region: Optional[str] = None
    dandi_target_region: Optional[str] = None


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "olfactory": DatasetConfig(
        name="olfactory",
        source_region="OB",
        target_region="PCx",
        n_sessions=10,  # Approximate
        in_channels=32,
        out_channels=32,
        sample_rate=1000.0,
    ),
    "pfc": DatasetConfig(
        name="pfc",
        source_region="PFC",
        target_region="CA1",
        n_sessions=8,  # Approximate
        in_channels=64,
        out_channels=32,
        sample_rate=1000.0,
    ),
    "dandi": DatasetConfig(
        name="dandi",
        source_region="AMY",
        target_region="HPC",
        n_sessions=18,  # 18 subjects
        in_channels=32,  # Variable
        out_channels=32,  # Variable
        sample_rate=1000.0,
        dandi_source_region="amygdala",
        dandi_target_region="hippocampus",
    ),
}


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters for Phase 4."""

    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Learning rate schedule
    lr_scheduler: str = "cosine"
    lr_min: float = 1e-6

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Gradient clipping
    grad_clip: float = 1.0

    # Loss
    loss_type: str = "huber"

    # Mixed precision
    use_amp: bool = True

    # Robustness
    checkpoint_every: int = 10
    max_nan_recovery: int = 100
    log_every: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "lr_scheduler": self.lr_scheduler,
            "lr_min": self.lr_min,
            "patience": self.patience,
            "grad_clip": self.grad_clip,
            "loss_type": self.loss_type,
            "use_amp": self.use_amp,
        }


# =============================================================================
# Phase 4 Configuration
# =============================================================================

@dataclass
class Phase4Config:
    """Configuration for Phase 4 generalization study.

    Attributes:
        datasets: List of datasets to evaluate
        split_modes: Split modes to test (intra, inter)
        seeds: Random seeds for multiple runs
        training: Training configuration
        output_dir: Directory for results
    """

    # Datasets to evaluate
    datasets: List[str] = field(default_factory=lambda: ["olfactory", "pfc", "dandi"])

    # Split modes
    split_modes: List[SplitMode] = field(
        default_factory=lambda: [SplitMode.INTRA_SESSION, SplitMode.INTER_SESSION]
    )

    # Multiple seeds for statistical validity
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])

    # Training config
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Model configuration (use best from Phase 2/3)
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        "base_channels": 64,
        "n_downsample": 2,
        "attention_type": "cross_freq_v2",
        "cond_mode": "cross_attn_gated",
    })

    # Output
    output_dir: Path = field(default_factory=lambda: Path("results/phase4"))

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_workers: int = 4

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/phase4"))

    # Logging
    use_wandb: bool = False
    wandb_project: str = "neural-signal-translation"
    verbose: int = 1
    log_dir: Path = field(default_factory=lambda: Path("logs/phase4"))

    # Inter-session settings
    test_session_ratio: float = 0.2  # Fraction of sessions for test
    leave_one_out: bool = False  # Use leave-one-session-out CV

    def __post_init__(self):
        """Validate configuration."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Validate datasets
        for ds in self.datasets:
            if ds not in DATASET_CONFIGS:
                raise ValueError(f"Unknown dataset: {ds}")

    def get_dataset_config(self, dataset: str) -> DatasetConfig:
        """Get configuration for a specific dataset."""
        return DATASET_CONFIGS[dataset]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "datasets": self.datasets,
            "split_modes": [m.value for m in self.split_modes],
            "seeds": self.seeds,
            "training": self.training.to_dict(),
            "model_config": self.model_config,
            "output_dir": str(self.output_dir),
            "device": self.device,
            "test_session_ratio": self.test_session_ratio,
            "leave_one_out": self.leave_one_out,
        }

    @property
    def total_runs(self) -> int:
        """Total number of training runs."""
        return len(self.datasets) * len(self.split_modes) * len(self.seeds)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.

    Attributes:
        dataset: Dataset name
        split_mode: Intra or inter session
        seed: Random seed
    """

    dataset: str
    split_mode: SplitMode
    seed: int

    # Dataset info (filled from DATASET_CONFIGS)
    in_channels: int = 32
    out_channels: int = 32
    sample_rate: float = 1000.0

    def __post_init__(self):
        if self.dataset in DATASET_CONFIGS:
            cfg = DATASET_CONFIGS[self.dataset]
            self.in_channels = cfg.in_channels
            self.out_channels = cfg.out_channels
            self.sample_rate = cfg.sample_rate

    @property
    def name(self) -> str:
        """Experiment identifier."""
        return f"{self.dataset}_{self.split_mode.value}_seed{self.seed}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split_mode": self.split_mode.value,
            "seed": self.seed,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
        }
