"""
Configuration for Phase 2: Architecture Screening
=================================================

Defines configuration for neural architecture comparison study.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

# =============================================================================
# Architecture List
# =============================================================================

ARCHITECTURE_LIST: List[str] = [
    "linear",       # Linear baseline
    "simplecnn",    # Simple CNN
    "wavenet",      # WaveNet with dilated convolutions
    "fnet",         # Fourier-based mixing
    "vit",          # Vision Transformer 1D
    "performer",    # Linear attention transformer
    # "mamba",      # State-space model - disabled (FSDP incompatible, numerical instability)
    "condunet",     # Conditional U-Net (our method)
]

# Fast architectures for quick testing
FAST_ARCHITECTURES: List[str] = ["linear", "simplecnn"]


# =============================================================================
# Default Hyperparameters per Architecture
# =============================================================================

ARCHITECTURE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "linear": {
        "hidden_dim": 256,
    },
    "simplecnn": {
        "channels": [64, 128, 256],
        "kernel_size": 7,
    },
    "wavenet": {
        "residual_channels": 64,
        "skip_channels": 128,
        "n_layers": 8,
        "kernel_size": 3,
    },
    "fnet": {
        "hidden_dim": 256,
        "n_layers": 4,
        "dropout": 0.1,
    },
    "vit": {
        "embed_dim": 256,
        "n_heads": 8,
        "n_layers": 4,
        "patch_size": 50,
        "dropout": 0.1,
    },
    "performer": {
        "embed_dim": 256,
        "n_heads": 8,
        "n_layers": 4,
        "n_features": 64,  # Random features for FAVOR+
    },
    "mamba": {
        "d_model": 256,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "n_layers": 4,
    },
    "condunet": {
        "base_channels": 64,
        "n_downsample": 3,
        "attention_type": "cross_freq_v2",
        "use_film": True,
    },
}


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # cosine, step, plateau
    lr_min: float = 1e-6

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Gradient clipping
    grad_clip: float = 1.0

    # Loss function
    loss_fn: str = "l1"  # l1, mse, huber

    # Data augmentation
    use_augmentation: bool = True
    aug_noise_std: float = 0.1
    aug_time_shift: int = 50

    # Mixed precision
    use_amp: bool = True

    # Robustness settings for long runs
    checkpoint_every: int = 10  # Save checkpoint every N epochs
    max_nan_recovery: int = 100  # Max NaN batches before stopping
    log_every: int = 5  # Log progress every N epochs

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
            "loss_fn": self.loss_fn,
            "use_augmentation": self.use_augmentation,
            "use_amp": self.use_amp,
            "checkpoint_every": self.checkpoint_every,
            "max_nan_recovery": self.max_nan_recovery,
        }


# =============================================================================
# Phase 2 Configuration
# =============================================================================

@dataclass
class Phase2Config:
    """Configuration for Phase 2 architecture screening.

    Attributes:
        dataset: Dataset name ('olfactory', 'pfc', 'dandi')
        architectures: List of architectures to evaluate
        seeds: Random seeds for multiple runs
        training: Training configuration
        output_dir: Directory for results
        device: Torch device
        n_workers: DataLoader workers
    """

    # Dataset
    dataset: str = "olfactory"
    sample_rate: float = 1000.0

    # Architectures to evaluate
    architectures: List[str] = field(default_factory=lambda: ARCHITECTURE_LIST.copy())

    # Cross-validation settings
    n_folds: int = 5
    cv_seed: int = 42  # Seed for CV split reproducibility

    # Training config
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Output
    output_dir: Path = field(default_factory=lambda: Path("results/phase2"))

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_workers: int = 4

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/phase2"))

    # Logging
    use_wandb: bool = False
    wandb_project: str = "neural-signal-translation"
    verbose: int = 1
    log_dir: Path = field(default_factory=lambda: Path("logs/phase2"))

    def __post_init__(self):
        """Validate configuration."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Validate architectures
        for arch in self.architectures:
            if arch not in ARCHITECTURE_LIST:
                raise ValueError(f"Unknown architecture: {arch}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sample_rate": self.sample_rate,
            "architectures": self.architectures,
            "n_folds": self.n_folds,
            "cv_seed": self.cv_seed,
            "training": self.training.to_dict(),
            "output_dir": str(self.output_dir),
            "device": self.device,
        }

    @property
    def total_runs(self) -> int:
        """Total number of training runs (architectures × folds)."""
        return len(self.architectures) * self.n_folds


# =============================================================================
# Gate Threshold (from Phase 1)
# =============================================================================

def check_gate_threshold(arch_r2: float, classical_r2: float, margin: float = 0.10) -> bool:
    """Check if architecture passes the gate threshold.

    Args:
        arch_r2: R² achieved by neural architecture
        classical_r2: Best R² from Phase 1 classical baselines
        margin: Required improvement margin

    Returns:
        True if architecture passes the gate
    """
    return arch_r2 >= classical_r2 + margin
