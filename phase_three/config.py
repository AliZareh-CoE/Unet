"""
Configuration for Phase 3: CondUNet Ablation Studies
=====================================================

Defines configuration for systematic ablation experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch

# =============================================================================
# Ablation Variants
# =============================================================================

ATTENTION_VARIANTS: Dict[str, str] = {
    "none": "none",
    "basic": "basic",
    "cross_freq": "cross_freq",
    "cross_freq_v2": "cross_freq_v2",
}

CONDITIONING_VARIANTS: Dict[str, Dict[str, Any]] = {
    "none": {"cond_mode": "none", "use_odor_embedding": False},
    "odor": {"cond_mode": "film", "use_odor_embedding": True},
    "spectro_temporal": {"cond_mode": "cross_attn_gated", "use_odor_embedding": True},
}

LOSS_VARIANTS: Dict[str, Dict[str, Any]] = {
    "l1": {"loss_type": "l1", "loss_weights": None},
    "huber": {"loss_type": "huber", "loss_weights": None},
    "wavelet": {"loss_type": "wavelet", "loss_weights": None},
    "combined": {"loss_type": "combined", "loss_weights": {"huber": 1.0, "wavelet": 0.5}},
}

CAPACITY_VARIANTS: Dict[str, int] = {
    "small": 32,
    "medium": 64,
    "large": 128,
}

BIDIRECTIONAL_VARIANTS: Dict[str, bool] = {
    "unidirectional": False,
    "bidirectional": True,
}

AUGMENTATION_VARIANTS: Dict[str, bool] = {
    "none": False,
    "full": True,
}

# =============================================================================
# Simplified Ablation Configuration (One variant per study)
# =============================================================================

# For each study, select the "ablated" variant (feature removed/reduced)
# Baseline has all features ON, ablated versions remove one feature each
ABLATION_VARIANTS_SIMPLE: Dict[str, str] = {
    "attention": "none",           # Remove attention
    "conditioning": "none",        # Remove conditioning
    "loss": "l1",                  # Alternative loss (L1 instead of Huber)
    "capacity": "small",           # Reduced capacity
    "bidirectional": "unidirectional",  # Remove bidirectional
    "augmentation": "none",        # No augmentation
}

# =============================================================================
# Ablation Study Definitions
# =============================================================================

ABLATION_STUDIES: Dict[str, Dict[str, Any]] = {
    "attention": {
        "description": "Attention mechanism comparison",
        "variants": list(ATTENTION_VARIANTS.keys()),
        "parameter": "attention_type",
        "n_variants": 4,
    },
    "conditioning": {
        "description": "Conditioning strategy comparison",
        "variants": list(CONDITIONING_VARIANTS.keys()),
        "parameter": "conditioning",
        "n_variants": 3,
    },
    "loss": {
        "description": "Loss function comparison",
        "variants": list(LOSS_VARIANTS.keys()),
        "parameter": "loss",
        "n_variants": 4,
    },
    "capacity": {
        "description": "Model capacity comparison",
        "variants": list(CAPACITY_VARIANTS.keys()),
        "parameter": "base_channels",
        "n_variants": 3,
    },
    "bidirectional": {
        "description": "Training direction comparison",
        "variants": list(BIDIRECTIONAL_VARIANTS.keys()),
        "parameter": "bidirectional",
        "n_variants": 2,
    },
    "augmentation": {
        "description": "Data augmentation comparison",
        "variants": list(AUGMENTATION_VARIANTS.keys()),
        "parameter": "use_augmentation",
        "n_variants": 2,
    },
}


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment.

    Attributes:
        study: Name of the ablation study (attention, conditioning, etc.)
        variant: Specific variant being tested
        base_config: Base CondUNet configuration
    """

    study: str
    variant: str

    # CondUNet base configuration (defaults to optimal from Phase 2)
    attention_type: str = "cross_freq_v2"
    cond_mode: str = "cross_attn_gated"
    use_odor_embedding: bool = True
    base_channels: int = 64
    n_downsample: int = 2

    # Loss configuration
    loss_type: str = "huber"
    loss_weights: Optional[Dict[str, float]] = None

    # Training options
    use_augmentation: bool = True
    bidirectional: bool = True

    # Augmentation parameters
    aug_noise_std: float = 0.1
    aug_time_shift: int = 50

    def __post_init__(self):
        """Apply ablation variant modifications."""
        if self.study == "attention":
            self.attention_type = ATTENTION_VARIANTS[self.variant]
        elif self.study == "conditioning":
            cond_cfg = CONDITIONING_VARIANTS[self.variant]
            self.cond_mode = cond_cfg["cond_mode"]
            self.use_odor_embedding = cond_cfg["use_odor_embedding"]
        elif self.study == "loss":
            loss_cfg = LOSS_VARIANTS[self.variant]
            self.loss_type = loss_cfg["loss_type"]
            self.loss_weights = loss_cfg["loss_weights"]
        elif self.study == "capacity":
            self.base_channels = CAPACITY_VARIANTS[self.variant]
        elif self.study == "bidirectional":
            self.bidirectional = BIDIRECTIONAL_VARIANTS[self.variant]
        elif self.study == "augmentation":
            self.use_augmentation = AUGMENTATION_VARIANTS[self.variant]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study": self.study,
            "variant": self.variant,
            "attention_type": self.attention_type,
            "cond_mode": self.cond_mode,
            "use_odor_embedding": self.use_odor_embedding,
            "base_channels": self.base_channels,
            "n_downsample": self.n_downsample,
            "loss_type": self.loss_type,
            "loss_weights": self.loss_weights,
            "use_augmentation": self.use_augmentation,
            "bidirectional": self.bidirectional,
        }

    @property
    def name(self) -> str:
        """Experiment name for logging."""
        return f"{self.study}_{self.variant}"


@dataclass
class TrainingConfig:
    """Training hyperparameters for ablation experiments."""

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

    # Mixed precision
    use_amp: bool = True

    # Robustness settings
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
            "use_amp": self.use_amp,
            "checkpoint_every": self.checkpoint_every,
            "max_nan_recovery": self.max_nan_recovery,
        }


@dataclass
class Phase3Config:
    """Configuration for Phase 3 ablation studies.

    Attributes:
        dataset: Dataset name ('olfactory')
        studies: Which ablation studies to run
        n_folds: Number of cross-validation folds
        cv_seed: Random seed for CV splits
        training: Training configuration
        output_dir: Directory for results
    """

    # Dataset (Phase 3 uses olfactory only)
    dataset: str = "olfactory"
    sample_rate: float = 1000.0

    # Which ablation studies to run
    studies: List[str] = field(default_factory=lambda: list(ABLATION_STUDIES.keys()))

    # Cross-validation settings
    n_folds: int = 5
    cv_seed: int = 42

    # Training config
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Output
    output_dir: Path = field(default_factory=lambda: Path("results/phase3"))

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_workers: int = 4

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/phase3"))

    # Logging
    use_wandb: bool = False
    wandb_project: str = "neural-signal-translation"
    verbose: int = 1
    log_dir: Path = field(default_factory=lambda: Path("logs/phase3"))

    # Phase 2 baseline (best architecture R²)
    phase2_best_r2: Optional[float] = None

    def __post_init__(self):
        """Validate configuration."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Validate studies
        for study in self.studies:
            if study not in ABLATION_STUDIES:
                raise ValueError(f"Unknown ablation study: {study}")

    def get_ablation_configs(self) -> List[AblationConfig]:
        """Generate ablation configurations (one ablated variant per study)."""
        configs = []
        for study in self.studies:
            # Use simplified variant (one per study)
            variant = ABLATION_VARIANTS_SIMPLE[study]
            configs.append(AblationConfig(study=study, variant=variant))
        return configs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sample_rate": self.sample_rate,
            "studies": self.studies,
            "n_folds": self.n_folds,
            "cv_seed": self.cv_seed,
            "training": self.training.to_dict(),
            "output_dir": str(self.output_dir),
            "device": self.device,
            "phase2_best_r2": self.phase2_best_r2,
        }

    @property
    def total_runs(self) -> int:
        """Total number of runs: (1 baseline + n_studies ablations) × n_folds."""
        # 1 baseline + 6 ablation studies = 7 configs × 5 folds = 35
        return (1 + len(self.studies)) * self.n_folds


def get_baseline_config() -> AblationConfig:
    """Get the baseline (optimal) configuration.

    This represents the full CondUNet with all features enabled.
    """
    return AblationConfig(
        study="baseline",
        variant="full",
        attention_type="cross_freq_v2",
        cond_mode="cross_attn_gated",
        use_odor_embedding=True,
        base_channels=64,
        n_downsample=2,
        loss_type="combined",
        loss_weights={"huber": 1.0, "wavelet": 0.5},
        use_augmentation=True,
        bidirectional=True,
    )
