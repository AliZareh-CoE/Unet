"""
Configuration for Phase 3: CondUNet Ablation Studies
=====================================================

Supports two ablation protocols:
1. SUBTRACTIVE (traditional): Start with full model, remove components
2. ADDITIVE (build-up): Start with simple baseline, add components incrementally

The ADDITIVE protocol is recommended based on literature review:
- Avoids "compensatory masquerade" problem
- Clearer attribution of component contributions
- Aligned with nnU-Net (Nature Methods) and Transformer paper methodology
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch

# =============================================================================
# Ablation Protocol Selection
# =============================================================================

ABLATION_PROTOCOLS = ["additive", "subtractive"]

# =============================================================================
# ADDITIVE PROTOCOL: Incremental Component Analysis
# =============================================================================
# Start with simple baseline, add one component at a time
# Based on: nnU-Net (Nature Methods), Transformer paper methodology

# Ordered list of components to add incrementally
# Each stage adds ONE component on top of previous stage
INCREMENTAL_STAGES: List[Dict[str, Any]] = [
    {
        "stage": 0,
        "name": "simple_baseline",
        "description": "Plain U-Net: standard conv, no attention, no conditioning, L1 loss",
        "config": {
            "conv_type": "standard",
            "attention_type": "none",
            "cond_mode": "none",
            "use_odor_embedding": False,
            "loss_type": "l1",
            "use_augmentation": False,
            "bidirectional": False,
            "base_channels": 64,
        },
    },
    {
        "stage": 1,
        "name": "modern_conv",
        "description": "+ Modern convolutions (dilated depthwise separable + SE)",
        "adds": "conv_type",
        "config": {
            "conv_type": "modern",
            "attention_type": "none",
            "cond_mode": "none",
            "use_odor_embedding": False,
            "loss_type": "l1",
            "use_augmentation": False,
            "bidirectional": False,
            "base_channels": 64,
        },
    },
    {
        "stage": 2,
        "name": "attention",
        "description": "+ Cross-frequency attention mechanism",
        "adds": "attention_type",
        "config": {
            "conv_type": "modern",
            "attention_type": "cross_freq_v2",
            "cond_mode": "none",
            "use_odor_embedding": False,
            "loss_type": "l1",
            "use_augmentation": False,
            "bidirectional": False,
            "base_channels": 64,
        },
    },
    {
        "stage": 3,
        "name": "conditioning",
        "description": "+ Spectro-temporal auto-conditioning",
        "adds": "cond_mode",
        "config": {
            "conv_type": "modern",
            "attention_type": "cross_freq_v2",
            "cond_mode": "cross_attn_gated",
            "use_odor_embedding": True,
            "loss_type": "l1",
            "use_augmentation": False,
            "bidirectional": False,
            "base_channels": 64,
        },
    },
    {
        "stage": 4,
        "name": "wavelet_loss",
        "description": "+ Wavelet multi-scale loss",
        "adds": "loss_type",
        "config": {
            "conv_type": "modern",
            "attention_type": "cross_freq_v2",
            "cond_mode": "cross_attn_gated",
            "use_odor_embedding": True,
            "loss_type": "l1_wavelet",
            "use_augmentation": False,
            "bidirectional": False,
            "base_channels": 64,
        },
    },
    {
        "stage": 5,
        "name": "augmentation",
        "description": "+ Data augmentation suite",
        "adds": "use_augmentation",
        "config": {
            "conv_type": "modern",
            "attention_type": "cross_freq_v2",
            "cond_mode": "cross_attn_gated",
            "use_odor_embedding": True,
            "loss_type": "l1_wavelet",
            "use_augmentation": True,
            "bidirectional": False,
            "base_channels": 64,
        },
    },
    {
        "stage": 6,
        "name": "bidirectional",
        "description": "+ Bidirectional training (cycle consistency)",
        "adds": "bidirectional",
        "config": {
            "conv_type": "modern",
            "attention_type": "cross_freq_v2",
            "cond_mode": "cross_attn_gated",
            "use_odor_embedding": True,
            "loss_type": "l1_wavelet",
            "use_augmentation": True,
            "bidirectional": True,
            "base_channels": 64,
        },
    },
]

# Component order for reference (matches stage order)
COMPONENT_ORDER = [
    "conv_type",        # Stage 1: standard -> modern
    "attention_type",   # Stage 2: none -> cross_freq_v2
    "cond_mode",        # Stage 3: none -> cross_attn_gated
    "loss_type",        # Stage 4: l1 -> l1_wavelet
    "use_augmentation", # Stage 5: False -> True
    "bidirectional",    # Stage 6: False -> True
]

# =============================================================================
# SUBTRACTIVE PROTOCOL: Traditional Ablation (legacy)
# =============================================================================
# Start with full model, remove components one at a time

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
    "l1_wavelet": {"loss_type": "l1_wavelet", "loss_weights": None},
    "huber": {"loss_type": "huber", "loss_weights": None},
    "huber_wavelet": {"loss_type": "huber_wavelet", "loss_weights": None},
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

CONV_TYPE_VARIANTS: Dict[str, str] = {
    "standard": "standard",
    "modern": "modern",
}

# For subtractive: which variant to use when "ablating" (removing) each component
ABLATION_VARIANTS_SIMPLE: Dict[str, str] = {
    "attention": "none",
    "conditioning": "none",
    "loss": "l1",
    "capacity": "small",
    "bidirectional": "unidirectional",
    "augmentation": "none",
    "conv_type": "standard",
}

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
    "conv_type": {
        "description": "Convolution type comparison",
        "variants": list(CONV_TYPE_VARIANTS.keys()),
        "parameter": "conv_type",
        "n_variants": 2,
    },
}


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment.

    Works with both additive and subtractive protocols.
    """

    study: str
    variant: str

    # CondUNet configuration
    attention_type: str = "cross_freq_v2"
    cond_mode: str = "cross_attn_gated"
    use_odor_embedding: bool = True
    base_channels: int = 64
    n_downsample: int = 2
    conv_type: str = "modern"

    # Loss configuration
    loss_type: str = "l1_wavelet"
    loss_weights: Optional[Dict[str, float]] = None

    # Training options
    use_augmentation: bool = True
    bidirectional: bool = True

    # Augmentation parameters
    aug_noise_std: float = 0.1
    aug_time_shift: int = 50

    # Stage info for additive protocol
    stage: int = -1  # -1 = subtractive protocol

    def __post_init__(self):
        """Apply ablation variant modifications for subtractive protocol."""
        if self.stage >= 0:
            # Additive protocol: config already set from INCREMENTAL_STAGES
            return

        # Subtractive protocol: modify based on study/variant
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
        elif self.study == "conv_type":
            self.conv_type = CONV_TYPE_VARIANTS[self.variant]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study": self.study,
            "variant": self.variant,
            "stage": self.stage,
            "attention_type": self.attention_type,
            "cond_mode": self.cond_mode,
            "use_odor_embedding": self.use_odor_embedding,
            "base_channels": self.base_channels,
            "n_downsample": self.n_downsample,
            "conv_type": self.conv_type,
            "loss_type": self.loss_type,
            "loss_weights": self.loss_weights,
            "use_augmentation": self.use_augmentation,
            "bidirectional": self.bidirectional,
        }

    @property
    def name(self) -> str:
        """Experiment name for logging."""
        if self.stage >= 0:
            return f"stage{self.stage}_{self.variant}"
        return f"{self.study}_{self.variant}"

    @classmethod
    def from_stage(cls, stage_idx: int) -> "AblationConfig":
        """Create config from incremental stage definition."""
        if stage_idx < 0 or stage_idx >= len(INCREMENTAL_STAGES):
            raise ValueError(f"Invalid stage index: {stage_idx}")

        stage = INCREMENTAL_STAGES[stage_idx]
        cfg = stage["config"]

        return cls(
            study=f"stage{stage_idx}",
            variant=stage["name"],
            stage=stage_idx,
            conv_type=cfg["conv_type"],
            attention_type=cfg["attention_type"],
            cond_mode=cfg["cond_mode"],
            use_odor_embedding=cfg["use_odor_embedding"],
            loss_type=cfg["loss_type"],
            use_augmentation=cfg["use_augmentation"],
            bidirectional=cfg["bidirectional"],
            base_channels=cfg["base_channels"],
        )


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

    Supports both ADDITIVE (build-up) and SUBTRACTIVE (traditional) protocols.
    """

    # Dataset (Phase 3 uses olfactory only)
    dataset: str = "olfactory"
    sample_rate: float = 1000.0

    # Ablation protocol: "additive" (recommended) or "subtractive"
    protocol: str = "additive"

    # For subtractive protocol: which studies to run
    studies: List[str] = field(default_factory=lambda: list(ABLATION_STUDIES.keys()))

    # For additive protocol: which stages to run (default: all)
    stages: List[int] = field(default_factory=lambda: list(range(len(INCREMENTAL_STAGES))))

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

        if self.protocol not in ABLATION_PROTOCOLS:
            raise ValueError(f"Unknown protocol: {self.protocol}. Use 'additive' or 'subtractive'")

        # Validate studies (for subtractive protocol)
        for study in self.studies:
            if study not in ABLATION_STUDIES:
                raise ValueError(f"Unknown ablation study: {study}")

        # Validate stages (for additive protocol)
        for stage in self.stages:
            if stage < 0 or stage >= len(INCREMENTAL_STAGES):
                raise ValueError(f"Invalid stage: {stage}. Valid: 0-{len(INCREMENTAL_STAGES)-1}")

    def get_ablation_configs(self) -> List[AblationConfig]:
        """Generate ablation configurations based on selected protocol."""
        if self.protocol == "additive":
            return self._get_additive_configs()
        else:
            return self._get_subtractive_configs()

    def _get_additive_configs(self) -> List[AblationConfig]:
        """Generate configs for additive (build-up) protocol."""
        configs = []
        for stage_idx in self.stages:
            configs.append(AblationConfig.from_stage(stage_idx))
        return configs

    def _get_subtractive_configs(self) -> List[AblationConfig]:
        """Generate configs for subtractive (traditional) protocol."""
        configs = []
        for study in self.studies:
            variant = ABLATION_VARIANTS_SIMPLE[study]
            configs.append(AblationConfig(study=study, variant=variant))
        return configs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sample_rate": self.sample_rate,
            "protocol": self.protocol,
            "studies": self.studies,
            "stages": self.stages,
            "n_folds": self.n_folds,
            "cv_seed": self.cv_seed,
            "training": self.training.to_dict(),
            "output_dir": str(self.output_dir),
            "device": self.device,
            "phase2_best_r2": self.phase2_best_r2,
        }

    @property
    def total_runs(self) -> int:
        """Total number of training runs."""
        if self.protocol == "additive":
            # All stages × n_folds
            return len(self.stages) * self.n_folds
        else:
            # (1 baseline + n_studies ablations) × n_folds
            return (1 + len(self.studies)) * self.n_folds


def get_baseline_config() -> AblationConfig:
    """Get the baseline configuration.

    For ADDITIVE protocol: Stage 0 (simple baseline)
    For SUBTRACTIVE protocol: Full model (all features enabled)
    """
    # Return full model config (Stage 6 = final stage with all components)
    return AblationConfig.from_stage(len(INCREMENTAL_STAGES) - 1)


def get_simple_baseline_config() -> AblationConfig:
    """Get the simple baseline configuration (Stage 0)."""
    return AblationConfig.from_stage(0)


def print_protocol_summary(protocol: str = "additive"):
    """Print summary of the ablation protocol."""
    print("\n" + "=" * 70)
    print(f"ABLATION PROTOCOL: {protocol.upper()}")
    print("=" * 70)

    if protocol == "additive":
        print("\nINCREMENTAL COMPONENT ANALYSIS (Build-Up Approach)")
        print("Based on: nnU-Net (Nature Methods), Transformer paper methodology")
        print("-" * 70)
        print(f"{'Stage':<8}{'Name':<20}{'Description':<42}")
        print("-" * 70)
        for stage in INCREMENTAL_STAGES:
            print(f"{stage['stage']:<8}{stage['name']:<20}{stage['description']:<42}")
        print("-" * 70)
        print("\nEach stage adds ONE component on top of previous stage.")
        print("This shows the incremental contribution of each component.")
    else:
        print("\nTRADITIONAL ABLATION (Subtractive Approach)")
        print("-" * 70)
        print("Start with full model, remove one component at a time.")
        print(f"{'Study':<15}{'Ablated Variant':<20}{'Description':<35}")
        print("-" * 70)
        for study, info in ABLATION_STUDIES.items():
            variant = ABLATION_VARIANTS_SIMPLE[study]
            print(f"{study:<15}{variant:<20}{info['description']:<35}")
        print("-" * 70)

    print("=" * 70 + "\n")
