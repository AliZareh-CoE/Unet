"""
Phase 3 v2: Bulletproof Ablation Study Configuration
=====================================================

DESIGN PRINCIPLES:
1. Single Source of Truth: All defaults in ONE place (dataclasses)
2. Type Safety: Literal types enforce valid values at definition time
3. Pre-flight Validation: Instantiate model BEFORE training
4. Direct Python Calls: No subprocess, no CLI parsing
5. Experiment Registry: Hash configs, skip duplicates
6. Fail-Fast: Invalid config = immediate crash with clear error

Usage:
    config = ExperimentConfig.baseline()
    config = config.with_override(activation="gelu")
    validate_config(config)  # Instantiates model, runs forward pass
    run_experiment(config)   # Direct Python call, no subprocess
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional, Dict, Any, List
from enum import Enum


# =============================================================================
# Type-Safe Enums for All Hyperparameters
# =============================================================================

# If it's not in these literals, it CAN'T be used - caught at definition time
ActivationType = Literal["relu", "leaky_relu", "gelu", "silu", "mish"]
NormType = Literal["batch", "instance", "group", "layer", "none"]
SkipType = Literal["add", "concat"]  # ONLY implemented options!
AttentionType = Literal["none", "basic", "cross_freq_v2"]
ConvType = Literal["standard", "modern"]
CondMode = Literal["none", "cross_attn_gated"]
LossType = Literal["l1", "l1_wavelet", "huber", "huber_wavelet"]
OptimizerType = Literal["adamw", "adam", "sgd", "rmsprop"]
LRScheduleType = Literal["none", "cosine", "cosine_warmup", "step", "plateau"]
AugStrength = Literal["none", "light", "medium", "heavy"]


# =============================================================================
# Typed Configuration Dataclasses
# =============================================================================

@dataclass(frozen=True)  # Immutable - prevents accidental modification
class ModelConfig:
    """Model architecture configuration - ALL defaults in ONE place."""

    # Architecture
    in_channels: int = 32
    out_channels: int = 32
    base_channels: int = 64
    n_downsample: int = 2

    # Components
    conv_type: ConvType = "standard"
    attention_type: AttentionType = "none"
    n_heads: int = 4
    cond_mode: CondMode = "none"

    # Layer choices
    activation: ActivationType = "relu"
    norm_type: NormType = "batch"
    skip_type: SkipType = "concat"  # Original U-Net default
    dropout: float = 0.0

    def __post_init__(self):
        """Validate on creation."""
        if self.dropout < 0 or self.dropout > 0.5:
            raise ValueError(f"dropout must be in [0, 0.5], got {self.dropout}")
        if self.n_heads not in [1, 2, 4, 8, 16]:
            raise ValueError(f"n_heads must be power of 2 up to 16, got {self.n_heads}")
        if self.n_downsample < 1 or self.n_downsample > 5:
            raise ValueError(f"n_downsample must be in [1, 5], got {self.n_downsample}")


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration - ALL defaults in ONE place."""

    # Optimizer
    optimizer: OptimizerType = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Schedule
    lr_schedule: LRScheduleType = "cosine"
    lr_warmup_epochs: int = 5
    lr_min_ratio: float = 0.01

    # Training
    epochs: int = 80
    batch_size: int = 64
    early_stop_patience: int = 15

    # Loss
    loss_type: LossType = "l1"

    # Augmentation
    aug_strength: AugStrength = "none"

    # Bidirectional
    bidirectional: bool = False

    def __post_init__(self):
        """Validate on creation."""
        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


@dataclass(frozen=True)
class DataConfig:
    """Data configuration."""

    dataset: str = "olfactory"
    split_by_session: bool = True
    n_test_sessions: int = 4
    n_val_sessions: int = 1
    seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experiment configuration - immutable, hashable, validated."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Experiment metadata
    name: str = "unnamed"
    description: str = ""

    @classmethod
    def baseline(cls) -> ExperimentConfig:
        """Create baseline configuration (simple U-Net)."""
        return cls(
            model=ModelConfig(),
            training=TrainingConfig(),
            data=DataConfig(),
            name="baseline",
            description="Simple U-Net baseline",
        )

    def with_override(self, **kwargs) -> ExperimentConfig:
        """Create new config with overrides. Validates types automatically.

        Example:
            config.with_override(activation="gelu", learning_rate=1e-4)
        """
        model_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
        training_fields = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
        data_fields = {f.name for f in DataConfig.__dataclass_fields__.values()}

        model_overrides = {}
        training_overrides = {}
        data_overrides = {}
        meta_overrides = {}

        for key, value in kwargs.items():
            if key in model_fields:
                model_overrides[key] = value
            elif key in training_fields:
                training_overrides[key] = value
            elif key in data_fields:
                data_overrides[key] = value
            elif key in ["name", "description"]:
                meta_overrides[key] = value
            else:
                raise ValueError(f"Unknown config key: {key}")

        # Create new immutable configs
        new_model = ModelConfig(**{**asdict(self.model), **model_overrides})
        new_training = TrainingConfig(**{**asdict(self.training), **training_overrides})
        new_data = DataConfig(**{**asdict(self.data), **data_overrides})

        return ExperimentConfig(
            model=new_model,
            training=new_training,
            data=new_data,
            name=meta_overrides.get("name", self.name),
            description=meta_overrides.get("description", self.description),
        )

    def config_hash(self) -> str:
        """Generate unique hash for this config (for deduplication)."""
        # Only hash model + training params, not metadata
        hashable = {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
        }
        json_str = json.dumps(hashable, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for train.py compatibility."""
        result = {}
        result.update(asdict(self.model))
        result.update(asdict(self.training))
        result.update(asdict(self.data))
        return result

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump({
                "model": asdict(self.model),
                "training": asdict(self.training),
                "data": asdict(self.data),
                "name": self.name,
                "description": self.description,
                "config_hash": self.config_hash(),
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            model=ModelConfig(**data["model"]),
            training=TrainingConfig(**data["training"]),
            data=DataConfig(**data["data"]),
            name=data.get("name", "loaded"),
            description=data.get("description", ""),
        )


# =============================================================================
# Ablation Study Definition
# =============================================================================

@dataclass
class AblationVariant:
    """A single variant to test in an ablation."""
    name: str
    description: str
    overrides: Dict[str, Any]


@dataclass
class AblationGroup:
    """A group of variants to compare."""
    group_id: int
    name: str
    description: str
    parameter: str  # Which parameter this tests
    variants: List[AblationVariant]

    def get_configs(self, base: ExperimentConfig) -> List[ExperimentConfig]:
        """Generate all configs for this ablation group."""
        configs = []
        for variant in self.variants:
            config = base.with_override(
                name=f"{self.name}_{variant.name}",
                description=variant.description,
                **variant.overrides,
            )
            configs.append(config)
        return configs


# =============================================================================
# Pre-defined Ablation Groups (Type-Safe!)
# =============================================================================

ABLATION_GROUPS_V2: List[AblationGroup] = [
    # GROUP 1: Normalization
    AblationGroup(
        group_id=1,
        name="normalization",
        description="Normalization layer type",
        parameter="norm_type",
        variants=[
            AblationVariant("batch", "Batch Normalization", {"norm_type": "batch"}),
            AblationVariant("instance", "Instance Normalization", {"norm_type": "instance"}),
            AblationVariant("group", "Group Normalization", {"norm_type": "group"}),
            AblationVariant("layer", "Layer Normalization", {"norm_type": "layer"}),
            AblationVariant("none", "No Normalization", {"norm_type": "none"}),
        ],
    ),

    # GROUP 2: Activation
    AblationGroup(
        group_id=2,
        name="activation",
        description="Activation function",
        parameter="activation",
        variants=[
            AblationVariant("relu", "ReLU", {"activation": "relu"}),
            AblationVariant("gelu", "GELU", {"activation": "gelu"}),
            AblationVariant("silu", "SiLU/Swish", {"activation": "silu"}),
            AblationVariant("mish", "Mish", {"activation": "mish"}),
            AblationVariant("leaky_relu", "Leaky ReLU", {"activation": "leaky_relu"}),
        ],
    ),

    # GROUP 3: Skip Connection (ONLY implemented types!)
    AblationGroup(
        group_id=3,
        name="skip_connection",
        description="Skip connection type",
        parameter="skip_type",
        variants=[
            AblationVariant("concat", "Concatenation (U-Net)", {"skip_type": "concat"}),
            AblationVariant("add", "Addition (ResNet)", {"skip_type": "add"}),
            # NOTE: "attention" and "dense" removed - NOT IMPLEMENTED
        ],
    ),

    # GROUP 4: Convolution Type
    AblationGroup(
        group_id=4,
        name="conv_type",
        description="Convolution architecture",
        parameter="conv_type",
        variants=[
            AblationVariant("standard", "Standard Conv1d", {"conv_type": "standard"}),
            AblationVariant("modern", "Dilated Depthwise-Sep + SE", {"conv_type": "modern"}),
        ],
    ),

    # GROUP 5: Network Depth
    AblationGroup(
        group_id=5,
        name="depth",
        description="Encoder/decoder depth",
        parameter="n_downsample",
        variants=[
            AblationVariant("shallow", "2 levels (125 Hz Nyquist)", {"n_downsample": 2}),
            AblationVariant("medium", "3 levels (62 Hz Nyquist)", {"n_downsample": 3}),
            AblationVariant("deep", "4 levels (31 Hz Nyquist)", {"n_downsample": 4}),
        ],
    ),

    # GROUP 6: Network Width
    AblationGroup(
        group_id=6,
        name="width",
        description="Base channel count",
        parameter="base_channels",
        variants=[
            AblationVariant("narrow", "32 channels", {"base_channels": 32}),
            AblationVariant("medium", "64 channels", {"base_channels": 64}),
            AblationVariant("wide", "128 channels", {"base_channels": 128}),
        ],
    ),

    # GROUP 7: Attention
    AblationGroup(
        group_id=7,
        name="attention",
        description="Attention mechanism",
        parameter="attention_type",
        variants=[
            AblationVariant("none", "No attention", {"attention_type": "none"}),
            AblationVariant("basic", "Basic self-attention", {"attention_type": "basic"}),
            AblationVariant("cross_freq", "Cross-frequency v2", {"attention_type": "cross_freq_v2"}),
        ],
    ),

    # GROUP 8: Conditioning
    AblationGroup(
        group_id=8,
        name="conditioning",
        description="Conditioning mechanism",
        parameter="cond_mode",
        variants=[
            AblationVariant("none", "No conditioning", {"cond_mode": "none"}),
            AblationVariant("gated", "Gated cross-attention", {"cond_mode": "cross_attn_gated"}),
        ],
    ),

    # GROUP 9: Loss Function
    AblationGroup(
        group_id=9,
        name="loss",
        description="Training loss",
        parameter="loss_type",
        variants=[
            AblationVariant("l1", "L1 loss", {"loss_type": "l1"}),
            AblationVariant("huber", "Huber loss", {"loss_type": "huber"}),
            AblationVariant("l1_wavelet", "L1 + Wavelet", {"loss_type": "l1_wavelet"}),
        ],
    ),

    # GROUP 10: Optimizer
    AblationGroup(
        group_id=10,
        name="optimizer",
        description="Optimizer",
        parameter="optimizer",
        variants=[
            AblationVariant("adamw", "AdamW", {"optimizer": "adamw"}),
            AblationVariant("adam", "Adam", {"optimizer": "adam"}),
            AblationVariant("sgd", "SGD + Momentum", {"optimizer": "sgd"}),
        ],
    ),

    # GROUP 11: Learning Rate
    AblationGroup(
        group_id=11,
        name="learning_rate",
        description="Learning rate",
        parameter="learning_rate",
        variants=[
            AblationVariant("low", "1e-4", {"learning_rate": 1e-4}),
            AblationVariant("medium", "1e-3", {"learning_rate": 1e-3}),
            AblationVariant("high", "3e-3", {"learning_rate": 3e-3}),
        ],
    ),

    # GROUP 12: Batch Size
    AblationGroup(
        group_id=12,
        name="batch_size",
        description="Batch size",
        parameter="batch_size",
        variants=[
            AblationVariant("small", "32", {"batch_size": 32}),
            AblationVariant("medium", "64", {"batch_size": 64}),
            AblationVariant("large", "128", {"batch_size": 128}),
            AblationVariant("xlarge", "256", {"batch_size": 256}),
        ],
    ),

    # GROUP 13: Dropout
    AblationGroup(
        group_id=13,
        name="dropout",
        description="Dropout rate",
        parameter="dropout",
        variants=[
            AblationVariant("none", "0.0", {"dropout": 0.0}),
            AblationVariant("light", "0.1", {"dropout": 0.1}),
            AblationVariant("medium", "0.2", {"dropout": 0.2}),
        ],
    ),

    # GROUP 14: LR Schedule
    AblationGroup(
        group_id=14,
        name="lr_schedule",
        description="Learning rate schedule",
        parameter="lr_schedule",
        variants=[
            AblationVariant("none", "Constant", {"lr_schedule": "none"}),
            AblationVariant("cosine", "Cosine annealing", {"lr_schedule": "cosine"}),
            AblationVariant("warmup", "Cosine + warmup", {"lr_schedule": "cosine_warmup"}),
        ],
    ),

    # GROUP 15: Augmentation
    AblationGroup(
        group_id=15,
        name="augmentation",
        description="Data augmentation",
        parameter="aug_strength",
        variants=[
            AblationVariant("none", "No augmentation", {"aug_strength": "none"}),
            AblationVariant("light", "Light aug", {"aug_strength": "light"}),
            AblationVariant("medium", "Medium aug", {"aug_strength": "medium"}),
            AblationVariant("heavy", "Heavy aug", {"aug_strength": "heavy"}),
        ],
    ),
]


# =============================================================================
# Validation Functions
# =============================================================================

def validate_config(config: ExperimentConfig, verbose: bool = True) -> bool:
    """Validate config by instantiating model and running forward pass.

    This catches ALL configuration errors BEFORE training starts.

    Args:
        config: Experiment configuration to validate
        verbose: Print validation details

    Returns:
        True if valid

    Raises:
        RuntimeError: If config is invalid
    """
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models import CondUNet1D

    if verbose:
        print(f"Validating config: {config.name} [{config.config_hash()}]")

    try:
        # Instantiate model with exact config
        model = CondUNet1D(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            base=config.model.base_channels,
            n_downsample=config.model.n_downsample,
            conv_type=config.model.conv_type,
            attention_type=config.model.attention_type,
            n_heads=config.model.n_heads,
            cond_mode=config.model.cond_mode,
            activation=config.model.activation,
            norm_type=config.model.norm_type,
            skip_type=config.model.skip_type,
            dropout=config.model.dropout,
        )

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())

        # Run forward pass with dummy data
        batch_size = 2
        seq_len = 1000
        x = torch.randn(batch_size, config.model.in_channels, seq_len)
        cond = torch.randint(0, 5, (batch_size,))

        model.eval()
        with torch.no_grad():
            output = model(x, cond)

        # Verify output shape
        expected_shape = (batch_size, config.model.out_channels, seq_len)
        if output.shape != expected_shape:
            raise RuntimeError(f"Output shape {output.shape} != expected {expected_shape}")

        if verbose:
            print(f"  ✓ Model instantiated: {n_params:,} parameters")
            print(f"  ✓ Forward pass: input {x.shape} → output {output.shape}")

        return True

    except Exception as e:
        raise RuntimeError(f"Config validation FAILED: {e}") from e


def validate_all_groups(base_config: ExperimentConfig) -> None:
    """Pre-flight check: validate ALL ablation configs before running any.

    This ensures no experiment will crash mid-way due to config issues.
    """
    print("=" * 60)
    print("PRE-FLIGHT VALIDATION: Checking all ablation configurations")
    print("=" * 60)

    total_configs = 0
    for group in ABLATION_GROUPS_V2:
        print(f"\nGroup {group.group_id}: {group.name}")
        configs = group.get_configs(base_config)
        for config in configs:
            validate_config(config, verbose=False)
            print(f"  ✓ {config.name}")
            total_configs += 1

    print(f"\n{'=' * 60}")
    print(f"PRE-FLIGHT COMPLETE: {total_configs} configurations validated")
    print("=" * 60)


# =============================================================================
# Experiment Registry (Deduplication)
# =============================================================================

class ExperimentRegistry:
    """Track completed experiments to avoid duplicates."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self.completed = json.load(f)
        else:
            self.completed = {}

    def _save(self) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(self.completed, f, indent=2)

    def is_completed(self, config: ExperimentConfig) -> bool:
        """Check if this exact config has been run."""
        return config.config_hash() in self.completed

    def mark_completed(self, config: ExperimentConfig, results: Dict[str, Any]) -> None:
        """Mark config as completed with results."""
        self.completed[config.config_hash()] = {
            "name": config.name,
            "completed_at": datetime.now().isoformat(),
            "results": results,
        }
        self._save()

    def get_results(self, config: ExperimentConfig) -> Optional[Dict[str, Any]]:
        """Get results for a completed config."""
        entry = self.completed.get(config.config_hash())
        return entry["results"] if entry else None


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    # Test config creation
    config = ExperimentConfig.baseline()
    print(f"Baseline config hash: {config.config_hash()}")

    # Test override
    config2 = config.with_override(activation="gelu", learning_rate=1e-4)
    print(f"Modified config hash: {config2.config_hash()}")

    # Test validation (requires models.py)
    try:
        validate_config(config)
        print("Validation passed!")
    except Exception as e:
        print(f"Validation requires model imports: {e}")

    # Show all ablation variants
    print("\nAblation Groups:")
    for group in ABLATION_GROUPS_V2:
        print(f"  {group.group_id}. {group.name}: {len(group.variants)} variants")


from datetime import datetime
