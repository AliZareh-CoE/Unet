"""Configuration classes for modular neural signal translation.

This module provides dataclass-based configuration for:
- DataConfig: Data loading and preprocessing settings
- ModelConfig: Model architecture hyperparameters
- TrainConfig: Training hyperparameters
- All configurations are JSON-serializable for reproducibility

Usage:
    from config import DataConfig, ModelConfig, TrainConfig

    data_cfg = DataConfig(
        data_path="/path/to/data.npy",
        region1_channels=(0, 32),   # First 32 channels
        region2_channels=(32, 64),  # Next 32 channels
        sampling_rate=1000,
    )
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    Supports any data format with two regions, each having shape:
    (trials, channels, samples)

    The data can be provided as:
    1. Single array: (trials, total_channels, samples) - split by channel indices
    2. Single array: (trials, 2, channels, samples) - already split into regions
    3. Two arrays: region1.npy and region2.npy separately

    Attributes:
        data_path: Path to main data file (.npy)
        region1_channels: Channel range for region 1 as (start, end) tuple
        region2_channels: Channel range for region 2 as (start, end) tuple
        sampling_rate: Sampling rate in Hz
        meta_path: Optional path to metadata CSV
        label_column: Column name for labels in metadata CSV
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        seed: Random seed for reproducibility
    """
    data_path: str
    region1_channels: Tuple[int, int] = (0, 32)
    region2_channels: Tuple[int, int] = (32, 64)
    sampling_rate: float = 1000.0
    meta_path: Optional[str] = None
    label_column: Optional[str] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42

    # Optional: second data path for separate region files
    region2_data_path: Optional[str] = None

    # Data format hints
    data_format: str = "single"  # "single", "stacked", or "separate"
    # "single": (trials, channels, samples) - split by channel indices
    # "stacked": (trials, 2, channels, samples) - pre-split
    # "separate": two files, one per region

    @property
    def n_channels_region1(self) -> int:
        return self.region1_channels[1] - self.region1_channels[0]

    @property
    def n_channels_region2(self) -> int:
        return self.region2_channels[1] - self.region2_channels[0]

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataConfig":
        """Load configuration from JSON file."""
        data = json.loads(Path(path).read_text())
        # Convert lists back to tuples
        data["region1_channels"] = tuple(data["region1_channels"])
        data["region2_channels"] = tuple(data["region2_channels"])
        return cls(**data)


@dataclass
class ModelConfig:
    """Configuration for model architecture.

    The model is a conditional U-Net that translates signals from
    region1 to region2. Architecture adapts to input/output channel counts.

    Attributes:
        in_channels: Number of input channels (region1)
        out_channels: Number of output channels (region2)
        base_channels: Base feature channels (doubled at each level)
        n_downsample: Number of downsampling levels
        dropout: Dropout probability
        use_attention: Whether to use attention in bottleneck
        attention_type: Type of attention ("self", "cross_freq", "cross_freq_v2")
        norm_type: Normalization type ("instance", "batch", "group")
        cond_mode: Conditioning mode ("none", "cross_attn_gated")
        conv_type: Convolution type ("standard", "modern")
    """
    in_channels: int = 32
    out_channels: int = 32
    base_channels: int = 64
    n_downsample: int = 4
    dropout: float = 0.0
    use_attention: bool = True
    attention_type: str = "cross_freq_v2"
    norm_type: str = "batch"
    cond_mode: str = "none"
    conv_type: str = "modern"
    use_se: bool = True
    conv_kernel_size: int = 7
    conv_dilations: Tuple[int, ...] = (1, 4, 16, 32)

    # Conditioning embedding dimension (if using conditioning)
    cond_dim: int = 128
    n_classes: int = 0  # Number of classes for conditional generation

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        data = asdict(self)
        data["conv_dilations"] = list(data["conv_dilations"])
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelConfig":
        """Load configuration from JSON file."""
        data = json.loads(Path(path).read_text())
        data["conv_dilations"] = tuple(data["conv_dilations"])
        return cls(**data)


@dataclass
class TrainConfig:
    """Configuration for training.

    Attributes:
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        loss_type: Loss function type
        early_stop_patience: Epochs without improvement before stopping
        device: Device to train on ("cuda", "cpu", or specific GPU)
    """
    batch_size: int = 8
    num_epochs: int = 80
    learning_rate: float = 2e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0

    # Loss configuration
    loss_type: str = "l1_wavelet"  # "l1", "mse", "huber", "wavelet", "l1_wavelet"
    weight_l1: float = 1.0
    weight_wavelet: float = 3.0

    # Learning rate schedule
    lr_scheduler: str = "none"  # "none", "cosine", "plateau"
    lr_warmup_epochs: int = 5
    lr_min_ratio: float = 0.01

    # Early stopping
    early_stop_patience: int = 15

    # Training options
    grad_clip: float = 5.0
    use_amp: bool = True  # Automatic mixed precision
    seed: int = 42

    # Augmentation master switch
    aug_enabled: bool = True
    aug_noise_std: float = 0.05
    aug_time_shift_max: float = 0.1
    aug_channel_dropout_p: float = 0.1

    # Bidirectional training
    use_bidirectional: bool = False

    # Output paths
    output_dir: str = "artifacts"
    checkpoint_name: str = "best_model.pt"

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainConfig":
        """Load configuration from JSON file."""
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all sub-configs.

    This is the main configuration class that bundles data, model,
    and training configurations together.
    """
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    name: str = "experiment"
    description: str = ""

    def save(self, path: Union[str, Path]) -> None:
        """Save complete experiment config to JSON."""
        data = {
            "name": self.name,
            "description": self.description,
            "data": asdict(self.data),
            "model": asdict(self.model),
            "train": asdict(self.train),
        }
        # Convert tuples to lists for JSON
        data["data"]["region1_channels"] = list(data["data"]["region1_channels"])
        data["data"]["region2_channels"] = list(data["data"]["region2_channels"])
        data["model"]["conv_dilations"] = list(data["model"]["conv_dilations"])
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load complete experiment config from JSON."""
        data = json.loads(Path(path).read_text())

        # Convert lists back to tuples
        data["data"]["region1_channels"] = tuple(data["data"]["region1_channels"])
        data["data"]["region2_channels"] = tuple(data["data"]["region2_channels"])
        data["model"]["conv_dilations"] = tuple(data["model"]["conv_dilations"])

        return cls(
            name=data.get("name", "experiment"),
            description=data.get("description", ""),
            data=DataConfig(**data["data"]),
            model=ModelConfig(**data["model"]),
            train=TrainConfig(**data["train"]),
        )

    @classmethod
    def from_args(cls, args) -> "ExperimentConfig":
        """Create config from argparse namespace."""
        data_cfg = DataConfig(
            data_path=args.data_path,
            region1_channels=tuple(args.region1_channels),
            region2_channels=tuple(args.region2_channels),
            sampling_rate=args.sampling_rate,
            meta_path=getattr(args, "meta_path", None),
            label_column=getattr(args, "label_column", None),
            seed=args.seed,
        )

        model_cfg = ModelConfig(
            in_channels=data_cfg.n_channels_region1,
            out_channels=data_cfg.n_channels_region2,
            base_channels=args.base_channels,
            n_downsample=args.n_downsample,
            dropout=args.dropout,
            use_attention=args.use_attention,
            norm_type=args.norm_type,
            conv_type=args.conv_type,
        )

        train_cfg = TrainConfig(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            output_dir=args.output_dir,
        )

        return cls(data=data_cfg, model=model_cfg, train=train_cfg)
