"""
Configuration for Phase 5: Real-Time Continuous
================================================

Defines configuration for sliding window experiments and benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch


# =============================================================================
# Ablation Constants
# =============================================================================

# Window sizes in samples (at 1000 Hz: 1s, 2.5s, 5s)
WINDOW_SIZES: List[int] = [1000, 2500, 5000]

# Stride ratios (fraction of window size)
STRIDE_RATIOS: List[float] = [0.25, 0.5, 0.75]

# Batch sizes for latency benchmarks
BATCH_SIZES: List[int] = [1, 4, 16, 64]

# Real-time requirements
REALTIME_REQUIREMENTS = {
    "max_latency_ms": 100,  # < 100ms for batch=1
    "min_throughput_hz": 1000,  # > 1000 samples/sec
    "max_memory_gb": 4,  # < 4 GB GPU memory
}


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window processing.

    Attributes:
        window_size: Window size in samples
        stride_ratio: Stride as fraction of window size
        sample_rate: Sampling rate in Hz
    """

    window_size: int = 5000
    stride_ratio: float = 0.5
    sample_rate: float = 1000.0

    @property
    def stride(self) -> int:
        """Stride in samples."""
        return int(self.window_size * self.stride_ratio)

    @property
    def window_duration_ms(self) -> float:
        """Window duration in milliseconds."""
        return self.window_size / self.sample_rate * 1000

    @property
    def stride_duration_ms(self) -> float:
        """Stride duration in milliseconds."""
        return self.stride / self.sample_rate * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "stride_ratio": self.stride_ratio,
            "stride": self.stride,
            "sample_rate": self.sample_rate,
            "window_duration_ms": self.window_duration_ms,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for latency/throughput benchmarks.

    Attributes:
        batch_sizes: Batch sizes to test
        n_warmup: Warmup iterations
        n_iterations: Benchmark iterations
        device: Device to benchmark on
    """

    batch_sizes: List[int] = field(default_factory=lambda: BATCH_SIZES.copy())
    n_warmup: int = 10
    n_iterations: int = 100
    device: str = "cuda"

    # Model configuration
    window_size: int = 5000
    in_channels: int = 32
    out_channels: int = 32

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_sizes": self.batch_sizes,
            "n_warmup": self.n_warmup,
            "n_iterations": self.n_iterations,
            "device": self.device,
            "window_size": self.window_size,
        }


@dataclass
class TrainingConfig:
    """Training hyperparameters for Phase 5."""

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
            "patience": self.patience,
            "use_amp": self.use_amp,
        }


@dataclass
class Phase5Config:
    """Configuration for Phase 5 real-time experiments.

    Attributes:
        datasets: Datasets to evaluate
        window_sizes: Window sizes to test
        stride_ratios: Stride ratios to test
        benchmark: Benchmark configuration
        training: Training configuration
    """

    # Datasets
    datasets: List[str] = field(default_factory=lambda: ["olfactory", "pfc", "dandi"])

    # Sliding window ablations
    window_sizes: List[int] = field(default_factory=lambda: WINDOW_SIZES.copy())
    stride_ratios: List[float] = field(default_factory=lambda: STRIDE_RATIOS.copy())

    # Use single seed for ablations
    seeds: List[int] = field(default_factory=lambda: [42])

    # Benchmark config
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # Training config
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Model config (best from Phase 3)
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        "base_channels": 64,
        "n_downsample": 2,
        "attention_type": "cross_freq_v2",
    })

    # Output
    output_dir: Path = field(default_factory=lambda: Path("results/phase5"))

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_workers: int = 4

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/phase5"))

    # Logging
    verbose: int = 1
    log_dir: Path = field(default_factory=lambda: Path("logs/phase5"))

    # Compare with fixed window baseline
    compare_fixed_window: bool = True

    def __post_init__(self):
        """Validate configuration."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "datasets": self.datasets,
            "window_sizes": self.window_sizes,
            "stride_ratios": self.stride_ratios,
            "seeds": self.seeds,
            "benchmark": self.benchmark.to_dict(),
            "training": self.training.to_dict(),
            "model_config": self.model_config,
            "output_dir": str(self.output_dir),
            "device": self.device,
        }

    @property
    def total_window_ablations(self) -> int:
        """Total window size ablation runs."""
        return len(self.datasets) * len(self.window_sizes) * len(self.seeds)

    @property
    def total_stride_ablations(self) -> int:
        """Total stride ablation runs."""
        return len(self.datasets) * len(self.stride_ratios) * len(self.seeds)

    @property
    def total_runs(self) -> int:
        """Total training runs (window + stride ablations)."""
        # Window ablations (fixed stride=0.5)
        window_runs = len(self.datasets) * len(self.window_sizes)
        # Stride ablations (fixed window=5000)
        stride_runs = len(self.datasets) * len(self.stride_ratios)
        return (window_runs + stride_runs) * len(self.seeds)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    dataset: str
    window_size: int
    stride_ratio: float
    seed: int
    experiment_type: str = "window"  # "window" or "stride"

    @property
    def name(self) -> str:
        return f"{self.dataset}_w{self.window_size}_s{self.stride_ratio}_seed{self.seed}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "window_size": self.window_size,
            "stride_ratio": self.stride_ratio,
            "seed": self.seed,
            "experiment_type": self.experiment_type,
        }
