"""
Configuration for Phase 1: Classical Baselines
===============================================

Defines all configuration parameters for the classical baselines study.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# Neural Frequency Bands (Standard Neuroscience Definitions)
# =============================================================================

NEURAL_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),     # Delta: 1-4 Hz (deep sleep, slow oscillations)
    "theta": (4.0, 8.0),     # Theta: 4-8 Hz (memory, navigation)
    "alpha": (8.0, 12.0),    # Alpha: 8-12 Hz (relaxation, attention)
    "beta": (12.0, 30.0),    # Beta: 12-30 Hz (active thinking, focus)
    "gamma": (30.0, 100.0),  # Gamma: 30-100 Hz (cognition, perception)
}

# =============================================================================
# Baseline Method Names
# =============================================================================

BASELINE_METHODS: List[str] = [
    "wiener",           # Single-channel Wiener filter
    "wiener_mimo",      # Multi-input multi-output Wiener
    "ridge",            # Basic L2-regularized regression
    "ridge_temporal",   # Ridge with temporal context
    "ridge_cv",         # Cross-validated Ridge
    "var",              # Vector Autoregressive
    "var_exogenous",    # VAR with exogenous inputs
]

# Fast baselines for quick testing
FAST_BASELINES: List[str] = ["ridge", "wiener"]


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class Phase1Config:
    """Configuration for Phase 1 classical baselines study.

    Attributes:
        dataset: Dataset name ('olfactory', 'pfc', 'dandi', 'pcx1')
        n_folds: Number of cross-validation folds
        seed: Random seed for reproducibility
        n_bootstrap: Number of bootstrap samples for CI
        ci_level: Confidence interval level (e.g., 0.95 for 95% CI)
        sample_rate: Sampling rate in Hz
        output_dir: Directory for saving results
        baselines: List of baseline methods to evaluate
        use_gpu: Whether to use GPU acceleration
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        save_predictions: Whether to save model predictions
        verbose: Verbosity level
    """

    # Dataset settings
    dataset: str = "olfactory"
    sample_rate: float = 1000.0  # Hz

    # Cross-validation settings
    n_folds: int = 5
    seed: int = 42

    # Statistical settings
    n_bootstrap: int = 10000
    ci_level: float = 0.95

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("results/phase1"))

    # Methods to evaluate
    baselines: List[str] = field(default_factory=lambda: BASELINE_METHODS.copy())

    # Performance settings
    use_gpu: bool = True
    n_jobs: int = -1  # -1 means use all available CPUs

    # Boran MTL dataset settings
    boran_source_region: str = "hippocampus"
    boran_target_region: str = "entorhinal_cortex"

    # Additional options
    save_predictions: bool = False
    verbose: int = 1

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure output_dir is Path
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate dataset
        valid_datasets = {"olfactory", "pfc", "dandi", "pcx1", "ecog"}
        if self.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset: {self.dataset}. Must be one of {valid_datasets}")

        # Validate baselines
        for baseline in self.baselines:
            if baseline not in BASELINE_METHODS:
                raise ValueError(f"Unknown baseline: {baseline}. Must be one of {BASELINE_METHODS}")

        # Validate numerical parameters
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if not 0 < self.ci_level < 1:
            raise ValueError(f"ci_level must be in (0, 1), got {self.ci_level}")

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "dataset": self.dataset,
            "sample_rate": self.sample_rate,
            "n_folds": self.n_folds,
            "seed": self.seed,
            "n_bootstrap": self.n_bootstrap,
            "ci_level": self.ci_level,
            "output_dir": str(self.output_dir),
            "baselines": self.baselines,
            "use_gpu": self.use_gpu,
            "n_jobs": self.n_jobs,
            "save_predictions": self.save_predictions,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "Phase1Config":
        """Create configuration from dictionary."""
        return cls(**config_dict)


# =============================================================================
# Dataset-Specific Configurations
# =============================================================================

@dataclass
class OlfactoryDataConfig:
    """Configuration specific to the olfactory dataset."""

    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("UNET_DATA_DIR", "/data")))
    signal_file: str = "signal_windows_1khz.npy"
    meta_file: str = "signal_windows_meta_1khz.csv"

    # Data dimensions
    n_channels: int = 32
    time_points: int = 5000
    sample_rate: float = 1000.0

    # Region indices in data array
    ob_index: int = 0   # Olfactory Bulb (source)
    pcx_index: int = 1  # Piriform Cortex (target)

    @property
    def signal_path(self) -> Path:
        return self.data_dir / self.signal_file

    @property
    def meta_path(self) -> Path:
        return self.data_dir / self.meta_file


@dataclass
class PFCDataConfig:
    """Configuration specific to the PFC-CA1 dataset."""

    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("UNET_DATA_DIR", "/data")) / "pfc" / "processed_data")
    signal_file: str = "neural_data.npy"
    meta_file: str = "metadata.csv"

    # Data dimensions
    pfc_channels: int = 64    # Source: prefrontal cortex
    ca1_channels: int = 32    # Target: hippocampal CA1
    time_points: int = 6250
    sample_rate: float = 1250.0

    @property
    def signal_path(self) -> Path:
        return self.data_dir / self.signal_file


# =============================================================================
# Gate Threshold
# =============================================================================

# Neural methods must beat classical baseline by this margin
GATE_THRESHOLD: float = 0.10  # R² improvement required


def get_gate_threshold(best_classical_r2: float) -> float:
    """Calculate the gate threshold for neural methods.

    Neural methods must achieve R² >= best_classical + GATE_THRESHOLD

    Args:
        best_classical_r2: Best R² achieved by classical methods

    Returns:
        Minimum R² required for neural methods to pass the gate
    """
    return best_classical_r2 + GATE_THRESHOLD
