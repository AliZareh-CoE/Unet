"""
Nested Cross-Validation for LOSO
================================

Gold standard methodology for limited subjects/sessions:

OUTER LOOP: LOSO (Leave-One-Session-Out)
  - Final unbiased evaluation
  - Each session held out once as test set

INNER LOOP: k-fold CV on remaining sessions
  - Component/hyperparameter selection
  - Runs ablation within each outer fold
  - Winner used to train final model for that fold

This avoids ANY data leakage because:
1. Test session never seen during inner loop selection
2. Different sessions may select different optimal configs (that's ok!)
3. Final metric is mean ± std across outer folds

Multi-Dataset Support
=====================
Supports three datasets:
- olfactory: OB→PCx translation (session-based LOSO)
- pfc_hpc: PFC→CA1 translation (session-based LOSO)
- dandi_movie: Human iEEG movie watching (subject-based LOSO)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import copy

import numpy as np


# =============================================================================
# Dataset Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Dataset-specific configuration for LOSO.

    Encapsulates the unique characteristics of each dataset to enable
    dataset-agnostic LOSO cross-validation.
    """
    # Identification
    name: str
    description: str

    # Session/subject semantics
    # "session" = recording session (olfactory, pfc_hpc)
    # "subject" = individual subject/animal (dandi_movie)
    session_type: str

    # Data characteristics (may be 0 if variable/detected at runtime)
    in_channels: int
    out_channels: int
    sampling_rate: int

    # Source and target region names (for display)
    source_region: str
    target_region: str

    # For sliding window datasets (DANDI, PCx1)
    uses_sliding_window: bool = False
    default_window_size: int = 5000
    default_stride_ratio: float = 0.5

    # train.py dataset name (may differ from LOSO name)
    train_py_dataset_name: str = ""

    def __post_init__(self):
        if not self.train_py_dataset_name:
            self.train_py_dataset_name = self.name

    def copy(self) -> "DatasetConfig":
        """Create a copy of this config."""
        return copy.deepcopy(self)


# Pre-defined dataset configurations
DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "olfactory": DatasetConfig(
        name="olfactory",
        description="Olfactory bulb to piriform cortex translation",
        session_type="session",
        in_channels=32,
        out_channels=32,
        sampling_rate=1000,
        source_region="OB",
        target_region="PCx",
        uses_sliding_window=False,
        train_py_dataset_name="olfactory",
    ),
    "pfc_hpc": DatasetConfig(
        name="pfc_hpc",
        description="Prefrontal cortex to hippocampus (CA1) translation",
        session_type="session",
        in_channels=64,
        out_channels=32,
        sampling_rate=1250,
        source_region="PFC",
        target_region="CA1",
        uses_sliding_window=False,
        train_py_dataset_name="pfc",
    ),
    "dandi_movie": DatasetConfig(
        name="dandi_movie",
        description="Human iEEG movie watching (DANDI 000623)",
        session_type="subject",  # CRITICAL: uses subjects, not sessions
        in_channels=0,  # Variable - detected at runtime
        out_channels=0,  # Variable - detected at runtime
        sampling_rate=1000,
        source_region="amygdala",  # Default, can be changed
        target_region="hippocampus",  # Default, can be changed
        uses_sliding_window=True,
        default_window_size=5000,
        default_stride_ratio=0.5,
        train_py_dataset_name="dandi",
    ),
    "pcx1": DatasetConfig(
        name="pcx1",
        description="Continuous OB to PCx translation (1kHz LFP)",
        session_type="session",
        in_channels=32,
        out_channels=32,
        sampling_rate=1000,
        source_region="OB",
        target_region="PCx",
        uses_sliding_window=True,
        default_window_size=5000,
        default_stride_ratio=0.5,
        train_py_dataset_name="pcx1",
    ),
    "ecog": DatasetConfig(
        name="ecog",
        description="Miller ECoG Library: inter-region cortical translation",
        session_type="subject",  # LOSO holds out entire subjects
        in_channels=0,  # Variable - detected at runtime
        out_channels=0,  # Variable - detected at runtime
        sampling_rate=1000,
        source_region="frontal",  # Default, can be changed
        target_region="temporal",  # Default, can be changed
        uses_sliding_window=True,
        default_window_size=5000,
        default_stride_ratio=0.5,
        train_py_dataset_name="ecog",
    ),
}


def get_dataset_config(dataset: str) -> DatasetConfig:
    """Get the configuration for a dataset.

    Args:
        dataset: Dataset name (olfactory, pfc_hpc, dandi_movie)

    Returns:
        DatasetConfig for the specified dataset

    Raises:
        ValueError: If dataset is not recognized
    """
    if dataset not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset}'. Available datasets: {available}"
        )
    return DATASET_CONFIGS[dataset].copy()


@dataclass
class LOSOConfig:
    """Configuration for nested LOSO cross-validation.

    Supports multiple datasets with dataset-specific parameters.
    """

    # Dataset
    dataset: str = "olfactory"
    output_dir: Path = field(default_factory=lambda: Path("results/loso"))

    # Training (fixed across all experiments)
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    n_seeds: int = 3  # Number of random seeds per fold for robust results

    # Model defaults (optimized from cascading ablation study)
    arch: str = "condunet"
    base_channels: int = 256  # Optimized: 256 (was 128)
    n_downsample: int = 2
    attention_type: str = "none"  # Optimized: no attention
    cond_mode: str = "cross_attn_gated"
    conv_type: str = "modern"
    activation: str = "gelu"  # Optimized: gelu
    skip_type: str = "add"
    n_heads: int = 4
    conditioning: str = "spectro_temporal"

    # Optimizer (fixed)
    optimizer: str = "adamw"
    lr_schedule: str = "cosine_warmup"  # Optimized: cosine with warmup
    weight_decay: float = 0.0  # Optimized: no weight decay
    dropout: float = 0.0

    # Noise augmentation (optimized defaults for LOSO)
    use_noise_augmentation: bool = True  # Enable by default for LOSO
    noise_gaussian_std: float = 0.1
    noise_pink: bool = True
    noise_pink_std: float = 0.05
    noise_channel_dropout: float = 0.05
    noise_temporal_dropout: float = 0.02
    noise_prob: float = 0.5

    # Session adaptation
    use_session_stats: bool = False
    use_adaptive_scaling: bool = True
    session_use_spectral: bool = False
    use_bidirectional: bool = False

    # Wiener residual learning
    wiener_residual: bool = False
    wiener_alpha: float = 1.0

    # FSDP
    use_fsdp: bool = False
    fsdp_strategy: str = "full"
    nproc: Optional[int] = None  # Number of GPUs (None = auto-detect)

    # Nested CV settings
    inner_cv_folds: int = 3  # k-fold CV for inner loop
    run_inner_cv: bool = False  # Set True to run nested CV with ablation

    # Execution
    resume: bool = True
    verbose: bool = True
    save_models: bool = True
    generate_plots: bool = False
    folds_to_run: Optional[List[int]] = None

    # =========================================================================
    # Dataset-specific parameters
    # =========================================================================

    # DANDI Movie dataset options
    dandi_source_region: str = "amygdala"
    dandi_target_region: str = "hippocampus"
    dandi_window_size: int = 5000
    dandi_stride_ratio: float = 0.5

    # PCx1 dataset options
    pcx1_window_size: int = 5000
    pcx1_stride_ratio: float = 0.5

    # PFC/HPC dataset options
    pfc_resample_to_1khz: bool = False
    pfc_sliding_window: bool = False
    pfc_window_size: int = 2500
    pfc_stride_ratio: float = 0.5

    # Miller ECoG Library dataset options
    ecog_experiment: str = "fingerflex"
    ecog_source_region: str = "frontal"
    ecog_target_region: str = "temporal"
    ecog_window_size: int = 5000
    ecog_stride_ratio: float = 0.5

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        # Validate dataset
        if self.dataset not in DATASET_CONFIGS:
            available = ", ".join(DATASET_CONFIGS.keys())
            raise ValueError(
                f"Unknown dataset: '{self.dataset}'. Available: {available}"
            )

    def get_dataset_config(self) -> DatasetConfig:
        """Get the dataset-specific configuration.

        Returns a DatasetConfig with any overrides from this LOSOConfig applied.
        """
        ds_config = get_dataset_config(self.dataset)

        # Apply DANDI-specific overrides
        if self.dataset == "dandi_movie":
            ds_config.source_region = self.dandi_source_region
            ds_config.target_region = self.dandi_target_region
            ds_config.default_window_size = self.dandi_window_size
            ds_config.default_stride_ratio = self.dandi_stride_ratio
        elif self.dataset == "pcx1":
            ds_config.default_window_size = self.pcx1_window_size
            ds_config.default_stride_ratio = self.pcx1_stride_ratio
        elif self.dataset == "ecog":
            ds_config.source_region = self.ecog_source_region
            ds_config.target_region = self.ecog_target_region
            ds_config.default_window_size = self.ecog_window_size
            ds_config.default_stride_ratio = self.ecog_stride_ratio

        return ds_config

    def get_session_type_label(self) -> str:
        """Get human-readable label for session/subject type.

        Returns:
            'Session' for session-based datasets, 'Subject' for subject-based
        """
        ds_config = self.get_dataset_config()
        return "Subject" if ds_config.session_type == "subject" else "Session"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result


@dataclass
class LOSOFoldResult:
    """Result from a single LOSO fold."""
    fold_idx: int
    test_session: str  # Session or subject ID that was held out
    train_sessions: List[str]  # Sessions/subjects used for training
    val_r2: float
    val_loss: float
    val_corr: float = 0.0  # Correlation coefficient
    train_loss: float = 0.0
    per_session_r2: Dict[str, float] = field(default_factory=dict)
    per_session_corr: Dict[str, float] = field(default_factory=dict)  # Per-session correlations
    per_session_loss: Dict[str, float] = field(default_factory=dict)
    epochs_trained: int = 0
    total_time: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)

    # Dataset metadata (for multi-dataset support)
    dataset: str = ""
    session_type: str = ""  # "session" or "subject"
    n_train_samples: int = 0
    n_val_samples: int = 0

    # Legacy aliases
    @property
    def session(self) -> str:
        return self.test_session

    @property
    def best_val_r2(self) -> float:
        return self.val_r2

    @property
    def best_val_corr(self) -> float:
        return self.val_corr

    @property
    def best_val_mae(self) -> float:
        return self.val_loss

    @property
    def best_epoch(self) -> int:
        return self.epochs_trained

    @property
    def train_time(self) -> float:
        return self.total_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "fold_idx": self.fold_idx,
            "test_session": self.test_session,
            "train_sessions": self.train_sessions,
            "val_r2": self.val_r2,
            "val_corr": self.val_corr,
            "val_loss": self.val_loss,
            "train_loss": self.train_loss,
            "per_session_r2": self.per_session_r2,
            "per_session_corr": self.per_session_corr,
            "per_session_loss": self.per_session_loss,
            "epochs_trained": self.epochs_trained,
            "total_time": self.total_time,
            "config": self.config,
            "dataset": self.dataset,
            "session_type": self.session_type,
            "n_train_samples": self.n_train_samples,
            "n_val_samples": self.n_val_samples,
        }


@dataclass
class LOSOResult:
    """Aggregated LOSO results."""
    config: LOSOConfig
    fold_results: List[LOSOFoldResult]
    all_sessions: List[str]

    # Statistics (computed after all folds)
    mean_r2: float = 0.0
    std_r2: float = 0.0
    mean_corr: float = 0.0
    std_corr: float = 0.0
    mean_loss: float = 0.0
    std_loss: float = 0.0
    fold_r2s: List[float] = field(default_factory=list)
    fold_corrs: List[float] = field(default_factory=list)

    # Legacy aliases
    @property
    def mean_mae(self) -> float:
        return self.mean_loss

    @property
    def std_mae(self) -> float:
        return self.std_loss

    @property
    def sessions(self) -> List[str]:
        return self.all_sessions

    def compute_statistics(self) -> None:
        """Compute aggregate statistics from fold results."""
        if not self.fold_results:
            return

        r2_values = [r.val_r2 for r in self.fold_results]
        corr_values = [r.val_corr for r in self.fold_results if r.val_corr != 0.0]
        loss_values = [r.val_loss for r in self.fold_results]

        self.fold_r2s = r2_values
        self.mean_r2 = float(np.mean(r2_values))
        self.std_r2 = float(np.std(r2_values))
        self.mean_loss = float(np.mean(loss_values))
        self.std_loss = float(np.std(loss_values))

        # Compute correlation stats if available
        if corr_values:
            self.fold_corrs = corr_values
            self.mean_corr = float(np.mean(corr_values))
            self.std_corr = float(np.std(corr_values))

    def print_summary(self) -> None:
        """Print dataset-aware summary of LOSO results."""
        print("\n" + "=" * 70)
        print("LOSO CROSS-VALIDATION RESULTS")
        print("=" * 70)

        # Dataset information
        ds_config = self.config.get_dataset_config()
        session_label = "Subject" if ds_config.session_type == "subject" else "Session"

        print(f"\nDataset: {self.config.dataset}")
        print(f"Description: {ds_config.description}")
        print(f"Translation: {ds_config.source_region} → {ds_config.target_region}")
        print(f"CV Type: Leave-One-{session_label}-Out ({len(self.all_sessions)} folds)")

        # Per-fold results table - include correlation if available
        header_label = f"Held-Out {session_label}"
        has_corr = any(r.val_corr != 0.0 for r in self.fold_results)

        if has_corr:
            print(f"\n{'Fold':<6} {header_label:<20} {'R²':>10} {'Corr':>10} {'Loss':>10}")
            print("-" * 60)
            for r in self.fold_results:
                print(f"{r.fold_idx:<6} {r.test_session:<20} {r.val_r2:>10.4f} {r.val_corr:>10.4f} {r.val_loss:>10.4f}")
            print("-" * 60)
            print(f"{'Mean':<6} {'':<20} {self.mean_r2:>10.4f} {self.mean_corr:>10.4f} {self.mean_loss:>10.4f}")
            print(f"{'Std':<6} {'':<20} {self.std_r2:>10.4f} {self.std_corr:>10.4f} {self.std_loss:>10.4f}")
        else:
            print(f"\n{'Fold':<6} {header_label:<20} {'R²':>10} {'Loss':>10}")
            print("-" * 50)
            for r in self.fold_results:
                print(f"{r.fold_idx:<6} {r.test_session:<20} {r.val_r2:>10.4f} {r.val_loss:>10.4f}")
            print("-" * 50)
            print(f"{'Mean':<6} {'':<20} {self.mean_r2:>10.4f} {self.mean_loss:>10.4f}")
            print(f"{'Std':<6} {'':<20} {self.std_r2:>10.4f} {self.std_loss:>10.4f}")

        print()
        if has_corr:
            print(f"Final: R² = {self.mean_r2:.4f} ± {self.std_r2:.4f}, Corr = {self.mean_corr:.4f} ± {self.std_corr:.4f}")
        else:
            print(f"Final: R² = {self.mean_r2:.4f} ± {self.std_r2:.4f}")

        # Print per-session correlations if available
        self._print_per_session_summary()

        # Dataset-specific notes
        if self.config.dataset == "dandi_movie":
            print(f"\nNote: Leave-One-Subject-Out CV (each fold holds out one human subject)")
            print(f"      Source: {self.config.dandi_source_region}, Target: {self.config.dandi_target_region}")
        elif self.config.dataset == "ecog":
            print(f"\nNote: Leave-One-Subject-Out CV on Miller ECoG Library")
            print(f"      Experiment: {self.config.ecog_experiment}")
            print(f"      Source: {self.config.ecog_source_region} cortex, "
                  f"Target: {self.config.ecog_target_region} cortex")

    def _print_per_session_summary(self) -> None:
        """Print per-session correlation summary across all folds."""
        # Collect all per-session correlations across folds
        all_session_corrs: Dict[str, List[float]] = {}
        all_session_r2s: Dict[str, List[float]] = {}

        for fold_result in self.fold_results:
            for session, corr in fold_result.per_session_corr.items():
                if session not in all_session_corrs:
                    all_session_corrs[session] = []
                all_session_corrs[session].append(corr)

            for session, r2 in fold_result.per_session_r2.items():
                if session not in all_session_r2s:
                    all_session_r2s[session] = []
                all_session_r2s[session].append(r2)

        # Print if we have per-session data
        if all_session_corrs or all_session_r2s:
            print("\n" + "-" * 50)
            print("Per-Session Summary (across all folds where session was in validation):")

            sessions = sorted(set(all_session_corrs.keys()) | set(all_session_r2s.keys()))
            for session in sessions:
                r2_vals = all_session_r2s.get(session, [])
                corr_vals = all_session_corrs.get(session, [])

                r2_str = f"R²={np.mean(r2_vals):.4f}±{np.std(r2_vals):.4f}" if r2_vals else "R²=N/A"
                corr_str = f"Corr={np.mean(corr_vals):.4f}±{np.std(corr_vals):.4f}" if corr_vals else ""

                if corr_str:
                    print(f"  {session}: {r2_str}, {corr_str}")
                else:
                    print(f"  {session}: {r2_str}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "fold_results": [r.to_dict() for r in self.fold_results],
            "all_sessions": self.all_sessions,
            "mean_r2": self.mean_r2,
            "std_r2": self.std_r2,
            "mean_corr": self.mean_corr,
            "std_corr": self.std_corr,
            "mean_loss": self.mean_loss,
            "std_loss": self.std_loss,
            "fold_r2s": self.fold_r2s,
            "fold_corrs": self.fold_corrs,
        }
