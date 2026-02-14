"""Phase 4 configuration – datasets, training, and validation parameters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ── Frequency bands (Hz) used across all spectral analyses ────────────────
FREQUENCY_BANDS: Dict[str, tuple] = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "low_gamma": (30, 60),
    "high_gamma": (60, 150),
}

# ── Per-dataset defaults ──────────────────────────────────────────────────
# Maps dataset name (as used by train.py --dataset) to its properties.

@dataclass
class DatasetSpec:
    """Immutable specification for a single dataset."""
    train_name: str          # --dataset value for train.py
    display_name: str        # Human-readable name for figures
    source_region: str
    target_region: str
    in_channels: int
    out_channels: int
    sampling_rate: int       # Hz
    n_labels: int            # Number of classes for decoding (0 = regression only)
    label_column: str        # Column name for trial labels
    extra_train_args: List[str] = field(default_factory=list)


DATASETS: Dict[str, DatasetSpec] = {
    "olfactory": DatasetSpec(
        train_name="olfactory",
        display_name="Olfactory (OB→PCx)",
        source_region="OB",
        target_region="PCx",
        in_channels=32,
        out_channels=32,
        sampling_rate=1000,
        n_labels=7,
        label_column="odor_name",
        extra_train_args=["--split-by-session"],
    ),
    "pfc_hpc": DatasetSpec(
        train_name="pfc",
        display_name="PFC→CA1",
        source_region="PFC",
        target_region="CA1",
        in_channels=64,
        out_channels=32,
        sampling_rate=1250,
        n_labels=2,
        label_column="trial_type",
        extra_train_args=["--split-by-session"],
    ),
    "pfc_hpc_reverse": DatasetSpec(
        train_name="pfc",
        display_name="CA1→PFC",
        source_region="CA1",
        target_region="PFC",
        in_channels=32,
        out_channels=64,
        sampling_rate=1250,
        n_labels=2,
        label_column="trial_type",
        extra_train_args=["--split-by-session", "--pfc-reverse"],
    ),
    "dandi": DatasetSpec(
        train_name="dandi",
        display_name="DANDI iEEG (Amyg→Hipp)",
        source_region="amygdala",
        target_region="hippocampus",
        in_channels=0,    # detected at runtime
        out_channels=0,
        sampling_rate=1000,
        n_labels=0,       # continuous – no discrete labels
        label_column="",
        extra_train_args=[],
    ),
    "ecog": DatasetSpec(
        train_name="ecog",
        display_name="ECoG (Frontal→Temporal)",
        source_region="frontal",
        target_region="temporal",
        in_channels=0,
        out_channels=0,
        sampling_rate=1000,
        n_labels=0,
        label_column="",
        extra_train_args=[
            "--ecog-experiment", "motor_imagery",
            "--ecog-source-region", "frontal",
            "--ecog-target-region", "temporal",
        ],
    ),
    "boran": DatasetSpec(
        train_name="boran",
        display_name="Boran MTL (Hipp→EC)",
        source_region="hippocampus",
        target_region="entorhinal_cortex",
        in_channels=0,
        out_channels=0,
        sampling_rate=1000,
        n_labels=0,
        label_column="",
        extra_train_args=[],
    ),
}


# ── Phase 4 master config ────────────────────────────────────────────────

@dataclass
class Phase4Config:
    """Top-level configuration for Phase 4."""

    # Which datasets to run (keys into DATASETS)
    datasets: List[str] = field(default_factory=lambda: [
        "olfactory", "pfc_hpc", "pfc_hpc_reverse",
        "dandi", "ecog", "boran",
    ])

    # ── Training (aligned with LOSO-optimized config from ablation study) ─
    test_fraction: float = 0.30          # 70 / 30 split
    epochs: int = 80
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42
    n_gpus: int = 1                      # GPUs per training run

    # Architecture (LOSO-optimized)
    arch: str = "condunet"
    base_channels: int = 256             # optimized from 128
    n_downsample: int = 2
    attention_type: str = "none"         # ablation: no attention needed
    conv_type: str = "modern"            # modern depthwise separable
    activation: str = "gelu"
    skip_type: str = "add"              # additive skip connections
    cond_mode: str = "cross_attn_gated"  # gated cross-attention conditioning
    conditioning: str = "spectro_temporal"

    # Optimizer (LOSO-optimized)
    optimizer: str = "adamw"
    lr_schedule: str = "cosine_warmup"
    weight_decay: float = 0.0            # ablation: no weight decay
    dropout: float = 0.0

    # Noise augmentation (LOSO-optimized)
    use_noise_augmentation: bool = True
    noise_gaussian_std: float = 0.1
    noise_pink: bool = True
    noise_pink_std: float = 0.05
    noise_channel_dropout: float = 0.05
    noise_temporal_dropout: float = 0.02
    noise_prob: float = 0.5

    # Session adaptation (LOSO-optimized)
    use_adaptive_scaling: bool = True
    use_bidirectional: bool = False

    # ── Paths ─────────────────────────────────────────────────────────────
    synth_root: Path = Path("/data/synth")
    output_dir: Path = Path("results/phase4")
    checkpoint_dir: Path = Path("artifacts/checkpoints/phase4")

    # ── Validation toggles ────────────────────────────────────────────────
    run_spectral: bool = True
    run_cca: bool = True
    run_decoding: bool = True
    run_pid: bool = True
    run_fingerprint: bool = True

    # ── Spectral validation ───────────────────────────────────────────────
    nperseg: int = 1024                  # Welch PSD segment length
    pac_n_surrogates: int = 200          # PAC significance surrogates
    pac_phase_band: tuple = (4, 8)       # theta
    pac_amp_band: tuple = (30, 100)      # gamma

    # ── CCA / DCCA ────────────────────────────────────────────────────────
    cca_n_components: int = 10
    dcca_hidden_dim: int = 256
    dcca_epochs: int = 100
    dcca_lr: float = 1e-3

    # ── Decoding ──────────────────────────────────────────────────────────
    decode_classifiers: List[str] = field(default_factory=lambda: [
        "lda", "svm", "rf",
    ])
    decode_n_cv_folds: int = 5

    # ── PID ───────────────────────────────────────────────────────────────
    pid_lag_ms: int = 50                 # temporal PID prediction lag

    # ── Fingerprinting ────────────────────────────────────────────────────
    # (no extra params – uses session_ids from data or labels as proxy)

    def get_synth_dir(self, dataset: str) -> Path:
        """Return /data/synth/<dataset>/ path."""
        return self.synth_root / dataset

    def get_checkpoint_path(self, dataset: str) -> Path:
        """Return checkpoint path for a dataset model."""
        return self.checkpoint_dir / dataset / "best_model.pt"
