"""Stage 1 Training: Neural signal translation with UNet1D.

This script trains a UNet model for OB → PCx signal translation.
For post-hoc calibration (Stage 2), use calibrate.py after training.

Usage:
    # Train model (single GPU)
    python train.py --epochs 80

    # Distributed training with FSDP
    torchrun --nproc_per_node=4 train.py --epochs 80 --fsdp

    # Cross-session evaluation with 4 test sessions
    python train.py --epochs 80 --n-test-sessions 4

    # After training, run calibration:
    python calibrate.py --checkpoint artifacts/checkpoints/best_model.pt

Notes:
    - Auto-conditioning from input signal (--conditioning spectro_temporal) is the default
    - This is NOT odor conditioning - it learns conditioning from the signal dynamics
    - Session-based splitting is used for proper cross-session generalization
"""
from __future__ import annotations

import argparse
import faulthandler
import functools
import math
import os
import sys
import traceback
import warnings

# Enable faulthandler for better crash diagnostics in distributed training
faulthandler.enable()
from collections import defaultdict
from datetime import datetime, timedelta
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from tqdm import tqdm

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Suppress FSDP and distributed warnings
warnings.filterwarnings("ignore", message=".*Both mixed precision and an auto_wrap_policy.*")
warnings.filterwarnings("ignore", message=".*FSDP.state_dict_type.*deprecated.*")
warnings.filterwarnings("ignore", message=".*barrier.*using the device under current context.*")
warnings.filterwarnings("ignore", message=".*will be wrapped as separate FSDP.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")

# Local imports
from models import (
    CondUNet1D,
    build_wavelet_loss,
    pearson_batch,
    pearson_per_channel,
    cross_channel_correlation,
    explained_variance_torch,
    normalized_rmse_torch,
    plv_torch,
    pli_torch,
    psd_error_db_torch,
    psd_diff_db_torch,
    MAX_FREQ_HZ,
    # Conditioning encoders
    SpectroTemporalEncoder,
    CPCEncoder,
    VQVAEEncoder,
    FreqDisentangledEncoder,
    CycleConsistentEncoder,
    hilbert_torch,
    # Session matching for inference
    SessionMatcher,
)
from data import (
    prepare_data,
    prepare_pfc_data,
    create_dataloaders,
    create_pfc_dataloaders,
    create_pfc_sliding_window_dataloaders,
    create_single_session_dataloader,
    crop_to_target_torch,
    SAMPLING_RATE_HZ,
    PFC_SAMPLING_RATE_HZ,
    DatasetType,
    # PCx1 continuous data
    list_pcx1_sessions,
    load_pcx1_session,
    create_pcx1_dataloaders,
    get_pcx1_session_splits,
    PCX1_SAMPLING_RATE,
    # DANDI 000623 movie dataset
    prepare_dandi_data,
    create_dandi_dataloaders,
    DANDI_SAMPLING_RATE_HZ,
    DANDI_BRAIN_REGIONS,
    _DANDI_DATA_DIR,
)

# Phase 2 architecture imports (for --arch flag)
try:
    from phase_two.architectures import create_architecture
    PHASE2_ARCHS_AVAILABLE = True
except ImportError:
    PHASE2_ARCHS_AVAILABLE = False

# Recording system imports (for comprehensive analysis)
try:
    from recording import (
        RecordingSession,
        RecordingConfig,
        NeuroVisualizer,
    )
    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False
    RecordingSession = None
    RecordingConfig = None
    NeuroVisualizer = None

# Validation plotting (for end-of-training plots)
try:
    from validate_model import generate_training_plots
    VALIDATION_PLOTS_AVAILABLE = True
except ImportError:
    VALIDATION_PLOTS_AVAILABLE = False
    generate_training_plots = None


# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = Path("artifacts")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

# Default hyperparameters
DEFAULT_CONFIG = {
    # Training
    "batch_size": 8,
    "num_epochs": 80,
    "learning_rate": 0.0002,
    "beta1": 0.7595905764360957,
    "beta2": 0.920298282605139,
    "weight_decay": 0.0,            # L2 regularization (0.01-0.1 typical)
    "lr_scheduler": "none",         # "none", "cosine", "cosine_warmup"
    "lr_warmup_epochs": 5,          # Warmup epochs for cosine_warmup
    "lr_min_ratio": 0.01,           # Min lr as ratio of initial (for cosine)
    "early_stop_patience": 15,  # Increased for better PSD convergence
    "seed": 42,
    "generate_plots": True,         # Generate validation plots at end of training

    # Loss weights
    "weight_l1": 1.0,  # Also used for Huber weight
    "weight_wavelet": 3.0,  # Reduced from 10.0 for better balance
    "weight_spectral": 5.0,  # Increased for better PSD matching
    "cycle_lambda": 1.0,  # Cycle consistency weight

    # Loss type selection
    # Options: "l1", "huber", "wavelet", "l1_wavelet", "huber_wavelet"
    #   - "l1": L1/MAE only
    #   - "huber": Huber only (smooth L1, robust to outliers)
    #   - "wavelet": Wavelet only (time-frequency)
    #   - "l1_wavelet": L1 + Wavelet combined (default)
    #   - "huber_wavelet": Huber + Wavelet combined
    "loss_type": "l1_wavelet",

    # Model
    "base_channels": 64,
    "dropout": 0.0,
    "use_attention": True,
    "attention_type": "cross_freq_v2",  # Cross-frequency coupling attention (theta-gamma)
    "norm_type": "batch",
    "cond_mode": "cross_attn_gated",  # Cross-attention with gating (uses auto-conditioning from --conditioning)
    
    # U-Net depth (controls frequency resolution at bottleneck)
    # n_downsample=2: 4x downsample → 125 Hz Nyquist (full gamma, uses more memory)
    # n_downsample=3: 8x downsample → 62 Hz Nyquist (low gamma)
    # n_downsample=4: 16x downsample → 31 Hz Nyquist (default)
    "n_downsample": 4,

    # Modern convolutions (for improved correlation in Stage 1)
    "conv_type": "modern",  # "standard" (Conv1d k=3) or "modern" (multi-scale dilated depthwise sep + SE)
    "use_se": True,  # SE channel attention in modern conv blocks
    "conv_kernel_size": 7,  # Kernel size for modern convs (ConvNeXt-style)
    "conv_dilations": (1, 4, 16, 32),  # Multi-scale dilation rates tuned for LFP bands

    # Wavelet loss configuration
    "wavelet_family": "morlet",
    "wavelet_omega0": 3.0,
    "use_complex_morlet": False,

    # Loss toggles (set False to disable)
    "use_wavelet_loss": True,   # Time-frequency matching (Morlet wavelet decomposition)

    # Data augmentation (applied during training only)
    # Master toggle - set to False to disable ALL augmentations at once
    "aug_enabled": True,            # Master switch for all augmentations
    # HEAVY augmentation for cross-session generalization
    "aug_time_shift": True,         # Random circular time shift
    "aug_time_shift_max": 0.2,      # Max shift as fraction of signal length (20%)
    "aug_noise": True,              # Add Gaussian noise
    "aug_noise_std": 0.1,           # Noise std relative to signal std (heavy)
    "aug_channel_dropout": True,    # Randomly zero out channels
    "aug_channel_dropout_p": 0.2,   # Probability of dropping each channel (heavy)
    "aug_amplitude_scale": True,    # Random amplitude scaling
    "aug_amplitude_scale_range": (0.5, 1.5),  # Scale factor range (heavy - simulates gain drift)
    "aug_time_mask": True,          # Randomly mask time segments
    "aug_time_mask_ratio": 0.15,    # Fraction of time to mask
    "aug_mixup": True,              # Mixup: blend random sample pairs
    "aug_mixup_alpha": 0.4,         # Beta distribution alpha
    "aug_freq_mask": True,          # Frequency masking: zero out random freq bands
    "aug_freq_mask_max_bands": 3,   # Max number of frequency bands to mask
    "aug_freq_mask_max_width": 20,  # Max width of each masked band (in freq bins)

    # Session-specific augmentation (simulates cross-session variability)
    "aug_channel_scale": True,      # Per-channel random scaling (simulates electrode drift)
    "aug_channel_scale_range": (0.7, 1.4),  # Per-channel scale range
    "aug_dc_offset": True,          # Random DC offset per channel
    "aug_dc_offset_range": (-0.3, 0.3),  # DC offset range (relative to signal std)

    # Contrastive learning for session-invariant representations
    "use_contrastive": False,       # Enable CEBRA-style contrastive learning
    "contrastive_weight": 0.1,      # Weight for contrastive loss
    "contrastive_temperature": 0.1, # Temperature for InfoNCE loss
    "contrastive_mode": "temporal", # "temporal" (true CEBRA) or "label" (behavior-supervised)
    "contrastive_time_delta": 10,   # Time delta (in samples) for temporal positive pairs
    "contrastive_num_samples": 32,  # Number of time points to sample per trial

    # Bidirectional training
    "use_bidirectional": True,  # Train both OB→PCx and PCx→OB

    # Output scaling correction (learnable per-channel scale and bias)
    # Helps match target distribution, especially important for probabilistic losses
    "use_output_scaling": True,

    # Recording system (for Nature Methods publication)
    # WARNING: Recording is VERY slow - only enable for final runs!
    "enable_recording": False,  # Enable comprehensive recording
    "record_saliency": False,   # Compute saliency maps and Grad-CAM
    "record_neuroscience": False,  # Compute PAC, coherence, ERP, burst analysis
    "saliency_epoch_interval": 5,  # Compute saliency every N epochs
    "neuroscience_epoch_interval": 10,  # Compute neuroscience metrics every N epochs
    "recording_output_dir": "artifacts/recordings",  # Output directory for recordings

    # Session-based splitting (for cross-session generalization)
    # When True, entire recording sessions are held out for val/test
    # This tests true cross-session generalization (harder but more realistic)
    "split_by_session": False,  # Use session-based holdout instead of random splits
    "n_test_sessions": 1,       # Number of sessions to hold out for testing
    "n_val_sessions": 3,        # Number of sessions to hold out for validation
    "session_column": "recording_id",  # CSV column containing session/recording IDs
    "no_test_set": True,        # If True, no test set - all held-out sessions for validation
    "separate_val_sessions": True,  # If True, evaluate each val session separately
}


GRAD_CLIP = 5.0


# =============================================================================
# Logging
# =============================================================================

class TeeOutput:
    """Duplicate stdout/stderr to both console and a log file."""

    def __init__(self, log_file: Path, stream: TextIOWrapper):
        self.log_file = log_file
        self.stream = stream
        self.file = open(log_file, "a", buffering=1, encoding="utf-8")

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def isatty(self) -> bool:
        return self.stream.isatty()

    @property
    def encoding(self) -> str:
        return self.stream.encoding


def setup_logging(log_dir: Path = LOGS_DIR) -> Path:
    """Setup live logging to a timestamped file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Training Log - Started at {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

    sys.stdout = TeeOutput(log_file, sys.__stdout__)
    sys.stderr = TeeOutput(log_file, sys.__stderr__)

    print(f"Logging to: {log_file}")
    return log_file


# =============================================================================
# FSDP Configuration
# =============================================================================

def check_bf16_support() -> bool:
    """Check if BFloat16 is supported."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False


def get_fsdp_config(cpu_offload: bool = False, sharding_strategy: str = "full", num_gpus: int = 1) -> Dict[str, Any]:
    """Get FSDP configuration optimized for the given setup.

    Args:
        cpu_offload: Whether to offload params to CPU (for very large models)
        sharding_strategy: "full", "grad_op", or "no_shard"
        num_gpus: Number of GPUs (used for tuning all-gather behavior)
    """
    if check_bf16_support():
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif torch.cuda.is_available():
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        mixed_precision = None

    strategy_map = {
        "full": ShardingStrategy.FULL_SHARD,
        "grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        # Hybrid strategies for multi-node setups
        "hybrid": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    # For 8+ GPUs with good interconnect (NVLink), allow concurrent all-gathers
    # For smaller setups, limit to avoid memory spikes
    limit_all_gathers = num_gpus < 8

    config = {
        "sharding_strategy": strategy,
        "mixed_precision": mixed_precision,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "forward_prefetch": True,
        "limit_all_gathers": limit_all_gathers,
        "use_orig_params": True,
    }

    if cpu_offload:
        config["cpu_offload"] = CPUOffload(offload_params=True)

    return config


def wrap_model_fsdp(
    model: nn.Module,
    device_id: int,
    use_fsdp: bool = True,
    cpu_offload: bool = False,
    sharding_strategy: str = "full",
    compile_model: bool = False,
) -> nn.Module:
    """Wrap model with FSDP for distributed training.

    Optimized for multi-GPU setups (8x A100 recommended settings):
    - Uses SHARD_GRAD_OP for --fsdp-strategy=grad_op (less communication)
    - Increased min_num_params to 1M to reduce FSDP unit overhead
    - Concurrent all-gathers enabled for 8+ GPUs with NVLink
    """
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    if not use_fsdp or not dist.is_initialized():
        model = model.to(device)
        if compile_model and hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                model = torch.compile(model, mode="default", fullgraph=True)
                if is_primary():
                    print("Model compiled with torch.compile")
            except Exception as e:
                if is_primary():
                    print(f"Warning: torch.compile failed: {e}")
        return model

    model = model.to(device)
    num_gpus = get_world_size()
    fsdp_config = get_fsdp_config(
        cpu_offload=cpu_offload,
        sharding_strategy=sharding_strategy,
        num_gpus=num_gpus,
    )

    # Increased from 100K to 1M params to reduce FSDP unit overhead
    # Fewer FSDP units = fewer all-gather/reduce-scatter collective ops
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)

    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "sharding_strategy": fsdp_config["sharding_strategy"],
        "mixed_precision": fsdp_config["mixed_precision"],
        "backward_prefetch": fsdp_config["backward_prefetch"],
        "forward_prefetch": fsdp_config.get("forward_prefetch", True),
        "limit_all_gathers": fsdp_config["limit_all_gathers"],
        "use_orig_params": fsdp_config["use_orig_params"],
        "device_id": device_id,
        "sync_module_states": True,
    }

    if "cpu_offload" in fsdp_config:
        fsdp_kwargs["cpu_offload"] = fsdp_config["cpu_offload"]

    return FSDP(model, **fsdp_kwargs)


def get_checkpoint_cond_mode(path: Path) -> Optional[str]:
    """Extract cond_mode from checkpoint without loading full state dict.
    
    Returns:
        cond_mode string if found, None otherwise.
    """
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("cond_mode", None)


# =============================================================================
# Per-Channel Normalization
# =============================================================================

def per_channel_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply per-channel z-score normalization to a batch of signals.

    Each channel is normalized independently using its own mean and std
    computed over the time dimension.

    Args:
        x: Input tensor of shape (batch, channels, time)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor with same shape, where each channel has
        approximately zero mean and unit variance.
    """
    # Compute mean and std per channel (over time dimension)
    # Shape: (batch, channels, 1)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=eps)

    return (x - mean) / std


# =============================================================================
# Data Augmentation Functions
# =============================================================================

def aug_time_shift(x: torch.Tensor, max_shift: float = 0.1) -> torch.Tensor:
    """Apply random circular time shift to each sample in batch.

    Args:
        x: Input tensor (batch, channels, time)
        max_shift: Maximum shift as fraction of signal length

    Returns:
        Shifted tensor (same shape)
    """
    batch_size, _, time_len = x.shape
    max_shift_samples = int(time_len * max_shift)

    if max_shift_samples == 0:
        return x

    # Random shift per sample in batch
    shifts = torch.randint(-max_shift_samples, max_shift_samples + 1, (batch_size,), device=x.device)

    # Apply circular shift to each sample
    shifted = torch.zeros_like(x)
    for i in range(batch_size):
        shifted[i] = torch.roll(x[i], shifts=shifts[i].item(), dims=-1)

    return shifted


def aug_gaussian_noise(x: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    """Add Gaussian noise scaled relative to signal std.

    Args:
        x: Input tensor (batch, channels, time)
        noise_std: Noise std as fraction of signal std per channel

    Returns:
        Noisy tensor (same shape)
    """
    # Compute per-channel std
    signal_std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    noise = torch.randn_like(x) * signal_std * noise_std
    return x + noise


def aug_channel_dropout(x: torch.Tensor, dropout_p: float = 0.1) -> torch.Tensor:
    """Randomly zero out entire channels.

    Args:
        x: Input tensor (batch, channels, time)
        dropout_p: Probability of dropping each channel

    Returns:
        Tensor with some channels zeroed (same shape)
    """
    batch_size, n_channels, _ = x.shape
    # Create dropout mask (batch, channels, 1)
    mask = torch.bernoulli(torch.full((batch_size, n_channels, 1), 1 - dropout_p, device=x.device))
    return x * mask


def aug_amplitude_scale(x: torch.Tensor, scale_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    """Apply random amplitude scaling per sample.

    Args:
        x: Input tensor (batch, channels, time)
        scale_range: (min_scale, max_scale) tuple

    Returns:
        Scaled tensor (same shape)
    """
    batch_size = x.shape[0]
    min_scale, max_scale = scale_range
    # Random scale per sample (batch, 1, 1)
    scales = torch.empty(batch_size, 1, 1, device=x.device).uniform_(min_scale, max_scale)
    return x * scales


def aug_time_mask(x: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
    """Randomly mask contiguous time segments with zeros.

    Args:
        x: Input tensor (batch, channels, time)
        mask_ratio: Fraction of time to mask

    Returns:
        Masked tensor (same shape)
    """
    batch_size, _, time_len = x.shape
    mask_len = int(time_len * mask_ratio)

    if mask_len == 0:
        return x

    result = x.clone()
    for i in range(batch_size):
        # Random start position for mask
        start = torch.randint(0, time_len - mask_len + 1, (1,)).item()
        result[i, :, start:start + mask_len] = 0

    return result


def aug_mixup(
    ob: torch.Tensor,
    pcx: torch.Tensor,
    alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Mixup augmentation: blend random pairs of samples.

    Mixup creates virtual training examples by linearly interpolating
    between pairs of samples and their targets.

    Args:
        ob: Input tensor (batch, channels, time)
        pcx: Target tensor (batch, channels, time)
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        Tuple of (mixed_ob, mixed_pcx)
    """
    batch_size = ob.shape[0]
    if batch_size < 2:
        return ob, pcx

    # Sample mixing coefficients from Beta distribution
    # lam ~ Beta(alpha, alpha), typically alpha in [0.2, 0.4]
    lam = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(ob.device)
    lam = lam.view(-1, 1, 1)  # Shape for broadcasting

    # Random permutation for pairing
    perm = torch.randperm(batch_size, device=ob.device)

    # Mix samples: x_mix = lam * x + (1 - lam) * x[perm]
    ob_mixed = lam * ob + (1 - lam) * ob[perm]
    pcx_mixed = lam * pcx + (1 - lam) * pcx[perm]

    return ob_mixed, pcx_mixed


def aug_freq_mask(
    x: torch.Tensor,
    max_bands: int = 2,
    max_width: int = 10
) -> torch.Tensor:
    """Apply frequency masking: zero out random frequency bands.

    Similar to SpecAugment for audio, but applied to neural signals.
    Operates in frequency domain via FFT.

    Args:
        x: Input tensor (batch, channels, time)
        max_bands: Maximum number of frequency bands to mask
        max_width: Maximum width of each masked band (in freq bins)

    Returns:
        Frequency-masked tensor (same shape)
    """
    batch_size, n_channels, time_len = x.shape

    # FFT to frequency domain
    x_fft = torch.fft.rfft(x, dim=-1)
    n_freqs = x_fft.shape[-1]

    if n_freqs <= max_width:
        return x  # Signal too short for meaningful masking

    # Create frequency mask
    for i in range(batch_size):
        n_bands = torch.randint(1, max_bands + 1, (1,)).item()
        for _ in range(n_bands):
            width = torch.randint(1, max_width + 1, (1,)).item()
            start = torch.randint(0, n_freqs - width, (1,)).item()
            x_fft[i, :, start:start + width] = 0

    # Inverse FFT back to time domain
    x_masked = torch.fft.irfft(x_fft, n=time_len, dim=-1)

    return x_masked


def aug_channel_scale(
    x: torch.Tensor,
    scale_range: Tuple[float, float] = (0.7, 1.4)
) -> torch.Tensor:
    """Apply random per-channel scaling (simulates electrode drift/impedance changes).

    Each channel gets a different random scale factor, simulating how electrodes
    drift differently across recording sessions.

    Args:
        x: Input tensor (batch, channels, time)
        scale_range: (min_scale, max_scale) range for per-channel scaling

    Returns:
        Channel-scaled tensor (same shape)
    """
    batch_size, n_channels, _ = x.shape
    min_scale, max_scale = scale_range

    # Generate per-channel scales (same for all samples in batch for consistency)
    scales = torch.empty(1, n_channels, 1, device=x.device).uniform_(min_scale, max_scale)
    scales = scales.expand(batch_size, -1, -1)

    return x * scales


def aug_dc_offset(
    x: torch.Tensor,
    offset_range: Tuple[float, float] = (-0.3, 0.3)
) -> torch.Tensor:
    """Apply random per-channel DC offset (simulates baseline drift).

    Each channel gets a different random offset, simulating how electrode
    baselines drift differently across sessions.

    Args:
        x: Input tensor (batch, channels, time)
        offset_range: (min_offset, max_offset) range relative to signal std

    Returns:
        Offset-adjusted tensor (same shape)
    """
    batch_size, n_channels, _ = x.shape
    min_offset, max_offset = offset_range

    # Compute signal std for reference scaling
    signal_std = x.std(dim=-1, keepdim=True).mean()

    # Generate per-channel offsets
    offsets = torch.empty(1, n_channels, 1, device=x.device).uniform_(min_offset, max_offset)
    offsets = offsets.expand(batch_size, -1, -1) * signal_std

    return x + offsets


def apply_augmentations(
    ob: torch.Tensor,
    pcx: torch.Tensor,
    config: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply configured augmentations to input/target pairs.

    IMPORTANT: Augmentations that change timing (time_shift) are applied
    identically to both input and target to maintain correspondence.
    Other augmentations are applied independently or only to input.

    Args:
        ob: Input (OB) tensor (batch, channels, time)
        pcx: Target (PCx) tensor (batch, channels, time)
        config: Config dict with aug_* keys

    Returns:
        Tuple of (augmented_ob, augmented_pcx)
    """
    # Master toggle - if disabled, skip ALL augmentations
    if not config.get("aug_enabled", True):
        return ob, pcx

    # Time shift: apply same shift to both (maintains correspondence)
    if config.get("aug_time_shift", False):
        max_shift = config.get("aug_time_shift_max", 0.1)
        # Generate shifts once, apply to both
        batch_size, _, time_len = ob.shape
        max_shift_samples = int(time_len * max_shift)
        if max_shift_samples > 0:
            shifts = torch.randint(-max_shift_samples, max_shift_samples + 1, (batch_size,), device=ob.device)
            # Use list + stack to avoid inplace assignment on autograd-tracked tensors
            ob_shifted_list = []
            pcx_shifted_list = []
            for i in range(batch_size):
                ob_shifted_list.append(torch.roll(ob[i], shifts=shifts[i].item(), dims=-1))
                pcx_shifted_list.append(torch.roll(pcx[i], shifts=shifts[i].item(), dims=-1))
            ob = torch.stack(ob_shifted_list, dim=0)
            pcx = torch.stack(pcx_shifted_list, dim=0)

    # Noise: add independently to both (realistic noise augmentation)
    if config.get("aug_noise", False):
        noise_std = config.get("aug_noise_std", 0.05)
        ob = aug_gaussian_noise(ob, noise_std)
        pcx = aug_gaussian_noise(pcx, noise_std)

    # Channel dropout: apply independently (channels may differ between input/output)
    if config.get("aug_channel_dropout", False):
        dropout_p = config.get("aug_channel_dropout_p", 0.1)
        batch_size, n_channels_ob, _ = ob.shape
        _, n_channels_pcx, _ = pcx.shape
        mask_ob = torch.bernoulli(torch.full((batch_size, n_channels_ob, 1), 1 - dropout_p, device=ob.device))
        mask_pcx = torch.bernoulli(torch.full((batch_size, n_channels_pcx, 1), 1 - dropout_p, device=pcx.device))
        ob = ob * mask_ob
        pcx = pcx * mask_pcx

    # Amplitude scale: apply same scale to both (maintains relative amplitude)
    if config.get("aug_amplitude_scale", False):
        scale_range = config.get("aug_amplitude_scale_range", (0.8, 1.2))
        batch_size = ob.shape[0]
        min_scale, max_scale = scale_range
        scales = torch.empty(batch_size, 1, 1, device=ob.device).uniform_(min_scale, max_scale)
        ob = ob * scales
        pcx = pcx * scales

    # Time mask: apply same mask to both (maintains correspondence)
    if config.get("aug_time_mask", False):
        mask_ratio = config.get("aug_time_mask_ratio", 0.1)
        batch_size, n_ch_ob, time_len = ob.shape
        _, n_ch_pcx, _ = pcx.shape
        mask_len = int(time_len * mask_ratio)
        if mask_len > 0:
            # Create binary mask (1 = keep, 0 = zero out) - NO inplace operations
            mask_ob = torch.ones(batch_size, 1, time_len, device=ob.device, dtype=ob.dtype)
            mask_pcx = torch.ones(batch_size, 1, time_len, device=pcx.device, dtype=pcx.dtype)
            for i in range(batch_size):
                start = torch.randint(0, time_len - mask_len + 1, (1,)).item()
                mask_ob[i, :, start:start + mask_len] = 0
                mask_pcx[i, :, start:start + mask_len] = 0
            # Apply mask via multiplication (creates new tensor, no inplace)
            ob = ob * mask_ob
            pcx = pcx * mask_pcx

    # Mixup: blend random pairs (applied to both input and target coherently)
    if config.get("aug_mixup", False):
        alpha = config.get("aug_mixup_alpha", 0.4)
        ob, pcx = aug_mixup(ob, pcx, alpha)

    # Frequency masking: zero out random freq bands (same bands for both)
    if config.get("aug_freq_mask", False):
        max_bands = config.get("aug_freq_mask_max_bands", 2)
        max_width = config.get("aug_freq_mask_max_width", 10)
        batch_size, n_ch_ob, _ = ob.shape
        _, n_ch_pcx, _ = pcx.shape
        # Apply same frequency mask pattern to both tensors
        ob_fft = torch.fft.rfft(ob, dim=-1)
        pcx_fft = torch.fft.rfft(pcx, dim=-1)
        n_freqs = ob_fft.shape[-1]
        if n_freqs > max_width:
            # Create frequency mask (1 = keep, 0 = zero out) - NO inplace on FFT tensors
            freq_mask_ob = torch.ones(batch_size, 1, n_freqs, device=ob.device, dtype=ob_fft.dtype)
            freq_mask_pcx = torch.ones(batch_size, 1, n_freqs, device=pcx.device, dtype=pcx_fft.dtype)
            for i in range(batch_size):
                n_bands_to_mask = torch.randint(1, max_bands + 1, (1,)).item()
                for _ in range(n_bands_to_mask):
                    width = torch.randint(1, max_width + 1, (1,)).item()
                    start = torch.randint(0, n_freqs - width, (1,)).item()
                    freq_mask_ob[i, :, start:start + width] = 0
                    freq_mask_pcx[i, :, start:start + width] = 0
            # Apply mask via multiplication (no inplace operation on FFT tensors)
            ob_fft = ob_fft * freq_mask_ob
            pcx_fft = pcx_fft * freq_mask_pcx
            ob = torch.fft.irfft(ob_fft, n=ob.shape[-1], dim=-1)
            pcx = torch.fft.irfft(pcx_fft, n=pcx.shape[-1], dim=-1)

    # Session-specific augmentations (simulate cross-session variability)

    # Per-channel scaling: apply independently (simulates electrode drift)
    if config.get("aug_channel_scale", False):
        scale_range = config.get("aug_channel_scale_range", (0.7, 1.4))
        batch_size, n_channels_ob, _ = ob.shape
        _, n_channels_pcx, _ = pcx.shape
        min_scale, max_scale = scale_range
        # Per-channel scale for OB
        scales_ob = torch.empty(1, n_channels_ob, 1, device=ob.device).uniform_(min_scale, max_scale)
        scales_ob = scales_ob.expand(batch_size, -1, -1)
        ob = ob * scales_ob
        # Per-channel scale for PCx (different electrodes, different channel count)
        scales_pcx = torch.empty(1, n_channels_pcx, 1, device=pcx.device).uniform_(min_scale, max_scale)
        scales_pcx = scales_pcx.expand(batch_size, -1, -1)
        pcx = pcx * scales_pcx

    # DC offset: apply independently (baseline drift is independent per electrode)
    if config.get("aug_dc_offset", False):
        offset_range = config.get("aug_dc_offset_range", (-0.3, 0.3))
        ob = aug_dc_offset(ob, offset_range)
        pcx = aug_dc_offset(pcx, offset_range)

    # Covariance expansion augmentation: create synthetic sessions via random per-channel scale/shift
    # This helps the model learn session-invariant representations by exposing it to
    # more diverse covariance structures during training (from statistics literature)
    if config.get("use_cov_augment", False):
        cov_prob = config.get("cov_augment_prob", 0.5)
        if torch.rand(1).item() < cov_prob:
            batch_size, n_channels_ob, _ = ob.shape
            _, n_channels_pcx, _ = pcx.shape
            # Random per-channel scale (simulates different electrode impedances across sessions)
            scale_ob = torch.empty(batch_size, n_channels_ob, 1, device=ob.device).uniform_(0.8, 1.2)
            scale_pcx = torch.empty(batch_size, n_channels_pcx, 1, device=pcx.device).uniform_(0.8, 1.2)
            # Random per-channel shift relative to std (simulates baseline drift)
            shift_ob = torch.empty(batch_size, n_channels_ob, 1, device=ob.device).uniform_(-0.2, 0.2)
            shift_pcx = torch.empty(batch_size, n_channels_pcx, 1, device=pcx.device).uniform_(-0.2, 0.2)
            # Apply: x_aug = x * scale + shift * std(x)
            ob_std = ob.std(dim=-1, keepdim=True).clamp(min=1e-6)
            pcx_std = pcx.std(dim=-1, keepdim=True).clamp(min=1e-6)
            ob = ob * scale_ob + shift_ob * ob_std
            pcx = pcx * scale_pcx + shift_pcx * pcx_std

    return ob, pcx


# =============================================================================
# Contrastive Learning for Session-Invariant Representations
# =============================================================================

class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Projects encoder features to a lower-dimensional embedding space
    where contrastive loss is applied.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def info_nce_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """InfoNCE contrastive loss (CEBRA-style).

    Pulls together samples with the same label (odor), pushes apart different labels.
    This encourages the model to learn odor-specific representations that are
    invariant to session-specific noise (since different augmentations simulate
    different sessions).

    Args:
        embeddings: (batch, embedding_dim) normalized embeddings
        labels: (batch,) odor labels
        temperature: Temperature for softmax (lower = sharper)

    Returns:
        Scalar InfoNCE loss
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)

    batch_size = embeddings.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # Compute similarity matrix
    sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature

    # Create mask for positive pairs (same odor)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()).float()

    # Remove diagonal (self-similarity)
    eye_mask = torch.eye(batch_size, device=embeddings.device)
    pos_mask = pos_mask - eye_mask

    # Check if there are any positive pairs
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    # For numerical stability, subtract max
    sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()

    # Compute log-softmax over all pairs (excluding self)
    exp_sim = torch.exp(sim_matrix) * (1 - eye_mask)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Mean log-probability over positive pairs
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)

    # Only include samples that have at least one positive pair
    valid_mask = pos_mask.sum(dim=1) > 0
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    loss = -pos_log_prob[valid_mask].mean()

    return loss


def temporal_info_nce_loss(
    features: torch.Tensor,
    temperature: float = 0.1,
    time_delta: int = 10,
    num_samples: int = 32,
) -> torch.Tensor:
    """True CEBRA temporal contrastive loss.

    Forms positive pairs from time points that are close in time (within delta),
    and negative pairs from time points that are far apart.

    This is the core innovation of CEBRA: learning embeddings where temporally
    adjacent neural states map to similar embeddings, capturing the smooth
    temporal dynamics of neural activity.

    Args:
        features: (batch, channels, time) bottleneck features (NOT pooled)
        temperature: Temperature for softmax (lower = sharper)
        time_delta: Maximum time offset for positive pairs (in samples)
        num_samples: Number of time points to sample per trial

    Returns:
        Scalar InfoNCE loss
    """
    batch_size, n_channels, time_len = features.shape
    device = features.device

    if time_len < 2 * time_delta + 1:
        # Signal too short for temporal contrastive learning
        return torch.tensor(0.0, device=device)

    # Sample anchor time indices (avoid edges to allow positive sampling)
    # Valid range: [time_delta, time_len - time_delta - 1]
    valid_start = time_delta
    valid_end = time_len - time_delta
    valid_range = valid_end - valid_start

    if valid_range <= 0:
        return torch.tensor(0.0, device=device)

    # Limit number of samples to available range
    actual_num_samples = min(num_samples, valid_range)

    # Sample random anchor indices for each item in batch
    # Shape: (batch, num_samples)
    anchor_indices = torch.randint(
        valid_start, valid_end, (batch_size, actual_num_samples), device=device
    )

    # Sample positive indices (within time_delta of anchor)
    # For each anchor, sample a positive within [-time_delta, +time_delta] (excluding 0)
    pos_offsets = torch.randint(-time_delta, time_delta + 1, (batch_size, actual_num_samples), device=device)
    # Avoid sampling the anchor itself
    pos_offsets = torch.where(pos_offsets == 0, torch.ones_like(pos_offsets), pos_offsets)
    positive_indices = anchor_indices + pos_offsets

    # Clamp to valid range
    positive_indices = positive_indices.clamp(0, time_len - 1)

    # Extract features at anchor and positive positions
    # features: (batch, channels, time) -> need to gather at specific time indices
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, actual_num_samples)

    # Gather anchor features: (batch, num_samples, channels)
    anchor_feats = features.permute(0, 2, 1)[batch_idx, anchor_indices]  # (batch, num_samples, channels)
    positive_feats = features.permute(0, 2, 1)[batch_idx, positive_indices]  # (batch, num_samples, channels)

    # Flatten batch and samples: (batch * num_samples, channels)
    anchor_flat = anchor_feats.reshape(-1, n_channels)
    positive_flat = positive_feats.reshape(-1, n_channels)

    # Normalize embeddings
    anchor_flat = F.normalize(anchor_flat, dim=1)
    positive_flat = F.normalize(positive_flat, dim=1)

    n_total = anchor_flat.shape[0]
    if n_total < 2:
        return torch.tensor(0.0, device=device)

    # Compute similarity matrix: each anchor against all positives
    # Using all positives from all anchors as the candidate pool (contrastive negatives)
    sim_matrix = torch.mm(anchor_flat, positive_flat.t()) / temperature

    # The diagonal contains the positive pairs (anchor_i with positive_i)
    # All off-diagonal entries are negatives (from different anchors or different trials)

    # For numerical stability
    sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()

    # Create labels: each anchor should match its corresponding positive (diagonal)
    labels = torch.arange(n_total, device=device)

    # Cross-entropy loss (each anchor should be closest to its positive)
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


def get_encoder_features(
    model: nn.Module,
    x: torch.Tensor,
    cond: torch.Tensor
) -> torch.Tensor:
    """Extract bottleneck features from the UNet encoder.

    Runs the encoder part of the UNet and returns the bottleneck features
    (before the decoder). These features are used for contrastive learning.

    Args:
        model: CondUNet1D model
        x: Input tensor (batch, channels, time)
        cond: Conditioning tensor

    Returns:
        Bottleneck features (batch, feature_dim)
    """
    # Get conditioning embedding
    cond_embed = model.cond_encoder(cond)

    # Run through encoder blocks
    skips = []
    h = x
    for down in model.encoder:
        h = down(h, cond_embed)
        skips.append(h)

    # Bottleneck
    h = model.bottleneck(h, cond_embed)

    # Global average pool to get fixed-size features
    # h shape: (batch, channels, time)
    features = h.mean(dim=-1)  # (batch, channels)

    return features


def load_checkpoint(
    path: Path,
    model: nn.Module,
    reverse_model: Optional[nn.Module] = None,
    expected_cond_mode: Optional[str] = None,
) -> str:
    """Load checkpoint.

    Args:
        expected_cond_mode: If provided, validate checkpoint cond_mode matches.

    Returns:
        cond_mode from checkpoint (for verification)
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    # Check cond_mode compatibility (if saved in checkpoint)
    checkpoint_cond_mode = checkpoint.get("cond_mode", "unknown")
    if is_primary():
        print(f"  Checkpoint cond_mode: {checkpoint_cond_mode}")
    
    # Validate cond_mode if expected_cond_mode is provided
    if expected_cond_mode is not None and checkpoint_cond_mode != "unknown":
        if checkpoint_cond_mode != expected_cond_mode:
            raise ValueError(
                f"Checkpoint cond_mode mismatch!\n"
                f"  Checkpoint has: {checkpoint_cond_mode}\n"
                f"  Model expects:  {expected_cond_mode}\n"
                f"  The checkpoint was trained with a different conditioning architecture.\n"
                f"  For stage2 spectral shift experiments, use a checkpoint with matching cond_mode.\n"
                f"  Either:\n"
                f"    1. Set --cond-mode {checkpoint_cond_mode} to match the checkpoint, or\n"
                f"    2. Use a different checkpoint (set BEST_CHECKPOINT=/path/to/matching/checkpoint.pt)"
            )

    # Load UNet (forward) - handle FSDP
    # Wrap in inference_mode(False) for consistency with save_checkpoint
    if isinstance(model, FSDP):
        with torch.inference_mode(False):
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            ):
                model.load_state_dict(checkpoint["model"])
    else:
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])

    # Load reverse UNet - handle FSDP
    if reverse_model is not None and "reverse_model" in checkpoint:
        if isinstance(reverse_model, FSDP):
            with torch.inference_mode(False):
                with FSDP.state_dict_type(
                    reverse_model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                ):
                    reverse_model.load_state_dict(checkpoint["reverse_model"])
        else:
            if hasattr(reverse_model, 'module'):
                reverse_model.module.load_state_dict(checkpoint["reverse_model"])
            else:
                reverse_model.load_state_dict(checkpoint["reverse_model"])

    return checkpoint_cond_mode


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
    is_fsdp: bool = False,
    reverse_model: Optional[nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,  # Add data dict for split info
) -> None:
    """Save checkpoint with FSDP support (includes all models)."""
    if is_fsdp and isinstance(model, FSDP):
        # FSDP models need special handling - use context manager for each FSDP model
        # Ensure model is in train mode to avoid inference mode restrictions
        model.train()
        if reverse_model is not None:
            reverse_model.train()

        checkpoint = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            # Save model config for compatibility checking
            "cond_mode": config.get("cond_mode", "unknown") if config else "unknown",
        }

        # Save full config for validation
        if config is not None:
            checkpoint["config"] = config

        # Save split info for validation to use same sessions
        if data is not None:
            split_info = {}
            if "split_info" in data:
                split_info = data["split_info"].copy()
            if "val_idx" in data:
                split_info["val_idx"] = data["val_idx"].tolist() if hasattr(data["val_idx"], 'tolist') else list(data["val_idx"])
            if "test_idx" in data:
                split_info["test_idx"] = data["test_idx"].tolist() if hasattr(data["test_idx"], 'tolist') else list(data["test_idx"])
            if "train_idx" in data:
                split_info["train_idx"] = data["train_idx"].tolist() if hasattr(data["train_idx"], 'tolist') else list(data["train_idx"])
            checkpoint["split_info"] = split_info

        # Save forward model
        # Wrap in inference_mode(False) to avoid "Inplace update to inference tensor" errors
        # that can occur when FSDP gathers and offloads state dict to CPU
        with torch.inference_mode(False):
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                checkpoint["model"] = model.state_dict()

            # Save reverse model if exists (also FSDP)
            if reverse_model is not None and isinstance(reverse_model, FSDP):
                with FSDP.state_dict_type(
                    reverse_model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                ):
                    checkpoint["reverse_model"] = reverse_model.state_dict()

        # Save session matcher state for inference-time session matching
        if config is not None and "_session_matcher" in config:
            session_matcher = config["_session_matcher"]
            checkpoint["session_matcher_state"] = {
                "signatures": session_matcher.signatures,
                "session_names": session_matcher.session_names,
                "n_channels": session_matcher.n_channels,
                "n_warmup_samples": session_matcher.n_warmup_samples,
                "use_power_spectrum": session_matcher.use_power_spectrum,
                "sample_rate": session_matcher.sample_rate,
            }

        if is_primary():
            torch.save(checkpoint, path)
    else:
        if is_primary():
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            checkpoint = {
                "epoch": epoch,
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                # Save model config for compatibility checking
                "cond_mode": config.get("cond_mode", "unknown") if config else "unknown",
            }

            # Save full config for validation
            if config is not None:
                checkpoint["config"] = config

            # Save split info for validation to use same sessions
            if data is not None:
                split_info = {}
                if "split_info" in data:
                    split_info = data["split_info"].copy()
                if "val_idx" in data:
                    split_info["val_idx"] = data["val_idx"].tolist() if hasattr(data["val_idx"], 'tolist') else list(data["val_idx"])
                if "test_idx" in data:
                    split_info["test_idx"] = data["test_idx"].tolist() if hasattr(data["test_idx"], 'tolist') else list(data["test_idx"])
                if "train_idx" in data:
                    split_info["train_idx"] = data["train_idx"].tolist() if hasattr(data["train_idx"], 'tolist') else list(data["train_idx"])
                checkpoint["split_info"] = split_info

            # Save reverse model if exists
            if reverse_model is not None:
                rev_state = reverse_model.module.state_dict() if hasattr(reverse_model, 'module') else reverse_model.state_dict()
                checkpoint["reverse_model"] = rev_state

            # Save session matcher state for inference-time session matching
            if config is not None and "_session_matcher" in config:
                session_matcher = config["_session_matcher"]
                checkpoint["session_matcher_state"] = {
                    "signatures": session_matcher.signatures,
                    "session_names": session_matcher.session_names,
                    "n_channels": session_matcher.n_channels,
                    "n_warmup_samples": session_matcher.n_warmup_samples,
                    "use_power_spectrum": session_matcher.use_power_spectrum,
                    "sample_rate": session_matcher.sample_rate,
                }

            torch.save(checkpoint, path)


# =============================================================================
# Distributed Utilities
# =============================================================================

def dist_init_if_needed() -> None:
    """Initialize distributed training if running with torchrun."""
    if dist.is_available() and not dist.is_initialized() and "RANK" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=1800))


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_primary() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def sync_gradients_manual(model: nn.Module) -> None:
    """Manually synchronize gradients across all ranks using all_reduce.

    This is an alternative to DDP for small modules that have compatibility issues
    with DDP (like auto-conditioning encoders with auxiliary losses).

    Args:
        model: The model whose gradients should be synchronized
    """
    if not (dist.is_available() and dist.is_initialized()):
        return

    world_size = dist.get_world_size()
    if world_size == 1:
        return

    # All-reduce gradients across all ranks
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    rank = get_rank()
    effective = seed + rank
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)
    # benchmark=True enables cuDNN auto-tuning for FAST training
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    wavelet_loss: Optional[nn.Module] = None,
    compute_phase: bool = False,
    reverse_model: Optional[nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
    fast_mode: bool = True,  # Skip expensive metrics (PSD, phase, baseline) during training
    sampling_rate: int = SAMPLING_RATE_HZ,  # Sampling rate for PSD calculations
    cond_encoder: Optional[nn.Module] = None,
    envelope_matcher_fwd: Optional[nn.Module] = None,
    envelope_matcher_rev: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate model on a dataloader (supports bidirectional).

    Returns composite validation loss that mirrors training loss.

    Args:
        fast_mode: If True, skip expensive metrics (PSD, phase, baseline) for faster validation.
                   Use fast_mode=False only for final evaluation.
        cond_encoder: Optional conditioning encoder for auto-conditioning modes
    """
    model.eval()
    if reverse_model is not None:
        reverse_model.eval()
    if cond_encoder is not None:
        cond_encoder.eval()

    # Forward direction (OB→PCx)
    mse_list, mae_list, corr_list = [], [], []
    wavelet_list, plv_list, pli_list = [], [], []
    r2_list, nrmse_list = [], []
    psd_err_list, psd_diff_list = [], []

    # Reverse direction (PCx→OB)
    mse_list_rev, mae_list_rev, corr_list_rev = [], [], []
    wavelet_list_rev = []
    plv_list_rev, pli_list_rev = [], []
    r2_list_rev, nrmse_list_rev = [], []
    psd_err_list_rev, psd_diff_list_rev = [], []

    # Baseline: Raw OB vs PCx (natural difference between brain regions)
    # This provides context - how similar are the regions naturally?
    baseline_corr_list, baseline_r2_list, baseline_nrmse_list = [], [], []
    baseline_psd_err_list, baseline_psd_diff_list = [], []
    baseline_plv_list, baseline_pli_list = [], []

    # Per-channel metrics (for channel correspondence analysis)
    # Only computed in non-fast mode
    per_channel_corr_list = []  # List of [C] tensors
    cross_channel_corr_accumulated = None  # [C, C] accumulated cross-channel correlation

    # Determine compute dtype for FSDP mixed precision compatibility
    use_bf16 = config.get("fsdp_bf16", False) if config else False
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Debug: Track data statistics across batches
    debug_eval = config.get("debug_eval", False) if config else False
    batch_count = 0
    total_samples = 0
    ob_means, pcx_means = [], []
    ob_stds, pcx_stds = [], []
    raw_corrs = []

    # Use no_grad instead of inference_mode for FSDP compatibility
    # inference_mode marks tensors as "inference tensors" which breaks FSDP checkpoint saving
    with torch.no_grad():
        for batch in loader:
            # Handle both 3-tuple (legacy) and 4-tuple (with session_ids) formats
            if len(batch) == 4:
                ob, pcx, odor, session_ids_batch = batch
            else:
                ob, pcx, odor = batch
                session_ids_batch = odor  # Legacy fallback
            ob = ob.to(device, dtype=compute_dtype, non_blocking=True)
            pcx = pcx.to(device, dtype=compute_dtype, non_blocking=True)
            odor = odor.to(device, non_blocking=True)
            session_ids_batch = session_ids_batch.to(device, non_blocking=True)

            # Normalize target (PCx) for loss computation
            # NOTE: Input (OB) is normalized inside the model's forward()
            pcx = per_channel_normalize(pcx)

            # Compute conditioning embedding if using auto-conditioning
            cond_emb = None
            if cond_encoder is not None:
                cond_source = config.get("conditioning_source", "odor_onehot") if config else "odor_onehot"
                if cond_source == "spectro_temporal":
                    cond_emb = cond_encoder(ob)
                elif cond_source == "cpc":
                    cond_emb, _, _ = cond_encoder(ob)  # Ignore z_seq, context_seq at eval
                elif cond_source == "vqvae":
                    cond_emb, _ = cond_encoder(ob)
                elif cond_source == "freq_disentangled":
                    # FFT operations require float32, cast input if bfloat16
                    ob_fft = ob.float() if ob.dtype == torch.bfloat16 else ob
                    cond_emb, _ = cond_encoder(ob_fft)  # Ignore band_info at eval
                    if ob.dtype == torch.bfloat16:
                        cond_emb = cond_emb.to(torch.bfloat16)
                elif cond_source == "cycle_consistent":
                    # At eval time, only use input (no target needed for embedding)
                    cond_emb, _ = cond_encoder(ob, None)

            # Forward: OB → PCx
            # For CondUNet: use cond_emb if available, otherwise odor_ids
            # For other architectures: just pass input directly
            arch = config.get("arch", "condunet") if config else "condunet"
            # Use proper session_ids from dataloader (only needed for learnable session embedding)
            session_ids = session_ids_batch if config and config.get("use_session_embedding", False) else None
            if arch == "condunet":
                if cond_emb is not None:
                    pred = model(ob, cond_emb=cond_emb, session_ids=session_ids)
                else:
                    pred = model(ob, odor, session_ids=session_ids)
            else:
                # Other architectures: just pass input
                pred = model(ob)

            # Apply envelope histogram matching (closed-form correction for amplitude dynamics)
            # This corrects bursty vs smooth characteristics
            if envelope_matcher_fwd is not None:
                pred = envelope_matcher_fwd(pred, odor_ids=odor)

            pred_c = crop_to_target_torch(pred)
            pcx_c = crop_to_target_torch(pcx)
            ob_c = crop_to_target_torch(ob)

            with torch.amp.autocast(device_type='cuda', enabled=False):
                pred_f32 = pred_c.float()
                pcx_f32 = pcx_c.float()
                ob_f32 = ob_c.float()

                mse_list.append(F.mse_loss(pred_f32, pcx_f32).item())
                mae_list.append(F.l1_loss(pred_f32, pcx_f32).item())
                corr_list.append(pearson_batch(pred_f32, pcx_f32).item())
                r2_list.append(explained_variance_torch(pred_f32, pcx_f32).item())
                # Skip nrmse in fast_mode (redundant with mse/r2)
                if not fast_mode:
                    nrmse_list.append(normalized_rmse_torch(pred_f32, pcx_f32).item())

            if wavelet_loss is not None:
                wavelet_list.append(wavelet_loss(pred_f32, pcx_f32).item())

            # Skip expensive phase metrics in fast_mode
            if compute_phase and not fast_mode:
                plv_list.append(plv_torch(pred_f32, pcx_f32).item())
                pli_list.append(pli_torch(pred_f32, pcx_f32).item())

            # Skip expensive PSD metrics in fast_mode
            if not fast_mode:
                psd_err_list.append(psd_error_db_torch(pred_f32, pcx_f32, fs=sampling_rate).item())
                psd_diff_list.append(psd_diff_db_torch(pred_f32, pcx_f32, fs=sampling_rate).item())

            # Baseline metrics: Compare raw source vs target
            # For different channel counts (e.g., PFC 64ch → CA1 32ch), use mean across channels
            # This gives a meaningful "how similar are these brain regions" baseline
            if not fast_mode:
                if ob_f32.shape[1] == pcx_f32.shape[1]:
                    # Same channel count: direct comparison
                    ob_baseline = ob_f32
                    pcx_baseline = pcx_f32
                else:
                    # Different channel counts: compare mean signals (reduces to [B, 1, T])
                    ob_baseline = ob_f32.mean(dim=1, keepdim=True)
                    pcx_baseline = pcx_f32.mean(dim=1, keepdim=True)

                baseline_corr_list.append(pearson_batch(ob_baseline, pcx_baseline).item())
                baseline_r2_list.append(explained_variance_torch(ob_baseline, pcx_baseline).item())
                baseline_nrmse_list.append(normalized_rmse_torch(ob_baseline, pcx_baseline).item())
                baseline_psd_err_list.append(psd_error_db_torch(ob_baseline, pcx_baseline, fs=sampling_rate).item())
                baseline_psd_diff_list.append(psd_diff_db_torch(ob_baseline, pcx_baseline, fs=sampling_rate).item())
                if compute_phase:
                    baseline_plv_list.append(plv_torch(ob_baseline, pcx_baseline).item())
                    baseline_pli_list.append(pli_torch(ob_baseline, pcx_baseline).item())

                # Per-channel correlation analysis (for channel correspondence investigation)
                per_ch_corr = pearson_per_channel(pred_f32, pcx_f32)  # [C]
                per_channel_corr_list.append(per_ch_corr.cpu())

                # Cross-channel correlation matrix (which pred channels correlate with which target channels)
                cross_ch_corr = cross_channel_correlation(pred_f32, pcx_f32)  # [C, C]
                if cross_channel_corr_accumulated is None:
                    cross_channel_corr_accumulated = cross_ch_corr.cpu()
                else:
                    cross_channel_corr_accumulated += cross_ch_corr.cpu()

            # Reverse: PCx → OB (if reverse model exists)
            if reverse_model is not None:
                if cond_emb is not None:
                    pred_rev = reverse_model(pcx, cond_emb=cond_emb, session_ids=session_ids)
                else:
                    pred_rev = reverse_model(pcx, odor, session_ids=session_ids)

                # Apply envelope histogram matching (reverse direction)
                if envelope_matcher_rev is not None:
                    pred_rev = envelope_matcher_rev(pred_rev, odor_ids=odor)

                pred_rev_c = crop_to_target_torch(pred_rev)

                with torch.amp.autocast(device_type='cuda', enabled=False):
                    pred_rev_f32 = pred_rev_c.float()

                    mse_list_rev.append(F.mse_loss(pred_rev_f32, ob_f32).item())
                    mae_list_rev.append(F.l1_loss(pred_rev_f32, ob_f32).item())
                    corr_list_rev.append(pearson_batch(pred_rev_f32, ob_f32).item())
                    r2_list_rev.append(explained_variance_torch(pred_rev_f32, ob_f32).item())
                    if not fast_mode:
                        nrmse_list_rev.append(normalized_rmse_torch(pred_rev_f32, ob_f32).item())

                if wavelet_loss is not None:
                    wavelet_list_rev.append(wavelet_loss(pred_rev_f32, ob_f32).item())

                # Skip expensive phase metrics in fast_mode
                if compute_phase and not fast_mode:
                    plv_list_rev.append(plv_torch(pred_rev_f32, ob_f32).item())
                    pli_list_rev.append(pli_torch(pred_rev_f32, ob_f32).item())

                # Skip expensive PSD metrics in fast_mode
                if not fast_mode:
                    psd_err_list_rev.append(psd_error_db_torch(pred_rev_f32, ob_f32, fs=sampling_rate).item())
                    psd_diff_list_rev.append(psd_diff_db_torch(pred_rev_f32, ob_f32, fs=sampling_rate).item())

    # Forward results
    if not mse_list:
        # No batches processed (can happen with small datasets and many GPUs)
        return {
            "mse": 0.0,
            "mae": 0.0,
            "corr": 0.0,
            "r2": 0.0,
        }
    results = {
        "mse": float(np.mean(mse_list)),
        "mae": float(np.mean(mae_list)),
        "corr": float(np.mean(corr_list)),
        "r2": float(np.mean(r2_list)),
    }
    
    if nrmse_list:
        results["nrmse"] = float(np.mean(nrmse_list))
    if psd_err_list:
        results["psd_err_db"] = float(np.mean(psd_err_list))
    if psd_diff_list:
        results["psd_diff_db"] = float(np.mean(psd_diff_list))
    if wavelet_list:
        results["wavelet"] = float(np.mean(wavelet_list))
    if plv_list:
        results["plv"] = float(np.mean(plv_list))
    if pli_list:
        results["pli"] = float(np.mean(pli_list))

    # Reverse results (PCx→OB)
    if mse_list_rev:
        results["mse_rev"] = float(np.mean(mse_list_rev))
        results["mae_rev"] = float(np.mean(mae_list_rev))
        results["corr_rev"] = float(np.mean(corr_list_rev))
        results["r2_rev"] = float(np.mean(r2_list_rev))
    if nrmse_list_rev:
        results["nrmse_rev"] = float(np.mean(nrmse_list_rev))
    if psd_err_list_rev:
        results["psd_err_db_rev"] = float(np.mean(psd_err_list_rev))
    if psd_diff_list_rev:
        results["psd_diff_db_rev"] = float(np.mean(psd_diff_list_rev))
    if wavelet_list_rev:
        results["wavelet_rev"] = float(np.mean(wavelet_list_rev))
    if plv_list_rev:
        results["plv_rev"] = float(np.mean(plv_list_rev))
    if pli_list_rev:
        results["pli_rev"] = float(np.mean(pli_list_rev))

    # Baseline results: Raw OB vs PCx (only computed when not in fast_mode)
    if baseline_corr_list:
        results["baseline_corr"] = float(np.mean(baseline_corr_list))
        results["baseline_r2"] = float(np.mean(baseline_r2_list))
        results["baseline_nrmse"] = float(np.mean(baseline_nrmse_list))
        results["baseline_psd_err_db"] = float(np.mean(baseline_psd_err_list))
        results["baseline_psd_diff_db"] = float(np.mean(baseline_psd_diff_list))
    if baseline_plv_list:
        results["baseline_plv"] = float(np.mean(baseline_plv_list))
    if baseline_pli_list:
        results["baseline_pli"] = float(np.mean(baseline_pli_list))

    # Per-channel correlation metrics (for channel correspondence analysis)
    if per_channel_corr_list:
        # Stack and average per-channel correlations across batches
        per_ch_corr_stacked = torch.stack(per_channel_corr_list)  # [num_batches, C]
        per_ch_corr_mean = per_ch_corr_stacked.mean(dim=0)  # [C]

        results["per_channel_corr"] = per_ch_corr_mean.tolist()  # List of per-channel correlations
        results["per_channel_corr_std"] = per_ch_corr_stacked.std(dim=0).tolist()
        results["per_channel_corr_min"] = float(per_ch_corr_mean.min())
        results["per_channel_corr_max"] = float(per_ch_corr_mean.max())
        results["channel_corr_range"] = float(per_ch_corr_mean.max() - per_ch_corr_mean.min())

        # Diagonal dominance: how much does channel i predict channel i vs others?
        if cross_channel_corr_accumulated is not None:
            n_batches = len(per_channel_corr_list)
            cross_corr_mean = cross_channel_corr_accumulated / n_batches  # [C, C]
            results["cross_channel_corr_matrix"] = cross_corr_mean.tolist()

            # Diagonal = correspondence assumption (ch i -> ch i)
            diag = torch.diag(cross_corr_mean)
            # Off-diagonal mean = cross-channel leakage
            off_diag_mask = ~torch.eye(cross_corr_mean.size(0), dtype=torch.bool)
            off_diag = cross_corr_mean[off_diag_mask]

            results["diagonal_corr_mean"] = float(diag.mean())
            results["off_diagonal_corr_mean"] = float(off_diag.mean())
            results["diagonal_dominance"] = float(diag.mean() - off_diag.mean())

    # Compute composite validation loss (mirrors training loss)
    # This allows early stopping based on overall objective, not just correlation
    if config is not None:
        # L1 + wavelet
        w_l1 = config.get("weight_l1", 1.0)
        w_wav = config.get("weight_wavelet", 1.0) if config.get("use_wavelet_loss", True) else 0.0

        # Forward loss
        val_loss = w_l1 * results["mae"]
        if "wavelet" in results:
            val_loss += w_wav * results["wavelet"]

        # Reverse loss (if bidirectional)
        if "mae_rev" in results:
            val_loss += w_l1 * results["mae_rev"]
            if "wavelet_rev" in results:
                val_loss += w_wav * results["wavelet_rev"]

        results["loss"] = val_loss

    return results


# =============================================================================
# Training
# =============================================================================

def freeze_model_params(model: nn.Module) -> None:
    """Freeze all parameters in a model (UNet)."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model_params(model: nn.Module) -> None:
    """Unfreeze all parameters in a model (UNet)."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    """Count total parameters in a model (regardless of requires_grad)."""
    return sum(p.numel() for p in model.parameters())


def get_module(model: nn.Module) -> nn.Module:
    """Get underlying module from DDP/FSDP wrapper if present."""
    return model.module if hasattr(model, 'module') else model


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict[str, Any],
    wavelet_loss: Optional[nn.Module] = None,
    reverse_model: Optional[nn.Module] = None,
    epoch: int = 0,
    num_epochs: int = 0,
    cond_encoder: Optional[nn.Module] = None,
    projection_head_fwd: Optional[nn.Module] = None,
    projection_head_rev: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Train one epoch (supports bidirectional with cycle consistency).

    Args:
        cond_encoder: Optional conditioning encoder for auto-conditioning modes
    """
    model.train()
    if reverse_model is not None:
        reverse_model.train()
    if cond_encoder is not None:
        cond_encoder.train()
    if projection_head_fwd is not None:
        projection_head_fwd.train()
    if projection_head_rev is not None:
        projection_head_rev.train()

    # Use tensors for accumulation to avoid GPU-CPU sync during training
    # Only convert to floats at end of epoch for logging
    total_loss = torch.tensor(0.0, device=device)
    loss_components = defaultdict(lambda: torch.tensor(0.0, device=device))
    optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{num_epochs}",
        leave=True,
        position=0,
        ncols=100,
        disable=not is_primary(),
        file=sys.stdout,
    )
    # Determine compute dtype for FSDP mixed precision compatibility
    use_bf16 = config.get("fsdp_bf16", False)
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float32

    for batch_idx, batch in enumerate(pbar):
        # Handle both 3-tuple (legacy) and 4-tuple (with session_ids) formats
        if len(batch) == 4:
            ob, pcx, odor, session_ids_batch = batch
        else:
            ob, pcx, odor = batch
            session_ids_batch = odor  # Legacy fallback
        # non_blocking=True enables async CPU->GPU transfer (overlaps with compute)
        ob = ob.to(device, dtype=compute_dtype, non_blocking=True)
        pcx = pcx.to(device, dtype=compute_dtype, non_blocking=True)
        odor = odor.to(device, non_blocking=True)
        session_ids_batch = session_ids_batch.to(device, non_blocking=True)

        # Normalize target (PCx) for loss computation
        # NOTE: Input (OB) is normalized inside the model's forward()
        pcx = per_channel_normalize(pcx)

        # Apply data augmentation (training only)
        ob, pcx = apply_augmentations(ob, pcx, config)

        # IMPORTANT: Clone tensors after augmentation to avoid autograd inplace errors
        # This ensures ob/pcx are fresh tensors with no shared memory when used in
        # both cond_encoder(ob) and model(ob, ...) - prevents version tracking conflicts
        ob = ob.clone()
        pcx = pcx.clone()

        # Compute conditioning embedding
        # IMPORTANT: Set cond_encoder to eval mode during forward pass to prevent
        # BatchNorm running stats from being updated in-place. This avoids version
        # tracking conflicts during backward() when cond_emb is used in multiple
        # model calls. Gradients still flow for weight/bias training.
        cond_emb = None
        cond_loss = 0.0
        if cond_encoder is not None:
            cond_encoder_training = cond_encoder.training
            cond_encoder.eval()  # Prevent BatchNorm running stats updates
            cond_source = config.get("conditioning_source", "odor_onehot")
            if cond_source == "spectro_temporal":
                # SpectroTemporalEncoder: signal -> embedding
                cond_emb = cond_encoder(ob)
            elif cond_source == "cpc":
                # CPCEncoder: returns (embedding, z_seq, context_seq)
                cond_emb, z_seq, context_seq = cond_encoder(ob)
                # CPC InfoNCE contrastive loss - learns predictive representations
                cond_loss = 0.1 * cond_encoder.cpc_loss(z_seq, context_seq)
            elif cond_source == "vqvae":
                # VQVAEEncoder: returns (embedding, losses_dict)
                cond_emb, vq_losses = cond_encoder(ob)
                # Add VQ-VAE auxiliary losses (vq + commitment + reconstruction)
                cond_loss = (vq_losses["vq_loss"] +
                            0.25 * vq_losses["commitment_loss"] +
                            0.1 * vq_losses["recon_loss"])
            elif cond_source == "freq_disentangled":
                # FreqDisentangledEncoder: signal -> (embedding, band_info)
                # IMPORTANT: FFT operations require float32, cast input if bfloat16
                ob_fft = ob.float() if ob.dtype == torch.bfloat16 else ob
                cond_emb, band_info = cond_encoder(ob_fft)
                # Cast output back to compute dtype for consistency
                if ob.dtype == torch.bfloat16:
                    cond_emb = cond_emb.to(torch.bfloat16)
                # Add band power reconstruction loss (ensures meaningful embeddings)
                cond_loss = 0.1 * cond_encoder.recon_loss(band_info)
            elif cond_source == "cycle_consistent":
                # CycleConsistentEncoder: (input, target) -> (embedding, losses_dict)
                cond_emb, cycle_losses = cond_encoder(ob, pcx)
                # Add cycle consistency + reconstruction losses
                cond_loss = 0.0
                if "cycle_loss" in cycle_losses:
                    cond_loss = cond_loss + 0.1 * cycle_losses["cycle_loss"]
                if "recon_loss" in cycle_losses:
                    cond_loss = cond_loss + 0.1 * cycle_losses["recon_loss"]

            # Restore cond_encoder training mode (gradients still flow for training)
            if cond_encoder_training:
                cond_encoder.train()

        # Forward: OB → PCx
        # For CondUNet: use cond_emb if available, otherwise odor_ids
        # For other architectures: just pass input directly
        arch = config.get("arch", "condunet")
        contrastive_mode = config.get("contrastive_mode", "temporal")  # "temporal" (true CEBRA) or "label"
        use_contrastive = config.get("use_contrastive", False)
        # For label mode, require projection heads; for temporal mode, no projection heads needed
        if contrastive_mode == "label" and projection_head_fwd is None:
            use_contrastive = False
        use_temporal_cebra = use_contrastive and contrastive_mode == "temporal"

        if arch == "condunet":
            # CondUNet with conditioning
            # Use proper session_ids from dataloader (only needed for learnable session embedding)
            session_ids = session_ids_batch if config.get("use_session_embedding", False) else None
            if cond_emb is not None:
                fwd_result = model(
                    ob, cond_emb=cond_emb,
                    session_ids=session_ids,
                    return_bottleneck=(use_contrastive and not use_temporal_cebra),
                    return_bottleneck_temporal=use_temporal_cebra
                )
            else:
                fwd_result = model(
                    ob, odor,
                    session_ids=session_ids,
                    return_bottleneck=(use_contrastive and not use_temporal_cebra),
                    return_bottleneck_temporal=use_temporal_cebra
                )
        else:
            # Other architectures: just pass input
            fwd_result = model(ob)

        if use_contrastive and arch == "condunet":
            pred_raw, bottleneck_fwd = fwd_result
        else:
            pred_raw = fwd_result
            bottleneck_fwd = None

        pred_raw_c = crop_to_target_torch(pred_raw)
        pcx_c = crop_to_target_torch(pcx)
        ob_c = crop_to_target_torch(ob)

        # L1/Huber + wavelet loss
        loss_type = config.get("loss_type", "huber_wavelet")
        if loss_type in ("huber", "huber_wavelet"):
            recon_loss = config["weight_l1"] * F.huber_loss(pred_raw_c, pcx_c)
        else:
            recon_loss = config["weight_l1"] * F.l1_loss(pred_raw_c, pcx_c)
        loss = recon_loss
        loss_components["l1_fwd"] = loss_components["l1_fwd"] + recon_loss.detach()

        # Wavelet loss (forward)
        if config.get("use_wavelet_loss", True) and wavelet_loss is not None:
            w_loss = config["weight_wavelet"] * wavelet_loss(pred_raw_c, pcx_c)
            loss = loss + w_loss
            loss_components["wavelet_fwd"] = loss_components["wavelet_fwd"] + w_loss.detach()

        # Add conditioning encoder auxiliary loss if present
        if cond_loss != 0.0:
            loss = loss + cond_loss
            loss_components["cond_loss"] = loss_components["cond_loss"] + cond_loss.detach() if isinstance(cond_loss, torch.Tensor) else loss_components["cond_loss"] + cond_loss

        # Bidirectional training with cycle consistency
        bottleneck_rev = None  # Initialize for contrastive loss check later
        if reverse_model is not None:
            # Reverse: PCx → OB
            # IMPORTANT: Detach cond_emb for reverse_model to prevent computation graph
            # interconnection that causes BatchNorm version tracking conflicts.
            # The conditioning encoder still receives gradients from the forward model,
            # which provides sufficient supervision for learning.
            cond_emb_rev = cond_emb.detach() if cond_emb is not None else None
            # If contrastive learning enabled, also get bottleneck features
            if cond_emb_rev is not None:
                rev_result = reverse_model(
                    pcx, cond_emb=cond_emb_rev,
                    session_ids=session_ids,
                    return_bottleneck=(use_contrastive and not use_temporal_cebra),
                    return_bottleneck_temporal=use_temporal_cebra
                )
            else:
                rev_result = reverse_model(
                    pcx, odor,
                    session_ids=session_ids,
                    return_bottleneck=(use_contrastive and not use_temporal_cebra),
                    return_bottleneck_temporal=use_temporal_cebra
                )

            if use_contrastive:
                pred_rev_raw, bottleneck_rev = rev_result
            else:
                pred_rev_raw = rev_result

            pred_rev_raw_c = crop_to_target_torch(pred_rev_raw)

            # L1/Huber + wavelet (reverse)
            loss_type = config.get("loss_type", "huber_wavelet")
            if loss_type in ("huber", "huber_wavelet"):
                rev_loss = config["weight_l1"] * F.huber_loss(pred_rev_raw_c, ob_c)
            else:
                rev_loss = config["weight_l1"] * F.l1_loss(pred_rev_raw_c, ob_c)
            loss = loss + rev_loss
            loss_components["l1_rev"] = loss_components["l1_rev"] + rev_loss.detach()

            # Wavelet loss (reverse)
            if config.get("use_wavelet_loss", True) and wavelet_loss is not None:
                w_loss_rev = config["weight_wavelet"] * wavelet_loss(pred_rev_raw_c, ob_c)
                loss = loss + w_loss_rev
                loss_components["wavelet_rev"] = loss_components["wavelet_rev"] + w_loss_rev.detach()

            # Cycle consistency: OB → PCx → OB and PCx → OB → PCx
            # Models are already in eval mode (set at batch start) to prevent BatchNorm
            # running stats updates. Detached inputs break gradient flow back to main
            # predictions, but gradients DO flow through cycle_ob/cycle_pcx.

            # Cycle consistency: OB → PCx → OB
            if cond_emb_rev is not None:
                cycle_ob = reverse_model(pred_raw.detach(), cond_emb=cond_emb_rev, session_ids=session_ids)
            else:
                cycle_ob = reverse_model(pred_raw.detach(), odor, session_ids=session_ids)
            cycle_ob_c = crop_to_target_torch(cycle_ob)
            cycle_loss_ob = config.get("cycle_lambda", 1.0) * F.l1_loss(cycle_ob_c, ob_c)
            loss = loss + cycle_loss_ob
            loss_components["cycle_ob"] = loss_components["cycle_ob"] + cycle_loss_ob.detach()

            # Cycle consistency: PCx → OB → PCx
            if cond_emb_rev is not None:
                cycle_pcx = model(pred_rev_raw.detach(), cond_emb=cond_emb_rev)
            else:
                cycle_pcx = model(pred_rev_raw.detach(), odor)
            cycle_pcx_c = crop_to_target_torch(cycle_pcx)
            cycle_loss_pcx = config.get("cycle_lambda", 1.0) * F.l1_loss(cycle_pcx_c, pcx_c)
            loss = loss + cycle_loss_pcx
            loss_components["cycle_pcx"] = loss_components["cycle_pcx"] + cycle_loss_pcx.detach()

        # Contrastive loss for session-invariant learning (CEBRA-style)
        # Two modes:
        # 1. "temporal" (True CEBRA): positive pairs from nearby time points
        # 2. "label": positive pairs from same odor label
        if use_contrastive and bottleneck_fwd is not None:
            contrastive_weight = config.get("contrastive_weight", 0.1)
            contrastive_temp = config.get("contrastive_temperature", 0.1)
            time_delta = config.get("contrastive_time_delta", 10)
            num_samples = config.get("contrastive_num_samples", 32)

            # Debug: Print shapes and stats on first batch of first epoch
            if is_primary() and batch_idx == 0 and epoch == 1:
                print(f"\n{'='*60}")
                print(f"CEBRA CONTRASTIVE LEARNING - First Batch Debug")
                print(f"{'='*60}")
                print(f"  Mode: {contrastive_mode}")
                print(f"  Bottleneck features shape: {bottleneck_fwd.shape}")
                print(f"  Bottleneck dtype: {bottleneck_fwd.dtype}")
                print(f"  Bottleneck stats: mean={bottleneck_fwd.float().mean().item():.4f}, std={bottleneck_fwd.float().std().item():.4f}")

            if use_temporal_cebra:
                # TEMPORAL CEBRA: Positive pairs from time points within delta
                # bottleneck_fwd: (batch, channels, time) - unpooled temporal features
                if is_primary() and batch_idx == 0 and epoch == 1:
                    print(f"  Time delta: {time_delta} samples")
                    print(f"  Num samples per trial: {num_samples}")
                    if bottleneck_fwd.ndim == 3:
                        print(f"  Temporal extent: {bottleneck_fwd.shape[-1]} time steps")

                # Cast to float32 for stable loss computation
                contrastive_loss_fwd = temporal_info_nce_loss(
                    bottleneck_fwd.float(),
                    temperature=contrastive_temp,
                    time_delta=time_delta,
                    num_samples=num_samples
                )
            else:
                # LABEL-BASED: Positive pairs from same odor
                # bottleneck_fwd: (batch, channels) - pooled features
                fwd_embed = projection_head_fwd(bottleneck_fwd)  # -> (batch, 128)

                if is_primary() and batch_idx == 0 and epoch == 1:
                    print(f"  Projected embedding shape: {fwd_embed.shape}")
                    print(f"  Embedding stats: mean={fwd_embed.float().mean().item():.4f}, std={fwd_embed.float().std().item():.4f}")
                    print(f"  Odor labels in batch: {odor.tolist()}")
                    print(f"  Unique odors: {odor.unique().tolist()} (n={odor.unique().numel()})")
                    # Count positive pairs
                    labels = odor.view(-1, 1)
                    pos_mask = (labels == labels.t()).float()
                    n_pos_pairs = (pos_mask.sum() - pos_mask.shape[0]).item()  # Exclude diagonal
                    print(f"  Positive pairs in batch: {int(n_pos_pairs)} (same odor, different sample)")

                # Cast to float32 for stable loss computation
                contrastive_loss_fwd = info_nce_loss(fwd_embed.float(), odor, temperature=contrastive_temp)

            loss = loss + contrastive_weight * contrastive_loss_fwd
            loss_components["contrastive_fwd"] = loss_components["contrastive_fwd"] + contrastive_loss_fwd.detach()

            # Debug: Print loss on first batch of first epoch
            if is_primary() and batch_idx == 0 and epoch == 1:
                print(f"  Contrastive loss (fwd): {contrastive_loss_fwd.item():.4f}")
                print(f"  Weighted contrastive (fwd): {(contrastive_weight * contrastive_loss_fwd).item():.4f}")

            # Periodic print every 10 batches on first epoch
            if is_primary() and epoch == 1 and batch_idx > 0 and batch_idx % 10 == 0:
                print(f"  [Batch {batch_idx}] Contrastive fwd: {contrastive_loss_fwd.item():.4f}")

            # Reverse direction contrastive loss (on reverse model's bottleneck)
            if reverse_model is not None and bottleneck_rev is not None:
                if use_temporal_cebra:
                    contrastive_loss_rev = temporal_info_nce_loss(
                        bottleneck_rev.float(),
                        temperature=contrastive_temp,
                        time_delta=time_delta,
                        num_samples=num_samples
                    )
                elif projection_head_rev is not None:
                    rev_embed = projection_head_rev(bottleneck_rev)
                    contrastive_loss_rev = info_nce_loss(rev_embed.float(), odor, temperature=contrastive_temp)
                else:
                    contrastive_loss_rev = torch.tensor(0.0, device=device)

                loss = loss + contrastive_weight * contrastive_loss_rev
                loss_components["contrastive_rev"] = loss_components["contrastive_rev"] + contrastive_loss_rev.detach()

                if is_primary() and batch_idx == 0 and epoch == 1:
                    print(f"  Contrastive loss (rev): {contrastive_loss_rev.item():.4f}")
                    print(f"  Weighted contrastive (rev): {(contrastive_weight * contrastive_loss_rev).item():.4f}")

            if is_primary() and batch_idx == 0 and epoch == 1:
                print(f"  Total loss (with contrastive): {loss.item():.4f}")
                print(f"{'='*60}\n")

        # NaN detection before backward to prevent silent failures
        # Local check is cheap; if NaN detected, raise immediately
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}. Training aborted.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if reverse_model is not None:
            torch.nn.utils.clip_grad_norm_(reverse_model.parameters(), GRAD_CLIP)
        if cond_encoder is not None:
            torch.nn.utils.clip_grad_norm_(cond_encoder.parameters(), GRAD_CLIP)
        if projection_head_fwd is not None:
            torch.nn.utils.clip_grad_norm_(projection_head_fwd.parameters(), GRAD_CLIP)
        if projection_head_rev is not None:
            torch.nn.utils.clip_grad_norm_(projection_head_rev.parameters(), GRAD_CLIP)
            # Manual gradient sync for cond_encoder (not wrapped in DDP)
            sync_gradients_manual(cond_encoder)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Accumulate loss as tensor - NO .item() call to avoid GPU sync
        total_loss = total_loss + loss.detach()

    pbar.close()
    sys.stdout.flush()

    # Convert to floats ONLY at end of epoch (single GPU sync point)
    n_batches = len(loader)
    if n_batches == 0:
        # No batches processed (can happen with small datasets and many GPUs)
        return {
            "loss": 0.0,
            **{k: 0.0 for k in loss_components.keys()},
        }
    total_loss_val = total_loss.item() / n_batches
    return {
        "loss": total_loss_val,
        **{k: v.item() / n_batches for k, v in loss_components.items()},
    }


def train(
    config: Dict[str, Any],
    data: Dict[str, Any],
    use_fsdp: bool = False,
    fsdp_strategy: str = "full",
    cpu_offload: bool = False,
    compile_model: bool = False,
) -> Dict[str, Any]:
    """Main training function."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    set_seed(config.get("seed", 42))
    is_distributed = get_world_size() > 1

    # =========================================================================
    # STAGE 2 AUTO-DETECTION: Override cond_mode from checkpoint if stage2_only
    # =========================================================================
    stage2_only = config.get("stage2_only", False)
    if stage2_only:
        stage1_checkpoint = config.get("stage1_checkpoint", None)
        if stage1_checkpoint:
            ckpt_path = Path(stage1_checkpoint)
            # Smart path resolution - check common locations
            if not ckpt_path.exists():
                for candidate in [
                    CHECKPOINT_DIR / ckpt_path.name,
                    CHECKPOINT_DIR / ckpt_path,
                    Path("artifacts/checkpoints") / ckpt_path.name,
                ]:
                    if candidate.exists():
                        ckpt_path = candidate
                        break
            if ckpt_path.exists():
                checkpoint_cond_mode = get_checkpoint_cond_mode(ckpt_path)
                config_cond_mode = config.get("cond_mode", "film")
                if checkpoint_cond_mode is not None and checkpoint_cond_mode != config_cond_mode:
                    if is_primary():
                        print(f"\n{'='*70}")
                        print("AUTO-CORRECTING cond_mode FOR STAGE 2")
                        print(f"{'='*70}")
                        print(f"  Config specified:   {config_cond_mode}")
                        print(f"  Checkpoint has:     {checkpoint_cond_mode}")
                        print(f"  Using checkpoint's cond_mode: {checkpoint_cond_mode}")
                        print(f"{'='*70}\n")
                    config["cond_mode"] = checkpoint_cond_mode

    # Create dataloaders
    # Scale num_workers based on available CPUs (but cap to avoid overhead)
    import multiprocessing
    num_cpus = multiprocessing.cpu_count()
    num_workers = min(8, num_cpus // max(1, get_world_size()))  # Per-GPU workers

    # Create dataloaders based on dataset type
    if config.get("dataset_type") == "pcx1":
        # PCx1 already has loaders created
        loaders = {
            "train": data["train_loader"],
            "val": data["val_loader"],
        }
        # Add per-session val loaders if available
        if data.get("val_sessions_loaders"):
            loaders["val_sessions"] = data["val_sessions_loaders"]
        if is_primary():
            dataset_name = config.get("dataset_type", "").upper()
            print(f"{dataset_name} DataLoaders: {len(loaders['train'].dataset)} train windows, "
                  f"{len(loaders['val'].dataset)} val windows")
            if "val_sessions" in loaders:
                for sess_name, sess_loader in loaders["val_sessions"].items():
                    print(f"  Val session {sess_name}: {len(sess_loader.dataset)} windows")
    elif config.get("dataset_type") == "pfc":
        if config.get("pfc_sliding_window", False):
            # Use sliding window dataloaders for more training samples
            loaders = create_pfc_sliding_window_dataloaders(
                data,
                window_size=config.get("pfc_window_size", 2500),
                stride=config.get("pfc_stride"),
                val_stride=config.get("pfc_val_stride"),
                batch_size=config.get("batch_size", 16),
                num_workers=num_workers,
                use_sessions=config.get("split_by_session", False),
                distributed=is_distributed,
            )
            if is_primary():
                print(f"PFC sliding window DataLoaders: {len(loaders['train'].dataset)} train windows, "
                      f"{len(loaders['val'].dataset)} val windows")
        else:
            loaders = create_pfc_dataloaders(
                data,
                batch_size=config.get("batch_size", 16),
                num_workers=num_workers,
                distributed=is_distributed,
            )
    elif config.get("dataset_type") == "dandi":
        # DANDI uses pre-created datasets from prepare_dandi_data
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        dandi_datasets = config.get("_dandi_datasets", {})
        train_dataset = dandi_datasets.get("train_dataset")
        val_dataset = dandi_datasets.get("val_dataset")
        test_dataset = dandi_datasets.get("test_dataset")

        batch_size = config.get("batch_size", 16)

        # Custom collate function to convert DANDI dict format to (ob, pcx, odor, session_id) tuple
        def dandi_collate_fn(batch):
            """Convert DANDI batch dict to (source, target, label, session_id) tuple."""
            sources = torch.stack([item["source"] for item in batch])
            targets = torch.stack([item["target"] for item in batch])
            # DANDI has no odor labels or session IDs, use zeros as placeholder
            labels = torch.zeros(len(batch), dtype=torch.long)
            session_ids = torch.zeros(len(batch), dtype=torch.long)
            return sources, targets, labels, session_ids

        # Create samplers for distributed training
        train_sampler = DistributedSampler(train_dataset, seed=42) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=42) if is_distributed else None

        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": True,
            "collate_fn": dandi_collate_fn,
        }

        loaders = {
            "train": DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                drop_last=True,
                **loader_kwargs,
            ),
            "val": DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=val_sampler,
                drop_last=False,
                **loader_kwargs,
            ),
        }

        if test_dataset is not None and len(test_dataset) > 0:
            test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=42) if is_distributed else None
            loaders["test"] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=test_sampler,
                drop_last=False,
                **loader_kwargs,
            )

        # Update in_channels and out_channels from actual data
        # Get a sample batch to determine shapes (collate_fn returns tuple: source, target, label)
        sample_batch = next(iter(loaders["train"]))
        source_batch, target_batch, _ = sample_batch
        config["in_channels"] = source_batch.shape[1]  # [B, C, T]
        config["out_channels"] = target_batch.shape[1]

        if is_primary():
            print(f"DANDI DataLoaders: {len(train_dataset)} train, {len(val_dataset)} val windows")
            print(f"  Source channels: {config['in_channels']}, Target channels: {config['out_channels']}")
    else:
        loaders = create_dataloaders(
            data,
            batch_size=config.get("batch_size", 16),
            num_workers=num_workers,
            distributed=is_distributed,
        )

    # Create model (forward: source → target)
    # For olfactory: OB → PCx (32 → 32)
    # For PFC: PFC → CA1 (64 → 32)
    n_odors = data["n_odors"]  # Works for both (odors or trial types)
    in_channels = config.get("in_channels", 32)
    out_channels = config.get("out_channels", 32)
    attention_type = config.get("attention_type", "basic")
    conv_type = config.get("conv_type", "standard")
    arch = config.get("arch", "condunet")  # Architecture selection

    # Auto-detect number of sessions for session embedding if not specified
    if config.get("use_session_embedding", False) and config.get("n_sessions") is None:
        # Try to get number of sessions from data
        if "train_sessions" in data and "val_sessions" in data:
            all_sessions = list(set(data["train_sessions"]) | set(data.get("val_sessions", [])))
            config["n_sessions"] = len(all_sessions)
        elif "session_to_idx" in data:
            config["n_sessions"] = len(data["session_to_idx"])
        else:
            # Default fallback - can be overridden with --n-sessions
            config["n_sessions"] = 15  # Reasonable default for olfactory data
        if is_primary():
            print(f"Session embedding: auto-detected {config['n_sessions']} sessions")

    # Build session signatures for inference matching if session embedding is enabled
    session_matcher = None
    if config.get("n_sessions", 0) > 0 and "session_ids" in data and "ob" in data:
        session_matcher = SessionMatcher(
            n_channels=in_channels,
            n_warmup_samples=100,
            use_power_spectrum=True,
            sample_rate=config.get("sampling_rate", 1000.0),
        )
        # Register signatures for each unique session in training data
        session_ids_np = data["session_ids"]
        ob_data = data["ob"]
        train_idx = data.get("train_idx", np.arange(len(ob_data)))
        unique_sessions = np.unique(session_ids_np[train_idx])
        for sid in unique_sessions:
            session_mask = session_ids_np == sid
            session_data = ob_data[session_mask]
            if len(session_data) > 0:
                session_name = data.get("idx_to_session", {}).get(int(sid), f"session_{sid}")
                session_matcher.register_session(int(sid), session_data, session_name=session_name)
        # Store in config for saving with checkpoint
        config["_session_matcher"] = session_matcher
        if is_primary():
            print(f"Session signatures: computed for {len(unique_sessions)} training sessions")

    if is_primary():
        print(f"Architecture: {arch.upper()}")
        print(f"Model: {in_channels} input channels → {out_channels} output channels")
        if arch == "condunet":
            print(f"Conditions: {n_odors} classes")
            if config.get("n_sessions", 0) > 0:
                print(f"Session embedding: {config['n_sessions']} sessions, dim={config['session_emb_dim']}")

    # Create model based on architecture
    if arch == "condunet":
        # CondUNet with all its features
        model = CondUNet1D(
            in_channels=in_channels,
            out_channels=out_channels,
            base=config.get("base_channels", 128),
            n_odors=n_odors,
            dropout=config.get("dropout", 0.0),
            use_attention=config.get("use_attention", True),
            attention_type=attention_type,
            norm_type=config.get("norm_type", "batch"),
            cond_mode=config.get("cond_mode", "film"),
            # U-Net depth for frequency resolution
            n_downsample=config.get("n_downsample", 2),
            # Modern convolution options
            conv_type=conv_type,
            use_se=config.get("use_se", True),
            conv_kernel_size=config.get("conv_kernel_size", 7),
            dilations=config.get("conv_dilations", (1, 4, 16, 32)),
            # Output scaling correction (disabled if using adaptive scaling)
            use_output_scaling=config.get("use_output_scaling", True) and not config.get("use_adaptive_scaling", False),
            # Statistics-based session adaptation (Phase 3 Group 18)
            use_session_stats=config.get("use_session_stats", False),
            session_emb_dim=config.get("session_emb_dim", 32),
            session_use_spectral=config.get("session_use_spectral", False),
            # Learnable session embedding (lookup table approach)
            use_session_embedding=config.get("use_session_embedding", False),
            n_sessions=config.get("n_sessions", 0),
            # Other session adaptation methods
            use_adaptive_scaling=config.get("use_adaptive_scaling", False),
        )
    else:
        # Use Phase 2 architectures for comparison
        if not PHASE2_ARCHS_AVAILABLE:
            raise RuntimeError(f"Phase 2 architectures not available. Cannot use --arch {arch}")

        # Get time_steps from data
        time_steps = data.get("time_steps", 5000)
        if "ob" in data:
            time_steps = data["ob"].shape[-1]

        model = create_architecture(
            arch,
            in_channels=in_channels,
            out_channels=out_channels,
            time_steps=time_steps,
        )
        # Count parameters BEFORE FSDP wrapping (FSDP shards params across ranks)
        # Store in config for later use in results output
        model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        config["model_n_params"] = model_n_params
        if is_primary():
            print(f"  Parameters: {model_n_params:,}")

    # Create reverse model (target → source) for bidirectional training
    # Only for CondUNet - other architectures don't support conditioning
    reverse_model = None
    if config.get("use_bidirectional", False) and arch == "condunet":
        reverse_model = CondUNet1D(
            in_channels=out_channels,  # Reverse: target → source
            out_channels=in_channels,
            base=config.get("base_channels", 128),
            n_odors=n_odors,
            dropout=config.get("dropout", 0.0),
            use_attention=config.get("use_attention", True),
            attention_type=attention_type,
            norm_type=config.get("norm_type", "batch"),
            cond_mode=config.get("cond_mode", "film"),
            # U-Net depth (same as forward)
            n_downsample=config.get("n_downsample", 2),
            # Modern convolution options (same as forward)
            conv_type=conv_type,
            use_se=config.get("use_se", True),
            conv_kernel_size=config.get("conv_kernel_size", 7),
            dilations=config.get("conv_dilations", (1, 4, 16, 32)),
            # Output scaling correction (disabled if using adaptive scaling)
            use_output_scaling=config.get("use_output_scaling", True) and not config.get("use_adaptive_scaling", False),
            # Statistics-based session adaptation (same as forward)
            use_session_stats=config.get("use_session_stats", False),
            session_emb_dim=config.get("session_emb_dim", 32),
            session_use_spectral=config.get("session_use_spectral", False),
            # Learnable session embedding (same as forward model)
            use_session_embedding=config.get("use_session_embedding", False),
            n_sessions=config.get("n_sessions", 0),
            # Other session adaptation methods
            use_adaptive_scaling=config.get("use_adaptive_scaling", False),
        )
        if is_primary():
            print("Bidirectional training ENABLED")
            if conv_type == "modern":
                print(f"Using MODERN convolutions: dilations={config.get('conv_dilations', (1, 4, 16, 32))}, kernel_size={config.get('conv_kernel_size', 7)}, SE={config.get('use_se', True)}")
    elif config.get("use_bidirectional", False) and arch != "condunet":
        if is_primary():
            print("Warning: Bidirectional training only supported for CondUNet, disabling")

    # Create conditioning encoder for auto-conditioning modes
    # Only for CondUNet - other architectures don't support conditioning
    cond_source = config.get("conditioning_source", "odor_onehot")
    cond_encoder = None
    emb_dim = 128  # Must match model's emb_dim

    if cond_source != "odor_onehot" and arch == "condunet":
        if is_primary():
            print(f"Using auto-conditioning: {cond_source}")

        if cond_source == "spectro_temporal":
            cond_encoder = SpectroTemporalEncoder(
                in_channels=in_channels,
                emb_dim=emb_dim,
            )
        elif cond_source == "cpc":
            cond_encoder = CPCEncoder(
                in_channels=in_channels,
                embed_dim=emb_dim,
            )
        elif cond_source == "vqvae":
            cond_encoder = VQVAEEncoder(
                in_channels=in_channels,
                embed_dim=emb_dim,
            )
        elif cond_source == "freq_disentangled":
            cond_encoder = FreqDisentangledEncoder(
                in_channels=in_channels,
                embed_dim=emb_dim,
            )
        elif cond_source == "cycle_consistent":
            cond_encoder = CycleConsistentEncoder(
                in_channels=in_channels,
                out_channels=out_channels,
                embed_dim=emb_dim,
            )
        else:
            raise ValueError(f"Unknown conditioning source: {cond_source}")

        if is_primary():
            cond_params = sum(p.numel() for p in cond_encoder.parameters())
            print(f"Conditioning encoder parameters: {cond_params:,}")

    # Count actual parameters
    model_params = sum(p.numel() for p in model.parameters())
    rev_params = sum(p.numel() for p in reverse_model.parameters()) if reverse_model else 0

    # Log U-Net depth and frequency resolution (only for CondUNet)
    if arch == "condunet" and is_primary():
        n_downsample = config.get("n_downsample", 2)
        base_ch = config.get("base_channels", 128)
        downsample_factor = 2 ** n_downsample
        nyquist_hz = 1000 / (2 * downsample_factor)  # 1000 Hz sample rate

        # Calculate channel progression for logging
        channels = [base_ch]
        for i in range(n_downsample):
            channels.append(min(base_ch * (2 ** (i + 1)), base_ch * 8))

        print(f"U-Net depth: {n_downsample} levels → {downsample_factor}x downsample → bottleneck Nyquist = {nyquist_hz:.0f} Hz")
        print(f"Channel progression: {' → '.join(map(str, channels))} (bottleneck={channels[-1]})")

    if is_primary():
        print(f"Model parameters: {model_params:,} (forward) + {rev_params:,} (reverse) = {model_params + rev_params:,} total")

    # Enable gradient checkpointing if requested (reduces memory, allows larger batches)
    if config.get("gradient_checkpointing", False):
        model.set_gradient_checkpointing(True)
        if reverse_model is not None:
            reverse_model.set_gradient_checkpointing(True)
        if is_primary():
            print("Gradient checkpointing ENABLED (saves ~40% memory, costs ~30% more compute)")

    # Wrap for distributed training
    is_fsdp_wrapped = False
    if is_distributed:
        if use_fsdp:
            model = wrap_model_fsdp(
                model, local_rank,
                use_fsdp=True,
                cpu_offload=cpu_offload,
                sharding_strategy=fsdp_strategy,
                compile_model=compile_model,
            )
            if reverse_model is not None:
                reverse_model = wrap_model_fsdp(
                    reverse_model, local_rank,
                    use_fsdp=True,
                    cpu_offload=cpu_offload,
                    sharding_strategy=fsdp_strategy,
                    compile_model=compile_model,
                )
            # Conditioning encoder: just move to device (too small for FSDP sharding)
            # Also convert to bf16 if FSDP uses mixed precision
            # EXCEPTION: freq_disentangled uses FFT which requires float32
            if cond_encoder is not None:
                cond_source = config.get("conditioning_source", "odor_onehot")
                if check_bf16_support() and cond_source != "freq_disentangled":
                    cond_encoder = cond_encoder.to(device, dtype=torch.bfloat16)
                else:
                    # Keep in float32 for FFT-based encoders or non-bf16 systems
                    cond_encoder = cond_encoder.to(device)
            is_fsdp_wrapped = True
            # Set flag for train_epoch to use bf16 for data tensors
            config["fsdp_bf16"] = check_bf16_support()
            if is_primary():
                print(f"Using FSDP with {get_world_size()} GPUs")
            dist.barrier()
        else:
            # When using gradient checkpointing, we need find_unused_parameters=True
            # because checkpointing recomputes activations and some params may not
            # receive gradients until the recomputation happens
            # broadcast_buffers=False prevents DDP from synchronizing BatchNorm
            # running stats, which can cause inplace modification errors when
            # the same model is used multiple times per batch (e.g., cycle consistency)
            ddp_kwargs = {"device_ids": [local_rank], "broadcast_buffers": False}
            if config.get("gradient_checkpointing", False):
                ddp_kwargs["find_unused_parameters"] = True

            model = model.to(device)
            model = DDP(model, **ddp_kwargs)
            if reverse_model is not None:
                reverse_model = reverse_model.to(device)
                reverse_model = DDP(reverse_model, **ddp_kwargs)
            if cond_encoder is not None:
                cond_encoder = cond_encoder.to(device)
                # NOTE: We intentionally do NOT wrap cond_encoder in DDP.
                # Auto-conditioning encoders (CPC, VQVAE, FreqDisentangled, SpectroTemporal)
                # have complex forward passes with auxiliary losses that cause DDP crashes
                # even with find_unused_parameters=True. Instead, we manually sync gradients
                # using sync_gradients_manual() after backward() but before optimizer.step().
            if is_primary():
                print(f"Using DDP with {get_world_size()} GPUs (broadcast_buffers=False)")
                if config.get("gradient_checkpointing", False):
                    print("  (find_unused_parameters=True for gradient checkpointing)")
            dist.barrier()
    else:
        model = wrap_model_fsdp(model, local_rank, use_fsdp=False, compile_model=compile_model)
        if reverse_model is not None:
            reverse_model = wrap_model_fsdp(reverse_model, local_rank, use_fsdp=False, compile_model=compile_model)
        if cond_encoder is not None:
            cond_encoder = cond_encoder.to(device)

    # Import EnvelopeHistogramMatching for post-processing
    from models import EnvelopeHistogramMatching

    # Create projection heads for contrastive learning (if enabled)
    projection_head_fwd = None
    projection_head_rev = None
    use_contrastive = config.get("use_contrastive", False)
    contrastive_mode = config.get("contrastive_mode", "temporal")

    if use_contrastive:
        # CEBRA-style: Apply contrastive loss on BOTTLENECK features (encoder output)
        # This encourages the encoder to learn session-invariant representations
        # Bottleneck dim = min(base_channels * 2^n_downsample, base_channels * 8)
        base_ch = config.get("base_channels", 64)
        n_downsample = config.get("n_downsample", 4)
        bottleneck_dim = min(base_ch * (2 ** n_downsample), base_ch * 8)

        if contrastive_mode == "label":
            # Label-based mode: need projection heads for pooled features
            # Create projection heads (small MLP: bottleneck -> 256 -> 128)
            projection_head_fwd = ProjectionHead(
                in_dim=bottleneck_dim,  # Bottleneck channels (512 for default config)
                hidden_dim=256,
                out_dim=128,
            )
            projection_head_rev = ProjectionHead(
                in_dim=bottleneck_dim,  # Same for reverse (symmetric architecture)
                hidden_dim=256,
                out_dim=128,
            )

            # Handle device placement and dtype for FSDP compatibility
            if is_fsdp_wrapped and check_bf16_support():
                projection_head_fwd = projection_head_fwd.to(device, dtype=torch.bfloat16)
                projection_head_rev = projection_head_rev.to(device, dtype=torch.bfloat16)
            else:
                projection_head_fwd = projection_head_fwd.to(device)
                projection_head_rev = projection_head_rev.to(device)

            # Wrap with DDP for gradient sync (too small for FSDP sharding)
            if is_distributed and not is_fsdp_wrapped:
                projection_head_fwd = DDP(projection_head_fwd, device_ids=[local_rank])
                projection_head_rev = DDP(projection_head_rev, device_ids=[local_rank])

            if is_primary():
                contrastive_weight = config.get("contrastive_weight", 0.1)
                contrastive_temp = config.get("contrastive_temperature", 0.1)
                print(f"Contrastive learning ENABLED (label-based): weight={contrastive_weight}, temperature={contrastive_temp}")
                print(f"  Projection heads: {bottleneck_dim}->256->128 (bottleneck features)")
        else:
            # Temporal CEBRA mode: no projection heads needed
            # Loss is applied directly on normalized bottleneck features
            if is_primary():
                contrastive_weight = config.get("contrastive_weight", 0.1)
                contrastive_temp = config.get("contrastive_temperature", 0.1)
                time_delta = config.get("contrastive_time_delta", 10)
                num_samples = config.get("contrastive_num_samples", 32)
                print(f"Contrastive learning ENABLED (temporal CEBRA): weight={contrastive_weight}, temperature={contrastive_temp}")
                print(f"  Temporal params: time_delta={time_delta} samples, num_samples={num_samples} per trial")
                print(f"  No projection heads (loss on raw {bottleneck_dim}-dim bottleneck features)")

    # Define betas early since it's used by multiple optimizers
    betas = (config.get("beta1", 0.9), config.get("beta2", 0.999))

    # Create loss functions
    wavelet_loss = None
    if config.get("use_wavelet_loss", True):
        wavelet_loss = build_wavelet_loss(
            wavelet=config.get("wavelet_family", "morlet"),
            use_complex_morlet=config.get("use_complex_morlet", True),
            omega0=config.get("wavelet_omega0", 5.0),
        ).to(device)

    # Create optimizer with parameter groups
    lr = config.get("learning_rate", 1e-4)

    # Build parameter groups with different learning rates
    param_groups = [
        {"params": list(model.parameters()), "lr": lr},
    ]
    if reverse_model is not None:
        param_groups.append({"params": list(reverse_model.parameters()), "lr": lr})
    if cond_encoder is not None:
        # Conditioning encoder uses same lr as model
        param_groups.append({"params": list(cond_encoder.parameters()), "lr": lr, "name": "cond_encoder"})
    # Projection heads for contrastive learning (use same lr as model)
    if projection_head_fwd is not None:
        param_groups.append({"params": list(projection_head_fwd.parameters()), "lr": lr, "name": "projection_head_fwd"})
    if projection_head_rev is not None:
        param_groups.append({"params": list(projection_head_rev.parameters()), "lr": lr, "name": "projection_head_rev"})

    total_params = sum(len(list(pg["params"])) if not isinstance(pg["params"], list) else len(pg["params"]) for pg in param_groups)

    # Weight decay (L2 regularization)
    weight_decay = config.get("weight_decay", 0.0)

    if is_primary():
        print(f"Optimizer: {total_params} total params | lr={lr}, weight_decay={weight_decay}")

    optimizer = AdamW(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)

    # Learning rate scheduler configuration
    num_epochs = config.get("num_epochs", 80)
    lr_scheduler_type = config.get("lr_scheduler", "none")
    lr_warmup_epochs = config.get("lr_warmup_epochs", 5)
    lr_min_ratio = config.get("lr_min_ratio", 0.01)

    def make_lr_lambda():
        """Create lr lambda for a param group."""
        def lr_lambda(epoch):
            # Base multiplier from scheduler type
            if lr_scheduler_type == "none":
                return 1.0
            elif lr_scheduler_type == "cosine":
                # Cosine annealing from 1.0 to lr_min_ratio
                progress = epoch / max(num_epochs - 1, 1)
                return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            elif lr_scheduler_type == "cosine_warmup":
                # Linear warmup then cosine decay
                if epoch < lr_warmup_epochs:
                    return (epoch + 1) / lr_warmup_epochs
                else:
                    progress = (epoch - lr_warmup_epochs) / max(num_epochs - lr_warmup_epochs - 1, 1)
                    return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            else:
                return 1.0
        return lr_lambda

    # Build per-group lr lambdas
    lr_lambdas = [make_lr_lambda() for _ in param_groups]

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)

    if is_primary():
        print(f"LR scheduler: {lr_scheduler_type}" + (f" (warmup={lr_warmup_epochs}, min_ratio={lr_min_ratio})" if lr_scheduler_type != "none" else ""))

    # =========================================================================
    # Initialize Recording System (for Nature Methods publication)
    # =========================================================================
    recording_session = None
    if config.get("enable_recording", False) and RECORDING_AVAILABLE and is_primary():
        recording_config = RecordingConfig(
            record_losses=True,
            record_gradients=True,
            record_saliency=config.get("record_saliency", True),
            record_neuroscience=config.get("record_neuroscience", True),
            saliency_epoch_interval=config.get("saliency_epoch_interval", 5),
            neuroscience_epoch_interval=config.get("neuroscience_epoch_interval", 10),
            output_dir=Path(config.get("recording_output_dir", "artifacts/recordings")),
            use_tensorboard=True,
            use_hdf5=True,
            sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
            n_channels=config.get("out_channels", 32),  # Use output channels (what model produces)
        )
        recording_session = RecordingSession(
            recording_config,
            run_name=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        print(f"Recording system initialized: {recording_session.output_dir}")

    early_stop_patience = config.get("early_stop_patience", 8)

    # =============================================================================
    # Training loop
    # =============================================================================
    # Session embedding verification - print BEFORE training
    if config.get("use_session_embedding", False) and is_primary():
        # Get the actual model (unwrap from FSDP/DDP if needed)
        unwrapped = model.module if hasattr(model, 'module') else model
        if hasattr(unwrapped, 'session_embed') and unwrapped.session_embed is not None:
            with torch.no_grad():
                init_weight = unwrapped.session_embed.weight.clone()
                print(f"\n{'='*60}")
                print(f"[SESSION EMBED] BEFORE TRAINING:")
                print(f"  Weight shape: {init_weight.shape}")
                print(f"  Weight norm: {init_weight.norm().item():.6f}")
                print(f"  Weight mean: {init_weight.mean().item():.6f}")
                print(f"  Weight std: {init_weight.std().item():.6f}")
                print(f"  First 3 embeddings (norm): {[init_weight[i].norm().item() for i in range(min(3, init_weight.shape[0]))]}")
                print(f"{'='*60}\n")
                # Store for later comparison
                config['_session_embed_init_weight'] = init_weight.cpu()
        else:
            print(f"\n[SESSION EMBED] WARNING: use_session_embedding=True but model has no session_embed!\n")

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        if loaders.get("train_sampler") is not None:
            loaders["train_sampler"].set_epoch(epoch)

        train_metrics = train_epoch(
            model, loaders["train"], optimizer, device, config,
            wavelet_loss,
            reverse_model, epoch, num_epochs,
            cond_encoder=cond_encoder,
            projection_head_fwd=projection_head_fwd,
            projection_head_rev=projection_head_rev,
        )

        barrier()

        # Validation (skip some epochs if val_every > 1 for faster training)
        val_every = config.get("val_every", 1)
        should_validate = (epoch % val_every == 0) or (epoch == num_epochs) or (epoch == 1)

        if should_validate:
            val_metrics = evaluate(
                model, loaders["val"], device, wavelet_loss,
                compute_phase=False, reverse_model=reverse_model, config=config,
                fast_mode=True,
                sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
                cond_encoder=cond_encoder,
            )
            last_val_metrics = val_metrics  # Cache for non-validation epochs

            # Per-session validation (if separate_val_sessions is enabled)
            per_session_metrics = {}
            if "val_sessions" in loaders and config.get("separate_val_sessions", False):
                for sess_name, sess_loader in loaders["val_sessions"].items():
                    sess_metrics = evaluate(
                        model, sess_loader, device, wavelet_loss,
                        compute_phase=False, reverse_model=reverse_model, config=config,
                        fast_mode=True,
                        sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
                        cond_encoder=cond_encoder,
                    )
                    per_session_metrics[sess_name] = sess_metrics
        else:
            # Skip validation this epoch - use cached metrics
            val_metrics = last_val_metrics if 'last_val_metrics' in dir() else {"loss": float("inf"), "corr": 0, "r2": 0}
            per_session_metrics = {}

        # Sync val_loss across ranks (for early stopping)
        val_loss = val_metrics.get("loss", val_metrics.get("mae", float("inf")))  # fallback to mae if no composite
        if dist.is_initialized():
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()
            val_metrics["loss"] = val_loss

        # Track best (lower loss is better)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                model, optimizer, epoch,
                CHECKPOINT_DIR / "best_model.pt",
                is_fsdp=is_fsdp_wrapped,
                reverse_model=reverse_model,
                config=config,
                data=data,  # Save split info for validation
            )
        else:
            patience_counter += 1

        barrier()

        if is_primary():
            rev_str = ""
            if "corr_rev" in val_metrics:
                rev_str = f" | Rev: r={val_metrics['corr_rev']:.3f}, r²={val_metrics.get('r2_rev', 0):.3f}"

            # Add contrastive loss if present
            contr_str = ""
            if "contrastive_fwd" in train_metrics:
                contr_str = f" | Contr: {train_metrics['contrastive_fwd']:.3f}"
                if "contrastive_rev" in train_metrics:
                    contr_str += f"/{train_metrics['contrastive_rev']:.3f}"

            print(f"Epoch {epoch}/{num_epochs} | "
                  f"Train: {train_metrics['loss']:.3f} | Val: {val_metrics['loss']:.3f} | "
                  f"Fwd: r={val_metrics['corr']:.3f}, r²={val_metrics.get('r2', 0):.3f}{rev_str}{contr_str} | "
                  f"Best: {best_val_loss:.3f}")

            # Print per-session metrics if available
            if per_session_metrics:
                sess_strs = []
                for sess_name, sess_m in per_session_metrics.items():
                    sess_strs.append(f"  {sess_name}: r={sess_m['corr']:.3f}, r²={sess_m.get('r2', 0):.3f}")
                print("  Per-session: " + " | ".join(sess_strs))

            # Print session embedding weight stats every 10 epochs
            if config.get("use_session_embedding", False) and epoch % 10 == 0:
                unwrapped = model.module if hasattr(model, 'module') else model
                if hasattr(unwrapped, 'session_embed') and unwrapped.session_embed is not None:
                    with torch.no_grad():
                        w = unwrapped.session_embed.weight
                        print(f"  [Session Embed] norm={w.norm().item():.4f}, mean={w.mean().item():.4f}, std={w.std().item():.4f}")

            sys.stdout.flush()

            # Build history entry with per-session metrics if available
            history_entry = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            for sess_name, sess_m in per_session_metrics.items():
                for k, v in sess_m.items():
                    history_entry[f"val_{sess_name}_{k}"] = v
            history.append(history_entry)

            # =========================================================================
            # Recording: Log epoch metrics and run periodic analyses
            # =========================================================================
            if recording_session is not None:
                recording_session.set_epoch(epoch)

                # Log loss components
                recording_session.loss_recorder.record_epoch(epoch)

                # Log validation metrics
                for metric_name, value in val_metrics.items():
                    recording_session.log_scalar(f"val/{metric_name}", value, epoch)

                # Log training metrics
                for metric_name, value in train_metrics.items():
                    recording_session.log_scalar(f"train/{metric_name}", value, epoch)

                # Record gradient statistics
                if recording_session.should_record_weight_histograms(epoch):
                    # Get the underlying model (unwrap DDP/FSDP if needed)
                    base_model = model.module if hasattr(model, 'module') else model
                    recording_session.gradient_recorder.record_weight_distributions(
                        base_model, epoch, model_name="fwd"
                    )

                # Periodic saliency and Grad-CAM analysis
                if recording_session.should_compute_saliency(epoch):
                    try:
                        base_model = model.module if hasattr(model, 'module') else model
                        base_model.eval()

                        # Get a sample batch for saliency computation
                        sample_ob, sample_pcx, sample_odor = next(iter(loaders["val"]))
                        sample_ob = sample_ob[:8].to(device)  # Limit to 8 samples
                        sample_pcx = sample_pcx[:8].to(device)
                        sample_odor = sample_odor[:8].to(device)

                        # Compute input saliency
                        saliency = recording_session.saliency_analyzer.compute_input_saliency(
                            base_model, sample_ob, sample_odor, sample_pcx
                        )

                        # Store saliency map
                        if 'hdf5' in recording_session.backends:
                            recording_session.backends['hdf5'].log_saliency(
                                'input', saliency.cpu().numpy(), epoch
                            )

                        # Compute Grad-CAM at all levels
                        gradcam_all = recording_session.saliency_analyzer.compute_gradcam_all_levels(
                            base_model, sample_ob, sample_odor
                        )

                        # Store Grad-CAM
                        for layer_name, cam in gradcam_all.items():
                            if 'hdf5' in recording_session.backends:
                                recording_session.backends['hdf5'].log_gradcam(
                                    layer_name, cam.cpu().numpy(), epoch
                                )

                        # Compute importance metrics
                        channel_imp = recording_session.importance_analyzer.compute_channel_importance(saliency)
                        temporal_imp, window_centers = recording_session.importance_analyzer.compute_temporal_importance(saliency)
                        freq_imp = recording_session.importance_analyzer.compute_frequency_importance(sample_ob, saliency)

                        # Store importance metrics
                        if 'hdf5' in recording_session.backends:
                            recording_session.backends['hdf5'].log_importance(
                                'multi_dim', {
                                    'channel': channel_imp.cpu().numpy(),
                                    'temporal': temporal_imp.cpu().numpy(),
                                    'window_centers_ms': window_centers,
                                    'frequency': freq_imp,
                                }, epoch
                            )

                        base_model.train()
                        if is_primary():
                            print(f"    [Recording] Saliency & Grad-CAM computed for epoch {epoch}")

                    except Exception as e:
                        if is_primary():
                            print(f"    [Recording] Saliency computation failed: {e}")

                # Periodic neuroscience analysis (less frequent)
                if recording_session.should_compute_neuroscience(epoch):
                    try:
                        base_model = model.module if hasattr(model, 'module') else model
                        base_model.eval()

                        sample_ob, sample_pcx, sample_odor = next(iter(loaders["val"]))
                        sample_ob = sample_ob[:8].to(device)
                        sample_pcx = sample_pcx[:8].to(device)
                        sample_odor = sample_odor[:8].to(device)

                        # Run full neuroscience analysis
                        neuro_results = recording_session.neuro_analyzer.compute_full_analysis(
                            base_model, sample_ob, sample_odor, sample_pcx
                        )

                        # Store neuroscience results
                        for analysis_type, results in neuro_results.items():
                            if 'error' not in results and 'hdf5' in recording_session.backends:
                                recording_session.backends['hdf5'].log_neuroscience(
                                    analysis_type, results, epoch
                                )

                        # Log summary statistics
                        summary = recording_session.neuro_analyzer.generate_summary_statistics(neuro_results)
                        for stat_name, value in summary.items():
                            if isinstance(value, (int, float)):
                                recording_session.log_scalar(f"neuro/{stat_name}", value, epoch)

                        base_model.train()
                        if is_primary():
                            print(f"    [Recording] Neuroscience analysis completed for epoch {epoch}")

                    except Exception as e:
                        if is_primary():
                            print(f"    [Recording] Neuroscience analysis failed: {e}")

                # Flush backends periodically
                if epoch % 5 == 0:
                    recording_session.flush()

            # Step the lr scheduler (only primary has it)
            scheduler.step()

        # =====================================================================
        # Early stopping check - MUST be outside is_primary() block
        # All ranks must break together to avoid NCCL hangs
        # =====================================================================
        should_stop = patience_counter >= early_stop_patience

        # Broadcast early stopping decision to all ranks
        if dist.is_initialized():
            stop_tensor = torch.tensor([1 if should_stop else 0], dtype=torch.int32, device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1

        if should_stop:
            if is_primary():
                print(f"Early stopping at epoch {epoch} (patience {early_stop_patience} exceeded)")
            # Barrier before break to ensure all ranks are synchronized
            barrier()
            break

        # Stage 1 evaluation (if enabled)
        # NOTE: ALL ranks must participate in evaluate() because model is sharded with FSDP
        if config.get("eval_stage1", False):
            if is_primary():
                print(f"\n{'='*70}")
                print("STAGE 1 EVALUATION (before spectral fine-tuning)")
                print(f"{'='*70}")

            barrier()

            # Evaluate on VALIDATION set - ALL ranks must call this (FSDP requirement)
            val_metrics_stage1 = evaluate(
                model, loaders["val"], device, wavelet_loss,
                compute_phase=True, reverse_model=reverse_model, config=config,
                fast_mode=False,  # Full metrics for stage evaluation
                sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
            )

            barrier()

            # Evaluate on TEST set - ALL ranks must call this (FSDP requirement)
            test_metrics_stage1 = evaluate(
                model, loaders["test"], device, wavelet_loss,
                compute_phase=True, reverse_model=reverse_model, config=config,
                fast_mode=False,  # Full metrics for stage evaluation
                sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
            )

            barrier()

            if is_primary():
                # Validation set metrics
                print(f"Stage 1 Validation Metrics:")
                print(f"  R²: {val_metrics_stage1['r2']:.4f}")
                print(f"  Correlation: {val_metrics_stage1['corr']:.4f}")
                print(f"  MAE: {val_metrics_stage1['mae']:.4f}")
                if 'psd_err_db' in val_metrics_stage1:
                    print(f"  PSD Error: {val_metrics_stage1['psd_err_db']:.2f} dB")

                # Test set metrics
                print(f"Stage 1 Test Metrics:")
                print(f"  R²: {test_metrics_stage1['r2']:.4f}")
                print(f"  Correlation: {test_metrics_stage1['corr']:.4f}")
                print(f"  MAE: {test_metrics_stage1['mae']:.4f}")
                print(f"  NRMSE: {test_metrics_stage1.get('nrmse', 'N/A')}")
                if 'psd_err_db' in test_metrics_stage1:
                    print(f"  PSD Error: {test_metrics_stage1['psd_err_db']:.2f} dB")

                # Machine-parseable results (for tier1/tier1.5 scripts)
                # Validation metrics
                print(f"STAGE1_VAL_R2={val_metrics_stage1['r2']:.4f}")
                print(f"STAGE1_VAL_CORR={val_metrics_stage1['corr']:.4f}")
                print(f"STAGE1_VAL_MAE={val_metrics_stage1['mae']:.4f}")
                if 'psd_err_db' in val_metrics_stage1:
                    print(f"STAGE1_VAL_PSD_ERR_DB={val_metrics_stage1['psd_err_db']:.4f}")
                # Test metrics (keeping original names for backward compatibility)
                print(f"STAGE1_RESULT_R2={test_metrics_stage1['r2']:.4f}")
                print(f"STAGE1_RESULT_CORR={test_metrics_stage1['corr']:.4f}")
                print(f"STAGE1_RESULT_MAE={test_metrics_stage1['mae']:.4f}")
                print(f"STAGE1_RESULT_LOSS={best_val_loss:.4f}")
                if 'psd_err_db' in test_metrics_stage1:
                    print(f"STAGE1_RESULT_PSD_ERR_DB={test_metrics_stage1['psd_err_db']:.4f}")

    # =============================================================================
    # POST-HOC CALIBRATION MOVED TO calibrate.py
    # =============================================================================
    # Stage 2 (spectral bias + envelope matching) is now in a separate script.
    # Run: python calibrate.py --checkpoint artifacts/checkpoints/best_model.pt
    #
    # This keeps train.py focused on Stage 1 (UNet training) only.

    # Initialize envelope matchers to None (calibrate.py handles these)
    envelope_matcher_fwd = None
    envelope_matcher_rev = None

    # Final test evaluation (full metrics, fast_mode=False)
    # Skip if no test set (no_test_set=True means all held-out sessions are validation)
    has_test_set = len(data.get("test_idx", [])) > 0
    if has_test_set:
        test_metrics = evaluate(
            model, loaders["test"], device, wavelet_loss,
            compute_phase=True, reverse_model=reverse_model, config=config,
            fast_mode=False,  # Full metrics for final evaluation
            sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
            cond_encoder=cond_encoder,
            envelope_matcher_fwd=envelope_matcher_fwd,
            envelope_matcher_rev=envelope_matcher_rev,
        )
    else:
        test_metrics = {}
        if is_primary():
            print("\n" + "="*60)
            print("NO TEST SET (all held-out sessions used for validation)")
            print("="*60)
            print("Final evaluation uses per-session validation metrics above.")
            sys.stdout.flush()

    if is_primary() and has_test_set:
        print("\n" + "="*60)
        print("FINAL TEST RESULTS")
        print("="*60)

        # Determine if we're using mean-signal baseline (different channel counts)
        in_ch = config.get("in_channels", 32)
        out_ch = config.get("out_channels", 32)
        baseline_note = "" if in_ch == out_ch else " (mean-signal comparison)"

        # Baseline: Natural difference between source and target (context for comparison)
        print(f"Baseline (Raw Source vs Target - natural difference){baseline_note}:")
        base_corr = test_metrics.get('baseline_corr', 0)
        base_r2 = test_metrics.get('baseline_r2', 0)
        base_nrmse = test_metrics.get('baseline_nrmse', 0)
        base_psd_bias = test_metrics.get('baseline_psd_diff_db', 0)
        base_psd_err = test_metrics.get('baseline_psd_err_db', 0)
        print(f"  Correlation: {base_corr:.4f}")
        print(f"  R²: {base_r2:.4f}")
        print(f"  NRMSE: {base_nrmse:.4f}")
        print(f"  PSD Bias: {base_psd_bias:+.2f} dB (|err|={base_psd_err:.2f}dB)")
        if "baseline_plv" in test_metrics:
            print(f"  PLV: {test_metrics['baseline_plv']:.4f}")
            print(f"  PLI: {test_metrics['baseline_pli']:.4f}")

        # Forward direction with delta from baseline
        print("\nForward Direction (OB → PCx):" if config.get("dataset_type") != "pfc" else "\nForward Direction (PFC → CA1):")
        fwd_corr = test_metrics['corr']
        fwd_r2 = test_metrics['r2']
        fwd_nrmse = test_metrics['nrmse']
        psd_bias_fwd = test_metrics.get('psd_diff_db', 0)
        psd_err_fwd = test_metrics.get('psd_err_db', 0)
        # Show delta from baseline (positive = improvement)
        delta_corr = fwd_corr - base_corr
        delta_r2 = fwd_r2 - base_r2
        delta_nrmse = base_nrmse - fwd_nrmse  # Lower NRMSE is better, so flip sign
        delta_psd_err = base_psd_err - psd_err_fwd  # Lower err is better
        print(f"  Correlation: {fwd_corr:.4f} (Δ={delta_corr:+.4f})")
        print(f"  R²: {fwd_r2:.4f} (Δ={delta_r2:+.4f})")
        print(f"  NRMSE: {fwd_nrmse:.4f} (Δ={delta_nrmse:+.4f})")
        print(f"  PSD Bias: {psd_bias_fwd:+.2f} dB (|err|={psd_err_fwd:.2f}dB, Δerr={delta_psd_err:+.2f})")
        if "plv" in test_metrics:
            delta_plv = test_metrics['plv'] - test_metrics.get('baseline_plv', 0)
            print(f"  PLV: {test_metrics['plv']:.4f} (Δ={delta_plv:+.4f})")
            print(f"  PLI: {test_metrics['pli']:.4f}")

        if "corr_rev" in test_metrics:
            print("\nReverse Direction (PCx → OB):")
            rev_corr = test_metrics['corr_rev']
            rev_r2 = test_metrics['r2_rev']
            rev_nrmse = test_metrics['nrmse_rev']
            psd_bias_rev = test_metrics.get('psd_diff_db_rev', 0)
            psd_err_rev = test_metrics.get('psd_err_db_rev', 0)
            # Reverse uses same baseline (OB vs PCx) since it's symmetric
            delta_corr_rev = rev_corr - base_corr
            delta_r2_rev = rev_r2 - base_r2
            delta_nrmse_rev = base_nrmse - rev_nrmse
            delta_psd_err_rev = base_psd_err - psd_err_rev
            print(f"  Correlation: {rev_corr:.4f} (Δ={delta_corr_rev:+.4f})")
            print(f"  R²: {rev_r2:.4f} (Δ={delta_r2_rev:+.4f})")
            print(f"  NRMSE: {rev_nrmse:.4f} (Δ={delta_nrmse_rev:+.4f})")
            print(f"  PSD Bias: {psd_bias_rev:+.2f} dB (|err|={psd_err_rev:.2f}dB, Δerr={delta_psd_err_rev:+.2f})")
            if "plv_rev" in test_metrics:
                delta_plv_rev = test_metrics['plv_rev'] - test_metrics.get('baseline_plv', 0)
                print(f"  PLV: {test_metrics['plv_rev']:.4f} (Δ={delta_plv_rev:+.4f})")
                print(f"  PLI: {test_metrics['pli_rev']:.4f}")

    # =========================================================================
    # PER-SESSION TEST EVALUATION (for cross-session generalization analysis)
    # =========================================================================
    if is_primary() and has_test_set and "split_info" in data and "session_ids" in data:
        split_info = data.get("split_info", {})
        test_sessions = split_info.get("test_sessions", [])
        session_ids = data.get("session_ids")
        idx_to_session = data.get("idx_to_session", {})
        test_idx = data["test_idx"]

        if len(test_sessions) > 1:
            print("\n" + "=" * 70)
            print("PER-SESSION TEST RESULTS")
            print("=" * 70)

            # Create session name to int ID mapping
            session_to_id = {name: idx for idx, name in idx_to_session.items()}

            per_session_results = []

            for session_name in test_sessions:
                # Find indices for this session
                session_id = session_to_id.get(session_name)
                if session_id is None:
                    continue

                # Get indices for this session that are in test set
                session_mask = session_ids[test_idx] == session_id
                session_indices = test_idx[session_mask]

                if len(session_indices) == 0:
                    continue

                # Create dataloader for this session
                session_loader = create_single_session_dataloader(
                    data, session_name, session_indices,
                    batch_size=config.get("batch_size", 16),
                    num_workers=4,
                    distributed=False,  # Per-session eval on primary only
                )

                # Evaluate this session
                session_metrics = evaluate(
                    model, session_loader, device, wavelet_loss,
                    compute_phase=False, reverse_model=None, config=config,
                    fast_mode=False,  # Need full metrics for baseline
                    sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
                    cond_encoder=cond_encoder,
                )

                per_session_results.append({
                    "session": session_name,
                    "n_trials": len(session_indices),
                    "corr": session_metrics["corr"],
                    "baseline_corr": session_metrics.get("baseline_corr", 0),
                    "r2": session_metrics["r2"],
                    "psd_err_db": session_metrics.get("psd_err_db", 0),
                })

            # Print summary table
            if per_session_results:
                print(f"\n{'Session':<15} {'Trials':>8} {'Corr':>8} {'Baseline':>10} {'Δ Corr':>10} {'R²':>8}")
                print("-" * 70)

                for r in per_session_results:
                    delta = r['corr'] - r['baseline_corr']
                    print(f"{r['session']:<15} {r['n_trials']:>8} {r['corr']:>8.4f} {r['baseline_corr']:>10.4f} {delta:>+10.4f} {r['r2']:>8.4f}")

                # Compute aggregate stats
                avg_corr = np.mean([r['corr'] for r in per_session_results])
                avg_baseline = np.mean([r['baseline_corr'] for r in per_session_results])
                avg_delta = avg_corr - avg_baseline
                avg_r2 = np.mean([r['r2'] for r in per_session_results])
                total_trials = sum(r['n_trials'] for r in per_session_results)

                print("-" * 70)
                print(f"{'AVERAGE':<15} {total_trials:>8} {avg_corr:>8.4f} {avg_baseline:>10.4f} {avg_delta:>+10.4f} {avg_r2:>8.4f}")
                print("=" * 70)

                # Machine-parseable output
                print(f"\nRESULT_PER_SESSION_AVG_CORR={avg_corr:.4f}")
                print(f"RESULT_PER_SESSION_AVG_DELTA={avg_delta:.4f}")

    if is_distributed:
        dist.barrier()

    # =========================================================================
    # Generate Validation Plots (galleries, per-session, per-odor)
    # =========================================================================
    if is_primary() and VALIDATION_PLOTS_AVAILABLE and config.get("generate_plots", True):
        try:
            plots_dir = CHECKPOINT_DIR / "validation_plots"

            # Determine which dataloader to use for plots
            if has_test_set:
                plot_loader = loaders["test"]
                plot_indices = data.get("test_idx")
            else:
                plot_loader = loaders["val"]
                plot_indices = data.get("val_idx")

            # Get session info for the plots
            session_ids_for_plots = None
            idx_to_session = None
            if "session_ids" in data and plot_indices is not None:
                session_ids_for_plots = data["session_ids"][plot_indices]
                idx_to_session = data.get("idx_to_session", {})

            # Get the base models (unwrap FSDP/DDP if needed)
            base_model = model.module if hasattr(model, 'module') else model
            base_reverse_model = None
            if reverse_model is not None:
                base_reverse_model = reverse_model.module if hasattr(reverse_model, 'module') else reverse_model

            # Put models in eval mode
            base_model.eval()
            if base_reverse_model is not None:
                base_reverse_model.eval()

            generate_training_plots(
                model_fwd=base_model,
                model_rev=base_reverse_model,
                dataloader=plot_loader,
                device=device,
                vocab=data["vocab"],
                output_dir=plots_dir,
                config=config,
                session_ids=session_ids_for_plots,
                idx_to_session=idx_to_session,
                formats=["png"],  # Fast - only PNG for training
                quick=True,  # Essential plots only
            )
        except Exception as e:
            print(f"\nWarning: Failed to generate validation plots: {e}")
            traceback.print_exc()

    # =========================================================================
    # Recording: Final report generation and cleanup
    # =========================================================================
    if recording_session is not None:
        try:
            # Generate final summary figure
            visualizer = NeuroVisualizer(
                output_dir=recording_session.output_dir / "figures",
                format='pdf'
            )

            # Get loss curves
            loss_curves = recording_session.loss_recorder.get_loss_curves()
            if loss_curves:
                visualizer.plot_loss_curves(loss_curves, save_name='final_loss_curves')

            # Final saliency analysis
            base_model = model.module if hasattr(model, 'module') else model
            base_model.eval()

            sample_ob, sample_pcx, sample_odor = next(iter(loaders["val"]))
            sample_ob = sample_ob[:8].to(device)
            sample_pcx = sample_pcx[:8].to(device)
            sample_odor = sample_odor[:8].to(device)

            saliency = recording_session.saliency_analyzer.compute_input_saliency(
                base_model, sample_ob, sample_odor, sample_pcx
            )
            mean_saliency = saliency.mean(dim=0).cpu().numpy()

            visualizer.plot_saliency_heatmap(mean_saliency, save_name='final_saliency')

            # Final importance analysis
            channel_imp = recording_session.importance_analyzer.compute_channel_importance(saliency)
            freq_imp = recording_session.importance_analyzer.compute_frequency_importance(sample_ob, saliency)

            visualizer.plot_channel_importance(channel_imp.cpu().numpy(), save_name='final_channel_importance')
            visualizer.plot_frequency_importance(freq_imp, save_name='final_frequency_importance')

            # Final neuroscience analysis
            neuro_results = recording_session.neuro_analyzer.compute_full_analysis(
                base_model, sample_ob, sample_odor, sample_pcx
            )

            if 'pac' in neuro_results and 'error' not in neuro_results['pac']:
                visualizer.plot_pac_analysis(neuro_results['pac'], save_name='final_pac_analysis')

            if 'coherence' in neuro_results and 'error' not in neuro_results['coherence']:
                visualizer.plot_coherence_matrix(neuro_results['coherence'], save_name='final_coherence')

            if 'erp' in neuro_results and 'error' not in neuro_results['erp']:
                visualizer.plot_erp_components(neuro_results['erp'], save_name='final_erp')

            visualizer.close_all()

            # Export loss history to CSV
            recording_session.loss_recorder.export_to_csv(
                str(recording_session.output_dir / "data" / "training_metrics.csv")
            )

            print(f"\n[Recording] Final analysis complete. Output: {recording_session.output_dir}")

        except Exception as e:
            print(f"\n[Recording] Final report generation failed: {e}")

        finally:
            # Always close the recording session
            recording_session.close()

    # Session embedding verification - print AFTER training
    if config.get("use_session_embedding", False) and is_primary():
        # Get the actual model (unwrap from FSDP/DDP if needed)
        unwrapped = model.module if hasattr(model, 'module') else model
        if hasattr(unwrapped, 'session_embed') and unwrapped.session_embed is not None:
            with torch.no_grad():
                final_weight = unwrapped.session_embed.weight.clone()
                print(f"\n{'='*60}")
                print(f"[SESSION EMBED] AFTER TRAINING:")
                print(f"  Weight shape: {final_weight.shape}")
                print(f"  Weight norm: {final_weight.norm().item():.6f}")
                print(f"  Weight mean: {final_weight.mean().item():.6f}")
                print(f"  Weight std: {final_weight.std().item():.6f}")
                print(f"  First 3 embeddings (norm): {[final_weight[i].norm().item() for i in range(min(3, final_weight.shape[0]))]}")
                # Compare with initial weights if available
                if '_session_embed_init_weight' in config:
                    init_weight = config['_session_embed_init_weight'].to(final_weight.device)
                    weight_delta = (final_weight - init_weight).norm().item()
                    print(f"  WEIGHT CHANGE (L2 norm of delta): {weight_delta:.6f}")
                    if weight_delta > 0.001:
                        print(f"  ✓ Session embeddings CHANGED during training (delta > 0.001)")
                    else:
                        print(f"  ⚠ Session embeddings BARELY changed (delta={weight_delta:.6f})")
                print(f"{'='*60}\n")
        else:
            print(f"\n[SESSION EMBED] WARNING: use_session_embedding=True but model has no session_embed!\n")

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "history": history,
        "model": model,
        "reverse_model": reverse_model,
        "n_parameters": config.get("model_n_params", 0),  # Pre-FSDP param count
    }


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CondUNet1D for neural signal translation")

    # Architecture selection (for Phase 2 comparison)
    parser.add_argument("--arch", type=str, default="condunet",
                        choices=["condunet", "linear", "simplecnn", "wavenet", "vit"],
                        help="Architecture to train (default: condunet). Other options for Phase 2 comparison.")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="olfactory",
                        choices=["olfactory", "pfc", "pcx1", "dandi"],
                        help="Dataset to train on: 'olfactory' (OB→PCx trial-based), "
                             "'pfc' (PFC→CA1), 'pcx1' (continuous 1kHz LFP), or "
                             "'dandi' (DANDI 000623 human iEEG movie watching)")
    parser.add_argument("--resample-pfc", action="store_true",
                        help="Resample PFC dataset from 1250Hz to 1000Hz (for compatibility)")

    # PFC sliding window options
    parser.add_argument("--pfc-sliding-window", action="store_true",
                        help="Use sliding window training for PFC dataset (more training samples)")
    parser.add_argument("--pfc-window-size", type=int, default=2500,
                        help="Window size in samples for PFC sliding window (default: 2500 = 2s at 1250Hz)")
    parser.add_argument("--pfc-stride", type=int, default=None,
                        help="Stride between windows for PFC (default: window_size // 2 for 50%% overlap)")
    parser.add_argument("--pfc-stride-ratio", type=float, default=None,
                        help="Stride as ratio of window size for PFC (0.5 = 50%% overlap)")

    # PCx1 continuous dataset options
    parser.add_argument("--pcx1-window-size", type=int, default=5000,
                        help="Window size in samples for PCx1 continuous data (default: 5000 = 5 seconds)")
    parser.add_argument("--pcx1-stride", type=int, default=None,
                        help="Stride between windows for PCx1 (default: window_size // 2 for 50%% overlap)")
    parser.add_argument("--pcx1-stride-ratio", type=float, default=None,
                        help="Stride as ratio of window size (0.5 = 50%% overlap, 1.0 = no overlap). "
                             "Overrides --pcx1-stride if both specified. Higher = faster training.")
    parser.add_argument("--val-stride-multiplier", type=float, default=1.0,
                        help="Multiply validation stride by this factor for faster eval (default: 1.0). "
                             "Use 2.0-4.0 for faster validation without affecting training.")
    parser.add_argument("--pcx1-train-sessions", type=str, nargs="+", default=None,
                        help="Explicit training sessions for PCx1 (e.g., --pcx1-train-sessions 141208-1 141208-2)")
    parser.add_argument("--pcx1-val-sessions", type=str, nargs="+", default=None,
                        help="Explicit validation sessions for PCx1 (default: 4 random sessions)")
    parser.add_argument("--pcx1-n-val", type=int, default=4,
                        help="Number of validation sessions for PCx1 if not explicitly specified (default: 4)")

    # DANDI 000623 dataset options
    parser.add_argument("--dandi-source-region", type=str, default="amygdala",
                        choices=["amygdala", "hippocampus", "medial_frontal_cortex"],
                        help="Source brain region for DANDI dataset (default: amygdala)")
    parser.add_argument("--dandi-target-region", type=str, default="hippocampus",
                        choices=["amygdala", "hippocampus", "medial_frontal_cortex"],
                        help="Target brain region for DANDI dataset (default: hippocampus)")
    parser.add_argument("--dandi-window-size", type=int, default=5000,
                        help="Window size in samples for DANDI (default: 5000 = 5s at 1kHz)")
    parser.add_argument("--dandi-stride", type=int, default=None,
                        help="Stride between windows for DANDI (default: window_size // 2)")
    parser.add_argument("--dandi-stride-ratio", type=float, default=None,
                        help="Stride as ratio of window size for DANDI (0.5 = 50%% overlap)")
    parser.add_argument("--dandi-data-dir", type=str, default=None,
                        help="Directory containing DANDI NWB files (default: $UNET_DATA_DIR/movie)")

    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: from config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: from config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: from config)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Weight decay / L2 regularization (default: 0.0, try 0.01-0.1)")
    parser.add_argument("--lr-scheduler", type=str, default=None,
                        choices=["none", "cosine", "cosine_warmup"],
                        help="LR scheduler: 'none' (constant), 'cosine' (annealing), 'cosine_warmup' (warmup + cosine)")
    parser.add_argument("--lr-warmup-epochs", type=int, default=None,
                        help="Warmup epochs for cosine_warmup scheduler (default: 5)")
    parser.add_argument("--lr-min-ratio", type=float, default=None,
                        help="Min LR as ratio of initial for cosine scheduler (default: 0.01)")
    parser.add_argument("--base-channels", type=int, default=None, help="Base channels for model (default: from config)")
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP for distributed training")
    parser.add_argument("--fsdp-strategy", type=str, default="full",
                        choices=["full", "grad_op", "no_shard", "hybrid", "hybrid_zero2"],
                        help="FSDP sharding strategy. 'grad_op' recommended for 8x A100")
    parser.add_argument("--cpu-offload", action="store_true", help="Offload to CPU (large models)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (slower startup but faster training)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: from config)")

    # Training speed optimizations
    parser.add_argument("--val-every", type=int, default=1,
                        help="Validate every N epochs (default: 1). Use 5-10 for faster training.")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce memory and allow larger batches")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of DataLoader workers (default: auto based on CPU count)")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="DataLoader prefetch factor (default: 4)")

    # Session-based split arguments (for held-out session evaluation)
    # Use default=None so CLI doesn't override config file values
    parser.add_argument("--split-by-session", action="store_true", default=None,
                        help="Use session-based holdout instead of random splits. "
                             "Entire recording sessions are held out for test/val.")
    parser.add_argument("--no-split-by-session", action="store_true", default=None,
                        help="Disable session-based splits (use random splits)")
    parser.add_argument("--n-test-sessions", type=int, default=None,
                        help="Number of sessions to hold out for testing (default: from config)")
    parser.add_argument("--n-val-sessions", type=int, default=None,
                        help="Number of sessions to hold out for validation (default: from config)")
    parser.add_argument("--session-column", type=str, default=None,
                        help="CSV column name containing session/recording IDs (default: from config)")
    parser.add_argument("--test-sessions", type=str, nargs="+", default=None,
                        help="Explicit session names for test set (e.g., --test-sessions 141208-1 170614)")
    parser.add_argument("--val-sessions", type=str, nargs="+", default=None,
                        help="Explicit session names for val set (e.g., --val-sessions 170609 170619)")
    parser.add_argument("--force-recreate-splits", action="store_true",
                        help="Force recreation of data splits even if they exist on disk")
    parser.add_argument("--no-test-set", action="store_true", default=None,
                        help="No test set - all held-out sessions are used for validation only")
    parser.add_argument("--with-test-set", action="store_false", dest="no_test_set",
                        help="Keep test set (default)")
    parser.add_argument("--separate-val-sessions", action="store_true", default=None,
                        help="Evaluate each validation session separately (per-session metrics)")
    parser.add_argument("--no-separate-val-sessions", action="store_false", dest="separate_val_sessions",
                        help="Combine all validation sessions (default)")

    # Conditioning mode override
    COND_MODES = ["none", "cross_attn_gated"]
    parser.add_argument("--cond-mode", type=str, default=None,
                        choices=COND_MODES,
                        help="Override conditioning mode")

    # Attention type override (for ablation studies)
    ATTENTION_TYPES = ["none", "basic", "cross_freq", "cross_freq_v2"]
    parser.add_argument("--attention-type", type=str, default=None,
                        choices=ATTENTION_TYPES,
                        help="Override attention type (for Phase 3 ablation studies)")

    # Convolution type override (for ablation studies)
    CONV_TYPES = ["standard", "modern"]
    parser.add_argument("--conv-type", type=str, default=None,
                        choices=CONV_TYPES,
                        help="Override convolution type: 'standard' (basic Conv1d) or 'modern' (dilated depthwise separable + SE)")

    # Conditioning source: how conditioning embeddings are derived
    COND_SOURCES = ["odor_onehot", "spectro_temporal", "cpc", "vqvae", "freq_disentangled", "cycle_consistent"]
    parser.add_argument("--conditioning", type=str, default="spectro_temporal",
                        choices=COND_SOURCES,
                        help="Conditioning source: 'spectro_temporal' (default, auto-conditioning from signal dynamics), "
                             "'odor_onehot' (uses odor labels), "
                             "'cpc' (contrastive predictive coding), 'vqvae' (vector quantized), "
                             "'freq_disentangled' (per-band encoding), 'cycle_consistent' (cycle loss)")

    # Training mode control
    parser.add_argument("--no-bidirectional", action="store_true",
                        help="Disable bidirectional training (only train OB→PCx, no cycle consistency)")

    # Data augmentation
    parser.add_argument("--no-aug", action="store_true",
                        help="Disable all data augmentations")
    parser.add_argument("--aug-strength", type=str, default=None,
                        choices=["none", "light", "medium", "heavy"],
                        help="Augmentation strength level: "
                             "'none' = no augmentation, "
                             "'light' = time shift + noise only, "
                             "'medium' = + channel dropout + amplitude scale, "
                             "'heavy' = all augmentations (default)")

    # =========================================================================
    # Phase 3 Ablation Study Arguments (Nature Methods level)
    # =========================================================================

    # Normalization type
    NORM_TYPES = ["batch", "layer", "instance", "group", "rms", "none"]
    parser.add_argument("--norm-type", type=str, default=None,
                        choices=NORM_TYPES,
                        help="Normalization layer type for ablation studies")

    # Skip connection type
    SKIP_TYPES = ["add", "concat", "attention", "dense"]
    parser.add_argument("--skip-type", type=str, default=None,
                        choices=SKIP_TYPES,
                        help="Skip connection type: 'add' (residual), 'concat' (U-Net), 'attention' (gated), 'dense'")

    # Activation function
    ACTIVATION_TYPES = ["relu", "leaky_relu", "gelu", "silu", "mish"]
    parser.add_argument("--activation", type=str, default=None,
                        choices=ACTIVATION_TYPES,
                        help="Activation function for ablation studies")

    # Dropout rate
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout rate (0.0 to 0.5) for ablation studies")

    # Optimizer
    OPTIMIZER_TYPES = ["adamw", "adam", "sgd", "lion", "adafactor"]
    parser.add_argument("--optimizer", type=str, default=None,
                        choices=OPTIMIZER_TYPES,
                        help="Optimizer for ablation studies")

    # Learning rate schedule (alias for --lr-scheduler)
    LR_SCHEDULE_TYPES = ["cosine", "cosine_warmup", "step", "plateau", "onecycle", "constant"]
    parser.add_argument("--lr-schedule", type=str, default=None,
                        choices=LR_SCHEDULE_TYPES,
                        help="Learning rate schedule (alias for --lr-scheduler)")

    # Number of attention heads
    parser.add_argument("--n-heads", type=int, default=None,
                        help="Number of attention heads for ablation studies")

    # Network depth (number of downsample levels)
    parser.add_argument("--n-downsample", type=int, default=None,
                        help="Number of encoder/decoder downsample levels (depth) for ablation studies")

    # Session embedding for session-specific adjustments
    parser.add_argument("--use-session-embedding", action="store_true",
                        help="Enable session embedding for session-specific model adjustments")
    parser.add_argument("--n-sessions", type=int, default=None,
                        help="Number of sessions for session embedding (auto-detected if not specified)")
    parser.add_argument("--session-emb-dim", type=int, default=32,
                        help="Session embedding dimension (default: 32)")

    # NEW: Statistics-based session adaptation (Phase 3 Group 18)
    parser.add_argument("--use-session-stats", action="store_true",
                        help="Use statistics-based session conditioning (FiLM style) instead of ID embedding")
    parser.add_argument("--session-use-spectral", action="store_true",
                        help="Include spectral features in session statistics encoder")
    parser.add_argument("--use-adaptive-scaling", action="store_true",
                        help="Use session-adaptive output scaling (AdaIN style)")
    parser.add_argument("--use-cov-augment", action="store_true",
                        help="Use covariance expansion augmentation for synthetic sessions")
    parser.add_argument("--cov-augment-prob", type=float, default=0.5,
                        help="Probability of applying covariance augmentation (default: 0.5)")

    # Validation plot generation
    parser.add_argument("--generate-plots", action="store_true", default=None,
                        help="Generate validation plots at end of training (default: True)")
    parser.add_argument("--no-plots", action="store_false", dest="generate_plots",
                        help="Skip validation plot generation")

    # Output scaling correction (learnable per-channel scale and bias in model)
    parser.add_argument("--output-scaling", action="store_true", default=True,
                        help="Enable learnable per-channel output scaling in model (default: True)")
    parser.add_argument("--no-output-scaling", action="store_false", dest="output_scaling",
                        help="Disable output scaling correction in model")

    # Loss function selection (for tier1 fair comparison)
    LOSS_CHOICES = ["l1", "huber", "wavelet", "l1_wavelet", "huber_wavelet"]
    parser.add_argument("--loss", type=str, default=None,
                        choices=LOSS_CHOICES,
                        help="Loss function: 'l1' (L1/MAE only), 'huber' (Huber only), "
                             "'wavelet' (Wavelet only), 'l1_wavelet' (L1 + Wavelet), "
                             "'huber_wavelet' (Huber + Wavelet combined). "
                             "If not specified, uses config default (huber_wavelet)")

    # Phase 2 cross-validation integration
    parser.add_argument("--fold-indices-file", type=str, default=None,
                        help="Pickle file with train/val indices for CV fold (used by Phase 2 runner). "
                             "File should contain dict with 'train_idx' and 'val_idx' numpy arrays.")
    parser.add_argument("--output-results-file", type=str, default=None,
                        help="JSON file to save training results (used by Phase 2 runner). "
                             "Contains best_r2, best_mae, best_epoch, train_losses, val_r2s, etc.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold number for Phase 2 CV logging (0-indexed)")
    parser.add_argument("--checkpoint-prefix", type=str, default=None,
                        help="Prefix for checkpoint file names (e.g., 'wavenet_fold0')")

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize distributed
    dist_init_if_needed()

    # Setup logging
    if is_primary():
        setup_logging()
        print(f"Python: {sys.version}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
        print(f"Args: {vars(args)}")
        print()

    # Create output directories
    if is_primary():
        for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    # Prepare data
    if is_primary():
        print("Loading data...")
        print(f"Dataset: {args.dataset.upper()}")

    # Build config FIRST - only override from args if explicitly provided
    config = DEFAULT_CONFIG.copy()

    # Architecture selection (for Phase 2 comparison)
    config["arch"] = args.arch

    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay
    if args.lr_scheduler is not None:
        config["lr_scheduler"] = args.lr_scheduler
    if args.lr_warmup_epochs is not None:
        config["lr_warmup_epochs"] = args.lr_warmup_epochs
    if args.lr_min_ratio is not None:
        config["lr_min_ratio"] = args.lr_min_ratio
    if args.base_channels is not None:
        config["base_channels"] = args.base_channels
    if args.seed is not None:
        config["seed"] = args.seed

    # Training speed optimizations
    config["val_every"] = args.val_every  # Validate every N epochs
    config["gradient_checkpointing"] = args.gradient_checkpointing
    config["num_workers"] = args.num_workers  # None = auto
    config["prefetch_factor"] = args.prefetch_factor

    # Session-based split config (CLI overrides config if explicitly provided)
    if args.split_by_session:
        config["split_by_session"] = True
    elif args.no_split_by_session:
        config["split_by_session"] = False
    # else: use value from config
    if args.n_test_sessions is not None:
        config["n_test_sessions"] = args.n_test_sessions
    if args.n_val_sessions is not None:
        config["n_val_sessions"] = args.n_val_sessions
    if args.session_column is not None:
        config["session_column"] = args.session_column
    if args.no_test_set is not None:
        config["no_test_set"] = args.no_test_set
    if args.separate_val_sessions is not None:
        config["separate_val_sessions"] = args.separate_val_sessions

    # Session embedding configuration
    if args.use_session_embedding:
        # n_sessions will be auto-detected from data if not specified
        config["use_session_embedding"] = True
        if args.n_sessions is not None:
            config["n_sessions"] = args.n_sessions
        # session_emb_dim is set via CLI (default 32)
        config["session_emb_dim"] = args.session_emb_dim
    else:
        config["n_sessions"] = 0  # Disabled
        config["session_emb_dim"] = 32

    # NEW: Statistics-based session adaptation (Phase 3 Group 18)
    config["use_session_stats"] = args.use_session_stats
    config["session_use_spectral"] = args.session_use_spectral
    config["use_adaptive_scaling"] = args.use_adaptive_scaling
    config["use_cov_augment"] = args.use_cov_augment
    config["cov_augment_prob"] = args.cov_augment_prob

    # Print session split info
    if is_primary() and config["split_by_session"]:
        if config.get("no_test_set", False):
            print(f"Using SESSION-BASED SPLITS: NO TEST SET, {config['n_val_sessions']} val sessions")
        else:
            print(f"Using SESSION-BASED SPLITS: {config['n_test_sessions']} test, {config['n_val_sessions']} val sessions")
        print(f"Session column: {config['session_column']}")
        if config.get("separate_val_sessions", False):
            print("  Per-session validation ENABLED (metrics reported per session)")

    # Load data based on dataset choice
    if args.dataset == "pcx1":
        # PCx1 continuous LFP dataset with sliding windows
        if is_primary():
            print("Loading PCx1 continuous data with sliding windows...")

        # Get session splits
        all_sessions = list_pcx1_sessions()
        if is_primary():
            print(f"Available sessions: {all_sessions}")

        if args.pcx1_train_sessions and args.pcx1_val_sessions:
            # Use explicitly specified sessions
            train_sessions = args.pcx1_train_sessions
            val_sessions = args.pcx1_val_sessions
        else:
            # Random split: use pcx1_n_val for validation, rest for training
            train_sessions, val_sessions, _ = get_pcx1_session_splits(
                seed=config["seed"],
                n_val=args.pcx1_n_val,
                n_test=0,  # No test set for now
            )
            # Allow CLI override of val sessions
            if args.pcx1_val_sessions:
                val_sessions = args.pcx1_val_sessions
                train_sessions = [s for s in all_sessions if s not in val_sessions]

        # Calculate stride: --pcx1-stride-ratio overrides --pcx1-stride
        window_size = args.pcx1_window_size
        if args.pcx1_stride_ratio is not None:
            train_stride = int(window_size * args.pcx1_stride_ratio)
        elif args.pcx1_stride is not None:
            train_stride = args.pcx1_stride
        else:
            train_stride = window_size // 2  # Default: 50% overlap

        # Validation stride can be larger for faster eval
        val_stride_mult = args.val_stride_multiplier
        val_stride = int(train_stride * val_stride_mult)

        # Calculate data multiplication factor (how many times each sample is seen)
        overlap_ratio = 1.0 - (train_stride / window_size)
        data_mult = window_size / train_stride if train_stride > 0 else float('inf')

        if is_primary():
            print(f"Train sessions ({len(train_sessions)}): {train_sessions}")
            print(f"Val sessions ({len(val_sessions)}): {val_sessions}")
            print(f"Window size: {window_size} samples ({window_size / PCX1_SAMPLING_RATE:.1f} sec)")
            print(f"Train stride: {train_stride} samples ({train_stride / PCX1_SAMPLING_RATE:.2f} sec, {overlap_ratio*100:.0f}% overlap)")
            print(f"  -> Each time point seen in ~{data_mult:.1f}x windows (data multiplication)")
            if val_stride_mult > 1.0:
                print(f"Val stride: {val_stride} samples ({val_stride_mult:.1f}x train stride for faster eval)")

        # Auto-detect num_workers if not specified
        # When running multi-GPU, reduce workers per process to avoid shared memory exhaustion
        num_workers = config["num_workers"]
        if num_workers is None:
            import os
            world_size = get_world_size()
            cpu_count = os.cpu_count() or 8
            # With 8 GPUs, use 2 workers per GPU; with 1 GPU, use up to 8
            num_workers = max(2, min(8, cpu_count // max(1, world_size)))
            # Further reduce workers for large batch sizes to avoid /dev/shm exhaustion
            if config["batch_size"] >= 64:
                num_workers = max(1, num_workers // 2)
        prefetch_factor = config["prefetch_factor"]

        # Reduce prefetch and disable persistent_workers when running multi-GPU to save shared memory
        use_persistent = True
        if get_world_size() > 1:
            if prefetch_factor > 2:
                prefetch_factor = 2
            # Disable persistent_workers with many GPUs to avoid /dev/shm exhaustion
            use_persistent = False
            # Further reduce for large batches
            if config["batch_size"] >= 64:
                prefetch_factor = 1

        if is_primary():
            print(f"DataLoader: {num_workers} workers, prefetch={prefetch_factor}, persistent={use_persistent}")

        # Create dataloaders
        loaders = create_pcx1_dataloaders(
            train_sessions=train_sessions,
            val_sessions=val_sessions,
            test_sessions=None,
            window_size=window_size,
            stride=train_stride,
            val_stride=val_stride,
            batch_size=config["batch_size"],
            zscore_per_window=True,
            num_workers=num_workers,
            separate_val_sessions=config.get("separate_val_sessions", True),
            persistent_workers=use_persistent,
            prefetch_factor=prefetch_factor,
        )

        # Build a minimal data dict for compatibility with rest of training loop
        data = {
            "train_loader": loaders["train"],
            "val_loader": loaders["val"],
            "val_sessions_loaders": loaders.get("val_sessions"),  # For per-session eval
            "train_sessions": train_sessions,
            "val_sessions": val_sessions,
            "n_odors": 1,  # No odor conditioning for continuous data
            "vocab": {"none": 0},
        }

        config["dataset_type"] = "pcx1"
        config["in_channels"] = 32   # OB channels
        config["out_channels"] = 32  # PCx channels
        config["sampling_rate"] = PCX1_SAMPLING_RATE
        config["pcx1_window_size"] = window_size
        config["pcx1_stride"] = train_stride
        config["pcx1_val_stride"] = val_stride
        config["split_by_session"] = True  # Always session-based for PCx1

    elif args.dataset == "pfc":
        # PFC/Hippocampus dataset
        data = prepare_pfc_data(
            split_by_session=config["split_by_session"],
            n_test_sessions=config["n_test_sessions"],
            n_val_sessions=config["n_val_sessions"],
            force_recreate_splits=args.force_recreate_splits,
            resample_to_1khz=args.resample_pfc,
        )
        # Set dataset-specific config
        config["dataset_type"] = "pfc"
        config["in_channels"] = 64   # PFC channels
        config["out_channels"] = 32  # CA1 channels
        config["sampling_rate"] = SAMPLING_RATE_HZ if args.resample_pfc else PFC_SAMPLING_RATE_HZ
        # Map PFC data keys to generic names for training loop
        data["n_odors"] = data["n_labels"]  # trial types instead of odors

        # PFC sliding window configuration
        if args.pfc_sliding_window:
            config["pfc_sliding_window"] = True
            config["pfc_window_size"] = args.pfc_window_size
            # Calculate stride from ratio if specified
            if args.pfc_stride_ratio is not None:
                config["pfc_stride"] = int(args.pfc_window_size * args.pfc_stride_ratio)
            elif args.pfc_stride is not None:
                config["pfc_stride"] = args.pfc_stride
            else:
                config["pfc_stride"] = args.pfc_window_size // 2  # 50% overlap default
            # Validation stride: use window size for non-overlapping (faster eval)
            config["pfc_val_stride"] = int(args.pfc_window_size * args.val_stride_multiplier)
            if is_primary():
                print(f"PFC sliding window: size={config['pfc_window_size']}, "
                      f"stride={config['pfc_stride']}, val_stride={config['pfc_val_stride']}")

    elif args.dataset == "dandi":
        # DANDI 000623 human iEEG movie watching dataset
        # Calculate stride
        window_size = args.dandi_window_size
        if args.dandi_stride_ratio is not None:
            train_stride = int(window_size * args.dandi_stride_ratio)
        elif args.dandi_stride is not None:
            train_stride = args.dandi_stride
        else:
            train_stride = window_size // 2  # 50% overlap default

        val_stride = int(window_size * args.val_stride_multiplier)

        # Use default DANDI data dir from data.py if not specified
        dandi_data_dir = Path(args.dandi_data_dir) if args.dandi_data_dir else _DANDI_DATA_DIR

        if is_primary():
            print(f"\nLoading DANDI 000623 dataset...")
            print(f"  Data directory: {dandi_data_dir}")
            print(f"  Source region: {args.dandi_source_region}")
            print(f"  Target region: {args.dandi_target_region}")
            print(f"  Window size: {window_size}, stride: {train_stride}, val_stride: {val_stride}")

        # Prepare DANDI data - this returns datasets directly
        dandi_data = prepare_dandi_data(
            data_dir=dandi_data_dir,
            source_region=args.dandi_source_region,
            target_region=args.dandi_target_region,
            window_size=window_size,
            stride=train_stride,
            seed=config["seed"],
            verbose=is_primary(),  # Only print from rank 0 in distributed training
        )

        # Create a minimal data dict for compatibility with training loop
        # DANDI uses sliding window so we bypass the standard data loading
        data = {
            "train_idx": list(range(len(dandi_data["train_dataset"]))),
            "val_idx": list(range(len(dandi_data["val_dataset"]))),
            "test_idx": list(range(len(dandi_data["test_dataset"]))),
            "n_odors": 1,  # No conditioning labels for DANDI
            "vocab": {"movie": 0},  # Placeholder
        }

        # Store DANDI-specific config
        config["dataset_type"] = "dandi"
        config["dandi_sliding_window"] = True
        config["dandi_window_size"] = window_size
        config["dandi_stride"] = train_stride
        config["dandi_val_stride"] = val_stride
        config["dandi_source_region"] = args.dandi_source_region
        config["dandi_target_region"] = args.dandi_target_region
        config["dandi_data_dir"] = str(dandi_data_dir)
        config["sampling_rate"] = DANDI_SAMPLING_RATE_HZ

        # Channel counts from prepared data (normalized across all subjects)
        config["in_channels"] = dandi_data["n_source_channels"]
        config["out_channels"] = dandi_data["n_target_channels"]
        config["split_by_session"] = True  # Subject-based splits

        # Store the prepared datasets for later use
        config["_dandi_datasets"] = dandi_data

        if is_primary():
            print(f"  Train windows: {len(dandi_data['train_dataset'])}")
            print(f"  Val windows: {len(dandi_data['val_dataset'])}")
            print(f"  Test windows: {len(dandi_data['test_dataset'])}")
            print(f"  Source channels: {dandi_data['n_source_channels']}, Target channels: {dandi_data['n_target_channels']}")

    else:
        # Olfactory dataset (default)
        data = prepare_data(
            split_by_session=config["split_by_session"],
            n_test_sessions=config["n_test_sessions"],
            n_val_sessions=config["n_val_sessions"],
            session_column=config["session_column"],
            force_recreate_splits=args.force_recreate_splits,
            seed=config["seed"],
            test_sessions=args.test_sessions,
            val_sessions=args.val_sessions,
            no_test_set=config.get("no_test_set", False),
            separate_val_sessions=config.get("separate_val_sessions", False),
        )
        config["dataset_type"] = "olfactory"
        config["in_channels"] = 32   # OB channels
        config["out_channels"] = 32  # PCx channels
        config["sampling_rate"] = SAMPLING_RATE_HZ

        # Phase integration: override train/val/test indices if fold file provided
        if args.fold_indices_file is not None:
            import pickle
            with open(args.fold_indices_file, 'rb') as f:
                fold_data = pickle.load(f)
            data["train_idx"] = fold_data["train_idx"]
            data["val_idx"] = fold_data["val_idx"]
            # Support test_idx if provided (Phase 4), otherwise empty for CV (Phase 2)
            if "test_idx" in fold_data:
                data["test_idx"] = fold_data["test_idx"]
            else:
                data["test_idx"] = np.array([], dtype=int)
            if is_primary():
                fold_num = args.fold if args.fold is not None else "?"
                n_test = len(data["test_idx"])
                test_str = f", test={n_test}" if n_test > 0 else ""
                print(f"Custom split mode: fold {fold_num}, train={len(data['train_idx'])}, val={len(data['val_idx'])}{test_str}")

    # Conditioning from CLI
    if args.cond_mode is not None:
        config["cond_mode"] = args.cond_mode
    config["conditioning_source"] = args.conditioning  # odor_onehot, spectro_temporal, etc.

    # Attention type from CLI (for Phase 3 ablation studies)
    if args.attention_type is not None:
        config["attention_type"] = args.attention_type
        if is_primary():
            print(f"Attention type override: {args.attention_type}")

    # Convolution type from CLI (for Phase 3 ablation studies)
    if args.conv_type is not None:
        config["conv_type"] = args.conv_type
        if is_primary():
            print(f"Convolution type override: {args.conv_type}")

    # Disable bidirectional training if requested (for fair architecture comparison)
    if args.no_bidirectional:
        config["use_bidirectional"] = False
        if is_primary():
            print("Bidirectional training DISABLED (--no-bidirectional)")

    # Data augmentation - strength levels
    if args.no_aug or args.aug_strength == "none":
        config["aug_enabled"] = False
    elif args.aug_strength is not None:
        # Apply augmentation strength presets
        config["aug_enabled"] = True
        if args.aug_strength == "light":
            # Light: only time shift + noise (minimal but effective)
            config["aug_time_shift"] = True
            config["aug_noise"] = True
            config["aug_channel_dropout"] = False
            config["aug_amplitude_scale"] = False
            config["aug_time_mask"] = False
            config["aug_mixup"] = False
            config["aug_freq_mask"] = False
            config["aug_channel_scale"] = False
            config["aug_dc_offset"] = False
        elif args.aug_strength == "medium":
            # Medium: + channel dropout + amplitude scale
            config["aug_time_shift"] = True
            config["aug_noise"] = True
            config["aug_channel_dropout"] = True
            config["aug_amplitude_scale"] = True
            config["aug_time_mask"] = False
            config["aug_mixup"] = False
            config["aug_freq_mask"] = False
            config["aug_channel_scale"] = True
            config["aug_dc_offset"] = False
        elif args.aug_strength == "heavy":
            # Heavy: all augmentations (default config behavior)
            config["aug_time_shift"] = True
            config["aug_noise"] = True
            config["aug_channel_dropout"] = True
            config["aug_amplitude_scale"] = True
            config["aug_time_mask"] = True
            config["aug_mixup"] = True
            config["aug_freq_mask"] = True
            config["aug_channel_scale"] = True
            config["aug_dc_offset"] = True
        if is_primary():
            print(f"Augmentation strength: {args.aug_strength.upper()}")

    # Plot generation config
    if args.generate_plots is not None:
        config["generate_plots"] = args.generate_plots
    if is_primary() and not config.get("generate_plots", True):
        print("Validation plot generation DISABLED (--no-plots)")

    # Print augmentation config
    if is_primary():
        if not config.get("aug_enabled", True):
            print("Data augmentation DISABLED (master toggle off)")
        else:
            aug_active = [
                k for k in [
                    "aug_time_shift", "aug_noise", "aug_channel_dropout", "aug_amplitude_scale",
                    "aug_time_mask", "aug_mixup", "aug_freq_mask",
                    "aug_channel_scale", "aug_dc_offset",  # Session-specific augmentations
                ]
                if config.get(k, False)
            ]
            if aug_active:
                print(f"Data augmentation ENABLED: {', '.join(aug_active)}")
                if config.get("aug_channel_scale", False) or config.get("aug_dc_offset", False):
                    print("  [Session-invariance augmentations active: channel_scale, dc_offset]")
            else:
                print("Data augmentation: all individual augmentations disabled")

    # Loss function selection (for tier1 fair comparison)
    # Only override config if --loss is explicitly provided
    # --loss l1: L1 only (no wavelet)
    # --loss huber: Huber only (no wavelet)
    # --loss wavelet: Wavelet only (no L1)
    # --loss l1_wavelet: L1 + Wavelet combined
    # --loss huber_wavelet: Huber + Wavelet combined
    if args.loss is not None:
        config["loss_type"] = args.loss
        if args.loss == "l1":
            config["use_wavelet_loss"] = False
            if is_primary():
                print("Loss: L1 only (--loss l1)")
        elif args.loss == "huber":
            config["use_wavelet_loss"] = False
            if is_primary():
                print("Loss: Huber only (--loss huber)")
        elif args.loss == "wavelet":
            config["use_wavelet_loss"] = True
            config["weight_l1"] = 0.0  # Disable L1/Huber, use only wavelet
            if is_primary():
                print("Loss: Wavelet only (--loss wavelet)")
        elif args.loss == "l1_wavelet":
            config["use_wavelet_loss"] = True
            if is_primary():
                print("Loss: L1 + Wavelet combined (--loss l1_wavelet)")
        elif args.loss == "huber_wavelet":
            config["use_wavelet_loss"] = True
            if is_primary():
                print(f"Loss: Huber + Wavelet combined (--loss huber_wavelet, wavelet_weight={config['weight_wavelet']})")
    else:
        # Use config default - print what's being used
        loss_type = config.get("loss_type", "huber_wavelet")
        if is_primary():
            print(f"Loss: {loss_type} (from config)")

    # Output scaling correction in model (default: enabled)
    config["use_output_scaling"] = args.output_scaling if hasattr(args, 'output_scaling') else True
    if is_primary():
        print(f"Output scaling correction: {'ENABLED' if config['use_output_scaling'] else 'DISABLED'}")

    if is_primary():
        arch_name = config.get('arch', 'condunet').upper()
        print(f"\nTraining {arch_name} for {config['num_epochs']} epochs...")
        if config.get('arch', 'condunet') == 'condunet':
            print(f"Attention type: {config['attention_type']}")
            print(f"Convolution type: {config['conv_type']}")
            if config['conv_type'] == 'modern':
                print(f"  -> Multi-scale dilated depthwise separable + SE attention")
                print(f"  -> Dilations: {config['conv_dilations']}, Kernel: {config['conv_kernel_size']}")
        print(f"Config: {config}")
        print()

    # Train
    results = train(
        config=config,
        data=data,
        use_fsdp=args.fsdp,
        fsdp_strategy=args.fsdp_strategy,
        cpu_offload=args.cpu_offload,
        compile_model=args.compile,
    )

    # Synchronize all ranks after training completes
    # This ensures all ranks are in sync before any rank-specific cleanup
    # Prevents NCCL timeout when non-primary ranks exit before primary finishes logging
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception as e:
            if is_primary():
                print(f"Warning: Post-training barrier failed: {e}")

    if is_primary():
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)

        # Show split type
        if "split_info" in data:
            split_info = data["split_info"]
            print(f"Split type: SESSION HOLDOUT")
            print(f"Test sessions: {split_info['test_sessions']} ({split_info['n_test_trials']} trials)")
            print(f"Val sessions: {split_info['val_sessions']} ({split_info['n_val_trials']} trials)")
            print(f"Train sessions: {split_info['train_sessions']} ({split_info['n_train_trials']} trials)")
        else:
            print(f"Split type: Random stratified")

        print(f"Best validation loss: {results['best_val_loss']:.4f} (epoch {results['best_epoch']})")

        # Handle case where no test set exists (using validation sessions instead)
        if results['test_metrics']:
            print(f"Test loss: {results['test_metrics'].get('loss', 'N/A')}")
            print(f"Test correlation: {results['test_metrics']['corr']:.4f}")
            print(f"Test R2: {results['test_metrics']['r2']:.4f}")
        else:
            print("No test set (using held-out validation sessions instead)")

        print(f"Model saved to: {CHECKPOINT_DIR / 'best_model.pt'}")

        # Machine-parseable results for experiment scripts
        if results['test_metrics']:
            print(f"RESULT_CORR={results['test_metrics']['corr']:.4f}")
            print(f"RESULT_R2={results['test_metrics']['r2']:.4f}")
        print(f"RESULT_LOSS={results['best_val_loss']:.4f}")
        if "split_info" in data:
            print(f"RESULT_SPLIT_TYPE=session_holdout")
            if data['split_info'].get('test_sessions'):
                print(f"RESULT_TEST_SESSIONS={data['split_info']['test_sessions']}")

        # Phase 2 CV integration: save results to JSON file
        if args.output_results_file is not None:
            import json
            # Extract metrics from history
            history = results.get("history", [])
            train_losses = [h.get("loss", 0.0) for h in history]
            val_losses = [h.get("val_loss", 0.0) for h in history]
            val_r2s = [h.get("val_r2", 0.0) for h in history]
            val_corrs = [h.get("val_corr", 0.0) for h in history]
            val_maes = [h.get("val_mae", 0.0) for h in history]

            # Get best validation R2 from best epoch
            best_epoch = results["best_epoch"]
            best_val_r2 = val_r2s[best_epoch - 1] if best_epoch > 0 and len(val_r2s) >= best_epoch else 0.0
            best_val_mae = val_maes[best_epoch - 1] if best_epoch > 0 and len(val_maes) >= best_epoch else 0.0
            best_val_corr = val_corrs[best_epoch - 1] if best_epoch > 0 and len(val_corrs) >= best_epoch else 0.0

            # Get parameter count - prefer pre-computed value (works correctly with FSDP)
            # FSDP shards parameters across ranks, so counting from wrapped model gives wrong result
            n_params = results.get("n_parameters", 0)
            if n_params == 0:
                # Fallback: try to count from model (may be inaccurate with FSDP)
                model_obj = results.get("model")
                if model_obj is not None:
                    if hasattr(model_obj, 'module'):
                        n_params = sum(p.numel() for p in model_obj.module.parameters() if p.requires_grad)
                    else:
                        n_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)

            output_results = {
                "architecture": args.arch,
                "fold": args.fold if args.fold is not None else 0,
                "best_val_r2": best_val_r2,
                "best_val_mae": best_val_mae,
                "best_val_corr": best_val_corr,
                "best_val_loss": results["best_val_loss"],
                "best_epoch": best_epoch,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_r2s": val_r2s,
                "val_corrs": val_corrs,
                "total_time": results.get("total_time", 0.0),
                "epochs_trained": len(history),
                "n_parameters": n_params,
                "completed_successfully": True,
            }
            output_path = Path(args.output_results_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output_results, f, indent=2, default=str)
            print(f"Phase 2 results saved to: {output_path}")

    # Final cleanup for distributed training
    # Note: Synchronization is done after train() returns (before logging).
    # This just cleans up the process group - no barrier needed here since
    # all ranks were already synchronized and the is_primary() block only does logging.
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Print detailed traceback for debugging distributed training crashes
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"\n{'='*70}", file=sys.stderr, flush=True)
        print(f"FATAL ERROR on rank {rank} (local_rank {local_rank})", file=sys.stderr, flush=True)
        print(f"{'='*70}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr, flush=True)
        sys.stderr.flush()
        raise
    finally:
        # Ensure distributed process group is cleaned up even on errors
        # This prevents NCCL hangs when one rank crashes
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass  # Ignore cleanup errors
