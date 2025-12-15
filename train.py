"""Training pipeline for neural signal translation with CondUNet1D.

Usage:
    # Train model (single GPU)
    python train.py --epochs 80

    # Distributed training with FSDP
    torchrun --nproc_per_node=4 train.py --epochs 80 --fsdp
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
)
from data import (
    prepare_data,
    prepare_pfc_data,
    create_dataloaders,
    create_pfc_dataloaders,
    crop_to_target_torch,
    SAMPLING_RATE_HZ,
    PFC_SAMPLING_RATE_HZ,
    DatasetType,
)

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

    # Multi-stage training:
    # Stage 1: Train UNet (L1 + wavelet)
    # Stage 2: Post-hoc calibration (OptimalSpectralBias + EnvelopeHistogramMatching)
    "use_two_stage": True,        # Enable multi-stage training
    "spectral_finetune_epochs": 20,  # Epochs for Stage 2 (post-hoc calibration)
    "stage2_only": False,         # Skip Stage 1, load checkpoint and run Stage 2 only
    "stage1_checkpoint": "best_model.pt",    # Path to Stage 1 checkpoint

    # Model
    "base_channels": 64,
    "dropout": 0.0,
    "use_attention": True,
    "attention_type": "cross_freq_v2",  # Cross-frequency coupling attention (theta-gamma)
    "norm_type": "batch",
    "cond_mode": "cross_attn_gated",  # Cross-attention with output gating
    
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

    # Bidirectional training
    "use_bidirectional": True,  # Train both OB→PCx and PCx→OB

    # Spectral shift block
    "use_spectral_shift": True,  # Per-channel amplitude scaling for PSD correction
    "spectral_shift_mode": "adaptive",  # "flat", "frequency_band", or "adaptive" (signal-adaptive)
    "spectral_shift_conditional": True,  # If True, use odor-specific spectral bias
    "spectral_shift_per_channel": False,  # Not used by OptimalSpectralBias (global bias per band)
    "spectral_shift_init_fwd": 0.0,  # Not used by OptimalSpectralBias (bias computed from data)
    "spectral_shift_lr": 0.001,  # Not used by OptimalSpectralBias (no training)
    "spectral_shift_lr_decay": 0.95,  # Not used by OptimalSpectralBias (no training)
    "spectral_shift_band_width_hz": 2,  # None=use predefined neuro bands, or float (e.g., 2.0 for 2Hz uniform bands)
    "spectral_shift_compute_bias": True,  # Compute optimal bias directly from UNet output vs target PSD

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
    "n_val_sessions": 1,        # Number of sessions to hold out for validation
    "session_column": "recording_id",  # CSV column containing session/recording IDs
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
    # Time shift: apply same shift to both (maintains correspondence)
    if config.get("aug_time_shift", False):
        max_shift = config.get("aug_time_shift_max", 0.1)
        # Generate shifts once, apply to both
        batch_size, _, time_len = ob.shape
        max_shift_samples = int(time_len * max_shift)
        if max_shift_samples > 0:
            shifts = torch.randint(-max_shift_samples, max_shift_samples + 1, (batch_size,), device=ob.device)
            ob_shifted = torch.zeros_like(ob)
            pcx_shifted = torch.zeros_like(pcx)
            for i in range(batch_size):
                ob_shifted[i] = torch.roll(ob[i], shifts=shifts[i].item(), dims=-1)
                pcx_shifted[i] = torch.roll(pcx[i], shifts=shifts[i].item(), dims=-1)
            ob, pcx = ob_shifted, pcx_shifted

    # Noise: add independently to both (realistic noise augmentation)
    if config.get("aug_noise", False):
        noise_std = config.get("aug_noise_std", 0.05)
        ob = aug_gaussian_noise(ob, noise_std)
        pcx = aug_gaussian_noise(pcx, noise_std)

    # Channel dropout: apply same mask to both (maintains channel correspondence)
    if config.get("aug_channel_dropout", False):
        dropout_p = config.get("aug_channel_dropout_p", 0.1)
        batch_size, n_channels, _ = ob.shape
        mask = torch.bernoulli(torch.full((batch_size, n_channels, 1), 1 - dropout_p, device=ob.device))
        ob = ob * mask
        pcx = pcx * mask

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
        batch_size, _, time_len = ob.shape
        mask_len = int(time_len * mask_ratio)
        if mask_len > 0:
            for i in range(batch_size):
                start = torch.randint(0, time_len - mask_len + 1, (1,)).item()
                ob[i, :, start:start + mask_len] = 0
                pcx[i, :, start:start + mask_len] = 0

    # Mixup: blend random pairs (applied to both input and target coherently)
    if config.get("aug_mixup", False):
        alpha = config.get("aug_mixup_alpha", 0.4)
        ob, pcx = aug_mixup(ob, pcx, alpha)

    # Frequency masking: zero out random freq bands (same bands for both)
    if config.get("aug_freq_mask", False):
        max_bands = config.get("aug_freq_mask_max_bands", 2)
        max_width = config.get("aug_freq_mask_max_width", 10)
        # Apply same masking to both by concatenating, masking, and splitting
        batch_size = ob.shape[0]
        combined = torch.cat([ob, pcx], dim=0)  # (2*batch, channels, time)
        # Generate mask indices once
        combined_fft = torch.fft.rfft(combined, dim=-1)
        n_freqs = combined_fft.shape[-1]
        if n_freqs > max_width:
            for i in range(batch_size):
                n_bands_to_mask = torch.randint(1, max_bands + 1, (1,)).item()
                for _ in range(n_bands_to_mask):
                    width = torch.randint(1, max_width + 1, (1,)).item()
                    start = torch.randint(0, n_freqs - width, (1,)).item()
                    # Apply same mask to both ob[i] and pcx[i]
                    combined_fft[i, :, start:start + width] = 0
                    combined_fft[batch_size + i, :, start:start + width] = 0
            combined = torch.fft.irfft(combined_fft, n=ob.shape[-1], dim=-1)
            ob, pcx = combined[:batch_size], combined[batch_size:]

    # Session-specific augmentations (simulate cross-session variability)

    # Per-channel scaling: apply same scale pattern to both (simulates electrode drift)
    if config.get("aug_channel_scale", False):
        scale_range = config.get("aug_channel_scale_range", (0.7, 1.4))
        batch_size, n_channels, _ = ob.shape
        min_scale, max_scale = scale_range
        # Same per-channel scale for both OB and PCx (electrodes drift together in same rig)
        scales = torch.empty(1, n_channels, 1, device=ob.device).uniform_(min_scale, max_scale)
        scales = scales.expand(batch_size, -1, -1)
        ob = ob * scales
        # Different scale pattern for PCx (different electrodes)
        scales_pcx = torch.empty(1, n_channels, 1, device=pcx.device).uniform_(min_scale, max_scale)
        scales_pcx = scales_pcx.expand(batch_size, -1, -1)
        pcx = pcx * scales_pcx

    # DC offset: apply independently (baseline drift is independent per electrode)
    if config.get("aug_dc_offset", False):
        offset_range = config.get("aug_dc_offset_range", (-0.3, 0.3))
        ob = aug_dc_offset(ob, offset_range)
        pcx = aug_dc_offset(pcx, offset_range)

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
    spectral_shift_fwd: Optional[nn.Module] = None,
    spectral_shift_rev: Optional[nn.Module] = None,
    load_spectral_only: bool = False,
    skip_spectral: bool = False,
    expected_cond_mode: Optional[str] = None,
) -> str:
    """Load checkpoint (with option to load only SpectralShift for fine-tuning).

    Args:
        skip_spectral: If True, skip loading SpectralShift (use fresh init instead).
                       Useful when SpectralShift architecture changed (e.g., different band_width_hz).
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

    if not load_spectral_only:
        # Load UNet (forward) - handle FSDP
        if isinstance(model, FSDP):
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

    # Load SpectralShift (DDP-wrapped, not FSDP)
    # Skip if skip_spectral=True (allows using fresh init with different architecture)
    if not skip_spectral:
        if spectral_shift_fwd is not None and "spectral_shift_fwd" in checkpoint:
            if hasattr(spectral_shift_fwd, 'module'):
                spectral_shift_fwd.module.load_state_dict(checkpoint["spectral_shift_fwd"])
            else:
                spectral_shift_fwd.load_state_dict(checkpoint["spectral_shift_fwd"])

        if spectral_shift_rev is not None and "spectral_shift_rev" in checkpoint:
            if hasattr(spectral_shift_rev, 'module'):
                spectral_shift_rev.module.load_state_dict(checkpoint["spectral_shift_rev"])
            else:
                spectral_shift_rev.load_state_dict(checkpoint["spectral_shift_rev"])
    
    return checkpoint_cond_mode


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
    is_fsdp: bool = False,
    reverse_model: Optional[nn.Module] = None,
    spectral_shift_fwd: Optional[nn.Module] = None,
    spectral_shift_rev: Optional[nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save checkpoint with FSDP support (includes all models)."""
    if is_fsdp and isinstance(model, FSDP):
        # FSDP models need special handling - use context manager for each FSDP model
        checkpoint = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            # Save model config for compatibility checking
            "cond_mode": config.get("cond_mode", "unknown") if config else "unknown",
        }

        # Save forward model
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

        # Save spectral shifts (DDP-wrapped, not FSDP)
        if is_primary():
            if spectral_shift_fwd is not None:
                shift_fwd_state = spectral_shift_fwd.module.state_dict() if hasattr(spectral_shift_fwd, 'module') else spectral_shift_fwd.state_dict()
                checkpoint["spectral_shift_fwd"] = shift_fwd_state
            if spectral_shift_rev is not None:
                shift_rev_state = spectral_shift_rev.module.state_dict() if hasattr(spectral_shift_rev, 'module') else spectral_shift_rev.state_dict()
                checkpoint["spectral_shift_rev"] = shift_rev_state

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
            # Save reverse model if exists
            if reverse_model is not None:
                rev_state = reverse_model.module.state_dict() if hasattr(reverse_model, 'module') else reverse_model.state_dict()
                checkpoint["reverse_model"] = rev_state
            # Save spectral shifts if exist
            if spectral_shift_fwd is not None:
                shift_fwd_state = spectral_shift_fwd.module.state_dict() if hasattr(spectral_shift_fwd, 'module') else spectral_shift_fwd.state_dict()
                checkpoint["spectral_shift_fwd"] = shift_fwd_state
            if spectral_shift_rev is not None:
                shift_rev_state = spectral_shift_rev.module.state_dict() if hasattr(spectral_shift_rev, 'module') else spectral_shift_rev.state_dict()
                checkpoint["spectral_shift_rev"] = shift_rev_state
            torch.save(checkpoint, path)


# =============================================================================
# Spectral Shift Monitoring
# =============================================================================

def get_spectral_shift_db(spectral_shift_module, detailed: bool = False) -> float | dict | None:
    """Get dB shift from OptimalSpectralBias (handles DDP wrapping).

    Args:
        spectral_shift_module: OptimalSpectralBias (may be DDP-wrapped)
        detailed: If True, return dict with detailed info

    Returns:
        Mean dB shift (or detailed dict), or None if module is None
    """
    if spectral_shift_module is None:
        return None

    # Handle DDP wrapping - access the underlying module
    module = spectral_shift_module.module if hasattr(spectral_shift_module, 'module') else spectral_shift_module

    # OptimalSpectralBias stores bias in bias_db parameter [n_odors, n_bands]
    if hasattr(module, 'bias_db'):
        bias_db = module.bias_db.detach()
        mean_shift = bias_db.mean().item()
        if detailed:
            return {
                "mean": mean_shift,
                "per_odor_mean": bias_db.mean(dim=1).cpu().tolist(),  # Mean per odor
                "per_band_mean": bias_db.mean(dim=0).cpu().tolist(),  # Mean per band
                "mode": "optimal_bias",
            }
        return mean_shift
    return None


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
    spectral_shift_fwd: Optional[nn.Module] = None,
    spectral_shift_rev: Optional[nn.Module] = None,
    spectral_only: bool = False,
    disable_spectral: bool = False,
    fast_mode: bool = True,  # Skip expensive metrics (PSD, phase, baseline) during training
    sampling_rate: int = SAMPLING_RATE_HZ,  # Sampling rate for PSD calculations
    cond_encoder: Optional[nn.Module] = None,
    envelope_matcher_fwd: Optional[nn.Module] = None,
    envelope_matcher_rev: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate model on a dataloader (supports bidirectional).

    Returns composite validation loss that mirrors training loss.

    Args:
        spectral_only: If True, Stage 2 mode (UNet frozen, only SpectralShift active)
        disable_spectral: If True, disable SpectralShift (Stage 1 mode)
        fast_mode: If True, skip expensive metrics (PSD, phase, baseline) for faster validation.
                   Use fast_mode=False only for final evaluation.
        cond_encoder: Optional conditioning encoder for auto-conditioning modes
    """
    model.eval()
    if reverse_model is not None:
        reverse_model.eval()
    if cond_encoder is not None:
        cond_encoder.eval()
    if spectral_shift_fwd is not None:
        spectral_shift_fwd.eval()
    if spectral_shift_rev is not None:
        spectral_shift_rev.eval()

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

    # Determine compute dtype for FSDP mixed precision compatibility
    use_bf16 = config.get("fsdp_bf16", False) if config else False
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float32

    with torch.inference_mode():  # Faster than no_grad() - disables view tracking
        for ob, pcx, odor in loader:
            ob = ob.to(device, dtype=compute_dtype, non_blocking=True)
            pcx = pcx.to(device, dtype=compute_dtype, non_blocking=True)
            odor = odor.to(device, non_blocking=True)

            # Apply per-channel normalization if enabled (default: True)
            if config is not None and config.get("per_channel_norm", True):
                ob = per_channel_normalize(ob)
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

            # Forward: OB → PCx (use cond_emb if available, otherwise odor_ids)
            if cond_emb is not None:
                pred = model(ob, cond_emb=cond_emb)
            else:
                pred = model(ob, odor)

            # Apply spectral shift OUTSIDE the model (FSDP-safe)
            # Pass odor_ids for conditional spectral shift
            # NOTE: Skip in Stage 1 (disable_spectral=True)
            if spectral_shift_fwd is not None and not disable_spectral:
                pred = spectral_shift_fwd(pred, odor_ids=odor)

            # Apply envelope histogram matching (closed-form correction for amplitude dynamics)
            # This corrects bursty vs smooth characteristics - applied after spectral shift
            if envelope_matcher_fwd is not None and not disable_spectral:
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

            # Reverse: PCx → OB (if reverse model exists)
            if reverse_model is not None:
                if cond_emb is not None:
                    pred_rev = reverse_model(pcx, cond_emb=cond_emb)
                else:
                    pred_rev = reverse_model(pcx, odor)

                # Apply spectral shift with SEPARATE reverse module
                # Each direction learns independently - no inverse constraint
                # NOTE: Skip in Stage 1 (disable_spectral=True)
                if spectral_shift_rev is not None and not disable_spectral:
                    pred_rev = spectral_shift_rev(pred_rev, odor_ids=odor)

                # Apply envelope histogram matching (reverse direction)
                if envelope_matcher_rev is not None and not disable_spectral:
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

    # Compute composite validation loss (mirrors training loss)
    # This allows early stopping based on overall objective, not just correlation
    if config is not None:
        if spectral_only:
            # Stage 2: Use PSD error for validation (SpectralShift optimization)
            val_loss = results.get("psd_err_db", results["mae"])
            if "psd_err_db_rev" in results:
                val_loss += results["psd_err_db_rev"]
        else:
            # Stage 1: L1 + wavelet
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
    spectral_shift_fwd: Optional[nn.Module] = None,
    spectral_shift_rev: Optional[nn.Module] = None,
    stage2_spectral_only: bool = False,
    disable_spectral: bool = False,
    cond_encoder: Optional[nn.Module] = None,
    projection_head_fwd: Optional[nn.Module] = None,
    projection_head_rev: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Train one epoch (supports bidirectional with cycle consistency).

    Args:
        stage2_spectral_only: If True, Stage 2 mode (UNet frozen, SpectralShift active)
        disable_spectral: If True, disable SpectralShift application (Stage 1 mode)
        cond_encoder: Optional conditioning encoder for auto-conditioning modes
    """
    model.train()
    if reverse_model is not None:
        reverse_model.train()
    if cond_encoder is not None:
        cond_encoder.train()
    if spectral_shift_fwd is not None:
        spectral_shift_fwd.train()
    if spectral_shift_rev is not None:
        spectral_shift_rev.train()
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

    for ob, pcx, odor in pbar:
        # non_blocking=True enables async CPU->GPU transfer (overlaps with compute)
        ob = ob.to(device, dtype=compute_dtype, non_blocking=True)
        pcx = pcx.to(device, dtype=compute_dtype, non_blocking=True)
        odor = odor.to(device, non_blocking=True)

        # Apply per-channel normalization if enabled (default: True)
        if config.get("per_channel_norm", True):
            ob = per_channel_normalize(ob)
            pcx = per_channel_normalize(pcx)

        # Apply data augmentation (training only)
        ob, pcx = apply_augmentations(ob, pcx, config)

        # Compute conditioning embedding
        cond_emb = None
        cond_loss = 0.0
        if cond_encoder is not None:
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

        # Forward: OB → PCx (use cond_emb if available, otherwise odor_ids)
        # If contrastive learning enabled, also get bottleneck features
        use_contrastive = config.get("use_contrastive", False) and projection_head_fwd is not None
        if cond_emb is not None:
            fwd_result = model(ob, cond_emb=cond_emb, return_bottleneck=use_contrastive)
        else:
            fwd_result = model(ob, odor, return_bottleneck=use_contrastive)

        if use_contrastive:
            pred_raw, bottleneck_fwd = fwd_result
        else:
            pred_raw = fwd_result
            bottleneck_fwd = None

        pred_raw_c = crop_to_target_torch(pred_raw)
        pcx_c = crop_to_target_torch(pcx)
        ob_c = crop_to_target_torch(ob)

        # Apply spectral shift OUTSIDE the model (FSDP-safe)
        # CRITICAL: Detach pred_raw so UNet does NOT get gradients from spectral loss!
        # This ensures clean separation of responsibilities:
        # - UNet handles waveform matching (L1 + wavelet loss)
        # - SpectralShift ALONE handles PSD correction (spectral loss)
        # Without detach, UNet and SpectralShift fight each other trying to fix PSD.
        # NOTE: In Stage 1 (disable_spectral=True), skip SpectralShift entirely
        if spectral_shift_fwd is not None and not disable_spectral:
            pred_shifted = spectral_shift_fwd(pred_raw.detach(), odor_ids=odor)
            pred_shifted_c = crop_to_target_torch(pred_shifted)
        else:
            pred_shifted = pred_raw
            pred_shifted_c = pred_raw_c

        # Stage 2 (UNet frozen): OptimalSpectralBias computes bias from statistics (no training needed)
        # Stage 1 (UNet trainable): L1 + wavelet
        if stage2_spectral_only:
            # Stage 2: No loss needed - OptimalSpectralBias uses closed-form solution
            # Just use wavelet loss if available for validation consistency
            if config.get("use_wavelet_loss", True) and wavelet_loss is not None:
                loss = config["weight_wavelet"] * wavelet_loss(pred_shifted_c, pcx_c)
                loss_components["wavelet_fwd"] = loss_components["wavelet_fwd"] + loss.detach()
            else:
                loss = torch.tensor(0.0, device=device)

        else:
            # Stage 1: L1/Huber + wavelet
            # Reconstruction loss (forward) - uses pred_raw (no SpectralShift gradient)
            loss_type = config.get("loss_type", "huber_wavelet")
            if loss_type in ("huber", "huber_wavelet"):
                recon_loss = config["weight_l1"] * F.huber_loss(pred_raw_c, pcx_c)
            else:
                recon_loss = config["weight_l1"] * F.l1_loss(pred_raw_c, pcx_c)
            loss = recon_loss
            loss_components["l1_fwd"] = loss_components["l1_fwd"] + recon_loss.detach()

            # Wavelet loss (forward) - uses pred_raw (no SpectralShift gradient)
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
            # Reverse: PCx → OB (use same cond_emb from forward - conditioning is symmetric)
            # If contrastive learning enabled, also get bottleneck features
            if cond_emb is not None:
                rev_result = reverse_model(pcx, cond_emb=cond_emb, return_bottleneck=use_contrastive)
            else:
                rev_result = reverse_model(pcx, odor, return_bottleneck=use_contrastive)

            if use_contrastive:
                pred_rev_raw, bottleneck_rev = rev_result
            else:
                pred_rev_raw = rev_result

            pred_rev_raw_c = crop_to_target_torch(pred_rev_raw)

            # Apply spectral shift with SEPARATE reverse module
            # CRITICAL: Detach so reverse UNet doesn't get spectral loss gradients
            # SpectralShift alone handles PSD correction for reverse direction
            # NOTE: In Stage 1 (disable_spectral=True), skip SpectralShift entirely
            if spectral_shift_rev is not None and not disable_spectral:
                pred_rev_shifted = spectral_shift_rev(pred_rev_raw.detach(), odor_ids=odor)
                pred_rev_shifted_c = crop_to_target_torch(pred_rev_shifted)
            else:
                pred_rev_shifted = pred_rev_raw
                pred_rev_shifted_c = pred_rev_raw_c

            # Stage 2: No training needed for reverse (OptimalSpectralBias uses closed-form)
            if stage2_spectral_only:
                # Just use wavelet loss if available for validation consistency
                if config.get("use_wavelet_loss", True) and wavelet_loss is not None:
                    w_loss_rev = config["weight_wavelet"] * wavelet_loss(pred_rev_shifted_c, ob_c)
                    loss = loss + w_loss_rev
                    loss_components["wavelet_rev"] = loss_components["wavelet_rev"] + w_loss_rev.detach()
            else:
                # Stage 1: L1/Huber + wavelet
                # Reconstruction loss (reverse) - uses pred_rev_raw (no SpectralShift gradient)
                loss_type = config.get("loss_type", "huber_wavelet")
                if loss_type in ("huber", "huber_wavelet"):
                    rev_loss = config["weight_l1"] * F.huber_loss(pred_rev_raw_c, ob_c)
                else:
                    rev_loss = config["weight_l1"] * F.l1_loss(pred_rev_raw_c, ob_c)
                loss = loss + rev_loss
                loss_components["l1_rev"] = loss_components["l1_rev"] + rev_loss.detach()

                # Wavelet loss (reverse) - uses pred_rev_raw (no SpectralShift gradient)
                if config.get("use_wavelet_loss", True) and wavelet_loss is not None:
                    w_loss_rev = config["weight_wavelet"] * wavelet_loss(pred_rev_raw_c, ob_c)
                    loss = loss + w_loss_rev
                    loss_components["wavelet_rev"] = loss_components["wavelet_rev"] + w_loss_rev.detach()

                # Cycle consistency: OB → PCx → OB (use raw, no spectral shift in cycle)
                # Skip in Stage 2 since UNet is frozen
                if cond_emb is not None:
                    cycle_ob = reverse_model(pred_raw, cond_emb=cond_emb)
                else:
                    cycle_ob = reverse_model(pred_raw, odor)
                cycle_ob_c = crop_to_target_torch(cycle_ob)
                cycle_loss_ob = config.get("cycle_lambda", 1.0) * F.l1_loss(cycle_ob_c, ob_c)
                loss = loss + cycle_loss_ob
                loss_components["cycle_ob"] = loss_components["cycle_ob"] + cycle_loss_ob.detach()

                # Cycle consistency: PCx → OB → PCx (use raw, no spectral shift in cycle)
                if cond_emb is not None:
                    cycle_pcx = model(pred_rev_raw, cond_emb=cond_emb)
                else:
                    cycle_pcx = model(pred_rev_raw, odor)
                cycle_pcx_c = crop_to_target_torch(cycle_pcx)
                cycle_loss_pcx = config.get("cycle_lambda", 1.0) * F.l1_loss(cycle_pcx_c, pcx_c)
                loss = loss + cycle_loss_pcx
                loss_components["cycle_pcx"] = loss_components["cycle_pcx"] + cycle_loss_pcx.detach()

        # Contrastive loss for session-invariant learning (CEBRA-style)
        # Uses BOTTLENECK features (encoder output) with odor labels as positive pairs
        # This encourages the encoder to learn session-invariant representations:
        # same odor, different session/augmentation → similar bottleneck embedding
        if use_contrastive and bottleneck_fwd is not None:
            contrastive_weight = config.get("contrastive_weight", 0.1)
            contrastive_temp = config.get("contrastive_temperature", 0.1)

            # Project bottleneck features to embedding space
            # bottleneck_fwd: (batch, bottleneck_channels) - already pooled in model forward
            fwd_embed = projection_head_fwd(bottleneck_fwd)  # -> (batch, 128)

            # Compute contrastive loss (same odor = positive pair)
            # Cast to float32 for stable loss computation
            contrastive_loss_fwd = info_nce_loss(fwd_embed.float(), odor, temperature=contrastive_temp)
            loss = loss + contrastive_weight * contrastive_loss_fwd
            loss_components["contrastive_fwd"] = loss_components["contrastive_fwd"] + contrastive_loss_fwd.detach()

            # Reverse direction contrastive loss (on reverse model's bottleneck)
            if reverse_model is not None and projection_head_rev is not None and bottleneck_rev is not None:
                rev_embed = projection_head_rev(bottleneck_rev)
                contrastive_loss_rev = info_nce_loss(rev_embed.float(), odor, temperature=contrastive_temp)
                loss = loss + contrastive_weight * contrastive_loss_rev
                loss_components["contrastive_rev"] = loss_components["contrastive_rev"] + contrastive_loss_rev.detach()

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
        # NOTE: We intentionally do NOT clip gradients for SpectralShift modules.
        # They have only 32 params each with high lr (0.1), clipping would prevent convergence.
        # The spectral loss gradient is naturally bounded by log PSD differences.
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Accumulate loss as tensor - NO .item() call to avoid GPU sync
        total_loss = total_loss + loss.detach()

    pbar.close()
    sys.stdout.flush()

    # Convert to floats ONLY at end of epoch (single GPU sync point)
    n_batches = len(loader)
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
    if config.get("dataset_type") == "pfc":
        loaders = create_pfc_dataloaders(
            data,
            batch_size=config.get("batch_size", 16),
            num_workers=num_workers,
            distributed=is_distributed,
        )
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

    if is_primary():
        print(f"Model: {in_channels} input channels → {out_channels} output channels")
        print(f"Conditions: {n_odors} classes")

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
        use_spectral_shift=config.get("use_spectral_shift", True),
        # U-Net depth for frequency resolution
        n_downsample=config.get("n_downsample", 2),
        # Modern convolution options
        conv_type=conv_type,
        use_se=config.get("use_se", True),
        conv_kernel_size=config.get("conv_kernel_size", 7),
        dilations=config.get("conv_dilations", (1, 4, 16, 32)),
        # Output scaling correction
        use_output_scaling=config.get("use_output_scaling", True),
    )

    # Create reverse model (target → source) for bidirectional training
    reverse_model = None
    if config.get("use_bidirectional", False):
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
            use_spectral_shift=config.get("use_spectral_shift", True),
            # U-Net depth (same as forward)
            n_downsample=config.get("n_downsample", 2),
            # Modern convolution options (same as forward)
            conv_type=conv_type,
            use_se=config.get("use_se", True),
            conv_kernel_size=config.get("conv_kernel_size", 7),
            dilations=config.get("conv_dilations", (1, 4, 16, 32)),
            # Output scaling correction
            use_output_scaling=config.get("use_output_scaling", True),
        )
        if is_primary():
            print("Bidirectional training ENABLED")
            if conv_type == "modern":
                print(f"Using MODERN convolutions: dilations={config.get('conv_dilations', (1, 4, 16, 32))}, kernel_size={config.get('conv_kernel_size', 7)}, SE={config.get('use_se', True)}")

    # Create conditioning encoder for auto-conditioning modes
    cond_source = config.get("conditioning_source", "odor_onehot")
    cond_encoder = None
    emb_dim = 128  # Must match model's emb_dim

    if cond_source != "odor_onehot":
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

    # Log U-Net depth and frequency resolution
    n_downsample = config.get("n_downsample", 2)
    base_ch = config.get("base_channels", 128)
    downsample_factor = 2 ** n_downsample
    nyquist_hz = 1000 / (2 * downsample_factor)  # 1000 Hz sample rate
    
    # Calculate channel progression for logging
    channels = [base_ch]
    for i in range(n_downsample):
        channels.append(min(base_ch * (2 ** (i + 1)), base_ch * 8))
    
    # Count actual parameters
    model_params = sum(p.numel() for p in model.parameters())
    rev_params = sum(p.numel() for p in reverse_model.parameters()) if reverse_model else 0
    
    if is_primary():
        print(f"U-Net depth: {n_downsample} levels → {downsample_factor}x downsample → bottleneck Nyquist = {nyquist_hz:.0f} Hz")
        print(f"Channel progression: {' → '.join(map(str, channels))} (bottleneck={channels[-1]})")
        print(f"Model parameters: {model_params:,} (forward) + {rev_params:,} (reverse) = {model_params + rev_params:,} total")

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
            model = model.to(device)
            model = DDP(model, device_ids=[local_rank])
            if reverse_model is not None:
                reverse_model = reverse_model.to(device)
                reverse_model = DDP(reverse_model, device_ids=[local_rank])
            if cond_encoder is not None:
                cond_encoder = cond_encoder.to(device)
                # NOTE: We intentionally do NOT wrap cond_encoder in DDP.
                # Auto-conditioning encoders (CPC, VQVAE, FreqDisentangled, SpectroTemporal)
                # have complex forward passes with auxiliary losses that cause DDP crashes
                # even with find_unused_parameters=True. Instead, we manually sync gradients
                # using sync_gradients_manual() after backward() but before optimizer.step().
            if is_primary():
                print(f"Using DDP with {get_world_size()} GPUs")
            dist.barrier()
    else:
        model = wrap_model_fsdp(model, local_rank, use_fsdp=False, compile_model=compile_model)
        if reverse_model is not None:
            reverse_model = wrap_model_fsdp(reverse_model, local_rank, use_fsdp=False, compile_model=compile_model)
        if cond_encoder is not None:
            cond_encoder = cond_encoder.to(device)

    # Create SpectralShift modules OUTSIDE of FSDP (too small for sharding overhead)
    # but DO wrap with DDP for gradient synchronization in distributed training
    from models import OptimalSpectralBias, EnvelopeHistogramMatching

    spectral_shift_fwd = None
    spectral_shift_rev = None
    if config.get("use_spectral_shift", True):
        # Fixed per-odor spectral bias correction (no signal-adaptive network!)
        # Computes optimal bias directly from UNet output vs target PSD difference
        sampling_rate = config.get("sampling_rate", SAMPLING_RATE_HZ)
        fwd_out_channels = config.get("out_channels", 32)  # Forward output channels
        rev_out_channels = config.get("in_channels", 32)   # Reverse output = forward input
        n_odors = config.get("n_odors", 7)
        band_width_hz = config.get("spectral_shift_band_width_hz", None)

        spectral_shift_fwd = OptimalSpectralBias(
            n_channels=fwd_out_channels,
            n_odors=n_odors,
            sample_rate=sampling_rate,
            band_width_hz=band_width_hz,
        ).to(device)
        spectral_shift_rev = OptimalSpectralBias(
            n_channels=rev_out_channels,
            n_odors=n_odors,
            sample_rate=sampling_rate,
            band_width_hz=band_width_hz,
        ).to(device)

        if band_width_hz is not None:
            band_str = f", {spectral_shift_fwd.n_bands} bands @ {band_width_hz}Hz"
        else:
            band_str = ", 10 neuro bands"
        mode_str = f"optimal_bias (fixed per-odor, n_odors={n_odors}{band_str})"

        # Wrap with DDP for distributed training (gradient sync across ranks)
        # Note: NOT using FSDP since these are tiny
        # CRITICAL: Must use find_unused_parameters=True because in Stage 1 (two-stage training)
        # SpectralShift modules are created but NOT used in forward pass (disable_spectral=True).
        # Without this flag, DDP hangs waiting for gradient sync on unused parameters.
        if is_distributed:
            spectral_shift_fwd = DDP(spectral_shift_fwd, device_ids=[local_rank], find_unused_parameters=True)
            spectral_shift_rev = DDP(spectral_shift_rev, device_ids=[local_rank], find_unused_parameters=True)

        if is_primary():
            ddp_str = " (DDP-wrapped)" if is_distributed else ""
            print(f"SpectralShift created{ddp_str} mode={mode_str}")

    # Create projection heads for contrastive learning (if enabled)
    projection_head_fwd = None
    projection_head_rev = None
    use_contrastive = config.get("use_contrastive", False)

    if use_contrastive:
        # CEBRA-style: Apply contrastive loss on BOTTLENECK features (encoder output)
        # This encourages the encoder to learn session-invariant representations
        # Bottleneck dim = min(base_channels * 2^n_downsample, base_channels * 8)
        base_ch = config.get("base_channels", 64)
        n_downsample = config.get("n_downsample", 4)
        bottleneck_dim = min(base_ch * (2 ** n_downsample), base_ch * 8)

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
            print(f"Contrastive learning ENABLED (CEBRA-style): weight={contrastive_weight}, temperature={contrastive_temp}")
            print(f"  Projection heads: {bottleneck_dim}->256->128 (bottleneck features)")

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
    # SpectralShift needs MUCH higher lr because it's just 32 scalars trying to make dB-scale changes
    lr = config.get("learning_rate", 1e-4)
    spectral_shift_lr = config.get("spectral_shift_lr", 0.1)  # 500x higher for fast amplitude adaptation

    # Build parameter groups with different learning rates
    param_groups = [
        {"params": list(model.parameters()), "lr": lr},
    ]
    if reverse_model is not None:
        param_groups.append({"params": list(reverse_model.parameters()), "lr": lr})
    if cond_encoder is not None:
        # Conditioning encoder uses same lr as model
        param_groups.append({"params": list(cond_encoder.parameters()), "lr": lr, "name": "cond_encoder"})
    if spectral_shift_fwd is not None:
        param_groups.append({"params": list(spectral_shift_fwd.parameters()), "lr": spectral_shift_lr, "name": "spectral_shift_fwd"})
    if spectral_shift_rev is not None:
        param_groups.append({"params": list(spectral_shift_rev.parameters()), "lr": spectral_shift_lr, "name": "spectral_shift_rev"})
    # Projection heads for contrastive learning (use same lr as model)
    if projection_head_fwd is not None:
        param_groups.append({"params": list(projection_head_fwd.parameters()), "lr": lr, "name": "projection_head_fwd"})
    if projection_head_rev is not None:
        param_groups.append({"params": list(projection_head_rev.parameters()), "lr": lr, "name": "projection_head_rev"})

    total_params = sum(len(list(pg["params"])) if not isinstance(pg["params"], list) else len(pg["params"]) for pg in param_groups)

    # Weight decay (L2 regularization)
    weight_decay = config.get("weight_decay", 0.0)

    if is_primary():
        print(f"Optimizer: {total_params} total params | model lr={lr}, spectral_shift lr={spectral_shift_lr}, weight_decay={weight_decay}")

    optimizer = AdamW(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)

    # Learning rate scheduler configuration
    spectral_shift_lr_decay = config.get("spectral_shift_lr_decay", 0.95)
    num_epochs = config.get("num_epochs", 80)
    lr_scheduler_type = config.get("lr_scheduler", "none")
    lr_warmup_epochs = config.get("lr_warmup_epochs", 5)
    lr_min_ratio = config.get("lr_min_ratio", 0.01)

    def make_lr_lambda(is_spectral_shift: bool):
        """Create lr lambda for a param group."""
        def lr_lambda(epoch):
            # Base multiplier from scheduler type
            if lr_scheduler_type == "none":
                base_mult = 1.0
            elif lr_scheduler_type == "cosine":
                # Cosine annealing from 1.0 to lr_min_ratio
                progress = epoch / max(num_epochs - 1, 1)
                base_mult = lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            elif lr_scheduler_type == "cosine_warmup":
                # Linear warmup then cosine decay
                if epoch < lr_warmup_epochs:
                    base_mult = (epoch + 1) / lr_warmup_epochs
                else:
                    progress = (epoch - lr_warmup_epochs) / max(num_epochs - lr_warmup_epochs - 1, 1)
                    base_mult = lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            else:
                base_mult = 1.0

            # SpectralShift always gets exponential decay on top
            if is_spectral_shift:
                return base_mult * (spectral_shift_lr_decay ** epoch)
            return base_mult
        return lr_lambda

    # Build per-group lr lambdas
    lr_lambdas = []
    for pg in param_groups:
        is_spectral = pg.get("name", "").startswith("spectral_shift")
        lr_lambdas.append(make_lr_lambda(is_spectral))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)

    if is_primary():
        print(f"LR scheduler: {lr_scheduler_type}" + (f" (warmup={lr_warmup_epochs}, min_ratio={lr_min_ratio})" if lr_scheduler_type != "none" else ""))
        print(f"SpectralShift lr decay: {spectral_shift_lr_decay} per epoch")

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

    # Two-stage training info: UNet convergence → SpectralShift fine-tuning
    use_two_stage = config.get("use_two_stage", False)
    spectral_finetune_epochs = config.get("spectral_finetune_epochs", 10)
    stage2_only = config.get("stage2_only", False)
    stage1_checkpoint = config.get("stage1_checkpoint", None)

    # Validate stage2_only requirements
    if stage2_only:
        if stage1_checkpoint is None:
            raise ValueError("stage2_only requires stage1_checkpoint path to be set!")

        # Smart checkpoint path resolution - check multiple locations
        stage1_checkpoint = Path(stage1_checkpoint)
        if not stage1_checkpoint.exists():
            # Try common locations
            candidates = [
                CHECKPOINT_DIR / stage1_checkpoint.name,  # artifacts/checkpoints/<filename>
                CHECKPOINT_DIR / stage1_checkpoint,        # artifacts/checkpoints/<full_path>
                Path("artifacts/checkpoints") / stage1_checkpoint.name,
            ]
            found = False
            for candidate in candidates:
                if candidate.exists():
                    stage1_checkpoint = candidate
                    found = True
                    if is_primary():
                        print(f"[INFO] Resolved checkpoint path: {stage1_checkpoint}")
                    break
            if not found:
                raise FileNotFoundError(
                    f"Stage 1 checkpoint not found: {stage1_checkpoint}\n"
                    f"  Also checked: {[str(c) for c in candidates]}"
                )
        if is_primary():
            print(f"\n{'='*70}")
            print("STAGE 2 ONLY MODE")
            print(f"{'='*70}")
            print(f"  Skipping Stage 1 - loading checkpoint from:")
            print(f"    {stage1_checkpoint}")
            print(f"  Running Stage 2: Post-hoc calibration")
            print(f"{'='*70}\n")

    elif is_primary() and use_two_stage:
        print(f"\n{'='*70}")
        print("MULTI-STAGE TRAINING ENABLED")
        print(f"{'='*70}")
        print(f"  Stage 1: Train UNet + SpectralShift until early stopping")
        print(f"  Stage 2: Post-hoc calibration (OptimalSpectralBias + EnvelopeHistogramMatching)")
        print(f"{'='*70}\n")

    early_stop_patience = config.get("early_stop_patience", 8)

    # =============================================================================
    # STAGE 1: Train UNet + SpectralShift together until early stopping
    # (SKIPPED if stage2_only=True)
    # =============================================================================
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    skip_stage1 = stage2_only
    if not skip_stage1:
        if is_primary() and use_two_stage:
            print(f"\n{'='*70}")
            print("STAGE 1: Training UNet + SpectralShift together")
            print(f"{'='*70}\n")

        for epoch in range(1, num_epochs + 1):
            if loaders.get("train_sampler") is not None:
                loaders["train_sampler"].set_epoch(epoch)

            train_metrics = train_epoch(
                model, loaders["train"], optimizer, device, config,
                wavelet_loss,
                reverse_model, epoch, num_epochs,
                spectral_shift_fwd, spectral_shift_rev,
                disable_spectral=use_two_stage,  # Stage 1: Disable spectral if two-stage (pure UNet training)
                cond_encoder=cond_encoder,
                projection_head_fwd=projection_head_fwd,
                projection_head_rev=projection_head_rev,
            )

            barrier()

            val_metrics = evaluate(
                model, loaders["val"], device, wavelet_loss,
                compute_phase=False, reverse_model=reverse_model, config=config,
                spectral_shift_fwd=spectral_shift_fwd, spectral_shift_rev=spectral_shift_rev,
                disable_spectral=use_two_stage,  # Stage 1: Disable spectral if two-stage (pure UNet validation)
                fast_mode=True,  # Stage 1: Only compute r and r² (skip PSD metrics)
                sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
                cond_encoder=cond_encoder,
            )

            # Sync val_loss across ranks (for early stopping)
            val_loss = val_metrics.get("loss", val_metrics["mae"])  # fallback to mae if no composite
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
                    spectral_shift_fwd=spectral_shift_fwd,
                    spectral_shift_rev=spectral_shift_rev,
                    config=config,
                )
            else:
                patience_counter += 1

            barrier()

            if is_primary():
                rev_str = ""
                if "corr_rev" in val_metrics:
                    rev_str = f" | Rev: r={val_metrics['corr_rev']:.3f}, r²={val_metrics.get('r2_rev', 0):.3f}"

                print(f"Epoch {epoch}/{num_epochs} | "
                      f"Train: {train_metrics['loss']:.3f} | Val: {val_metrics['loss']:.3f} | "
                      f"Fwd: r={val_metrics['corr']:.3f}, r²={val_metrics.get('r2', 0):.3f}{rev_str} | "
                      f"Best: {best_val_loss:.3f}")
                sys.stdout.flush()

            history.append({"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}})

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

            # Step the lr scheduler (decays SpectralShift lr)
            scheduler.step()

            if patience_counter >= early_stop_patience:
                if is_primary():
                    print(f"Early stopping at epoch {epoch} (Stage 1 complete)")
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
                spectral_shift_fwd=spectral_shift_fwd, spectral_shift_rev=spectral_shift_rev,
                disable_spectral=use_two_stage,
                fast_mode=False,  # Full metrics for stage evaluation
                sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
            )

            barrier()

            # Evaluate on TEST set - ALL ranks must call this (FSDP requirement)
            test_metrics_stage1 = evaluate(
                model, loaders["test"], device, wavelet_loss,
                compute_phase=True, reverse_model=reverse_model, config=config,
                spectral_shift_fwd=spectral_shift_fwd, spectral_shift_rev=spectral_shift_rev,
                disable_spectral=use_two_stage,
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
    # STAGE 2: POST-HOC CALIBRATION (OptimalSpectralBias + EnvelopeHistogramMatching)
    # =============================================================================
    # Only run Stage 2 if:
    # 1. Two-stage training is enabled OR stage2_only mode
    # 2. SpectralShift is available
    # 3. spectral_finetune_epochs > 0 (skip if --skip-spectral-finetune was used)

    # Initialize envelope matchers OUTSIDE the block so they're always defined
    envelope_matcher_fwd = None
    envelope_matcher_rev = None

    should_run_stage2 = (
        (use_two_stage or stage2_only) and
        spectral_shift_fwd is not None and
        spectral_finetune_epochs > 0
    )

    if should_run_stage2:
        # Determine checkpoint path: user-provided (stage2_only) or best from stage1
        if stage2_only:
            # Loading from user-provided checkpoint
            checkpoint_path = stage1_checkpoint
            checkpoint_source = f"user-provided: {stage1_checkpoint}"
        else:
            # Use Stage 1 checkpoint
            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
            checkpoint_source = f"best from Stage 1 (epoch {best_epoch})"

        if is_primary():
            print(f"\n{'='*70}")
            print(f"STAGE 2: POST-HOC CALIBRATION (OptimalSpectralBias + EnvelopeHistogramMatching)")
            print(f"{'='*70}")
            print(f"Loading UNet from checkpoint: {checkpoint_source}")

        # Load UNet checkpoint
        barrier()
        expected_cond = config.get("cond_mode") if stage2_only else None
        load_checkpoint(
            checkpoint_path,
            model, reverse_model,
            spectral_shift_fwd, spectral_shift_rev,
            load_spectral_only=False,
            skip_spectral=stage2_only,
            expected_cond_mode=expected_cond,
        )
        barrier()

        if stage2_only and is_primary():
            print(f"SpectralShift: using FRESH initialization (not loaded from checkpoint)")

        # FREEZE UNet parameters (forward and reverse) - stays frozen for all of Stage 2
        freeze_model_params(model)
        if reverse_model is not None:
            freeze_model_params(reverse_model)

        spectral_shift_lr_finetune = config.get("spectral_shift_lr", 0.1)

        # Split epochs between phases (can be configured)
        phase_epochs = spectral_finetune_epochs // 2
        if phase_epochs < 1:
            phase_epochs = spectral_finetune_epochs  # If only 1 epoch, use it for both

        # =====================================================================
        # COMPUTE OPTIMAL BIAS FROM DATA (if enabled)
        # =====================================================================
        compute_bias_from_data = config.get("spectral_shift_compute_bias", True)

        if compute_bias_from_data and spectral_shift_fwd is not None:
            if is_primary():
                print(f"\n{'='*70}")
                print("COMPUTING OPTIMAL BIAS FROM UNet OUTPUT vs TARGET PSD")
                print(f"{'='*70}")

            # Collect UNet outputs and targets for all training data
            model.eval()
            if reverse_model is not None:
                reverse_model.eval()

            all_unet_outputs = []
            all_targets = []
            all_odor_ids = []

            # Use ONLY TRAIN data for computing bias (never val or test!)
            # This ensures proper train/test separation
            bias_loaders = [("train", loaders["train"])]

            with torch.no_grad():
                for loader_name, loader in bias_loaders:
                    for ob_batch, pcx_batch, odor_batch in tqdm(loader, desc=f"Computing bias ({loader_name})", disable=not is_primary()):
                        ob_batch = ob_batch.to(device)
                        pcx_batch = pcx_batch.to(device)
                        odor_batch = odor_batch.to(device)

                        # Apply per-channel normalization if enabled
                        if config.get("per_channel_norm", True):
                            ob_batch = per_channel_normalize(ob_batch)
                            pcx_batch = per_channel_normalize(pcx_batch)

                        # Forward pass through UNet (frozen)
                        unet_out = model(ob_batch, odor_batch)

                        all_unet_outputs.append(unet_out.cpu())
                        all_targets.append(pcx_batch.cpu())
                        all_odor_ids.append(odor_batch.cpu())

            # Concatenate all batches
            all_unet_outputs = torch.cat(all_unet_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_odor_ids = torch.cat(all_odor_ids, dim=0)

            if is_primary():
                print(f"Collected {len(all_unet_outputs)} samples")

            # Compute and set optimal bias for FORWARD SpectralShift
            spectral_shift_fwd_module = get_module(spectral_shift_fwd)
            spectral_shift_fwd_module.set_bias_from_data(
                all_unet_outputs.to(device),
                all_targets.to(device),
                all_odor_ids.to(device),
            )

            # Also compute for REVERSE if exists
            if spectral_shift_rev is not None and reverse_model is not None:
                all_rev_outputs = []
                all_rev_targets = []
                all_rev_odor_ids = []

                with torch.no_grad():
                    for loader_name, loader in bias_loaders:
                        for ob_batch, pcx_batch, odor_batch in tqdm(loader, desc=f"Computing reverse bias ({loader_name})", disable=not is_primary()):
                            ob_batch = ob_batch.to(device)
                            pcx_batch = pcx_batch.to(device)
                            odor_batch = odor_batch.to(device)

                            # Apply per-channel normalization if enabled
                            if config.get("per_channel_norm", True):
                                ob_batch = per_channel_normalize(ob_batch)
                                pcx_batch = per_channel_normalize(pcx_batch)

                            # Reverse pass: PCx -> OB
                            rev_out = reverse_model(pcx_batch, odor_batch)

                            all_rev_outputs.append(rev_out.cpu())
                            all_rev_targets.append(ob_batch.cpu())
                            all_rev_odor_ids.append(odor_batch.cpu())

                all_rev_outputs = torch.cat(all_rev_outputs, dim=0)
                all_rev_targets = torch.cat(all_rev_targets, dim=0)
                all_rev_odor_ids = torch.cat(all_rev_odor_ids, dim=0)

                spectral_shift_rev_module = get_module(spectral_shift_rev)
                spectral_shift_rev_module.set_bias_from_data(
                    all_rev_outputs.to(device),
                    all_rev_targets.to(device),
                    all_rev_odor_ids.to(device),
                )

            # Clean up spectral bias data
            del all_unet_outputs, all_odor_ids
            if spectral_shift_rev is not None:
                del all_rev_outputs, all_rev_targets, all_rev_odor_ids
            torch.cuda.empty_cache()

            barrier()

        # =====================================================================
        # COMPUTE ENVELOPE MATCHING FROM TARGET DATA
        # =====================================================================
        # EnvelopeHistogramMatching corrects amplitude dynamics (bursty vs smooth)
        # This is the third closed-form correction after output scaling and spectral bias
        use_envelope_matching = config.get("use_envelope_matching", True)

        if use_envelope_matching and spectral_shift_fwd is not None:
            if is_primary():
                print(f"\n{'='*70}")
                print("COMPUTING ENVELOPE MATCHING FROM TARGET DATA")
                print(f"{'='*70}")

            n_odors = config.get("n_odors", 7)

            # Create envelope matchers
            envelope_matcher_fwd = EnvelopeHistogramMatching(n_odors=n_odors).to(device)
            envelope_matcher_rev = EnvelopeHistogramMatching(n_odors=n_odors).to(device) if reverse_model is not None else None

            # Collect target data for fitting (reuse from spectral bias if available)
            all_targets_fwd = []
            all_targets_rev = []
            all_odor_ids_env = []

            with torch.no_grad():
                # Use ONLY TRAIN data for envelope statistics (never val or test!)
                # This ensures proper train/test separation
                for ob_batch, pcx_batch, odor_batch in tqdm(loaders["train"], desc="Collecting train targets", disable=not is_primary()):
                    pcx_batch = pcx_batch.to(device)
                    ob_batch = ob_batch.to(device)
                    odor_batch = odor_batch.to(device)

                    # Apply per-channel normalization if enabled
                    if config.get("per_channel_norm", True):
                        ob_batch = per_channel_normalize(ob_batch)
                        pcx_batch = per_channel_normalize(pcx_batch)

                    all_targets_fwd.append(pcx_batch.cpu())
                    all_targets_rev.append(ob_batch.cpu())
                    all_odor_ids_env.append(odor_batch.cpu())

            all_targets_fwd = torch.cat(all_targets_fwd, dim=0)
            all_targets_rev = torch.cat(all_targets_rev, dim=0)
            all_odor_ids_env = torch.cat(all_odor_ids_env, dim=0)

            # Fit envelope matchers
            envelope_matcher_fwd.fit(all_targets_fwd.to(device), all_odor_ids_env.to(device))
            if envelope_matcher_rev is not None:
                envelope_matcher_rev.fit(all_targets_rev.to(device), all_odor_ids_env.to(device))

            # Clean up
            del all_targets_fwd, all_targets_rev, all_odor_ids_env
            torch.cuda.empty_cache()

            barrier()

        # =====================================================================
        # DEBUG: Save distribution plots (envelope, PSD, instantaneous frequency)
        # =====================================================================
        if is_primary() and envelope_matcher_fwd is not None:
            print(f"\n{'='*70}")
            print("DEBUG: Computing distributions (envelope, PSD, inst. frequency)")
            print(f"{'='*70}")

            import json
            from models import hilbert_torch
            from scipy.signal import welch

            # Create dedicated debug folder
            debug_dir = Path("debug_plots")
            debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Saving debug plots to: {debug_dir.absolute()}")

            sampling_rate = config.get("sampling_rate", SAMPLING_RATE_HZ)

            debug_stats = {
                "target": {"mean": [], "std": [], "cv": []},
                "pred_raw": {"mean": [], "std": [], "cv": []},
                "pred_corrected": {"mean": [], "std": [], "cv": []},
            }

            # Collect ALL values for plotting
            all_target_env = []
            all_pred_raw_env = []
            all_pred_corr_env = []

            # Collect signals for PSD
            all_target_signals = []
            all_pred_raw_signals = []
            all_pred_corr_signals = []

            # Collect instantaneous frequencies
            all_target_inst_freq = []
            all_pred_raw_inst_freq = []
            all_pred_corr_inst_freq = []

            # Sample batches from validation set
            n_debug_batches = min(10, len(loaders["val"]))
            debug_iter = iter(loaders["val"])

            with torch.no_grad():
                for batch_idx in range(n_debug_batches):
                    ob_batch, pcx_batch, odor_batch = next(debug_iter)
                    ob_batch = ob_batch.to(device)
                    pcx_batch = pcx_batch.to(device)
                    odor_batch = odor_batch.to(device)

                    if config.get("per_channel_norm", True):
                        ob_batch = per_channel_normalize(ob_batch)
                        pcx_batch = per_channel_normalize(pcx_batch)

                    # Get UNet prediction
                    if cond_encoder is not None:
                        cond_source = config.get("conditioning_source", "odor_onehot")
                        if cond_source == "spectro_temporal":
                            cond_emb = cond_encoder(ob_batch)
                        else:
                            cond_emb = None
                        pred_raw = model(ob_batch, cond_emb=cond_emb) if cond_emb is not None else model(ob_batch, odor_batch)
                    else:
                        pred_raw = model(ob_batch, odor_batch)

                    # Apply spectral shift
                    pred_shifted = spectral_shift_fwd(pred_raw, odor_ids=odor_batch)

                    # Apply envelope correction
                    pred_corrected = envelope_matcher_fwd(pred_shifted, odor_ids=odor_batch)

                    # Compute analytic signals and envelopes
                    B, C, T = pcx_batch.shape

                    target_analytic = hilbert_torch(pcx_batch.view(B*C, T).float())
                    pred_raw_analytic = hilbert_torch(pred_shifted.view(B*C, T).float())
                    pred_corr_analytic = hilbert_torch(pred_corrected.view(B*C, T).float())

                    target_env = target_analytic.abs()
                    pred_raw_env = pred_raw_analytic.abs()
                    pred_corr_env = pred_corr_analytic.abs()

                    # Compute instantaneous frequency from phase derivative
                    # inst_freq = d(phase)/dt / (2*pi) * sampling_rate
                    target_phase = target_analytic.angle()
                    pred_raw_phase = pred_raw_analytic.angle()
                    pred_corr_phase = pred_corr_analytic.angle()

                    # Unwrap phase and compute derivative
                    def compute_inst_freq(phase, fs):
                        # phase: [N, T]
                        phase_np = phase.cpu().numpy()
                        inst_freq_list = []
                        for i in range(phase_np.shape[0]):
                            unwrapped = np.unwrap(phase_np[i])
                            # Derivative (central difference)
                            inst_freq = np.diff(unwrapped) * fs / (2 * np.pi)
                            inst_freq_list.append(inst_freq)
                        return np.concatenate(inst_freq_list)

                    target_inst_freq = compute_inst_freq(target_phase, sampling_rate)
                    pred_raw_inst_freq = compute_inst_freq(pred_raw_phase, sampling_rate)
                    pred_corr_inst_freq = compute_inst_freq(pred_corr_phase, sampling_rate)

                    # Collect envelope values
                    all_target_env.append(target_env.flatten().cpu().numpy())
                    all_pred_raw_env.append(pred_raw_env.flatten().cpu().numpy())
                    all_pred_corr_env.append(pred_corr_env.flatten().cpu().numpy())

                    # Collect signals for PSD (flatten channels, keep time)
                    # Convert to float32 first (bfloat16 not supported by numpy)
                    all_target_signals.append(pcx_batch.view(B*C, T).float().cpu().numpy())
                    all_pred_raw_signals.append(pred_shifted.view(B*C, T).float().cpu().numpy())
                    all_pred_corr_signals.append(pred_corrected.view(B*C, T).float().cpu().numpy())

                    # Collect instantaneous frequencies
                    all_target_inst_freq.append(target_inst_freq)
                    all_pred_raw_inst_freq.append(pred_raw_inst_freq)
                    all_pred_corr_inst_freq.append(pred_corr_inst_freq)

                    # Envelope stats
                    debug_stats["target"]["mean"].append(target_env.mean().item())
                    debug_stats["target"]["std"].append(target_env.std().item())
                    debug_stats["target"]["cv"].append((target_env.std() / target_env.mean().clamp(min=1e-8)).item())

                    debug_stats["pred_raw"]["mean"].append(pred_raw_env.mean().item())
                    debug_stats["pred_raw"]["std"].append(pred_raw_env.std().item())
                    debug_stats["pred_raw"]["cv"].append((pred_raw_env.std() / pred_raw_env.mean().clamp(min=1e-8)).item())

                    debug_stats["pred_corrected"]["mean"].append(pred_corr_env.mean().item())
                    debug_stats["pred_corrected"]["std"].append(pred_corr_env.std().item())
                    debug_stats["pred_corrected"]["cv"].append((pred_corr_env.std() / pred_corr_env.mean().clamp(min=1e-8)).item())

            # Concatenate all values
            all_target_env = np.concatenate(all_target_env)
            all_pred_raw_env = np.concatenate(all_pred_raw_env)
            all_pred_corr_env = np.concatenate(all_pred_corr_env)

            all_target_signals = np.vstack(all_target_signals)  # [N_total, T]
            all_pred_raw_signals = np.vstack(all_pred_raw_signals)
            all_pred_corr_signals = np.vstack(all_pred_corr_signals)

            all_target_inst_freq = np.concatenate(all_target_inst_freq)
            all_pred_raw_inst_freq = np.concatenate(all_pred_raw_inst_freq)
            all_pred_corr_inst_freq = np.concatenate(all_pred_corr_inst_freq)

            # Aggregate stats
            for key in ["target", "pred_raw", "pred_corrected"]:
                debug_stats[key]["mean_avg"] = sum(debug_stats[key]["mean"]) / len(debug_stats[key]["mean"])
                debug_stats[key]["std_avg"] = sum(debug_stats[key]["std"]) / len(debug_stats[key]["std"])
                debug_stats[key]["cv_avg"] = sum(debug_stats[key]["cv"]) / len(debug_stats[key]["cv"])

            # Print summary
            print(f"\nEnvelope Statistics (averaged over {n_debug_batches} batches):")
            print(f"  TARGET:     mean={debug_stats['target']['mean_avg']:.4f}, std={debug_stats['target']['std_avg']:.4f}, CV={debug_stats['target']['cv_avg']:.4f}")
            print(f"  PRED (raw): mean={debug_stats['pred_raw']['mean_avg']:.4f}, std={debug_stats['pred_raw']['std_avg']:.4f}, CV={debug_stats['pred_raw']['cv_avg']:.4f}")
            print(f"  PRED (fix): mean={debug_stats['pred_corrected']['mean_avg']:.4f}, std={debug_stats['pred_corrected']['std_avg']:.4f}, CV={debug_stats['pred_corrected']['cv_avg']:.4f}")

            print("\nGenerating debug plots...")

            # =====================================================================
            # PLOT 1: ENVELOPE DISTRIBUTIONS (overlay)
            # =====================================================================
            fig, ax = plt.subplots(figsize=(10, 6))

            all_vals = np.concatenate([all_target_env, all_pred_raw_env, all_pred_corr_env])
            bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 100)

            ax.hist(all_target_env, bins=bins, alpha=0.5, color='green', label=f'Target (μ={np.mean(all_target_env):.3f}, σ={np.std(all_target_env):.3f})', density=True)
            ax.hist(all_pred_raw_env, bins=bins, alpha=0.5, color='red', label=f'Pred Raw (μ={np.mean(all_pred_raw_env):.3f}, σ={np.std(all_pred_raw_env):.3f})', density=True)
            ax.hist(all_pred_corr_env, bins=bins, alpha=0.5, color='blue', label=f'Pred Corrected (μ={np.mean(all_pred_corr_env):.3f}, σ={np.std(all_pred_corr_env):.3f})', density=True)

            ax.set_title('Envelope Distribution Comparison')
            ax.set_xlabel('Envelope Amplitude')
            ax.set_ylabel('Density')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(debug_dir / "envelope_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {debug_dir / 'envelope_distribution.png'}")

            # =====================================================================
            # PLOT 2: PSD COMPARISON (Welch method)
            # =====================================================================
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Compute average PSD using Welch
            nperseg = min(1024, all_target_signals.shape[1] // 4)

            # Average PSD across all signals
            def compute_avg_psd(signals, fs, nperseg):
                psds = []
                for sig in signals[:100]:  # Limit to 100 signals for speed
                    f, psd = welch(sig, fs=fs, nperseg=nperseg)
                    psds.append(psd)
                return f, np.mean(psds, axis=0), np.std(psds, axis=0)

            f_target, psd_target, psd_target_std = compute_avg_psd(all_target_signals, sampling_rate, nperseg)
            f_raw, psd_raw, psd_raw_std = compute_avg_psd(all_pred_raw_signals, sampling_rate, nperseg)
            f_corr, psd_corr, psd_corr_std = compute_avg_psd(all_pred_corr_signals, sampling_rate, nperseg)

            # Linear scale PSD
            axes[0].semilogy(f_target, psd_target, 'g-', linewidth=2, label='Target')
            axes[0].semilogy(f_raw, psd_raw, 'r-', linewidth=2, label='Pred (raw)')
            axes[0].semilogy(f_corr, psd_corr, 'b-', linewidth=2, label='Pred (corrected)')
            axes[0].fill_between(f_target, psd_target - psd_target_std, psd_target + psd_target_std, alpha=0.2, color='green')
            axes[0].fill_between(f_raw, psd_raw - psd_raw_std, psd_raw + psd_raw_std, alpha=0.2, color='red')
            axes[0].fill_between(f_corr, psd_corr - psd_corr_std, psd_corr + psd_corr_std, alpha=0.2, color='blue')
            axes[0].set_xlabel('Frequency (Hz)')
            axes[0].set_ylabel('PSD (log scale)')
            axes[0].set_title('Power Spectral Density (Welch)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlim([0, min(150, sampling_rate/2)])

            # PSD difference in dB
            eps = 1e-10
            psd_diff_raw = 10 * np.log10((psd_raw + eps) / (psd_target + eps))
            psd_diff_corr = 10 * np.log10((psd_corr + eps) / (psd_target + eps))

            axes[1].plot(f_target, psd_diff_raw, 'r-', linewidth=2, label=f'Raw - Target (mean: {np.mean(psd_diff_raw):.2f} dB)')
            axes[1].plot(f_target, psd_diff_corr, 'b-', linewidth=2, label=f'Corrected - Target (mean: {np.mean(psd_diff_corr):.2f} dB)')
            axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
            axes[1].set_xlabel('Frequency (Hz)')
            axes[1].set_ylabel('PSD Difference (dB)')
            axes[1].set_title('PSD Difference from Target')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim([0, min(150, sampling_rate/2)])
            axes[1].set_ylim([-10, 10])

            plt.tight_layout()
            plt.savefig(debug_dir / "psd_welch.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {debug_dir / 'psd_welch.png'}")

            # =====================================================================
            # PLOT 3: INSTANTANEOUS FREQUENCY DISTRIBUTION
            # =====================================================================
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Filter to valid frequency range (0 to Nyquist)
            max_freq = sampling_rate / 2
            valid_mask_target = (all_target_inst_freq > 0) & (all_target_inst_freq < max_freq)
            valid_mask_raw = (all_pred_raw_inst_freq > 0) & (all_pred_raw_inst_freq < max_freq)
            valid_mask_corr = (all_pred_corr_inst_freq > 0) & (all_pred_corr_inst_freq < max_freq)

            target_freq_valid = all_target_inst_freq[valid_mask_target]
            raw_freq_valid = all_pred_raw_inst_freq[valid_mask_raw]
            corr_freq_valid = all_pred_corr_inst_freq[valid_mask_corr]

            # Histogram of instantaneous frequencies
            freq_bins = np.linspace(0, min(150, max_freq), 100)

            axes[0].hist(target_freq_valid, bins=freq_bins, alpha=0.5, color='green', label=f'Target (median: {np.median(target_freq_valid):.1f} Hz)', density=True)
            axes[0].hist(raw_freq_valid, bins=freq_bins, alpha=0.5, color='red', label=f'Pred Raw (median: {np.median(raw_freq_valid):.1f} Hz)', density=True)
            axes[0].hist(corr_freq_valid, bins=freq_bins, alpha=0.5, color='blue', label=f'Pred Corrected (median: {np.median(corr_freq_valid):.1f} Hz)', density=True)

            axes[0].set_xlabel('Instantaneous Frequency (Hz)')
            axes[0].set_ylabel('Density')
            axes[0].set_title('Instantaneous Frequency Distribution')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # CDF of instantaneous frequencies
            target_sorted = np.sort(target_freq_valid)
            raw_sorted = np.sort(raw_freq_valid)
            corr_sorted = np.sort(corr_freq_valid)

            step_t = max(1, len(target_sorted) // 1000)
            step_r = max(1, len(raw_sorted) // 1000)
            step_c = max(1, len(corr_sorted) // 1000)

            axes[1].plot(target_sorted[::step_t], np.linspace(0, 1, len(target_sorted))[::step_t], 'g-', linewidth=2, label='Target')
            axes[1].plot(raw_sorted[::step_r], np.linspace(0, 1, len(raw_sorted))[::step_r], 'r-', linewidth=2, label='Pred (raw)')
            axes[1].plot(corr_sorted[::step_c], np.linspace(0, 1, len(corr_sorted))[::step_c], 'b-', linewidth=2, label='Pred (corrected)')

            axes[1].set_xlabel('Instantaneous Frequency (Hz)')
            axes[1].set_ylabel('Cumulative Probability')
            axes[1].set_title('Instantaneous Frequency CDF')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim([0, min(150, max_freq)])

            plt.tight_layout()
            plt.savefig(debug_dir / "instantaneous_frequency.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {debug_dir / 'instantaneous_frequency.png'}")

            # =====================================================================
            # PLOT 4: Q-Q PLOTS (envelope)
            # =====================================================================
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            n_qq = min(10000, len(all_target_env))
            idx = np.random.choice(len(all_target_env), n_qq, replace=False)

            target_sorted = np.sort(all_target_env[idx])
            raw_sorted = np.sort(all_pred_raw_env[idx])
            corr_sorted = np.sort(all_pred_corr_env[idx])

            axes[0].scatter(target_sorted, raw_sorted, alpha=0.3, s=1, color='red')
            axes[0].plot([target_sorted.min(), target_sorted.max()], [target_sorted.min(), target_sorted.max()], 'k--', label='y=x')
            axes[0].set_xlabel('Target Envelope Quantiles')
            axes[0].set_ylabel('Pred (Raw) Envelope Quantiles')
            axes[0].set_title('Q-Q Plot: Raw vs Target')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].scatter(target_sorted, corr_sorted, alpha=0.3, s=1, color='blue')
            axes[1].plot([target_sorted.min(), target_sorted.max()], [target_sorted.min(), target_sorted.max()], 'k--', label='y=x')
            axes[1].set_xlabel('Target Envelope Quantiles')
            axes[1].set_ylabel('Pred (Corrected) Envelope Quantiles')
            axes[1].set_title('Q-Q Plot: Corrected vs Target')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(debug_dir / "envelope_qq.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {debug_dir / 'envelope_qq.png'}")

            # =====================================================================
            # PLOT 5: SIGNAL COMPARISON (Real vs Generated vs Corrected + Overlap)
            # =====================================================================
            # 4 columns: Real | Generated | Corrected | Overlap (all three)
            n_signals_to_plot = 4  # Number of example signals to show
            fig, axes = plt.subplots(n_signals_to_plot, 4, figsize=(20, 3 * n_signals_to_plot))

            # Pick random signals to display
            n_available = all_target_signals.shape[0]
            signal_indices = np.random.choice(n_available, min(n_signals_to_plot, n_available), replace=False)

            # Time axis (use first 1000 samples for clarity)
            T_display = min(1000, all_target_signals.shape[1])
            t = np.arange(T_display) / sampling_rate * 1000  # Convert to ms

            for row, sig_idx in enumerate(signal_indices):
                real_sig = all_target_signals[sig_idx, :T_display]
                gen_sig = all_pred_raw_signals[sig_idx, :T_display]
                corr_sig = all_pred_corr_signals[sig_idx, :T_display]

                # Compute correlations for titles
                corr_gen = np.corrcoef(real_sig, gen_sig)[0, 1]
                corr_corr = np.corrcoef(real_sig, corr_sig)[0, 1]

                # Column 1: Real signal
                axes[row, 0].plot(t, real_sig, 'g-', linewidth=0.8, label='Real')
                axes[row, 0].set_ylabel(f'Sig {sig_idx}', fontsize=10)
                if row == 0:
                    axes[row, 0].set_title('Real (Target)', fontsize=12, fontweight='bold')
                axes[row, 0].grid(True, alpha=0.3)
                axes[row, 0].set_xlim([t[0], t[-1]])

                # Column 2: Generated (raw + spectral shift)
                axes[row, 1].plot(t, gen_sig, 'r-', linewidth=0.8, label='Generated')
                if row == 0:
                    axes[row, 1].set_title('Generated (UNet+SpectralShift)', fontsize=12, fontweight='bold')
                axes[row, 1].set_ylabel(f'r={corr_gen:.3f}', fontsize=9, color='red')
                axes[row, 1].grid(True, alpha=0.3)
                axes[row, 1].set_xlim([t[0], t[-1]])

                # Column 3: Corrected (with envelope matching)
                axes[row, 2].plot(t, corr_sig, 'b-', linewidth=0.8, label='Corrected')
                if row == 0:
                    axes[row, 2].set_title('Corrected (+EnvelopeMatch)', fontsize=12, fontweight='bold')
                axes[row, 2].set_ylabel(f'r={corr_corr:.3f}', fontsize=9, color='blue')
                axes[row, 2].grid(True, alpha=0.3)
                axes[row, 2].set_xlim([t[0], t[-1]])

                # Column 4: Overlap of all three
                axes[row, 3].plot(t, real_sig, 'g-', linewidth=1.0, alpha=0.8, label='Real')
                axes[row, 3].plot(t, gen_sig, 'r-', linewidth=0.8, alpha=0.6, label='Generated')
                axes[row, 3].plot(t, corr_sig, 'b--', linewidth=0.8, alpha=0.6, label='Corrected')
                if row == 0:
                    axes[row, 3].set_title('Overlap Comparison', fontsize=12, fontweight='bold')
                    axes[row, 3].legend(loc='upper right', fontsize=8)
                axes[row, 3].grid(True, alpha=0.3)
                axes[row, 3].set_xlim([t[0], t[-1]])

                # Add x-label to bottom row
                if row == n_signals_to_plot - 1:
                    for col in range(4):
                        axes[row, col].set_xlabel('Time (ms)')

            plt.suptitle('Signal Comparison: Real vs Generated vs Corrected', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(debug_dir / "signal_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {debug_dir / 'signal_comparison.png'}")

            # Save stats to JSON
            with open(debug_dir / "debug_stats.json", "w") as f:
                json.dump(debug_stats, f, indent=2)
            print(f"  Saved: {debug_dir / 'debug_stats.json'}")

            print(f"\nAll debug plots saved to: {debug_dir.absolute()}")
            print(f"{'='*70}\n")

        # =====================================================================
        # STAGE 2: Optimal Bias Applied (No Training Needed!)
        # =====================================================================
        # With OptimalSpectralBias, we just compute the bias from data and apply it.
        # No gradient-based training is required - the bias is fixed per-odor.

        if is_primary():
            print(f"\n{'='*70}")
            print("STAGE 2: Optimal Spectral Bias Applied")
            print(f"{'='*70}")
            print("  Bias computed directly from UNet output vs target PSD difference")
            print("  No training required - bias is fixed per-odor per-band")
            print(f"{'='*70}")

        # Freeze all bias parameters (no training)
        for param in spectral_shift_fwd.parameters():
            param.requires_grad = False
        if spectral_shift_rev is not None:
            for param in spectral_shift_rev.parameters():
                param.requires_grad = False

        # Evaluate with optimal bias applied
        val_metrics = evaluate(
            model, loaders["val"], device, wavelet_loss,
            compute_phase=False, reverse_model=reverse_model, config=config,
            spectral_shift_fwd=spectral_shift_fwd, spectral_shift_rev=spectral_shift_rev,
            spectral_only=True, fast_mode=False,
            sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
            cond_encoder=cond_encoder,
        )

        psd_err_fwd = val_metrics.get("psd_err_db", float("inf"))
        psd_bias_fwd = val_metrics.get('psd_diff_db', 0)
        psd_err_rev = val_metrics.get("psd_err_db_rev", float("inf"))
        psd_bias_rev = val_metrics.get('psd_diff_db_rev', 0)

        if is_primary():
            print(f"\nValidation with Optimal Bias:")
            print(f"  Forward: PSD_err={psd_err_fwd:.2f}dB (bias={psd_bias_fwd:+.2f}dB)")
            if spectral_shift_rev is not None:
                print(f"  Reverse: PSD_err={psd_err_rev:.2f}dB (bias={psd_bias_rev:+.2f}dB)")

        # Save checkpoint with computed bias
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model, optimizer, num_epochs,  # Use Stage 1 optimizer/epoch since no Stage 2 training
            CHECKPOINT_DIR / "best_model_stage2.pt",
            is_fsdp=is_fsdp_wrapped,
            reverse_model=reverse_model,
            spectral_shift_fwd=spectral_shift_fwd,
            spectral_shift_rev=spectral_shift_rev,
            config=config,
        )

        if is_primary():
            print(f"\n{'='*70}")
            print("STAGE 2 COMPLETE: Post-hoc Calibration Applied")
            print(f"{'='*70}")
            print(f"  Optimal bias computed and saved (no training)")
            print(f"  FWD PSD_err = {psd_err_fwd:.2f} dB")
            if spectral_shift_rev is not None:
                print(f"  REV PSD_err = {psd_err_rev:.2f} dB")
            print(f"{'='*70}\n")

    # Final test evaluation (full metrics, fast_mode=False)
    test_metrics = evaluate(
        model, loaders["test"], device, wavelet_loss,
        compute_phase=True, reverse_model=reverse_model, config=config,
        spectral_shift_fwd=spectral_shift_fwd, spectral_shift_rev=spectral_shift_rev,
        fast_mode=False,  # Full metrics for final evaluation
        sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
        cond_encoder=cond_encoder,
        envelope_matcher_fwd=envelope_matcher_fwd,
        envelope_matcher_rev=envelope_matcher_rev,
    )

    if is_primary():
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

    if is_distributed:
        dist.barrier()

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

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "history": history,
        "model": model,
        "reverse_model": reverse_model,
    }


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CondUNet1D for neural signal translation")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="olfactory",
                        choices=["olfactory", "pfc"],
                        help="Dataset to train on: 'olfactory' (OB→PCx) or 'pfc' (PFC→CA1)")
    parser.add_argument("--resample-pfc", action="store_true",
                        help="Resample PFC dataset from 1250Hz to 1000Hz (for compatibility)")
    
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

    # Contrastive learning for cross-session generalization (CEBRA-style)
    parser.add_argument("--use-contrastive", action="store_true", default=None,
                        help="Enable CEBRA-style contrastive learning for session-invariant representations")
    parser.add_argument("--no-contrastive", action="store_true", default=None,
                        help="Disable contrastive learning (default)")
    parser.add_argument("--contrastive-weight", type=float, default=None,
                        help="Weight for contrastive loss (default: 0.1)")
    parser.add_argument("--contrastive-temperature", type=float, default=None,
                        help="Temperature for InfoNCE loss (default: 0.1)")

    # Conditioning mode override
    COND_MODES = ["none", "cross_attn_gated"]
    parser.add_argument("--cond-mode", type=str, default=None,
                        choices=COND_MODES,
                        help="Override conditioning mode")

    # Conditioning source: how conditioning embeddings are derived
    COND_SOURCES = ["odor_onehot", "spectro_temporal", "cpc", "vqvae", "freq_disentangled", "cycle_consistent"]
    parser.add_argument("--conditioning", type=str, default="spectro_temporal",
                        choices=COND_SOURCES,
                        help="Conditioning source: 'spectro_temporal' (default, auto-conditioning from signal dynamics), "
                             "'odor_onehot' (uses odor labels), "
                             "'cpc' (contrastive predictive coding), 'vqvae' (vector quantized), "
                             "'freq_disentangled' (per-band encoding), 'cycle_consistent' (cycle loss)")

    # Stage 1 evaluation
    parser.add_argument("--eval-stage1", action="store_true",
                        help="Evaluate and save metrics after stage 1 (before spectral fine-tuning)")

    # Spectral finetune epochs override
    parser.add_argument("--spectral-finetune-epochs", type=int, default=None,
                        help="Override number of spectral fine-tuning epochs")

    # Spectral shift block configuration
    parser.add_argument("--spectral-shift-mode", type=str, default=None,
                        choices=["flat", "frequency_band"],
                        help="Spectral shift mode: 'flat' (per-channel) or 'frequency_band' (per-band)")
    parser.add_argument("--spectral-shift-conditional", type=lambda x: x.lower() == 'true',
                        default=None, metavar="BOOL",
                        help="Learn odor-specific spectral shifts (True/False)")
    parser.add_argument("--spectral-shift-band-width", type=float, default=None,
                        help="Band width in Hz for frequency_band mode (e.g., 2.0, 4.0, 8.0)")
    parser.add_argument("--spectral-shift-per-channel", type=lambda x: x.lower() == 'true',
                        default=None, metavar="BOOL",
                        help="Learn per-channel spectral shifts (True/False)")
    parser.add_argument("--spectral-shift-init-db", type=float, default=None,
                        help="Initial dB shift value for spectral shift block")
    parser.add_argument("--spectral-shift-lr", type=float, default=None,
                        help="Learning rate for spectral shift parameters")
    # Note: OptimalSpectralBias always computes bias directly from data (no training needed)

    # Stage control
    parser.add_argument("--skip-spectral-finetune", action="store_true",
                        help="Skip Stage 2 (post-hoc calibration)")
    parser.add_argument("--stage2-only", action="store_true",
                        help="Skip Stage 1, load checkpoint and run Stage 2 (calibration)")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to checkpoint to load for stage2-only mode")

    # Training mode control
    parser.add_argument("--no-bidirectional", action="store_true",
                        help="Disable bidirectional training (only train OB→PCx, no cycle consistency)")

    # Per-channel normalization (applied during training)
    parser.add_argument("--per-channel-norm", action="store_true", default=True,
                        help="Apply per-channel z-score normalization to each batch (default: True)")
    parser.add_argument("--no-per-channel-norm", action="store_false", dest="per_channel_norm",
                        help="Disable per-channel normalization")

    # Data augmentation (default=None means use config value, not override)
    parser.add_argument("--aug-time-shift", action="store_true", default=None,
                        help="Enable random circular time shift augmentation")
    parser.add_argument("--no-aug-time-shift", action="store_false", dest="aug_time_shift",
                        help="Disable time shift augmentation")
    parser.add_argument("--aug-time-shift-max", type=float, default=None,
                        help="Max time shift as fraction of signal length (default: 0.1)")
    parser.add_argument("--aug-noise", action="store_true", default=None,
                        help="Enable Gaussian noise augmentation")
    parser.add_argument("--no-aug-noise", action="store_false", dest="aug_noise",
                        help="Disable noise augmentation")
    parser.add_argument("--aug-noise-std", type=float, default=None,
                        help="Noise std relative to signal std (default: 0.05)")
    parser.add_argument("--aug-channel-dropout", action="store_true", default=None,
                        help="Enable random channel dropout augmentation")
    parser.add_argument("--no-aug-channel-dropout", action="store_false", dest="aug_channel_dropout",
                        help="Disable channel dropout augmentation")
    parser.add_argument("--aug-channel-dropout-p", type=float, default=None,
                        help="Probability of dropping each channel (default: 0.1)")
    parser.add_argument("--aug-amplitude-scale", action="store_true", default=None,
                        help="Enable random amplitude scaling augmentation")
    parser.add_argument("--no-aug-amplitude-scale", action="store_false", dest="aug_amplitude_scale",
                        help="Disable amplitude scale augmentation")
    parser.add_argument("--aug-amplitude-scale-min", type=float, default=None,
                        help="Min amplitude scale factor (default: 0.8)")
    parser.add_argument("--aug-amplitude-scale-max", type=float, default=None,
                        help="Max amplitude scale factor (default: 1.2)")
    parser.add_argument("--aug-time-mask", action="store_true", default=None,
                        help="Enable random time masking augmentation")
    parser.add_argument("--no-aug-time-mask", action="store_false", dest="aug_time_mask",
                        help="Disable time mask augmentation")
    parser.add_argument("--aug-time-mask-ratio", type=float, default=None,
                        help="Fraction of time to mask (default: 0.1)")
    parser.add_argument("--aug-mixup", action="store_true", default=None,
                        help="Enable Mixup augmentation (blend random sample pairs)")
    parser.add_argument("--no-aug-mixup", action="store_false", dest="aug_mixup",
                        help="Disable Mixup augmentation")
    parser.add_argument("--aug-mixup-alpha", type=float, default=None,
                        help="Mixup Beta distribution alpha (default: 0.4, higher=more mixing)")
    parser.add_argument("--aug-freq-mask", action="store_true", default=None,
                        help="Enable frequency masking augmentation")
    parser.add_argument("--no-aug-freq-mask", action="store_false", dest="aug_freq_mask",
                        help="Disable frequency masking augmentation")
    parser.add_argument("--aug-freq-mask-max-bands", type=int, default=None,
                        help="Max number of frequency bands to mask (default: 2)")
    parser.add_argument("--aug-freq-mask-max-width", type=int, default=None,
                        help="Max width of each masked band in freq bins (default: 10)")

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

    # Contrastive learning config (CLI overrides config)
    if args.use_contrastive:
        config["use_contrastive"] = True
    elif args.no_contrastive:
        config["use_contrastive"] = False
    if args.contrastive_weight is not None:
        config["contrastive_weight"] = args.contrastive_weight
    if args.contrastive_temperature is not None:
        config["contrastive_temperature"] = args.contrastive_temperature

    # Print contrastive learning info
    if is_primary() and config.get("use_contrastive", False):
        print(f"CONTRASTIVE LEARNING ENABLED: weight={config['contrastive_weight']}, temp={config['contrastive_temperature']}")

    # Print session split info
    if is_primary() and config["split_by_session"]:
        print(f"Using SESSION-BASED SPLITS: {config['n_test_sessions']} test, {config['n_val_sessions']} val sessions")
        print(f"Session column: {config['session_column']}")

    # Load data based on dataset choice
    if args.dataset == "pfc":
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
        )
        config["dataset_type"] = "olfactory"
        config["in_channels"] = 32   # OB channels
        config["out_channels"] = 32  # PCx channels
        config["sampling_rate"] = SAMPLING_RATE_HZ

    # Conditioning from CLI
    if args.cond_mode is not None:
        config["cond_mode"] = args.cond_mode
    config["conditioning_source"] = args.conditioning  # odor_onehot, spectro_temporal, etc.
    if args.spectral_finetune_epochs is not None:
        config["spectral_finetune_epochs"] = args.spectral_finetune_epochs

    # Spectral shift configuration from CLI
    if args.spectral_shift_mode is not None:
        config["spectral_shift_mode"] = args.spectral_shift_mode
    if args.spectral_shift_conditional is not None:
        config["spectral_shift_conditional"] = args.spectral_shift_conditional
    if args.spectral_shift_band_width is not None:
        config["spectral_shift_band_width_hz"] = args.spectral_shift_band_width
    if args.spectral_shift_per_channel is not None:
        config["spectral_shift_per_channel"] = args.spectral_shift_per_channel
    if args.spectral_shift_init_db is not None:
        config["spectral_shift_init_fwd"] = args.spectral_shift_init_db
    if args.spectral_shift_lr is not None:
        config["spectral_shift_lr"] = args.spectral_shift_lr
    # Note: OptimalSpectralBias computes bias directly from data (no training config needed)

    # Stage control from CLI
    if args.skip_spectral_finetune:
        config["spectral_finetune_epochs"] = 0  # Disables Stage 2 (post-hoc calibration)
    if args.stage2_only:
        config["stage2_only"] = True
        if args.load_checkpoint:
            config["stage1_checkpoint"] = args.load_checkpoint
        else:
            # Default to best_model.pt in artifacts/checkpoints
            config["stage1_checkpoint"] = str(CHECKPOINT_DIR / "best_model.pt")

    # Stage 1 evaluation flag
    config["eval_stage1"] = args.eval_stage1

    # Disable bidirectional training if requested (for fair architecture comparison)
    if args.no_bidirectional:
        config["use_bidirectional"] = False
        if is_primary():
            print("Bidirectional training DISABLED (--no-bidirectional)")

    # Data augmentation config from CLI (only override if explicitly set, not None)
    if args.aug_time_shift is not None:
        config["aug_time_shift"] = args.aug_time_shift
    if args.aug_time_shift_max is not None:
        config["aug_time_shift_max"] = args.aug_time_shift_max
    if args.aug_noise is not None:
        config["aug_noise"] = args.aug_noise
    if args.aug_noise_std is not None:
        config["aug_noise_std"] = args.aug_noise_std
    if args.aug_channel_dropout is not None:
        config["aug_channel_dropout"] = args.aug_channel_dropout
    if args.aug_channel_dropout_p is not None:
        config["aug_channel_dropout_p"] = args.aug_channel_dropout_p
    if args.aug_amplitude_scale is not None:
        config["aug_amplitude_scale"] = args.aug_amplitude_scale
    if args.aug_amplitude_scale_min is not None or args.aug_amplitude_scale_max is not None:
        # Get current range or default
        current_range = config.get("aug_amplitude_scale_range", (0.8, 1.2))
        min_val = args.aug_amplitude_scale_min if args.aug_amplitude_scale_min is not None else current_range[0]
        max_val = args.aug_amplitude_scale_max if args.aug_amplitude_scale_max is not None else current_range[1]
        config["aug_amplitude_scale_range"] = (min_val, max_val)
    if args.aug_time_mask is not None:
        config["aug_time_mask"] = args.aug_time_mask
    if args.aug_time_mask_ratio is not None:
        config["aug_time_mask_ratio"] = args.aug_time_mask_ratio
    if args.aug_mixup is not None:
        config["aug_mixup"] = args.aug_mixup
    if args.aug_mixup_alpha is not None:
        config["aug_mixup_alpha"] = args.aug_mixup_alpha
    if args.aug_freq_mask is not None:
        config["aug_freq_mask"] = args.aug_freq_mask
    if args.aug_freq_mask_max_bands is not None:
        config["aug_freq_mask_max_bands"] = args.aug_freq_mask_max_bands
    if args.aug_freq_mask_max_width is not None:
        config["aug_freq_mask_max_width"] = args.aug_freq_mask_max_width

    # Print augmentation config if any are enabled
    aug_enabled = [
        k for k in [
            "aug_time_shift", "aug_noise", "aug_channel_dropout", "aug_amplitude_scale",
            "aug_time_mask", "aug_mixup", "aug_freq_mask",
            "aug_channel_scale", "aug_dc_offset"  # Session-specific augmentations
        ]
        if config.get(k, False)
    ]
    if aug_enabled and is_primary():
        print(f"Data augmentation ENABLED: {', '.join(aug_enabled)}")
        if config.get("aug_channel_scale", False) or config.get("aug_dc_offset", False):
            print("  [Session-invariance augmentations active: channel_scale, dc_offset]")

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

    # Per-channel normalization (default: enabled)
    config["per_channel_norm"] = args.per_channel_norm if hasattr(args, 'per_channel_norm') else True
    if is_primary():
        print(f"Per-channel normalization: {'ENABLED' if config['per_channel_norm'] else 'DISABLED'}")

    # Output scaling correction in model (default: enabled)
    config["use_output_scaling"] = args.output_scaling if hasattr(args, 'output_scaling') else True
    if is_primary():
        print(f"Output scaling correction: {'ENABLED' if config['use_output_scaling'] else 'DISABLED'}")

    if is_primary():
        print(f"\nTraining CondUNet1D for {config['num_epochs']} epochs...")
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
        print(f"Test loss: {results['test_metrics'].get('loss', 'N/A')}")
        print(f"Test correlation: {results['test_metrics']['corr']:.4f}")
        print(f"Test R2: {results['test_metrics']['r2']:.4f}")
        print(f"Model saved to: {CHECKPOINT_DIR / 'best_model.pt'}")
        # Machine-parseable results for experiment scripts
        print(f"RESULT_CORR={results['test_metrics']['corr']:.4f}")
        print(f"RESULT_R2={results['test_metrics']['r2']:.4f}")
        print(f"RESULT_LOSS={results['best_val_loss']:.4f}")
        if "split_info" in data:
            print(f"RESULT_SPLIT_TYPE=session_holdout")
            print(f"RESULT_TEST_SESSIONS={data['split_info']['test_sessions']}")


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
