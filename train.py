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
    HighFrequencySpectralLoss,
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
    # Residual distribution correction
    ResidualDistributionCorrector,
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

    # Two-stage training: UNet convergence → SpectralShift fine-tuning
    "use_two_stage": True,        # Enable two-stage training (UNet converge → freeze → SpectralShift fine-tune)
    "spectral_finetune_epochs": 20,  # Extra epochs to fine-tune ONLY SpectralShift after UNet converges
    "stage2_only": False,         # Skip Stage 1, load checkpoint and run ONLY Stage 2 (SpectralShift fine-tuning)
    "stage1_checkpoint": "best_model.pt",    # Path to Stage 1 checkpoint (required if stage2_only=True)

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
    "use_spectral_loss": True,  # PSD matching (log-domain spectral loss)

    # Bidirectional training
    "use_bidirectional": True,  # Train both OB→PCx and PCx→OB

    # Spectral shift block
    "use_spectral_shift": True,  # Per-channel amplitude scaling for PSD correction
    "spectral_shift_mode": "adaptive",  # "flat", "frequency_band", or "adaptive" (signal-adaptive)
    "spectral_shift_conditional": True,  # If True, learn odor-specific spectral shifts
    "spectral_shift_per_channel": True,  # If True, learn separate shifts per channel
    "spectral_shift_init_fwd": 0.0,  # Initial dB shift for OB→PCx (rev uses inverse automatically)
    "spectral_shift_lr": 0.001,  # Learning rate for spectral shift (adaptive mode uses lower lr)
    "spectral_shift_lr_decay": 0.95,  # Decay spectral shift lr each epoch (0.95^30 ≈ 0.21)
    "spectral_shift_band_width_hz": 2,  # None=use predefined neuro bands, or float (e.g., 2.0 for 2Hz uniform bands)
    "spectral_shift_max_db": 12.0,  # Maximum allowed shift in dB (for adaptive mode)
    "spectral_shift_hidden_dim": 128,  # Hidden dimension for adaptive mode network

    # Enhanced odor conditioning options for AdaptiveSpectralShift
    "spectral_shift_use_odor_base": True,   # Per-odor learnable base shifts per frequency band
    "spectral_shift_use_odor_scale": True,  # Per-odor learnable scale factors
    "spectral_shift_use_film": True,        # FiLM-style odor modulation of band features
    "spectral_shift_film_hidden_mult": 2,   # Hidden dim multiplier for FiLM network (hidden_dim * mult)

    # Envelope loss (for spectral shift training, helps match signal envelope distribution)
    # IMPORTANT: Applied with detach() so only SpectralShift gets gradients, not UNet
    "use_envelope_loss": True,              # Enable envelope distribution matching loss
    "envelope_loss_weight": 1.0,            # Weight for envelope loss
    "envelope_loss_n_bins": 64,             # Number of histogram bins for distribution matching
    "envelope_loss_type": "kl",             # Loss type: "kl" (KL divergence), "wasserstein", "mse"

    # Output scaling correction (learnable per-channel scale and bias)
    # Helps match target distribution, especially important for probabilistic losses
    "use_output_scaling": True,

    # Residual distribution correction (replaces old Rayleigh-based approach)
    # Learns to correct systematic distributional errors in UNet envelope output
    # Key insight: Learn from actual target distribution, not a theoretical prior
    # IMPORTANT: Gradients are isolated from UNet via .detach()
    "use_residual_correction": False,  # Enable residual distribution correction
    "residual_correction_n_bins": 32,  # Number of quantile bins for correction curve
    "residual_correction_hidden_dim": 64,  # Hidden dim for correction network
    "residual_correction_max": 0.5,  # Maximum correction (±50%)
    "residual_correction_lambda": 0.1,  # Weight for distribution loss
    "residual_correction_lr": 1e-3,  # Learning rate (separate optimizer)

    # Recording system (for Nature Methods publication)
    # WARNING: Recording is VERY slow - only enable for final runs!
    "enable_recording": False,  # Enable comprehensive recording
    "record_saliency": False,   # Compute saliency maps and Grad-CAM
    "record_neuroscience": False,  # Compute PAC, coherence, ERP, burst analysis
    "saliency_epoch_interval": 5,  # Compute saliency every N epochs
    "neuroscience_epoch_interval": 10,  # Compute neuroscience metrics every N epochs
    "recording_output_dir": "artifacts/recordings",  # Output directory for recordings
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
    """Get dB shift from SpectralShiftBlock or FrequencyBandSpectralShift (handles DDP wrapping).

    Args:
        spectral_shift_module: SpectralShiftBlock or FrequencyBandSpectralShift (may be DDP-wrapped)
        detailed: If True, return dict with detailed info

    Returns:
        Mean dB shift (or detailed dict), or None if module is None
    """
    if spectral_shift_module is None:
        return None

    import math
    # Handle DDP wrapping - access the underlying module
    module = spectral_shift_module.module if hasattr(spectral_shift_module, 'module') else spectral_shift_module

    # Check if it's FrequencyBandSpectralShift (has get_shift_db_dict method)
    if hasattr(module, 'get_shift_db_dict'):
        band_shifts = module.get_shift_db_dict()  # Total shifts (includes global gain)
        mean_shift = sum(band_shifts.values()) / len(band_shifts)
        # Get global gain separately if available
        global_gain_db = module.get_global_gain_db() if hasattr(module, 'get_global_gain_db') else 0.0
        if detailed:
            result = {
                "mean": mean_shift,
                "global_gain": global_gain_db,
                "bands": band_shifts,
                "mode": "frequency_band",
            }
            # Add per-odor shifts if conditional
            if getattr(module, 'conditional', False) and hasattr(module, 'get_odor_shift_db_dict'):
                result["conditional"] = True
                result["odor_shifts"] = module.get_odor_shift_db_dict()
            return result
        return mean_shift

    # Original SpectralShiftBlock (per-channel)
    if hasattr(module, 'log_scale'):
        log_scale = module.log_scale
        if log_scale.numel() > 0:
            shift_db = 20.0 * log_scale / math.log(10)
            if detailed:
                return {
                    "mean": shift_db.mean().item(),
                    "std": shift_db.std().item(),
                    "min": shift_db.min().item(),
                    "max": shift_db.max().item(),
                    "per_channel": shift_db.detach().cpu().tolist(),
                    "mode": "flat",
                }
            return shift_db.mean().item()
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
    spectral_loss: Optional[nn.Module] = None,
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
    residual_corrector: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate model on a dataloader (supports bidirectional).

    Returns composite validation loss that mirrors training loss.

    Args:
        spectral_only: If True, composite loss uses ONLY spectral loss (Stage 2 mode)
        disable_spectral: If True, disable SpectralShift and exclude spectral from composite loss (Stage 1 mode)
        fast_mode: If True, skip expensive metrics (PSD, phase, baseline) for faster validation.
                   Use fast_mode=False only for final evaluation.
        cond_encoder: Optional conditioning encoder for auto-conditioning modes
        residual_corrector: Optional ResidualDistributionCorrector for envelope correction
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
    if residual_corrector is not None:
        residual_corrector.eval()

    # Forward direction (OB→PCx)
    mse_list, mae_list, corr_list = [], [], []
    wavelet_list, plv_list, pli_list = [], [], []
    r2_list, nrmse_list = [], []
    psd_err_list, psd_diff_list = [], []

    # Reverse direction (PCx→OB)
    mse_list_rev, mae_list_rev, corr_list_rev = [], [], []
    wavelet_list_rev, spectral_list_rev = [], []
    plv_list_rev, pli_list_rev = [], []
    r2_list_rev, nrmse_list_rev = [], []
    psd_err_list_rev, psd_diff_list_rev = [], []

    # Baseline: Raw OB vs PCx (natural difference between brain regions)
    # This provides context - how similar are the regions naturally?
    baseline_corr_list, baseline_r2_list, baseline_nrmse_list = [], [], []
    baseline_psd_err_list, baseline_psd_diff_list = [], []
    baseline_plv_list, baseline_pli_list = [], []

    # For composite loss (forward)
    spectral_list = []

    # Envelope metrics (residual correction)
    env_cv2_list = []  # Envelope CV² of prediction
    env_cv2_target_list = []  # Target envelope CV² (ground truth)
    mean_correction_list = []  # Mean correction magnitude

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

            # Apply residual distribution correction (learns to match target envelope)
            # This is the FINAL output that should be used for metrics
            if residual_corrector is not None and cond_emb is not None:
                pred = residual_corrector(pred, cond_emb)

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

            if spectral_loss is not None:
                spectral_list.append(spectral_loss(pred_f32, pcx_f32).item())

            # Skip expensive phase metrics in fast_mode
            if compute_phase and not fast_mode:
                plv_list.append(plv_torch(pred_f32, pcx_f32).item())
                pli_list.append(pli_torch(pred_f32, pcx_f32).item())

            # Skip expensive PSD metrics in fast_mode (spectral_loss already covers frequency info)
            if not fast_mode:
                psd_err_list.append(psd_error_db_torch(pred_f32, pcx_f32, fs=sampling_rate).item())
                psd_diff_list.append(psd_diff_db_torch(pred_f32, pcx_f32, fs=sampling_rate).item())

            # Envelope metrics for residual correction monitoring (on CROPPED signals)
            if residual_corrector is not None and cond_emb is not None:
                # Compute envelope using Hilbert transform
                analytic_pred = hilbert_torch(pred_f32)
                envelope_pred = torch.abs(analytic_pred)

                analytic_target = hilbert_torch(pcx_f32)
                envelope_target = torch.abs(analytic_target)

                # Envelope CV² = Var(A) / Mean(A)²
                env_mean = envelope_pred.mean(dim=-1, keepdim=True).clamp(min=1e-8)
                env_var = ((envelope_pred - env_mean) ** 2).mean(dim=-1)
                env_cv2 = (env_var / (env_mean.squeeze(-1) ** 2)).mean()
                env_cv2_list.append(env_cv2.item())

                # Target envelope CV² for comparison
                tgt_mean = envelope_target.mean(dim=-1, keepdim=True).clamp(min=1e-8)
                tgt_var = ((envelope_target - tgt_mean) ** 2).mean(dim=-1)
                tgt_cv2 = (tgt_var / (tgt_mean.squeeze(-1) ** 2)).mean()
                env_cv2_target_list.append(tgt_cv2.item())

                # Get mean correction magnitude
                corr_module = residual_corrector.module if hasattr(residual_corrector, 'module') else residual_corrector
                mean_corr = corr_module.get_mean_correction(cond_emb.detach()).mean()
                mean_correction_list.append(mean_corr.item())

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

                if spectral_loss is not None:
                    spectral_list_rev.append(spectral_loss(pred_rev_f32, ob_f32).item())

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
    if spectral_list:
        results["spectral"] = float(np.mean(spectral_list))
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
    if spectral_list_rev:
        results["spectral_rev"] = float(np.mean(spectral_list_rev))
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

    # Envelope metrics (residual correction)
    if env_cv2_list:
        results["env_cv2"] = float(np.mean(env_cv2_list))  # Pred envelope CV²
    if env_cv2_target_list:
        results["env_cv2_target"] = float(np.mean(env_cv2_target_list))  # Target envelope CV²
    if mean_correction_list:
        results["mean_correction"] = float(np.mean(mean_correction_list))  # Mean correction magnitude

    # Compute composite validation loss (mirrors training loss)
    # This allows early stopping based on overall objective, not just correlation
    if config is not None:
        if spectral_only:
            # Stage 2: ONLY spectral loss (UNet frozen, pure PSD optimization)
            w_spec = config.get("weight_spectral", 1.0) if config.get("use_spectral_loss", True) else 0.0
            val_loss = 0.0
            if "spectral" in results:
                val_loss += w_spec * results["spectral"]
            if "spectral_rev" in results:
                val_loss += w_spec * results["spectral_rev"]
        else:
            # Stage 1: L1 + wavelet (spectral excluded if disable_spectral=True)
            w_l1 = config.get("weight_l1", 1.0)
            w_wav = config.get("weight_wavelet", 1.0) if config.get("use_wavelet_loss", True) else 0.0
            # Spectral weight is 0 if disabled (Stage 1 mode)
            w_spec = 0.0 if disable_spectral else (config.get("weight_spectral", 1.0) if config.get("use_spectral_loss", True) else 0.0)

            # Forward loss
            val_loss = w_l1 * results["mae"]
            if "wavelet" in results:
                val_loss += w_wav * results["wavelet"]
            if "spectral" in results and not disable_spectral:
                val_loss += w_spec * results["spectral"]

            # Reverse loss (if bidirectional)
            if "mae_rev" in results:
                val_loss += w_l1 * results["mae_rev"]
                if "wavelet_rev" in results:
                    val_loss += w_wav * results["wavelet_rev"]
                if "spectral_rev" in results and not disable_spectral:
                    val_loss += w_spec * results["spectral_rev"]

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


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict[str, Any],
    wavelet_loss: Optional[nn.Module] = None,
    spectral_loss: Optional[nn.Module] = None,
    reverse_model: Optional[nn.Module] = None,
    epoch: int = 0,
    num_epochs: int = 0,
    spectral_shift_fwd: Optional[nn.Module] = None,
    spectral_shift_rev: Optional[nn.Module] = None,
    stage2_spectral_only: bool = False,
    disable_spectral: bool = False,
    cond_encoder: Optional[nn.Module] = None,
    residual_corrector: Optional[nn.Module] = None,
    residual_optimizer: Optional[torch.optim.Optimizer] = None,
    prob_loss: Optional[nn.Module] = None,
    envelope_loss: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Train one epoch (supports bidirectional with cycle consistency).

    Args:
        stage2_spectral_only: If True, ONLY use spectral loss (Stage 2 mode where UNet is frozen)
        disable_spectral: If True, disable SpectralShift application and spectral loss (Stage 1 mode)
        cond_encoder: Optional conditioning encoder for auto-conditioning modes
        residual_corrector: Optional ResidualDistributionCorrector for envelope correction
        residual_optimizer: Optional separate optimizer for residual corrector
        envelope_loss: Optional envelope distribution matching loss (for SpectralShift)
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
    if residual_corrector is not None:
        residual_corrector.train()

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
        if cond_emb is not None:
            pred_raw = model(ob, cond_emb=cond_emb)
        else:
            pred_raw = model(ob, odor)

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

        # Stage 2 (UNet frozen): ONLY spectral loss for PSD correction
        # Stage 1 (UNet trainable): L1 + wavelet (NO spectral)
        if stage2_spectral_only:
            # ONLY spectral loss - UNet is frozen, we're fine-tuning SpectralShift for PSD
            if config.get("use_spectral_loss", True) and spectral_loss is not None:
                loss = config.get("weight_spectral", 1.0) * spectral_loss(pred_shifted_c, pcx_c)
                loss_components["spectral_fwd"] = loss_components["spectral_fwd"] + loss.detach()
            else:
                loss = torch.tensor(0.0, device=device)

            # Envelope loss (forward) - helps SpectralShift match envelope distribution
            # Uses pred_shifted (SpectralShift gets gradients)
            if config.get("use_envelope_loss", True) and envelope_loss is not None:
                env_loss = config.get("envelope_loss_weight", 1.0) * envelope_loss(pred_shifted_c, pcx_c)
                loss = loss + env_loss
                loss_components["envelope_fwd"] = loss_components["envelope_fwd"] + env_loss.detach()
        else:
            # Stage 1: L1/Huber + wavelet (spectral disabled if disable_spectral=True)
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

            # Spectral loss (forward) - DISABLED in Stage 1, enabled in joint training
            # uses pred_shifted (SpectralShift DOES get gradients here)
            if not disable_spectral and config.get("use_spectral_loss", True) and spectral_loss is not None:
                spec_loss = config.get("weight_spectral", 1.0) * spectral_loss(pred_shifted_c, pcx_c)
                loss = loss + spec_loss
                loss_components["spectral_fwd"] = loss_components["spectral_fwd"] + spec_loss.detach()

            # Envelope loss (forward) - DISABLED in Stage 1, enabled in joint training
            # Uses pred_shifted (SpectralShift gets gradients)
            if not disable_spectral and config.get("use_envelope_loss", True) and envelope_loss is not None:
                env_loss = config.get("envelope_loss_weight", 1.0) * envelope_loss(pred_shifted_c, pcx_c)
                loss = loss + env_loss
                loss_components["envelope_fwd"] = loss_components["envelope_fwd"] + env_loss.detach()

            # Probabilistic loss (forward) - for tier 2.5, added ON TOP of base loss
            if prob_loss is not None:
                p_loss = prob_loss(pred_raw_c, pcx_c)
                loss = loss + p_loss
                loss_components["prob_fwd"] = loss_components["prob_fwd"] + p_loss.detach()

        # Add conditioning encoder auxiliary loss if present
        if cond_loss != 0.0:
            loss = loss + cond_loss
            loss_components["cond_loss"] = loss_components["cond_loss"] + cond_loss.detach() if isinstance(cond_loss, torch.Tensor) else loss_components["cond_loss"] + cond_loss

        # Bidirectional training with cycle consistency
        if reverse_model is not None:
            # Reverse: PCx → OB (use same cond_emb from forward - conditioning is symmetric)
            if cond_emb is not None:
                pred_rev_raw = reverse_model(pcx, cond_emb=cond_emb)
            else:
                pred_rev_raw = reverse_model(pcx, odor)

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

            # Stage 2: ONLY spectral loss for reverse direction too
            if stage2_spectral_only:
                # ONLY spectral loss - UNet frozen, fine-tuning SpectralShift
                if config.get("use_spectral_loss", True) and spectral_loss is not None:
                    spec_loss_rev = config.get("weight_spectral", 1.0) * spectral_loss(pred_rev_shifted_c, ob_c)
                    loss = loss + spec_loss_rev
                    loss_components["spectral_rev"] = loss_components["spectral_rev"] + spec_loss_rev.detach()

                # Envelope loss (reverse) - helps SpectralShift match envelope distribution
                if config.get("use_envelope_loss", True) and envelope_loss is not None:
                    env_loss_rev = config.get("envelope_loss_weight", 1.0) * envelope_loss(pred_rev_shifted_c, ob_c)
                    loss = loss + env_loss_rev
                    loss_components["envelope_rev"] = loss_components["envelope_rev"] + env_loss_rev.detach()
            else:
                # Stage 1: L1/Huber + wavelet (spectral disabled if disable_spectral=True)
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

                # Spectral loss (reverse) - DISABLED in Stage 1, enabled in joint training
                # uses pred_rev_shifted (SpectralShift DOES get gradients)
                if not disable_spectral and config.get("use_spectral_loss", True) and spectral_loss is not None:
                    spec_loss_rev = config.get("weight_spectral", 1.0) * spectral_loss(pred_rev_shifted_c, ob_c)
                    loss = loss + spec_loss_rev
                    loss_components["spectral_rev"] = loss_components["spectral_rev"] + spec_loss_rev.detach()

                # Envelope loss (reverse) - DISABLED in Stage 1, enabled in joint training
                if not disable_spectral and config.get("use_envelope_loss", True) and envelope_loss is not None:
                    env_loss_rev = config.get("envelope_loss_weight", 1.0) * envelope_loss(pred_rev_shifted_c, ob_c)
                    loss = loss + env_loss_rev
                    loss_components["envelope_rev"] = loss_components["envelope_rev"] + env_loss_rev.detach()

                # Probabilistic loss (reverse) - for tier 2.5
                if prob_loss is not None:
                    p_loss_rev = prob_loss(pred_rev_raw_c, ob_c)
                    loss = loss + p_loss_rev
                    loss_components["prob_rev"] = loss_components["prob_rev"] + p_loss_rev.detach()

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

        # =====================================================================
        # Residual Distribution Correction (experimental)
        # CRITICAL: Input is DETACHED to ensure gradient isolation
        # Distribution losses train ONLY the corrector, not UNet
        # Key: Learns to match actual target distribution, not a theoretical prior
        #
        # IMPORTANT: Apply corrector AFTER spectral shift to match evaluation flow!
        # Training:   UNet → SpectralShift → ResidualCorrector → Loss
        # Evaluation: UNet → SpectralShift → ResidualCorrector → Metrics
        # =====================================================================
        residual_loss = torch.tensor(0.0, device=device)
        if residual_corrector is not None and cond_emb is not None:
            # Use spectral-shifted output if available, otherwise raw output
            # This matches the evaluation pipeline exactly
            if spectral_shift_fwd is not None and not disable_spectral:
                corrector_input = pred_shifted.detach()
            else:
                corrector_input = pred_raw.detach()

            # Apply residual correction with DETACHED input
            # This prevents correction gradients from flowing back to UNet/SpectralShift
            y_corrected = residual_corrector(corrector_input, cond_emb.detach())

            # IMPORTANT: Crop BEFORE computing loss to match target dimensions!
            y_corrected_c = crop_to_target_torch(y_corrected)

            # Compute distribution matching loss against TARGET
            # Uses 1D optimal transport (sorted envelope comparison)
            corr_module = residual_corrector.module if hasattr(residual_corrector, 'module') else residual_corrector
            corr_losses = corr_module.compute_loss(y_corrected_c, pcx_c)

            # Weighted loss
            lambda_dist = config.get("residual_correction_lambda", 0.1)
            residual_loss = lambda_dist * corr_losses["distribution_loss"]

            loss_components["dist_loss"] = loss_components["dist_loss"] + corr_losses["distribution_loss"].detach()

            # Separate backward/step for residual corrector (always use separate optimizer)
            if residual_optimizer is not None:
                residual_loss.backward()
                torch.nn.utils.clip_grad_norm_(residual_corrector.parameters(), GRAD_CLIP)
                residual_optimizer.step()
                residual_optimizer.zero_grad(set_to_none=True)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if reverse_model is not None:
            torch.nn.utils.clip_grad_norm_(reverse_model.parameters(), GRAD_CLIP)
        if cond_encoder is not None:
            torch.nn.utils.clip_grad_norm_(cond_encoder.parameters(), GRAD_CLIP)
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
    from models import SpectralShiftBlock, FrequencyBandSpectralShift, AdaptiveSpectralShift

    spectral_shift_fwd = None
    spectral_shift_rev = None
    if config.get("use_spectral_shift", True):
        # Initialize with predefined shift (from baseline PSD difference)
        # SEPARATE modules for forward and reverse - no inverse constraint!
        # This fixes the gradient conflict where both directions need same-sign correction.
        init_shift_fwd = config.get("spectral_shift_init_fwd", 0.0)  # OB→PCx
        init_shift_rev = config.get("spectral_shift_init_rev", 0.0)  # PCx→OB (independent)
        shift_mode = config.get("spectral_shift_mode", "adaptive")
        shift_conditional = config.get("spectral_shift_conditional", True)
        shift_per_channel = config.get("spectral_shift_per_channel", False)
        
        # Get correct channel counts for each direction
        # Forward: source → target (e.g., PFC 64ch → CA1 32ch), SpectralShift operates on OUTPUT
        # Reverse: target → source (e.g., CA1 32ch → PFC 64ch), SpectralShift operates on OUTPUT
        fwd_out_channels = config.get("out_channels", 32)  # Forward output channels
        rev_out_channels = config.get("in_channels", 32)   # Reverse output = forward input
        sampling_rate = config.get("sampling_rate", SAMPLING_RATE_HZ)

        if shift_mode == "adaptive":
            # Dynamic signal-adaptive spectral shift with odor conditioning
            # Analyzes input signal's spectral content and predicts per-band shifts
            n_odors = config.get("n_odors", 7)
            emb_dim = config.get("emb_dim", 128)
            band_width_hz = config.get("spectral_shift_band_width_hz", None)
            max_shift_db = config.get("spectral_shift_max_db", 12.0)
            hidden_dim = config.get("spectral_shift_hidden_dim", 64)
            # Enhanced odor conditioning options
            use_odor_base = config.get("spectral_shift_use_odor_base", True)
            use_odor_scale = config.get("spectral_shift_use_odor_scale", True)
            use_film = config.get("spectral_shift_use_film", True)
            film_hidden_mult = config.get("spectral_shift_film_hidden_mult", 2)

            spectral_shift_fwd = AdaptiveSpectralShift(
                n_channels=fwd_out_channels,
                n_odors=n_odors,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                sample_rate=sampling_rate,
                band_width_hz=band_width_hz,
                max_shift_db=max_shift_db,
                per_channel=shift_per_channel,
                use_odor_base=use_odor_base,
                use_odor_scale=use_odor_scale,
                use_film=use_film,
                film_hidden_mult=film_hidden_mult,
            ).to(device)
            spectral_shift_rev = AdaptiveSpectralShift(
                n_channels=rev_out_channels,
                n_odors=n_odors,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                sample_rate=sampling_rate,
                band_width_hz=band_width_hz,
                max_shift_db=max_shift_db,
                per_channel=shift_per_channel,
                use_odor_base=use_odor_base,
                use_odor_scale=use_odor_scale,
                use_film=use_film,
                film_hidden_mult=film_hidden_mult,
            ).to(device)
            if band_width_hz is not None:
                band_str = f", {spectral_shift_fwd.n_bands} bands @ {band_width_hz}Hz"
            else:
                band_str = ", 10 neuro bands"
            per_ch_str = ", per_channel" if shift_per_channel else ""
            mode_str = f"adaptive (signal-aware, n_odors={n_odors}, max={max_shift_db}dB{band_str}{per_ch_str})"
        elif shift_mode == "frequency_band":
            # Per-frequency-band scaling (delta/theta/alpha/beta/gamma or uniform)
            # With optional odor conditioning (different shifts per odor)
            # SEPARATE modules for forward and reverse - each learns independently
            n_odors = config.get("n_odors", 7)
            band_width_hz = config.get("spectral_shift_band_width_hz", None)
            spectral_shift_fwd = FrequencyBandSpectralShift(
                sample_rate=sampling_rate,
                init_shift_db=init_shift_fwd,
                per_channel=shift_per_channel,
                n_odors=n_odors,
                conditional=shift_conditional,
                band_width_hz=band_width_hz,
                n_channels=fwd_out_channels,
            ).to(device)
            spectral_shift_rev = FrequencyBandSpectralShift(
                sample_rate=sampling_rate,
                init_shift_db=init_shift_rev,
                per_channel=shift_per_channel,
                n_odors=n_odors,
                conditional=shift_conditional,
                band_width_hz=band_width_hz,
                n_channels=rev_out_channels,
            ).to(device)
            cond_str = f", conditional (n_odors={n_odors})" if shift_conditional else ""
            if band_width_hz is not None:
                band_str = f", {spectral_shift_fwd.n_bands} bands @ {band_width_hz}Hz"
            else:
                band_str = ", 10 neuro bands"
            per_ch_str = ", per_channel" if shift_per_channel else ""
            mode_str = f"frequency_band (separate fwd/rev{cond_str}{band_str}{per_ch_str})"
        else:
            # Flat per-channel scaling (original) - no conditioning support
            # SEPARATE modules for forward and reverse
            spectral_shift_fwd = SpectralShiftBlock(n_channels=fwd_out_channels, init_shift_db=init_shift_fwd).to(device)
            spectral_shift_rev = SpectralShiftBlock(n_channels=rev_out_channels, init_shift_db=init_shift_rev).to(device)
            mode_str = f"flat (fwd={fwd_out_channels}ch, rev={rev_out_channels}ch, separate)"

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
            print(f"SpectralShift created{ddp_str} mode={mode_str} (fwd init: {init_shift_fwd:+.1f}dB, rev init: {init_shift_rev:+.1f}dB)")

    # Define betas early since it's used by multiple optimizers
    betas = (config.get("beta1", 0.9), config.get("beta2", 0.999))

    # Create residual distribution corrector (experimental)
    # IMPORTANT: Gradient isolation - corrector input is DETACHED from UNet
    # This ensures UNet training is not affected by distribution losses
    # Key improvement: Learns to match actual target distribution, not a theoretical prior
    residual_corrector = None
    residual_optimizer = None
    if config.get("use_residual_correction", False):
        # Requires spectro_temporal conditioning to work
        if cond_source != "spectro_temporal":
            if is_primary():
                print("WARNING: Residual correction requires spectro_temporal conditioning. Disabling.")
        else:
            residual_corrector = ResidualDistributionCorrector(
                condition_dim=emb_dim,
                n_bins=config.get("residual_correction_n_bins", 32),
                hidden_dim=config.get("residual_correction_hidden_dim", 64),
                max_correction=config.get("residual_correction_max", 0.5),
            )

            # Convert to bf16 if FSDP uses mixed precision (match cond_encoder dtype)
            # NOTE: Correction uses FFT internally which needs float32,
            # but the correction_net (Linear layers) should match input dtype
            if config.get("fsdp_bf16", False):
                residual_corrector = residual_corrector.to(device, dtype=torch.bfloat16)
            else:
                residual_corrector = residual_corrector.to(device)

            # Wrap with DDP for distributed training
            if is_distributed:
                residual_corrector = DDP(residual_corrector, device_ids=[local_rank])

            if is_primary():
                corr_params = sum(p.numel() for p in residual_corrector.parameters())
                print(f"Residual corrector created: {corr_params:,} params, n_bins={config.get('residual_correction_n_bins', 32)}, max_correction={config.get('residual_correction_max', 0.5)}")

            # Always use separate optimizer for gradient isolation
            residual_optimizer = AdamW(
                residual_corrector.parameters(),
                lr=config.get("residual_correction_lr", 1e-3),
                betas=betas,
            )
            if is_primary():
                print(f"Residual corrector using separate optimizer, lr={config.get('residual_correction_lr', 1e-3)}")

    # Create loss functions
    wavelet_loss = None
    if config.get("use_wavelet_loss", True):
        wavelet_loss = build_wavelet_loss(
            wavelet=config.get("wavelet_family", "morlet"),
            use_complex_morlet=config.get("use_complex_morlet", True),
            omega0=config.get("wavelet_omega0", 5.0),
        ).to(device)

    spectral_loss = None
    if config.get("use_spectral_loss", True):
        spectral_loss = HighFrequencySpectralLoss(
            sample_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
            low_freq_cutoff=0.0,
            high_freq_boost=1.0,
            max_freq=MAX_FREQ_HZ,
            use_log_psd=True,
        ).to(device)

    # Envelope loss (for SpectralShift training - matches envelope distribution)
    # IMPORTANT: Applied with detach() so only SpectralShift gets gradients, not UNet
    envelope_loss = None
    if config.get("use_envelope_loss", True):
        from models import EnvelopeLoss
        envelope_loss = EnvelopeLoss(
            n_bins=config.get("envelope_loss_n_bins", 64),
            loss_type=config.get("envelope_loss_type", "kl"),
        ).to(device)
        if is_primary():
            print(f"Envelope loss: {config.get('envelope_loss_type', 'kl')} (weight={config.get('envelope_loss_weight', 1.0)}, bins={config.get('envelope_loss_n_bins', 64)})")

    # Probabilistic loss (tier 2.5 - added ON TOP of base loss)
    prob_loss = None
    prob_loss_type = config.get("prob_loss_type", "none")
    prob_loss_weight = config.get("prob_loss_weight", 1.0)
    if prob_loss_type != "none":
        try:
            from experiments.study3_loss.losses.neural_probabilistic_losses import (
                create_neural_prob_loss
            )
            prob_loss = create_neural_prob_loss(prob_loss_type, weight=prob_loss_weight).to(device)
            if is_primary():
                print(f"Probabilistic loss: {prob_loss_type} (weight={prob_loss_weight})")
        except Exception as e:
            if is_primary():
                print(f"Warning: Could not create probabilistic loss '{prob_loss_type}': {e}")
            prob_loss = None

    # Create optimizer with parameter groups
    # SpectralShift needs MUCH higher lr because it's just 32 scalars trying to make dB-scale changes
    lr = config.get("learning_rate", 1e-4)
    spectral_shift_lr = config.get("spectral_shift_lr", 0.1)  # 500x higher for fast amplitude adaptation
    # Note: betas is defined earlier (before distribution block) since it's used by multiple optimizers

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
    # Note: Residual corrector always uses separate optimizer for gradient isolation
    # Add probabilistic loss parameters (if any) with model learning rate
    if prob_loss is not None:
        prob_params = list(prob_loss.parameters())
        if prob_params:
            param_groups.append({"params": prob_params, "lr": lr, "name": "prob_loss"})

    total_params = sum(len(list(pg["params"])) if not isinstance(pg["params"], list) else len(pg["params"]) for pg in param_groups)
    if is_primary():
        print(f"Optimizer: {total_params} total params | model lr={lr}, spectral_shift lr={spectral_shift_lr}")

    optimizer = AdamW(param_groups, lr=lr, betas=betas)

    # Learning rate scheduler for SpectralShift
    # Decay SpectralShift lr by gamma each epoch (start high, converge smoothly)
    # Model lr stays constant, only SpectralShift decays
    spectral_shift_lr_decay = config.get("spectral_shift_lr_decay", 0.95)  # decay per epoch
    num_epochs = config.get("num_epochs", 80)

    def lr_lambda(_epoch):
        """Return lr multiplier. Model lr stays constant."""
        return 1.0  # Model lr unchanged

    def spectral_shift_lr_lambda(epoch):
        """Exponential decay for SpectralShift lr."""
        return spectral_shift_lr_decay ** epoch

    # Build per-group lr lambdas
    lr_lambdas = []
    for pg in param_groups:
        if pg.get("name", "").startswith("spectral_shift"):
            lr_lambdas.append(spectral_shift_lr_lambda)
        else:
            lr_lambdas.append(lr_lambda)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)

    if is_primary():
        print(f"SpectralShift lr scheduler: decay={spectral_shift_lr_decay} per epoch")

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
            raise ValueError("stage2_only=True requires stage1_checkpoint path to be set!")
        stage1_checkpoint = Path(stage1_checkpoint)
        if not stage1_checkpoint.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_checkpoint}")
        if is_primary():
            print(f"\n{'='*70}")
            print("STAGE 2 ONLY MODE")
            print(f"{'='*70}")
            print(f"  Skipping Stage 1 - loading checkpoint from:")
            print(f"    {stage1_checkpoint}")
            print(f"  Fine-tuning ONLY SpectralShift for {spectral_finetune_epochs} epochs")
            print(f"{'='*70}\n")

    elif is_primary() and use_two_stage:
        print(f"\n{'='*70}")
        print("TWO-STAGE TRAINING ENABLED")
        print(f"{'='*70}")
        print(f"  Stage 1: Train UNet + SpectralShift until early stopping")
        print(f"  Stage 2: FREEZE best UNet, fine-tune ONLY SpectralShift for {spectral_finetune_epochs} epochs")
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

    if not stage2_only:
        if is_primary() and use_two_stage:
            print(f"\n{'='*70}")
            print("STAGE 1: Training UNet + SpectralShift together")
            print(f"{'='*70}\n")

        for epoch in range(1, num_epochs + 1):
            if loaders.get("train_sampler") is not None:
                loaders["train_sampler"].set_epoch(epoch)

            train_metrics = train_epoch(
                model, loaders["train"], optimizer, device, config,
                wavelet_loss, spectral_loss,
                reverse_model, epoch, num_epochs,
                spectral_shift_fwd, spectral_shift_rev,
                disable_spectral=use_two_stage,  # Stage 1: Disable spectral if two-stage (pure UNet training)
                cond_encoder=cond_encoder,
                residual_corrector=residual_corrector,
                residual_optimizer=residual_optimizer,
                prob_loss=prob_loss,
                envelope_loss=envelope_loss,
            )

            barrier()

            val_metrics = evaluate(
                model, loaders["val"], device, wavelet_loss, spectral_loss,
                compute_phase=False, reverse_model=reverse_model, config=config,
                spectral_shift_fwd=spectral_shift_fwd, spectral_shift_rev=spectral_shift_rev,
                disable_spectral=use_two_stage,  # Stage 1: Disable spectral if two-stage (pure UNet validation)
                fast_mode=True,  # Stage 1: Only compute r and r² (skip PSD metrics)
                sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
                cond_encoder=cond_encoder,
                residual_corrector=residual_corrector,
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
                model, loaders["val"], device, wavelet_loss, spectral_loss,
                compute_phase=True, reverse_model=reverse_model, config=config,
                spectral_shift_fwd=spectral_shift_fwd, spectral_shift_rev=spectral_shift_rev,
                disable_spectral=use_two_stage,
                fast_mode=False,  # Full metrics for stage evaluation
                sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
            )

            barrier()

            # Evaluate on TEST set - ALL ranks must call this (FSDP requirement)
            test_metrics_stage1 = evaluate(
                model, loaders["test"], device, wavelet_loss, spectral_loss,
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
    # STAGE 2: FREEZE best UNet, fine-tune ONLY SpectralShift (if enabled)
    # =============================================================================
    # Only run Stage 2 if:
    # 1. Two-stage training is enabled OR stage2_only mode
    # 2. SpectralShift is available
    # 3. spectral_finetune_epochs > 0 (skip if --skip-spectral-finetune was used)
    should_run_stage2 = (
        (use_two_stage or stage2_only) and 
        spectral_shift_fwd is not None and 
        spectral_finetune_epochs > 0
    )
    
    if should_run_stage2:
        # Determine checkpoint path: user-provided (stage2_only) or best from stage1
        if stage2_only:
            checkpoint_path = stage1_checkpoint
            checkpoint_source = f"user-provided: {stage1_checkpoint}"
        else:
            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
            checkpoint_source = f"best from Stage 1 (epoch {best_epoch})"

        if is_primary():
            print(f"\n{'='*70}")
            print(f"STAGE 2: Fine-tuning ONLY SpectralShift for PSD correction")
            print(f"{'='*70}")
            print(f"Loading UNet from checkpoint: {checkpoint_source}")

        # Load UNet checkpoint
        # In stage2_only mode: skip loading SpectralShift (use fresh init with current config)
        # This allows changing SpectralShift architecture (band_width_hz, conditional, etc.)
        # Also validate cond_mode matches when loading for stage2
        barrier()
        expected_cond = config.get("cond_mode") if stage2_only else None
        load_checkpoint(
            checkpoint_path,
            model, reverse_model,
            spectral_shift_fwd, spectral_shift_rev,
            load_spectral_only=False,  # Load UNet
            skip_spectral=stage2_only,  # Skip SpectralShift in stage2_only mode (use fresh init)
            expected_cond_mode=expected_cond,  # Validate cond_mode for stage2_only
        )
        barrier()

        if stage2_only and is_primary():
            print(f"SpectralShift: using FRESH initialization (not loaded from checkpoint)")

        # FREEZE UNet parameters (forward and reverse)
        freeze_model_params(model)
        if reverse_model is not None:
            freeze_model_params(reverse_model)

        if is_primary():
            unet_params = count_trainable_params(model)
            rev_params = count_trainable_params(reverse_model) if reverse_model is not None else 0
            shift_fwd_params = count_trainable_params(spectral_shift_fwd)
            shift_rev_params = count_trainable_params(spectral_shift_rev) if spectral_shift_rev is not None else 0
            total_trainable = unet_params + rev_params + shift_fwd_params + shift_rev_params
            print(f"\nFROZEN UNet from: {checkpoint_source}")
            print(f"Trainable params: UNet={unet_params}, Rev={rev_params}, ShiftFwd={shift_fwd_params}, ShiftRev={shift_rev_params}")
            print(f"Total trainable: {total_trainable} (should be ONLY SpectralShift)\n")

        # Reset optimizer to ONLY SpectralShift parameters with high LR
        spectral_shift_lr_finetune = config.get("spectral_shift_lr", 0.1)
        param_groups_stage2 = []
        if spectral_shift_fwd is not None:
            param_groups_stage2.append({"params": list(spectral_shift_fwd.parameters()), "lr": spectral_shift_lr_finetune})
        if spectral_shift_rev is not None:
            param_groups_stage2.append({"params": list(spectral_shift_rev.parameters()), "lr": spectral_shift_lr_finetune})

        optimizer_stage2 = AdamW(param_groups_stage2, lr=spectral_shift_lr_finetune, betas=betas)

        # Reset early stopping for fine-tuning phase
        # IMPORTANT: Use PSD error (not loss) for best model selection in Stage 2
        # SpectralShift's purpose is to correct PSD, so we should select based on PSD error!
        best_psd_err_stage2 = float("inf")
        best_epoch_stage2 = 0
        patience_counter_stage2 = 0

        for epoch in range(1, spectral_finetune_epochs + 1):
            if loaders.get("train_sampler") is not None:
                loaders["train_sampler"].set_epoch(epoch + num_epochs)

            train_metrics = train_epoch(
                model, loaders["train"], optimizer_stage2, device, config,
                wavelet_loss, spectral_loss,
                reverse_model, epoch, spectral_finetune_epochs,
                spectral_shift_fwd, spectral_shift_rev,
                stage2_spectral_only=True,  # ONLY spectral loss - UNet frozen, pure PSD optimization
                cond_encoder=cond_encoder,
                residual_corrector=residual_corrector,
                residual_optimizer=residual_optimizer,
                prob_loss=prob_loss,
                envelope_loss=envelope_loss,
            )

            barrier()

            val_metrics = evaluate(
                model, loaders["val"], device, wavelet_loss, spectral_loss,
                compute_phase=False, reverse_model=reverse_model, config=config,
                spectral_shift_fwd=spectral_shift_fwd, spectral_shift_rev=spectral_shift_rev,
                spectral_only=True,  # Stage 2: Early stopping based on ONLY spectral loss
                fast_mode=False,  # Include PSD metrics to monitor frequency reconstruction
                sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
                cond_encoder=cond_encoder,
                residual_corrector=residual_corrector,
            )

            # Sync val_loss across ranks
            val_loss = val_metrics.get("loss", val_metrics["mae"])
            if dist.is_initialized():
                val_loss_tensor = torch.tensor(val_loss, device=device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                val_loss = val_loss_tensor.item()
                val_metrics["loss"] = val_loss

            # CRITICAL: Track best based on PSD ERROR, not loss!
            # SpectralShift's purpose is to correct PSD mismatch, so we select best model
            # based on actual PSD reconstruction quality (lower error = better)
            psd_err_fwd = val_metrics.get("psd_err_db", float("inf"))
            psd_err_rev = val_metrics.get("psd_err_db_rev", psd_err_fwd)  # Use fwd if rev not available
            # Average forward and reverse PSD errors for bidirectional training
            avg_psd_err = (psd_err_fwd + psd_err_rev) / 2.0 if "psd_err_db_rev" in val_metrics else psd_err_fwd

            # Sync PSD error across ranks
            if dist.is_initialized():
                psd_err_tensor = torch.tensor(avg_psd_err, device=device)
                dist.all_reduce(psd_err_tensor, op=dist.ReduceOp.AVG)
                avg_psd_err = psd_err_tensor.item()

            is_best_stage2 = avg_psd_err < best_psd_err_stage2
            if is_best_stage2:
                best_psd_err_stage2 = avg_psd_err
                best_epoch_stage2 = epoch
                patience_counter_stage2 = 0
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    model, optimizer_stage2, epoch,
                    CHECKPOINT_DIR / "best_model_stage2.pt",
                    is_fsdp=is_fsdp_wrapped,
                    reverse_model=reverse_model,
                    spectral_shift_fwd=spectral_shift_fwd,
                    spectral_shift_rev=spectral_shift_rev,
                    config=config,
                )
            else:
                patience_counter_stage2 += 1

            barrier()

            if is_primary():
                rev_str = ""
                if "corr_rev" in val_metrics:
                    psd_err_rev_db = val_metrics.get('psd_err_db_rev', 0)
                    psd_bias_rev = val_metrics.get('psd_diff_db_rev', 0)
                    rev_str = f" | Rev: r={val_metrics['corr_rev']:.3f}, PSD_err={psd_err_rev_db:.2f}dB (bias={psd_bias_rev:+.1f}dB)"

                psd_err_fwd_db = val_metrics.get('psd_err_db', 0)
                psd_bias_fwd = val_metrics.get('psd_diff_db', 0)
                print(f"[Stage2] Epoch {epoch}/{spectral_finetune_epochs} | "
                      f"Train: {train_metrics['loss']:.3f} | Val: {val_metrics['loss']:.3f} | "
                      f"Fwd: r={val_metrics['corr']:.3f}, PSD_err={psd_err_fwd_db:.2f}dB (bias={psd_bias_fwd:+.1f}dB){rev_str} | "
                      f"Best PSD_err: {best_psd_err_stage2:.2f}dB [UNet FROZEN]")
                sys.stdout.flush()

            if patience_counter_stage2 >= early_stop_patience:
                if is_primary():
                    print(f"Early stopping at epoch {epoch} (Stage 2 complete)")
                break

        # Load best stage2 checkpoint for final eval
        if is_primary():
            print(f"\nLoading best Stage 2 checkpoint from epoch {best_epoch_stage2} (PSD_err={best_psd_err_stage2:.2f}dB)...")
        barrier()
        load_checkpoint(
            CHECKPOINT_DIR / "best_model_stage2.pt",
            model, reverse_model,
            spectral_shift_fwd, spectral_shift_rev,
            load_spectral_only=False,
        )
        barrier()

    # Final test evaluation (full metrics, fast_mode=False)
    test_metrics = evaluate(
        model, loaders["test"], device, wavelet_loss, spectral_loss,
        compute_phase=True, reverse_model=reverse_model, config=config,
        spectral_shift_fwd=spectral_shift_fwd, spectral_shift_rev=spectral_shift_rev,
        fast_mode=False,  # Full metrics for final evaluation
        sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
        cond_encoder=cond_encoder,
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
    parser.add_argument("--base-channels", type=int, default=None, help="Base channels for model (default: from config)")
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP for distributed training")
    parser.add_argument("--fsdp-strategy", type=str, default="full",
                        choices=["full", "grad_op", "no_shard", "hybrid", "hybrid_zero2"],
                        help="FSDP sharding strategy. 'grad_op' recommended for 8x A100")
    parser.add_argument("--cpu-offload", action="store_true", help="Offload to CPU (large models)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (slower startup but faster training)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: from config)")

    # Session-based split arguments (for held-out session evaluation)
    parser.add_argument("--split-by-session", action="store_true",
                        help="Use session-based holdout instead of random splits. "
                             "Entire recording sessions are held out for test/val.")
    parser.add_argument("--n-test-sessions", type=int, default=1,
                        help="Number of sessions to hold out for testing (requires --split-by-session)")
    parser.add_argument("--n-val-sessions", type=int, default=1,
                        help="Number of sessions to hold out for validation (requires --split-by-session)")
    parser.add_argument("--session-column", type=str, default="recording_id",
                        help="CSV column name containing session/recording IDs")
    parser.add_argument("--force-recreate-splits", action="store_true",
                        help="Force recreation of data splits even if they exist on disk")

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

    # Stage control
    parser.add_argument("--skip-spectral-finetune", action="store_true",
                        help="Skip Stage 2 (spectral fine-tuning) - only run Stage 1")
    parser.add_argument("--stage2-only", action="store_true",
                        help="Skip Stage 1, load checkpoint and run ONLY Stage 2 (SpectralShift fine-tuning)")
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

    # Output scaling correction (learnable per-channel scale and bias in model)
    parser.add_argument("--output-scaling", action="store_true", default=True,
                        help="Enable learnable per-channel output scaling in model (default: True)")
    parser.add_argument("--no-output-scaling", action="store_false", dest="output_scaling",
                        help="Disable output scaling correction in model")

    # Enhanced odor conditioning options for AdaptiveSpectralShift
    parser.add_argument("--spectral-shift-odor-base", action="store_true", default=True,
                        help="Enable per-odor learnable base shifts (default: True)")
    parser.add_argument("--no-spectral-shift-odor-base", action="store_false", dest="spectral_shift_odor_base",
                        help="Disable per-odor base shifts")
    parser.add_argument("--spectral-shift-odor-scale", action="store_true", default=True,
                        help="Enable per-odor learnable scale factors (default: True)")
    parser.add_argument("--no-spectral-shift-odor-scale", action="store_false", dest="spectral_shift_odor_scale",
                        help="Disable per-odor scale factors")
    parser.add_argument("--spectral-shift-film", action="store_true", default=True,
                        help="Enable FiLM-style odor modulation (default: True)")
    parser.add_argument("--no-spectral-shift-film", action="store_false", dest="spectral_shift_film",
                        help="Disable FiLM modulation")
    parser.add_argument("--spectral-shift-film-hidden-mult", type=int, default=2,
                        help="Hidden dim multiplier for FiLM network (default: 2)")

    # Envelope loss (for SpectralShift training)
    parser.add_argument("--envelope-loss", action="store_true", default=True,
                        help="Enable envelope distribution matching loss (default: True)")
    parser.add_argument("--no-envelope-loss", action="store_false", dest="envelope_loss",
                        help="Disable envelope loss")
    parser.add_argument("--envelope-loss-weight", type=float, default=1.0,
                        help="Weight for envelope loss (default: 1.0)")
    parser.add_argument("--envelope-loss-bins", type=int, default=64,
                        help="Number of histogram bins for envelope loss (default: 64)")
    parser.add_argument("--envelope-loss-type", type=str, default="kl",
                        choices=["kl", "wasserstein", "mse"],
                        help="Envelope loss type: kl, wasserstein, or mse (default: kl)")

    # Loss function selection (for tier1 fair comparison)
    LOSS_CHOICES = ["l1", "huber", "wavelet", "l1_wavelet", "huber_wavelet"]
    parser.add_argument("--loss", type=str, default=None,
                        choices=LOSS_CHOICES,
                        help="Loss function: 'l1' (L1/MAE only), 'huber' (Huber only), "
                             "'wavelet' (Wavelet only), 'l1_wavelet' (L1 + Wavelet), "
                             "'huber_wavelet' (Huber + Wavelet combined). "
                             "If not specified, uses config default (huber_wavelet)")

    # Probabilistic loss (for tier 2.5 - added ON TOP of base loss)
    PROB_LOSS_CHOICES = [
        "none",               # No probabilistic loss (baseline)
        "gaussian_nll",       # Gaussian negative log-likelihood
        "laplacian_nll",      # Laplacian NLL (robust to outliers)
        "rayleigh",           # Signal envelope distribution
        "von_mises",          # Phase distribution (circular statistics)
        "kl_divergence",      # Distribution matching
        "cauchy_nll",         # Heavy-tailed (very robust)
        "student_t_nll",      # Controllable heavy tails
        "gumbel",             # Peak/extreme value distribution
        "gamma",              # Positive-valued signals
        "log_normal",         # Multiplicative processes
        "mixture",            # Gaussian mixture (multi-modal)
    ]
    parser.add_argument("--prob-loss", type=str, default="none",
                        choices=PROB_LOSS_CHOICES,
                        help="Probabilistic loss to ADD on top of base loss (for tier 2.5). "
                             "Default: 'none' (no probabilistic loss)")
    parser.add_argument("--prob-loss-weight", type=float, default=1.0,
                        help="Weight for probabilistic loss (default: 1.0)")

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
        if args.split_by_session:
            print(f"Using SESSION-BASED SPLITS: {args.n_test_sessions} test, {args.n_val_sessions} val sessions")
            print(f"Session column: {args.session_column}")

    # Build config FIRST - only override from args if explicitly provided
    config = DEFAULT_CONFIG.copy()
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.base_channels is not None:
        config["base_channels"] = args.base_channels
    if args.seed is not None:
        config["seed"] = args.seed

    # Load data based on dataset choice
    if args.dataset == "pfc":
        # PFC/Hippocampus dataset
        data = prepare_pfc_data(
            split_by_session=args.split_by_session,
            n_test_sessions=args.n_test_sessions,
            n_val_sessions=args.n_val_sessions,
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
            split_by_session=args.split_by_session,
            n_test_sessions=args.n_test_sessions,
            n_val_sessions=args.n_val_sessions,
            session_column=args.session_column,
            force_recreate_splits=args.force_recreate_splits,
            seed=config["seed"],
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

    # Stage control from CLI
    if args.skip_spectral_finetune:
        config["spectral_finetune_epochs"] = 0  # Disables Stage 2
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

    # Probabilistic loss (tier 2.5) - added ON TOP of base loss
    config["prob_loss_type"] = args.prob_loss if hasattr(args, 'prob_loss') else "none"
    config["prob_loss_weight"] = args.prob_loss_weight if hasattr(args, 'prob_loss_weight') else 1.0
    if config["prob_loss_type"] != "none" and is_primary():
        print(f"Probabilistic loss: {config['prob_loss_type']} (weight={config['prob_loss_weight']})")

    # Per-channel normalization (default: enabled)
    config["per_channel_norm"] = args.per_channel_norm if hasattr(args, 'per_channel_norm') else True
    if is_primary():
        print(f"Per-channel normalization: {'ENABLED' if config['per_channel_norm'] else 'DISABLED'}")

    # Output scaling correction in model (default: enabled)
    config["use_output_scaling"] = args.output_scaling if hasattr(args, 'output_scaling') else True
    if is_primary():
        print(f"Output scaling correction: {'ENABLED' if config['use_output_scaling'] else 'DISABLED'}")

    # Enhanced odor conditioning for SpectralShift
    config["spectral_shift_use_odor_base"] = getattr(args, 'spectral_shift_odor_base', True)
    config["spectral_shift_use_odor_scale"] = getattr(args, 'spectral_shift_odor_scale', True)
    config["spectral_shift_use_film"] = getattr(args, 'spectral_shift_film', True)
    config["spectral_shift_film_hidden_mult"] = getattr(args, 'spectral_shift_film_hidden_mult', 2)

    # Envelope loss configuration
    config["use_envelope_loss"] = getattr(args, 'envelope_loss', True)
    config["envelope_loss_weight"] = getattr(args, 'envelope_loss_weight', 1.0)
    config["envelope_loss_n_bins"] = getattr(args, 'envelope_loss_bins', 64)
    config["envelope_loss_type"] = getattr(args, 'envelope_loss_type', 'kl')

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
