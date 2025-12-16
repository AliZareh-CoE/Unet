"""Validation script for neural signal translation models.

This script loads trained checkpoints and generates comprehensive evaluation plots
comparing real vs generated signals for BOTH directions (OB→PCx and PCx→OB).

Usage:
    python validate_model.py --checkpoint artifacts/checkpoints/best_model_stage2.pt
    python validate_model.py --checkpoint path/to/checkpoint.pt --split test --output_dir results/
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as scipy_signal
from scipy import stats
from scipy.signal import welch, coherence, butter, sosfilt, hilbert
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CLI
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from models import (
    CondUNet1D,
    OptimalSpectralBias,
    EnvelopeHistogramMatching,
    SAMPLING_RATE_HZ as MODEL_SAMPLING_RATE,
)
from data import (
    prepare_data,
    create_dataloaders,
    prepare_pfc_data,
    create_pfc_dataloaders,
    create_single_session_dataloader,
    crop_to_target_torch,
    SAMPLING_RATE_HZ,
    PFC_SAMPLING_RATE_HZ,
    CROP_START,
    CROP_END,
    get_odor_name,
    PFC_CHANNELS,
    CA1_CHANNELS,
    NUM_TRIAL_TYPES,
)
from train import DEFAULT_CONFIG

# Dataset type enumeration
from enum import Enum
class DatasetType(Enum):
    OLFACTORY = "olfactory"
    PFC = "pfc"

# Recording system imports for post-hoc analysis
try:
    from recording import (
        SaliencyAnalyzer,
        ImportanceAnalyzer,
        NeuroInterpretabilityAnalyzer,
        NeuroVisualizer,
    )
    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False
    SaliencyAnalyzer = None
    ImportanceAnalyzer = None
    NeuroInterpretabilityAnalyzer = None
    NeuroVisualizer = None


# =============================================================================
# Configuration
# =============================================================================

# Model config derived from train.py DEFAULT_CONFIG
DEFAULT_MODEL_CONFIG = {
    "in_channels": 32,
    "out_channels": 32,
    "base_channels": DEFAULT_CONFIG["base_channels"],
    "n_odors": 7,
    "emb_dim": 128,
    "dropout": DEFAULT_CONFIG["dropout"],
    "use_attention": DEFAULT_CONFIG["use_attention"],
    "attention_type": DEFAULT_CONFIG["attention_type"],
    "norm_type": DEFAULT_CONFIG["norm_type"],
    "cond_mode": DEFAULT_CONFIG["cond_mode"],
    "use_spectral_shift": DEFAULT_CONFIG["use_spectral_shift"],
    "conv_type": DEFAULT_CONFIG["conv_type"],
    "use_se": DEFAULT_CONFIG["use_se"],
    "conv_kernel_size": DEFAULT_CONFIG["conv_kernel_size"],
    "conv_dilations": DEFAULT_CONFIG["conv_dilations"],
    "n_downsample": DEFAULT_CONFIG["n_downsample"],
    "use_output_scaling": DEFAULT_CONFIG.get("use_output_scaling", True),
}

# Spectral shift config derived from train.py DEFAULT_CONFIG
# Uses OptimalSpectralBias (fixed per-odor PSD correction)
DEFAULT_SPECTRAL_SHIFT_CONFIG = {
    "sample_rate": SAMPLING_RATE_HZ,
    "n_channels": 32,
    "n_odors": 7,
    "band_width_hz": DEFAULT_CONFIG.get("spectral_shift_band_width_hz", None),
}


def per_channel_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply per-channel z-score normalization to a batch of signals.

    For each channel independently, normalize to zero mean and unit variance.
    This helps handle amplitude variations across channels and recordings.

    Args:
        x: Input tensor of shape [B, C, T]
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape
    """
    # Compute mean and std per channel (over time dimension)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=eps)
    return (x - mean) / std


def get_dataset_config(dataset_type: DatasetType) -> Dict[str, Any]:
    """Get dataset-specific configuration for model and spectral shift.
    
    Args:
        dataset_type: Which dataset to configure for
        
    Returns:
        Dict with 'in_channels', 'out_channels', 'n_conditions', 'sampling_rate',
        'source_name', 'target_name', 'condition_name'
    """
    if dataset_type == DatasetType.PFC:
        return {
            "in_channels": PFC_CHANNELS,       # 64 PFC channels
            "out_channels": CA1_CHANNELS,      # 32 CA1 channels
            "n_conditions": NUM_TRIAL_TYPES,   # 2 (Left/Right)
            "sampling_rate": PFC_SAMPLING_RATE_HZ,  # 1250 Hz
            "source_name": "PFC",
            "target_name": "CA1",
            "condition_name": "trial_type",
        }
    else:  # OLFACTORY
        return {
            "in_channels": 32,                 # 32 OB channels
            "out_channels": 32,                # 32 PCx channels
            "n_conditions": 7,                 # 7 odors
            "sampling_rate": SAMPLING_RATE_HZ, # 1000 Hz
            "source_name": "OB",
            "target_name": "PCx",
            "condition_name": "odor",
        }


# Plot styling
COLORS = {
    "real": "#1f77b4",      # Blue
    "generated": "#ff7f0e", # Orange
    "error": "#d62728",     # Red
    "forward": "#2ca02c",   # Green
    "reverse": "#9467bd",   # Purple
}

# Analysis constants (centralized for easy tuning)
DEFAULT_NPERSEG = 512              # Welch PSD segment length
DEFAULT_N_BOOTSTRAP = 1000         # Bootstrap iterations for CI
DEFAULT_N_PERMUTATIONS = 5000      # Permutation test iterations
FREQ_LIMIT_HZ = 100                # Max frequency for PSD plots
STIMULUS_ONSET_SEC = 2.0           # Stimulus onset time marker
MAX_VIS_SAMPLES = 100000           # Max samples for visualizations
DEFAULT_RANDOM_SEED = 42           # Default seed for reproducibility

# Frequency bands for analysis
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "low_gamma": (30, 50),
    "high_gamma": (50, 100),
}

# Default output formats for publication
DEFAULT_FORMATS = ["png", "svg", "pdf"]


# =============================================================================
# Multi-Format Save Utility
# =============================================================================

def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    name: str,
    formats: List[str] = None,
):
    """Save figure in multiple formats for publication.

    Args:
        fig: Matplotlib figure to save
        output_dir: Base output directory
        name: Filename (without extension)
        formats: List of formats to save (default: png, svg, pdf)
    """
    if formats is None:
        formats = DEFAULT_FORMATS

    for fmt in formats:
        fmt_dir = output_dir / fmt
        fmt_dir.mkdir(parents=True, exist_ok=True)

        dpi = 300 if fmt == "png" else None
        fig.savefig(
            fmt_dir / f"{name}.{fmt}",
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            transparent=(fmt in ["svg", "pdf"]),
            facecolor="white" if fmt == "png" else "none",
        )


# =============================================================================
# Statistical Utility Functions
# =============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: Data array
        statistic: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap resamples
        ci: Confidence level (default: 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    boot_stats = np.array([
        statistic(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return tuple(np.percentile(boot_stats, [alpha * 100, (1 - alpha) * 100]))


def compute_effect_size_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute Cohen's d with bootstrap confidence interval.

    Returns:
        Dict with 'd', 'ci_lower', 'ci_upper'
    """
    d = compute_cohens_d(group1, group2)

    rng = np.random.RandomState(seed)
    boot_ds = []
    for _ in range(n_bootstrap):
        idx1 = rng.choice(len(group1), size=len(group1), replace=True)
        idx2 = rng.choice(len(group2), size=len(group2), replace=True)
        boot_ds.append(compute_cohens_d(group1[idx1], group2[idx2]))

    alpha = (1 - ci) / 2
    ci_bounds = np.percentile(boot_ds, [alpha * 100, (1 - alpha) * 100])

    return {"d": d, "ci_lower": ci_bounds[0], "ci_upper": ci_bounds[1]}


def bland_altman_stats(
    measured: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, float]:
    """Compute Bland-Altman agreement statistics.

    Args:
        measured: Measured/predicted values
        reference: Reference/ground truth values

    Returns:
        Dict with bias, lower_loa, upper_loa (limits of agreement)
    """
    diff = measured - reference
    mean_vals = (measured + reference) / 2

    bias = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # 95% limits of agreement
    lower_loa = bias - 1.96 * std_diff
    upper_loa = bias + 1.96 * std_diff

    return {
        "bias": bias,
        "std_diff": std_diff,
        "lower_loa": lower_loa,
        "upper_loa": upper_loa,
        "mean_vals": mean_vals,
        "diff": diff,
    }


# =============================================================================
# Model Loading
# =============================================================================

def get_model_config(
    dataset_type: DatasetType = DatasetType.OLFACTORY,
    n_conditions: Optional[int] = None,
    checkpoint_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get model configuration for the specified dataset.
    
    Args:
        dataset_type: Which dataset the model was trained on
        n_conditions: Override number of conditions (odors/trial_types)
        checkpoint_config: Config from checkpoint (takes priority if available)
        
    Returns:
        Model configuration dict
    """
    config = DEFAULT_MODEL_CONFIG.copy()
    ds_config = get_dataset_config(dataset_type)
    
    # Use checkpoint config if available, else dataset defaults
    if checkpoint_config is not None:
        config["in_channels"] = checkpoint_config.get("in_channels", ds_config["in_channels"])
        config["out_channels"] = checkpoint_config.get("out_channels", ds_config["out_channels"])
        config["n_odors"] = checkpoint_config.get("n_odors", ds_config["n_conditions"])
    else:
        config["in_channels"] = ds_config["in_channels"]
        config["out_channels"] = ds_config["out_channels"]
        config["n_odors"] = n_conditions if n_conditions else ds_config["n_conditions"]
    
    return config


def create_model(config: Dict[str, Any], device: torch.device) -> CondUNet1D:
    """Create a CondUNet1D model from config."""
    model = CondUNet1D(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        base=config["base_channels"],
        n_odors=config["n_odors"],
        emb_dim=config["emb_dim"],
        dropout=config["dropout"],
        use_attention=config["use_attention"],
        attention_type=config["attention_type"],
        norm_type=config["norm_type"],
        cond_mode=config["cond_mode"],
        use_spectral_shift=config["use_spectral_shift"],
        conv_type=config["conv_type"],
        use_se=config["use_se"],
        conv_kernel_size=config["conv_kernel_size"],
        dilations=config["conv_dilations"],
        n_downsample=config["n_downsample"],
        use_output_scaling=config.get("use_output_scaling", True),
    )
    return model.to(device)


def create_spectral_shift(config: Dict[str, Any], device: torch.device):
    """Create a spectral shift module from config.

    Uses OptimalSpectralBias - fixed per-odor PSD correction (not trainable).
    """
    shift = OptimalSpectralBias(
        n_channels=config["n_channels"],
        n_odors=config["n_odors"],
        sample_rate=config.get("sample_rate", SAMPLING_RATE_HZ),
        band_width_hz=config.get("band_width_hz", None),
    )
    return shift.to(device)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    dataset_type: DatasetType = DatasetType.OLFACTORY,
    n_conditions: Optional[int] = None,
) -> Tuple[CondUNet1D, CondUNet1D, nn.Module, nn.Module, Dict[str, Any]]:
    """Load models and spectral shifts from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load models onto
        dataset_type: Which dataset the model was trained on
        n_conditions: Override number of conditions (from data)

    Returns:
        Tuple of (model_fwd, model_rev, spectral_shift_fwd, spectral_shift_rev, checkpoint_config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config from checkpoint if available
    checkpoint_config = checkpoint.get("config", None)
    ds_config = get_dataset_config(dataset_type)
    
    # Determine actual channel counts and conditions
    if checkpoint_config is not None:
        in_channels = checkpoint_config.get("in_channels", ds_config["in_channels"])
        out_channels = checkpoint_config.get("out_channels", ds_config["out_channels"])
        n_odors = checkpoint_config.get("n_odors", ds_config["n_conditions"])
        sampling_rate = checkpoint_config.get("sampling_rate", ds_config["sampling_rate"])
        print(f"  Using config from checkpoint: {in_channels}→{out_channels}ch, {n_odors} conditions")
    else:
        # No config in checkpoint - try to infer from weight shapes
        model_state = checkpoint.get("model", {})
        
        # Infer n_odors from embedding weight shape: embed.weight has shape [n_odors, emb_dim]
        if "embed.weight" in model_state:
            inferred_n_odors = model_state["embed.weight"].shape[0]
        else:
            inferred_n_odors = None
            
        # Infer in_channels from first conv layer: inc.conv.0.branches.0.conv.weight has shape [out, in, kernel]
        # The input channels are the second dimension
        if "inc.conv.0.branches.0.conv.weight" in model_state:
            inferred_in_channels = model_state["inc.conv.0.branches.0.conv.weight"].shape[1]
        else:
            inferred_in_channels = None
            
        # Infer out_channels from output layer if present, or from reverse model
        # outc.conv.weight has shape [out_channels, in_features, 1]
        if "outc.conv.weight" in model_state:
            inferred_out_channels = model_state["outc.conv.weight"].shape[0]
        else:
            inferred_out_channels = None
            
        # Use inferred values if available, otherwise fall back to dataset defaults
        if inferred_in_channels is not None:
            in_channels = inferred_in_channels
            print(f"  Inferred in_channels={in_channels} from checkpoint weights")
        else:
            in_channels = ds_config["in_channels"]
            
        if inferred_out_channels is not None:
            out_channels = inferred_out_channels
            print(f"  Inferred out_channels={out_channels} from checkpoint weights")
        else:
            out_channels = ds_config["out_channels"]
            
        if inferred_n_odors is not None:
            n_odors = inferred_n_odors
            print(f"  Inferred n_conditions={n_odors} from checkpoint weights")
        else:
            n_odors = n_conditions if n_conditions else ds_config["n_conditions"]
            
        # Determine sampling rate based on inferred channels
        # PFC: 64→32, Olfactory: 32→32
        if in_channels == 64 and out_channels == 32:
            sampling_rate = PFC_SAMPLING_RATE_HZ
            # Update region names for PFC
            ds_config = get_dataset_config(DatasetType.PFC)
            print(f"  Detected PFC dataset from channel configuration (64→32)")
        else:
            sampling_rate = ds_config["sampling_rate"]
            
        print(f"  Final config: {in_channels}→{out_channels}ch, {n_odors} conditions, {sampling_rate}Hz")

    # Get model config with correct channels
    model_config = get_model_config(dataset_type, n_conditions, checkpoint_config)
    # Override with inferred values
    model_config["in_channels"] = in_channels
    model_config["out_channels"] = out_channels
    model_config["n_odors"] = n_odors
    
    # Build spectral shift config with correct channels and sampling rate
    spectral_config = DEFAULT_SPECTRAL_SHIFT_CONFIG.copy()
    spectral_config["n_odors"] = n_odors
    spectral_config["sample_rate"] = sampling_rate
    
    # Check if SpectralShift was trained with different n_odors
    # This can happen if training had a bug or used defaults
    spectral_n_odors_fwd = n_odors
    spectral_n_odors_rev = n_odors
    if "spectral_shift_fwd" in checkpoint:
        ss_state = checkpoint["spectral_shift_fwd"]
        if "embed.weight" in ss_state:
            spectral_n_odors_fwd = ss_state["embed.weight"].shape[0]
            if spectral_n_odors_fwd != n_odors:
                print(f"  Warning: SpectralShift fwd was trained with {spectral_n_odors_fwd} conditions (model has {n_odors})")
    if "spectral_shift_rev" in checkpoint:
        ss_state = checkpoint["spectral_shift_rev"]
        if "embed.weight" in ss_state:
            spectral_n_odors_rev = ss_state["embed.weight"].shape[0]
            if spectral_n_odors_rev != n_odors:
                print(f"  Warning: SpectralShift rev was trained with {spectral_n_odors_rev} conditions (model has {n_odors})")
    
    # Forward SpectralShift: operates on forward output (out_channels)
    spectral_config_fwd = spectral_config.copy()
    spectral_config_fwd["n_channels"] = out_channels
    spectral_config_fwd["n_odors"] = spectral_n_odors_fwd  # Use actual trained n_odors
    
    # Reverse SpectralShift: operates on reverse output (in_channels)  
    spectral_config_rev = spectral_config.copy()
    spectral_config_rev["n_channels"] = in_channels
    spectral_config_rev["n_odors"] = spectral_n_odors_rev  # Use actual trained n_odors

    # Create forward model (source → target)
    model_fwd = create_model(model_config, device)
    model_fwd.load_state_dict(checkpoint["model"])
    model_fwd.eval()

    # Create reverse model (target → source)
    model_rev_config = model_config.copy()
    model_rev_config["in_channels"] = out_channels   # Reverse: target → source
    model_rev_config["out_channels"] = in_channels
    model_rev = create_model(model_rev_config, device)
    if "reverse_model" in checkpoint:
        model_rev.load_state_dict(checkpoint["reverse_model"])
    model_rev.eval()

    # Create spectral shift modules with correct channel counts
    spectral_shift_fwd = create_spectral_shift(spectral_config_fwd, device)
    spectral_shift_rev = create_spectral_shift(spectral_config_rev, device)

    if "spectral_shift_fwd" in checkpoint:
        spectral_shift_fwd.load_state_dict(checkpoint["spectral_shift_fwd"])
    if "spectral_shift_rev" in checkpoint:
        spectral_shift_rev.load_state_dict(checkpoint["spectral_shift_rev"])

    spectral_shift_fwd.eval()
    spectral_shift_rev.eval()
    
    # Build effective config for return
    effective_config = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "n_conditions": n_odors,
        "sampling_rate": sampling_rate,
        "source_name": ds_config["source_name"],
        "target_name": ds_config["target_name"],
        "condition_name": ds_config["condition_name"],
    }

    return model_fwd, model_rev, spectral_shift_fwd, spectral_shift_rev, effective_config


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(
    model_fwd: CondUNet1D,
    model_rev: CondUNet1D,
    spectral_shift_fwd: nn.Module,
    spectral_shift_rev: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    vocab: Dict[str, int],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Generate signals for all samples in dataloader.
    
    Args:
        model_fwd: Forward translation model (source → target)
        model_rev: Reverse translation model (target → source)
        spectral_shift_fwd: SpectralShift for forward output
        spectral_shift_rev: SpectralShift for reverse output
        dataloader: DataLoader for evaluation data
        device: Torch device
        vocab: Label vocabulary (name → id)
        config: Dataset config with 'source_name', 'target_name', 'condition_name'

    Returns:
        Dictionary with keys:
        - source_real: Real source signals [N, C_src, T]
        - target_real: Real target signals [N, C_tgt, T]
        - target_gen: Generated target signals (from source) [N, C_tgt, T]
        - source_gen: Generated source signals (from target) [N, C_src, T]
        - condition_ids: Condition label indices [N]
        - condition_names: Condition names [N]
        - config: Dataset configuration
        
        Also provides legacy keys for backward compatibility:
        - ob_real, pcx_real, pcx_gen, ob_gen, odor_ids, odor_names
    """
    # Reverse vocab: id -> name
    id_to_name = {v: k for k, v in vocab.items()}

    source_real_list = []
    target_real_list = []
    target_gen_list = []
    source_gen_list = []
    condition_ids_list = []
    condition_names_list = []

    with torch.no_grad():
        for source, target, condition in tqdm(dataloader, desc="Generating signals"):
            source = source.to(device)
            target = target.to(device)
            condition = condition.to(device)

            # Apply per-channel normalization (same as training)
            source_norm = per_channel_normalize(source)
            target_norm = per_channel_normalize(target)

            # Forward: source → target
            target_pred = model_fwd(source_norm, condition)
            target_pred = spectral_shift_fwd(target_pred, odor_ids=condition)

            # Reverse: target → source
            source_pred = model_rev(target_norm, condition)
            source_pred = spectral_shift_rev(source_pred, odor_ids=condition)

            # Crop to target window
            # IMPORTANT: Use normalized versions for fair comparison with model outputs
            # Model was trained on normalized data, so predictions are in normalized space
            source_cropped = crop_to_target_torch(source_norm)
            target_cropped = crop_to_target_torch(target_norm)
            target_pred_cropped = crop_to_target_torch(target_pred)
            source_pred_cropped = crop_to_target_torch(source_pred)

            # Store
            source_real_list.append(source_cropped.cpu().numpy())
            target_real_list.append(target_cropped.cpu().numpy())
            target_gen_list.append(target_pred_cropped.cpu().numpy())
            source_gen_list.append(source_pred_cropped.cpu().numpy())
            condition_np = condition.cpu().numpy()
            condition_ids_list.append(condition_np)

            # Get condition names (reuse converted array)
            for cid in condition_np:
                condition_names_list.append(id_to_name.get(cid, f"condition_{cid}"))

    # Build result dict with generic names
    result = {
        "source_real": np.concatenate(source_real_list, axis=0),
        "target_real": np.concatenate(target_real_list, axis=0),
        "target_gen": np.concatenate(target_gen_list, axis=0),
        "source_gen": np.concatenate(source_gen_list, axis=0),
        "condition_ids": np.concatenate(condition_ids_list, axis=0),
        "condition_names": condition_names_list,
        "config": config,
        # Include sampling rate for plotting functions
        "sampling_rate": config.get("sampling_rate", SAMPLING_RATE_HZ) if config else SAMPLING_RATE_HZ,
    }
    
    # Add legacy keys for backward compatibility with existing plotting functions
    result["ob_real"] = result["source_real"]
    result["pcx_real"] = result["target_real"]
    result["pcx_gen"] = result["target_gen"]
    result["ob_gen"] = result["source_gen"]
    result["odor_ids"] = result["condition_ids"]
    result["odor_names"] = result["condition_names"]
    
    return result


def get_region_labels(signals: Dict[str, Any]) -> Dict[str, str]:
    """Get human-readable region labels for plotting based on dataset config.
    
    Args:
        signals: The signals dict which includes 'config' key
        
    Returns:
        Dict with keys: 'source', 'target', 'condition', 'forward_title', 'reverse_title'
    """
    config = signals.get("config", {})
    source = config.get("source_name", "Source")
    target = config.get("target_name", "Target")
    condition = config.get("condition_name", "condition")
    
    return {
        "source": source,
        "target": target,
        "condition": condition,
        "forward_title": f"Forward: {source} → {target}",
        "reverse_title": f"Reverse: {target} → {source}",
        "real_source": f"Real {source}",
        "real_target": f"Real {target}",
        "gen_source": f"Generated {source}",
        "gen_target": f"Generated {target}",
    }


# =============================================================================
# Analysis Utilities
# =============================================================================


class PSDCache:
    """Cache for PSD computations to avoid redundant Welch calculations.

    This cache stores PSD results keyed by array identity and parameters,
    eliminating redundant computations across multiple plotting functions.
    """

    def __init__(self):
        self._cache: Dict[Tuple[int, Tuple[int, ...], float, int], Tuple[np.ndarray, np.ndarray]] = {}

    def _get_key(self, signals: np.ndarray, fs: float, nperseg: int) -> Tuple[int, Tuple[int, ...], float, int]:
        """Generate cache key from signal array identity and parameters."""
        return (id(signals), signals.shape, fs, nperseg)

    def get(self, signals: np.ndarray, fs: float, nperseg: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve cached PSD if available."""
        key = self._get_key(signals, fs, nperseg)
        return self._cache.get(key)

    def set(self, signals: np.ndarray, fs: float, nperseg: int,
            freqs: np.ndarray, psd: np.ndarray) -> None:
        """Store PSD result in cache."""
        key = self._get_key(signals, fs, nperseg)
        self._cache[key] = (freqs, psd)

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# Global PSD cache instance
_psd_cache = PSDCache()


def compute_psd(
    signals: np.ndarray,
    fs: float = SAMPLING_RATE_HZ,
    nperseg: int = 512,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density using Welch's method (vectorized with caching).

    Args:
        signals: Array of shape [N, C, T] or [C, T]
        fs: Sampling frequency
        nperseg: Length of each segment for Welch
        use_cache: Whether to use caching (default True)

    Returns:
        freqs: Frequency array
        psd: PSD array (same shape as input except last dim is freq)
    """
    was_2d = signals.ndim == 2
    if was_2d:
        signals = signals[np.newaxis, ...]

    # Check cache first
    if use_cache:
        cached = _psd_cache.get(signals, fs, nperseg)
        if cached is not None:
            freqs, psd = cached
            return (freqs, psd[0]) if was_2d else (freqs, psd)

    N, C, T = signals.shape

    # Get frequency array from a single sample
    freqs, _ = scipy_signal.welch(signals[0, 0], fs=fs, nperseg=nperseg)
    n_freqs = len(freqs)

    # Vectorized: reshape to 2D and compute all at once
    signals_flat = signals.reshape(N * C, T)
    _, psd_flat = scipy_signal.welch(signals_flat, fs=fs, nperseg=nperseg, axis=-1)
    psd = psd_flat.reshape(N, C, n_freqs)

    # Cache result
    if use_cache:
        _psd_cache.set(signals, fs, nperseg, freqs, psd)

    return (freqs, psd[0]) if was_2d else (freqs, psd)


def compute_correlation(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation per trial (vectorized).

    Args:
        pred: Predicted signals [N, C, T]
        target: Target signals [N, C, T]

    Returns:
        correlations: Array of correlations [N]
    """
    N = pred.shape[0]

    # Flatten each trial: [N, C*T]
    pred_flat = pred.reshape(N, -1)
    target_flat = target.reshape(N, -1)

    # Vectorized Pearson correlation: mean-center, then normalized dot product
    pred_centered = pred_flat - pred_flat.mean(axis=1, keepdims=True)
    target_centered = target_flat - target_flat.mean(axis=1, keepdims=True)

    numerator = np.sum(pred_centered * target_centered, axis=1)
    pred_std = np.sqrt(np.sum(pred_centered ** 2, axis=1))
    target_std = np.sqrt(np.sum(target_centered ** 2, axis=1))

    correlations = numerator / (pred_std * target_std + 1e-10)

    return correlations


def compute_band_power(psd: np.ndarray, freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """Compute power in a frequency band.

    Args:
        psd: PSD array [N, C, F] or [C, F]
        freqs: Frequency array [F]
        band: (low_freq, high_freq) tuple

    Returns:
        power: Band power [N, C] or [C]
    """
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    return np.mean(psd[..., idx], axis=-1)


def compute_plv(signal1: np.ndarray, signal2: np.ndarray, fs: float = SAMPLING_RATE_HZ) -> float:
    """Compute Phase Locking Value between two signals.

    Args:
        signal1, signal2: 1D signals
        fs: Sampling frequency

    Returns:
        PLV value between 0 and 1
    """
    # Hilbert transform to get analytic signal
    analytic1 = scipy_signal.hilbert(signal1)
    analytic2 = scipy_signal.hilbert(signal2)

    # Get instantaneous phase
    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)

    # Compute PLV
    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    return plv


def compute_channel_correlation_matrix(signals: np.ndarray) -> np.ndarray:
    """Compute channel-to-channel correlation matrix efficiently.

    Uses a single np.corrcoef call instead of O(C^2) nested loops.

    Args:
        signals: Array of shape [N, C, T] where N=trials, C=channels, T=timepoints

    Returns:
        Correlation matrix of shape [C, C]
    """
    n_channels = signals.shape[1]
    # Flatten trials and time: reshape from [N, C, T] to [C, N*T]
    flattened = signals.transpose(1, 0, 2).reshape(n_channels, -1)
    # Single corrcoef call computes full C x C matrix
    return np.corrcoef(flattened)


# =============================================================================
# Plotting Functions
# =============================================================================

def setup_plot_style():
    """Set up consistent plot styling."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })


def plot_psd_comparison(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 1: Global PSD comparison for both directions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = get_region_labels(signals)

    # Forward: source → target
    freqs, psd_real = compute_psd(signals["target_real"], fs)
    _, psd_gen = compute_psd(signals["target_gen"], fs)

    # Average across trials and channels
    psd_real_mean = psd_real.mean(axis=(0, 1))
    psd_real_std = psd_real.mean(axis=1).std(axis=0)
    psd_gen_mean = psd_gen.mean(axis=(0, 1))
    psd_gen_std = psd_gen.mean(axis=1).std(axis=0)

    # Convert to dB
    psd_real_db = 10 * np.log10(psd_real_mean + 1e-10)
    psd_gen_db = 10 * np.log10(psd_gen_mean + 1e-10)

    ax = axes[0]
    ax.plot(freqs, psd_real_db, color=COLORS["real"], label=labels["real_target"], linewidth=2)
    ax.plot(freqs, psd_gen_db, color=COLORS["generated"], label=labels["gen_target"], linewidth=2)
    ax.fill_between(freqs, psd_real_db - 1, psd_real_db + 1, alpha=0.2, color=COLORS["real"])
    ax.fill_between(freqs, psd_gen_db - 1, psd_gen_db + 1, alpha=0.2, color=COLORS["generated"])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(labels["forward_title"])
    ax.set_xlim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reverse: target → source
    freqs, psd_real = compute_psd(signals["source_real"], fs)
    _, psd_gen = compute_psd(signals["source_gen"], fs)

    psd_real_mean = psd_real.mean(axis=(0, 1))
    psd_gen_mean = psd_gen.mean(axis=(0, 1))
    psd_real_db = 10 * np.log10(psd_real_mean + 1e-10)
    psd_gen_db = 10 * np.log10(psd_gen_mean + 1e-10)

    ax = axes[1]
    ax.plot(freqs, psd_real_db, color=COLORS["real"], label=labels["real_source"], linewidth=2)
    ax.plot(freqs, psd_gen_db, color=COLORS["generated"], label=labels["gen_source"], linewidth=2)
    ax.fill_between(freqs, psd_real_db - 1, psd_real_db + 1, alpha=0.2, color=COLORS["real"])
    ax.fill_between(freqs, psd_gen_db - 1, psd_gen_db + 1, alpha=0.2, color=COLORS["generated"])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(labels["reverse_title"])
    ax.set_xlim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Power Spectral Density: Real vs Generated", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "01_psd_comparison", formats)
    plt.close()


def plot_psd_per_odor(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 2: Per-condition PSD comparison."""
    labels = get_region_labels(signals)
    condition_names = np.unique(signals["condition_names"])
    n_conditions = len(condition_names)

    fig, axes = plt.subplots(n_conditions, 2, figsize=(14, 3 * n_conditions))

    for i, cond in enumerate(condition_names):
        mask = np.array(signals["condition_names"]) == cond

        # Forward: source → target
        freqs, psd_real = compute_psd(signals["target_real"][mask], fs)
        _, psd_gen = compute_psd(signals["target_gen"][mask], fs)

        psd_real_db = 10 * np.log10(psd_real.mean(axis=(0, 1)) + 1e-10)
        psd_gen_db = 10 * np.log10(psd_gen.mean(axis=(0, 1)) + 1e-10)

        ax = axes[i, 0]
        ax.plot(freqs, psd_real_db, color=COLORS["real"], label="Real", linewidth=1.5)
        ax.plot(freqs, psd_gen_db, color=COLORS["generated"], label="Generated", linewidth=1.5)
        ax.set_xlim(0, 100)
        ax.set_ylabel(f"{cond}\nPower (dB)")
        if i == 0:
            ax.set_title(labels["forward_title"])
            ax.legend()
        if i == n_conditions - 1:
            ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, alpha=0.3)

        # Reverse: target → source
        freqs, psd_real = compute_psd(signals["source_real"][mask], fs)
        _, psd_gen = compute_psd(signals["source_gen"][mask], fs)

        psd_real_db = 10 * np.log10(psd_real.mean(axis=(0, 1)) + 1e-10)
        psd_gen_db = 10 * np.log10(psd_gen.mean(axis=(0, 1)) + 1e-10)

        ax = axes[i, 1]
        ax.plot(freqs, psd_real_db, color=COLORS["real"], label="Real", linewidth=1.5)
        ax.plot(freqs, psd_gen_db, color=COLORS["generated"], label="Generated", linewidth=1.5)
        ax.set_xlim(0, 100)
        if i == 0:
            ax.set_title(labels["reverse_title"])
            ax.legend()
        if i == n_conditions - 1:
            ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Per-{labels['condition'].title()} Power Spectral Density", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "02_psd_per_odor", formats)
    plt.close()


def plot_mean_waveform(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 3: Mean waveform with variance."""
    labels = get_region_labels(signals)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time axis (after cropping: 0.5s to 4.5s)
    T = signals["target_real"].shape[-1]
    time = np.arange(T) / fs + 0.5

    # Forward: source → target (average across trials and channels)
    real_mean = signals["target_real"].mean(axis=(0, 1))
    real_std = signals["target_real"].mean(axis=1).std(axis=0)
    gen_mean = signals["target_gen"].mean(axis=(0, 1))
    gen_std = signals["target_gen"].mean(axis=1).std(axis=0)

    ax = axes[0]
    ax.plot(time, real_mean, color=COLORS["real"], label=labels["real_target"], linewidth=1.5)
    ax.fill_between(time, real_mean - real_std, real_mean + real_std, alpha=0.2, color=COLORS["real"])
    ax.plot(time, gen_mean, color=COLORS["generated"], label=labels["gen_target"], linewidth=1.5)
    ax.fill_between(time, gen_mean - gen_std, gen_mean + gen_std, alpha=0.2, color=COLORS["generated"])
    ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5, label="Stimulus onset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(labels["forward_title"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reverse: target → source
    real_mean = signals["source_real"].mean(axis=(0, 1))
    real_std = signals["source_real"].mean(axis=1).std(axis=0)
    gen_mean = signals["source_gen"].mean(axis=(0, 1))
    gen_std = signals["source_gen"].mean(axis=1).std(axis=0)

    ax = axes[1]
    ax.plot(time, real_mean, color=COLORS["real"], label=labels["real_source"], linewidth=1.5)
    ax.fill_between(time, real_mean - real_std, real_mean + real_std, alpha=0.2, color=COLORS["real"])
    ax.plot(time, gen_mean, color=COLORS["generated"], label=labels["gen_source"], linewidth=1.5)
    ax.fill_between(time, gen_mean - gen_std, gen_mean + gen_std, alpha=0.2, color=COLORS["generated"])
    ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5, label="Stimulus onset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(labels["reverse_title"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Mean Waveform with Variance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "03_mean_waveform", formats)
    plt.close()


def plot_waveform_per_odor(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 4: Per-condition mean waveform with variance."""
    labels = get_region_labels(signals)
    src, tgt, cond_label = labels["source"], labels["target"], labels["condition"]
    condition_names = np.unique(signals["condition_names"])
    n_conditions = len(condition_names)

    T = signals["target_real"].shape[-1]
    time = np.arange(T) / fs + 0.5

    fig, axes = plt.subplots(n_conditions, 2, figsize=(14, 2.5 * n_conditions))

    for i, cond in enumerate(condition_names):
        mask = np.array(signals["condition_names"]) == cond

        # Forward: source → target
        real_mean = signals["target_real"][mask].mean(axis=(0, 1))
        real_std = signals["target_real"][mask].mean(axis=1).std(axis=0)
        gen_mean = signals["target_gen"][mask].mean(axis=(0, 1))
        gen_std = signals["target_gen"][mask].mean(axis=1).std(axis=0)

        ax = axes[i, 0]
        ax.plot(time, real_mean, color=COLORS["real"], label="Real", linewidth=1)
        ax.fill_between(time, real_mean - real_std, real_mean + real_std, alpha=0.2, color=COLORS["real"])
        ax.plot(time, gen_mean, color=COLORS["generated"], label="Generated", linewidth=1)
        ax.fill_between(time, gen_mean - gen_std, gen_mean + gen_std, alpha=0.2, color=COLORS["generated"])
        ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.3)
        ax.set_ylabel(f"{cond}")
        if i == 0:
            ax.set_title(f"Forward: {src}→{tgt}")
            ax.legend(loc="upper right")
        if i == n_conditions - 1:
            ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

        # Reverse: target → source
        real_mean = signals["source_real"][mask].mean(axis=(0, 1))
        real_std = signals["source_real"][mask].mean(axis=1).std(axis=0)
        gen_mean = signals["source_gen"][mask].mean(axis=(0, 1))
        gen_std = signals["source_gen"][mask].mean(axis=1).std(axis=0)

        ax = axes[i, 1]
        ax.plot(time, real_mean, color=COLORS["real"], label="Real", linewidth=1)
        ax.fill_between(time, real_mean - real_std, real_mean + real_std, alpha=0.2, color=COLORS["real"])
        ax.plot(time, gen_mean, color=COLORS["generated"], label="Generated", linewidth=1)
        ax.fill_between(time, gen_mean - gen_std, gen_mean + gen_std, alpha=0.2, color=COLORS["generated"])
        ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.3)
        if i == 0:
            ax.set_title(f"Reverse: {tgt}→{src}")
            ax.legend(loc="upper right")
        if i == n_conditions - 1:
            ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Per-{cond_label} Mean Waveform with Variance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "04_waveform_per_odor", formats)
    plt.close()


def plot_psd_distribution(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 5: PSD distribution in dB (violin/box plot)."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Forward: source → target
    freqs, psd_real = compute_psd(signals["target_real"], fs)
    _, psd_gen = compute_psd(signals["target_gen"], fs)

    # Compute total power per trial (mean across channels, sum across frequencies)
    power_real = 10 * np.log10(psd_real.mean(axis=1).mean(axis=-1) + 1e-10)
    power_gen = 10 * np.log10(psd_gen.mean(axis=1).mean(axis=-1) + 1e-10)

    ax = axes[0]
    data = [power_real, power_gen]
    parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor(COLORS["real"])
    parts['bodies'][1].set_facecolor(COLORS["generated"])
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Real {tgt}", f"Generated {tgt}"])
    ax.set_ylabel("Total Power (dB)")
    ax.set_title(labels["forward_title"])
    ax.grid(True, alpha=0.3)

    # Reverse: target → source
    freqs, psd_real = compute_psd(signals["source_real"], fs)
    _, psd_gen = compute_psd(signals["source_gen"], fs)

    power_real = 10 * np.log10(psd_real.mean(axis=1).mean(axis=-1) + 1e-10)
    power_gen = 10 * np.log10(psd_gen.mean(axis=1).mean(axis=-1) + 1e-10)

    ax = axes[1]
    data = [power_real, power_gen]
    parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor(COLORS["real"])
    parts['bodies'][1].set_facecolor(COLORS["generated"])
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Real {src}", f"Generated {src}"])
    ax.set_ylabel("Total Power (dB)")
    ax.set_title(labels["reverse_title"])
    ax.grid(True, alpha=0.3)

    plt.suptitle("PSD Distribution in dB", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "05_psd_distribution", formats)
    plt.close()


def plot_correlation_histogram(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
):
    """Plot 6: Correlation distribution histogram."""
    labels = get_region_labels(signals)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Forward: source → target
    corr_fwd = compute_correlation(signals["target_gen"], signals["target_real"])

    ax = axes[0]
    ax.hist(corr_fwd, bins=50, color=COLORS["forward"], alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(corr_fwd), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(corr_fwd):.3f}")
    ax.axvline(np.median(corr_fwd), color="orange", linestyle=":", linewidth=2, label=f"Median: {np.median(corr_fwd):.3f}")
    ax.set_xlabel("Pearson Correlation")
    ax.set_ylabel("Count")
    ax.set_title(labels["forward_title"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reverse: target → source
    corr_rev = compute_correlation(signals["source_gen"], signals["source_real"])

    ax = axes[1]
    ax.hist(corr_rev, bins=50, color=COLORS["reverse"], alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(corr_rev), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(corr_rev):.3f}")
    ax.axvline(np.median(corr_rev), color="orange", linestyle=":", linewidth=2, label=f"Median: {np.median(corr_rev):.3f}")
    ax.set_xlabel("Pearson Correlation")
    ax.set_ylabel("Count")
    ax.set_title(labels["reverse_title"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Trial Correlation Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "06_correlation_histogram", formats)
    plt.close()


def plot_spectrograms(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 7: Time-frequency spectrograms."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Use averaged signal for spectrogram (mean across trials, first channel for clarity)
    nperseg = 256
    noverlap = 240

    # Forward direction
    tgt_real_avg = signals["target_real"].mean(axis=0)[0]  # First channel
    tgt_gen_avg = signals["target_gen"].mean(axis=0)[0]

    f, t, Sxx_real = scipy_signal.spectrogram(tgt_real_avg, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Sxx_gen = scipy_signal.spectrogram(tgt_gen_avg, fs=fs, nperseg=nperseg, noverlap=noverlap)

    t = t + 0.5  # Adjust time offset

    # Limit to 100 Hz
    freq_mask = f <= 100
    f = f[freq_mask]
    Sxx_real = Sxx_real[freq_mask]
    Sxx_gen = Sxx_gen[freq_mask]

    vmin = min(10 * np.log10(Sxx_real + 1e-10).min(), 10 * np.log10(Sxx_gen + 1e-10).min())
    vmax = max(10 * np.log10(Sxx_real + 1e-10).max(), 10 * np.log10(Sxx_gen + 1e-10).max())

    ax = axes[0, 0]
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx_real + 1e-10), shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.axvline(x=2.0, color="white", linestyle="--", alpha=0.7)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Real {tgt} (Forward)")
    plt.colorbar(im, ax=ax, label="Power (dB)")

    ax = axes[0, 1]
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx_gen + 1e-10), shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.axvline(x=2.0, color="white", linestyle="--", alpha=0.7)
    ax.set_title(f"Generated {tgt} (Forward)")
    plt.colorbar(im, ax=ax, label="Power (dB)")

    # Reverse direction
    src_real_avg = signals["source_real"].mean(axis=0)[0]
    src_gen_avg = signals["source_gen"].mean(axis=0)[0]

    f, t, Sxx_real = scipy_signal.spectrogram(src_real_avg, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Sxx_gen = scipy_signal.spectrogram(src_gen_avg, fs=fs, nperseg=nperseg, noverlap=noverlap)

    t = t + 0.5
    freq_mask = f <= 100
    f = f[freq_mask]
    Sxx_real = Sxx_real[freq_mask]
    Sxx_gen = Sxx_gen[freq_mask]

    vmin = min(10 * np.log10(Sxx_real + 1e-10).min(), 10 * np.log10(Sxx_gen + 1e-10).min())
    vmax = max(10 * np.log10(Sxx_real + 1e-10).max(), 10 * np.log10(Sxx_gen + 1e-10).max())

    ax = axes[1, 0]
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx_real + 1e-10), shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.axvline(x=2.0, color="white", linestyle="--", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Real {src} (Reverse)")
    plt.colorbar(im, ax=ax, label="Power (dB)")

    ax = axes[1, 1]
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx_gen + 1e-10), shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.axvline(x=2.0, color="white", linestyle="--", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Generated {src} (Reverse)")
    plt.colorbar(im, ax=ax, label="Power (dB)")

    plt.suptitle("Time-Frequency Spectrograms (Trial-Averaged)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "07_spectrograms", formats)
    plt.close()


def plot_channel_correlation(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
):
    """Plot 8: Cross-channel correlation matrices."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Forward: source → target
    # Compute correlation matrix (correlation between channels across time, averaged over trials)
    corr_real_fwd = compute_channel_correlation_matrix(signals["target_real"])
    corr_gen_fwd = compute_channel_correlation_matrix(signals["target_gen"])

    vmin, vmax = -1, 1

    ax = axes[0, 0]
    im = ax.imshow(corr_real_fwd, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title(f"Real {tgt}")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(corr_gen_fwd, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title(f"Generated {tgt}")
    ax.set_xlabel("Channel")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    diff = corr_real_fwd - corr_gen_fwd
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_title(f"Difference (MAE={np.abs(diff).mean():.3f})")
    ax.set_xlabel("Channel")
    plt.colorbar(im, ax=ax)

    # Reverse: target → source
    corr_real_rev = compute_channel_correlation_matrix(signals["source_real"])
    corr_gen_rev = compute_channel_correlation_matrix(signals["source_gen"])

    ax = axes[1, 0]
    im = ax.imshow(corr_real_rev, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title(f"Real {src}")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(corr_gen_rev, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title(f"Generated {src}")
    ax.set_xlabel("Channel")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    diff = corr_real_rev - corr_gen_rev
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_title(f"Difference (MAE={np.abs(diff).mean():.3f})")
    ax.set_xlabel("Channel")
    plt.colorbar(im, ax=ax)

    plt.suptitle("Cross-Channel Correlation Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "08_channel_correlation", formats)
    plt.close()


def plot_band_power(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 9: Frequency band power comparison."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    band_names = list(FREQ_BANDS.keys())
    n_bands = len(band_names)
    x = np.arange(n_bands)
    width = 0.35

    # Forward: source → target
    freqs, psd_real = compute_psd(signals["target_real"], fs)
    _, psd_gen = compute_psd(signals["target_gen"], fs)

    power_real = []
    power_gen = []
    for band_name, band in FREQ_BANDS.items():
        power_real.append(compute_band_power(psd_real, freqs, band).mean())
        power_gen.append(compute_band_power(psd_gen, freqs, band).mean())

    power_real_db = 10 * np.log10(np.array(power_real) + 1e-10)
    power_gen_db = 10 * np.log10(np.array(power_gen) + 1e-10)

    ax = axes[0]
    ax.bar(x - width/2, power_real_db, width, label=f"Real {tgt}", color=COLORS["real"])
    ax.bar(x + width/2, power_gen_db, width, label=f"Generated {tgt}", color=COLORS["generated"])
    ax.set_ylabel("Power (dB)")
    ax.set_title(labels["forward_title"])
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Reverse: target → source
    freqs, psd_real = compute_psd(signals["source_real"], fs)
    _, psd_gen = compute_psd(signals["source_gen"], fs)

    power_real = []
    power_gen = []
    for band_name, band in FREQ_BANDS.items():
        power_real.append(compute_band_power(psd_real, freqs, band).mean())
        power_gen.append(compute_band_power(psd_gen, freqs, band).mean())

    power_real_db = 10 * np.log10(np.array(power_real) + 1e-10)
    power_gen_db = 10 * np.log10(np.array(power_gen) + 1e-10)

    ax = axes[1]
    ax.bar(x - width/2, power_real_db, width, label=f"Real {src}", color=COLORS["real"])
    ax.bar(x + width/2, power_gen_db, width, label=f"Generated {src}", color=COLORS["generated"])
    ax.set_ylabel("Power (dB)")
    ax.set_title(labels["reverse_title"])
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Frequency Band Power Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "09_band_power", formats)
    plt.close()


def plot_phase_coherence(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 10: Phase coherence (PLV) across frequency bands."""
    labels = get_region_labels(signals)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    band_names = list(FREQ_BANDS.keys())
    n_bands = len(band_names)
    x = np.arange(n_bands)
    width = 0.35

    # Sample a subset of trials for computational efficiency
    n_samples = min(50, signals["target_real"].shape[0])
    indices = np.random.choice(signals["target_real"].shape[0], n_samples, replace=False)

    # Forward: source → target
    plv_fwd = []
    for band_name, (low, high) in FREQ_BANDS.items():
        plv_vals = []
        for idx in indices:
            # Filter signal to band (use first channel)
            real_sig = signals["target_real"][idx, 0]
            gen_sig = signals["target_gen"][idx, 0]

            # Bandpass filter
            sos = scipy_signal.butter(4, [low, high], btype="band", fs=fs, output="sos")
            real_filtered = scipy_signal.sosfilt(sos, real_sig)
            gen_filtered = scipy_signal.sosfilt(sos, gen_sig)

            plv_vals.append(compute_plv(real_filtered, gen_filtered, fs))
        plv_fwd.append(np.mean(plv_vals))

    # Reverse: target → source
    plv_rev = []
    for band_name, (low, high) in FREQ_BANDS.items():
        plv_vals = []
        for idx in indices:
            real_sig = signals["source_real"][idx, 0]
            gen_sig = signals["source_gen"][idx, 0]

            sos = scipy_signal.butter(4, [low, high], btype="band", fs=fs, output="sos")
            real_filtered = scipy_signal.sosfilt(sos, real_sig)
            gen_filtered = scipy_signal.sosfilt(sos, gen_sig)

            plv_vals.append(compute_plv(real_filtered, gen_filtered, fs))
        plv_rev.append(np.mean(plv_vals))

    ax = axes[0]
    ax.bar(x, plv_fwd, color=COLORS["forward"], edgecolor="black")
    ax.set_ylabel("Phase Locking Value (PLV)")
    ax.set_title(labels["forward_title"])
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x, plv_rev, color=COLORS["reverse"], edgecolor="black")
    ax.set_ylabel("Phase Locking Value (PLV)")
    ax.set_title(labels["reverse_title"])
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Phase Coherence (PLV) Between Real and Generated", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "10_phase_coherence", formats)
    plt.close()


def plot_error_spectrum(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 11: Error vs frequency spectrum."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Forward: source → target
    freqs, psd_real_fwd = compute_psd(signals["target_real"], fs)
    _, psd_gen_fwd = compute_psd(signals["target_gen"], fs)

    # Reverse: target → source
    _, psd_real_rev = compute_psd(signals["source_real"], fs)
    _, psd_gen_rev = compute_psd(signals["source_gen"], fs)

    # Compute error in dB
    error_fwd = np.abs(
        10 * np.log10(psd_gen_fwd.mean(axis=(0, 1)) + 1e-10) -
        10 * np.log10(psd_real_fwd.mean(axis=(0, 1)) + 1e-10)
    )
    error_rev = np.abs(
        10 * np.log10(psd_gen_rev.mean(axis=(0, 1)) + 1e-10) -
        10 * np.log10(psd_real_rev.mean(axis=(0, 1)) + 1e-10)
    )

    ax.plot(freqs, error_fwd, color=COLORS["forward"], label=f"Forward ({src}→{tgt})", linewidth=2)
    ax.plot(freqs, error_rev, color=COLORS["reverse"], label=f"Reverse ({tgt}→{src})", linewidth=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Absolute PSD Error (dB)")
    ax.set_title("PSD Error vs Frequency")
    ax.set_xlim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add band annotations
    for band_name, (low, high) in FREQ_BANDS.items():
        ax.axvspan(low, high, alpha=0.1, color="gray")
        ax.text((low + high) / 2, ax.get_ylim()[1] * 0.95, band_name,
                ha="center", va="top", fontsize=8, rotation=90)

    plt.tight_layout()
    save_figure(fig, output_dir, "11_error_spectrum", formats)
    plt.close()


def plot_temporal_correlation(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
    window_ms: int = 250,
):
    """Plot 12: Sliding window temporal correlation."""
    labels = get_region_labels(signals)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window_samples = int(window_ms * fs / 1000)
    T = signals["target_real"].shape[-1]
    n_windows = T // window_samples

    time_centers = np.array([(i + 0.5) * window_ms / 1000 + 0.5 for i in range(n_windows)])

    # Forward: source → target
    corr_windows_fwd = []
    for i in range(n_windows):
        start = i * window_samples
        end = (i + 1) * window_samples

        corrs = []
        for trial in range(signals["target_real"].shape[0]):
            real_win = signals["target_real"][trial, :, start:end].flatten()
            gen_win = signals["target_gen"][trial, :, start:end].flatten()
            corrs.append(np.corrcoef(real_win, gen_win)[0, 1])
        corr_windows_fwd.append(np.mean(corrs))

    ax = axes[0]
    ax.plot(time_centers, corr_windows_fwd, color=COLORS["forward"], linewidth=2, marker="o")
    ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5, label="Stimulus onset")
    ax.axhline(y=np.mean(corr_windows_fwd), color="red", linestyle=":", alpha=0.5, label=f"Mean: {np.mean(corr_windows_fwd):.3f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Correlation")
    ax.set_title(labels["forward_title"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reverse: target → source
    corr_windows_rev = []
    for i in range(n_windows):
        start = i * window_samples
        end = (i + 1) * window_samples

        corrs = []
        for trial in range(signals["source_real"].shape[0]):
            real_win = signals["source_real"][trial, :, start:end].flatten()
            gen_win = signals["source_gen"][trial, :, start:end].flatten()
            corrs.append(np.corrcoef(real_win, gen_win)[0, 1])
        corr_windows_rev.append(np.mean(corrs))

    ax = axes[1]
    ax.plot(time_centers, corr_windows_rev, color=COLORS["reverse"], linewidth=2, marker="o")
    ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5, label="Stimulus onset")
    ax.axhline(y=np.mean(corr_windows_rev), color="red", linestyle=":", alpha=0.5, label=f"Mean: {np.mean(corr_windows_rev):.3f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Correlation")
    ax.set_title(labels["reverse_title"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Sliding Window Correlation ({window_ms}ms windows)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "12_temporal_correlation", formats)
    plt.close()


def plot_amplitude_qq(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
):
    """Plot 13: Amplitude distribution QQ plots."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Forward: source → target
    real_flat = signals["target_real"].flatten()
    gen_flat = signals["target_gen"].flatten()

    # Sample for efficiency
    n_samples = min(100000, len(real_flat))
    idx = np.random.choice(len(real_flat), n_samples, replace=False)

    real_sorted = np.sort(real_flat[idx])
    gen_sorted = np.sort(gen_flat[idx])

    ax = axes[0]
    ax.scatter(real_sorted, gen_sorted, alpha=0.1, s=1, color=COLORS["forward"])
    lim = [min(real_sorted.min(), gen_sorted.min()), max(real_sorted.max(), gen_sorted.max())]
    ax.plot(lim, lim, "r--", linewidth=2, label="Perfect match")
    ax.set_xlabel(f"Real {tgt} Quantiles")
    ax.set_ylabel(f"Generated {tgt} Quantiles")
    ax.set_title(labels["forward_title"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reverse: target → source
    real_flat = signals["source_real"].flatten()
    gen_flat = signals["source_gen"].flatten()

    idx = np.random.choice(len(real_flat), n_samples, replace=False)

    real_sorted = np.sort(real_flat[idx])
    gen_sorted = np.sort(gen_flat[idx])

    ax = axes[1]
    ax.scatter(real_sorted, gen_sorted, alpha=0.1, s=1, color=COLORS["reverse"])
    lim = [min(real_sorted.min(), gen_sorted.min()), max(real_sorted.max(), gen_sorted.max())]
    ax.plot(lim, lim, "r--", linewidth=2, label="Perfect match")
    ax.set_xlabel(f"Real {src} Quantiles")
    ax.set_ylabel(f"Generated {src} Quantiles")
    ax.set_title(labels["reverse_title"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Amplitude Distribution QQ Plots", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "13_amplitude_qq", formats)
    plt.close()


def plot_channel_performance(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
):
    """Plot 14: Per-channel performance heatmap."""
    labels = get_region_labels(signals)
    n_channels_tgt = signals["target_real"].shape[1]
    n_channels_src = signals["source_real"].shape[1]

    # Compute per-channel correlation
    corr_fwd = np.zeros(n_channels_tgt)
    corr_rev = np.zeros(n_channels_src)

    for c in range(n_channels_tgt):
        corr_fwd[c] = np.corrcoef(
            signals["target_gen"][:, c, :].flatten(),
            signals["target_real"][:, c, :].flatten()
        )[0, 1]
    for c in range(n_channels_src):
        corr_rev[c] = np.corrcoef(
            signals["source_gen"][:, c, :].flatten(),
            signals["source_real"][:, c, :].flatten()
        )[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap style bar plot
    ax = axes[0]
    colors = plt.cm.RdYlGn((corr_fwd - corr_fwd.min()) / (corr_fwd.max() - corr_fwd.min() + 1e-6))
    ax.bar(range(n_channels_tgt), corr_fwd, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=np.mean(corr_fwd), color="red", linestyle="--", label=f"Mean: {np.mean(corr_fwd):.3f}")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Correlation")
    ax.set_title(labels["forward_title"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    colors = plt.cm.RdYlGn((corr_rev - corr_rev.min()) / (corr_rev.max() - corr_rev.min() + 1e-6))
    ax.bar(range(n_channels_src), corr_rev, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=np.mean(corr_rev), color="red", linestyle="--", label=f"Mean: {np.mean(corr_rev):.3f}")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Correlation")
    ax.set_title(labels["reverse_title"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Per-Channel Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "14_channel_performance", formats)
    plt.close()


def plot_error_heatmap(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 15: Error heatmap (channels x time)."""
    labels = get_region_labels(signals)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Downsample time for visualization (every 10th point)
    downsample = 10
    T = signals["target_real"].shape[-1]
    time = np.arange(0, T, downsample) / fs + 0.5
    n_channels_tgt = signals["target_real"].shape[1]
    n_channels_src = signals["source_real"].shape[1]

    # Forward: source → target
    error_fwd = np.abs(signals["target_gen"] - signals["target_real"]).mean(axis=0)[:, ::downsample]

    ax = axes[0]
    im = ax.imshow(error_fwd, aspect="auto", cmap="hot",
                   extent=[time[0], time[-1], n_channels_tgt - 0.5, -0.5])
    ax.axvline(x=2.0, color="white", linestyle="--", alpha=0.7)
    ax.set_ylabel("Channel")
    ax.set_title(labels["forward_title"])
    plt.colorbar(im, ax=ax, label="Mean Absolute Error")

    # Reverse: target → source
    error_rev = np.abs(signals["source_gen"] - signals["source_real"]).mean(axis=0)[:, ::downsample]

    ax = axes[1]
    im = ax.imshow(error_rev, aspect="auto", cmap="hot",
                   extent=[time[0], time[-1], n_channels_src - 0.5, -0.5])
    ax.axvline(x=2.0, color="white", linestyle="--", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(labels["reverse_title"])
    plt.colorbar(im, ax=ax, label="Mean Absolute Error")

    plt.suptitle("Error Heatmap (Channels x Time)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "15_error_heatmap", formats)
    plt.close()


# =============================================================================
# NEW: Nature Methods Quality Plots (16-25)
# =============================================================================

def plot_example_trials(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
    n_examples: int = 16,
):
    """Plot 16: Example trials grid showing individual comparisons."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    T = signals["target_real"].shape[-1]
    time = np.arange(T) / fs + 0.5

    # Randomly select trials (use local RNG to avoid global state mutation)
    n_trials = signals["target_real"].shape[0]
    rng = np.random.RandomState(42)
    indices = rng.choice(n_trials, min(n_examples, n_trials), replace=False)

    for i, idx in enumerate(indices):
        ax = axes[i]

        # Use first channel for clarity
        real = signals["target_real"][idx, 0, :]
        gen = signals["target_gen"][idx, 0, :]
        corr = np.corrcoef(real, gen)[0, 1]

        ax.plot(time, real, color=COLORS["real"], alpha=0.7, linewidth=1, label="Real")
        ax.plot(time, gen, color=COLORS["generated"], alpha=0.7, linewidth=1, label="Gen")
        ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.3)
        ax.set_title(f"Trial {idx} (r={corr:.3f})", fontsize=9)

        if i >= 12:
            ax.set_xlabel("Time (s)", fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(labelsize=7)

        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    plt.suptitle(f"Example Trials: Real vs Generated (Forward {src}→{tgt})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "16_example_trials", formats)
    plt.close()


def plot_cycle_consistency(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 17: Cycle consistency analysis."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # For cycle consistency, we need the models to compute round-trips
    # Since we have generated signals, we can compute cycle correlation
    # by comparing original with what would be reconstructed

    # Forward cycle: source → target_gen → source_reconstructed
    # We approximate this by correlating source_real with source_gen (reverse of target_gen)
    # This is a simplification - ideally we'd run the full cycle

    # Compute per-trial cycle correlations
    n_trials = signals["source_real"].shape[0]

    # Forward cycle approximation: correlate original source with reverse-generated source
    cycle_corr_fwd = np.array([
        np.corrcoef(signals["source_real"][i].flatten(), signals["source_gen"][i].flatten())[0, 1]
        for i in range(n_trials)
    ])

    # Reverse cycle approximation
    cycle_corr_rev = np.array([
        np.corrcoef(signals["target_real"][i].flatten(), signals["target_gen"][i].flatten())[0, 1]
        for i in range(n_trials)
    ])

    # Plot 1: Cycle correlation histograms
    ax = axes[0, 0]
    ax.hist(cycle_corr_fwd, bins=30, color=COLORS["forward"], alpha=0.7, label=f"{src} real vs {src} gen")
    ax.axvline(np.mean(cycle_corr_fwd), color="red", linestyle="--", label=f"Mean: {np.mean(cycle_corr_fwd):.3f}")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Count")
    ax.set_title(f"{src}: Real vs Generated")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(cycle_corr_rev, bins=30, color=COLORS["reverse"], alpha=0.7, label=f"{tgt} real vs {tgt} gen")
    ax.axvline(np.mean(cycle_corr_rev), color="red", linestyle="--", label=f"Mean: {np.mean(cycle_corr_rev):.3f}")
    ax.set_xlabel("Correlation")
    ax.set_title(f"{tgt}: Real vs Generated")
    ax.legend(fontsize=8)

    # Plot 2: Scatter of correlations
    ax = axes[0, 2]
    ax.scatter(cycle_corr_fwd, cycle_corr_rev, alpha=0.5, s=20)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5)
    ax.set_xlabel(f"{src} Correlation")
    ax.set_ylabel(f"{tgt} Correlation")
    ax.set_title("Bidirectional Consistency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Plot 3: Example waveforms showing reconstruction
    T = signals["source_real"].shape[-1]
    time = np.arange(T) / fs + 0.5

    # Best reconstruction example
    best_idx = np.argmax(cycle_corr_fwd)
    ax = axes[1, 0]
    ax.plot(time, signals["source_real"][best_idx, 0], color=COLORS["real"], label=f"{src} Real", alpha=0.8)
    ax.plot(time, signals["source_gen"][best_idx, 0], color=COLORS["generated"], label=f"{src} Gen", alpha=0.8)
    ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Best {src} Reconstruction (r={cycle_corr_fwd[best_idx]:.3f})")
    ax.legend(fontsize=8)

    # Median reconstruction
    median_idx = np.argsort(cycle_corr_fwd)[len(cycle_corr_fwd) // 2]
    ax = axes[1, 1]
    ax.plot(time, signals["source_real"][median_idx, 0], color=COLORS["real"], label=f"{src} Real", alpha=0.8)
    ax.plot(time, signals["source_gen"][median_idx, 0], color=COLORS["generated"], label=f"{src} Gen", alpha=0.8)
    ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Median {src} Reconstruction (r={cycle_corr_fwd[median_idx]:.3f})")
    ax.legend(fontsize=8)

    # Summary statistics
    ax = axes[1, 2]
    ax.axis("off")
    stats_text = (
        f"Cycle Consistency Summary\n"
        f"{'='*30}\n\n"
        f"{src} (Real vs Gen from {tgt}):\n"
        f"  Mean r: {np.mean(cycle_corr_fwd):.4f}\n"
        f"  Std: {np.std(cycle_corr_fwd):.4f}\n"
        f"  Median: {np.median(cycle_corr_fwd):.4f}\n\n"
        f"{tgt} (Real vs Gen from {src}):\n"
        f"  Mean r: {np.mean(cycle_corr_rev):.4f}\n"
        f"  Std: {np.std(cycle_corr_rev):.4f}\n"
        f"  Median: {np.median(cycle_corr_rev):.4f}"
    )
    ax.text(0.1, 0.5, stats_text, fontsize=11, family="monospace",
            verticalalignment="center", transform=ax.transAxes)

    plt.suptitle("Cycle Consistency Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "17_cycle_consistency", formats)
    plt.close()


def plot_statistical_summary(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
):
    """Plot 18: Statistical summary with effect sizes and p-values."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Compute correlations
    corr_fwd = compute_correlation(signals["target_gen"], signals["target_real"])
    corr_rev = compute_correlation(signals["source_gen"], signals["source_real"])

    # Compute MSE per trial
    mse_fwd = np.mean((signals["target_gen"] - signals["target_real"]) ** 2, axis=(1, 2))
    mse_rev = np.mean((signals["source_gen"] - signals["source_real"]) ** 2, axis=(1, 2))

    # Plot 1: Forest plot of effect sizes
    ax = axes[0]
    metrics = ["Correlation", "1 - MSE"]
    fwd_means = [np.mean(corr_fwd), 1 - np.mean(mse_fwd)]
    rev_means = [np.mean(corr_rev), 1 - np.mean(mse_rev)]

    # Bootstrap CIs
    corr_fwd_ci = bootstrap_ci(corr_fwd)
    corr_rev_ci = bootstrap_ci(corr_rev)
    mse_fwd_ci = bootstrap_ci(1 - mse_fwd)
    mse_rev_ci = bootstrap_ci(1 - mse_rev)

    y_pos = np.array([0, 1])
    width = 0.35

    # Forward
    ax.barh(y_pos - width/2, fwd_means, width, color=COLORS["forward"], alpha=0.7, label=f"Forward ({src}→{tgt})")
    ax.errorbar(fwd_means, y_pos - width/2,
                xerr=[[fwd_means[0] - corr_fwd_ci[0], fwd_means[1] - mse_fwd_ci[0]],
                      [corr_fwd_ci[1] - fwd_means[0], mse_fwd_ci[1] - fwd_means[1]]],
                fmt="none", color="black", capsize=3)

    # Reverse
    ax.barh(y_pos + width/2, rev_means, width, color=COLORS["reverse"], alpha=0.7, label=f"Reverse ({tgt}→{src})")
    ax.errorbar(rev_means, y_pos + width/2,
                xerr=[[rev_means[0] - corr_rev_ci[0], rev_means[1] - mse_rev_ci[0]],
                      [corr_rev_ci[1] - rev_means[0], mse_rev_ci[1] - rev_means[1]]],
                fmt="none", color="black", capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Value (with 95% CI)")
    ax.set_title("Performance Metrics")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis="x")

    # Plot 2: Effect size comparison (Cohen's d)
    ax = axes[1]

    # Compare forward vs reverse
    d_corr = compute_effect_size_ci(corr_fwd, corr_rev)
    d_mse = compute_effect_size_ci(mse_fwd, mse_rev)

    effect_sizes = [d_corr["d"], d_mse["d"]]
    ci_lower = [d_corr["ci_lower"], d_mse["ci_lower"]]
    ci_upper = [d_corr["ci_upper"], d_mse["ci_upper"]]

    colors = ["green" if es > 0 else "red" for es in effect_sizes]
    ax.barh(y_pos, effect_sizes, color=colors, alpha=0.7)
    ax.errorbar(effect_sizes, y_pos,
                xerr=[[es - cl for es, cl in zip(effect_sizes, ci_lower)],
                      [cu - es for es, cu in zip(effect_sizes, ci_upper)]],
                fmt="none", color="black", capsize=3)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(x=0.2, color="gray", linestyle="--", alpha=0.5, label="Small (0.2)")
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5, label="Medium (0.5)")
    ax.axvline(x=-0.2, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=-0.5, color="gray", linestyle=":", alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(["Correlation", "MSE"])
    ax.set_xlabel("Cohen's d (Forward - Reverse)")
    ax.set_title("Effect Size: Forward vs Reverse")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

    # Plot 3: Statistical test results
    ax = axes[2]
    ax.axis("off")

    # Paired t-tests
    t_corr, p_corr = stats.ttest_rel(corr_fwd, corr_rev)
    t_mse, p_mse = stats.ttest_rel(mse_fwd, mse_rev)

    stats_text = (
        f"Statistical Tests\n"
        f"{'='*40}\n\n"
        f"Paired t-test (Forward vs Reverse):\n\n"
        f"Correlation:\n"
        f"  t = {t_corr:.3f}, p = {p_corr:.2e}\n"
        f"  Cohen's d = {d_corr['d']:.3f} [{d_corr['ci_lower']:.3f}, {d_corr['ci_upper']:.3f}]\n\n"
        f"MSE:\n"
        f"  t = {t_mse:.3f}, p = {p_mse:.2e}\n"
        f"  Cohen's d = {d_mse['d']:.3f} [{d_mse['ci_lower']:.3f}, {d_mse['ci_upper']:.3f}]\n\n"
        f"{'='*40}\n"
        f"n = {len(corr_fwd)} samples\n"
        f"Bootstrap: 1000 resamples, 95% CI"
    )
    ax.text(0.1, 0.5, stats_text, fontsize=10, family="monospace",
            verticalalignment="center", transform=ax.transAxes)

    plt.suptitle("Statistical Summary with Effect Sizes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "18_statistical_summary", formats)
    plt.close()


def plot_bland_altman(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
):
    """Plot 19: Bland-Altman agreement plots."""
    labels = get_region_labels(signals)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sample points for visualization (use local RNG)
    n_samples = min(10000, signals["target_real"].size)
    rng = np.random.RandomState(42)
    idx = rng.choice(signals["target_real"].size, n_samples, replace=False)

    # Forward: target
    real_flat = signals["target_real"].flatten()[idx]
    gen_flat = signals["target_gen"].flatten()[idx]

    ba_stats_fwd = bland_altman_stats(gen_flat, real_flat)

    ax = axes[0]
    ax.scatter(ba_stats_fwd["mean_vals"], ba_stats_fwd["diff"], alpha=0.1, s=1, color=COLORS["forward"])
    ax.axhline(y=ba_stats_fwd["bias"], color="red", linestyle="-", linewidth=2, label=f"Bias: {ba_stats_fwd['bias']:.4f}")
    ax.axhline(y=ba_stats_fwd["upper_loa"], color="gray", linestyle="--", linewidth=1.5, label=f"+1.96 SD: {ba_stats_fwd['upper_loa']:.4f}")
    ax.axhline(y=ba_stats_fwd["lower_loa"], color="gray", linestyle="--", linewidth=1.5, label=f"-1.96 SD: {ba_stats_fwd['lower_loa']:.4f}")
    ax.set_xlabel("Mean of Real and Generated")
    ax.set_ylabel("Difference (Generated - Real)")
    ax.set_title(labels["forward_title"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Reverse: source
    real_flat = signals["source_real"].flatten()[idx]
    gen_flat = signals["source_gen"].flatten()[idx]

    ba_stats_rev = bland_altman_stats(gen_flat, real_flat)

    ax = axes[1]
    ax.scatter(ba_stats_rev["mean_vals"], ba_stats_rev["diff"], alpha=0.1, s=1, color=COLORS["reverse"])
    ax.axhline(y=ba_stats_rev["bias"], color="red", linestyle="-", linewidth=2, label=f"Bias: {ba_stats_rev['bias']:.4f}")
    ax.axhline(y=ba_stats_rev["upper_loa"], color="gray", linestyle="--", linewidth=1.5, label=f"+1.96 SD: {ba_stats_rev['upper_loa']:.4f}")
    ax.axhline(y=ba_stats_rev["lower_loa"], color="gray", linestyle="--", linewidth=1.5, label=f"-1.96 SD: {ba_stats_rev['lower_loa']:.4f}")
    ax.set_xlabel("Mean of Real and Generated")
    ax.set_ylabel("Difference (Generated - Real)")
    ax.set_title(labels["reverse_title"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Bland-Altman Agreement Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "19_bland_altman", formats)
    plt.close()

    return {"forward": ba_stats_fwd, "reverse": ba_stats_rev}


def plot_coherence_spectrum(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 20: Magnitude-squared coherence spectrum."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    nperseg = 512

    # Forward: target real vs generated
    # Average coherence across all trials and channels
    n_trials = signals["target_real"].shape[0]
    n_channels = signals["target_real"].shape[1]

    # Compute coherence for a subset of trials (computational efficiency)
    n_sample = min(50, n_trials)
    sample_idx = np.random.choice(n_trials, n_sample, replace=False)

    coherence_fwd_list = []
    for idx in sample_idx:
        for c in range(n_channels):
            f, Cxy = scipy_signal.coherence(
                signals["target_real"][idx, c],
                signals["target_gen"][idx, c],
                fs=fs, nperseg=nperseg
            )
            coherence_fwd_list.append(Cxy)

    coherence_fwd = np.array(coherence_fwd_list)
    coherence_fwd_mean = coherence_fwd.mean(axis=0)
    coherence_fwd_std = coherence_fwd.std(axis=0)

    # Limit to 100 Hz
    freq_mask = f <= 100
    f = f[freq_mask]
    coherence_fwd_mean = coherence_fwd_mean[freq_mask]
    coherence_fwd_std = coherence_fwd_std[freq_mask]

    ax = axes[0]
    ax.plot(f, coherence_fwd_mean, color=COLORS["forward"], linewidth=2)
    ax.fill_between(f, coherence_fwd_mean - coherence_fwd_std, coherence_fwd_mean + coherence_fwd_std,
                    alpha=0.2, color=COLORS["forward"])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.set_title(labels["forward_title"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reverse: source real vs generated
    coherence_rev_list = []
    for idx in sample_idx:
        for c in range(n_channels):
            f, Cxy = scipy_signal.coherence(
                signals["source_real"][idx, c],
                signals["source_gen"][idx, c],
                fs=fs, nperseg=nperseg
            )
            coherence_rev_list.append(Cxy)

    coherence_rev = np.array(coherence_rev_list)
    coherence_rev_mean = coherence_rev.mean(axis=0)
    coherence_rev_std = coherence_rev.std(axis=0)

    freq_mask = f <= 100
    f = f[freq_mask]
    coherence_rev_mean = coherence_rev_mean[freq_mask]
    coherence_rev_std = coherence_rev_std[freq_mask]

    ax = axes[1]
    ax.plot(f, coherence_rev_mean, color=COLORS["reverse"], linewidth=2)
    ax.fill_between(f, coherence_rev_mean - coherence_rev_std, coherence_rev_mean + coherence_rev_std,
                    alpha=0.2, color=COLORS["reverse"])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.set_title(labels["reverse_title"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Magnitude-Squared Coherence Spectrum", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "20_coherence_spectrum", formats)
    plt.close()


def plot_residual_analysis(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
):
    """Plot 21: Residual analysis (normality, autocorrelation)."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Compute residuals
    residuals_fwd = (signals["target_gen"] - signals["target_real"]).flatten()
    residuals_rev = (signals["source_gen"] - signals["source_real"]).flatten()

    # Sample for visualization (use local RNG)
    n_samples = min(100000, len(residuals_fwd))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(residuals_fwd), n_samples, replace=False)

    # Forward direction
    # Histogram
    ax = axes[0, 0]
    ax.hist(residuals_fwd[idx], bins=100, density=True, color=COLORS["forward"], alpha=0.7)
    # Fit normal
    mu, std = np.mean(residuals_fwd), np.std(residuals_fwd)
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), "r-", linewidth=2, label=f"N({mu:.3f}, {std:.3f})")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Forward: Residual Distribution")
    ax.legend()

    # Q-Q plot
    ax = axes[0, 1]
    stats.probplot(residuals_fwd[idx], dist="norm", plot=ax)
    ax.set_title("Forward: Q-Q Plot")

    # Autocorrelation (sample one trial)
    ax = axes[0, 2]
    trial_residuals = signals["target_gen"][0, 0, :] - signals["target_real"][0, 0, :]
    max_lag = min(100, len(trial_residuals) // 4)
    autocorr = [np.corrcoef(trial_residuals[:-lag], trial_residuals[lag:])[0, 1]
                for lag in range(1, max_lag)]
    ax.plot(range(1, max_lag), autocorr, color=COLORS["forward"], linewidth=1.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(y=1.96/np.sqrt(len(trial_residuals)), color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=-1.96/np.sqrt(len(trial_residuals)), color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Forward: Residual Autocorrelation")

    # Reverse direction
    # Histogram
    ax = axes[1, 0]
    ax.hist(residuals_rev[idx], bins=100, density=True, color=COLORS["reverse"], alpha=0.7)
    mu, std = np.mean(residuals_rev), np.std(residuals_rev)
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), "r-", linewidth=2, label=f"N({mu:.3f}, {std:.3f})")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Reverse: Residual Distribution")
    ax.legend()

    # Q-Q plot
    ax = axes[1, 1]
    stats.probplot(residuals_rev[idx], dist="norm", plot=ax)
    ax.set_title("Reverse: Q-Q Plot")

    # Autocorrelation
    ax = axes[1, 2]
    trial_residuals = signals["source_gen"][0, 0, :] - signals["source_real"][0, 0, :]
    autocorr = [np.corrcoef(trial_residuals[:-lag], trial_residuals[lag:])[0, 1]
                for lag in range(1, max_lag)]
    ax.plot(range(1, max_lag), autocorr, color=COLORS["reverse"], linewidth=1.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(y=1.96/np.sqrt(len(trial_residuals)), color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=-1.96/np.sqrt(len(trial_residuals)), color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Reverse: Residual Autocorrelation")

    plt.suptitle("Residual Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "21_residual_analysis", formats)
    plt.close()


def plot_odor_metrics_heatmap(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 22: Per-odor performance heatmap."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    odor_names = np.unique(signals["odor_names"])
    n_odors = len(odor_names)

    # Metrics to compute
    metric_names = ["Correlation", "PSD Error (dB)", "MSE", "PLV"]
    n_metrics = len(metric_names)

    # Initialize arrays for forward and reverse
    metrics_fwd = np.zeros((n_odors, n_metrics))
    metrics_rev = np.zeros((n_odors, n_metrics))

    for i, odor in enumerate(odor_names):
        mask = np.array(signals["odor_names"]) == odor

        # Forward metrics
        corr = compute_correlation(signals["target_gen"][mask], signals["target_real"][mask])
        metrics_fwd[i, 0] = np.mean(corr)

        freqs, psd_real = compute_psd(signals["target_real"][mask], fs)
        _, psd_gen = compute_psd(signals["target_gen"][mask], fs)
        psd_err = np.abs(
            10 * np.log10(psd_gen.mean(axis=(0, 1)) + 1e-10) -
            10 * np.log10(psd_real.mean(axis=(0, 1)) + 1e-10)
        ).mean()
        metrics_fwd[i, 1] = psd_err

        mse = np.mean((signals["target_gen"][mask] - signals["target_real"][mask]) ** 2)
        metrics_fwd[i, 2] = mse

        # Simplified PLV (first channel, first trial)
        if mask.sum() > 0:
            real_sig = signals["target_real"][mask][0, 0]
            gen_sig = signals["target_gen"][mask][0, 0]
            plv = compute_plv(real_sig, gen_sig, fs)
            metrics_fwd[i, 3] = plv

        # Reverse metrics
        corr = compute_correlation(signals["source_gen"][mask], signals["source_real"][mask])
        metrics_rev[i, 0] = np.mean(corr)

        freqs, psd_real = compute_psd(signals["source_real"][mask], fs)
        _, psd_gen = compute_psd(signals["source_gen"][mask], fs)
        psd_err = np.abs(
            10 * np.log10(psd_gen.mean(axis=(0, 1)) + 1e-10) -
            10 * np.log10(psd_real.mean(axis=(0, 1)) + 1e-10)
        ).mean()
        metrics_rev[i, 1] = psd_err

        mse = np.mean((signals["source_gen"][mask] - signals["source_real"][mask]) ** 2)
        metrics_rev[i, 2] = mse

        if mask.sum() > 0:
            real_sig = signals["source_real"][mask][0, 0]
            gen_sig = signals["source_gen"][mask][0, 0]
            plv = compute_plv(real_sig, gen_sig, fs)
            metrics_rev[i, 3] = plv

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Forward heatmap
    ax = axes[0]
    # Normalize each column for visualization
    metrics_fwd_norm = metrics_fwd.copy()
    for j in range(n_metrics):
        col = metrics_fwd[:, j]
        if j == 1 or j == 2:  # PSD error and MSE - lower is better, invert
            metrics_fwd_norm[:, j] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-10)
        else:  # Correlation and PLV - higher is better
            metrics_fwd_norm[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-10)

    im = ax.imshow(metrics_fwd_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.set_yticks(range(n_odors))
    ax.set_yticklabels(odor_names)
    ax.set_title(labels["forward_title"])

    # Add text annotations
    for i in range(n_odors):
        for j in range(n_metrics):
            val = metrics_fwd[i, j]
            text = f"{val:.3f}" if j != 1 else f"{val:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if metrics_fwd_norm[i, j] < 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Normalized Performance")

    # Reverse heatmap
    ax = axes[1]
    metrics_rev_norm = metrics_rev.copy()
    for j in range(n_metrics):
        col = metrics_rev[:, j]
        if j == 1 or j == 2:
            metrics_rev_norm[:, j] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-10)
        else:
            metrics_rev_norm[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-10)

    im = ax.imshow(metrics_rev_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.set_yticks(range(n_odors))
    ax.set_yticklabels(odor_names)
    ax.set_title(labels["reverse_title"])

    for i in range(n_odors):
        for j in range(n_metrics):
            val = metrics_rev[i, j]
            text = f"{val:.3f}" if j != 1 else f"{val:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if metrics_rev_norm[i, j] < 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Normalized Performance")

    plt.suptitle("Per-Odor Performance Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "22_odor_metrics_heatmap", formats)
    plt.close()


def plot_bootstrap_ci(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """Plot 23: Bootstrap confidence intervals for all metrics."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Compute correlations
    corr_fwd = compute_correlation(signals["target_gen"], signals["target_real"])
    corr_rev = compute_correlation(signals["source_gen"], signals["source_real"])

    # MSE per trial
    mse_fwd = np.mean((signals["target_gen"] - signals["target_real"]) ** 2, axis=(1, 2))
    mse_rev = np.mean((signals["source_gen"] - signals["source_real"]) ** 2, axis=(1, 2))

    # Bootstrap for each metric (bootstrap_ci has its own seed parameter)
    metrics = {
        "forward": {
            "correlation": {"mean": np.mean(corr_fwd), "ci": bootstrap_ci(corr_fwd, n_bootstrap=n_bootstrap)},
            "mse": {"mean": np.mean(mse_fwd), "ci": bootstrap_ci(mse_fwd, n_bootstrap=n_bootstrap)},
        },
        "reverse": {
            "correlation": {"mean": np.mean(corr_rev), "ci": bootstrap_ci(corr_rev, n_bootstrap=n_bootstrap)},
            "mse": {"mean": np.mean(mse_rev), "ci": bootstrap_ci(mse_rev, n_bootstrap=n_bootstrap)},
        }
    }

    # Plot forward
    ax = axes[0]
    metric_names = ["Correlation", "1 - MSE"]
    means = [metrics["forward"]["correlation"]["mean"], 1 - metrics["forward"]["mse"]["mean"]]
    ci_lower = [metrics["forward"]["correlation"]["ci"][0], 1 - metrics["forward"]["mse"]["ci"][1]]
    ci_upper = [metrics["forward"]["correlation"]["ci"][1], 1 - metrics["forward"]["mse"]["ci"][0]]

    y_pos = np.arange(len(metric_names))
    ax.barh(y_pos, means, color=COLORS["forward"], alpha=0.7)
    ax.errorbar(means, y_pos,
                xerr=[[m - l for m, l in zip(means, ci_lower)],
                      [u - m for m, u in zip(means, ci_upper)]],
                fmt="none", color="black", capsize=5, capthick=2)

    for i, (m, l, u) in enumerate(zip(means, ci_lower, ci_upper)):
        ax.text(m + 0.02, i, f"{m:.4f}\n[{l:.4f}, {u:.4f}]", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_names)
    ax.set_xlabel("Value")
    ax.set_title(f"{labels['forward_title']} (95% CI)")
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="x")

    # Plot reverse
    ax = axes[1]
    means = [metrics["reverse"]["correlation"]["mean"], 1 - metrics["reverse"]["mse"]["mean"]]
    ci_lower = [metrics["reverse"]["correlation"]["ci"][0], 1 - metrics["reverse"]["mse"]["ci"][1]]
    ci_upper = [metrics["reverse"]["correlation"]["ci"][1], 1 - metrics["reverse"]["mse"]["ci"][0]]

    ax.barh(y_pos, means, color=COLORS["reverse"], alpha=0.7)
    ax.errorbar(means, y_pos,
                xerr=[[m - l for m, l in zip(means, ci_lower)],
                      [u - m for m, u in zip(means, ci_upper)]],
                fmt="none", color="black", capsize=5, capthick=2)

    for i, (m, l, u) in enumerate(zip(means, ci_lower, ci_upper)):
        ax.text(m + 0.02, i, f"{m:.4f}\n[{l:.4f}, {u:.4f}]", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_names)
    ax.set_xlabel("Value")
    ax.set_title(f"{labels['reverse_title']} (95% CI)")
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle(f"Bootstrap Confidence Intervals (n={n_bootstrap})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "23_bootstrap_ci", formats)
    plt.close()

    return metrics


def plot_power_ratio(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 24: Power ratio spectrum (PSD_gen / PSD_real)."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Forward: target
    freqs, psd_real = compute_psd(signals["target_real"], fs)
    _, psd_gen = compute_psd(signals["target_gen"], fs)

    # Mean PSD across trials and channels
    psd_real_mean = psd_real.mean(axis=(0, 1))
    psd_gen_mean = psd_gen.mean(axis=(0, 1))

    # Power ratio
    ratio_fwd = psd_gen_mean / (psd_real_mean + 1e-10)
    ratio_fwd_db = 10 * np.log10(ratio_fwd + 1e-10)

    # Limit to 100 Hz
    freq_mask = freqs <= 100
    freqs_plot = freqs[freq_mask]
    ratio_fwd_db_plot = ratio_fwd_db[freq_mask]

    ax = axes[0]
    ax.plot(freqs_plot, ratio_fwd_db_plot, color=COLORS["forward"], linewidth=2)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Perfect ratio (0 dB)")
    ax.fill_between(freqs_plot, -3, 3, alpha=0.1, color="green", label="±3 dB tolerance")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Ratio (dB)")
    ax.set_title(labels["forward_title"])
    ax.set_xlim(0, 100)
    ax.set_ylim(-10, 10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reverse: source
    freqs, psd_real = compute_psd(signals["source_real"], fs)
    _, psd_gen = compute_psd(signals["source_gen"], fs)

    psd_real_mean = psd_real.mean(axis=(0, 1))
    psd_gen_mean = psd_gen.mean(axis=(0, 1))

    ratio_rev = psd_gen_mean / (psd_real_mean + 1e-10)
    ratio_rev_db = 10 * np.log10(ratio_rev + 1e-10)

    freq_mask = freqs <= 100
    freqs_plot = freqs[freq_mask]
    ratio_rev_db_plot = ratio_rev_db[freq_mask]

    ax = axes[1]
    ax.plot(freqs_plot, ratio_rev_db_plot, color=COLORS["reverse"], linewidth=2)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Perfect ratio (0 dB)")
    ax.fill_between(freqs_plot, -3, 3, alpha=0.1, color="green", label="±3 dB tolerance")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Ratio (dB)")
    ax.set_title(labels["reverse_title"])
    ax.set_xlim(0, 100)
    ax.set_ylim(-10, 10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Power Spectral Density Ratio (Generated / Real)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "24_power_ratio", formats)
    plt.close()


def plot_cross_correlation(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 25: Cross-correlation lag analysis."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    max_lag = 100  # samples = 100ms at 1kHz

    # Forward: target
    # Compute cross-correlation for a subset of trials (use local RNG)
    n_trials = min(50, signals["target_real"].shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(signals["target_real"].shape[0], n_trials, replace=False)

    xcorr_fwd_list = []
    for idx in sample_idx:
        # Use first channel
        real = signals["target_real"][idx, 0]
        gen = signals["target_gen"][idx, 0]

        # Normalize
        real = (real - real.mean()) / (real.std() + 1e-10)
        gen = (gen - gen.mean()) / (gen.std() + 1e-10)

        # Cross-correlation at various lags
        xcorr = np.correlate(gen, real, mode="full")
        xcorr = xcorr / len(real)  # Normalize
        mid = len(xcorr) // 2
        xcorr_fwd_list.append(xcorr[mid - max_lag:mid + max_lag + 1])

    xcorr_fwd = np.array(xcorr_fwd_list)
    xcorr_fwd_mean = xcorr_fwd.mean(axis=0)
    xcorr_fwd_std = xcorr_fwd.std(axis=0)

    lags = np.arange(-max_lag, max_lag + 1)
    lag_ms = lags * 1000 / fs

    ax = axes[0]
    ax.plot(lag_ms, xcorr_fwd_mean, color=COLORS["forward"], linewidth=2)
    ax.fill_between(lag_ms, xcorr_fwd_mean - xcorr_fwd_std, xcorr_fwd_mean + xcorr_fwd_std,
                    alpha=0.2, color=COLORS["forward"])
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Zero lag")
    peak_lag = lag_ms[np.argmax(xcorr_fwd_mean)]
    ax.axvline(x=peak_lag, color="green", linestyle=":", linewidth=1.5, label=f"Peak: {peak_lag:.1f} ms")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title(labels["forward_title"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reverse: source
    xcorr_rev_list = []
    for idx in sample_idx:
        real = signals["source_real"][idx, 0]
        gen = signals["source_gen"][idx, 0]

        real = (real - real.mean()) / (real.std() + 1e-10)
        gen = (gen - gen.mean()) / (gen.std() + 1e-10)

        xcorr = np.correlate(gen, real, mode="full")
        xcorr = xcorr / len(real)
        mid = len(xcorr) // 2
        xcorr_rev_list.append(xcorr[mid - max_lag:mid + max_lag + 1])

    xcorr_rev = np.array(xcorr_rev_list)
    xcorr_rev_mean = xcorr_rev.mean(axis=0)
    xcorr_rev_std = xcorr_rev.std(axis=0)

    ax = axes[1]
    ax.plot(lag_ms, xcorr_rev_mean, color=COLORS["reverse"], linewidth=2)
    ax.fill_between(lag_ms, xcorr_rev_mean - xcorr_rev_std, xcorr_rev_mean + xcorr_rev_std,
                    alpha=0.2, color=COLORS["reverse"])
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Zero lag")
    peak_lag = lag_ms[np.argmax(xcorr_rev_mean)]
    ax.axvline(x=peak_lag, color="green", linestyle=":", linewidth=1.5, label=f"Peak: {peak_lag:.1f} ms")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title(labels["reverse_title"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Cross-Correlation Lag Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "25_cross_correlation", formats)
    plt.close()


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_summary_metrics(
    signals: Dict[str, np.ndarray],
    fs: float = SAMPLING_RATE_HZ,
) -> Dict[str, Any]:
    """Compute summary metrics for both directions."""
    metrics = {}
    labels = get_region_labels(signals)
    fwd_key = f"forward_{labels['source']}_to_{labels['target']}"
    rev_key = f"reverse_{labels['target']}_to_{labels['source']}"

    # Forward: source → target
    corr_fwd = compute_correlation(signals["target_gen"], signals["target_real"])

    freqs, psd_real = compute_psd(signals["target_real"], fs)
    _, psd_gen = compute_psd(signals["target_gen"], fs)
    psd_error_fwd = np.abs(
        10 * np.log10(psd_gen.mean(axis=(0, 1)) + 1e-10) -
        10 * np.log10(psd_real.mean(axis=(0, 1)) + 1e-10)
    ).mean()

    mse_fwd = np.mean((signals["target_gen"] - signals["target_real"]) ** 2)

    metrics[fwd_key] = {
        "correlation_mean": float(np.mean(corr_fwd)),
        "correlation_std": float(np.std(corr_fwd)),
        "correlation_median": float(np.median(corr_fwd)),
        "psd_error_db": float(psd_error_fwd),
        "mse": float(mse_fwd),
        "n_samples": int(len(corr_fwd)),
    }

    # Reverse: target → source
    corr_rev = compute_correlation(signals["source_gen"], signals["source_real"])

    freqs, psd_real = compute_psd(signals["source_real"], fs)
    _, psd_gen = compute_psd(signals["source_gen"], fs)
    psd_error_rev = np.abs(
        10 * np.log10(psd_gen.mean(axis=(0, 1)) + 1e-10) -
        10 * np.log10(psd_real.mean(axis=(0, 1)) + 1e-10)
    ).mean()

    mse_rev = np.mean((signals["source_gen"] - signals["source_real"]) ** 2)

    metrics[rev_key] = {
        "correlation_mean": float(np.mean(corr_rev)),
        "correlation_std": float(np.std(corr_rev)),
        "correlation_median": float(np.median(corr_rev)),
        "psd_error_db": float(psd_error_rev),
        "mse": float(mse_rev),
        "n_samples": int(len(corr_rev)),
    }
    
    # Store keys for later reference
    metrics["_keys"] = {"forward": fwd_key, "reverse": rev_key}

    return metrics


def compute_statistical_tests(
    signals: Dict[str, np.ndarray],
    fs: float = SAMPLING_RATE_HZ,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """Compute statistical tests including effect sizes and p-values."""
    # Note: scipy.stats is already imported at module level
    results = {"statistical_tests": {}, "per_condition_stats": {}}

    # Compute correlations for both directions using generic keys
    corr_fwd = compute_correlation(signals["target_gen"], signals["target_real"])
    corr_rev = compute_correlation(signals["source_gen"], signals["source_real"])

    # Paired t-test between forward and reverse correlations
    t_stat, p_value = stats.ttest_rel(corr_fwd, corr_rev)
    results["statistical_tests"]["correlation_ttest_pvalue"] = float(p_value)
    results["statistical_tests"]["correlation_ttest_statistic"] = float(t_stat)

    # Cohen's d effect size for correlations
    cohens_d = compute_cohens_d(corr_fwd, corr_rev)
    results["statistical_tests"]["cohens_d_correlation"] = float(cohens_d)

    # Bootstrap 95% CI for mean correlations
    ci_fwd = bootstrap_ci(corr_fwd, np.mean, n_bootstrap=n_bootstrap)
    ci_rev = bootstrap_ci(corr_rev, np.mean, n_bootstrap=n_bootstrap)
    results["statistical_tests"]["corr_fwd_ci_95"] = [float(ci_fwd[0]), float(ci_fwd[1])]
    results["statistical_tests"]["corr_rev_ci_95"] = [float(ci_rev[0]), float(ci_rev[1])]

    # Bland-Altman statistics for amplitude agreement
    real_amp_fwd = signals["target_real"].flatten()
    gen_amp_fwd = signals["target_gen"].flatten()
    # Sample for efficiency
    n_samples = min(100000, len(real_amp_fwd))
    idx = np.random.choice(len(real_amp_fwd), n_samples, replace=False)
    ba_stats = bland_altman_stats(real_amp_fwd[idx], gen_amp_fwd[idx])
    results["statistical_tests"]["bland_altman_fwd"] = {
        "bias": float(ba_stats["bias"]),
        "loa_lower": float(ba_stats["lower_loa"]),
        "loa_upper": float(ba_stats["upper_loa"]),
    }

    real_amp_rev = signals["source_real"].flatten()
    gen_amp_rev = signals["source_gen"].flatten()
    idx = np.random.choice(len(real_amp_rev), n_samples, replace=False)
    ba_stats = bland_altman_stats(real_amp_rev[idx], gen_amp_rev[idx])
    results["statistical_tests"]["bland_altman_rev"] = {
        "bias": float(ba_stats["bias"]),
        "loa_lower": float(ba_stats["lower_loa"]),
        "loa_upper": float(ba_stats["upper_loa"]),
    }

    # Per-condition statistics
    condition_names = np.unique(signals["condition_names"])
    for cond in condition_names:
        mask = np.array(signals["condition_names"]) == cond
        corr_fwd_cond = compute_correlation(
            signals["target_gen"][mask], signals["target_real"][mask]
        )
        corr_rev_cond = compute_correlation(
            signals["source_gen"][mask], signals["source_real"][mask]
        )

        results["per_condition_stats"][cond] = {
            "n_samples": int(np.sum(mask)),
            "corr_fwd_mean": float(np.mean(corr_fwd_cond)),
            "corr_fwd_std": float(np.std(corr_fwd_cond)),
            "corr_rev_mean": float(np.mean(corr_rev_cond)),
            "corr_rev_std": float(np.std(corr_rev_cond)),
        }

    return results


def compute_bootstrap_results(
    signals: Dict[str, np.ndarray],
    fs: float = SAMPLING_RATE_HZ,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """Compute bootstrap confidence intervals for all key metrics."""
    labels = get_region_labels(signals)
    fwd_key = f"forward_{labels['source']}_to_{labels['target']}"
    rev_key = f"reverse_{labels['target']}_to_{labels['source']}"
    
    results = {fwd_key: {}, rev_key: {}, "bootstrap_params": {}}

    results["bootstrap_params"] = {
        "n_bootstrap": n_bootstrap,
        "ci_level": 0.95,
        "random_seed": 42,
    }

    # Forward: source → target (bootstrap_ci uses its own seeded RNG)
    corr_fwd = compute_correlation(signals["target_gen"], signals["target_real"])
    mse_fwd = np.mean((signals["target_gen"] - signals["target_real"]) ** 2, axis=(1, 2))

    results[fwd_key]["correlation"] = {
        "mean": float(np.mean(corr_fwd)),
        "ci_95": [float(x) for x in bootstrap_ci(corr_fwd, np.mean, n_bootstrap)],
        "std": float(np.std(corr_fwd)),
    }

    results[fwd_key]["mse"] = {
        "mean": float(np.mean(mse_fwd)),
        "ci_95": [float(x) for x in bootstrap_ci(mse_fwd, np.mean, n_bootstrap)],
        "std": float(np.std(mse_fwd)),
    }

    # Compute PSD error per trial
    freqs, psd_real_fwd = compute_psd(signals["target_real"], fs)
    _, psd_gen_fwd = compute_psd(signals["target_gen"], fs)
    psd_error_fwd = np.abs(
        10 * np.log10(psd_gen_fwd.mean(axis=1) + 1e-10) -
        10 * np.log10(psd_real_fwd.mean(axis=1) + 1e-10)
    ).mean(axis=1)

    results[fwd_key]["psd_error_db"] = {
        "mean": float(np.mean(psd_error_fwd)),
        "ci_95": [float(x) for x in bootstrap_ci(psd_error_fwd, np.mean, n_bootstrap)],
        "std": float(np.std(psd_error_fwd)),
    }

    # Reverse: target → source
    corr_rev = compute_correlation(signals["source_gen"], signals["source_real"])
    mse_rev = np.mean((signals["source_gen"] - signals["source_real"]) ** 2, axis=(1, 2))

    results[rev_key]["correlation"] = {
        "mean": float(np.mean(corr_rev)),
        "ci_95": [float(x) for x in bootstrap_ci(corr_rev, np.mean, n_bootstrap)],
        "std": float(np.std(corr_rev)),
    }

    results[rev_key]["mse"] = {
        "mean": float(np.mean(mse_rev)),
        "ci_95": [float(x) for x in bootstrap_ci(mse_rev, np.mean, n_bootstrap)],
        "std": float(np.std(mse_rev)),
    }

    freqs, psd_real_rev = compute_psd(signals["source_real"], fs)
    _, psd_gen_rev = compute_psd(signals["source_gen"], fs)
    psd_error_rev = np.abs(
        10 * np.log10(psd_gen_rev.mean(axis=1) + 1e-10) -
        10 * np.log10(psd_real_rev.mean(axis=1) + 1e-10)
    ).mean(axis=1)

    results[rev_key]["psd_error_db"] = {
        "mean": float(np.mean(psd_error_rev)),
        "ci_95": [float(x) for x in bootstrap_ci(psd_error_rev, np.mean, n_bootstrap)],
        "std": float(np.std(psd_error_rev)),
    }
    
    # Store keys for later reference
    results["_keys"] = {"forward": fwd_key, "reverse": rev_key}

    return results


# =============================================================================
# Cross-Region Comparison Plots (Natural vs Generated relationships)
# =============================================================================

def plot_cross_region_correlation(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """
    Plot 26: Compare correlation between source-target regions for natural vs generated signals.

    Shows whether the model preserves the natural inter-regional dynamics.
    Natural: corr(real_source, real_target)
    Generated: corr(gen_source, gen_target) where gen_source = reverse(real_target), gen_target = forward(real_source)
    """
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    if formats is None:
        formats = ["png", "svg", "pdf"]

    source_real = signals["source_real"]      # (N, 32, T)
    target_real = signals["target_real"]    # (N, 32, T)
    source_gen = signals["source_gen"]        # Generated from target→source
    target_gen = signals["target_gen"]      # Generated from source→target

    n_samples = source_real.shape[0]

    # Compute per-trial cross-region correlations (average across channels)
    nat_corrs = []
    gen_corrs = []

    for i in range(n_samples):
        # Natural: correlation between real source and real target (flattened across channels)
        src_flat = source_real[i].flatten()
        tgt_flat = target_real[i].flatten()
        nat_corrs.append(np.corrcoef(src_flat, tgt_flat)[0, 1])

        # Generated: correlation between gen source (from target) and gen target (from source)
        src_gen_flat = source_gen[i].flatten()
        tgt_gen_flat = target_gen[i].flatten()
        gen_corrs.append(np.corrcoef(src_gen_flat, tgt_gen_flat)[0, 1])

    nat_corrs = np.array(nat_corrs)
    gen_corrs = np.array(gen_corrs)

    # Also compute per-channel correlations
    n_channels = source_real.shape[1]
    nat_channel_corrs = np.zeros((n_samples, n_channels))
    gen_channel_corrs = np.zeros((n_samples, n_channels))

    for i in range(n_samples):
        for ch in range(n_channels):
            nat_channel_corrs[i, ch] = np.corrcoef(source_real[i, ch], target_real[i, ch])[0, 1]
            gen_channel_corrs[i, ch] = np.corrcoef(source_gen[i, ch], target_gen[i, ch])[0, 1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Histogram of cross-region correlations
    ax = axes[0, 0]
    ax.hist(nat_corrs, bins=30, alpha=0.7, label=f'Natural (μ={np.mean(nat_corrs):.3f})', color=COLORS["real"])
    ax.hist(gen_corrs, bins=30, alpha=0.7, label=f'Generated (μ={np.mean(gen_corrs):.3f})', color=COLORS["generated"])
    ax.axvline(np.mean(nat_corrs), color=COLORS["real"], linestyle='--', linewidth=2)
    ax.axvline(np.mean(gen_corrs), color=COLORS["generated"], linestyle='--', linewidth=2)
    ax.set_xlabel(f"Cross-Region Correlation ({src} vs {tgt})")
    ax.set_ylabel("Count")
    ax.set_title("A. Cross-Region Correlation Distribution")
    ax.legend()

    # Panel B: Scatter plot - Natural vs Generated correlation
    ax = axes[0, 1]
    ax.scatter(nat_corrs, gen_corrs, alpha=0.5, s=20)
    min_val = min(nat_corrs.min(), gen_corrs.min())
    max_val = max(nat_corrs.max(), gen_corrs.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')
    ax.set_xlabel("Natural Cross-Region Correlation")
    ax.set_ylabel("Generated Cross-Region Correlation")
    ax.set_title("B. Natural vs Generated Cross-Region Correlation")
    r = np.corrcoef(nat_corrs, gen_corrs)[0, 1]
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12, va='top')
    ax.legend()

    # Panel C: Per-channel mean correlation comparison
    ax = axes[1, 0]
    nat_ch_mean = np.mean(nat_channel_corrs, axis=0)
    gen_ch_mean = np.mean(gen_channel_corrs, axis=0)
    x = np.arange(n_channels)
    width = 0.35
    ax.bar(x - width/2, nat_ch_mean, width, label='Natural', color=COLORS["real"], alpha=0.8)
    ax.bar(x + width/2, gen_ch_mean, width, label='Generated', color=COLORS["generated"], alpha=0.8)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Mean Cross-Region Correlation")
    ax.set_title("C. Per-Channel Cross-Region Correlation")
    ax.set_xticks(x[::4])
    ax.legend()

    # Panel D: Difference (Natural - Generated) per channel
    ax = axes[1, 1]
    diff = nat_ch_mean - gen_ch_mean
    colors = [COLORS["real"] if d > 0 else COLORS["generated"] for d in diff]
    ax.bar(x, diff, color=colors, alpha=0.8)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Correlation Difference (Natural - Generated)")
    ax.set_title("D. Cross-Region Correlation Preservation Error")
    ax.set_xticks(x[::4])

    plt.tight_layout()
    save_figure(fig, output_dir, "26_cross_region_correlation", formats)
    plt.close(fig)


def plot_cross_region_coherence(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """
    Plot 27: Cross-region coherence spectrum comparing natural vs generated.

    Shows frequency-resolved similarity between source and target regions.
    """
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    if formats is None:
        formats = ["png", "svg", "pdf"]

    source_real = signals["source_real"]
    target_real = signals["target_real"]
    source_gen = signals["source_gen"]
    target_gen = signals["target_gen"]

    n_samples = min(50, source_real.shape[0])  # Limit for computational efficiency
    nperseg = min(512, source_real.shape[2] // 4)

    # Compute coherence for a few representative channels
    channels_to_plot = [0, 8, 16, 24]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, ch in enumerate(channels_to_plot):
        ax = axes[idx // 2, idx % 2]

        # Accumulate coherence across samples
        nat_coherences = []
        gen_coherences = []

        for i in range(n_samples):
            # Natural coherence: real source vs real target
            f, Cxy_nat = coherence(source_real[i, ch], target_real[i, ch], fs=fs, nperseg=nperseg)
            nat_coherences.append(Cxy_nat)

            # Generated coherence: gen source vs gen target
            f, Cxy_gen = coherence(source_gen[i, ch], target_gen[i, ch], fs=fs, nperseg=nperseg)
            gen_coherences.append(Cxy_gen)

        nat_coherences = np.array(nat_coherences)
        gen_coherences = np.array(gen_coherences)

        # Plot mean ± SEM
        nat_mean = np.mean(nat_coherences, axis=0)
        nat_sem = np.std(nat_coherences, axis=0) / np.sqrt(n_samples)
        gen_mean = np.mean(gen_coherences, axis=0)
        gen_sem = np.std(gen_coherences, axis=0) / np.sqrt(n_samples)

        # Limit frequency range
        freq_mask = f <= 100
        f_plot = f[freq_mask]

        ax.plot(f_plot, nat_mean[freq_mask], color=COLORS["real"], label='Natural', linewidth=2)
        ax.fill_between(f_plot, (nat_mean - nat_sem)[freq_mask], (nat_mean + nat_sem)[freq_mask],
                       color=COLORS["real"], alpha=0.3)

        ax.plot(f_plot, gen_mean[freq_mask], color=COLORS["generated"], label='Generated', linewidth=2)
        ax.fill_between(f_plot, (gen_mean - gen_sem)[freq_mask], (gen_mean + gen_sem)[freq_mask],
                       color=COLORS["generated"], alpha=0.3)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Coherence")
        ax.set_title(f"Channel {ch}")
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Cross-Region Coherence: Natural ({src}-{tgt}) vs Generated ({src}-{tgt})", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, "27_cross_region_coherence", formats)
    plt.close(fig)


def plot_cross_region_plv(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """
    Plot 28: Cross-region Phase Locking Value (PLV) comparison.

    Compares phase synchronization between source-target in natural vs generated signals.
    """
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    if formats is None:
        formats = ["png", "svg", "pdf"]

    source_real = signals["source_real"]
    target_real = signals["target_real"]
    source_gen = signals["source_gen"]
    target_gen = signals["target_gen"]

    # Frequency bands for PLV
    bands = {
        "Delta (1-4 Hz)": (1, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-12 Hz)": (8, 12),
        "Beta (12-30 Hz)": (12, 30),
        "Low Gamma (30-50 Hz)": (30, 50),
        "High Gamma (50-100 Hz)": (50, 100),
    }

    n_samples = min(100, source_real.shape[0])
    n_channels = source_real.shape[1]

    nat_plv_per_band = {band: [] for band in bands}
    gen_plv_per_band = {band: [] for band in bands}

    for band_name, (low, high) in bands.items():
        # Bandpass filter
        sos = butter(4, [low, high], btype='band', fs=fs, output='sos')

        nat_plvs = []
        gen_plvs = []

        for i in range(n_samples):
            trial_nat_plv = []
            trial_gen_plv = []

            for ch in range(n_channels):
                # Filter signals
                source_real_filt = sosfilt(sos, source_real[i, ch])
                target_real_filt = sosfilt(sos, target_real[i, ch])
                source_gen_filt = sosfilt(sos, source_gen[i, ch])
                target_gen_filt = sosfilt(sos, target_gen[i, ch])

                # Compute instantaneous phase using Hilbert transform
                phase_source_real = np.angle(hilbert(source_real_filt))
                phase_target_real = np.angle(hilbert(target_real_filt))
                phase_source_gen = np.angle(hilbert(source_gen_filt))
                phase_target_gen = np.angle(hilbert(target_gen_filt))

                # PLV = |mean(exp(j*(phase1 - phase2)))|
                nat_plv = np.abs(np.mean(np.exp(1j * (phase_source_real - phase_target_real))))
                gen_plv = np.abs(np.mean(np.exp(1j * (phase_source_gen - phase_target_gen))))

                trial_nat_plv.append(nat_plv)
                trial_gen_plv.append(gen_plv)

            nat_plvs.append(np.mean(trial_nat_plv))
            gen_plvs.append(np.mean(trial_gen_plv))

        nat_plv_per_band[band_name] = nat_plvs
        gen_plv_per_band[band_name] = gen_plvs

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Grouped bar plot
    ax = axes[0]
    band_names = list(bands.keys())
    x = np.arange(len(band_names))
    width = 0.35

    nat_means = [np.mean(nat_plv_per_band[b]) for b in band_names]
    nat_stds = [np.std(nat_plv_per_band[b]) for b in band_names]
    gen_means = [np.mean(gen_plv_per_band[b]) for b in band_names]
    gen_stds = [np.std(gen_plv_per_band[b]) for b in band_names]

    bars1 = ax.bar(x - width/2, nat_means, width, yerr=nat_stds, label='Natural',
                   color=COLORS["real"], alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, gen_means, width, yerr=gen_stds, label='Generated',
                   color=COLORS["generated"], alpha=0.8, capsize=3)

    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Cross-Region PLV")
    ax.set_title("A. Cross-Region Phase Locking Value by Band")
    ax.set_xticks(x)
    ax.set_xticklabels([b.split()[0] for b in band_names], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])

    # Panel B: PLV preservation (scatter)
    ax = axes[1]
    all_nat = []
    all_gen = []
    colors_scatter = []
    color_map = plt.cm.viridis(np.linspace(0, 1, len(bands)))

    for idx, band_name in enumerate(band_names):
        all_nat.extend(nat_plv_per_band[band_name])
        all_gen.extend(gen_plv_per_band[band_name])
        colors_scatter.extend([color_map[idx]] * len(nat_plv_per_band[band_name]))

    ax.scatter(all_nat, all_gen, c=colors_scatter, alpha=0.5, s=20)
    ax.plot([0, 1], [0, 1], 'k--', label='y=x')
    ax.set_xlabel("Natural Cross-Region PLV")
    ax.set_ylabel("Generated Cross-Region PLV")
    ax.set_title("B. PLV Preservation Across Bands")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add legend for bands
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[i], label=band_names[i].split()[0])
                      for i in range(len(band_names))]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    save_figure(fig, output_dir, "28_cross_region_plv", formats)
    plt.close(fig)


def plot_cross_region_psd_ratio(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """
    Plot 29: Cross-region PSD ratio comparison.

    Compares the ratio PSD(target)/PSD(source) for natural vs generated signals.
    This shows if the model preserves the relative power between regions.
    """
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    if formats is None:
        formats = ["png", "svg", "pdf"]

    src_real = signals["source_real"]
    tgt_real = signals["target_real"]
    src_gen = signals["source_gen"]
    tgt_gen = signals["target_gen"]

    nperseg = min(512, src_real.shape[2] // 4)

    # Compute PSD ratio for natural and generated
    all_nat_ratios = []
    all_gen_ratios = []

    for i in range(src_real.shape[0]):
        for ch in range(src_real.shape[1]):
            # Natural PSD
            f, psd_src_real = welch(src_real[i, ch], fs=fs, nperseg=nperseg)
            _, psd_tgt_real = welch(tgt_real[i, ch], fs=fs, nperseg=nperseg)

            # Generated PSD
            _, psd_src_gen = welch(src_gen[i, ch], fs=fs, nperseg=nperseg)
            _, psd_tgt_gen = welch(tgt_gen[i, ch], fs=fs, nperseg=nperseg)

            # Compute ratios (add small epsilon to avoid division by zero)
            eps = 1e-10
            nat_ratio = psd_tgt_real / (psd_src_real + eps)
            gen_ratio = psd_tgt_gen / (psd_src_gen + eps)

            all_nat_ratios.append(nat_ratio)
            all_gen_ratios.append(gen_ratio)

    all_nat_ratios = np.array(all_nat_ratios)
    all_gen_ratios = np.array(all_gen_ratios)

    # Limit frequency range
    freq_mask = f <= 100
    f_plot = f[freq_mask]

    nat_mean = np.mean(all_nat_ratios[:, freq_mask], axis=0)
    nat_std = np.std(all_nat_ratios[:, freq_mask], axis=0)
    gen_mean = np.mean(all_gen_ratios[:, freq_mask], axis=0)
    gen_std = np.std(all_gen_ratios[:, freq_mask], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: PSD ratio spectrum (linear scale)
    ax = axes[0, 0]
    ax.semilogy(f_plot, nat_mean, color=COLORS["real"], label='Natural', linewidth=2)
    ax.fill_between(f_plot, nat_mean - nat_std, nat_mean + nat_std,
                   color=COLORS["real"], alpha=0.3)
    ax.semilogy(f_plot, gen_mean, color=COLORS["generated"], label='Generated', linewidth=2)
    ax.fill_between(f_plot, gen_mean - gen_std, gen_mean + gen_std,
                   color=COLORS["generated"], alpha=0.3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"PSD Ratio ({tgt}/{src})")
    ax.set_title("A. Cross-Region PSD Ratio Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])

    # Panel B: Ratio difference (natural - generated)
    ax = axes[0, 1]
    ratio_diff = nat_mean - gen_mean
    ax.plot(f_plot, ratio_diff, color='purple', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.fill_between(f_plot, ratio_diff, 0, where=ratio_diff > 0,
                   color=COLORS["real"], alpha=0.3, label='Natural > Gen')
    ax.fill_between(f_plot, ratio_diff, 0, where=ratio_diff < 0,
                   color=COLORS["generated"], alpha=0.3, label='Generated > Nat')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Ratio Difference (Nat - Gen)")
    ax.set_title("B. PSD Ratio Preservation Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])

    # Panel C: Per-band ratio comparison
    ax = axes[1, 0]
    bands = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (12, 30),
        "Gamma": (30, 100),
    }

    band_nat_ratios = []
    band_gen_ratios = []
    band_names = []

    for band_name, (low, high) in bands.items():
        band_mask = (f >= low) & (f <= high)
        nat_band = np.mean(all_nat_ratios[:, band_mask], axis=1)
        gen_band = np.mean(all_gen_ratios[:, band_mask], axis=1)
        band_nat_ratios.append(np.mean(nat_band))
        band_gen_ratios.append(np.mean(gen_band))
        band_names.append(band_name)

    x = np.arange(len(band_names))
    width = 0.35
    ax.bar(x - width/2, band_nat_ratios, width, label='Natural', color=COLORS["real"], alpha=0.8)
    ax.bar(x + width/2, band_gen_ratios, width, label='Generated', color=COLORS["generated"], alpha=0.8)
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Mean PSD Ratio (PCx/OB)")
    ax.set_title("C. Per-Band Cross-Region Power Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.legend()
    ax.set_yscale('log')

    # Panel D: Scatter plot of natural vs generated ratios (band-averaged)
    ax = axes[1, 1]

    # Sample a subset for clarity
    n_sample = min(500, all_nat_ratios.shape[0])
    indices = np.random.choice(all_nat_ratios.shape[0], n_sample, replace=False)

    nat_mean_per_sample = np.mean(all_nat_ratios[indices, freq_mask], axis=1)
    gen_mean_per_sample = np.mean(all_gen_ratios[indices, freq_mask], axis=1)

    ax.scatter(nat_mean_per_sample, gen_mean_per_sample, alpha=0.3, s=10)
    min_val = min(nat_mean_per_sample.min(), gen_mean_per_sample.min())
    max_val = max(nat_mean_per_sample.max(), gen_mean_per_sample.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')
    ax.set_xlabel("Natural PSD Ratio")
    ax.set_ylabel("Generated PSD Ratio")
    ax.set_title("D. PSD Ratio Scatter (per channel/trial)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    r = np.corrcoef(nat_mean_per_sample, gen_mean_per_sample)[0, 1]
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12, va='top')

    plt.tight_layout()
    save_figure(fig, output_dir, "29_cross_region_psd_ratio", formats)
    plt.close(fig)


def plot_cross_region_xcorr(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """
    Plot 30: Cross-region cross-correlation comparison.

    Compares the temporal relationship between source and target signals.
    """
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    if formats is None:
        formats = ["png", "svg", "pdf"]

    src_real = signals["source_real"]
    tgt_real = signals["target_real"]
    src_gen = signals["source_gen"]
    tgt_gen = signals["target_gen"]

    n_samples = min(50, src_real.shape[0])
    max_lag = int(0.1 * fs)  # 100ms max lag

    # Compute cross-correlation for representative channels
    channels = [0, 8, 16, 24]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, ch in enumerate(channels):
        ax = axes[idx // 2, idx % 2]

        nat_xcorrs = []
        gen_xcorrs = []

        for i in range(n_samples):
            # Normalize signals
            s_r = (src_real[i, ch] - src_real[i, ch].mean()) / (src_real[i, ch].std() + 1e-8)
            t_r = (tgt_real[i, ch] - tgt_real[i, ch].mean()) / (tgt_real[i, ch].std() + 1e-8)
            s_g = (src_gen[i, ch] - src_gen[i, ch].mean()) / (src_gen[i, ch].std() + 1e-8)
            t_g = (tgt_gen[i, ch] - tgt_gen[i, ch].mean()) / (tgt_gen[i, ch].std() + 1e-8)

            # Cross-correlation (natural)
            nat_xcorr = np.correlate(s_r, t_r, mode='full')
            nat_xcorr = nat_xcorr / len(s_r)  # Normalize
            center = len(nat_xcorr) // 2
            nat_xcorr = nat_xcorr[center - max_lag:center + max_lag + 1]
            nat_xcorrs.append(nat_xcorr)

            # Cross-correlation (generated)
            gen_xcorr = np.correlate(s_g, t_g, mode='full')
            gen_xcorr = gen_xcorr / len(s_g)
            gen_xcorr = gen_xcorr[center - max_lag:center + max_lag + 1]
            gen_xcorrs.append(gen_xcorr)

        nat_xcorrs = np.array(nat_xcorrs)
        gen_xcorrs = np.array(gen_xcorrs)

        lags = np.arange(-max_lag, max_lag + 1) / fs * 1000  # Convert to ms

        nat_mean = np.mean(nat_xcorrs, axis=0)
        nat_sem = np.std(nat_xcorrs, axis=0) / np.sqrt(n_samples)
        gen_mean = np.mean(gen_xcorrs, axis=0)
        gen_sem = np.std(gen_xcorrs, axis=0) / np.sqrt(n_samples)

        ax.plot(lags, nat_mean, color=COLORS["real"], label='Natural', linewidth=2)
        ax.fill_between(lags, nat_mean - nat_sem, nat_mean + nat_sem,
                       color=COLORS["real"], alpha=0.3)
        ax.plot(lags, gen_mean, color=COLORS["generated"], label='Generated', linewidth=2)
        ax.fill_between(lags, gen_mean - gen_sem, gen_mean + gen_sem,
                       color=COLORS["generated"], alpha=0.3)

        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Cross-Correlation")
        ax.set_title(f"Channel {ch}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Cross-Region Cross-Correlation: Natural vs Generated {src}-{tgt}", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, "30_cross_region_xcorr", formats)
    plt.close(fig)


def plot_cross_region_band_power(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """
    Plot 31: Cross-region band power scatter comparison.

    Scatter plots of source vs target band power for natural and generated signals.
    Shows whether the model preserves the natural power relationship.
    """
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    if formats is None:
        formats = ["png", "svg", "pdf"]

    src_real = signals["source_real"]
    tgt_real = signals["target_real"]
    src_gen = signals["source_gen"]
    tgt_gen = signals["target_gen"]

    bands = {
        "Delta (1-4 Hz)": (1, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Beta (12-30 Hz)": (12, 30),
        "Gamma (30-100 Hz)": (30, 100),
    }

    nperseg = min(512, src_real.shape[2] // 4)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
        # Compute band power for all samples/channels
        nat_src_power = []
        nat_tgt_power = []
        gen_src_power = []
        gen_tgt_power = []

        for i in range(src_real.shape[0]):
            for ch in range(src_real.shape[1]):
                f, psd_src_real = welch(src_real[i, ch], fs=fs, nperseg=nperseg)
                _, psd_tgt_real = welch(tgt_real[i, ch], fs=fs, nperseg=nperseg)
                _, psd_src_gen = welch(src_gen[i, ch], fs=fs, nperseg=nperseg)
                _, psd_tgt_gen = welch(tgt_gen[i, ch], fs=fs, nperseg=nperseg)

                # Get band power (mean PSD in band)
                band_mask = (f >= low) & (f <= high)
                nat_src_power.append(np.mean(psd_src_real[band_mask]))
                nat_tgt_power.append(np.mean(psd_tgt_real[band_mask]))
                gen_src_power.append(np.mean(psd_src_gen[band_mask]))
                gen_tgt_power.append(np.mean(psd_tgt_gen[band_mask]))

        nat_src_power = np.array(nat_src_power)
        nat_tgt_power = np.array(nat_tgt_power)
        gen_src_power = np.array(gen_src_power)
        gen_tgt_power = np.array(gen_tgt_power)

        # Subsample for plotting
        n_plot = min(500, len(nat_src_power))
        indices = np.random.choice(len(nat_src_power), n_plot, replace=False)

        # Top row: Natural
        ax = axes[0, band_idx]
        ax.scatter(nat_src_power[indices], nat_tgt_power[indices], alpha=0.3, s=10, color=COLORS["real"])

        # Fit line
        z = np.polyfit(np.log10(nat_src_power[indices] + 1e-10),
                       np.log10(nat_tgt_power[indices] + 1e-10), 1)
        r = np.corrcoef(np.log10(nat_src_power[indices] + 1e-10),
                       np.log10(nat_tgt_power[indices] + 1e-10))[0, 1]

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f"{src} Power")
        ax.set_ylabel(f"{tgt} Power")
        ax.set_title(f"Natural - {band_name.split()[0]}")
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=10, va='top')

        # Bottom row: Generated
        ax = axes[1, band_idx]
        ax.scatter(gen_src_power[indices], gen_tgt_power[indices], alpha=0.3, s=10, color=COLORS["generated"])

        r_gen = np.corrcoef(np.log10(gen_src_power[indices] + 1e-10),
                           np.log10(gen_tgt_power[indices] + 1e-10))[0, 1]

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f"{src} Power")
        ax.set_ylabel(f"{tgt} Power")
        ax.set_title(f"Generated - {band_name.split()[0]}")
        ax.text(0.05, 0.95, f'r = {r_gen:.3f}', transform=ax.transAxes, fontsize=10, va='top')

    fig.suptitle(f"Cross-Region Band Power: {src} vs {tgt}\n(Top: Natural, Bottom: Generated)", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, "31_cross_region_band_power", formats)
    plt.close(fig)


# =============================================================================
# Advanced Statistical Functions (Nature Methods Quality)
# =============================================================================

def compute_effect_size(group1: np.ndarray, group2: np.ndarray, method: str = "cohens_d") -> Dict[str, float]:
    """Compute effect size with confidence intervals."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if method == "cohens_d":
        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    elif method == "hedges_g":
        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        d = d * correction
        se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2))) * correction
    elif method == "cliff_delta":
        greater = np.sum(group1[:, None] > group2[None, :])
        less = np.sum(group1[:, None] < group2[None, :])
        d = (greater - less) / (n1 * n2)
        se = np.sqrt((1 - d**2) / (n1 * n2))
    else:
        d, se = 0, 0

    ci_95 = (d - 1.96 * se, d + 1.96 * se)
    return {"effect_size": d, "se": se, "ci_95_lower": ci_95[0], "ci_95_upper": ci_95[1], "method": method}


def permutation_test(group1: np.ndarray, group2: np.ndarray, n_permutations: int = 5000) -> Dict[str, float]:
    """Permutation test for comparing two groups."""
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    observed = np.mean(group1) - np.mean(group2)

    perm_stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = np.random.permutation(combined)
        perm_stats[i] = np.mean(perm[:n1]) - np.mean(perm[n1:])

    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
    return {"observed_statistic": observed, "p_value": p_value, "n_permutations": n_permutations}


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    threshold = (np.arange(1, n + 1) / n) * alpha

    below_threshold = sorted_p <= threshold
    if np.any(below_threshold):
        max_k = np.max(np.where(below_threshold)[0])
        rejected = np.zeros(n, dtype=bool)
        rejected[sorted_idx[:max_k + 1]] = True
    else:
        rejected = np.zeros(n, dtype=bool)

    adjusted_p = np.zeros(n)
    adjusted_p[sorted_idx] = np.minimum.accumulate((n / np.arange(1, n + 1)) * sorted_p)[::-1][::-1]
    adjusted_p = np.minimum(adjusted_p, 1.0)
    return rejected, adjusted_p


def comprehensive_stats(real_data: np.ndarray, gen_data: np.ndarray, n_bootstrap: int = 1000) -> Dict[str, Any]:
    """Compute comprehensive statistics for Nature Methods quality."""
    results = {}
    results["real_mean"], results["real_std"] = float(np.mean(real_data)), float(np.std(real_data))
    results["gen_mean"], results["gen_std"] = float(np.mean(gen_data)), float(np.std(gen_data))

    results["cohens_d"] = compute_effect_size(real_data, gen_data, "cohens_d")
    results["hedges_g"] = compute_effect_size(real_data, gen_data, "hedges_g")

    t_stat, t_pval = stats.ttest_ind(real_data, gen_data)
    results["ttest"] = {"t_statistic": float(t_stat), "p_value": float(t_pval)}

    u_stat, u_pval = stats.mannwhitneyu(real_data, gen_data, alternative='two-sided')
    results["mann_whitney"] = {"u_statistic": float(u_stat), "p_value": float(u_pval)}

    ks_stat, ks_pval = stats.ks_2samp(real_data, gen_data)
    results["ks_test"] = {"ks_statistic": float(ks_stat), "p_value": float(ks_pval)}

    results["permutation_test"] = permutation_test(real_data, gen_data, n_permutations=min(n_bootstrap, 5000))

    mean_diffs = []
    for _ in range(n_bootstrap):
        real_boot = np.random.choice(real_data, size=len(real_data), replace=True)
        gen_boot = np.random.choice(gen_data, size=len(gen_data), replace=True)
        mean_diffs.append(np.mean(real_boot) - np.mean(gen_boot))
    results["bootstrap_mean_diff"] = {
        "mean": float(np.mean(mean_diffs)),
        "ci_95": [float(np.percentile(mean_diffs, 2.5)), float(np.percentile(mean_diffs, 97.5))],
    }
    return results


def add_stats_annotation(ax: plt.Axes, stats_dict: Dict[str, Any], position: str = "top_right", fontsize: int = 8):
    """Add statistical annotations to a plot axis."""
    lines = []
    if "cohens_d" in stats_dict:
        d = stats_dict["cohens_d"]["effect_size"]
        ci = (stats_dict["cohens_d"]["ci_95_lower"], stats_dict["cohens_d"]["ci_95_upper"])
        lines.append(f"d={d:.3f} [{ci[0]:.2f},{ci[1]:.2f}]")
    if "ttest" in stats_dict:
        p = stats_dict["ttest"]["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        lines.append(f"t-test: p={p:.2e} {sig}")
    if "mann_whitney" in stats_dict:
        p = stats_dict["mann_whitney"]["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        lines.append(f"M-W: p={p:.2e} {sig}")
    if "ks_test" in stats_dict:
        ks = stats_dict["ks_test"]["ks_statistic"]
        p = stats_dict["ks_test"]["p_value"]
        lines.append(f"KS={ks:.3f}, p={p:.2e}")

    text = "\n".join(lines)
    pos_map = {"top_right": (0.98, 0.98, "right", "top"), "top_left": (0.02, 0.98, "left", "top"),
               "bottom_right": (0.98, 0.02, "right", "bottom"), "bottom_left": (0.02, 0.02, "left", "bottom")}
    x, y, ha, va = pos_map.get(position, pos_map["top_right"])
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize, verticalalignment=va, horizontalalignment=ha,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"))


def significance_stars(p_value: float) -> str:
    """Return significance stars for p-value."""
    if p_value < 0.001: return "***"
    elif p_value < 0.01: return "**"
    elif p_value < 0.05: return "*"
    else: return "ns"


# =============================================================================
# Real/Imaginary (Analytic Signal) Analysis Plots
# =============================================================================

def plot_analytic_magnitude(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 32: Distribution of analytic signal magnitude (envelope)."""
    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    if formats is None:
        formats = ["png", "svg", "pdf"]

    tgt_real, tgt_gen = signals["target_real"], signals["target_gen"]
    src_real, src_gen = signals["source_real"], signals["source_gen"]
    n_samples = min(100, tgt_real.shape[0])

    def get_envelope_stats(data, n_samples):
        envelopes = []
        for i in range(n_samples):
            for ch in range(data.shape[1]):
                envelope = np.abs(hilbert(data[i, ch]))
                envelopes.extend(envelope)
        return np.array(envelopes)

    tgt_real_env = get_envelope_stats(tgt_real, n_samples)
    tgt_gen_env = get_envelope_stats(tgt_gen, n_samples)
    src_real_env = get_envelope_stats(src_real, n_samples)
    src_gen_env = get_envelope_stats(src_gen, n_samples)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Target region envelope distribution
    ax = axes[0, 0]
    ax.hist(tgt_real_env, bins=100, alpha=0.7, label='Real', color=COLORS["real"], density=True)
    ax.hist(tgt_gen_env, bins=100, alpha=0.7, label='Generated', color=COLORS["generated"], density=True)
    ax.set_xlabel("Envelope Magnitude")
    ax.set_ylabel("Density")
    ax.set_title(f"A. {tgt} Envelope Distribution")
    ax.legend()
    ax.set_xlim([0, np.percentile(np.concatenate([tgt_real_env, tgt_gen_env]), 99)])
    stats_tgt = comprehensive_stats(tgt_real_env[::100], tgt_gen_env[::100], n_bootstrap=500)
    add_stats_annotation(ax, stats_tgt, position="top_right", fontsize=7)

    # Source region envelope distribution
    ax = axes[0, 1]
    ax.hist(src_real_env, bins=100, alpha=0.7, label='Real', color=COLORS["real"], density=True)
    ax.hist(src_gen_env, bins=100, alpha=0.7, label='Generated', color=COLORS["generated"], density=True)
    ax.set_xlabel("Envelope Magnitude")
    ax.set_ylabel("Density")
    ax.set_title(f"B. {src} Envelope Distribution")
    ax.legend()
    ax.set_xlim([0, np.percentile(np.concatenate([src_real_env, src_gen_env]), 99)])
    stats_src = comprehensive_stats(src_real_env[::100], src_gen_env[::100], n_bootstrap=500)
    add_stats_annotation(ax, stats_src, position="top_right", fontsize=7)

    # Q-Q plot
    ax = axes[0, 2]
    tgt_real_q = np.percentile(tgt_real_env, np.linspace(1, 99, 100))
    tgt_gen_q = np.percentile(tgt_gen_env, np.linspace(1, 99, 100))
    src_real_q = np.percentile(src_real_env, np.linspace(1, 99, 100))
    src_gen_q = np.percentile(src_gen_env, np.linspace(1, 99, 100))
    ax.scatter(tgt_real_q, tgt_gen_q, alpha=0.7, s=20, label=tgt)
    ax.scatter(src_real_q, src_gen_q, alpha=0.7, s=20, label=src)
    max_val = max(tgt_real_q.max(), src_real_q.max())
    ax.plot([0, max_val], [0, max_val], 'k--', label='y=x')
    ax.set_xlabel("Real Envelope Quantiles")
    ax.set_ylabel("Generated Envelope Quantiles")
    ax.set_title("C. Envelope Q-Q Plot")
    ax.legend()

    # Mean envelope per trial
    ax = axes[1, 0]
    tgt_real_trial = [np.mean(np.abs(hilbert(tgt_real[i]))) for i in range(n_samples)]
    tgt_gen_trial = [np.mean(np.abs(hilbert(tgt_gen[i]))) for i in range(n_samples)]
    ax.scatter(tgt_real_trial, tgt_gen_trial, alpha=0.5, s=30)
    r = np.corrcoef(tgt_real_trial, tgt_gen_trial)[0, 1]
    ax.plot([min(tgt_real_trial), max(tgt_real_trial)], [min(tgt_real_trial), max(tgt_real_trial)], 'k--')
    ax.set_xlabel("Real Mean Envelope")
    ax.set_ylabel("Generated Mean Envelope")
    ax.set_title(f"D. {tgt} Trial-wise Envelope (r={r:.3f})")

    # Envelope variance per trial
    ax = axes[1, 1]
    tgt_real_var = [np.var(np.abs(hilbert(tgt_real[i]))) for i in range(n_samples)]
    tgt_gen_var = [np.var(np.abs(hilbert(tgt_gen[i]))) for i in range(n_samples)]
    ax.scatter(tgt_real_var, tgt_gen_var, alpha=0.5, s=30)
    r = np.corrcoef(tgt_real_var, tgt_gen_var)[0, 1]
    ax.set_xlabel("Real Envelope Variance")
    ax.set_ylabel("Generated Envelope Variance")
    ax.set_title(f"E. {tgt} Envelope Variance (r={r:.3f})")

    # Effect size summary
    ax = axes[1, 2]
    effect_sizes = {tgt: stats_tgt["cohens_d"]["effect_size"], src: stats_src["cohens_d"]["effect_size"]}
    y_pos = np.arange(len(effect_sizes))
    ax.barh(y_pos, list(effect_sizes.values()), color=[COLORS["real"], COLORS["generated"]], alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(-0.2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(effect_sizes.keys()))
    ax.set_xlabel("Cohen's d")
    ax.set_title("F. Effect Size Summary")

    plt.tight_layout()
    save_figure(fig, output_dir, "32_analytic_magnitude", formats)
    plt.close(fig)


def plot_phase_distribution(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 33: Phase distribution comparison from analytic signal."""
    if formats is None:
        formats = ["png", "svg", "pdf"]

    labels = get_region_labels(signals)
    tgt = labels["target"]
    tgt_real, tgt_gen = signals["target_real"], signals["target_gen"]
    n_samples = min(50, tgt_real.shape[0])

    bands = {"Theta (4-8 Hz)": (4, 8), "Beta (12-30 Hz)": (12, 30), "Gamma (30-80 Hz)": (30, 80)}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
        sos = butter(4, [low, high], btype='band', fs=fs, output='sos')
        tgt_real_phases, tgt_gen_phases = [], []

        for i in range(n_samples):
            for ch in range(min(8, tgt_real.shape[1])):
                tgt_real_phases.extend(np.angle(hilbert(sosfilt(sos, tgt_real[i, ch]))))
                tgt_gen_phases.extend(np.angle(hilbert(sosfilt(sos, tgt_gen[i, ch]))))

        tgt_real_phases = np.array(tgt_real_phases)
        tgt_gen_phases = np.array(tgt_gen_phases)

        # Phase histogram
        ax = axes[0, band_idx]
        bins = np.linspace(-np.pi, np.pi, 37)
        real_hist, _ = np.histogram(tgt_real_phases, bins=bins, density=True)
        gen_hist, _ = np.histogram(tgt_gen_phases, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_centers, real_hist, color=COLORS["real"], label='Real', linewidth=2)
        ax.plot(bin_centers, gen_hist, color=COLORS["generated"], label='Generated', linewidth=2)
        ax.set_xlabel("Phase (radians)")
        ax.set_ylabel("Density")
        ax.set_title(f"A{band_idx+1}. {band_name}")
        ax.legend()

        ks_stat, ks_p = stats.ks_2samp(tgt_real_phases[::100], tgt_gen_phases[::100])
        ax.text(0.02, 0.98, f"KS={ks_stat:.3f}\np={ks_p:.2e} {significance_stars(ks_p)}",
               transform=ax.transAxes, fontsize=8, va='top', bbox=dict(facecolor='white', alpha=0.8))

        # Phase difference
        ax = axes[1, band_idx]
        n_compare = min(len(tgt_real_phases), 100000)
        idx = np.random.choice(len(tgt_real_phases), n_compare, replace=False)
        phase_diff = np.angle(np.exp(1j * (tgt_real_phases[idx] - tgt_gen_phases[idx])))
        ax.hist(phase_diff, bins=50, density=True, alpha=0.7, color='purple')
        ax.axvline(0, color='black', linestyle='--')
        ax.set_xlabel("Phase Difference (Real - Gen)")
        ax.set_ylabel("Density")
        ax.set_title(f"B{band_idx+1}. {band_name} Alignment")

    fig.suptitle("Instantaneous Phase Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, "33_phase_distribution", formats)
    plt.close(fig)


def plot_instantaneous_frequency(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 34: Instantaneous frequency comparison."""
    if formats is None:
        formats = ["png", "svg", "pdf"]

    labels = get_region_labels(signals)
    tgt = labels["target"]
    tgt_real, tgt_gen = signals["target_real"], signals["target_gen"]
    n_samples = min(50, tgt_real.shape[0])

    def compute_inst_freq(signal, fs):
        phase = np.unwrap(np.angle(hilbert(signal)))
        inst_freq = np.diff(phase) * fs / (2 * np.pi)
        return inst_freq[(inst_freq > 0) & (inst_freq < fs/2)]

    tgt_real_if, tgt_gen_if = [], []
    for i in range(n_samples):
        for ch in range(min(8, tgt_real.shape[1])):
            tgt_real_if.extend(compute_inst_freq(tgt_real[i, ch], fs))
            tgt_gen_if.extend(compute_inst_freq(tgt_gen[i, ch], fs))

    tgt_real_if = np.array(tgt_real_if)
    tgt_gen_if = np.array(tgt_gen_if)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Overall IF distribution
    ax = axes[0, 0]
    bins = np.linspace(0, 150, 100)
    ax.hist(tgt_real_if, bins=bins, alpha=0.7, label='Real', color=COLORS["real"], density=True)
    ax.hist(tgt_gen_if, bins=bins, alpha=0.7, label='Generated', color=COLORS["generated"], density=True)
    ax.set_xlabel("Instantaneous Frequency (Hz)")
    ax.set_ylabel("Density")
    ax.set_title("A. IF Distribution (0-150 Hz)")
    ax.legend()
    stats_if = comprehensive_stats(tgt_real_if[::100], tgt_gen_if[::100], n_bootstrap=500)
    add_stats_annotation(ax, stats_if, position="top_right", fontsize=7)

    # Zoomed IF
    ax = axes[0, 1]
    bins = np.linspace(0, 50, 100)
    ax.hist(tgt_real_if[tgt_real_if < 50], bins=bins, alpha=0.7, label='Real', color=COLORS["real"], density=True)
    ax.hist(tgt_gen_if[tgt_gen_if < 50], bins=bins, alpha=0.7, label='Generated', color=COLORS["generated"], density=True)
    ax.set_xlabel("Instantaneous Frequency (Hz)")
    ax.set_ylabel("Density")
    ax.set_title("B. IF Distribution (0-50 Hz)")
    ax.legend()

    # Q-Q plot
    ax = axes[0, 2]
    percentiles = np.linspace(1, 99, 100)
    real_q = np.percentile(tgt_real_if, percentiles)
    gen_q = np.percentile(tgt_gen_if, percentiles)
    ax.scatter(real_q, gen_q, alpha=0.7, s=20)
    ax.plot([0, real_q.max()], [0, real_q.max()], 'k--')
    ax.set_xlabel("Real IF Quantiles (Hz)")
    ax.set_ylabel("Generated IF Quantiles (Hz)")
    ax.set_title("C. IF Q-Q Plot")

    # Mean IF per trial
    ax = axes[1, 0]
    real_trial_if = [np.mean(compute_inst_freq(tgt_real[i, 0], fs)) for i in range(n_samples)]
    gen_trial_if = [np.mean(compute_inst_freq(tgt_gen[i, 0], fs)) for i in range(n_samples)]
    ax.scatter(real_trial_if, gen_trial_if, alpha=0.5, s=30)
    r = np.corrcoef(real_trial_if, gen_trial_if)[0, 1]
    ax.set_xlabel("Real Mean IF (Hz)")
    ax.set_ylabel("Generated Mean IF (Hz)")
    ax.set_title(f"D. Trial-wise Mean IF (r={r:.3f})")

    # IF variance
    ax = axes[1, 1]
    real_var = [np.var(compute_inst_freq(tgt_real[i, 0], fs)) for i in range(n_samples)]
    gen_var = [np.var(compute_inst_freq(tgt_gen[i, 0], fs)) for i in range(n_samples)]
    ax.scatter(real_var, gen_var, alpha=0.5, s=30)
    r = np.corrcoef(real_var, gen_var)[0, 1]
    ax.set_xlabel("Real IF Variance")
    ax.set_ylabel("Generated IF Variance")
    ax.set_title(f"E. IF Variance (r={r:.3f})")

    # CDF
    ax = axes[1, 2]
    sorted_real = np.sort(tgt_real_if[tgt_real_if < 100])
    sorted_gen = np.sort(tgt_gen_if[tgt_gen_if < 100])
    ax.plot(sorted_real, np.linspace(0, 1, len(sorted_real)), color=COLORS["real"], label='Real', linewidth=2)
    ax.plot(sorted_gen, np.linspace(0, 1, len(sorted_gen)), color=COLORS["generated"], label='Generated', linewidth=2)
    ax.set_xlabel("Instantaneous Frequency (Hz)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("F. IF CDF")
    ax.legend()

    plt.tight_layout()
    save_figure(fig, output_dir, "34_instantaneous_frequency", formats)
    plt.close(fig)


def plot_envelope_dynamics(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 35: Analytic signal envelope dynamics comparison."""
    if formats is None:
        formats = ["png", "svg", "pdf"]

    labels = get_region_labels(signals)
    tgt = labels["target"]
    tgt_real, tgt_gen = signals["target_real"], signals["target_gen"]
    n_samples = min(50, tgt_real.shape[0])
    bands = {"Theta (4-8 Hz)": (4, 8), "Beta (12-30 Hz)": (12, 30), "Gamma (30-80 Hz)": (30, 80)}

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
        sos = butter(4, [low, high], btype='band', fs=fs, output='sos')
        real_envs, gen_envs = [], []

        for i in range(n_samples):
            real_envs.append(np.abs(hilbert(sosfilt(sos, tgt_real[i, 0]))))
            gen_envs.append(np.abs(hilbert(sosfilt(sos, tgt_gen[i, 0]))))

        real_envs, gen_envs = np.array(real_envs), np.array(gen_envs)
        t = np.arange(real_envs.shape[1]) / fs * 1000

        # Mean envelope trace
        ax = axes[band_idx, 0]
        ax.plot(t, np.mean(real_envs, axis=0), color=COLORS["real"], label='Real', linewidth=1.5)
        ax.fill_between(t, np.mean(real_envs, axis=0) - np.std(real_envs, axis=0)/np.sqrt(n_samples),
                       np.mean(real_envs, axis=0) + np.std(real_envs, axis=0)/np.sqrt(n_samples),
                       color=COLORS["real"], alpha=0.3)
        ax.plot(t, np.mean(gen_envs, axis=0), color=COLORS["generated"], label='Generated', linewidth=1.5)
        ax.fill_between(t, np.mean(gen_envs, axis=0) - np.std(gen_envs, axis=0)/np.sqrt(n_samples),
                       np.mean(gen_envs, axis=0) + np.std(gen_envs, axis=0)/np.sqrt(n_samples),
                       color=COLORS["generated"], alpha=0.3)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Envelope")
        ax.set_title(f"{band_name} Mean Envelope")
        ax.legend(fontsize=8)

        # Autocorrelation
        ax = axes[band_idx, 1]
        max_lag = int(0.5 * fs)
        real_acorrs, gen_acorrs = [], []
        for i in range(n_samples):
            r_norm = (real_envs[i] - np.mean(real_envs[i])) / (np.std(real_envs[i]) + 1e-8)
            g_norm = (gen_envs[i] - np.mean(gen_envs[i])) / (np.std(gen_envs[i]) + 1e-8)
            r_acorr = np.correlate(r_norm, r_norm, mode='full')[len(r_norm)-1:len(r_norm)-1+max_lag] / len(r_norm)
            g_acorr = np.correlate(g_norm, g_norm, mode='full')[len(g_norm)-1:len(g_norm)-1+max_lag] / len(g_norm)
            real_acorrs.append(r_acorr)
            gen_acorrs.append(g_acorr)

        lags = np.arange(max_lag) / fs * 1000
        ax.plot(lags, np.mean(real_acorrs, axis=0), color=COLORS["real"], label='Real', linewidth=2)
        ax.plot(lags, np.mean(gen_acorrs, axis=0), color=COLORS["generated"], label='Generated', linewidth=2)
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"{band_name} Envelope Autocorr")
        ax.legend(fontsize=8)

        # Envelope correlation
        ax = axes[band_idx, 2]
        env_corrs = [np.corrcoef(real_envs[i], gen_envs[i])[0, 1] for i in range(n_samples)]
        ax.hist(env_corrs, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(np.mean(env_corrs), color='red', linestyle='--', linewidth=2, label=f'μ={np.mean(env_corrs):.3f}')
        ax.set_xlabel("Correlation (Real vs Gen)")
        ax.set_ylabel("Count")
        ax.set_title(f"{band_name} Envelope Corr")
        ax.legend()
        t_stat, p_val = stats.ttest_1samp(env_corrs, 0)
        ax.text(0.02, 0.98, f"t={t_stat:.2f}, p={p_val:.2e} {significance_stars(p_val)}",
               transform=ax.transAxes, fontsize=8, va='top', bbox=dict(facecolor='white', alpha=0.8))

    fig.suptitle("Envelope Dynamics Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, "35_envelope_dynamics", formats)
    plt.close(fig)


def plot_complex_plane(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 36: Complex plane (Argand diagram) trajectory comparison."""
    if formats is None:
        formats = ["png", "svg", "pdf"]

    labels = get_region_labels(signals)
    tgt = labels["target"]
    tgt_real, tgt_gen = signals["target_real"], signals["target_gen"]
    trial_idx, ch = 0, 0
    bands = {"Theta (4-8 Hz)": (4, 8), "Beta (12-30 Hz)": (12, 30), "Gamma (30-80 Hz)": (30, 80)}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
        sos = butter(4, [low, high], btype='band', fs=fs, output='sos')
        real_analytic = hilbert(sosfilt(sos, tgt_real[trial_idx, ch]))
        gen_analytic = hilbert(sosfilt(sos, tgt_gen[trial_idx, ch]))

        seg_len = int(0.5 * fs)
        start = int(1 * fs)
        real_seg = real_analytic[start:start + seg_len]
        gen_seg = gen_analytic[start:start + seg_len]

        # Trajectory
        ax = axes[0, band_idx]
        ax.plot(np.real(real_seg), np.imag(real_seg), color=COLORS["real"], alpha=0.7, linewidth=0.5, label='Real')
        ax.plot(np.real(gen_seg), np.imag(gen_seg), color=COLORS["generated"], alpha=0.7, linewidth=0.5, label='Gen')
        ax.scatter([np.real(real_seg[0])], [np.imag(real_seg[0])], color=COLORS["real"], s=50, marker='o', zorder=5)
        ax.scatter([np.real(gen_seg[0])], [np.imag(gen_seg[0])], color=COLORS["generated"], s=50, marker='o', zorder=5)
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        ax.set_title(f"{band_name} Trajectory")
        ax.legend(fontsize=8)
        ax.axis('equal')

        # Distribution difference
        ax = axes[1, band_idx]
        n_samples = min(30, tgt_real.shape[0])
        real_parts_r, imag_parts_r, real_parts_g, imag_parts_g = [], [], [], []
        for i in range(n_samples):
            r_a = hilbert(sosfilt(sos, tgt_real[i, ch]))
            g_a = hilbert(sosfilt(sos, tgt_gen[i, ch]))
            real_parts_r.extend(np.real(r_a))
            imag_parts_r.extend(np.imag(r_a))
            real_parts_g.extend(np.real(g_a))
            imag_parts_g.extend(np.imag(g_a))

        lim = np.percentile(np.abs(real_parts_r), 99)
        h_r, xe, ye = np.histogram2d(real_parts_r, imag_parts_r, bins=50, range=[[-lim, lim], [-lim, lim]], density=True)
        h_g, _, _ = np.histogram2d(real_parts_g, imag_parts_g, bins=50, range=[[-lim, lim], [-lim, lim]], density=True)
        diff = h_r - h_g
        im = ax.imshow(diff.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, ax=ax, label='Density Diff')
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        ax.set_title(f"{band_name} Distribution Diff")

        ks_r, p_r = stats.ks_2samp(real_parts_r[::100], real_parts_g[::100])
        ks_i, p_i = stats.ks_2samp(imag_parts_r[::100], imag_parts_g[::100])
        ax.text(0.02, 0.98, f"Real: KS={ks_r:.3f}\nImag: KS={ks_i:.3f}", transform=ax.transAxes, fontsize=7, va='top',
               bbox=dict(facecolor='white', alpha=0.8))

    fig.suptitle("Complex Plane Analysis: Analytic Signal", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, "36_complex_plane", formats)
    plt.close(fig)


# =============================================================================
# Phase-Amplitude Coupling (PAC) Analysis
# =============================================================================

def compute_pac_mi(signal: np.ndarray, fs: float, phase_freq: Tuple[float, float],
                   amp_freq: Tuple[float, float], n_bins: int = 18) -> float:
    """Compute Phase-Amplitude Coupling using Modulation Index (Tort et al. 2010)."""
    sos_phase = butter(4, phase_freq, btype='band', fs=fs, output='sos')
    phase = np.angle(hilbert(sosfilt(sos_phase, signal)))

    sos_amp = butter(4, amp_freq, btype='band', fs=fs, output='sos')
    amplitude = np.abs(hilbert(sosfilt(sos_amp, signal)))

    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_indices = np.clip(np.digitize(phase, phase_bins) - 1, 0, n_bins - 1)

    mean_amp = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            mean_amp[i] = np.mean(amplitude[mask])

    if np.sum(mean_amp) > 0:
        p = mean_amp / np.sum(mean_amp) + 1e-10
        mi = np.sum(p * np.log(p / (1/n_bins))) / np.log(n_bins)
    else:
        mi = 0.0
    return mi


def compute_pac_comodulogram(signal: np.ndarray, fs: float, phase_freqs: np.ndarray,
                             amp_freqs: np.ndarray, phase_bw: float = 2.0, amp_bw: float = 10.0) -> np.ndarray:
    """Compute PAC comodulogram matrix."""
    comod = np.zeros((len(phase_freqs), len(amp_freqs)))
    for i, pf in enumerate(phase_freqs):
        for j, af in enumerate(amp_freqs):
            phase_band = (max(1, pf - phase_bw/2), pf + phase_bw/2)
            amp_band = (max(af - amp_bw/2, pf + phase_bw), af + amp_bw/2)
            if amp_band[0] < amp_band[1]:
                comod[i, j] = compute_pac_mi(signal, fs, phase_band, amp_band)
    return comod


def plot_pac_comodulogram(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 37: PAC Comodulogram comparison."""
    if formats is None:
        formats = ["png", "svg", "pdf"]

    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    tgt_real, tgt_gen = signals["target_real"], signals["target_gen"]
    src_real, src_gen = signals["source_real"], signals["source_gen"]

    phase_freqs = np.arange(4, 20, 2)
    amp_freqs = np.arange(30, 120, 5)
    n_samples = min(20, tgt_real.shape[0])

    def compute_avg_comod(data, n_samples):
        comod_sum = np.zeros((len(phase_freqs), len(amp_freqs)))
        count = 0
        for i in range(n_samples):
            for ch in range(min(4, data.shape[1])):
                comod_sum += compute_pac_comodulogram(data[i, ch], fs, phase_freqs, amp_freqs)
                count += 1
        return comod_sum / count if count > 0 else comod_sum

    print("  Computing PAC comodulograms...")
    tgt_real_comod = compute_avg_comod(tgt_real, n_samples)
    tgt_gen_comod = compute_avg_comod(tgt_gen, n_samples)
    src_real_comod = compute_avg_comod(src_real, n_samples)
    src_gen_comod = compute_avg_comod(src_gen, n_samples)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    vmax = max(tgt_real_comod.max(), tgt_gen_comod.max(), src_real_comod.max(), src_gen_comod.max())

    for idx, (data, title) in enumerate([(tgt_real_comod, f"A. {tgt} Real"), (tgt_gen_comod, f"B. {tgt} Gen"),
                                          (src_real_comod, f"E. {src} Real"), (src_gen_comod, f"F. {src} Gen")]):
        row, col = idx // 2, idx % 2
        if idx >= 2: row = 1
        ax = axes[row, col]
        im = ax.imshow(data.T, origin='lower', aspect='auto',
                      extent=[phase_freqs[0], phase_freqs[-1], amp_freqs[0], amp_freqs[-1]], cmap='hot', vmin=0, vmax=vmax)
        ax.set_xlabel("Phase Freq (Hz)")
        ax.set_ylabel("Amp Freq (Hz)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='MI')

    # Differences
    ax = axes[0, 2]
    diff_tgt = tgt_real_comod - tgt_gen_comod
    vmax_d = max(abs(diff_tgt.min()), abs(diff_tgt.max()))
    im = ax.imshow(diff_tgt.T, origin='lower', aspect='auto',
                  extent=[phase_freqs[0], phase_freqs[-1], amp_freqs[0], amp_freqs[-1]], cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
    ax.set_xlabel("Phase Freq (Hz)")
    ax.set_ylabel("Amp Freq (Hz)")
    ax.set_title(f"C. {tgt} Diff (Real-Gen)")
    plt.colorbar(im, ax=ax, label='ΔMI')

    ax = axes[1, 2]
    diff_src = src_real_comod - src_gen_comod
    vmax_d = max(abs(diff_src.min()), abs(diff_src.max()))
    im = ax.imshow(diff_src.T, origin='lower', aspect='auto',
                  extent=[phase_freqs[0], phase_freqs[-1], amp_freqs[0], amp_freqs[-1]], cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
    ax.set_xlabel("Phase Freq (Hz)")
    ax.set_ylabel("Amp Freq (Hz)")
    ax.set_title(f"G. {src} Diff (Real-Gen)")
    plt.colorbar(im, ax=ax, label='ΔMI')

    # Correlations
    ax = axes[0, 3]
    ax.scatter(tgt_real_comod.flatten(), tgt_gen_comod.flatten(), alpha=0.5, s=20)
    ax.plot([0, vmax], [0, vmax], 'k--')
    r = np.corrcoef(tgt_real_comod.flatten(), tgt_gen_comod.flatten())[0, 1]
    ax.set_xlabel("Real MI")
    ax.set_ylabel("Gen MI")
    ax.set_title(f"D. {tgt} PAC (r={r:.3f})")
    stats_tgt = comprehensive_stats(tgt_real_comod.flatten(), tgt_gen_comod.flatten(), n_bootstrap=500)
    add_stats_annotation(ax, stats_tgt, position="bottom_right", fontsize=7)

    ax = axes[1, 3]
    ax.scatter(src_real_comod.flatten(), src_gen_comod.flatten(), alpha=0.5, s=20)
    ax.plot([0, vmax], [0, vmax], 'k--')
    r = np.corrcoef(src_real_comod.flatten(), src_gen_comod.flatten())[0, 1]
    ax.set_xlabel("Real MI")
    ax.set_ylabel("Gen MI")
    ax.set_title(f"H. {src} PAC (r={r:.3f})")
    stats_src = comprehensive_stats(src_real_comod.flatten(), src_gen_comod.flatten(), n_bootstrap=500)
    add_stats_annotation(ax, stats_src, position="bottom_right", fontsize=7)

    fig.suptitle("Phase-Amplitude Coupling (PAC) Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, "37_pac_comodulogram", formats)
    plt.close(fig)


def plot_pac_modulation_index(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 38: PAC Modulation Index comparison across frequency pairs."""
    if formats is None:
        formats = ["png", "svg", "pdf"]

    labels = get_region_labels(signals)
    src, tgt = labels["source"], labels["target"]
    
    tgt_real, tgt_gen = signals["target_real"], signals["target_gen"]
    src_real, src_gen = signals["source_real"], signals["source_gen"]

    coupling_pairs = [
        ("Theta-LowGamma", (4, 8), (30, 50)), ("Theta-HighGamma", (4, 8), (50, 100)),
        ("Alpha-LowGamma", (8, 12), (30, 50)), ("Alpha-HighGamma", (8, 12), (50, 100)),
        ("Beta-LowGamma", (12, 20), (30, 50)), ("Beta-HighGamma", (12, 20), (50, 100)),
    ]

    n_samples = min(30, tgt_real.shape[0])
    n_channels = min(8, tgt_real.shape[1])

    results = {p[0]: {"tgt_real": [], "tgt_gen": [], "src_real": [], "src_gen": []} for p in coupling_pairs}

    print("  Computing PAC MI for coupling pairs...")
    for pair_name, phase_band, amp_band in coupling_pairs:
        for i in range(n_samples):
            for ch in range(n_channels):
                results[pair_name]["tgt_real"].append(compute_pac_mi(tgt_real[i, ch], fs, phase_band, amp_band))
                results[pair_name]["tgt_gen"].append(compute_pac_mi(tgt_gen[i, ch], fs, phase_band, amp_band))
                results[pair_name]["src_real"].append(compute_pac_mi(src_real[i, ch], fs, phase_band, amp_band))
                results[pair_name]["src_gen"].append(compute_pac_mi(src_gen[i, ch], fs, phase_band, amp_band))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (pair_name, _, _) in enumerate(coupling_pairs):
        ax = axes[idx]
        tgt_r, tgt_g = np.array(results[pair_name]["tgt_real"]), np.array(results[pair_name]["tgt_gen"])
        src_r, src_g = np.array(results[pair_name]["src_real"]), np.array(results[pair_name]["src_gen"])

        bp = ax.boxplot([tgt_r, tgt_g, src_r, src_g], positions=[0, 0.8, 2, 2.8], widths=0.6, patch_artist=True)
        for patch, color in zip(bp['boxes'], [COLORS["real"], COLORS["generated"], COLORS["real"], COLORS["generated"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks([0.4, 2.4])
        ax.set_xticklabels([tgt, src])
        ax.set_ylabel("Modulation Index")
        ax.set_title(pair_name)

        _, p_tgt = stats.ttest_rel(tgt_r, tgt_g)
        _, p_src = stats.ttest_rel(src_r, src_g)
        y_max = max(tgt_r.max(), tgt_g.max(), src_r.max(), src_g.max())
        ax.plot([0, 0.8], [y_max * 1.05, y_max * 1.05], 'k-', linewidth=1)
        ax.text(0.4, y_max * 1.07, significance_stars(p_tgt), ha='center', fontsize=10)
        ax.plot([2, 2.8], [y_max * 1.05, y_max * 1.05], 'k-', linewidth=1)
        ax.text(2.4, y_max * 1.07, significance_stars(p_src), ha='center', fontsize=10)

        d_tgt = compute_effect_size(tgt_r, tgt_g)["effect_size"]
        d_src = compute_effect_size(src_r, src_g)["effect_size"]
        ax.text(0.02, 0.02, f"{tgt} d={d_tgt:.2f}\n{src} d={d_src:.2f}", transform=ax.transAxes, fontsize=8, va='bottom',
               bbox=dict(facecolor='white', alpha=0.8))

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS["real"], alpha=0.7, label='Real'),
                      Patch(facecolor=COLORS["generated"], alpha=0.7, label='Generated')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("PAC Modulation Index by Frequency Pair", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, "38_pac_modulation_index", formats)
    plt.close(fig)


# =============================================================================
# New Gallery and Per-Session Plots (39-44)
# =============================================================================

def plot_per_odor_signal_gallery(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
    n_examples_per_odor: int = 3,
):
    """Plot 39: Per-odor signal gallery showing real vs generated for each condition.

    Shows example waveforms for each odor/condition, making it easy to see how well
    the model captures condition-specific neural dynamics.
    """
    if formats is None:
        formats = ["png", "svg", "pdf"]

    labels = get_region_labels(signals)
    tgt = labels["target"]

    # Get unique odors
    odor_names = np.array(signals["odor_names"])
    unique_odors = np.unique(odor_names)
    n_odors = len(unique_odors)

    # Compute time axis
    T = signals["target_real"].shape[-1]
    time = np.arange(T) / fs + 0.5  # Offset by 0.5s (crop start)

    # Create figure: rows=odors, cols=examples
    fig, axes = plt.subplots(n_odors, n_examples_per_odor, figsize=(4*n_examples_per_odor, 3*n_odors))
    if n_odors == 1:
        axes = axes.reshape(1, -1)

    rng = np.random.RandomState(42)

    for i, odor in enumerate(unique_odors):
        mask = odor_names == odor
        odor_indices = np.where(mask)[0]

        # Sample examples for this odor
        n_avail = len(odor_indices)
        n_show = min(n_examples_per_odor, n_avail)
        selected = rng.choice(odor_indices, n_show, replace=False)

        for j, idx in enumerate(selected):
            ax = axes[i, j]

            # Use first channel for clarity
            real = signals["target_real"][idx, 0, :]
            gen = signals["target_gen"][idx, 0, :]
            corr = np.corrcoef(real, gen)[0, 1]

            ax.plot(time, real, color=COLORS["real"], alpha=0.8, linewidth=1, label="Real")
            ax.plot(time, gen, color=COLORS["generated"], alpha=0.8, linewidth=1, label="Gen")
            ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.3, label="Odor onset")

            ax.set_title(f"r={corr:.3f}", fontsize=10)

            if j == 0:
                ax.set_ylabel(f"{odor}\nAmplitude", fontsize=9)
            if i == n_odors - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper right")

            ax.tick_params(labelsize=8)

        # Fill empty columns if fewer examples
        for j in range(n_show, n_examples_per_odor):
            axes[i, j].axis('off')

    plt.suptitle(f"Per-Odor Signal Gallery: Real vs Generated {tgt}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "39_per_odor_signal_gallery", formats)
    plt.close(fig)


def plot_success_gallery(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
    n_examples: int = 16,
):
    """Plot 40: Success gallery - trials with highest correlation.

    Shows the best-performing trials to understand what the model does well.
    """
    if formats is None:
        formats = ["png", "svg", "pdf"]

    labels = get_region_labels(signals)
    tgt = labels["target"]

    # Compute per-trial correlations (using first channel)
    n_trials = signals["target_real"].shape[0]
    correlations = np.array([
        np.corrcoef(signals["target_real"][i, 0, :], signals["target_gen"][i, 0, :])[0, 1]
        for i in range(n_trials)
    ])

    # Get top N trials
    top_indices = np.argsort(correlations)[-n_examples:][::-1]  # Descending

    # Time axis
    T = signals["target_real"].shape[-1]
    time = np.arange(T) / fs + 0.5

    # Create grid
    n_rows = 4
    n_cols = (n_examples + n_rows - 1) // n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten()

    odor_names = np.array(signals["odor_names"])

    for i, idx in enumerate(top_indices):
        if i >= len(axes):
            break
        ax = axes[i]

        real = signals["target_real"][idx, 0, :]
        gen = signals["target_gen"][idx, 0, :]
        corr = correlations[idx]
        odor = odor_names[idx]

        ax.plot(time, real, color=COLORS["real"], alpha=0.8, linewidth=1, label="Real")
        ax.plot(time, gen, color=COLORS["generated"], alpha=0.8, linewidth=1, label="Gen")
        ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.3)

        ax.set_title(f"#{i+1} Trial {idx} ({odor})\nr={corr:.4f}", fontsize=10)

        if i >= len(axes) - n_cols:
            ax.set_xlabel("Time (s)", fontsize=9)
        if i % n_cols == 0:
            ax.set_ylabel("Amplitude", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

        ax.tick_params(labelsize=8)
        # Add green border for success
        for spine in ax.spines.values():
            spine.set_edgecolor('#2ecc71')
            spine.set_linewidth(2)

    # Hide unused axes
    for i in range(len(top_indices), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f"SUCCESS GALLERY: Top {n_examples} Best {tgt} Reconstructions",
                 fontsize=14, fontweight="bold", color="#2ecc71")
    plt.tight_layout()
    save_figure(fig, output_dir, "40_success_gallery", formats)
    plt.close(fig)


def plot_failure_gallery(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
    n_examples: int = 16,
):
    """Plot 41: Failure gallery - trials with lowest correlation.

    Shows the worst-performing trials to diagnose failure modes.
    """
    if formats is None:
        formats = ["png", "svg", "pdf"]

    labels = get_region_labels(signals)
    tgt = labels["target"]

    # Compute per-trial correlations (using first channel)
    n_trials = signals["target_real"].shape[0]
    correlations = np.array([
        np.corrcoef(signals["target_real"][i, 0, :], signals["target_gen"][i, 0, :])[0, 1]
        for i in range(n_trials)
    ])

    # Get bottom N trials
    bottom_indices = np.argsort(correlations)[:n_examples]  # Ascending

    # Time axis
    T = signals["target_real"].shape[-1]
    time = np.arange(T) / fs + 0.5

    # Create grid
    n_rows = 4
    n_cols = (n_examples + n_rows - 1) // n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten()

    odor_names = np.array(signals["odor_names"])

    for i, idx in enumerate(bottom_indices):
        if i >= len(axes):
            break
        ax = axes[i]

        real = signals["target_real"][idx, 0, :]
        gen = signals["target_gen"][idx, 0, :]
        corr = correlations[idx]
        odor = odor_names[idx]

        ax.plot(time, real, color=COLORS["real"], alpha=0.8, linewidth=1, label="Real")
        ax.plot(time, gen, color=COLORS["generated"], alpha=0.8, linewidth=1, label="Gen")
        ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.3)

        ax.set_title(f"#{i+1} Trial {idx} ({odor})\nr={corr:.4f}", fontsize=10)

        if i >= len(axes) - n_cols:
            ax.set_xlabel("Time (s)", fontsize=9)
        if i % n_cols == 0:
            ax.set_ylabel("Amplitude", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

        ax.tick_params(labelsize=8)
        # Add red border for failure
        for spine in ax.spines.values():
            spine.set_edgecolor('#e74c3c')
            spine.set_linewidth(2)

    # Hide unused axes
    for i in range(len(bottom_indices), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f"FAILURE GALLERY: Bottom {n_examples} Worst {tgt} Reconstructions",
                 fontsize=14, fontweight="bold", color="#e74c3c")
    plt.tight_layout()
    save_figure(fig, output_dir, "41_failure_gallery", formats)
    plt.close(fig)


def plot_per_session_metrics(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 42: Per-session performance metrics summary.

    Shows how model performance varies across recording sessions.
    Requires session_ids in signals dict.
    """
    if formats is None:
        formats = ["png", "svg", "pdf"]

    # Check if session info is available
    if "session_ids" not in signals:
        print("  Skipping per-session metrics (no session_ids in signals)")
        return

    labels = get_region_labels(signals)
    tgt = labels["target"]

    session_ids = signals["session_ids"]
    unique_sessions = np.unique(session_ids)
    n_sessions = len(unique_sessions)

    # Get session names if available
    idx_to_session = signals.get("idx_to_session", {})

    # Compute per-session metrics
    session_names = []
    corr_means = []
    corr_stds = []
    r2_means = []
    mse_means = []
    n_trials_list = []

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        n_trials = mask.sum()
        n_trials_list.append(n_trials)

        sess_name = idx_to_session.get(sess_id, f"Session {sess_id}")
        session_names.append(sess_name)

        # Compute correlations for this session
        correlations = np.array([
            np.corrcoef(signals["target_real"][i, 0, :], signals["target_gen"][i, 0, :])[0, 1]
            for i in np.where(mask)[0]
        ])
        corr_means.append(np.mean(correlations))
        corr_stds.append(np.std(correlations))

        # Compute R2
        real_flat = signals["target_real"][mask].flatten()
        gen_flat = signals["target_gen"][mask].flatten()
        ss_res = np.sum((real_flat - gen_flat) ** 2)
        ss_tot = np.sum((real_flat - real_flat.mean()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        r2_means.append(r2)

        # Compute MSE
        mse = np.mean((signals["target_real"][mask] - signals["target_gen"][mask]) ** 2)
        mse_means.append(mse)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(n_sessions)
    width = 0.6

    # Plot 1: Correlation by session
    ax = axes[0, 0]
    bars = ax.bar(x, corr_means, width, yerr=corr_stds, capsize=4,
                  color=COLORS["generated"], alpha=0.8, edgecolor='black')
    ax.axhline(np.mean(corr_means), color='red', linestyle='--', label=f'Mean: {np.mean(corr_means):.3f}')
    ax.set_xticks(x)
    ax.set_xticklabels(session_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Correlation", fontsize=11)
    ax.set_title("Correlation by Session", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim([0, 1])

    # Color bars by performance
    for bar, val in zip(bars, corr_means):
        if val > np.mean(corr_means):
            bar.set_color('#2ecc71')  # Green for above average
        else:
            bar.set_color('#e74c3c')  # Red for below average

    # Plot 2: R² by session
    ax = axes[0, 1]
    bars = ax.bar(x, r2_means, width, color=COLORS["forward"], alpha=0.8, edgecolor='black')
    ax.axhline(np.mean(r2_means), color='red', linestyle='--', label=f'Mean: {np.mean(r2_means):.3f}')
    ax.set_xticks(x)
    ax.set_xticklabels(session_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("R² by Session", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Plot 3: MSE by session
    ax = axes[1, 0]
    bars = ax.bar(x, mse_means, width, color=COLORS["reverse"], alpha=0.8, edgecolor='black')
    ax.axhline(np.mean(mse_means), color='red', linestyle='--', label=f'Mean: {np.mean(mse_means):.4f}')
    ax.set_xticks(x)
    ax.set_xticklabels(session_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("MSE", fontsize=11)
    ax.set_title("MSE by Session", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Plot 4: Trial count by session
    ax = axes[1, 1]
    bars = ax.bar(x, n_trials_list, width, color='steelblue', alpha=0.8, edgecolor='black')
    ax.axhline(np.mean(n_trials_list), color='red', linestyle='--', label=f'Mean: {np.mean(n_trials_list):.0f}')
    ax.set_xticks(x)
    ax.set_xticklabels(session_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Number of Trials", fontsize=11)
    ax.set_title("Trial Count by Session", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle(f"Per-Session Performance Summary: {tgt}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "42_per_session_metrics", formats)
    plt.close(fig)


def plot_per_session_signal_examples(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
    n_examples_per_session: int = 2,
):
    """Plot 43: Signal examples from each session.

    Shows representative examples from each recording session.
    """
    if formats is None:
        formats = ["png", "svg", "pdf"]

    if "session_ids" not in signals:
        print("  Skipping per-session signal examples (no session_ids)")
        return

    labels = get_region_labels(signals)
    tgt = labels["target"]

    session_ids = signals["session_ids"]
    unique_sessions = np.unique(session_ids)
    n_sessions = len(unique_sessions)
    idx_to_session = signals.get("idx_to_session", {})

    T = signals["target_real"].shape[-1]
    time = np.arange(T) / fs + 0.5

    rng = np.random.RandomState(42)

    # Create figure
    fig, axes = plt.subplots(n_sessions, n_examples_per_session,
                             figsize=(5*n_examples_per_session, 2.5*n_sessions))
    if n_sessions == 1:
        axes = axes.reshape(1, -1)

    for i, sess_id in enumerate(unique_sessions):
        mask = session_ids == sess_id
        sess_indices = np.where(mask)[0]
        sess_name = idx_to_session.get(sess_id, f"Session {sess_id}")

        # Compute correlations for this session to show representative examples
        correlations = np.array([
            np.corrcoef(signals["target_real"][idx, 0, :], signals["target_gen"][idx, 0, :])[0, 1]
            for idx in sess_indices
        ])

        # Select examples: median performers (representative)
        sorted_idx = np.argsort(correlations)
        mid = len(sorted_idx) // 2
        selected = sess_indices[sorted_idx[max(0, mid-1):mid+n_examples_per_session-1]]

        for j, idx in enumerate(selected[:n_examples_per_session]):
            ax = axes[i, j]

            real = signals["target_real"][idx, 0, :]
            gen = signals["target_gen"][idx, 0, :]
            corr = np.corrcoef(real, gen)[0, 1]

            ax.plot(time, real, color=COLORS["real"], alpha=0.8, linewidth=1, label="Real")
            ax.plot(time, gen, color=COLORS["generated"], alpha=0.8, linewidth=1, label="Gen")
            ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.3)

            ax.set_title(f"Trial {idx} (r={corr:.3f})", fontsize=10)

            if j == 0:
                ax.set_ylabel(f"{sess_name}\nAmplitude", fontsize=9)
            if i == n_sessions - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper right")

            ax.tick_params(labelsize=8)

    plt.suptitle(f"Per-Session Signal Examples: {tgt}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "43_per_session_signals", formats)
    plt.close(fig)


def plot_correlation_distribution_by_odor(
    signals: Dict[str, np.ndarray],
    output_dir: Path,
    formats: List[str] = None,
    fs: float = SAMPLING_RATE_HZ,
):
    """Plot 44: Correlation distribution violin plot by odor.

    Shows the distribution of per-trial correlations for each odor/condition.
    """
    if formats is None:
        formats = ["png", "svg", "pdf"]

    labels = get_region_labels(signals)
    tgt = labels["target"]
    condition_name = labels.get("condition", "Odor")

    odor_names = np.array(signals["odor_names"])
    unique_odors = np.unique(odor_names)
    n_odors = len(unique_odors)

    # Compute per-trial correlations
    n_trials = signals["target_real"].shape[0]
    correlations = np.array([
        np.corrcoef(signals["target_real"][i, 0, :], signals["target_gen"][i, 0, :])[0, 1]
        for i in range(n_trials)
    ])

    # Organize by odor
    odor_correlations = {odor: correlations[odor_names == odor] for odor in unique_odors}

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Violin plot
    ax = axes[0]
    positions = np.arange(n_odors)
    violin_data = [odor_correlations[odor] for odor in unique_odors]

    parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS["generated"])
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('red')

    ax.set_xticks(positions)
    ax.set_xticklabels(unique_odors, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Correlation", fontsize=11)
    ax.set_xlabel(condition_name, fontsize=11)
    ax.set_title(f"Correlation Distribution by {condition_name}", fontsize=12, fontweight="bold")
    ax.axhline(np.mean(correlations), color='gray', linestyle='--', alpha=0.5, label=f'Overall mean: {np.mean(correlations):.3f}')
    ax.legend(fontsize=9)
    ax.set_ylim([min(0, correlations.min()-0.1), 1.05])

    # Plot 2: Box plot with individual points
    ax = axes[1]
    bp = ax.boxplot(violin_data, positions=positions, patch_artist=True, widths=0.6)
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS["forward"])
        patch.set_alpha(0.7)

    # Add individual points with jitter
    for i, odor in enumerate(unique_odors):
        y = odor_correlations[odor]
        x = np.random.normal(i, 0.1, size=len(y))
        ax.scatter(x, y, alpha=0.3, color='black', s=10)

    ax.set_xticks(positions)
    ax.set_xticklabels(unique_odors, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Correlation", fontsize=11)
    ax.set_xlabel(condition_name, fontsize=11)
    ax.set_title(f"Correlation Box Plot by {condition_name}", fontsize=12, fontweight="bold")

    # Add mean annotation for each odor
    means = [np.mean(odor_correlations[odor]) for odor in unique_odors]
    for i, m in enumerate(means):
        ax.annotate(f'{m:.3f}', xy=(i, m), xytext=(5, 0), textcoords='offset points',
                   fontsize=8, color='red', fontweight='bold')

    plt.suptitle(f"Per-{condition_name} Correlation Analysis: {tgt}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "44_correlation_by_odor", formats)
    plt.close(fig)


# =============================================================================
# Training Integration API
# =============================================================================

def generate_training_plots(
    model_fwd: nn.Module,
    model_rev: nn.Module,
    spectral_shift_fwd: nn.Module,
    spectral_shift_rev: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    vocab: Dict[str, int],
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    session_ids: Optional[np.ndarray] = None,
    idx_to_session: Optional[Dict[int, str]] = None,
    formats: List[str] = None,
    quick: bool = True,
):
    """Generate validation plots after training.

    This is the main API for generating plots from train.py at the end of training.

    Args:
        model_fwd: Forward translation model (already on device)
        model_rev: Reverse translation model (already on device)
        spectral_shift_fwd: SpectralShift for forward output
        spectral_shift_rev: SpectralShift for reverse output
        dataloader: DataLoader for validation/test data
        device: Torch device
        vocab: Label vocabulary (name → id)
        output_dir: Directory to save plots
        config: Model/dataset config with 'source_name', 'target_name', 'sampling_rate'
        session_ids: Optional array of session IDs for each sample
        idx_to_session: Optional mapping from session ID to session name
        formats: Output formats (default: png only for training speed)
        quick: If True, only generate essential plots (galleries, per-odor, per-session)

    Returns:
        Dict with summary metrics
    """
    if formats is None:
        formats = ["png"]  # Quick mode only saves PNG

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("GENERATING VALIDATION PLOTS")
    print("="*60)

    # Generate signals
    print("Generating signals from validation data...")
    signals = generate_signals(
        model_fwd, model_rev,
        spectral_shift_fwd, spectral_shift_rev,
        dataloader, device, vocab,
        config=config,
    )

    # Add session info if provided
    if session_ids is not None:
        signals["session_ids"] = session_ids
    if idx_to_session is not None:
        signals["idx_to_session"] = idx_to_session

    print(f"  Generated {signals['target_gen'].shape[0]} samples")

    fs = config.get("sampling_rate", SAMPLING_RATE_HZ) if config else SAMPLING_RATE_HZ

    # Define which plots to generate
    if quick:
        # Essential plots only
        plot_functions = [
            ("39_per_odor_signal_gallery", plot_per_odor_signal_gallery),
            ("40_success_gallery", plot_success_gallery),
            ("41_failure_gallery", plot_failure_gallery),
            ("42_per_session_metrics", plot_per_session_metrics),
            ("43_per_session_signals", plot_per_session_signal_examples),
            ("44_correlation_by_odor", plot_correlation_distribution_by_odor),
            ("16_example_trials", plot_example_trials),
            ("06_correlation_histogram", plot_correlation_histogram),
        ]
    else:
        # All gallery and summary plots
        plot_functions = [
            ("01_psd_comparison", plot_psd_comparison),
            ("03_mean_waveform", plot_mean_waveform),
            ("06_correlation_histogram", plot_correlation_histogram),
            ("14_channel_performance", plot_channel_performance),
            ("16_example_trials", plot_example_trials),
            ("22_odor_metrics_heatmap", plot_odor_metrics_heatmap),
            ("39_per_odor_signal_gallery", plot_per_odor_signal_gallery),
            ("40_success_gallery", plot_success_gallery),
            ("41_failure_gallery", plot_failure_gallery),
            ("42_per_session_metrics", plot_per_session_metrics),
            ("43_per_session_signals", plot_per_session_signal_examples),
            ("44_correlation_by_odor", plot_correlation_distribution_by_odor),
        ]

    print(f"Generating {len(plot_functions)} plots...")

    import inspect
    for plot_name, plot_func in plot_functions:
        try:
            sig = inspect.signature(plot_func)
            if 'fs' in sig.parameters:
                plot_func(signals, output_dir, formats, fs=fs)
            else:
                plot_func(signals, output_dir, formats)
            print(f"  Saved: {plot_name}")
        except Exception as e:
            print(f"  ERROR in {plot_name}: {e}")

    # Compute summary metrics
    print("\nComputing summary metrics...")
    metrics = compute_summary_metrics(signals, fs=fs)

    metrics_path = output_dir / "training_summary_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    print(f"\nPlots saved to: {output_dir}")
    print("="*60)

    return metrics


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Validate neural signal translation model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/validation_plots",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        choices=["auto", "olfactory", "pfc"],
        help="Dataset to validate: 'auto' (detect from checkpoint), 'olfactory' (OB→PCx), or 'pfc' (PFC→CA1)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,svg,pdf",
        help="Comma-separated output formats (default: png,svg,pdf)"
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for CI estimation"
    )
    parser.add_argument(
        "--run_gradient_analysis",
        action="store_true",
        help="Run gradient-based analysis (saliency, Grad-CAM, neuroscience metrics)"
    )
    parser.add_argument(
        "--no_gradient_analysis",
        action="store_true",
        help="Explicitly disable gradient analysis (overrides --run_gradient_analysis)"
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip generating standard plots (useful with --run_gradient_analysis to run only gradient analysis)"
    )
    parser.add_argument(
        "--only_gradient_analysis",
        action="store_true",
        help="Run ONLY gradient analysis, skip all standard plots and metrics (shortcut for --skip_plots --run_gradient_analysis)"
    )
    parser.add_argument(
        "--save_numeric_results",
        action="store_true",
        default=True,
        help="Save all numeric results to JSON and HDF5 (default: True)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file for validation settings"
    )
    return parser.parse_args()


# Default validation config (can be overridden by --config file)
DEFAULT_VALIDATION_CONFIG = {
    # Gradient analysis settings
    "enable_gradient_analysis": True,  # Master switch for gradient-based analysis
    "run_saliency": True,              # Input saliency maps
    "run_gradcam": True,               # Grad-CAM at all levels
    "run_importance": True,            # Channel/temporal/frequency importance
    "run_neuroscience": True,          # PAC, coherence, ERP, bursts, waves

    # Output settings
    "save_numeric_results": True,      # Save all numeric results
    "save_hdf5": True,                 # Save to HDF5 format
    "save_json": True,                 # Save to JSON format
    "save_csv": True,                  # Save tabular results to CSV

    # Analysis parameters
    "n_samples_for_analysis": 16,      # Number of samples for gradient analysis
    "temporal_window_size": 500,       # Window size for temporal importance (samples)
}


def main():
    args = parse_args()

    # Load config from file or use defaults
    config = DEFAULT_VALIDATION_CONFIG.copy()
    if args.config is not None:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                user_config = json.load(f)
                config.update(user_config)
            print(f"Loaded config from: {config_path}")

    # Command-line overrides
    if args.run_gradient_analysis:
        config["enable_gradient_analysis"] = True
    if args.no_gradient_analysis:
        config["enable_gradient_analysis"] = False
    if not args.save_numeric_results:
        config["save_numeric_results"] = False

    # Setup
    setup_plot_style()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse formats
    formats = [f.strip() for f in args.formats.split(",")]
    print(f"Output formats: {formats}")
    
    # Determine dataset type - can auto-detect from checkpoint
    if args.dataset == "auto":
        # Pre-scan checkpoint to determine dataset
        print(f"\nAuto-detecting dataset from checkpoint...")
        checkpoint_preview = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        
        # Check for saved config first
        if "config" in checkpoint_preview and checkpoint_preview["config"]:
            ckpt_cfg = checkpoint_preview["config"]
            in_ch = ckpt_cfg.get("in_channels", 32)
            out_ch = ckpt_cfg.get("out_channels", 32)
        else:
            # Infer from weights
            model_state = checkpoint_preview.get("model", {})
            if "inc.conv.0.branches.0.conv.weight" in model_state:
                in_ch = model_state["inc.conv.0.branches.0.conv.weight"].shape[1]
            else:
                in_ch = 32
            if "outc.conv.weight" in model_state:
                out_ch = model_state["outc.conv.weight"].shape[0]
            else:
                out_ch = 32
                
        del checkpoint_preview  # Free memory
        
        # Determine dataset type based on channels
        if in_ch == 64 and out_ch == 32:
            dataset_type = DatasetType.PFC
            print(f"  Detected PFC dataset ({in_ch}→{out_ch}ch)")
        else:
            dataset_type = DatasetType.OLFACTORY
            print(f"  Detected olfactory dataset ({in_ch}→{out_ch}ch)")
    else:
        dataset_type = DatasetType.PFC if args.dataset == "pfc" else DatasetType.OLFACTORY
    
    ds_config = get_dataset_config(dataset_type)

    print(f"Device: {device}")
    print(f"Dataset: {dataset_type.value} ({ds_config['source_name']}→{ds_config['target_name']}, {ds_config['in_channels']}→{ds_config['out_channels']}ch)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Split: {args.split}")
    print(f"Gradient analysis: {'ENABLED' if config['enable_gradient_analysis'] else 'DISABLED'}")

    # Load checkpoint to get split info (use same validation/test sets as training)
    print("\nLoading checkpoint for split info...")
    checkpoint_full = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_split_info = checkpoint_full.get("split_info", {})

    # Load data based on dataset type
    print("Loading data...")
    if dataset_type == DatasetType.PFC:
        data = prepare_pfc_data()
    else:
        data = prepare_data()

    # Check if checkpoint has the split indices we need
    split_key = f"{args.split}_idx"
    if split_key in checkpoint_split_info:
        # Use the SAME indices that were used during training!
        split_indices = np.array(checkpoint_split_info[split_key])

        # Handle empty split (e.g., test split when no_test_set=True)
        if len(split_indices) == 0:
            # Try to fall back to val split if test is empty
            if args.split == "test" and "val_idx" in checkpoint_split_info:
                print(f"  WARNING: test split has 0 samples (no_test_set mode was used)")
                print(f"  Automatically falling back to validation split...")
                split_indices = np.array(checkpoint_split_info["val_idx"])
                args.split = "val"  # Update for downstream code
                print(f"  Using val indices from checkpoint ({len(split_indices)} samples)")
            else:
                raise ValueError(
                    f"Split '{args.split}' has 0 samples in checkpoint. "
                    f"Available splits: {[k.replace('_idx', '') for k in checkpoint_split_info.keys() if k.endswith('_idx')]}. "
                    f"Try using --split val"
                )
        else:
            print(f"  Using {args.split} indices from checkpoint ({len(split_indices)} samples)")

        # Print session info if available
        if "val_sessions" in checkpoint_split_info and args.split == "val":
            print(f"  Validation sessions: {checkpoint_split_info['val_sessions']}")
        if "test_sessions" in checkpoint_split_info and args.split == "test":
            print(f"  Test sessions: {checkpoint_split_info['test_sessions']}")

        # Create dataloader with checkpoint indices
        dataloader = create_single_session_dataloader(
            data, f"checkpoint_{args.split}", split_indices,
            batch_size=args.batch_size, num_workers=4, distributed=False
        )
    else:
        # Fallback: checkpoint doesn't have split info, use new splits
        print(f"  WARNING: Checkpoint has no {split_key}, creating new random splits!")
        print(f"  (This means validation is NOT on the same data as training)")
        if dataset_type == DatasetType.PFC:
            loaders = create_pfc_dataloaders(data, batch_size=args.batch_size, num_workers=4)
        else:
            loaders = create_dataloaders(data, batch_size=args.batch_size, num_workers=4)
        dataloader = loaders[args.split]

    del checkpoint_full  # Free memory
    
    # Get number of conditions from data (handle different key names)
    n_conditions = data.get("n_odors", data.get("n_labels", data.get("n_conditions", ds_config["n_conditions"])))
    sampling_rate = data.get("sampling_rate", ds_config["sampling_rate"])
    print(f"  Loaded {len(dataloader.dataset)} samples, {n_conditions} conditions, {sampling_rate}Hz")

    # Load models
    print("Loading models...")
    model_fwd, model_rev, spectral_shift_fwd, spectral_shift_rev, model_config = load_checkpoint(
        Path(args.checkpoint), device, dataset_type=dataset_type, n_conditions=n_conditions
    )
    print(f"  Forward: {model_config['in_channels']}→{model_config['out_channels']}ch")
    print(f"  Reverse: {model_config['out_channels']}→{model_config['in_channels']}ch")

    # Generate signals
    print("Generating signals...")
    signals = generate_signals(
        model_fwd, model_rev,
        spectral_shift_fwd, spectral_shift_rev,
        dataloader, device, data["vocab"],
        config=model_config,
    )

    print(f"Generated {signals['target_gen'].shape[0]} samples")
    print(f"Source signal shape: {signals['source_real'].shape}")
    print(f"Target signal shape: {signals['target_real'].shape}")

    # Add session info to signals if available (for per-session plots)
    if "session_ids" in data:
        # Use split_indices from checkpoint (already computed above)
        split_key = f"{args.split}_idx"
        if split_key in checkpoint_split_info:
            # Use checkpoint indices
            split_indices = np.array(checkpoint_split_info[split_key])
        elif split_key in data:
            # Fallback to data indices
            split_indices = data[split_key]
        else:
            split_indices = None

        if split_indices is not None:
            signals["session_ids"] = data["session_ids"][split_indices]
            signals["idx_to_session"] = data.get("idx_to_session", {})
            print(f"  Added session info for {len(np.unique(signals['session_ids']))} sessions")

    # Handle --only_gradient_analysis shortcut
    if args.only_gradient_analysis:
        args.skip_plots = True
        args.run_gradient_analysis = True
        config["enable_gradient_analysis"] = True

    # Generate all plots (44 total) - skip if --skip_plots
    if args.skip_plots:
        print("\nSkipping standard plots (--skip_plots enabled)")
    else:
        print("\nGenerating plots (44 total)...")

    # Define all plot functions (name without extension, function)
    plot_functions = [
        # Original 15 plots
        ("01_psd_comparison", plot_psd_comparison),
        ("02_psd_per_odor", plot_psd_per_odor),
        ("03_mean_waveform", plot_mean_waveform),
        ("04_waveform_per_odor", plot_waveform_per_odor),
        ("05_psd_distribution", plot_psd_distribution),
        ("06_correlation_histogram", plot_correlation_histogram),
        ("07_spectrograms", plot_spectrograms),
        ("08_channel_correlation", plot_channel_correlation),
        ("09_band_power", plot_band_power),
        ("10_phase_coherence", plot_phase_coherence),
        ("11_error_spectrum", plot_error_spectrum),
        ("12_temporal_correlation", plot_temporal_correlation),
        ("13_amplitude_qq", plot_amplitude_qq),
        ("14_channel_performance", plot_channel_performance),
        ("15_error_heatmap", plot_error_heatmap),
        # Nature Methods quality plots (16-25)
        ("16_example_trials", plot_example_trials),
        ("17_cycle_consistency", plot_cycle_consistency),
        ("18_statistical_summary", plot_statistical_summary),
        ("19_bland_altman", plot_bland_altman),
        ("20_coherence_spectrum", plot_coherence_spectrum),
        ("21_residual_analysis", plot_residual_analysis),
        ("22_odor_metrics_heatmap", plot_odor_metrics_heatmap),
        ("23_bootstrap_ci", plot_bootstrap_ci),
        ("24_power_ratio", plot_power_ratio),
        ("25_cross_correlation", plot_cross_correlation),
        # Cross-region comparison plots (26-31)
        ("26_cross_region_correlation", plot_cross_region_correlation),
        ("27_cross_region_coherence", plot_cross_region_coherence),
        ("28_cross_region_plv", plot_cross_region_plv),
        ("29_cross_region_psd_ratio", plot_cross_region_psd_ratio),
        ("30_cross_region_xcorr", plot_cross_region_xcorr),
        ("31_cross_region_band_power", plot_cross_region_band_power),
        # Analytic signal (Real/Imaginary) analysis plots (32-36)
        ("32_analytic_magnitude", plot_analytic_magnitude),
        ("33_phase_distribution", plot_phase_distribution),
        ("34_instantaneous_frequency", plot_instantaneous_frequency),
        ("35_envelope_dynamics", plot_envelope_dynamics),
        ("36_complex_plane", plot_complex_plane),
        # Phase-Amplitude Coupling (PAC) plots (37-38)
        ("37_pac_comodulogram", plot_pac_comodulogram),
        ("38_pac_modulation_index", plot_pac_modulation_index),
        # Gallery and per-session plots (39-44)
        ("39_per_odor_signal_gallery", plot_per_odor_signal_gallery),
        ("40_success_gallery", plot_success_gallery),
        ("41_failure_gallery", plot_failure_gallery),
        ("42_per_session_metrics", plot_per_session_metrics),
        ("43_per_session_signals", plot_per_session_signal_examples),
        ("44_correlation_by_odor", plot_correlation_distribution_by_odor),
    ]

    # Skip plots if --skip_plots or --only_gradient_analysis
    if not args.skip_plots:
        # Get sampling rate from signals for proper spectral analysis
        fs = signals.get("sampling_rate", SAMPLING_RATE_HZ)
        
        for plot_name, plot_func in tqdm(plot_functions, desc="Generating plots"):
            try:
                # Call plot function - it handles saving internally with save_figure
                # Pass fs for functions that need sampling rate
                import inspect
                sig = inspect.signature(plot_func)
                if 'fs' in sig.parameters:
                    plot_func(signals, output_dir, formats, fs=fs)
                else:
                    plot_func(signals, output_dir, formats)
                print(f"  Saved: {plot_name}")
            except Exception as e:
                print(f"  ERROR in {plot_name}: {e}")
                traceback.print_exc()

        # Compute and save summary metrics
        print("\nComputing summary metrics...")
        metrics = compute_summary_metrics(signals, fs=fs)

        metrics_path = output_dir / "summary_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

        # Compute and save statistical tests
        print("Computing statistical tests...")
        statistical_tests = compute_statistical_tests(signals, n_bootstrap=args.n_bootstrap, fs=fs)

        stats_path = output_dir / "statistical_tests.json"
        with open(stats_path, "w") as f:
            json.dump(statistical_tests, f, indent=2)
        print(f"Saved statistical tests to: {stats_path}")

        # Compute and save bootstrap results
        print("Computing bootstrap confidence intervals...")
        bootstrap_results = compute_bootstrap_results(signals, n_bootstrap=args.n_bootstrap, fs=fs)

        bootstrap_path = output_dir / "bootstrap_results.json"
        with open(bootstrap_path, "w") as f:
            json.dump(bootstrap_results, f, indent=2)
        print(f"Saved bootstrap results to: {bootstrap_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        # Get dynamic keys from metrics
        fwd_key = metrics.get("_keys", {}).get("forward", "forward")
        rev_key = metrics.get("_keys", {}).get("reverse", "reverse")
        labels = get_region_labels(signals)

        print(f"\nForward ({labels['source']} → {labels['target']}):")
        print(f"  Correlation: {metrics[fwd_key]['correlation_mean']:.4f} ± {metrics[fwd_key]['correlation_std']:.4f}")
        print(f"  PSD Error: {metrics[fwd_key]['psd_error_db']:.2f} dB")
        print(f"  MSE: {metrics[fwd_key]['mse']:.6f}")

        print(f"\nReverse ({labels['target']} → {labels['source']}):")
        print(f"  Correlation: {metrics[rev_key]['correlation_mean']:.4f} ± {metrics[rev_key]['correlation_std']:.4f}")
        print(f"  PSD Error: {metrics[rev_key]['psd_error_db']:.2f} dB")
        print(f"  MSE: {metrics[rev_key]['mse']:.6f}")

        if 'statistical_tests' in statistical_tests:
            print("\nStatistical Tests:")
            st = statistical_tests['statistical_tests']
            if 'correlation_ttest_pvalue' in st:
                print(f"  Correlation t-test p-value: {st['correlation_ttest_pvalue']:.4e}")
            if 'cohens_d_correlation' in st:
                print(f"  Cohen's d (correlation): {st['cohens_d_correlation']:.3f}")

        print(f"\nOutput structure:")
        for fmt in formats:
            fmt_dir = output_dir / fmt
            if fmt_dir.exists():
                n_files = len(list(fmt_dir.glob(f"*.{fmt}")))
                print(f"  {fmt}/: {n_files} files")
        print(f"  summary_metrics.json")
        print(f"  statistical_tests.json")
        print(f"  bootstrap_results.json")
        print("=" * 60)

    # Run gradient-based analysis if enabled in config
    if config["enable_gradient_analysis"]:
        if not RECORDING_AVAILABLE:
            print("\nWARNING: Recording module not available. Skipping gradient analysis.")
        else:
            print("\n" + "=" * 60)
            print("GRADIENT-BASED ANALYSIS (Post-hoc)")
            print("=" * 60)
            run_gradient_analysis(
                model_fwd=model_fwd,
                model_rev=model_rev,
                spectral_shift_fwd=spectral_shift_fwd,
                spectral_shift_rev=spectral_shift_rev,
                dataloader=dataloader,
                device=device,
                output_dir=output_dir,
                formats=formats,
                vocab=data["vocab"],
                config=config,
            )
    else:
        print("\nGradient analysis: SKIPPED (disabled in config)")


def run_gradient_analysis(
    model_fwd: CondUNet1D,
    model_rev: CondUNet1D,
    spectral_shift_fwd: Optional[nn.Module],
    spectral_shift_rev: Optional[nn.Module],
    dataloader,
    device: torch.device,
    output_dir: Path,
    formats: List[str],
    vocab: Dict[str, int],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run comprehensive gradient-based analysis for post-hoc interpretability.

    This function performs:
    1. Input saliency analysis (which input features drive outputs)
    2. Grad-CAM at all encoder/decoder levels
    3. Channel/temporal/frequency importance analysis
    4. Neuroscience-specific metrics (PAC gradients, coherence importance, etc.)

    Args:
        model_fwd: Forward translation model (OB → PCx)
        model_rev: Reverse translation model (PCx → OB)
        spectral_shift_fwd: Forward spectral shift block
        spectral_shift_rev: Reverse spectral shift block
        dataloader: Data loader for analysis samples
        device: Computation device
        output_dir: Directory to save results
        formats: Output formats for plots
        vocab: Odor vocabulary mapping
        config: Configuration dictionary controlling analysis options

    Returns:
        Dictionary containing all analysis results
    """
    if not RECORDING_AVAILABLE:
        print("Recording module not available. Skipping gradient analysis.")
        return {}

    # Use default config if not provided
    if config is None:
        config = DEFAULT_VALIDATION_CONFIG.copy()

    # Create output directory for gradient analysis
    gradient_dir = output_dir / "gradient_analysis"
    gradient_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzers
    print("\nInitializing gradient analyzers...")
    saliency_analyzer = SaliencyAnalyzer(fs=SAMPLING_RATE_HZ, n_channels=32)
    importance_analyzer = ImportanceAnalyzer(fs=SAMPLING_RATE_HZ, n_channels=32)
    neuro_analyzer = NeuroInterpretabilityAnalyzer(fs=SAMPLING_RATE_HZ, n_channels=32)
    visualizer = NeuroVisualizer(output_dir=gradient_dir)

    # Collect a batch of samples for analysis
    print("Collecting samples for analysis...")
    batch = next(iter(dataloader))
    source_real, target_real, condition_ids = batch[0].to(device), batch[1].to(device), batch[2].to(device)

    # Use configured number of samples for detailed analysis
    n_samples = min(config.get("n_samples_for_analysis", 16), source_real.shape[0])
    source_real = source_real[:n_samples]
    target_real = target_real[:n_samples]
    condition_ids = condition_ids[:n_samples]

    # Initialize results structure
    results = {
        "forward": {},
        "reverse": {},
        "neuroscience": {},
        "config": config.copy(),  # Store config for reproducibility
        "metadata": {
            "n_samples": n_samples,
            "device": str(device),
        },
    }

    # Initialize variables for optional analyses
    fwd_saliency = None
    fwd_gradcam = None
    fwd_channel_importance = None
    fwd_temporal_importance = None
    fwd_freq_importance = None
    rev_saliency = None
    rev_gradcam = None
    rev_channel_importance = None
    rev_temporal_importance = None
    rev_freq_importance = None

    # ==========================================================================
    # Forward model analysis (Source → Target)
    # ==========================================================================
    print("\n--- Forward Model (Source → Target) Analysis ---")

    # 1. Input Saliency
    if config.get("run_saliency", True):
        print("  Computing input saliency...")
        fwd_saliency = saliency_analyzer.compute_input_saliency(
            model=model_fwd,
            input_ob=source_real,
            odor_ids=condition_ids,
            target_pcx=target_real,
        )
        results["forward"]["input_saliency"] = fwd_saliency
    else:
        print("  Skipping input saliency (disabled)")

    # 2. Grad-CAM at all levels
    if config.get("run_gradcam", True):
        print("  Computing Grad-CAM at all encoder/decoder levels...")
        fwd_gradcam = saliency_analyzer.compute_gradcam_all_levels(
            model=model_fwd,
            input_ob=source_real,
            odor_ids=condition_ids,
        )
        results["forward"]["gradcam"] = fwd_gradcam
    else:
        print("  Skipping Grad-CAM (disabled)")

    # 3-5. Importance analysis (requires saliency to be computed first)
    if config.get("run_importance", True) and fwd_saliency is not None:
        # Channel importance
        print("  Computing channel importance...")
        fwd_channel_importance = importance_analyzer.compute_channel_importance(
            saliency=fwd_saliency,
        )
        results["forward"]["channel_importance"] = fwd_channel_importance

        # Temporal importance
        print("  Computing temporal importance...")
        window_ms = config.get("temporal_window_ms", 100.0)
        fwd_temporal_importance, fwd_temporal_centers = importance_analyzer.compute_temporal_importance(
            saliency=fwd_saliency,
            window_ms=window_ms,
        )
        results["forward"]["temporal_importance"] = fwd_temporal_importance
        results["forward"]["temporal_centers_ms"] = fwd_temporal_centers

        # Frequency importance
        print("  Computing frequency band importance...")
        fwd_freq_importance = importance_analyzer.compute_frequency_importance(
            input_signal=source_real,
            saliency=fwd_saliency,
        )
        results["forward"]["frequency_importance"] = fwd_freq_importance
    elif config.get("run_importance", True):
        print("  Skipping importance analysis (saliency not computed)")
    else:
        print("  Skipping importance analysis (disabled)")

    # ==========================================================================
    # Reverse model analysis (Target → Source)
    # ==========================================================================
    print("\n--- Reverse Model (Target → Source) Analysis ---")

    # 1. Input Saliency
    if config.get("run_saliency", True):
        print("  Computing input saliency...")
        rev_saliency = saliency_analyzer.compute_input_saliency(
            model=model_rev,
            input_ob=target_real,
            odor_ids=condition_ids,
            target_pcx=source_real,
        )
        results["reverse"]["input_saliency"] = rev_saliency
    else:
        print("  Skipping input saliency (disabled)")

    # 2. Grad-CAM at all levels
    if config.get("run_gradcam", True):
        print("  Computing Grad-CAM at all encoder/decoder levels...")
        rev_gradcam = saliency_analyzer.compute_gradcam_all_levels(
            model=model_rev,
            input_ob=target_real,
            odor_ids=condition_ids,
        )
        results["reverse"]["gradcam"] = rev_gradcam
    else:
        print("  Skipping Grad-CAM (disabled)")

    # 3-5. Importance analysis (requires saliency to be computed first)
    if config.get("run_importance", True) and rev_saliency is not None:
        # Channel importance
        print("  Computing channel importance...")
        rev_channel_importance = importance_analyzer.compute_channel_importance(
            saliency=rev_saliency,
        )
        results["reverse"]["channel_importance"] = rev_channel_importance

        # Temporal importance
        print("  Computing temporal importance...")
        window_ms = config.get("temporal_window_ms", 100.0)
        rev_temporal_importance, rev_temporal_centers = importance_analyzer.compute_temporal_importance(
            saliency=rev_saliency,
            window_ms=window_ms,
        )
        results["reverse"]["temporal_importance"] = rev_temporal_importance
        results["reverse"]["temporal_centers_ms"] = rev_temporal_centers

        # Frequency importance
        print("  Computing frequency band importance...")
        rev_freq_importance = importance_analyzer.compute_frequency_importance(
            input_signal=target_real,
            saliency=rev_saliency,
        )
        results["reverse"]["frequency_importance"] = rev_freq_importance
    elif config.get("run_importance", True):
        print("  Skipping importance analysis (saliency not computed)")
    else:
        print("  Skipping importance analysis (disabled)")

    # ==========================================================================
    # Neuroscience-specific analysis
    # ==========================================================================
    target_gen = None
    source_gen = None

    if config.get("run_neuroscience", True):
        print("\n--- Neuroscience Interpretability Analysis ---")

        # Generate translations for neuroscience analysis
        model_fwd.eval()
        model_rev.eval()
        with torch.no_grad():
            # Apply per-channel normalization (same as training)
            source_norm = per_channel_normalize(source_real)
            target_norm = per_channel_normalize(target_real)
            target_gen = model_fwd(source_norm, condition_ids)
            source_gen = model_rev(target_norm, condition_ids)
            if spectral_shift_fwd is not None:
                target_gen = spectral_shift_fwd(target_gen, condition_ids)
            if spectral_shift_rev is not None:
                source_gen = spectral_shift_rev(source_gen, condition_ids)

        # 1. Phase-Amplitude Coupling gradients
        print("  Computing PAC gradient importance...")
        try:
            pac_results = neuro_analyzer.compute_pac_gradients(
                model=model_fwd,
                input_ob=source_real,
                odor_ids=condition_ids,
                target_pcx=target_real,
            )
            results["neuroscience"]["pac_gradients"] = pac_results
        except Exception as e:
            print(f"    Warning: PAC gradient computation failed: {e}")

        # 2. Cross-electrode coherence importance
        print("  Computing coherence importance...")
        try:
            coherence_results = neuro_analyzer.compute_coherence_importance(
                model=model_fwd,
                input_ob=source_real,
                odor_ids=condition_ids,
                target_pcx=target_real,
            )
            results["neuroscience"]["coherence_importance"] = coherence_results
        except Exception as e:
            print(f"    Warning: Coherence importance computation failed: {e}")

        # 3. ERP component saliency
        print("  Computing ERP saliency...")
        try:
            erp_results = neuro_analyzer.compute_erp_saliency(
                model=model_fwd,
                input_ob=source_real,
                odor_ids=condition_ids,
                target_pcx=target_real,
            )
            results["neuroscience"]["erp_saliency"] = erp_results
        except Exception as e:
            print(f"    Warning: ERP saliency computation failed: {e}")

        # 4. Burst importance
        print("  Computing oscillation burst importance...")
        try:
            burst_results = neuro_analyzer.compute_burst_importance(
                model=model_fwd,
                input_ob=source_real,
                odor_ids=condition_ids,
                target_pcx=target_real,
            )
            results["neuroscience"]["burst_importance"] = burst_results
        except Exception as e:
            print(f"    Warning: Burst importance computation failed: {e}")

        # 5. Traveling wave analysis
        print("  Computing traveling wave analysis...")
        try:
            wave_results = neuro_analyzer.compute_traveling_wave_analysis(
                model=model_fwd,
                input_ob=source_real,
                odor_ids=condition_ids,
                target_pcx=target_real,
            )
            results["neuroscience"]["traveling_waves"] = wave_results
        except Exception as e:
            print(f"    Warning: Traveling wave analysis failed: {e}")
    else:
        print("\n--- Neuroscience Analysis: SKIPPED (disabled in config) ---")

    # ==========================================================================
    # Generate visualizations
    # ==========================================================================
    print("\n--- Generating Gradient Analysis Plots ---")

    # Collect condition names
    inv_vocab = {v: k for k, v in vocab.items()}
    condition_names = [inv_vocab.get(cid.item(), f"cond_{cid.item()}") for cid in condition_ids]

    try:
        # Saliency heatmaps
        print("  Generating saliency heatmaps...")
        if fwd_saliency is not None:
            # Average across batch, convert to numpy
            fwd_sal_np = fwd_saliency.mean(dim=0).cpu().numpy() if torch.is_tensor(fwd_saliency) else fwd_saliency.mean(axis=0)
            visualizer.plot_saliency_heatmap(
                saliency=fwd_sal_np,
                title="Forward Model (OB→PCx) Input Saliency",
                save_name="fwd_input_saliency",
                fs=SAMPLING_RATE_HZ,
            )
        if rev_saliency is not None:
            rev_sal_np = rev_saliency.mean(dim=0).cpu().numpy() if torch.is_tensor(rev_saliency) else rev_saliency.mean(axis=0)
            visualizer.plot_saliency_heatmap(
                saliency=rev_sal_np,
                title="Reverse Model (PCx→OB) Input Saliency",
                save_name="rev_input_saliency",
                fs=SAMPLING_RATE_HZ,
            )

        # Grad-CAM hierarchy
        print("  Generating Grad-CAM hierarchy plots...")
        if fwd_gradcam:
            # Convert tensors to numpy
            fwd_gradcam_np = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in fwd_gradcam.items()}
            visualizer.plot_gradcam_hierarchy(
                gradcam_dict=fwd_gradcam_np,
                title="Forward Model Grad-CAM Hierarchy",
                save_name="fwd_gradcam_hierarchy",
            )
        if rev_gradcam:
            rev_gradcam_np = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in rev_gradcam.items()}
            visualizer.plot_gradcam_hierarchy(
                gradcam_dict=rev_gradcam_np,
                title="Reverse Model Grad-CAM Hierarchy",
                save_name="rev_gradcam_hierarchy",
            )

        # Channel importance
        print("  Generating channel importance plots...")
        if fwd_channel_importance is not None:
            fwd_ch_np = fwd_channel_importance.cpu().numpy() if torch.is_tensor(fwd_channel_importance) else fwd_channel_importance
            visualizer.plot_channel_importance(
                channel_importance=fwd_ch_np,
                title="Forward Model Channel Importance",
                save_name="fwd_channel_importance",
            )
        if rev_channel_importance is not None:
            rev_ch_np = rev_channel_importance.cpu().numpy() if torch.is_tensor(rev_channel_importance) else rev_channel_importance
            visualizer.plot_channel_importance(
                channel_importance=rev_ch_np,
                title="Reverse Model Channel Importance",
                save_name="rev_channel_importance",
            )

        # Frequency importance
        print("  Generating frequency importance plots...")
        if fwd_freq_importance is not None:
            visualizer.plot_frequency_importance(
                freq_importance=fwd_freq_importance,
                title="Forward Model Frequency Band Importance",
                save_name="fwd_frequency_importance",
            )
        if rev_freq_importance is not None:
            visualizer.plot_frequency_importance(
                freq_importance=rev_freq_importance,
                title="Reverse Model Frequency Band Importance",
                save_name="rev_frequency_importance",
            )

        # Temporal importance
        print("  Generating temporal importance plots...")
        if fwd_temporal_importance is not None:
            fwd_temp_np = fwd_temporal_importance.cpu().numpy() if torch.is_tensor(fwd_temporal_importance) else fwd_temporal_importance
            fwd_centers = results["forward"].get("temporal_centers_ms", np.arange(len(fwd_temp_np)) * 100)
            visualizer.plot_temporal_importance(
                temporal_importance=fwd_temp_np,
                window_centers_ms=fwd_centers,
                title="Forward Model Temporal Importance",
                save_name="fwd_temporal_importance",
            )
        if rev_temporal_importance is not None:
            rev_temp_np = rev_temporal_importance.cpu().numpy() if torch.is_tensor(rev_temporal_importance) else rev_temporal_importance
            rev_centers = results["reverse"].get("temporal_centers_ms", np.arange(len(rev_temp_np)) * 100)
            visualizer.plot_temporal_importance(
                temporal_importance=rev_temp_np,
                window_centers_ms=rev_centers,
                title="Reverse Model Temporal Importance",
                save_name="rev_temporal_importance",
            )

        # Neuroscience-specific plots
        if "pac_gradients" in results["neuroscience"]:
            print("  Generating PAC analysis plots...")
            visualizer.plot_pac_analysis(
                pac_results=results["neuroscience"]["pac_gradients"],
                title="Phase-Amplitude Coupling Gradient Importance",
                save_name="pac_gradient_analysis",
            )

        if "coherence_importance" in results["neuroscience"]:
            print("  Generating coherence importance plots...")
            visualizer.plot_coherence_matrix(
                coherence_results=results["neuroscience"]["coherence_importance"],
                title="Cross-Electrode Coherence Importance",
                save_name="coherence_importance",
            )

        if "erp_saliency" in results["neuroscience"]:
            print("  Generating ERP saliency plots...")
            visualizer.plot_erp_components(
                erp_results=results["neuroscience"]["erp_saliency"],
                title="ERP Component Saliency",
                save_name="erp_saliency",
            )

    except Exception as e:
        print(f"  Warning: Visualization failed: {e}")
        traceback.print_exc()

    # ==========================================================================
    # Save numeric results (controlled by config)
    # ==========================================================================
    if not config.get("save_numeric_results", True):
        print("\n  Numeric result saving: SKIPPED (disabled in config)")
    else:
        print("\n  Saving gradient analysis numeric results...")

        # Helper function to convert tensors to numpy for saving
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        # Helper function to recursively convert dict values
        def convert_dict_to_numpy(d):
            result = {}
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.detach().cpu().numpy()
                elif isinstance(v, dict):
                    result[k] = convert_dict_to_numpy(v)
                elif isinstance(v, (list, tuple)):
                    result[k] = [to_numpy(x) if isinstance(x, torch.Tensor) else x for x in v]
                else:
                    result[k] = v
            return result

        # ==========================================================================
        # Save to HDF5 (comprehensive array storage)
        # ==========================================================================
        if config.get("save_hdf5", True):
            try:
                import h5py

                results_path = gradient_dir / "gradient_analysis_results.h5"
                with h5py.File(results_path, "w") as f:
                    # Save metadata
                    meta_grp = f.create_group("metadata")
                    meta_grp.attrs["n_samples"] = n_samples
                    meta_grp.attrs["device"] = str(device)
                    meta_grp.create_dataset("condition_ids", data=condition_ids.cpu().numpy())

                    # Save config for reproducibility
                    config_grp = f.create_group("config")
                    for k, v in config.items():
                        if isinstance(v, (int, float, bool, str)):
                            config_grp.attrs[k] = v

                    # Save forward results
                    fwd_grp = f.create_group("forward")
                    if fwd_saliency is not None:
                        fwd_grp.create_dataset("input_saliency", data=to_numpy(fwd_saliency), compression="gzip")
                    if fwd_channel_importance is not None:
                        fwd_grp.create_dataset("channel_importance", data=to_numpy(fwd_channel_importance), compression="gzip")
                    if fwd_temporal_importance is not None:
                        fwd_grp.create_dataset("temporal_importance", data=to_numpy(fwd_temporal_importance), compression="gzip")
                    if fwd_freq_importance is not None and isinstance(fwd_freq_importance, dict):
                        freq_grp = fwd_grp.create_group("frequency_importance")
                        for band_name, importance in fwd_freq_importance.items():
                            if importance is not None:
                                # Scalars don't support compression
                                data = to_numpy(importance)
                                if np.isscalar(data) or data.ndim == 0:
                                    freq_grp.create_dataset(band_name, data=data)
                                else:
                                    freq_grp.create_dataset(band_name, data=data, compression="gzip")
                    if fwd_gradcam is not None:
                        gradcam_grp = fwd_grp.create_group("gradcam")
                        for layer_name, cam in fwd_gradcam.items():
                            if cam is not None:
                                gradcam_grp.create_dataset(layer_name, data=to_numpy(cam), compression="gzip")

                    # Save reverse results
                    rev_grp = f.create_group("reverse")
                    if rev_saliency is not None:
                        rev_grp.create_dataset("input_saliency", data=to_numpy(rev_saliency), compression="gzip")
                    if rev_channel_importance is not None:
                        rev_grp.create_dataset("channel_importance", data=to_numpy(rev_channel_importance), compression="gzip")
                    if rev_temporal_importance is not None:
                        rev_grp.create_dataset("temporal_importance", data=to_numpy(rev_temporal_importance), compression="gzip")
                    if rev_freq_importance is not None and isinstance(rev_freq_importance, dict):
                        freq_grp = rev_grp.create_group("frequency_importance")
                        for band_name, importance in rev_freq_importance.items():
                            if importance is not None:
                                # Scalars don't support compression
                                data = to_numpy(importance)
                                if np.isscalar(data) or data.ndim == 0:
                                    freq_grp.create_dataset(band_name, data=data)
                                else:
                                    freq_grp.create_dataset(band_name, data=data, compression="gzip")
                    if rev_gradcam is not None:
                        gradcam_grp = rev_grp.create_group("gradcam")
                        for layer_name, cam in rev_gradcam.items():
                            if cam is not None:
                                gradcam_grp.create_dataset(layer_name, data=to_numpy(cam), compression="gzip")

                    # Save neuroscience results
                    neuro_grp = f.create_group("neuroscience")
                    for key, value in results["neuroscience"].items():
                        if isinstance(value, dict):
                            sub_grp = neuro_grp.create_group(key)
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, (torch.Tensor, np.ndarray)):
                                    data = to_numpy(sub_value)
                                    if data.size > 0:
                                        sub_grp.create_dataset(sub_key, data=data, compression="gzip")
                                elif isinstance(sub_value, (int, float)):
                                    sub_grp.attrs[sub_key] = sub_value
                        elif isinstance(value, (torch.Tensor, np.ndarray)):
                            data = to_numpy(value)
                            if data.size > 0:
                                neuro_grp.create_dataset(key, data=data, compression="gzip")

                print(f"    HDF5 saved to: {results_path}")

            except Exception as e:
                print(f"    Warning: Could not save HDF5 results: {e}")
                traceback.print_exc()

        # ==========================================================================
        # Save comprehensive JSON with numeric summaries
        # ==========================================================================
        if config.get("save_json", True):
            try:
                # Compute summary statistics for all numeric results
                def compute_stats(arr):
                    """Compute summary stats for numpy array."""
                    if arr is None:
                        return None
                    # Handle scalar values
                    if isinstance(arr, (int, float)):
                        return {"value": float(arr)}
                    arr = to_numpy(arr)
                    if arr.size == 0:
                        return None
                    return {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "median": float(np.median(arr)),
                        "shape": list(arr.shape),
                    }

                summary = {
                    "metadata": {
                        "n_samples_analyzed": n_samples,
                        "condition_ids": condition_ids.cpu().tolist(),
                        "device": str(device),
                        "analysis_complete": True,
                    },
                    "config": {k: v for k, v in config.items() if isinstance(v, (int, float, bool, str))},
                    "forward": {
                        "input_saliency_stats": compute_stats(fwd_saliency),
                        "channel_importance_stats": compute_stats(fwd_channel_importance),
                        "temporal_importance_stats": compute_stats(fwd_temporal_importance),
                        # freq_importance is dict of floats, not arrays
                        "frequency_importance": fwd_freq_importance if fwd_freq_importance else None,
                        "gradcam_levels": list(fwd_gradcam.keys()) if fwd_gradcam else [],
                    },
                    "reverse": {
                        "input_saliency_stats": compute_stats(rev_saliency),
                        "channel_importance_stats": compute_stats(rev_channel_importance),
                        "temporal_importance_stats": compute_stats(rev_temporal_importance),
                        # freq_importance is dict of floats, not arrays
                        "frequency_importance": rev_freq_importance if rev_freq_importance else None,
                        "gradcam_levels": list(rev_gradcam.keys()) if rev_gradcam else [],
                    },
                    "neuroscience": {
                        "metrics_computed": list(results["neuroscience"].keys()),
                    },
                }

                # Add detailed neuroscience stats
                for metric_name, metric_data in results["neuroscience"].items():
                    if isinstance(metric_data, dict):
                        summary["neuroscience"][f"{metric_name}_stats"] = {
                            k: compute_stats(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                            for k, v in metric_data.items()
                            if isinstance(v, (torch.Tensor, np.ndarray, int, float))
                        }

                summary_path = gradient_dir / "gradient_analysis_summary.json"
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                print(f"    JSON summary saved to: {summary_path}")

            except Exception as e:
                print(f"    Warning: Could not save summary JSON: {e}")
                traceback.print_exc()

        # ==========================================================================
        # Save tabular results to CSV
        # ==========================================================================
        if config.get("save_csv", True):
            try:
                import csv

                # Channel importance CSV
                if fwd_channel_importance is not None or rev_channel_importance is not None:
                    csv_path = gradient_dir / "channel_importance.csv"
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["channel_idx", "forward_importance", "reverse_importance"])
                        n_channels = 32  # Assuming 32 channels
                        fwd_imp = to_numpy(fwd_channel_importance) if fwd_channel_importance is not None else [None] * n_channels
                        rev_imp = to_numpy(rev_channel_importance) if rev_channel_importance is not None else [None] * n_channels
                        for i in range(min(len(fwd_imp) if fwd_imp is not None else 0, n_channels)):
                            writer.writerow([
                                i,
                                float(fwd_imp[i]) if fwd_imp is not None and i < len(fwd_imp) else "",
                                float(rev_imp[i]) if rev_imp is not None and i < len(rev_imp) else "",
                            ])
                    print(f"    Channel importance CSV saved to: {csv_path}")

                # Frequency importance CSV
                if fwd_freq_importance is not None or rev_freq_importance is not None:
                    csv_path = gradient_dir / "frequency_importance.csv"
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["frequency_band", "forward_importance_mean", "forward_importance_std",
                                       "reverse_importance_mean", "reverse_importance_std"])
                        all_bands = set()
                        if fwd_freq_importance:
                            all_bands.update(fwd_freq_importance.keys())
                        if rev_freq_importance:
                            all_bands.update(rev_freq_importance.keys())
                        for band in sorted(all_bands):
                            fwd_val = to_numpy(fwd_freq_importance.get(band)) if fwd_freq_importance else None
                            rev_val = to_numpy(rev_freq_importance.get(band)) if rev_freq_importance else None
                            writer.writerow([
                                band,
                                float(np.mean(fwd_val)) if fwd_val is not None else "",
                                float(np.std(fwd_val)) if fwd_val is not None else "",
                                float(np.mean(rev_val)) if rev_val is not None else "",
                                float(np.std(rev_val)) if rev_val is not None else "",
                            ])
                    print(f"    Frequency importance CSV saved to: {csv_path}")

            except Exception as e:
                print(f"    Warning: Could not save CSV results: {e}")

    print("\n  Gradient analysis complete!")
    print(f"  Results saved to: {gradient_dir}")

    return results


if __name__ == "__main__":
    main()
