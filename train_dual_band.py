"""Training script for Dual-Band U-Net.

Usage:
    # Single GPU training
    python train_dual_band.py --epochs 80

    # Multi-GPU training with FSDP
    torchrun --nproc_per_node=8 train_dual_band.py --epochs 80 --fsdp

    # Continue from checkpoint
    python train_dual_band.py --resume artifacts/checkpoints_dual_band/best_model.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Local imports
from dual_band_unet import DualBandUNet, DualBandLoss
from data import prepare_data, create_dataloaders
from train import (
    set_seed,
    is_primary,
    get_world_size,
    barrier,
    per_channel_normalize,
    CHECKPOINT_DIR,
    SAMPLING_RATE_HZ,
)

# Constants
DUAL_BAND_CHECKPOINT_DIR = Path("artifacts/checkpoints_dual_band")
LOG_DIR = Path("artifacts/logs")


def create_dual_band_model(
    config: Dict[str, Any],
    n_odors: int = 7,
) -> DualBandUNet:
    """Create DualBandUNet model from config."""
    model = DualBandUNet(
        in_channels=config.get("in_channels", 32),
        out_channels=config.get("out_channels", 32),
        base_channels=config.get("base_channels", 64),
        n_odors=n_odors,
        emb_dim=config.get("emb_dim", 128),
        cutoff_freq=config.get("cutoff_freq", 30.0),
        sampling_rate=config.get("sampling_rate", 1000.0),
        low_band=tuple(config.get("low_band", [1.0, 30.0])),
        high_band=tuple(config.get("high_band", [30.0, 100.0])),
        n_downsample_low=config.get("n_downsample_low", 4),
        n_downsample_high=config.get("n_downsample_high", 2),
        use_differentiable_decomp=config.get("use_differentiable_decomp", True),
        dropout=config.get("dropout", 0.0),
        use_attention=config.get("use_attention", True),
        attention_type=config.get("attention_type", "cross_freq_v2"),
        norm_type=config.get("norm_type", "batch"),
        cond_mode=config.get("cond_mode", "cross_attn_gated"),
        conv_type=config.get("conv_type", "modern"),
        share_embedding=config.get("share_embedding", True),
        reconstruction_mode=config.get("reconstruction_mode", "sum"),
        use_se=config.get("use_se", True),
        conv_kernel_size=config.get("conv_kernel_size", 7),
        dilations=tuple(config.get("dilations", [1, 4, 16, 32])),
    )
    return model


def evaluate_dual_band(
    model: DualBandUNet,
    loader: torch.utils.data.DataLoader,
    loss_fn: DualBandLoss,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Returns:
        Dictionary with metrics including band-wise losses and correlations.
    """
    model.eval()
    total_losses = defaultdict(float)
    correlations = {"full": [], "low": [], "high": []}
    n_batches = 0

    with torch.no_grad():
        for ob_batch, pcx_batch, odor_batch in loader:
            ob_batch = ob_batch.to(device)
            pcx_batch = pcx_batch.to(device)
            odor_batch = odor_batch.to(device)

            # Normalize if needed
            if config.get("per_channel_norm", True):
                ob_batch = per_channel_normalize(ob_batch)
                pcx_batch = per_channel_normalize(pcx_batch)

            # Forward pass with band outputs
            pred, pred_low, pred_high = model(ob_batch, odor_batch, return_bands=True)

            # Decompose target
            target_low, target_high = model.decompose_target(pcx_batch)

            # Compute loss
            loss, loss_dict = loss_fn(
                pred, pcx_batch,
                pred_low, target_low,
                pred_high, target_high
            )

            # Accumulate losses
            for k, v in loss_dict.items():
                total_losses[k] += v
            n_batches += 1

            # Compute correlations (per-channel, averaged)
            def batch_correlation(a, b):
                """Compute mean correlation across batch and channels."""
                B, C, T = a.shape
                corrs = []
                for bi in range(B):
                    for ci in range(C):
                        a_flat = a[bi, ci].float().cpu().numpy()
                        b_flat = b[bi, ci].float().cpu().numpy()
                        if np.std(a_flat) > 1e-6 and np.std(b_flat) > 1e-6:
                            corr = np.corrcoef(a_flat, b_flat)[0, 1]
                            if not np.isnan(corr):
                                corrs.append(corr)
                return np.mean(corrs) if corrs else 0.0

            correlations["full"].append(batch_correlation(pred, pcx_batch))
            correlations["low"].append(batch_correlation(pred_low, target_low))
            correlations["high"].append(batch_correlation(pred_high, target_high))

    # Average metrics
    metrics = {k: v / n_batches for k, v in total_losses.items()}
    metrics["corr_full"] = np.mean(correlations["full"])
    metrics["corr_low"] = np.mean(correlations["low"])
    metrics["corr_high"] = np.mean(correlations["high"])

    return metrics


def compute_band_psd(
    model: DualBandUNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    n_batches: int = 10,
) -> Dict[str, np.ndarray]:
    """Compute average PSD for target and predictions in each band.

    Returns:
        Dictionary with 'freqs', 'target_*', 'pred_*' PSD arrays.
    """
    from scipy.signal import welch

    model.eval()
    sampling_rate = config.get("sampling_rate", 1000.0)

    all_psds = {
        "target_full": [], "pred_full": [],
        "target_low": [], "pred_low": [],
        "target_high": [], "pred_high": [],
    }

    with torch.no_grad():
        for i, (ob_batch, pcx_batch, odor_batch) in enumerate(loader):
            if i >= n_batches:
                break

            ob_batch = ob_batch.to(device)
            pcx_batch = pcx_batch.to(device)
            odor_batch = odor_batch.to(device)

            if config.get("per_channel_norm", True):
                ob_batch = per_channel_normalize(ob_batch)
                pcx_batch = per_channel_normalize(pcx_batch)

            pred, pred_low, pred_high = model(ob_batch, odor_batch, return_bands=True)
            target_low, target_high = model.decompose_target(pcx_batch)

            # Compute PSDs for each signal
            signals = {
                "target_full": pcx_batch,
                "pred_full": pred,
                "target_low": target_low,
                "pred_low": pred_low,
                "target_high": target_high,
                "pred_high": pred_high,
            }

            for name, sig in signals.items():
                sig_np = sig.float().cpu().numpy().reshape(-1, sig.shape[-1])
                for s in sig_np[:10]:  # Limit per batch
                    f, psd = welch(s, fs=sampling_rate, nperseg=min(1024, len(s)//4))
                    all_psds[name].append(psd)

    # Average PSDs
    results = {"freqs": f}
    for name in all_psds:
        results[name] = np.mean(all_psds[name], axis=0)

    return results


def plot_band_comparison(
    psd_data: Dict[str, np.ndarray],
    save_path: Path,
    low_band: Tuple[float, float] = (1, 30),
    high_band: Tuple[float, float] = (30, 100),
):
    """Create comparison plots for band-wise performance."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    freqs = psd_data["freqs"]
    max_freq = 120

    # Row 1: PSD comparisons
    # Full band
    axes[0, 0].semilogy(freqs, psd_data["target_full"], 'g-', label='Target', linewidth=2)
    axes[0, 0].semilogy(freqs, psd_data["pred_full"], 'b--', label='Prediction', linewidth=2)
    axes[0, 0].set_xlim([0, max_freq])
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('PSD')
    axes[0, 0].set_title('Full Band PSD')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Low band
    axes[0, 1].semilogy(freqs, psd_data["target_low"], 'g-', label='Target', linewidth=2)
    axes[0, 1].semilogy(freqs, psd_data["pred_low"], 'b--', label='Prediction', linewidth=2)
    axes[0, 1].axvline(low_band[0], color='r', linestyle=':', alpha=0.5)
    axes[0, 1].axvline(low_band[1], color='r', linestyle=':', alpha=0.5, label=f'Band: {low_band}Hz')
    axes[0, 1].set_xlim([0, max_freq])
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('PSD')
    axes[0, 1].set_title(f'Low Band ({low_band[0]}-{low_band[1]} Hz) PSD')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # High band
    axes[0, 2].semilogy(freqs, psd_data["target_high"], 'g-', label='Target', linewidth=2)
    axes[0, 2].semilogy(freqs, psd_data["pred_high"], 'b--', label='Prediction', linewidth=2)
    axes[0, 2].axvline(high_band[0], color='r', linestyle=':', alpha=0.5)
    axes[0, 2].axvline(high_band[1], color='r', linestyle=':', alpha=0.5, label=f'Band: {high_band}Hz')
    axes[0, 2].set_xlim([0, max_freq])
    axes[0, 2].set_xlabel('Frequency (Hz)')
    axes[0, 2].set_ylabel('PSD')
    axes[0, 2].set_title(f'High Band ({high_band[0]}-{high_band[1]} Hz) PSD')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: PSD differences in dB
    eps = 1e-10

    diff_full = 10 * np.log10((psd_data["pred_full"] + eps) / (psd_data["target_full"] + eps))
    axes[1, 0].plot(freqs, diff_full, 'b-', linewidth=2)
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].fill_between(freqs, -3, 3, alpha=0.2, color='green', label='±3dB')
    axes[1, 0].set_xlim([0, max_freq])
    axes[1, 0].set_ylim([-15, 15])
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('PSD Difference (dB)')
    axes[1, 0].set_title(f'Full Band Error (mean: {np.mean(np.abs(diff_full[freqs < max_freq])):.2f} dB)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    diff_low = 10 * np.log10((psd_data["pred_low"] + eps) / (psd_data["target_low"] + eps))
    axes[1, 1].plot(freqs, diff_low, 'b-', linewidth=2)
    axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].fill_between(freqs, -3, 3, alpha=0.2, color='green', label='±3dB')
    axes[1, 1].set_xlim([0, max_freq])
    axes[1, 1].set_ylim([-15, 15])
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('PSD Difference (dB)')
    low_mask = (freqs >= low_band[0]) & (freqs <= low_band[1])
    axes[1, 1].set_title(f'Low Band Error (in-band mean: {np.mean(np.abs(diff_low[low_mask])):.2f} dB)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    diff_high = 10 * np.log10((psd_data["pred_high"] + eps) / (psd_data["target_high"] + eps))
    axes[1, 2].plot(freqs, diff_high, 'b-', linewidth=2)
    axes[1, 2].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 2].fill_between(freqs, -3, 3, alpha=0.2, color='green', label='±3dB')
    axes[1, 2].set_xlim([0, max_freq])
    axes[1, 2].set_ylim([-15, 15])
    axes[1, 2].set_xlabel('Frequency (Hz)')
    axes[1, 2].set_ylabel('PSD Difference (dB)')
    high_mask = (freqs >= high_band[0]) & (freqs <= high_band[1])
    axes[1, 2].set_title(f'High Band Error (in-band mean: {np.mean(np.abs(diff_high[high_mask])):.2f} dB)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_dual_band(
    config: Dict[str, Any],
    data: Dict[str, Any],
    use_fsdp: bool = False,
    resume_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Main training function for Dual-Band U-Net."""

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    set_seed(config.get("seed", 42))
    is_distributed = get_world_size() > 1

    # Create dataloaders
    import multiprocessing
    num_cpus = multiprocessing.cpu_count()
    num_workers = min(8, num_cpus // max(1, get_world_size()))

    loaders = create_dataloaders(
        data,
        batch_size=config.get("batch_size", 8),
        num_workers=num_workers,
        distributed=is_distributed,
    )

    # Create model
    n_odors = data["n_odors"]
    model = create_dual_band_model(config, n_odors)

    if is_primary():
        n_params = sum(p.numel() for p in model.parameters())
        n_params_low = sum(p.numel() for p in model.unet_low.parameters())
        n_params_high = sum(p.numel() for p in model.unet_high.parameters())
        print(f"\nDualBandUNet created:")
        print(f"  Total parameters: {n_params:,}")
        print(f"  Low branch: {n_params_low:,}")
        print(f"  High branch: {n_params_high:,}")
        print(f"  Low band: {config.get('low_band', [1, 30])} Hz ({config.get('n_downsample_low', 4)} downsample)")
        print(f"  High band: {config.get('high_band', [30, 100])} Hz ({config.get('n_downsample_high', 2)} downsample)")

    # Move to device
    model = model.to(device)

    # FSDP wrapping
    if use_fsdp and is_distributed:
        if is_primary():
            print("Using FSDP")

        # Mixed precision
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Auto-wrap policy
        auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=1e6)

        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            mixed_precision=mp_policy,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
        )
        is_fsdp_wrapped = True
    elif is_distributed:
        model = DDP(model, device_ids=[local_rank])
        is_fsdp_wrapped = False
    else:
        is_fsdp_wrapped = False

    # Loss function
    loss_fn = DualBandLoss(
        base_loss=nn.L1Loss(),
        lambda_low=config.get("lambda_low", 0.5),
        lambda_high=config.get("lambda_high", 0.5),
        use_spectral_loss=config.get("use_spectral_loss", True),
        spectral_weight=config.get("spectral_weight", 1.0),
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 2e-4),
        betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
        weight_decay=config.get("weight_decay", 0.01),
    )

    # Scheduler
    num_epochs = config.get("num_epochs", 80)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_path is not None and resume_path.exists():
        if is_primary():
            print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        if is_fsdp_wrapped:
            # Need to use FSDP state dict loading
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    # Training loop
    DUAL_BAND_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [], "val_loss": [],
        "train_loss_low": [], "val_loss_low": [],
        "train_loss_high": [], "val_loss_high": [],
        "val_corr_full": [], "val_corr_low": [], "val_corr_high": [],
    }

    if is_primary():
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"  Batch size: {config.get('batch_size', 8)}")
        print(f"  Learning rate: {config.get('learning_rate', 2e-4)}")
        print(f"  Lambda low/high: {config.get('lambda_low', 0.5)}/{config.get('lambda_high', 0.5)}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_losses = defaultdict(float)
        n_batches = 0

        # Training
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{num_epochs}", disable=not is_primary())
        for ob_batch, pcx_batch, odor_batch in pbar:
            ob_batch = ob_batch.to(device)
            pcx_batch = pcx_batch.to(device)
            odor_batch = odor_batch.to(device)

            if config.get("per_channel_norm", True):
                ob_batch = per_channel_normalize(ob_batch)
                pcx_batch = per_channel_normalize(pcx_batch)

            optimizer.zero_grad()

            # Forward pass
            pred, pred_low, pred_high = model(ob_batch, odor_batch, return_bands=True)
            target_low, target_high = model.module.decompose_target(pcx_batch) if hasattr(model, 'module') else model.decompose_target(pcx_batch)

            # Compute loss
            loss, loss_dict = loss_fn(
                pred, pcx_batch,
                pred_low, target_low,
                pred_high, target_high
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accumulate
            for k, v in loss_dict.items():
                train_losses[k] += v
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss_dict['loss_total']:.4f}",
                "low": f"{loss_dict['loss_low']:.4f}",
                "high": f"{loss_dict['loss_high']:.4f}",
            })

        scheduler.step()

        # Average training losses
        for k in train_losses:
            train_losses[k] /= n_batches

        # Validation
        val_metrics = evaluate_dual_band(model, loaders["val"], loss_fn, device, config)

        # Log
        history["train_loss"].append(train_losses["loss_total"])
        history["val_loss"].append(val_metrics["loss_total"])
        history["train_loss_low"].append(train_losses["loss_low"])
        history["val_loss_low"].append(val_metrics["loss_low"])
        history["train_loss_high"].append(train_losses["loss_high"])
        history["val_loss_high"].append(val_metrics["loss_high"])
        history["val_corr_full"].append(val_metrics["corr_full"])
        history["val_corr_low"].append(val_metrics["corr_low"])
        history["val_corr_high"].append(val_metrics["corr_high"])

        if is_primary():
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train: loss={train_losses['loss_total']:.4f} (low={train_losses['loss_low']:.4f}, high={train_losses['loss_high']:.4f})")
            print(f"  Val:   loss={val_metrics['loss_total']:.4f} (low={val_metrics['loss_low']:.4f}, high={val_metrics['loss_high']:.4f})")
            print(f"  Corr:  full={val_metrics['corr_full']:.4f}, low={val_metrics['corr_low']:.4f}, high={val_metrics['corr_high']:.4f}")

        # Save best model
        val_loss = val_metrics["loss_total"]
        if val_loss < best_val_loss and is_primary():
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict() if not is_fsdp_wrapped else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config,
                "history": history,
            }
            torch.save(checkpoint, DUAL_BAND_CHECKPOINT_DIR / "best_model.pt")
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0 and is_primary():
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config,
                "history": history,
            }
            torch.save(checkpoint, DUAL_BAND_CHECKPOINT_DIR / f"checkpoint_epoch{epoch+1}.pt")

        barrier()

    # Final evaluation
    if is_primary():
        print("\n" + "="*70)
        print("Final Evaluation on Test Set")
        print("="*70)

    test_metrics = evaluate_dual_band(model, loaders["test"], loss_fn, device, config)

    if is_primary():
        print(f"Test Loss: {test_metrics['loss_total']:.4f}")
        print(f"  Low band: {test_metrics['loss_low']:.4f}")
        print(f"  High band: {test_metrics['loss_high']:.4f}")
        print(f"Test Correlation:")
        print(f"  Full: {test_metrics['corr_full']:.4f}")
        print(f"  Low:  {test_metrics['corr_low']:.4f}")
        print(f"  High: {test_metrics['corr_high']:.4f}")

        # Generate band comparison plots
        print("\nGenerating band comparison plots...")
        psd_data = compute_band_psd(model, loaders["test"], device, config)
        plot_band_comparison(
            psd_data,
            DUAL_BAND_CHECKPOINT_DIR / "band_comparison.png",
            low_band=tuple(config.get("low_band", [1, 30])),
            high_band=tuple(config.get("high_band", [30, 100])),
        )
        print(f"Saved: {DUAL_BAND_CHECKPOINT_DIR / 'band_comparison.png'}")

        # Save final results
        results = {
            "test_metrics": test_metrics,
            "history": history,
            "config": config,
        }
        with open(DUAL_BAND_CHECKPOINT_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    return {"test_metrics": test_metrics, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Band U-Net")
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channels")
    parser.add_argument("--cutoff", type=float, default=30.0, help="Frequency cutoff (Hz)")
    parser.add_argument("--lambda_low", type=float, default=0.5, help="Low band loss weight")
    parser.add_argument("--lambda_high", type=float, default=0.5, help="High band loss weight")
    parser.add_argument("--n_downsample_low", type=int, default=4, help="Downsample levels for low branch")
    parser.add_argument("--n_downsample_high", type=int, default=2, help="Downsample levels for high branch")
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Initialize distributed
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    # Config
    config = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "base_channels": args.base_channels,
        "cutoff_freq": args.cutoff,
        "lambda_low": args.lambda_low,
        "lambda_high": args.lambda_high,
        "n_downsample_low": args.n_downsample_low,
        "n_downsample_high": args.n_downsample_high,
        "low_band": [1.0, args.cutoff],
        "high_band": [args.cutoff, 100.0],
        "seed": args.seed,
        "in_channels": 32,
        "out_channels": 32,
        "sampling_rate": 1000.0,
        "per_channel_norm": True,
        "use_spectral_loss": True,
        "spectral_weight": 1.0,
        "emb_dim": 128,
        "use_attention": True,
        "attention_type": "cross_freq_v2",
        "norm_type": "batch",
        "cond_mode": "cross_attn_gated",
        "conv_type": "modern",
        "share_embedding": True,
        "reconstruction_mode": "sum",
        "use_se": True,
        "conv_kernel_size": 7,
        "dilations": [1, 4, 16, 32],
        "use_differentiable_decomp": True,
    }

    if is_primary():
        print("="*70)
        print("Dual-Band U-Net Training")
        print("="*70)
        print(f"Config: {json.dumps(config, indent=2)}")

    # Load data
    if is_primary():
        print("\nLoading data...")

    data = prepare_data(seed=config["seed"])

    if is_primary():
        print(f"  Train: {len(data['train']['ob'])} samples")
        print(f"  Val: {len(data['val']['ob'])} samples")
        print(f"  Test: {len(data['test']['ob'])} samples")
        print(f"  Odors: {data['n_odors']}")

    # Train
    resume_path = Path(args.resume) if args.resume else None
    results = train_dual_band(config, data, use_fsdp=args.fsdp, resume_path=resume_path)

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
