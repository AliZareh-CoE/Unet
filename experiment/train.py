"""Training script for neural signal translation.

This script provides a modular training pipeline for the U-Net model with:
- FSDP/DDP distributed training support
- Session-based data splitting
- Per-channel normalization
- Output scaling

Usage:
    # Single GPU
    python train.py --data-path /data/signals.npy --epochs 80

    # Distributed with FSDP
    torchrun --nproc_per_node=8 train.py --fsdp --epochs 80

    # With session-based splitting
    python train.py --split-by-session --n-val-sessions 4
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# FSDP imports
try:
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
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore", message=".*FSDP.*")
warnings.filterwarnings("ignore", message=".*barrier.*")

from config import DataConfig, ModelConfig, TrainConfig, ExperimentConfig
from data import load_data, create_dataloaders, get_data_info, NormStats, compute_norm_stats
from models import UNet1D, CombinedLoss, pearson_corr, explained_variance


# =============================================================================
# Distributed Training Utilities
# =============================================================================

def setup_distributed() -> Tuple[int, int, bool]:
    """Setup distributed training environment.

    Returns:
        rank: Global rank of this process
        world_size: Total number of processes
        is_distributed: Whether distributed training is active
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, world_size, True
    return 0, 1, False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_primary() -> bool:
    """Check if this is the primary process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    """Get world size."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    """Get global rank."""
    return dist.get_rank() if dist.is_initialized() else 0


# =============================================================================
# FSDP Utilities
# =============================================================================

def get_fsdp_config(sharding_strategy: str = "full") -> Dict[str, Any]:
    """Get FSDP configuration."""
    if not FSDP_AVAILABLE:
        return {}

    # Mixed precision
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    strategy_map = {
        "full": ShardingStrategy.FULL_SHARD,
        "grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }

    return {
        "sharding_strategy": strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD),
        "mixed_precision": mixed_precision,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "forward_prefetch": True,
        "limit_all_gathers": True,
        "use_orig_params": True,
    }


def wrap_model_fsdp(model: nn.Module, device_id: int, fsdp_config: Dict[str, Any]) -> nn.Module:
    """Wrap model with FSDP."""
    if not FSDP_AVAILABLE:
        return model.cuda(device_id)

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000_000
    )

    return FSDP(
        model.cuda(device_id),
        auto_wrap_policy=auto_wrap_policy,
        device_id=device_id,
        sync_module_states=True,
        **fsdp_config,
    )


def save_fsdp_checkpoint(model: nn.Module, optimizer, path: Path, extra: Dict = None):
    """Save FSDP checkpoint."""
    if not FSDP_AVAILABLE or not isinstance(model, FSDP):
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if extra:
            state.update(extra)
        if is_primary():
            torch.save(state, path)
        return

    # FSDP full state dict
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if extra:
            state.update(extra)
        if is_primary():
            torch.save(state, path)


# =============================================================================
# Training Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_channel_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-channel z-score normalization."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=eps)
    return (x - mean) / std


# =============================================================================
# Data Augmentation
# =============================================================================

def augment_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    noise_std: float = 0.05,
    time_shift_max: float = 0.1,
    channel_dropout_p: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply data augmentation."""
    B, C, T = x.shape

    # Add noise
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std * x.std()

    # Circular time shift
    if time_shift_max > 0:
        max_shift = int(T * time_shift_max)
        shifts = torch.randint(-max_shift, max_shift + 1, (B,))
        for i, shift in enumerate(shifts):
            if shift != 0:
                x[i] = torch.roll(x[i], shift.item(), dims=-1)
                y[i] = torch.roll(y[i], shift.item(), dims=-1)

    # Channel dropout
    if channel_dropout_p > 0:
        mask = torch.rand(B, C, 1, device=x.device) > channel_dropout_p
        x = x * mask.float()

    return x, y


# =============================================================================
# Output Scaling Module
# =============================================================================

class OutputScaling(nn.Module):
    """Learnable per-channel output scaling."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, n_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, n_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """Training manager with distributed support."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args,
        device: torch.device,
        output_dir: Path,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        self.output_dir = output_dir

        # Output scaling
        if args.output_scaling:
            self.output_scaler = OutputScaling(args.out_channels).to(device)
        else:
            self.output_scaler = None

        # Loss function
        self.loss_fn = CombinedLoss(
            loss_type=args.loss_type,
            weight_l1=1.0,
            weight_wavelet=3.0 if "wavelet" in args.loss_type else 0.0,
            sample_rate=args.sampling_rate,
        )

        # Optimizer - include output scaler params if exists
        params = list(model.parameters())
        if self.output_scaler is not None:
            params += list(self.output_scaler.parameters())

        self.optimizer = AdamW(
            params,
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )

        # Scheduler
        if args.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
            )
        elif args.lr_scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
        else:
            self.scheduler = None

        # Gradient scaler for mixed precision (when not using FSDP)
        self.scaler = torch.amp.GradScaler('cuda') if args.amp and not args.fsdp else None

        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_corr": []}

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if self.output_scaler:
            self.output_scaler.train()

        total_loss = 0.0
        n_batches = 0

        # Progress bar only on primary
        loader = self.train_loader
        if is_primary():
            loader = tqdm(loader, desc=f"Epoch {self.epoch}", leave=False)

        for batch in loader:
            x = batch["input"].to(self.device)
            y = batch["target"].to(self.device)

            # Per-channel normalization
            if self.args.per_channel_norm:
                x = per_channel_normalize(x)
                y = per_channel_normalize(y)

            # Augmentation
            if self.args.aug_enabled:
                x, y = augment_batch(
                    x, y,
                    noise_std=self.args.aug_noise_std,
                    time_shift_max=self.args.aug_time_shift_max,
                    channel_dropout_p=self.args.aug_channel_dropout_p,
                )

            # Forward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred = self.model(x)
                    if self.output_scaler:
                        pred = self.output_scaler(pred)
                    loss, _ = self.loss_fn(pred, y)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # FSDP handles mixed precision internally
                pred = self.model(x)
                if self.output_scaler:
                    pred = self.output_scaler(pred)
                loss, _ = self.loss_fn(pred, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if is_primary() and hasattr(loader, 'set_postfix'):
                loader.set_postfix({"loss": f"{loss.item():.4f}"})

        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        if self.output_scaler:
            self.output_scaler.eval()

        total_loss = 0.0
        total_corr = 0.0
        n_batches = 0

        for batch in self.val_loader:
            x = batch["input"].to(self.device)
            y = batch["target"].to(self.device)

            if self.args.per_channel_norm:
                x = per_channel_normalize(x)
                y = per_channel_normalize(y)

            pred = self.model(x)
            if self.output_scaler:
                pred = self.output_scaler(pred)

            loss, _ = self.loss_fn(pred, y)
            corr = pearson_corr(pred, y)

            total_loss += loss.item()
            total_corr += corr.item()
            n_batches += 1

        # Reduce across processes
        if dist.is_initialized():
            metrics = torch.tensor([total_loss, total_corr, n_batches], device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, total_corr, n_batches = metrics.tolist()

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_corr": total_corr / max(n_batches, 1),
        }

    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save checkpoint."""
        extra = {
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "args": vars(self.args),
        }
        if self.output_scaler is not None:
            extra["output_scaler_state_dict"] = self.output_scaler.state_dict()

        path = self.output_dir / "checkpoints" / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        save_fsdp_checkpoint(self.model, self.optimizer, path, extra)

        if is_best and is_primary():
            best_path = self.output_dir / "checkpoints" / "best_model.pt"
            save_fsdp_checkpoint(self.model, self.optimizer, best_path, extra)

    def train(self) -> Dict[str, list]:
        """Run full training loop."""
        if is_primary():
            print(f"\nStarting training on {get_world_size()} GPU(s)")
            print(f"Training samples: {len(self.train_loader.dataset)}")
            print(f"Validation samples: {len(self.val_loader.dataset)}")
            print("-" * 60)

        for self.epoch in range(self.args.epochs):
            # Set epoch for distributed sampler
            if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.epoch)

            # Train
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["train_loss"])

            # Validate
            val_metrics = self.validate()
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_corr"].append(val_metrics["val_corr"])

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Check for improvement
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(f"epoch_{self.epoch:03d}.pt", is_best=is_best)

            # Logging
            if is_primary():
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {self.epoch:3d} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Corr: {val_metrics['val_corr']:.4f} | "
                    f"LR: {lr:.2e}"
                    + (" *" if is_best else "")
                )

            # Early stopping
            if self.patience_counter >= self.args.early_stop_patience:
                if is_primary():
                    print(f"\nEarly stopping at epoch {self.epoch}")
                break

        if is_primary():
            print("-" * 60)
            print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

        return self.history


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train neural signal translation model")

    # Data arguments
    parser.add_argument("--data-path", type=str, default="/data/signal_windows_1khz.npy",
                        help="Path to data file (.npy)")
    parser.add_argument("--region1", type=int, nargs=2, default=[0, 32],
                        help="Channel range for region 1")
    parser.add_argument("--region2", type=int, nargs=2, default=[32, 64],
                        help="Channel range for region 2")
    parser.add_argument("--sampling-rate", type=float, default=1000.0,
                        help="Sampling rate in Hz")
    parser.add_argument("--meta-path", type=str, default="/data/signal_windows_meta_1khz.csv",
                        help="Path to metadata CSV")
    parser.add_argument("--label-column", type=str, default="odor_name",
                        help="Label column in metadata")

    # Session-based splitting
    parser.add_argument("--split-by-session", action="store_true",
                        help="Use session-based splitting")
    parser.add_argument("--n-val-sessions", type=int, default=4,
                        help="Number of validation sessions")
    parser.add_argument("--n-test-sessions", type=int, default=1,
                        help="Number of test sessions")
    parser.add_argument("--force-recreate-splits", action="store_true",
                        help="Force recreate data splits")

    # Model arguments
    parser.add_argument("--base-channels", type=int, default=64,
                        help="Base feature channels")
    parser.add_argument("--n-downsample", type=int, default=4,
                        help="Number of downsampling levels")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout probability")
    parser.add_argument("--no-attention", action="store_true",
                        help="Disable attention")
    parser.add_argument("--norm-type", type=str, default="batch",
                        choices=["batch", "instance", "group"])
    parser.add_argument("--conv-type", type=str, default="modern",
                        choices=["standard", "modern"])

    # Normalization and scaling
    parser.add_argument("--per-channel-norm", action="store_true",
                        help="Apply per-channel normalization")
    parser.add_argument("--output-scaling", action="store_true",
                        help="Use learnable output scaling")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss-type", type=str, default="l1_wavelet",
                        choices=["l1", "mse", "huber", "l1_wavelet"])
    parser.add_argument("--lr-scheduler", type=str, default="none",
                        choices=["none", "cosine", "plateau"])
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    # Augmentation
    parser.add_argument("--no-aug", action="store_true", help="Disable augmentation")
    parser.add_argument("--aug-noise-std", type=float, default=0.05)
    parser.add_argument("--aug-time-shift-max", type=float, default=0.1)
    parser.add_argument("--aug-channel-dropout-p", type=float, default=0.1)

    # Mixed precision
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")

    # Distributed training
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP")
    parser.add_argument("--fsdp-strategy", type=str, default="full",
                        choices=["full", "grad_op", "no_shard"])
    parser.add_argument("--ddp", action="store_true", help="Use DDP instead of FSDP")

    # I/O
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Setup distributed
    rank, world_size, is_distributed = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Seed
    set_seed(args.seed + rank)

    # Derived args
    args.amp = not args.no_amp
    args.aug_enabled = not args.no_aug
    args.in_channels = args.region1[1] - args.region1[0]
    args.out_channels = args.region2[1] - args.region2[0]

    # Output directory
    output_dir = Path(args.output_dir)
    if is_primary():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)

    if is_primary():
        print("=" * 60)
        print("Neural Signal Translation Training")
        print("=" * 60)
        print(f"Data: {args.data_path}")
        print(f"Region 1: channels {args.region1}")
        print(f"Region 2: channels {args.region2}")
        print(f"Device: {device} (world_size={world_size})")
        print(f"FSDP: {args.fsdp}, Strategy: {args.fsdp_strategy}")
        print("=" * 60)

    # Create data config
    data_cfg = DataConfig(
        data_path=args.data_path,
        region1_channels=tuple(args.region1),
        region2_channels=tuple(args.region2),
        sampling_rate=args.sampling_rate,
        meta_path=args.meta_path,
        label_column=args.label_column,
        seed=args.seed,
        data_format="stacked",  # Default OB/PCx format
    )

    # Load data
    if is_primary():
        print("\nLoading data...")

    region1, region2, labels = load_data(data_cfg)

    if is_primary():
        info = get_data_info(region1, region2)
        print(f"  Trials: {info['n_trials']}")
        print(f"  Region 1 channels: {info['region1_channels']}")
        print(f"  Region 2 channels: {info['region2_channels']}")
        print(f"  Samples: {info['n_samples']}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        region1, region2, labels, data_cfg,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Wrap with distributed sampler
    if is_distributed:
        train_sampler = DistributedSampler(train_loader.dataset, shuffle=True)
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    # Create model
    model_cfg = ModelConfig(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_channels=args.base_channels,
        n_downsample=args.n_downsample,
        dropout=args.dropout,
        use_attention=not args.no_attention,
        norm_type=args.norm_type,
        conv_type=args.conv_type,
    )

    model = UNet1D.from_config(model_cfg)

    if is_primary():
        print(f"\nModel parameters: {model.count_parameters():,}")

    # Wrap model for distributed training
    if args.fsdp and is_distributed and FSDP_AVAILABLE:
        fsdp_config = get_fsdp_config(args.fsdp_strategy)
        model = wrap_model_fsdp(model, local_rank, fsdp_config)
        if is_primary():
            print("Model wrapped with FSDP")
    elif args.ddp and is_distributed:
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])
        if is_primary():
            print("Model wrapped with DDP")
    else:
        model = model.to(device)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        device=device,
        output_dir=output_dir,
    )

    # Resume if specified
    if args.resume:
        if is_primary():
            print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.epoch = checkpoint.get("epoch", 0) + 1
        trainer.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    # Train
    if is_primary():
        print("\n" + "=" * 60)

    try:
        history = trainer.train()

        # Save history
        if is_primary():
            history_path = output_dir / "training_history.json"
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            print(f"\nTraining history saved to {history_path}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
