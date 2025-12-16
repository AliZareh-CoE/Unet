"""Training script for neural signal translation.

This script provides a clean, modular training pipeline for the U-Net model.
It supports any data format with two brain regions of shape (trials, channels, samples).

Usage:
    # Basic training
    python train.py --data-path /path/to/data.npy --region1 0 32 --region2 32 64

    # With configuration file
    python train.py --config experiment.json

    # Resume training
    python train.py --config experiment.json --resume checkpoint.pt

Example:
    # Train on olfactory data (OB -> PCx)
    python train.py \\
        --data-path /data/signals.npy \\
        --region1 0 32 \\
        --region2 32 64 \\
        --epochs 80 \\
        --batch-size 8

    # Train on PFC/HPC data
    python train.py \\
        --data-path /data/pfc_hpc.npy \\
        --region1 0 64 \\
        --region2 64 96 \\
        --sampling-rate 1250
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DataConfig, ModelConfig, TrainConfig, ExperimentConfig
from data import load_data, create_dataloaders, get_data_info
from models import UNet1D, CombinedLoss, pearson_corr, explained_variance


# =============================================================================
# Training Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get compute device."""
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Data Augmentation
# =============================================================================

def augment_batch(
    batch: Dict[str, torch.Tensor],
    noise_std: float = 0.05,
    time_shift_max: float = 0.1,
    channel_dropout_p: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Apply data augmentation to a batch.

    Args:
        batch: Batch with 'input' and 'target' tensors
        noise_std: Gaussian noise standard deviation
        time_shift_max: Max circular time shift fraction
        channel_dropout_p: Channel dropout probability

    Returns:
        Augmented batch
    """
    x = batch["input"]
    y = batch["target"]
    B, C, T = x.shape

    # Add noise
    if noise_std > 0:
        noise = torch.randn_like(x) * noise_std * x.std()
        x = x + noise

    # Circular time shift (same shift for input and target)
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

    batch["input"] = x
    batch["target"] = y
    return batch


# =============================================================================
# Training Loop
# =============================================================================

class Trainer:
    """Training manager for neural signal translation models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: TrainConfig,
        data_cfg: DataConfig,
        device: torch.device,
        output_dir: Path,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg
        self.device = device
        self.output_dir = output_dir

        # Create output directories
        self.checkpoint_dir = output_dir / "checkpoints"
        self.log_dir = output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.loss_fn = CombinedLoss(
            loss_type=train_cfg.loss_type,
            weight_l1=train_cfg.weight_l1,
            weight_wavelet=train_cfg.weight_wavelet,
            sample_rate=data_cfg.sampling_rate,
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg.learning_rate,
            betas=(train_cfg.beta1, train_cfg.beta2),
            weight_decay=train_cfg.weight_decay,
        )

        # Learning rate scheduler
        if train_cfg.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg.num_epochs,
                eta_min=train_cfg.learning_rate * train_cfg.lr_min_ratio,
            )
        elif train_cfg.lr_scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5,
            )
        else:
            self.scheduler = None

        # Gradient scaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if train_cfg.use_amp and device.type == "cuda" else None

        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_corr": []}

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", leave=False)
        for batch in pbar:
            # Move to device
            x = batch["input"].to(self.device)
            y = batch["target"].to(self.device)

            # Augmentation
            if self.train_cfg.aug_enabled:
                batch_aug = {"input": x, "target": y}
                batch_aug = augment_batch(
                    batch_aug,
                    noise_std=self.train_cfg.aug_noise_std,
                    time_shift_max=self.train_cfg.aug_time_shift_max,
                    channel_dropout_p=self.train_cfg.aug_channel_dropout_p,
                )
                x, y = batch_aug["input"], batch_aug["target"]

            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred = self.model(x)
                    loss, _ = self.loss_fn(pred, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x)
                loss, _ = self.loss_fn(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_corr = 0.0
        n_batches = 0

        for batch in self.val_loader:
            x = batch["input"].to(self.device)
            y = batch["target"].to(self.device)

            pred = self.model(x)
            loss, _ = self.loss_fn(pred, y)
            corr = pearson_corr(pred, y)

            total_loss += loss.item()
            total_corr += corr.item()
            n_batches += 1

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_corr": total_corr / max(n_batches, 1),
        }

    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "train_cfg": self.train_cfg.__dict__ if hasattr(self.train_cfg, '__dict__') else {},
            "data_cfg": self.data_cfg.__dict__ if hasattr(self.data_cfg, '__dict__') else {},
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint.get("history", self.history)
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def train(self) -> Dict[str, list]:
        """Run full training loop."""
        print(f"\nStarting training on {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 60)

        for self.epoch in range(self.epoch, self.train_cfg.num_epochs):
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
            if self.patience_counter >= self.train_cfg.early_stop_patience:
                print(f"\nEarly stopping at epoch {self.epoch}")
                break

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
    parser.add_argument("--data-path", type=str, help="Path to data file (.npy)")
    parser.add_argument("--region1", type=int, nargs=2, default=[0, 32],
                        help="Channel range for region 1 (start end)")
    parser.add_argument("--region2", type=int, nargs=2, default=[32, 64],
                        help="Channel range for region 2 (start end)")
    parser.add_argument("--sampling-rate", type=float, default=1000.0,
                        help="Sampling rate in Hz")
    parser.add_argument("--meta-path", type=str, help="Path to metadata CSV")
    parser.add_argument("--label-column", type=str, help="Label column in metadata")

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
                        choices=["batch", "instance", "group"],
                        help="Normalization type")
    parser.add_argument("--conv-type", type=str, default="modern",
                        choices=["standard", "modern"],
                        help="Convolution type")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--loss-type", type=str, default="l1_wavelet",
                        choices=["l1", "mse", "huber", "l1_wavelet"],
                        help="Loss function type")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Augmentation
    parser.add_argument("--no-aug", action="store_true",
                        help="Disable augmentation")

    # I/O arguments
    parser.add_argument("--output-dir", type=str, default="artifacts",
                        help="Output directory")
    parser.add_argument("--config", type=str,
                        help="Path to config JSON file")
    parser.add_argument("--resume", type=str,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str,
                        help="Device to use (cuda/cpu)")

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Load config from file or create from args
    if args.config and Path(args.config).exists():
        config = ExperimentConfig.load(args.config)
        data_cfg = config.data
        model_cfg = config.model
        train_cfg = config.train
    else:
        # Create configs from arguments
        if not args.data_path:
            print("Error: --data-path is required when not using --config")
            sys.exit(1)

        data_cfg = DataConfig(
            data_path=args.data_path,
            region1_channels=tuple(args.region1),
            region2_channels=tuple(args.region2),
            sampling_rate=args.sampling_rate,
            meta_path=args.meta_path,
            label_column=args.label_column,
            seed=args.seed,
        )

        model_cfg = ModelConfig(
            in_channels=data_cfg.n_channels_region1,
            out_channels=data_cfg.n_channels_region2,
            base_channels=args.base_channels,
            n_downsample=args.n_downsample,
            dropout=args.dropout,
            use_attention=not args.no_attention,
            norm_type=args.norm_type,
            conv_type=args.conv_type,
        )

        train_cfg = TrainConfig(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            loss_type=args.loss_type,
            use_amp=not args.no_amp,
            seed=args.seed,
            aug_enabled=not args.no_aug,
            output_dir=args.output_dir,
        )

    # Setup
    set_seed(train_cfg.seed)
    device = get_device(args.device)
    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = ExperimentConfig(data=data_cfg, model=model_cfg, train=train_cfg)
    config.save(output_dir / "config.json")

    print("=" * 60)
    print("Neural Signal Translation Training")
    print("=" * 60)
    print(f"Data: {data_cfg.data_path}")
    print(f"Region 1: channels {data_cfg.region1_channels}")
    print(f"Region 2: channels {data_cfg.region2_channels}")
    print(f"Sampling rate: {data_cfg.sampling_rate} Hz")
    print(f"Device: {device}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    region1, region2, labels = load_data(data_cfg)
    info = get_data_info(region1, region2)
    print(f"  Trials: {info['n_trials']}")
    print(f"  Region 1 channels: {info['region1_channels']}")
    print(f"  Region 2 channels: {info['region2_channels']}")
    print(f"  Samples per trial: {info['n_samples']}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        region1, region2, labels, data_cfg,
        batch_size=train_cfg.batch_size,
    )

    # Create model
    print("\nCreating model...")
    model = UNet1D.from_config(model_cfg)
    print(f"  Parameters: {model.count_parameters():,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        device=device,
        output_dir=output_dir,
    )

    # Resume if specified
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    # Train
    print("\n" + "=" * 60)
    history = trainer.train()

    # Save final training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # Load best model
    best_path = output_dir / "checkpoints" / "best_model.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    test_loss = 0.0
    test_corr = 0.0
    test_ev = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            pred = model(x)

            loss, _ = CombinedLoss(loss_type=train_cfg.loss_type)(pred, y)
            corr = pearson_corr(pred, y)
            ev = explained_variance(pred, y)

            test_loss += loss.item()
            test_corr += corr.item()
            test_ev += ev.item()
            n_batches += 1

    print(f"Test Loss: {test_loss / n_batches:.4f}")
    print(f"Test Correlation: {test_corr / n_batches:.4f}")
    print(f"Test Explained Variance: {test_ev / n_batches:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
