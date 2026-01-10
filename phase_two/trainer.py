"""
Training Loop for Phase 2
=========================

PyTorch training loop with:
- Mixed precision training (AMP)
- Learning rate scheduling (cosine with warmup)
- Early stopping
- Checkpointing
- Comprehensive logging
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.cuda.amp import GradScaler, autocast

from .config import TrainingConfig


# =============================================================================
# Metrics
# =============================================================================

def compute_r2(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute R² (coefficient of determination).

    Args:
        y_pred: Predictions [B, C, T]
        y_true: Ground truth [B, C, T]

    Returns:
        R² value
    """
    with torch.no_grad():
        # Flatten to [B*C*T]
        pred_flat = y_pred.reshape(-1)
        true_flat = y_true.reshape(-1)

        ss_res = torch.sum((true_flat - pred_flat) ** 2)
        ss_tot = torch.sum((true_flat - torch.mean(true_flat)) ** 2)

        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return r2.item()


def compute_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute MAE."""
    with torch.no_grad():
        return torch.mean(torch.abs(y_pred - y_true)).item()


# =============================================================================
# Data Augmentation
# =============================================================================

class SignalAugmentation:
    """Data augmentation for neural signals."""

    def __init__(
        self,
        noise_std: float = 0.1,
        time_shift: int = 50,
        p: float = 0.5,
    ):
        self.noise_std = noise_std
        self.time_shift = time_shift
        self.p = p

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation.

        Args:
            x: Input [B, C, T]
            y: Target [B, C, T]

        Returns:
            Augmented (x, y)
        """
        B, C, T = x.shape

        # Gaussian noise
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # Time shift (same shift for x and y to maintain alignment)
        if torch.rand(1).item() < self.p and self.time_shift > 0:
            shift = torch.randint(-self.time_shift, self.time_shift + 1, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)
                y = torch.roll(y, shifts=shift, dims=-1)

        return x, y


# =============================================================================
# Training Result
# =============================================================================

@dataclass
class TrainingResult:
    """Results from a single training run."""

    architecture: str
    seed: int

    # Final metrics
    best_val_r2: float
    best_val_mae: float
    best_epoch: int

    # Training history
    train_losses: List[float]
    val_losses: List[float]
    val_r2s: List[float]

    # Timing
    total_time: float
    epochs_trained: int

    # Model info
    n_parameters: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "seed": self.seed,
            "best_val_r2": self.best_val_r2,
            "best_val_mae": self.best_val_mae,
            "best_epoch": self.best_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_r2s": self.val_r2s,
            "total_time": self.total_time,
            "epochs_trained": self.epochs_trained,
            "n_parameters": self.n_parameters,
        }


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """Training loop for Phase 2 architectures.

    Features:
        - Mixed precision training (AMP)
        - Cosine learning rate schedule with warmup
        - Early stopping
        - Best model checkpointing
        - Comprehensive metrics logging

    Args:
        model: PyTorch model
        config: Training configuration
        device: Torch device
        checkpoint_dir: Directory for saving checkpoints
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Loss function
        if config.loss_fn == "l1":
            self.criterion = nn.L1Loss()
        elif config.loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif config.loss_fn == "huber":
            self.criterion = nn.HuberLoss()
        else:
            self.criterion = nn.L1Loss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp and device == "cuda" else None
        self.use_amp = config.use_amp and device == "cuda"

        # Augmentation
        self.augmentation = SignalAugmentation(
            noise_std=config.aug_noise_std,
            time_shift=config.aug_time_shift,
        ) if config.use_augmentation else None

        # State
        self.best_val_loss = float('inf')
        self.best_val_r2 = -float('inf')
        self.patience_counter = 0

    def _get_lr_scheduler(self, n_steps: int) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        warmup_steps = self.config.warmup_epochs * (n_steps // self.config.epochs)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
            return max(self.config.lr_min / self.config.learning_rate,
                      0.5 * (1 + np.cos(np.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            # Apply augmentation
            if self.augmentation is not None:
                x, y = self.augmentation(x, y)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    y_pred = self.model(x)
                    loss = self.criterion(y_pred, y)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Validate model.

        Returns:
            (loss, r2, mae)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

            total_loss += loss.item()
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())

        # Compute metrics
        y_pred = torch.cat(all_preds, dim=0)
        y_true = torch.cat(all_targets, dim=0)

        loss = total_loss / len(dataloader)
        r2 = compute_r2(y_pred, y_true)
        mae = compute_mae(y_pred, y_true)

        return loss, r2, mae

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        arch_name: str = "model",
        seed: int = 42,
        verbose: bool = True,
    ) -> TrainingResult:
        """Train model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            arch_name: Architecture name (for logging)
            seed: Random seed used
            verbose: Whether to print progress

        Returns:
            TrainingResult with all metrics
        """
        n_steps = self.config.epochs * len(train_loader)
        scheduler = self._get_lr_scheduler(n_steps)

        # History
        train_losses = []
        val_losses = []
        val_r2s = []

        best_epoch = 0
        best_val_r2 = -float('inf')
        best_val_mae = float('inf')

        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Step scheduler
            scheduler.step()

            # Validate
            val_loss, val_r2, val_mae = self.validate(val_loader)

            # Record history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_r2s.append(val_r2)

            # Check for improvement
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_val_mae = val_mae
                best_epoch = epoch + 1
                self.patience_counter = 0

                # Save checkpoint
                if self.checkpoint_dir:
                    self._save_checkpoint(arch_name, seed)
            else:
                self.patience_counter += 1

            # Log
            if verbose and (epoch + 1) % 5 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs}: "
                      f"train_loss={train_loss:.4f}, val_r2={val_r2:.4f}, "
                      f"val_mae={val_mae:.4f}, lr={lr:.2e}")

            # Early stopping
            if self.patience_counter >= self.config.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        total_time = time.time() - start_time

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return TrainingResult(
            architecture=arch_name,
            seed=seed,
            best_val_r2=best_val_r2,
            best_val_mae=best_val_mae,
            best_epoch=best_epoch,
            train_losses=train_losses,
            val_losses=val_losses,
            val_r2s=val_r2s,
            total_time=total_time,
            epochs_trained=len(train_losses),
            n_parameters=n_params,
        )

    def _save_checkpoint(self, arch_name: str, seed: int):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{arch_name}_seed{seed}_best.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    n_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch data loaders.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size
        n_workers: Number of data loader workers

    Returns:
        (train_loader, val_loader)
    """
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float()

    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
