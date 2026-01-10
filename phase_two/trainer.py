"""
Training Loop for Phase 2
=========================

PyTorch training loop with:
- Mixed precision training (AMP)
- Learning rate scheduling (cosine with warmup)
- Early stopping
- Checkpointing (best model + periodic)
- Comprehensive logging
- NaN/Inf detection and recovery
- Resume from checkpoint
- Memory monitoring
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.cuda.amp import GradScaler, autocast

from .config import TrainingConfig


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_dir: Optional[Path] = None, name: str = "phase2") -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_dir / f"{name}_{timestamp}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


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


def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check if tensor contains NaN or Inf values.

    Returns:
        True if NaN/Inf detected
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    return has_nan or has_inf


def get_gpu_memory_mb() -> Optional[float]:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return None


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

    # Robustness info
    nan_detected: bool = False
    nan_recovery_count: int = 0
    early_stopped: bool = False
    error_message: Optional[str] = None
    peak_gpu_memory_mb: Optional[float] = None
    completed_successfully: bool = True

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
            "nan_detected": self.nan_detected,
            "nan_recovery_count": self.nan_recovery_count,
            "early_stopped": self.early_stopped,
            "error_message": self.error_message,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "completed_successfully": self.completed_successfully,
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
        - Best model checkpointing + periodic checkpoints
        - Comprehensive metrics logging
        - NaN/Inf detection and recovery
        - Resume from checkpoint
        - Memory monitoring

    Args:
        model: PyTorch model
        config: Training configuration
        device: Torch device
        checkpoint_dir: Directory for saving checkpoints
        log_dir: Directory for log files
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Setup logging
        self.logger = setup_logging(log_dir, "phase2_trainer")

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
        self.nan_recovery_count = 0
        self.nan_detected = False
        self.peak_gpu_memory_mb = 0.0

        # For resume
        self.start_epoch = 0
        self._best_model_state = None

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

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, bool]:
        """Train for one epoch.

        Returns:
            (average_loss, nan_detected)
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        nan_in_epoch = False

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            # Apply augmentation
            if self.augmentation is not None:
                x, y = self.augmentation(x, y)

            # Forward pass
            self.optimizer.zero_grad()

            try:
                if self.use_amp:
                    with autocast():
                        y_pred = self.model(x)
                        loss = self.criterion(y_pred, y)

                    # Check for NaN/Inf
                    if check_nan_inf(loss, "loss"):
                        self.logger.warning(f"NaN/Inf detected in loss at batch {n_batches}")
                        nan_in_epoch = True
                        self.nan_detected = True
                        self.nan_recovery_count += 1
                        # Skip this batch
                        continue

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    y_pred = self.model(x)
                    loss = self.criterion(y_pred, y)

                    # Check for NaN/Inf
                    if check_nan_inf(loss, "loss"):
                        self.logger.warning(f"NaN/Inf detected in loss at batch {n_batches}")
                        nan_in_epoch = True
                        self.nan_detected = True
                        self.nan_recovery_count += 1
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                # Track GPU memory
                if self.device == "cuda":
                    mem = get_gpu_memory_mb()
                    if mem and mem > self.peak_gpu_memory_mb:
                        self.peak_gpu_memory_mb = mem

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.error(f"GPU OOM at batch {n_batches}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise
                else:
                    self.logger.error(f"Runtime error at batch {n_batches}: {e}")
                    raise

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, nan_in_epoch

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

    def load_checkpoint(self, path: Path) -> int:
        """Load checkpoint and return epoch to resume from.

        Args:
            path: Path to checkpoint file

        Returns:
            Epoch number to resume from
        """
        if not path.exists():
            self.logger.warning(f"Checkpoint not found: {path}")
            return 0

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_val_r2 = checkpoint.get('best_val_r2', -float('inf'))
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        self.logger.info(f"Loaded checkpoint from epoch {self.start_epoch}")
        return self.start_epoch

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        arch_name: str = "model",
        seed: int = 42,
        verbose: bool = True,
        resume_from: Optional[Path] = None,
        checkpoint_every: int = 10,
    ) -> TrainingResult:
        """Train model with robust error handling.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            arch_name: Architecture name (for logging)
            seed: Random seed used
            verbose: Whether to print progress
            resume_from: Optional checkpoint path to resume from
            checkpoint_every: Save checkpoint every N epochs

        Returns:
            TrainingResult with all metrics
        """
        self.logger.info(f"Starting training for {arch_name} (seed={seed})")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # Resume if requested
        if resume_from:
            self.load_checkpoint(resume_from)

        n_steps = self.config.epochs * len(train_loader)
        scheduler = self._get_lr_scheduler(n_steps)

        # History
        train_losses = []
        val_losses = []
        val_r2s = []

        best_epoch = 0
        best_val_r2 = self.best_val_r2
        best_val_mae = float('inf')
        early_stopped = False
        error_message = None
        completed_successfully = True

        start_time = time.time()

        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                epoch_start = time.time()

                # Train
                train_loss, nan_in_epoch = self.train_epoch(train_loader)

                # Handle excessive NaN
                if self.nan_recovery_count > 100:
                    error_message = "Too many NaN values detected, stopping training"
                    self.logger.error(error_message)
                    completed_successfully = False
                    break

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
                    self._best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }

                    # Save best checkpoint
                    if self.checkpoint_dir:
                        self._save_checkpoint(arch_name, seed, is_best=True, epoch=epoch+1)
                else:
                    self.patience_counter += 1

                # Periodic checkpoint
                if self.checkpoint_dir and (epoch + 1) % checkpoint_every == 0:
                    self._save_checkpoint(arch_name, seed, is_best=False, epoch=epoch+1)

                # Log
                epoch_time = time.time() - epoch_start
                if verbose and (epoch + 1) % 5 == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    mem_str = f", GPU={self.peak_gpu_memory_mb:.0f}MB" if self.device == "cuda" else ""
                    self.logger.info(
                        f"Epoch {epoch+1:3d}/{self.config.epochs}: "
                        f"train_loss={train_loss:.4f}, val_r2={val_r2:.4f}, "
                        f"val_mae={val_mae:.4f}, lr={lr:.2e}, "
                        f"time={epoch_time:.1f}s{mem_str}"
                    )

                # Early stopping
                if self.patience_counter >= self.config.patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    early_stopped = True
                    break

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            error_message = "Interrupted by user"
            completed_successfully = False

        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e)}"
            self.logger.error(f"Training failed: {error_message}")
            self.logger.error(traceback.format_exc())
            completed_successfully = False

        total_time = time.time() - start_time

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Restore best model if available
        if self._best_model_state is not None and completed_successfully:
            self.model.load_state_dict(self._best_model_state)
            self.logger.info(f"Restored best model from epoch {best_epoch}")

        self.logger.info(
            f"Training completed: best_val_r2={best_val_r2:.4f} at epoch {best_epoch}, "
            f"total_time={total_time/60:.1f}min"
        )

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
            nan_detected=self.nan_detected,
            nan_recovery_count=self.nan_recovery_count,
            early_stopped=early_stopped,
            error_message=error_message,
            peak_gpu_memory_mb=self.peak_gpu_memory_mb if self.device == "cuda" else None,
            completed_successfully=completed_successfully,
        )

    def _save_checkpoint(
        self,
        arch_name: str,
        seed: int,
        is_best: bool = True,
        epoch: int = 0,
    ):
        """Save model checkpoint.

        Args:
            arch_name: Architecture name
            seed: Random seed
            is_best: Whether this is the best model checkpoint
            epoch: Current epoch number
        """
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_val_r2': self.best_val_r2,
            'best_val_loss': self.best_val_loss,
            'architecture': arch_name,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
        }

        if is_best:
            path = self.checkpoint_dir / f"{arch_name}_seed{seed}_best.pt"
        else:
            path = self.checkpoint_dir / f"{arch_name}_seed{seed}_epoch{epoch}.pt"

        torch.save(checkpoint, path)
        self.logger.debug(f"Saved checkpoint: {path}")


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
