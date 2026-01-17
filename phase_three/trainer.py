"""
Trainer for Phase 3 Ablation Studies
=====================================

Handles training loops for CondUNet ablation experiments.
"""

from __future__ import annotations

import time
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from .config import AblationConfig, TrainingConfig
from .losses import build_loss


class AblationTrainer:
    """Trainer for ablation experiments.

    Handles training with configurable model variants, loss functions,
    and training settings.
    """

    def __init__(
        self,
        model: nn.Module,
        ablation_config: AblationConfig,
        training_config: TrainingConfig,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.ablation_config = ablation_config
        self.training_config = training_config
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Note: Session ID mapping is NO LONGER NEEDED with statistics-based conditioning
        # The model computes session statistics from input signal, automatically
        # generalizing to unseen sessions without explicit mapping.

        # Build loss function based on ablation config
        self.criterion = build_loss(
            ablation_config.loss_type,
            ablation_config.loss_weights,
        ).to(device)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = self._build_scheduler()

        # Mixed precision
        self.use_amp = training_config.use_amp and device == "cuda"
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")

        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_r2 = float("-inf")
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_r2s: List[float] = []
        self.patience_counter = 0

        # NaN recovery
        self.nan_count = 0

    def _build_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler."""
        cfg = self.training_config

        if cfg.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.epochs,
                eta_min=cfg.lr_min,
            )
        elif cfg.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.epochs // 3,
                gamma=0.1,
            )
        elif cfg.lr_scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=cfg.patience // 2,
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.

        Note: With statistics-based session conditioning, the model automatically
        computes session statistics from input. No explicit session IDs needed.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Handle different batch formats: 2, 3, or 4 elements
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
                odor_ids = batch[2] if len(batch) > 2 else None
                # Note: session_ids (batch[3]) are no longer needed
                # Statistics-based conditioning computes from input signal
            else:
                x, y = batch, batch
                odor_ids = None

            x = x.to(self.device)
            y = y.to(self.device)
            if odor_ids is not None:
                odor_ids = odor_ids.to(self.device)

            # Apply augmentation if enabled
            if self.ablation_config.use_augmentation:
                x, y = self._augment(x, y)

            # Forward pass with mixed precision
            self.optimizer.zero_grad()

            # Determine what to pass to model (session stats computed automatically)
            use_odor = odor_ids is not None and self.ablation_config.use_odor_embedding

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    if use_odor:
                        pred = self.model(x, odor_ids)
                    else:
                        pred = self.model(x)
                    loss, _ = self.criterion(pred, y)

                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    self.nan_count += 1
                    if self.nan_count > self.training_config.max_nan_recovery:
                        raise RuntimeError(f"Too many NaN batches: {self.nan_count}")
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.training_config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if use_odor:
                    pred = self.model(x, odor_ids)
                else:
                    pred = self.model(x)
                loss, _ = self.criterion(pred, y)

                if torch.isnan(loss) or torch.isinf(loss):
                    self.nan_count += 1
                    if self.nan_count > self.training_config.max_nan_recovery:
                        raise RuntimeError(f"Too many NaN batches: {self.nan_count}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.training_config.grad_clip
                )
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate model and compute R².

        Note: With statistics-based session conditioning, the model automatically
        handles unseen sessions by computing statistics from the input signal.
        No explicit session ID mapping is needed.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in dataloader:
            # Handle different batch formats: 2, 3, or 4 elements
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
                odor_ids = batch[2] if len(batch) > 2 else None
                # Note: session_ids are no longer needed for statistics-based conditioning
                # The model computes session statistics directly from input
            else:
                x, y = batch, batch
                odor_ids = None

            x = x.to(self.device)
            y = y.to(self.device)
            if odor_ids is not None:
                odor_ids = odor_ids.to(self.device)

            # Model forward - session stats computed automatically from input
            use_odor = odor_ids is not None and self.ablation_config.use_odor_embedding

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    if use_odor:
                        pred = self.model(x, odor_ids)
                    else:
                        pred = self.model(x)
                    loss, _ = self.criterion(pred, y)
            else:
                if use_odor:
                    pred = self.model(x, odor_ids)
                else:
                    pred = self.model(x)
                loss, _ = self.criterion(pred, y)

            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

        # Compute R²
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        r2 = self._compute_r2(preds, targets)

        return total_loss / len(dataloader), r2

    def _compute_r2(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute R² (coefficient of determination)."""
        # Flatten to [N, features]
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        ss_res = ((target_flat - pred_flat) ** 2).sum()
        ss_tot = ((target_flat - target_flat.mean()) ** 2).sum()

        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return r2.item()

    def _augment(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation."""
        # Noise augmentation
        if self.ablation_config.aug_noise_std > 0:
            noise_x = torch.randn_like(x) * self.ablation_config.aug_noise_std
            noise_y = torch.randn_like(y) * self.ablation_config.aug_noise_std
            x = x + noise_x
            y = y + noise_y

        # Time shift augmentation
        shift = self.ablation_config.aug_time_shift
        if shift > 0 and x.shape[-1] > shift:
            shift_amount = torch.randint(-shift, shift + 1, (1,)).item()
            if shift_amount != 0:
                x = torch.roll(x, shifts=shift_amount, dims=-1)
                y = torch.roll(y, shifts=shift_amount, dims=-1)

        return x, y

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Full training loop.

        Args:
            train_loader: Training data
            val_loader: Validation data
            epochs: Number of epochs (overrides config)

        Returns:
            Training history dictionary
        """
        epochs = epochs or self.training_config.epochs
        start_time = time.time()

        for epoch in range(epochs):
            self.epoch = epoch

            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss, val_r2 = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_r2s.append(val_r2)

            # Learning rate scheduling
            if self.training_config.lr_scheduler == "plateau":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Early stopping check
            if val_r2 > self.best_val_r2 + self.training_config.min_delta:
                self.best_val_r2 = val_r2
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save checkpoint
                if self.checkpoint_dir:
                    self._save_checkpoint("best")
            else:
                self.patience_counter += 1

            # Periodic checkpoint
            if (
                self.checkpoint_dir
                and (epoch + 1) % self.training_config.checkpoint_every == 0
            ):
                self._save_checkpoint(f"epoch_{epoch + 1}")

            # Logging
            if (epoch + 1) % self.training_config.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch + 1}/{epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"R²: {val_r2:.4f} | LR: {lr:.2e}"
                )

            # Early stopping
            if self.patience_counter >= self.training_config.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        total_time = time.time() - start_time

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_r2s": self.val_r2s,
            "best_val_r2": self.best_val_r2,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": len(self.train_losses),
            "total_time": total_time,
        }

    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{self.ablation_config.name}_{name}.pt"

        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_r2": self.best_val_r2,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "val_r2s": self.val_r2s,
                "ablation_config": self.ablation_config.to_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_r2 = checkpoint["best_val_r2"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.val_r2s = checkpoint["val_r2s"]
