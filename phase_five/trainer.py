"""
Trainer for Phase 5: Real-Time Continuous
==========================================

Handles training with sliding window data.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import TrainingConfig, ExperimentConfig


class Phase5Trainer:
    """Trainer for Phase 5 sliding window experiments."""

    def __init__(
        self,
        model: nn.Module,
        training_config: TrainingConfig,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.training_config = training_config
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Loss function
        if training_config.loss_type == "huber":
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.L1Loss()

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=training_config.epochs,
            eta_min=training_config.lr_min,
        )

        # Mixed precision
        self.use_amp = training_config.use_amp and device == "cuda"
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")

        # State
        self.epoch = 0
        self.best_val_r2 = float("-inf")
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_r2s: List[float] = []
        self.nan_count = 0
        self.patience_counter = 0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    pred = self.model(x)
                    loss = self.criterion(pred, y)

                if torch.isnan(loss):
                    self.nan_count += 1
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.training_config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)

                if torch.isnan(loss):
                    self.nan_count += 1
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
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in dataloader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)

            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

        # Compute R²
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        pred_flat = preds.reshape(-1)
        target_flat = targets.reshape(-1)
        ss_res = ((target_flat - pred_flat) ** 2).sum()
        ss_tot = ((target_flat - target_flat.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        return total_loss / len(dataloader), r2.item()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Full training loop."""
        epochs = epochs or self.training_config.epochs
        start_time = time.time()

        for epoch in range(epochs):
            self.epoch = epoch

            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            val_loss, val_r2 = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_r2s.append(val_r2)

            self.scheduler.step()

            # Early stopping
            if val_r2 > self.best_val_r2 + self.training_config.min_delta:
                self.best_val_r2 = val_r2
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Logging
            if (epoch + 1) % self.training_config.log_every == 0:
                print(f"    Epoch {epoch + 1}/{epochs} | "
                      f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | R²: {val_r2:.4f}")

            if self.patience_counter >= self.training_config.patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_r2s": self.val_r2s,
            "best_val_r2": self.best_val_r2,
            "epochs_trained": len(self.train_losses),
            "total_time": time.time() - start_time,
        }
