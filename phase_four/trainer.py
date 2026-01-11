"""
Trainer for Phase 4: Inter vs Intra Session
============================================

Handles training loops for generalization experiments.
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
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import TrainingConfig, ExperimentConfig


class Phase4Trainer:
    """Trainer for Phase 4 experiments.

    Handles training with session-aware evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        experiment_config: ExperimentConfig,
        training_config: TrainingConfig,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.experiment_config = experiment_config
        self.training_config = training_config
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Loss function
        if training_config.loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif training_config.loss_type == "huber":
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()

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

        # Training state
        self.epoch = 0
        self.best_val_r2 = float("-inf")
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_r2s: List[float] = []
        self.patience_counter = 0
        self.nan_count = 0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                x, y = batch, batch

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    pred = self.model(x)
                    loss = self.criterion(pred, y)

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
                pred = self.model(x)
                loss = self.criterion(pred, y)

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
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate model.

        Returns:
            loss, r2, predictions, targets
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in dataloader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                x, y = batch, batch

            x = x.to(self.device)
            y = y.to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)

            total_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

        # Concatenate
        predictions = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Compute R²
        r2 = self._compute_r2(predictions, targets)
        avg_loss = total_loss / len(dataloader)

        return avg_loss, r2, predictions, targets

    def _compute_r2(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute R² (coefficient of determination)."""
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - target_flat.mean()) ** 2)

        return float(1 - ss_res / (ss_tot + 1e-8))

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

            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss, val_r2, _, _ = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_r2s.append(val_r2)

            # Scheduler step
            self.scheduler.step()

            # Early stopping check
            if val_r2 > self.best_val_r2 + self.training_config.min_delta:
                self.best_val_r2 = val_r2
                self.patience_counter = 0

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
                    f"    Epoch {epoch + 1}/{epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"R²: {val_r2:.4f} | LR: {lr:.2e}"
                )

            # Early stopping
            if self.patience_counter >= self.training_config.patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

        total_time = time.time() - start_time

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_r2s": self.val_r2s,
            "best_val_r2": self.best_val_r2,
            "epochs_trained": len(self.train_losses),
            "total_time": total_time,
        }

    def _save_checkpoint(self, name: str):
        """Save checkpoint."""
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{self.experiment_config.name}_{name}.pt"

        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_r2": self.best_val_r2,
                "experiment_config": self.experiment_config.to_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_r2 = checkpoint["best_val_r2"]
