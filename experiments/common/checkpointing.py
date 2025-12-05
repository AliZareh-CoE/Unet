"""
Trial Checkpointing for Nature Methods HPO
===========================================

Save and resume trial state for crash recovery.
Critical for long-running HPO studies.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TrialCheckpoint:
    """Manages trial checkpoints for crash recovery."""

    def __init__(
        self,
        checkpoint_dir: Path,
        study_name: str,
        trial_number: int,
        save_frequency: int = 10,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Base directory for checkpoints
            study_name: Name of the study
            trial_number: Trial number
            save_frequency: Save every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.study_name = study_name
        self.trial_number = trial_number
        self.save_frequency = save_frequency

        self.trial_dir = self.checkpoint_dir / study_name / f"trial_{trial_number}"
        self.trial_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.trial_dir / "checkpoint.pt"
        self.config_path = self.trial_dir / "config.json"
        self.metrics_path = self.trial_dir / "metrics.json"

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        best_score: Optional[float] = None,
        reverse_model: Optional[nn.Module] = None,
    ) -> None:
        """Save checkpoint.

        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: LR scheduler (optional)
            metrics: Current metrics
            config: Trial configuration
            best_score: Best score so far
            reverse_model: Reverse model for bidirectional training
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": best_score,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if reverse_model is not None:
            checkpoint["reverse_model_state_dict"] = reverse_model.state_dict()

        # Save checkpoint atomically (write to temp, then rename)
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        temp_path.rename(self.checkpoint_path)

        # Save config
        if config is not None:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)

        # Append metrics
        if metrics is not None:
            metrics_entry = {"epoch": epoch, **metrics}
            if self.metrics_path.exists():
                with open(self.metrics_path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = []
            all_metrics.append(metrics_entry)
            with open(self.metrics_path, "w") as f:
                json.dump(all_metrics, f, indent=2)

        logger.debug(f"Saved checkpoint at epoch {epoch}")

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        reverse_model: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[int, Optional[float]]:
        """Load checkpoint.

        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            scheduler: LR scheduler (optional)
            reverse_model: Reverse model (optional)
            device: Target device

        Returns:
            (start_epoch, best_score): Epoch to resume from and best score
        """
        if not self.checkpoint_path.exists():
            logger.info("No checkpoint found, starting fresh")
            return 0, None

        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if reverse_model is not None and "reverse_model_state_dict" in checkpoint:
            reverse_model.load_state_dict(checkpoint["reverse_model_state_dict"])

        epoch = checkpoint["epoch"]
        best_score = checkpoint.get("best_score")

        logger.info(f"Resumed from epoch {epoch}, best_score={best_score}")
        return epoch + 1, best_score

    def should_save(self, epoch: int) -> bool:
        """Check if should save at this epoch.

        Args:
            epoch: Current epoch

        Returns:
            True if should save
        """
        return epoch % self.save_frequency == 0

    def cleanup(self) -> None:
        """Remove checkpoint directory (call after successful completion)."""
        if self.trial_dir.exists():
            shutil.rmtree(self.trial_dir)
            logger.debug(f"Cleaned up checkpoint directory: {self.trial_dir}")

    def exists(self) -> bool:
        """Check if checkpoint exists."""
        return self.checkpoint_path.exists()

    def get_config(self) -> Optional[Dict[str, Any]]:
        """Load saved config."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return json.load(f)
        return None

    def get_metrics(self) -> list:
        """Load saved metrics history."""
        if self.metrics_path.exists():
            with open(self.metrics_path, "r") as f:
                return json.load(f)
        return []


class BestModelSaver:
    """Save best model based on validation metric."""

    def __init__(
        self,
        save_dir: Path,
        study_name: str,
        metric_name: str = "val_r2",
        mode: str = "max",
    ):
        """Initialize best model saver.

        Args:
            save_dir: Directory to save models
            study_name: Study name
            metric_name: Metric to track
            mode: "max" or "min"
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.study_name = study_name
        self.metric_name = metric_name
        self.mode = mode

        self.best_score = None
        self.best_trial = None
        self.best_config = None

    def update(
        self,
        trial_number: int,
        score: float,
        model: nn.Module,
        config: Dict[str, Any],
        reverse_model: Optional[nn.Module] = None,
    ) -> bool:
        """Update best model if score is better.

        Args:
            trial_number: Trial number
            score: Current score
            model: Model to potentially save
            config: Trial config
            reverse_model: Reverse model (optional)

        Returns:
            True if this is new best
        """
        is_better = False

        if self.best_score is None:
            is_better = True
        elif self.mode == "max":
            is_better = score > self.best_score
        else:
            is_better = score < self.best_score

        if is_better:
            self.best_score = score
            self.best_trial = trial_number
            self.best_config = config

            # Save model
            save_path = self.save_dir / f"best_{self.study_name}.pt"
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "trial_number": trial_number,
                "score": score,
                "metric_name": self.metric_name,
            }
            if reverse_model is not None:
                checkpoint["reverse_model_state_dict"] = reverse_model.state_dict()

            torch.save(checkpoint, save_path)
            logger.info(
                f"New best {self.metric_name}: {score:.4f} (trial {trial_number})"
            )

            # Also save config
            config_path = self.save_dir / f"best_{self.study_name}_config.json"
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "config": config,
                        "trial_number": trial_number,
                        "score": score,
                    },
                    f,
                    indent=2,
                    default=str,
                )

        return is_better

    def load_best(
        self,
        model: nn.Module,
        reverse_model: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[Dict[str, Any]]:
        """Load best model.

        Args:
            model: Model to load into
            reverse_model: Reverse model (optional)
            device: Target device

        Returns:
            Checkpoint dict or None if not found
        """
        save_path = self.save_dir / f"best_{self.study_name}.pt"
        if not save_path.exists():
            return None

        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if reverse_model is not None and "reverse_model_state_dict" in checkpoint:
            reverse_model.load_state_dict(checkpoint["reverse_model_state_dict"])

        return checkpoint


def cleanup_old_checkpoints(
    checkpoint_dir: Path,
    study_name: str,
    keep_last_n: int = 5,
) -> None:
    """Remove old trial checkpoints, keeping only the last N.

    Args:
        checkpoint_dir: Base checkpoint directory
        study_name: Study name
        keep_last_n: Number of checkpoints to keep
    """
    study_dir = checkpoint_dir / study_name
    if not study_dir.exists():
        return

    trial_dirs = sorted(
        study_dir.glob("trial_*"),
        key=lambda p: int(p.name.split("_")[1]),
        reverse=True,
    )

    for trial_dir in trial_dirs[keep_last_n:]:
        shutil.rmtree(trial_dir)
        logger.debug(f"Removed old checkpoint: {trial_dir}")


class TopKCheckpointSaver:
    """Save only top K models by score to manage storage.

    For Nature Methods: Saves only top 3 checkpoints per study
    to avoid filling up storage during extensive HPO.

    Args:
        save_dir: Directory to save checkpoints
        study_name: Name of the study
        k: Number of top checkpoints to keep (default: 3)
        metric_name: Metric to rank by
        mode: "max" or "min" (maximize or minimize metric)
    """

    def __init__(
        self,
        save_dir: Path,
        study_name: str,
        k: int = 3,
        metric_name: str = "val_r2",
        mode: str = "max",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.study_name = study_name
        self.k = k
        self.metric_name = metric_name
        self.mode = mode

        # Track saved checkpoints: [(trial_id, score, path), ...]
        self.saved_checkpoints: list = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Load info about existing checkpoints."""
        index_path = self.save_dir / f"{self.study_name}_topk_index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                data = json.load(f)
                self.saved_checkpoints = [
                    (c["trial_id"], c["score"], Path(c["path"]))
                    for c in data.get("checkpoints", [])
                ]

    def _save_index(self) -> None:
        """Save index of checkpoints."""
        index_path = self.save_dir / f"{self.study_name}_topk_index.json"
        data = {
            "study_name": self.study_name,
            "k": self.k,
            "metric_name": self.metric_name,
            "mode": self.mode,
            "checkpoints": [
                {"trial_id": tid, "score": score, "path": str(path)}
                for tid, score, path in self.saved_checkpoints
            ],
        }
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _is_top_k(self, score: float) -> bool:
        """Check if score qualifies for top K."""
        if len(self.saved_checkpoints) < self.k:
            return True

        if self.mode == "max":
            worst_score = min(s for _, s, _ in self.saved_checkpoints)
            return score > worst_score
        else:
            worst_score = max(s for _, s, _ in self.saved_checkpoints)
            return score < worst_score

    def _remove_worst(self) -> None:
        """Remove the worst checkpoint to make room."""
        if not self.saved_checkpoints:
            return

        # Find worst
        if self.mode == "max":
            worst_idx = min(range(len(self.saved_checkpoints)),
                          key=lambda i: self.saved_checkpoints[i][1])
        else:
            worst_idx = max(range(len(self.saved_checkpoints)),
                          key=lambda i: self.saved_checkpoints[i][1])

        _, _, worst_path = self.saved_checkpoints[worst_idx]

        # Remove file
        if worst_path.exists():
            worst_path.unlink()
            logger.debug(f"Removed checkpoint: {worst_path}")

        # Remove from list
        del self.saved_checkpoints[worst_idx]

    def maybe_save(
        self,
        trial_id: int,
        score: float,
        model: nn.Module,
        config: Dict[str, Any],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> bool:
        """Save checkpoint if it qualifies for top K.

        Args:
            trial_id: Trial identifier
            score: Metric score
            model: Model to save
            config: Configuration dict
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state

        Returns:
            True if checkpoint was saved
        """
        if not self._is_top_k(score):
            return False

        # Make room if needed
        if len(self.saved_checkpoints) >= self.k:
            self._remove_worst()

        # Save new checkpoint
        checkpoint_name = f"{self.study_name}_trial{trial_id}_{self.metric_name}{score:.4f}.pt"
        save_path = self.save_dir / checkpoint_name

        checkpoint = {
            "trial_id": trial_id,
            "score": score,
            "metric_name": self.metric_name,
            "config": config,
            "model_state_dict": model.state_dict(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(checkpoint, save_path)

        # Update tracking
        self.saved_checkpoints.append((trial_id, score, save_path))
        self._save_index()

        logger.info(f"Saved top-{self.k} checkpoint: trial {trial_id}, {self.metric_name}={score:.4f}")
        return True

    def get_best(self) -> Optional[Tuple[int, float, Path]]:
        """Get the best checkpoint.

        Returns:
            (trial_id, score, path) or None
        """
        if not self.saved_checkpoints:
            return None

        if self.mode == "max":
            return max(self.saved_checkpoints, key=lambda x: x[1])
        else:
            return min(self.saved_checkpoints, key=lambda x: x[1])

    def get_all(self) -> list:
        """Get all saved checkpoints sorted by score.

        Returns:
            List of (trial_id, score, path) tuples
        """
        reverse = self.mode == "max"
        return sorted(self.saved_checkpoints, key=lambda x: x[1], reverse=reverse)

    def load_best(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint.

        Args:
            model: Model to load into
            device: Target device

        Returns:
            Checkpoint dict or None
        """
        best = self.get_best()
        if best is None:
            return None

        _, _, path = best
        if not path.exists():
            return None

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
