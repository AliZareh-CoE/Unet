"""
Training Loop for Phase 2
=========================

PyTorch training loop with:
- Mixed precision training (AMP)
- FSDP (Fully Sharded Data Parallel) for multi-GPU training
- Learning rate scheduling (cosine with warmup)
- Early stopping
- Checkpointing (best model + periodic)
- Comprehensive logging
- NaN/Inf detection and recovery
- Resume from checkpoint
- Memory monitoring

FSDP Usage:
    torchrun --nproc_per_node=8 -m phase_two.runner --fsdp --epochs 80 --batch-size 64
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress PyTorch C++ warnings BEFORE importing torch
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("TORCH_LOGS", "-all")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "0")
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "0")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# Disable PyTorch's internal warnings
torch.set_warn_always(False)

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*FSDP.state_dict_type.*")
warnings.filterwarnings("ignore", message=".*Both mixed precision and an auto_wrap_policy.*")
warnings.filterwarnings("ignore", message=".*Deallocating Tensor.*")

# FSDP imports
try:
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        CPUOffload,
        BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        ModuleWrapPolicy,
    )
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullStateDictConfig,
        StateDictType,
    )
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

from .config import TrainingConfig

# Add project root to path
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from utils.dsp_metrics import compute_all_dsp_metrics, compute_batch_metrics
    DSP_AVAILABLE = True
except ImportError:
    DSP_AVAILABLE = False


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
# Distributed Training Utilities
# =============================================================================

def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_initialized() if FSDP_AVAILABLE else False


def get_rank() -> int:
    """Get current process rank (0 if not distributed)."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed() -> Tuple[int, int]:
    """Initialize distributed training environment.

    Returns:
        (rank, world_size)
    """
    if not FSDP_AVAILABLE:
        return 0, 1

    # Check if launched with torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set CUDA device before init_process_group
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
            )

        return rank, world_size

    return 0, 1


def cleanup_distributed():
    """Clean up distributed training resources."""
    if is_distributed():
        dist.destroy_process_group()


def get_fsdp_config(
    strategy: str = "grad_op",
    use_amp: bool = True,
    cpu_offload: bool = False,
) -> Dict[str, Any]:
    """Get FSDP configuration.

    Args:
        strategy: Sharding strategy ('full', 'grad_op', 'no_shard', 'hybrid')
        use_amp: Use mixed precision
        cpu_offload: Offload parameters to CPU

    Returns:
        Dict with FSDP configuration
    """
    if not FSDP_AVAILABLE:
        return {}

    # Map strategy names
    strategy_map = {
        "full": ShardingStrategy.FULL_SHARD,
        "grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid": ShardingStrategy.HYBRID_SHARD,
    }
    sharding_strategy = strategy_map.get(strategy, ShardingStrategy.SHARD_GRAD_OP)

    # Mixed precision policy
    mp_policy = None
    if use_amp:
        mp_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    # CPU offload
    offload_config = CPUOffload(offload_params=True) if cpu_offload else None

    return {
        "sharding_strategy": sharding_strategy,
        "mixed_precision": mp_policy,
        "cpu_offload": offload_config,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "device_id": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "auto_wrap_policy": size_based_auto_wrap_policy,
    }


def wrap_model_fsdp(
    model: nn.Module,
    fsdp_config: Dict[str, Any],
    min_num_params: int = 1_000_000,
) -> nn.Module:
    """Wrap model with FSDP.

    Args:
        model: PyTorch model
        fsdp_config: FSDP configuration from get_fsdp_config()
        min_num_params: Minimum parameters for auto-wrapping sublayers

    Returns:
        FSDP-wrapped model
    """
    if not FSDP_AVAILABLE:
        return model

    from functools import partial

    # Create auto-wrap policy based on parameter count
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=min_num_params,
    )

    wrapped = FSDP(
        model,
        sharding_strategy=fsdp_config.get("sharding_strategy", ShardingStrategy.SHARD_GRAD_OP),
        mixed_precision=fsdp_config.get("mixed_precision"),
        cpu_offload=fsdp_config.get("cpu_offload"),
        backward_prefetch=fsdp_config.get("backward_prefetch"),
        device_id=fsdp_config.get("device_id"),
        auto_wrap_policy=auto_wrap_policy,
    )

    return wrapped


def get_fsdp_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get full state dict from FSDP model.

    Must be called on all ranks, but only rank 0 will have the full state dict.
    """
    if not FSDP_AVAILABLE or not isinstance(model, FSDP):
        return model.state_dict()

    # Configure to get full state dict on rank 0
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        state_dict = model.state_dict()

    return state_dict


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
    """Results from a single training run (one fold of cross-validation)."""

    architecture: str
    fold: int  # CV fold index (0 to n_folds-1)

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

    # DSP Metrics (computed on final validation predictions)
    dsp_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "architecture": self.architecture,
            "fold": self.fold,
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
            "dsp_metrics": self.dsp_metrics,
        }
        return result


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """Training loop for Phase 2 architectures.

    Features:
        - Mixed precision training (AMP)
        - FSDP (Fully Sharded Data Parallel) for multi-GPU training
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
        use_fsdp: Enable FSDP for distributed training
        fsdp_strategy: FSDP sharding strategy ('full', 'grad_op', 'no_shard', 'hybrid')
        fsdp_cpu_offload: Enable CPU offloading for FSDP

    FSDP Usage:
        torchrun --nproc_per_node=8 -m phase_two.runner --fsdp --fsdp-strategy grad_op
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        use_fsdp: bool = False,
        fsdp_strategy: str = "grad_op",
        fsdp_cpu_offload: bool = False,
    ):
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.use_fsdp = use_fsdp and FSDP_AVAILABLE and is_distributed()
        self.rank = get_rank()
        self.world_size = get_world_size()
        self._is_main = is_main_process()

        # Setup logging (only on main process)
        if self._is_main:
            self.logger = setup_logging(log_dir, "phase2_trainer")
        else:
            # Null logger for non-main processes
            self.logger = logging.getLogger("phase2_trainer_null")
            self.logger.addHandler(logging.NullHandler())

        # Wrap model with FSDP if enabled
        if self.use_fsdp:
            self._log_info(f"Initializing FSDP with strategy={fsdp_strategy}, world_size={self.world_size}")
            fsdp_config = get_fsdp_config(
                strategy=fsdp_strategy,
                use_amp=config.use_amp,
                cpu_offload=fsdp_cpu_offload,
            )
            self.model = wrap_model_fsdp(model, fsdp_config)
            # FSDP handles device placement
            self.device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        else:
            self.model = model.to(device)

        # Loss function
        if config.loss_fn == "l1":
            self.criterion = nn.L1Loss()
        elif config.loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif config.loss_fn == "huber":
            self.criterion = nn.HuberLoss()
        else:
            self.criterion = nn.L1Loss()

        # Optimizer (use model.parameters() after FSDP wrapping)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Mixed precision (FSDP has its own mixed precision handling)
        if self.use_fsdp:
            # FSDP handles mixed precision internally
            self.scaler = None
            self.use_amp = False
        else:
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

    def _log_info(self, msg: str):
        """Log info only on main process."""
        if self._is_main:
            self.logger.info(msg)

    def _log_warning(self, msg: str):
        """Log warning only on main process."""
        if self._is_main:
            self.logger.warning(msg)

    def _log_error(self, msg: str):
        """Log error only on main process."""
        if self._is_main:
            self.logger.error(msg)

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
                        self._log_warning(f"NaN/Inf detected in loss at batch {n_batches}")
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
                        self._log_warning(f"NaN/Inf detected in loss at batch {n_batches}")
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
                if "cuda" in str(self.device):
                    mem = get_gpu_memory_mb()
                    if mem and mem > self.peak_gpu_memory_mb:
                        self.peak_gpu_memory_mb = mem

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._log_error(f"GPU OOM at batch {n_batches}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise
                else:
                    self._log_error(f"Runtime error at batch {n_batches}: {e}")
                    raise

        # Reduce loss across processes if distributed
        if self.use_fsdp:
            loss_tensor = torch.tensor([total_loss, n_batches], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss, n_batches = loss_tensor[0].item(), loss_tensor[1].item()

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
        n_batches = 0

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())

        # Compute metrics
        y_pred = torch.cat(all_preds, dim=0)
        y_true = torch.cat(all_targets, dim=0)

        # Gather metrics from all processes if distributed
        if self.use_fsdp:
            # Reduce loss
            loss_tensor = torch.tensor([total_loss, n_batches], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss, n_batches = loss_tensor[0].item(), loss_tensor[1].item()

            # Compute R2 components locally then reduce
            # R2 = 1 - SS_res / SS_tot
            # SS_res = sum((y_true - y_pred)^2)
            # SS_tot = sum((y_true - mean(y_true))^2)
            pred_flat = y_pred.reshape(-1)
            true_flat = y_true.reshape(-1)
            local_n = torch.tensor([len(true_flat)], dtype=torch.float64, device=self.device)
            local_sum = torch.tensor([true_flat.sum().item()], dtype=torch.float64, device=self.device)
            local_ss_res = torch.tensor([((true_flat - pred_flat) ** 2).sum().item()], dtype=torch.float64, device=self.device)
            local_mae_sum = torch.tensor([torch.abs(true_flat - pred_flat).sum().item()], dtype=torch.float64, device=self.device)
            local_sum_sq = torch.tensor([(true_flat ** 2).sum().item()], dtype=torch.float64, device=self.device)

            # All-reduce to get global values
            dist.all_reduce(local_n, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_ss_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_mae_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM)

            global_n = local_n.item()
            global_mean = local_sum.item() / max(global_n, 1)
            global_ss_res = local_ss_res.item()
            # SS_tot = sum(y^2) - n * mean^2 = sum(y^2) - (sum(y))^2 / n
            global_ss_tot = local_sum_sq.item() - (local_sum.item() ** 2) / max(global_n, 1)

            r2 = 1.0 - global_ss_res / max(global_ss_tot, 1e-8)
            mae = local_mae_sum.item() / max(global_n, 1)
            loss = total_loss / max(n_batches, 1)
        else:
            loss = total_loss / max(n_batches, 1)
            r2 = compute_r2(y_pred, y_true)
            mae = compute_mae(y_pred, y_true)

        return loss, r2, mae

    @torch.no_grad()
    def compute_dsp_metrics(
        self,
        dataloader: DataLoader,
        sample_rate: float = 1000.0,
    ) -> Dict[str, Any]:
        """Compute comprehensive DSP metrics on validation set.

        Args:
            dataloader: Validation data loader
            sample_rate: Sampling rate in Hz

        Returns:
            Dictionary of DSP metrics
        """
        # Only compute on main process
        if not self._is_main:
            return {}

        if not DSP_AVAILABLE:
            self._log_warning("DSP metrics module not available")
            return {}

        self.model.eval()
        all_preds = []
        all_targets = []

        for x, y in dataloader:
            x = x.to(self.device)
            y_pred = self.model(x)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.numpy())

        # Concatenate
        y_pred = np.concatenate(all_preds, axis=0)  # [N, C, T]
        y_true = np.concatenate(all_targets, axis=0)

        # Compute batch metrics (sample subset for efficiency)
        N, C, T = y_pred.shape
        max_samples = min(N, 50)
        sample_indices = np.linspace(0, N - 1, max_samples, dtype=int)

        # Average across channels, compute per sample
        y_pred_avg = np.mean(y_pred[sample_indices], axis=1)  # [N_sample, T]
        y_true_avg = np.mean(y_true[sample_indices], axis=1)

        try:
            dsp = compute_batch_metrics(
                y_true_avg, y_pred_avg,
                fs=sample_rate,
                nperseg=min(256, T // 4)
            )

            # Extract key metrics with proper naming
            result = {
                "plv_mean": dsp.get("plv_mean", 0),
                "pli_mean": dsp.get("pli_mean", 0),
                "wpli_mean": dsp.get("wpli_mean", 0),
                "coherence_mean": dsp.get("coherence_mean_mean", 0),
                "envelope_corr_mean": dsp.get("envelope_correlation_mean", 0),
                "reconstruction_snr_db": dsp.get("reconstruction_snr_db_mean", 0),
                "psd_correlation": dsp.get("psd_correlation_mean", 0),
                "mutual_info": dsp.get("normalized_mi_mean", 0),
                "cross_corr_max": dsp.get("cross_correlation_max_mean", 0),
            }

            return result

        except Exception as e:
            self._log_warning(f"DSP metrics computation failed: {e}")
            return {}

    def load_checkpoint(self, path: Path) -> int:
        """Load checkpoint and return epoch to resume from.

        Args:
            path: Path to checkpoint file

        Returns:
            Epoch number to resume from
        """
        if not path.exists():
            self._log_warning(f"Checkpoint not found: {path}")
            return 0

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_val_r2 = checkpoint.get('best_val_r2', -float('inf'))
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        self._log_info(f"Loaded checkpoint from epoch {self.start_epoch}")
        return self.start_epoch

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        arch_name: str = "model",
        fold: int = 0,
        verbose: bool = True,
        resume_from: Optional[Path] = None,
        checkpoint_every: int = 10,
    ) -> TrainingResult:
        """Train model with robust error handling.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            arch_name: Architecture name (for logging)
            fold: Cross-validation fold index
            verbose: Whether to print progress
            resume_from: Optional checkpoint path to resume from
            checkpoint_every: Save checkpoint every N epochs

        Returns:
            TrainingResult with all metrics
        """
        self._log_info(f"Starting training for {arch_name} (fold={fold})")
        self._log_info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        if self.use_fsdp:
            self._log_info(f"Using FSDP with {self.world_size} GPUs")

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

                # Set epoch for distributed sampler (if using)
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)

                # Train
                train_loss, nan_in_epoch = self.train_epoch(train_loader)

                # Handle excessive NaN
                if self.nan_recovery_count > 100:
                    error_message = "Too many NaN values detected, stopping training"
                    self._log_error(error_message)
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

                    # Get model state dict (FSDP-aware)
                    if self.use_fsdp:
                        self._best_model_state = get_fsdp_state_dict(self.model)
                    else:
                        self._best_model_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }

                    # Save best checkpoint (only on main process)
                    if self.checkpoint_dir and self._is_main:
                        self._save_checkpoint(arch_name, fold, is_best=True, epoch=epoch+1)
                else:
                    self.patience_counter += 1

                # Periodic checkpoint (only on main process)
                if self.checkpoint_dir and (epoch + 1) % checkpoint_every == 0 and self._is_main:
                    self._save_checkpoint(arch_name, fold, is_best=False, epoch=epoch+1)

                # Log (only on main process)
                epoch_time = time.time() - epoch_start
                if verbose and (epoch + 1) % 5 == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    mem_str = f", GPU={self.peak_gpu_memory_mb:.0f}MB" if "cuda" in str(self.device) else ""
                    self._log_info(
                        f"Epoch {epoch+1:3d}/{self.config.epochs}: "
                        f"train_loss={train_loss:.4f}, val_r2={val_r2:.4f}, "
                        f"val_mae={val_mae:.4f}, lr={lr:.2e}, "
                        f"time={epoch_time:.1f}s{mem_str}"
                    )

                # Early stopping
                if self.patience_counter >= self.config.patience:
                    self._log_info(f"Early stopping at epoch {epoch+1}")
                    early_stopped = True
                    break

                # Sync all processes at end of epoch
                if self.use_fsdp:
                    dist.barrier()

        except KeyboardInterrupt:
            self._log_warning("Training interrupted by user")
            error_message = "Interrupted by user"
            completed_successfully = False

        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e)}"
            self._log_error(f"Training failed: {error_message}")
            self._log_error(traceback.format_exc())
            completed_successfully = False

        total_time = time.time() - start_time

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Restore best model if available (not done for FSDP as state is sharded)
        if self._best_model_state is not None and completed_successfully and not self.use_fsdp:
            self.model.load_state_dict(self._best_model_state)
            self._log_info(f"Restored best model from epoch {best_epoch}")

        # Compute DSP metrics on final predictions (main process only)
        dsp_metrics = {}
        if completed_successfully and DSP_AVAILABLE and self._is_main:
            self._log_info("Computing DSP metrics...")
            dsp_metrics = self.compute_dsp_metrics(val_loader)
            if dsp_metrics:
                self._log_info(
                    f"DSP: PLV={dsp_metrics.get('plv_mean', 0):.4f}, "
                    f"Coh={dsp_metrics.get('coherence_mean', 0):.4f}, "
                    f"SNR={dsp_metrics.get('reconstruction_snr_db', 0):.1f}dB"
                )

        self._log_info(
            f"Training completed: best_val_r2={best_val_r2:.4f} at epoch {best_epoch}, "
            f"total_time={total_time/60:.1f}min"
        )

        return TrainingResult(
            architecture=arch_name,
            fold=fold,
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
            peak_gpu_memory_mb=self.peak_gpu_memory_mb if "cuda" in str(self.device) else None,
            completed_successfully=completed_successfully,
            dsp_metrics=dsp_metrics if dsp_metrics else None,
        )

    def _save_checkpoint(
        self,
        arch_name: str,
        fold: int,
        is_best: bool = True,
        epoch: int = 0,
    ):
        """Save model checkpoint.

        Args:
            arch_name: Architecture name
            fold: Cross-validation fold index
            is_best: Whether this is the best model checkpoint
            epoch: Current epoch number

        Note:
            For FSDP, only the main process (rank 0) saves checkpoints.
            get_fsdp_state_dict() must be called on all ranks, but the
            actual saving is done only on rank 0.
        """
        if self.checkpoint_dir is None:
            return

        # Only main process saves
        if not self._is_main:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get model state dict (FSDP-aware)
        if self.use_fsdp:
            model_state = get_fsdp_state_dict(self.model)
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_val_r2': self.best_val_r2,
            'best_val_loss': self.best_val_loss,
            'architecture': arch_name,
            'fold': fold,
            'timestamp': datetime.now().isoformat(),
            'fsdp': self.use_fsdp,
        }

        if is_best:
            path = self.checkpoint_dir / f"{arch_name}_fold{fold}_best.pt"
        else:
            path = self.checkpoint_dir / f"{arch_name}_fold{fold}_epoch{epoch}.pt"

        torch.save(checkpoint, path)
        self.logger.debug(f"Saved checkpoint: {path}")


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    n_workers: int = 4,
    use_distributed: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch data loaders.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size
        n_workers: Number of data loader workers
        use_distributed: Use DistributedSampler for multi-GPU training

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

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    shuffle_train = True

    if use_distributed and is_distributed():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False,
            drop_last=False,
        )
        shuffle_train = False  # Sampler handles shuffling

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=n_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
