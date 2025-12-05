"""
Robust Error Handling for Nature Methods HPO
=============================================

Production-grade error handling with logging, recovery, and cleanup.
Every trial MUST use these wrappers - ZERO tolerance for silent failures.
"""

from __future__ import annotations

import functools
import gc
import logging
import sys
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar

import torch

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    optuna = None
    TrialPruned = Exception

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TrialError(Exception):
    """Base exception for trial failures."""

    def __init__(self, message: str, trial_number: Optional[int] = None):
        self.trial_number = trial_number
        super().__init__(message)


class OOMError(TrialError):
    """Out of memory error."""
    pass


class NaNError(TrialError):
    """NaN/Inf detected in loss or gradients."""
    pass


class ConvergenceError(TrialError):
    """Model failed to converge."""
    pass


def cleanup_gpu_memory(device: Optional[torch.device] = None) -> None:
    """Aggressively clean up GPU memory.

    Args:
        device: Specific device to clean (None = all)
    """
    gc.collect()

    if torch.cuda.is_available():
        if device is not None and device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def get_gpu_memory_info(device: torch.device) -> Dict[str, float]:
    """Get GPU memory usage info.

    Args:
        device: CUDA device

    Returns:
        Dictionary with memory stats in GB
    """
    if device.type != "cuda":
        return {}

    with torch.cuda.device(device):
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9

    return {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "max_allocated_gb": round(max_allocated, 3),
        "total_gb": round(total, 3),
        "available_gb": round(total - reserved, 3),
    }


@contextmanager
def trial_context(
    trial: Optional["optuna.Trial"] = None,
    device: Optional[torch.device] = None,
    cleanup_on_exit: bool = True,
):
    """Context manager for safe trial execution.

    Handles cleanup and error logging automatically.

    Args:
        trial: Optuna trial (for logging)
        device: GPU device for cleanup
        cleanup_on_exit: Whether to clean GPU memory on exit

    Yields:
        None

    Raises:
        TrialPruned: On any error (for Optuna compatibility)
    """
    trial_num = trial.number if trial else "N/A"

    try:
        if device and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        yield

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"Trial {trial_num}: OOM - {e}")
        if trial:
            trial.set_user_attr("error", "OOM")
            trial.set_user_attr("error_message", str(e))
        cleanup_gpu_memory(device)
        raise TrialPruned(f"OOM: {e}")

    except NaNError as e:
        logger.error(f"Trial {trial_num}: NaN detected - {e}")
        if trial:
            trial.set_user_attr("error", "NaN")
            trial.set_user_attr("error_message", str(e))
        raise TrialPruned(f"NaN: {e}")

    except ConvergenceError as e:
        logger.error(f"Trial {trial_num}: Convergence failed - {e}")
        if trial:
            trial.set_user_attr("error", "convergence")
            trial.set_user_attr("error_message", str(e))
        raise TrialPruned(f"Convergence: {e}")

    except TrialPruned:
        # Re-raise pruning (this is expected behavior)
        raise

    except KeyboardInterrupt:
        logger.warning(f"Trial {trial_num}: Interrupted by user")
        raise

    except Exception as e:
        # Log full traceback for unexpected errors
        logger.error(
            f"Trial {trial_num}: Unexpected error - {type(e).__name__}: {e}\n"
            f"{traceback.format_exc()}"
        )
        if trial:
            trial.set_user_attr("error", type(e).__name__)
            trial.set_user_attr("error_message", str(e))
            trial.set_user_attr("traceback", traceback.format_exc())
        raise TrialPruned(f"Unexpected error: {e}")

    finally:
        if cleanup_on_exit:
            cleanup_gpu_memory(device)

        if device and device.type == "cuda" and trial:
            mem_info = get_gpu_memory_info(device)
            trial.set_user_attr("peak_memory_gb", mem_info.get("max_allocated_gb", 0))


def robust_objective(
    func: Callable[..., float],
) -> Callable[..., float]:
    """Decorator to make objective functions robust.

    Wraps objective with error handling, cleanup, and logging.

    Args:
        func: Objective function to wrap

    Returns:
        Wrapped function
    """

    @functools.wraps(func)
    def wrapper(trial: "optuna.Trial", *args, **kwargs) -> float:
        device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        with trial_context(trial, device):
            return func(trial, *args, **kwargs)

    return wrapper


def check_loss_valid(
    loss: torch.Tensor,
    epoch: int,
    trial: Optional["optuna.Trial"] = None,
) -> None:
    """Check if loss is valid, raise if not.

    Args:
        loss: Loss tensor
        epoch: Current epoch (for logging)
        trial: Optuna trial (for attributes)

    Raises:
        NaNError: If loss is NaN or Inf
    """
    if torch.isnan(loss):
        if trial:
            trial.set_user_attr("nan_epoch", epoch)
        raise NaNError(f"Loss became NaN at epoch {epoch}")

    if torch.isinf(loss):
        if trial:
            trial.set_user_attr("inf_epoch", epoch)
        raise NaNError(f"Loss became Inf at epoch {epoch}")


def check_gradients_valid(
    model: torch.nn.Module,
    epoch: int,
    trial: Optional["optuna.Trial"] = None,
    max_grad_norm: float = 1000.0,
) -> None:
    """Check if gradients are valid.

    Args:
        model: Model to check
        epoch: Current epoch
        trial: Optuna trial
        max_grad_norm: Maximum allowed gradient norm

    Raises:
        NaNError: If gradients are invalid
    """
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                if trial:
                    trial.set_user_attr("nan_grad_param", name)
                    trial.set_user_attr("nan_grad_epoch", epoch)
                raise NaNError(f"NaN gradient in {name} at epoch {epoch}")

            if torch.isinf(param.grad).any():
                if trial:
                    trial.set_user_attr("inf_grad_param", name)
                    trial.set_user_attr("inf_grad_epoch", epoch)
                raise NaNError(f"Inf gradient in {name} at epoch {epoch}")

            total_norm += param.grad.data.norm(2).item() ** 2

    total_norm = total_norm ** 0.5

    if total_norm > max_grad_norm:
        logger.warning(
            f"Epoch {epoch}: Gradient norm {total_norm:.2f} exceeds {max_grad_norm}"
        )


def safe_backward(
    loss: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_grad_norm: float = 5.0,
    epoch: int = 0,
    trial: Optional["optuna.Trial"] = None,
) -> float:
    """Safe backward pass with gradient clipping and validation.

    Args:
        loss: Loss tensor
        model: Model
        optimizer: Optimizer
        clip_grad_norm: Max gradient norm for clipping
        epoch: Current epoch
        trial: Optuna trial

    Returns:
        Loss value

    Raises:
        NaNError: If loss or gradients are invalid
    """
    # Check loss before backward
    check_loss_valid(loss, epoch, trial)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Check gradients
    check_gradients_valid(model, epoch, trial)

    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

    # Step optimizer
    optimizer.step()

    return loss.item()


class EarlyStopping:
    """Early stopping with patience and best model tracking."""

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        """Initialize early stopping.

        Args:
            patience: Epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "max" or "min" for metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, score: float, epoch: int) -> bool:
        """Update early stopping state.

        Args:
            score: Current metric value
            epoch: Current epoch

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """Setup logging for HPO runs.

    Args:
        log_file: Path to log file (None = stdout only)
        level: Logging level
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    # Reduce verbosity of some libraries
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
