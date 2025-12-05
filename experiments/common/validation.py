"""
Input/Output Validation for Nature Methods HPO
===============================================

Robust validation utilities to catch errors before they propagate.
ZERO tolerance for runtime errors - validate everything upfront.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[Optional[int], ...],
    name: str = "tensor",
) -> None:
    """Validate tensor has expected shape.

    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (None = any size for that dim)
        name: Name for error messages

    Raises:
        ValidationError: If shape doesn't match
    """
    if len(tensor.shape) != len(expected_shape):
        raise ValidationError(
            f"{name} has {len(tensor.shape)} dims, expected {len(expected_shape)}. "
            f"Shape: {tensor.shape}, expected: {expected_shape}"
        )

    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValidationError(
                f"{name} dim {i} is {actual}, expected {expected}. "
                f"Shape: {tensor.shape}, expected: {expected_shape}"
            )


def validate_not_nan_inf(
    tensor: torch.Tensor,
    name: str = "tensor",
) -> None:
    """Check tensor has no NaN or Inf values.

    Args:
        tensor: Tensor to check
        name: Name for error messages

    Raises:
        ValidationError: If NaN or Inf found
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        raise ValidationError(
            f"{name} contains {nan_count} NaN values "
            f"(shape: {tensor.shape}, dtype: {tensor.dtype})"
        )

    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        raise ValidationError(
            f"{name} contains {inf_count} Inf values "
            f"(shape: {tensor.shape}, dtype: {tensor.dtype})"
        )


def validate_finite_gradients(
    model: nn.Module,
    name: str = "model",
) -> None:
    """Check all gradients are finite.

    Args:
        model: Model to check
        name: Name for error messages

    Raises:
        ValidationError: If any gradient is NaN or Inf
    """
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                raise ValidationError(
                    f"{name}.{param_name} has NaN gradients"
                )
            if torch.isinf(param.grad).any():
                raise ValidationError(
                    f"{name}.{param_name} has Inf gradients"
                )


def validate_model_output(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
    expected_channels: int = 32,
    name: str = "model_output",
) -> None:
    """Validate model output shape and values.

    Args:
        output: Model output tensor
        input_tensor: Input tensor (for batch size reference)
        expected_channels: Expected number of output channels
        name: Name for error messages
    """
    B = input_tensor.shape[0]

    # Check shape
    validate_tensor_shape(
        output,
        (B, expected_channels, None),
        name=name,
    )

    # Check finite values
    validate_not_nan_inf(output, name=name)


def validate_loss(
    loss: torch.Tensor,
    name: str = "loss",
) -> None:
    """Validate loss is finite and scalar.

    Args:
        loss: Loss tensor
        name: Name for error messages

    Raises:
        ValidationError: If loss is invalid
    """
    if loss.numel() != 1:
        raise ValidationError(
            f"{name} is not scalar, has shape {loss.shape}"
        )

    if torch.isnan(loss):
        raise ValidationError(f"{name} is NaN")

    if torch.isinf(loss):
        raise ValidationError(f"{name} is Inf")

    if loss.item() < 0:
        logger.warning(f"{name} is negative: {loss.item():.6f}")


def validate_hyperparameters(
    config: Dict[str, Any],
    required_keys: List[str],
    name: str = "config",
) -> None:
    """Validate hyperparameter configuration.

    Args:
        config: Configuration dictionary
        required_keys: Keys that must be present
        name: Name for error messages

    Raises:
        ValidationError: If required keys missing
    """
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValidationError(
            f"{name} missing required keys: {missing}"
        )


def validate_data_loader(
    loader,
    expected_batch_keys: Tuple[str, ...] = ("ob", "pcx", "odor"),
    name: str = "loader",
) -> None:
    """Validate data loader produces expected format.

    Args:
        loader: DataLoader to validate
        expected_batch_keys: Expected keys in each batch
        name: Name for error messages
    """
    try:
        batch = next(iter(loader))
    except StopIteration:
        raise ValidationError(f"{name} is empty")

    if isinstance(batch, (tuple, list)):
        if len(batch) != len(expected_batch_keys):
            raise ValidationError(
                f"{name} yields {len(batch)} items, expected {len(expected_batch_keys)}"
            )
        for i, item in enumerate(batch):
            if not isinstance(item, torch.Tensor):
                raise ValidationError(
                    f"{name} item {i} is {type(item)}, expected Tensor"
                )
            validate_not_nan_inf(item, f"{name}[{i}]")
    else:
        raise ValidationError(
            f"{name} yields {type(batch)}, expected tuple/list"
        )


def validate_memory_available(
    estimated_gb: float,
    device: torch.device,
    safety_margin: float = 0.2,
) -> None:
    """Check if enough GPU memory is available.

    Args:
        estimated_gb: Estimated memory needed in GB
        device: Target device
        safety_margin: Additional margin (fraction)

    Raises:
        ValidationError: If insufficient memory
    """
    if device.type != "cuda":
        return  # No GPU memory to check

    # Get available memory
    torch.cuda.synchronize(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
    used_mem = torch.cuda.memory_allocated(device) / 1e9
    available_mem = total_mem - used_mem

    required_mem = estimated_gb * (1 + safety_margin)

    if available_mem < required_mem:
        raise ValidationError(
            f"Insufficient GPU memory: {available_mem:.2f}GB available, "
            f"{required_mem:.2f}GB required (with {safety_margin*100:.0f}% margin)"
        )

    logger.info(
        f"Memory check passed: {available_mem:.2f}GB available, "
        f"{estimated_gb:.2f}GB estimated"
    )


def estimate_model_memory(
    model: nn.Module,
    batch_size: int,
    input_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> float:
    """Estimate peak memory usage in GB.

    Args:
        model: Model to estimate
        batch_size: Batch size
        input_shape: Input shape (C, T)
        dtype: Data type

    Returns:
        Estimated memory in GB
    """
    # Parameter memory
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters())

    # Gradient memory (same as params)
    grad_mem = param_mem

    # Optimizer states (Adam: 2x params for momentum and variance)
    optim_mem = 2 * param_mem

    # Input/output memory
    bytes_per_element = 4 if dtype == torch.float32 else 2
    input_mem = batch_size * np.prod(input_shape) * bytes_per_element
    output_mem = input_mem  # Assume same size output

    # Activation memory (rough estimate: 4x input for U-Net style)
    activation_mem = 4 * input_mem

    total_bytes = param_mem + grad_mem + optim_mem + 2 * input_mem + activation_mem
    return total_bytes / 1e9


def run_forward_pass_test(
    model: nn.Module,
    input_shape: Tuple[int, int, int],  # (B, C, T)
    device: torch.device,
    odor_classes: int = 7,
) -> bool:
    """Run smoke test forward pass.

    Args:
        model: Model to test
        input_shape: Input shape (B, C, T)
        device: Target device
        odor_classes: Number of odor classes

    Returns:
        True if test passes

    Raises:
        ValidationError: If test fails
    """
    B, C, T = input_shape

    try:
        model.eval()
        with torch.no_grad():
            x = torch.randn(B, C, T, device=device)
            odor = torch.randint(0, odor_classes, (B,), device=device)

            output = model(x, odor)

            validate_model_output(output, x, expected_channels=C)

        logger.info(f"Forward pass test passed: {input_shape} -> {output.shape}")
        return True

    except Exception as e:
        raise ValidationError(f"Forward pass test failed: {e}")


def run_gradient_flow_test(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    device: torch.device,
    odor_classes: int = 7,
) -> bool:
    """Test that gradients flow through the model.

    Args:
        model: Model to test
        input_shape: Input shape (B, C, T)
        device: Target device
        odor_classes: Number of odor classes

    Returns:
        True if test passes

    Raises:
        ValidationError: If test fails
    """
    B, C, T = input_shape

    try:
        model.train()
        x = torch.randn(B, C, T, device=device, requires_grad=True)
        odor = torch.randint(0, odor_classes, (B,), device=device)

        output = model(x, odor)
        loss = output.mean()
        loss.backward()

        # Check input gradients
        if x.grad is None:
            raise ValidationError("No gradients flow to input")
        validate_not_nan_inf(x.grad, "input.grad")

        # Check model gradients
        validate_finite_gradients(model)

        # Check at least some parameters have gradients
        params_with_grad = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        total_params = sum(1 for _ in model.parameters())

        if params_with_grad == 0:
            raise ValidationError("No parameters received gradients")

        logger.info(
            f"Gradient flow test passed: {params_with_grad}/{total_params} "
            f"params have gradients"
        )
        return True

    except Exception as e:
        raise ValidationError(f"Gradient flow test failed: {e}")
    finally:
        model.zero_grad()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters.

    Args:
        model: Model to count

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": round(total / 1e6, 2),
        "trainable_millions": round(trainable / 1e6, 2),
    }
