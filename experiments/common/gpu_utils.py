"""
GPU Utilities for Accelerated Computing
========================================

Provides CuPy GPU acceleration with automatic fallback to NumPy.

Usage:
    from experiments.common.gpu_utils import get_array_module, to_numpy

    xp = get_array_module(use_gpu=True)  # Returns cupy if available, else numpy
    arr = xp.array([1, 2, 3])
    result = to_numpy(arr)  # Always returns numpy array
"""

from __future__ import annotations

from typing import Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Try to import CuPy
_CUPY_AVAILABLE = False
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None


def is_cupy_available() -> bool:
    """Check if CuPy is available."""
    return _CUPY_AVAILABLE


def get_array_module(use_gpu: bool = True):
    """Get the array module to use (cupy or numpy).

    Args:
        use_gpu: Whether to try using GPU acceleration

    Returns:
        cupy module if available and use_gpu=True, else numpy
    """
    if use_gpu and _CUPY_AVAILABLE:
        return cp
    return np


def to_numpy(arr) -> np.ndarray:
    """Convert array to numpy (handles both cupy and numpy arrays).

    Args:
        arr: Input array (numpy or cupy)

    Returns:
        NumPy array
    """
    if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return arr.get()
    return np.asarray(arr)


def to_gpu(arr: np.ndarray):
    """Convert numpy array to GPU array if CuPy is available.

    Args:
        arr: NumPy array

    Returns:
        CuPy array if available, else the original numpy array
    """
    if _CUPY_AVAILABLE:
        return cp.asarray(arr)
    return arr


class ArrayModule:
    """Context manager for GPU array operations with automatic memory cleanup.

    Usage:
        with ArrayModule(use_gpu=True) as xp:
            X_gpu = xp.asarray(X)
            result = xp.linalg.solve(A, b)
            result_cpu = to_numpy(result)
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np

    def __enter__(self):
        return self.xp

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clear GPU memory cache
        if self.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()
        return False


def gpu_fft(x: np.ndarray, n: int = None, axis: int = -1) -> np.ndarray:
    """Compute FFT with GPU acceleration if available.

    Args:
        x: Input array
        n: Length of FFT
        axis: Axis along which to compute FFT

    Returns:
        FFT result as numpy array
    """
    if _CUPY_AVAILABLE:
        x_gpu = cp.asarray(x)
        result = cp.fft.fft(x_gpu, n=n, axis=axis)
        return result.get()
    return np.fft.fft(x, n=n, axis=axis)


def gpu_ifft(x: np.ndarray, n: int = None, axis: int = -1) -> np.ndarray:
    """Compute inverse FFT with GPU acceleration if available.

    Args:
        x: Input array
        n: Length of IFFT
        axis: Axis along which to compute IFFT

    Returns:
        IFFT result as numpy array
    """
    if _CUPY_AVAILABLE:
        x_gpu = cp.asarray(x)
        result = cp.fft.ifft(x_gpu, n=n, axis=axis)
        return result.get()
    return np.fft.ifft(x, n=n, axis=axis)


def gpu_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve linear system Ax = b with GPU acceleration.

    Args:
        a: Coefficient matrix
        b: Ordinate values

    Returns:
        Solution x as numpy array
    """
    if _CUPY_AVAILABLE:
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        result = cp.linalg.solve(a_gpu, b_gpu)
        return result.get()
    return np.linalg.solve(a, b)


def gpu_inv(a: np.ndarray) -> np.ndarray:
    """Compute matrix inverse with GPU acceleration.

    Args:
        a: Matrix to invert

    Returns:
        Inverse matrix as numpy array
    """
    if _CUPY_AVAILABLE:
        a_gpu = cp.asarray(a)
        result = cp.linalg.inv(a_gpu)
        return result.get()
    return np.linalg.inv(a)


def gpu_einsum(subscripts: str, *operands) -> np.ndarray:
    """Compute einsum with GPU acceleration.

    Args:
        subscripts: Einsum subscripts string
        *operands: Input arrays

    Returns:
        Result as numpy array
    """
    if _CUPY_AVAILABLE:
        operands_gpu = [cp.asarray(op) for op in operands]
        result = cp.einsum(subscripts, *operands_gpu)
        return result.get()
    return np.einsum(subscripts, *operands)


def gpu_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiplication with GPU acceleration.

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        Result as numpy array
    """
    if _CUPY_AVAILABLE:
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        result = cp.matmul(a_gpu, b_gpu)
        return result.get()
    return np.matmul(a, b)
