"""
Sliding Window Processing for Phase 5
======================================

Implements sliding window data loading and continuous inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import SlidingWindowConfig


# =============================================================================
# Sliding Window Dataset
# =============================================================================

class SlidingWindowDataset(Dataset):
    """Dataset that extracts sliding windows from continuous signals.

    Args:
        source: Source signal [n_samples, n_channels] or [n_channels, n_samples]
        target: Target signal [n_samples, n_channels] or [n_channels, n_samples]
        window_size: Window size in samples
        stride: Stride in samples
        channel_first: Whether input is channel-first format

    Example:
        >>> dataset = SlidingWindowDataset(source, target, window_size=5000, stride=2500)
        >>> x, y = dataset[0]
        >>> print(x.shape)  # [n_channels, window_size]
    """

    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        window_size: int,
        stride: int,
        channel_first: bool = False,
    ):
        # Convert to channel-last if needed
        if channel_first:
            source = source.T
            target = target.T

        self.source = source  # [n_samples, n_channels]
        self.target = target
        self.window_size = window_size
        self.stride = stride

        # Compute number of windows
        n_samples = source.shape[0]
        self.n_windows = max(1, (n_samples - window_size) // stride + 1)

        # Store metadata
        self.n_samples = n_samples
        self.n_channels_source = source.shape[1]
        self.n_channels_target = target.shape[1]

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.window_size

        # Extract window
        x = self.source[start:end].T  # [n_channels, window_size]
        y = self.target[start:end].T

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    @property
    def coverage(self) -> float:
        """Fraction of signal covered by windows."""
        covered = min(self.n_windows * self.stride + self.window_size, self.n_samples)
        return covered / self.n_samples


def create_sliding_windows(
    source: np.ndarray,
    target: np.ndarray,
    config: SlidingWindowConfig,
    channel_first: bool = False,
) -> SlidingWindowDataset:
    """Create sliding window dataset from continuous signals.

    Args:
        source: Source signal array
        target: Target signal array
        config: Sliding window configuration
        channel_first: Whether arrays are channel-first

    Returns:
        SlidingWindowDataset
    """
    return SlidingWindowDataset(
        source=source,
        target=target,
        window_size=config.window_size,
        stride=config.stride,
        channel_first=channel_first,
    )


# =============================================================================
# Continuous Inference
# =============================================================================

class ContinuousInference:
    """Handles continuous real-time inference with sliding windows.

    Args:
        model: Trained model
        window_size: Window size in samples
        stride: Stride in samples
        device: Device for inference

    Example:
        >>> inference = ContinuousInference(model, window_size=5000, stride=2500)
        >>> for pred in inference.process(signal):
        ...     # pred is prediction for each window
    """

    def __init__(
        self,
        model: nn.Module,
        window_size: int,
        stride: int,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.window_size = window_size
        self.stride = stride
        self.device = device

        # Buffer for incoming samples
        self.buffer = None
        self.n_channels = None

    def reset(self):
        """Reset the buffer."""
        self.buffer = None

    @torch.no_grad()
    def process_window(self, window: np.ndarray) -> np.ndarray:
        """Process a single window.

        Args:
            window: Input window [n_channels, window_size]

        Returns:
            Prediction [n_channels, window_size]
        """
        x = torch.tensor(window, dtype=torch.float32, device=self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension

        pred = self.model(x)
        return pred.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def process_continuous(
        self,
        signal: np.ndarray,
        channel_first: bool = True,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Process continuous signal with sliding windows.

        Args:
            signal: Continuous signal [n_channels, n_samples] or [n_samples, n_channels]
            channel_first: Whether signal is channel-first

        Yields:
            (start_sample, prediction) tuples
        """
        if not channel_first:
            signal = signal.T

        n_channels, n_samples = signal.shape

        # Process windows
        start = 0
        while start + self.window_size <= n_samples:
            window = signal[:, start:start + self.window_size]
            pred = self.process_window(window)
            yield start, pred
            start += self.stride

    @torch.no_grad()
    def reconstruct_continuous(
        self,
        signal: np.ndarray,
        channel_first: bool = True,
        overlap_mode: str = "average",
    ) -> np.ndarray:
        """Reconstruct continuous prediction from overlapping windows.

        Args:
            signal: Input signal
            channel_first: Whether signal is channel-first
            overlap_mode: How to handle overlaps ("average", "last", "first")

        Returns:
            Reconstructed continuous prediction
        """
        if not channel_first:
            signal = signal.T

        n_channels, n_samples = signal.shape

        # Allocate output
        output = np.zeros((n_channels, n_samples), dtype=np.float32)
        counts = np.zeros(n_samples, dtype=np.float32)

        # Process all windows
        for start, pred in self.process_continuous(signal, channel_first=True):
            end = start + self.window_size

            if overlap_mode == "average":
                output[:, start:end] += pred
                counts[start:end] += 1
            elif overlap_mode == "last":
                output[:, start:end] = pred
            elif overlap_mode == "first":
                mask = counts[start:end] == 0
                output[:, start:end][:, mask] = pred[:, mask]
                counts[start:end] = np.maximum(counts[start:end], 1)

        # Normalize for averaging
        if overlap_mode == "average":
            counts = np.maximum(counts, 1)
            output = output / counts

        return output


def continuous_inference(
    model: nn.Module,
    signal: np.ndarray,
    window_size: int,
    stride: int,
    device: str = "cuda",
    overlap_mode: str = "average",
) -> np.ndarray:
    """Convenience function for continuous inference.

    Args:
        model: Trained model
        signal: Input signal [n_channels, n_samples]
        window_size: Window size
        stride: Stride
        device: Device
        overlap_mode: Overlap handling mode

    Returns:
        Continuous prediction
    """
    inference = ContinuousInference(model, window_size, stride, device)
    return inference.reconstruct_continuous(signal, overlap_mode=overlap_mode)


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_continuous_signal(
    duration_sec: float,
    sample_rate: float = 1000.0,
    n_channels: int = 32,
    add_structure: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic continuous neural signal.

    Args:
        duration_sec: Duration in seconds
        sample_rate: Sampling rate in Hz
        n_channels: Number of channels
        add_structure: Whether to add oscillatory structure

    Returns:
        source, target arrays [n_channels, n_samples]
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Generate source with oscillations
    source = np.random.randn(n_channels, n_samples) * 0.5

    if add_structure:
        # Add oscillations at different frequencies
        for i, freq in enumerate([4, 8, 12, 30, 60]):
            phase = np.random.rand(n_channels, 1) * 2 * np.pi
            amp = np.random.rand(n_channels, 1) * 0.3
            source += amp * np.sin(2 * np.pi * freq * t + phase)

    # Generate target as filtered/transformed source
    target = source.copy()

    # Add some transformation
    target = target + 0.3 * np.random.randn(n_channels, n_samples)

    # Smooth target
    kernel_size = 51
    kernel = np.ones(kernel_size) / kernel_size
    for c in range(n_channels):
        target[c] = np.convolve(target[c], kernel, mode='same')

    return source.astype(np.float32), target.astype(np.float32)


def generate_synthetic_dataset(
    n_samples: int = 1000,
    window_size: int = 5000,
    n_channels: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic dataset for testing.

    Args:
        n_samples: Number of trial samples
        window_size: Length of each sample
        n_channels: Number of channels

    Returns:
        source, target arrays [n_samples, n_channels, window_size]
    """
    sources = []
    targets = []

    for _ in range(n_samples):
        src, tgt = generate_continuous_signal(
            duration_sec=window_size / 1000.0,
            n_channels=n_channels,
        )
        sources.append(src)
        targets.append(tgt)

    return np.stack(sources), np.stack(targets)
