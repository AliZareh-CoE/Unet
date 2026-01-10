"""
Phase 5: Real-Time Continuous Processing
=========================================

This module implements real-time continuous signal translation using
sliding windows, with latency and throughput benchmarks.

Experiments:
    - Window size ablation: 1000, 2500, 5000 samples
    - Stride ratio ablation: 0.25, 0.5, 0.75
    - Latency benchmarks: batch sizes 1, 4, 16, 64
    - Fixed vs sliding window comparison

Datasets:
    - Olfactory (OB → PCx)
    - PFC (PFC → CA1)
    - DANDI (AMY → HPC)

Usage:
    python -m phase_five.runner --epochs 60

Author: Neural Signal Translation Team
"""

__version__ = "1.0.0"

from .config import (
    Phase5Config,
    SlidingWindowConfig,
    BenchmarkConfig,
    WINDOW_SIZES,
    STRIDE_RATIOS,
    BATCH_SIZES,
)
from .sliding_window import (
    SlidingWindowDataset,
    create_sliding_windows,
    continuous_inference,
)
from .benchmark import (
    LatencyBenchmark,
    run_latency_benchmark,
    run_throughput_benchmark,
    run_memory_benchmark,
)
from .runner import run_phase5, Phase5Result
from .visualization import Phase5Visualizer

__all__ = [
    "Phase5Config",
    "SlidingWindowConfig",
    "BenchmarkConfig",
    "WINDOW_SIZES",
    "STRIDE_RATIOS",
    "BATCH_SIZES",
    "SlidingWindowDataset",
    "create_sliding_windows",
    "continuous_inference",
    "LatencyBenchmark",
    "run_latency_benchmark",
    "run_throughput_benchmark",
    "run_memory_benchmark",
    "run_phase5",
    "Phase5Result",
    "Phase5Visualizer",
]
