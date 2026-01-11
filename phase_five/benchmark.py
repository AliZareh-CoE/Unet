"""
Benchmarking Utilities for Phase 5
===================================

Implements latency, throughput, and memory benchmarks.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import BenchmarkConfig, REALTIME_REQUIREMENTS


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class LatencyResult:
    """Results from latency benchmark."""

    batch_size: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n_iterations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "n_iterations": self.n_iterations,
        }

    @property
    def meets_realtime(self) -> bool:
        """Check if meets real-time requirements (batch_size=1)."""
        if self.batch_size == 1:
            return self.p95_ms < REALTIME_REQUIREMENTS["max_latency_ms"]
        return True


@dataclass
class ThroughputResult:
    """Results from throughput benchmark."""

    batch_size: int
    samples_per_sec: float
    windows_per_sec: float
    n_iterations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "samples_per_sec": self.samples_per_sec,
            "windows_per_sec": self.windows_per_sec,
            "n_iterations": self.n_iterations,
        }

    @property
    def meets_realtime(self) -> bool:
        """Check if meets real-time throughput requirements."""
        return self.samples_per_sec > REALTIME_REQUIREMENTS["min_throughput_hz"]


@dataclass
class MemoryResult:
    """Results from memory benchmark."""

    batch_size: int
    model_memory_mb: float
    peak_memory_mb: float
    allocated_memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "model_memory_mb": self.model_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "allocated_memory_mb": self.allocated_memory_mb,
        }

    @property
    def peak_memory_gb(self) -> float:
        return self.peak_memory_mb / 1024

    @property
    def meets_realtime(self) -> bool:
        """Check if meets memory requirements."""
        return self.peak_memory_gb < REALTIME_REQUIREMENTS["max_memory_gb"]


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    latency: Dict[int, LatencyResult]
    throughput: Dict[int, ThroughputResult]
    memory: Dict[int, MemoryResult]
    model_params: int
    window_size: int
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency": {k: v.to_dict() for k, v in self.latency.items()},
            "throughput": {k: v.to_dict() for k, v in self.throughput.items()},
            "memory": {k: v.to_dict() for k, v in self.memory.items()},
            "model_params": self.model_params,
            "window_size": self.window_size,
            "device": self.device,
        }

    @property
    def meets_all_requirements(self) -> bool:
        """Check if all real-time requirements are met."""
        # Check batch_size=1 latency
        if 1 in self.latency and not self.latency[1].meets_realtime:
            return False

        # Check throughput (any batch size)
        if not any(t.meets_realtime for t in self.throughput.values()):
            return False

        # Check memory (batch_size=1)
        if 1 in self.memory and not self.memory[1].meets_realtime:
            return False

        return True


# =============================================================================
# Benchmark Class
# =============================================================================

class LatencyBenchmark:
    """Benchmark latency, throughput, and memory for a model.

    Args:
        model: Model to benchmark
        config: Benchmark configuration

    Example:
        >>> benchmark = LatencyBenchmark(model, config)
        >>> results = benchmark.run_all()
    """

    def __init__(
        self,
        model: nn.Module,
        config: BenchmarkConfig,
    ):
        self.model = model.to(config.device)
        self.model.eval()
        self.config = config
        self.device = config.device

        # Count parameters
        self.n_params = sum(p.numel() for p in model.parameters())

    def run_latency(self, batch_size: int) -> LatencyResult:
        """Run latency benchmark for a specific batch size."""
        # Create input
        x = torch.randn(
            batch_size,
            self.config.in_channels,
            self.config.window_size,
            device=self.device,
        )

        latencies = []

        # Warmup
        with torch.no_grad():
            for _ in range(self.config.n_warmup):
                _ = self.model(x)

        # Synchronize before timing
        if self.device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        with torch.no_grad():
            for _ in range(self.config.n_iterations):
                if self.device == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = self.model(x)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        latencies = np.array(latencies)

        return LatencyResult(
            batch_size=batch_size,
            mean_ms=float(np.mean(latencies)),
            std_ms=float(np.std(latencies)),
            min_ms=float(np.min(latencies)),
            max_ms=float(np.max(latencies)),
            p50_ms=float(np.percentile(latencies, 50)),
            p95_ms=float(np.percentile(latencies, 95)),
            p99_ms=float(np.percentile(latencies, 99)),
            n_iterations=self.config.n_iterations,
        )

    def run_throughput(self, batch_size: int) -> ThroughputResult:
        """Run throughput benchmark."""
        x = torch.randn(
            batch_size,
            self.config.in_channels,
            self.config.window_size,
            device=self.device,
        )

        # Warmup
        with torch.no_grad():
            for _ in range(self.config.n_warmup):
                _ = self.model(x)

        if self.device == "cuda":
            torch.cuda.synchronize()

        # Measure total time for n_iterations
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(self.config.n_iterations):
                _ = self.model(x)

        if self.device == "cuda":
            torch.cuda.synchronize()

        total_time = time.perf_counter() - start

        # Calculate throughput
        total_windows = batch_size * self.config.n_iterations
        windows_per_sec = total_windows / total_time
        samples_per_sec = windows_per_sec * self.config.window_size

        return ThroughputResult(
            batch_size=batch_size,
            samples_per_sec=samples_per_sec,
            windows_per_sec=windows_per_sec,
            n_iterations=self.config.n_iterations,
        )

    def run_memory(self, batch_size: int) -> MemoryResult:
        """Run memory benchmark."""
        if self.device != "cuda":
            return MemoryResult(
                batch_size=batch_size,
                model_memory_mb=0,
                peak_memory_mb=0,
                allocated_memory_mb=0,
            )

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

        # Measure model memory
        model_memory = torch.cuda.memory_allocated() / 1024 / 1024

        # Run inference to measure peak
        x = torch.randn(
            batch_size,
            self.config.in_channels,
            self.config.window_size,
            device=self.device,
        )

        with torch.no_grad():
            _ = self.model(x)

        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        allocated_memory = torch.cuda.memory_allocated() / 1024 / 1024

        return MemoryResult(
            batch_size=batch_size,
            model_memory_mb=model_memory,
            peak_memory_mb=peak_memory,
            allocated_memory_mb=allocated_memory,
        )

    def run_all(self) -> BenchmarkResults:
        """Run all benchmarks for all batch sizes."""
        latency_results = {}
        throughput_results = {}
        memory_results = {}

        for batch_size in self.config.batch_sizes:
            print(f"  Benchmarking batch_size={batch_size}...")

            latency_results[batch_size] = self.run_latency(batch_size)
            throughput_results[batch_size] = self.run_throughput(batch_size)
            memory_results[batch_size] = self.run_memory(batch_size)

            # Print summary
            lat = latency_results[batch_size]
            thr = throughput_results[batch_size]
            mem = memory_results[batch_size]

            print(f"    Latency: {lat.mean_ms:.2f} ± {lat.std_ms:.2f} ms")
            print(f"    Throughput: {thr.samples_per_sec:.0f} samples/sec")
            print(f"    Peak Memory: {mem.peak_memory_mb:.1f} MB")

        return BenchmarkResults(
            latency=latency_results,
            throughput=throughput_results,
            memory=memory_results,
            model_params=self.n_params,
            window_size=self.config.window_size,
            device=self.device,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def run_latency_benchmark(
    model: nn.Module,
    batch_sizes: List[int],
    window_size: int = 5000,
    in_channels: int = 32,
    n_iterations: int = 100,
    device: str = "cuda",
) -> Dict[int, LatencyResult]:
    """Run latency benchmark for multiple batch sizes."""
    config = BenchmarkConfig(
        batch_sizes=batch_sizes,
        window_size=window_size,
        in_channels=in_channels,
        n_iterations=n_iterations,
        device=device,
    )

    benchmark = LatencyBenchmark(model, config)

    results = {}
    for batch_size in batch_sizes:
        results[batch_size] = benchmark.run_latency(batch_size)

    return results


def run_throughput_benchmark(
    model: nn.Module,
    batch_sizes: List[int],
    window_size: int = 5000,
    in_channels: int = 32,
    n_iterations: int = 100,
    device: str = "cuda",
) -> Dict[int, ThroughputResult]:
    """Run throughput benchmark for multiple batch sizes."""
    config = BenchmarkConfig(
        batch_sizes=batch_sizes,
        window_size=window_size,
        in_channels=in_channels,
        n_iterations=n_iterations,
        device=device,
    )

    benchmark = LatencyBenchmark(model, config)

    results = {}
    for batch_size in batch_sizes:
        results[batch_size] = benchmark.run_throughput(batch_size)

    return results


def run_memory_benchmark(
    model: nn.Module,
    batch_sizes: List[int],
    window_size: int = 5000,
    in_channels: int = 32,
    device: str = "cuda",
) -> Dict[int, MemoryResult]:
    """Run memory benchmark for multiple batch sizes."""
    config = BenchmarkConfig(
        batch_sizes=batch_sizes,
        window_size=window_size,
        in_channels=in_channels,
        device=device,
    )

    benchmark = LatencyBenchmark(model, config)

    results = {}
    for batch_size in batch_sizes:
        results[batch_size] = benchmark.run_memory(batch_size)

    return results


def check_realtime_feasibility(results: BenchmarkResults) -> Dict[str, bool]:
    """Check if model meets real-time requirements.

    Returns:
        Dictionary of requirement checks
    """
    checks = {
        "latency_ok": results.latency.get(1, LatencyResult(1, 0, 0, 0, 0, 0, 0, 0, 0)).meets_realtime,
        "throughput_ok": any(t.meets_realtime for t in results.throughput.values()),
        "memory_ok": results.memory.get(1, MemoryResult(1, 0, 0, 0)).meets_realtime,
        "all_ok": results.meets_all_requirements,
    }

    return checks
