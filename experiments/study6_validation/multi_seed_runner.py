"""
Multi-Seed Runner for Final Validation
======================================

Runs top configurations with multiple seeds for statistical robustness.
"""

from __future__ import annotations

import gc
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SeedResult:
    """Result from a single seed run."""
    seed: int
    train_r2: float
    val_r2: float
    test_r2: float
    train_loss: float
    val_loss: float
    n_epochs: int
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiSeedResult:
    """Aggregated results across multiple seeds."""
    config_name: str
    config: Dict[str, Any]
    seed_results: List[SeedResult]

    @property
    def mean_r2(self) -> float:
        return np.mean([r.test_r2 for r in self.seed_results])

    @property
    def std_r2(self) -> float:
        return np.std([r.test_r2 for r in self.seed_results])

    @property
    def ci_95(self) -> tuple:
        """95% confidence interval."""
        values = [r.test_r2 for r in self.seed_results]
        mean = np.mean(values)
        se = np.std(values) / np.sqrt(len(values))
        return (mean - 1.96 * se, mean + 1.96 * se)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "config": self.config,
            "mean_r2": self.mean_r2,
            "std_r2": self.std_r2,
            "ci_95_lower": self.ci_95[0],
            "ci_95_upper": self.ci_95[1],
            "n_seeds": len(self.seed_results),
            "all_r2_values": [r.test_r2 for r in self.seed_results],
            "seed_results": [
                {
                    "seed": r.seed,
                    "train_r2": r.train_r2,
                    "val_r2": r.val_r2,
                    "test_r2": r.test_r2,
                }
                for r in self.seed_results
            ],
        }


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_seeds(base_seed: int, n_seeds: int) -> List[int]:
    """Generate deterministic list of seeds."""
    seeds = []
    for i in range(n_seeds):
        seed_str = f"{base_seed}_run_{i}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**31)
        seeds.append(seed)
    return seeds


class MultiSeedRunner:
    """Run training with multiple seeds for robustness validation.

    Args:
        model_factory: Function that creates model given config
        train_fn: Function that trains model and returns metrics
        n_seeds: Number of seeds to use
        base_seed: Base seed for generating deterministic seeds
    """

    def __init__(
        self,
        model_factory,
        train_fn,
        n_seeds: int = 10,
        base_seed: int = 42,
    ):
        self.model_factory = model_factory
        self.train_fn = train_fn
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.seeds = generate_seeds(base_seed, n_seeds)

    def run_single_seed(
        self,
        config: Dict[str, Any],
        seed: int,
        train_loader,
        val_loader,
        test_loader,
        device: torch.device,
        n_epochs: int = 80,
    ) -> SeedResult:
        """Run training with a single seed."""

        seed_everything(seed)

        # Create model
        model = self.model_factory(config).to(device)

        # Train
        result = self.train_fn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            config=config,
            n_epochs=n_epochs,
        )

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return SeedResult(
            seed=seed,
            train_r2=result.get("train_r2", 0.0),
            val_r2=result.get("val_r2", 0.0),
            test_r2=result.get("test_r2", 0.0),
            train_loss=result.get("train_loss", 0.0),
            val_loss=result.get("val_loss", 0.0),
            n_epochs=result.get("n_epochs", n_epochs),
            config=config,
        )

    def run_config(
        self,
        config_name: str,
        config: Dict[str, Any],
        train_loader,
        val_loader,
        test_loader,
        device: torch.device,
        n_epochs: int = 80,
        progress_callback=None,
    ) -> MultiSeedResult:
        """Run all seeds for a single configuration."""

        seed_results = []

        for i, seed in enumerate(self.seeds):
            if progress_callback:
                progress_callback(f"Config {config_name}, seed {i+1}/{len(self.seeds)}")

            result = self.run_single_seed(
                config=config,
                seed=seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                n_epochs=n_epochs,
            )
            seed_results.append(result)

        return MultiSeedResult(
            config_name=config_name,
            config=config,
            seed_results=seed_results,
        )

    def run_all_configs(
        self,
        configs: Dict[str, Dict[str, Any]],
        train_loader,
        val_loader,
        test_loader,
        device: torch.device,
        n_epochs: int = 80,
    ) -> List[MultiSeedResult]:
        """Run all configurations with all seeds."""

        results = []

        for config_name, config in configs.items():
            print(f"\n{'='*50}")
            print(f"Running config: {config_name}")
            print(f"{'='*50}")

            result = self.run_config(
                config_name=config_name,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                n_epochs=n_epochs,
            )
            results.append(result)

            print(f"  Mean R²: {result.mean_r2:.4f} ± {result.std_r2:.4f}")
            print(f"  95% CI: [{result.ci_95[0]:.4f}, {result.ci_95[1]:.4f}]")

        return results


def run_bootstrap_analysis(
    results: List[MultiSeedResult],
    n_bootstrap: int = 10000,
) -> Dict[str, Dict[str, Any]]:
    """Run bootstrap analysis for confidence intervals.

    Args:
        results: Multi-seed results for each config
        n_bootstrap: Number of bootstrap samples

    Returns:
        Bootstrap analysis results
    """
    bootstrap_results = {}

    for result in results:
        values = np.array([r.test_r2 for r in result.seed_results])
        n = len(values)

        # Bootstrap
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)

        bootstrap_results[result.config_name] = {
            "original_mean": float(np.mean(values)),
            "bootstrap_mean": float(np.mean(bootstrap_means)),
            "bootstrap_std": float(np.std(bootstrap_means)),
            "ci_95_lower": float(np.percentile(bootstrap_means, 2.5)),
            "ci_95_upper": float(np.percentile(bootstrap_means, 97.5)),
            "ci_99_lower": float(np.percentile(bootstrap_means, 0.5)),
            "ci_99_upper": float(np.percentile(bootstrap_means, 99.5)),
        }

    return bootstrap_results
