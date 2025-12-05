"""
Negative Controls for Validation
================================

Implements negative control experiments to verify model behavior:
- Shuffled labels: Model should fail (chance performance)
- Time-reversed signals: Tests temporal dependency
- Random input: Tests information content
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class NegativeControlResult:
    """Result from a negative control experiment."""
    control_type: str
    baseline_r2: float  # Original model R²
    control_r2: float   # Negative control R²
    expected_r2: float  # Expected R² (e.g., 0 for shuffled)
    passed: bool        # Whether control behaved as expected
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_type": self.control_type,
            "baseline_r2": self.baseline_r2,
            "control_r2": self.control_r2,
            "expected_r2": self.expected_r2,
            "passed": self.passed,
            "details": self.details or {},
        }


class NegativeControlRunner:
    """Run negative control experiments for validation.

    Args:
        model_factory: Function to create model
        train_fn: Training function
        baseline_r2_threshold: Minimum baseline R² to proceed
        control_r2_threshold: Maximum R² for negative controls to pass
    """

    def __init__(
        self,
        model_factory,
        train_fn,
        baseline_r2_threshold: float = 0.3,
        control_r2_threshold: float = 0.1,
    ):
        self.model_factory = model_factory
        self.train_fn = train_fn
        self.baseline_r2_threshold = baseline_r2_threshold
        self.control_r2_threshold = control_r2_threshold

    def run_shuffled_labels(
        self,
        config: Dict[str, Any],
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        device: torch.device,
        n_epochs: int = 40,
    ) -> NegativeControlResult:
        """Test with shuffled labels (should give ~0 R²).

        Shuffles the correspondence between input and output signals.
        Model should not be able to learn meaningful mapping.

        Args:
            config: Model configuration
            train_data, train_labels: Training data
            test_data, test_labels: Test data
            device: Device
            n_epochs: Training epochs

        Returns:
            NegativeControlResult
        """
        # First get baseline
        baseline_result = self._train_and_evaluate(
            config, train_data, train_labels, test_data, test_labels,
            device, n_epochs
        )
        baseline_r2 = baseline_result["test_r2"]

        if baseline_r2 < self.baseline_r2_threshold:
            return NegativeControlResult(
                control_type="shuffled_labels",
                baseline_r2=baseline_r2,
                control_r2=float("nan"),
                expected_r2=0.0,
                passed=False,
                details={"error": f"Baseline R² ({baseline_r2:.4f}) below threshold"}
            )

        # Shuffle labels
        perm = torch.randperm(train_labels.shape[0])
        shuffled_labels = train_labels[perm]

        # Train with shuffled
        control_result = self._train_and_evaluate(
            config, train_data, shuffled_labels, test_data, test_labels,
            device, n_epochs
        )
        control_r2 = control_result["test_r2"]

        # Check if control R² is appropriately low
        passed = control_r2 < self.control_r2_threshold

        return NegativeControlResult(
            control_type="shuffled_labels",
            baseline_r2=baseline_r2,
            control_r2=control_r2,
            expected_r2=0.0,
            passed=passed,
            details={
                "baseline_train_r2": baseline_result.get("train_r2", 0),
                "control_train_r2": control_result.get("train_r2", 0),
            }
        )

    def run_time_reversed(
        self,
        config: Dict[str, Any],
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        device: torch.device,
        n_epochs: int = 40,
    ) -> NegativeControlResult:
        """Test with time-reversed signals.

        Reverses the temporal dimension. For causal models,
        this should significantly degrade performance.

        Args:
            config: Model configuration
            train_data, train_labels: Training data [N, C, T]
            test_data, test_labels: Test data
            device: Device
            n_epochs: Training epochs

        Returns:
            NegativeControlResult
        """
        # Baseline
        baseline_result = self._train_and_evaluate(
            config, train_data, train_labels, test_data, test_labels,
            device, n_epochs
        )
        baseline_r2 = baseline_result["test_r2"]

        if baseline_r2 < self.baseline_r2_threshold:
            return NegativeControlResult(
                control_type="time_reversed",
                baseline_r2=baseline_r2,
                control_r2=float("nan"),
                expected_r2=baseline_r2 * 0.5,
                passed=False,
                details={"error": "Baseline R² below threshold"}
            )

        # Reverse time dimension
        train_data_rev = train_data.flip(dims=[-1])
        train_labels_rev = train_labels.flip(dims=[-1])
        test_data_rev = test_data.flip(dims=[-1])

        # Train on reversed, test on original
        control_result = self._train_and_evaluate(
            config, train_data_rev, train_labels_rev, test_data, test_labels,
            device, n_epochs
        )
        control_r2 = control_result["test_r2"]

        # For strongly temporal models, expect significant degradation
        expected_degradation = 0.5  # Expect at least 50% R² loss
        expected_r2 = baseline_r2 * (1 - expected_degradation)
        passed = control_r2 < expected_r2

        return NegativeControlResult(
            control_type="time_reversed",
            baseline_r2=baseline_r2,
            control_r2=control_r2,
            expected_r2=expected_r2,
            passed=passed,
            details={
                "r2_reduction": 1 - control_r2 / baseline_r2 if baseline_r2 > 0 else 0,
            }
        )

    def run_random_input(
        self,
        config: Dict[str, Any],
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        device: torch.device,
        n_epochs: int = 40,
    ) -> NegativeControlResult:
        """Test with random input signals.

        Replaces input with random noise. Model should not learn.

        Args:
            config: Model configuration
            train_data, train_labels: Training data
            test_data, test_labels: Test data
            device: Device
            n_epochs: Training epochs

        Returns:
            NegativeControlResult
        """
        # Baseline
        baseline_result = self._train_and_evaluate(
            config, train_data, train_labels, test_data, test_labels,
            device, n_epochs
        )
        baseline_r2 = baseline_result["test_r2"]

        # Random input with same statistics
        random_train = torch.randn_like(train_data)
        random_test = torch.randn_like(test_data)

        # Scale to match original statistics
        random_train = random_train * train_data.std() + train_data.mean()
        random_test = random_test * test_data.std() + test_data.mean()

        # Train with random input
        control_result = self._train_and_evaluate(
            config, random_train, train_labels, random_test, test_labels,
            device, n_epochs
        )
        control_r2 = control_result["test_r2"]

        passed = control_r2 < self.control_r2_threshold

        return NegativeControlResult(
            control_type="random_input",
            baseline_r2=baseline_r2,
            control_r2=control_r2,
            expected_r2=0.0,
            passed=passed,
        )

    def run_all_controls(
        self,
        config: Dict[str, Any],
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        device: torch.device,
        n_epochs: int = 40,
    ) -> List[NegativeControlResult]:
        """Run all negative control experiments.

        Args:
            config: Model configuration
            train_data, train_labels, test_data, test_labels: Data
            device: Device
            n_epochs: Training epochs

        Returns:
            List of NegativeControlResults
        """
        results = []

        # Shuffled labels
        print("  Running shuffled labels control...")
        results.append(self.run_shuffled_labels(
            config, train_data, train_labels, test_data, test_labels,
            device, n_epochs
        ))

        # Time reversed
        print("  Running time-reversed control...")
        results.append(self.run_time_reversed(
            config, train_data, train_labels, test_data, test_labels,
            device, n_epochs
        ))

        # Random input
        print("  Running random input control...")
        results.append(self.run_random_input(
            config, train_data, train_labels, test_data, test_labels,
            device, n_epochs
        ))

        return results

    def _train_and_evaluate(
        self,
        config: Dict[str, Any],
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        device: torch.device,
        n_epochs: int,
    ) -> Dict[str, float]:
        """Helper to train and evaluate model."""
        from torch.utils.data import TensorDataset, DataLoader

        # Create loaders
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16)

        # Create and train model
        model = self.model_factory(config).to(device)
        result = self.train_fn(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            test_loader=test_loader,
            device=device,
            config=config,
            n_epochs=n_epochs,
        )

        return result


def summarize_negative_controls(
    results: List[NegativeControlResult],
) -> Dict[str, Any]:
    """Summarize negative control results.

    Args:
        results: List of negative control results

    Returns:
        Summary dictionary
    """
    n_passed = sum(1 for r in results if r.passed)
    n_total = len(results)

    return {
        "total_controls": n_total,
        "passed": n_passed,
        "failed": n_total - n_passed,
        "pass_rate": n_passed / n_total if n_total > 0 else 0,
        "all_passed": n_passed == n_total,
        "results": [r.to_dict() for r in results],
    }
