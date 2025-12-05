"""
Confidence Interval Computation
===============================

Bootstrap-based confidence interval computation for Nature Methods reporting.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: Array of samples
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap iterations
        ci_level: Confidence level (default: 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.randint(0, n, size=n)
        bootstrap_sample = data[indices]
        bootstrap_stats[i] = statistic(bootstrap_sample)

    # Compute percentiles
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    point_estimate = statistic(data)

    return point_estimate, ci_lower, ci_upper


def paired_bootstrap_test(
    data_a: np.ndarray,
    data_b: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Paired bootstrap test for comparing two conditions.

    Tests H0: statistic(A) = statistic(B)

    Args:
        data_a: Samples from condition A
        data_b: Samples from condition B (must be same length)
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed

    Returns:
        (difference, p_value): Point difference and two-sided p-value
    """
    assert len(data_a) == len(data_b), "Paired test requires equal lengths"

    if seed is not None:
        np.random.seed(seed)

    n = len(data_a)
    observed_diff = statistic(data_a) - statistic(data_b)

    # Bootstrap under null (paired differences should be symmetric around 0)
    differences = data_a - data_b
    centered_diffs = differences - np.mean(differences)

    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        bootstrap_sample = centered_diffs[indices]
        bootstrap_diffs[i] = statistic(bootstrap_sample + np.mean(differences))

    # Two-sided p-value
    p_value = np.mean(np.abs(bootstrap_diffs - np.mean(bootstrap_diffs)) >= np.abs(observed_diff))

    return observed_diff, p_value


def compute_effect_size(
    data_treatment: np.ndarray,
    data_control: np.ndarray,
) -> Dict[str, float]:
    """Compute effect size metrics.

    Args:
        data_treatment: Treatment group samples
        data_control: Control group samples

    Returns:
        Dictionary with effect size metrics
    """
    mean_t = np.mean(data_treatment)
    mean_c = np.mean(data_control)
    std_t = np.std(data_treatment, ddof=1)
    std_c = np.std(data_control, ddof=1)

    # Pooled standard deviation
    n_t = len(data_treatment)
    n_c = len(data_control)
    pooled_std = np.sqrt(
        ((n_t - 1) * std_t**2 + (n_c - 1) * std_c**2) / (n_t + n_c - 2)
    )

    # Cohen's d
    cohens_d = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0

    # Percent improvement
    percent_improvement = (mean_t - mean_c) / abs(mean_c) * 100 if mean_c != 0 else 0

    return {
        "cohens_d": cohens_d,
        "percent_improvement": percent_improvement,
        "mean_difference": mean_t - mean_c,
        "pooled_std": pooled_std,
    }


def format_ci_string(
    point: float,
    ci_lower: float,
    ci_upper: float,
    precision: int = 4,
) -> str:
    """Format confidence interval for publication.

    Args:
        point: Point estimate
        ci_lower: Lower CI bound
        ci_upper: Upper CI bound
        precision: Decimal places

    Returns:
        Formatted string like "0.8234 [0.8012, 0.8456]"
    """
    return f"{point:.{precision}f} [{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]"


def bootstrap_ci_difference(
    data_a: np.ndarray,
    data_b: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compute bootstrap CI for the difference between two groups.

    Args:
        data_a: First group samples
        data_b: Second group samples
        n_bootstrap: Bootstrap iterations
        ci_level: Confidence level
        seed: Random seed

    Returns:
        (difference, ci_lower, ci_upper)
    """
    if seed is not None:
        np.random.seed(seed)

    n_a, n_b = len(data_a), len(data_b)
    bootstrap_diffs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample_a = data_a[np.random.randint(0, n_a, size=n_a)]
        sample_b = data_b[np.random.randint(0, n_b, size=n_b)]
        bootstrap_diffs[i] = np.mean(sample_a) - np.mean(sample_b)

    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
    point_diff = np.mean(data_a) - np.mean(data_b)

    return point_diff, ci_lower, ci_upper


def summarize_with_ci(
    values: List[float],
    name: str = "metric",
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
) -> Dict[str, Union[str, float]]:
    """Create summary dictionary with confidence intervals.

    Args:
        values: List of values
        name: Name of the metric
        n_bootstrap: Bootstrap iterations
        ci_level: Confidence level

    Returns:
        Summary dictionary
    """
    arr = np.array(values)
    mean, ci_lower, ci_upper = bootstrap_ci(arr, np.mean, n_bootstrap, ci_level)
    _, std_lower, std_upper = bootstrap_ci(arr, np.std, n_bootstrap, ci_level)

    return {
        f"{name}_mean": mean,
        f"{name}_std": np.std(arr),
        f"{name}_ci_lower": ci_lower,
        f"{name}_ci_upper": ci_upper,
        f"{name}_ci_string": format_ci_string(mean, ci_lower, ci_upper),
        f"{name}_n": len(values),
    }
