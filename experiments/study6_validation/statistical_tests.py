"""
Statistical Tests for Final Validation
======================================

Implements statistical significance tests for Nature Methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    method_1: str = ""
    method_2: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "effect_size": self.effect_size,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "method_1": self.method_1,
            "method_2": self.method_2,
        }


def paired_t_test(
    values_1: np.ndarray,
    values_2: np.ndarray,
    alpha: float = 0.05,
    method_1: str = "",
    method_2: str = "",
) -> TestResult:
    """Paired t-test for comparing two methods on same seeds.

    Args:
        values_1: R² values from method 1 [n_seeds]
        values_2: R² values from method 2 [n_seeds]
        alpha: Significance level
        method_1: Name of method 1
        method_2: Name of method 2

    Returns:
        TestResult with statistics
    """
    assert len(values_1) == len(values_2), "Must have same number of samples"

    statistic, p_value = stats.ttest_rel(values_1, values_2)

    # Effect size (Cohen's d)
    diff = values_1 - values_2
    effect_size = np.mean(diff) / np.std(diff)

    # 95% CI for difference
    se = stats.sem(diff)
    ci_lower = np.mean(diff) - 1.96 * se
    ci_upper = np.mean(diff) + 1.96 * se

    return TestResult(
        test_name="paired_t_test",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        effect_size=float(effect_size),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method_1=method_1,
        method_2=method_2,
    )


def wilcoxon_test(
    values_1: np.ndarray,
    values_2: np.ndarray,
    alpha: float = 0.05,
    method_1: str = "",
    method_2: str = "",
) -> TestResult:
    """Wilcoxon signed-rank test (non-parametric).

    More robust than t-test when normality assumption is violated.

    Args:
        values_1: R² values from method 1 [n_seeds]
        values_2: R² values from method 2 [n_seeds]
        alpha: Significance level

    Returns:
        TestResult
    """
    statistic, p_value = stats.wilcoxon(values_1, values_2)

    # Effect size (r = Z / sqrt(N))
    n = len(values_1)
    z = stats.norm.ppf(p_value / 2)
    effect_size = abs(z) / np.sqrt(n)

    return TestResult(
        test_name="wilcoxon_signed_rank",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        effect_size=float(effect_size),
        method_1=method_1,
        method_2=method_2,
    )


def permutation_test(
    values_1: np.ndarray,
    values_2: np.ndarray,
    n_permutations: int = 10000,
    alpha: float = 0.05,
    method_1: str = "",
    method_2: str = "",
) -> TestResult:
    """Permutation test for significance.

    Non-parametric test with minimal assumptions.

    Args:
        values_1, values_2: R² values from two methods
        n_permutations: Number of permutations
        alpha: Significance level

    Returns:
        TestResult
    """
    observed_diff = np.mean(values_1) - np.mean(values_2)

    # Combine values
    combined = np.concatenate([values_1, values_2])
    n = len(values_1)

    # Permutation
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n]) - np.mean(combined[n:])
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)

    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return TestResult(
        test_name="permutation_test",
        statistic=float(observed_diff),
        p_value=float(p_value),
        significant=p_value < alpha,
        method_1=method_1,
        method_2=method_2,
    )


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[bool], float]:
    """Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values
        alpha: Desired family-wise error rate

    Returns:
        List of significance decisions, corrected alpha
    """
    corrected_alpha = alpha / len(p_values)
    significant = [p < corrected_alpha for p in p_values]
    return significant, corrected_alpha


def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[bool], List[float]]:
    """Benjamini-Hochberg FDR correction.

    Controls False Discovery Rate rather than family-wise error.

    Args:
        p_values: List of p-values
        alpha: Desired FDR level

    Returns:
        List of significance decisions, adjusted p-values
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Calculate BH threshold for each rank
    thresholds = [(i + 1) / n * alpha for i in range(n)]

    # Find largest k where p_k <= threshold_k
    significant_sorted = sorted_p <= thresholds
    if not np.any(significant_sorted):
        return [False] * n, list(p_values)

    k = np.max(np.where(significant_sorted)[0])

    # Reject all hypotheses 1..k
    significant = [False] * n
    for i in range(k + 1):
        significant[sorted_indices[i]] = True

    # Adjusted p-values
    adjusted = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        adjusted[idx] = min(1.0, sorted_p[i] * n / (i + 1))

    return significant, list(adjusted)


class StatisticalComparison:
    """Compare multiple methods with proper statistical tests.

    Args:
        alpha: Significance level
        correction: Multiple comparison correction ("bonferroni", "bh", or None)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        correction: str = "bh",
    ):
        self.alpha = alpha
        self.correction = correction
        self.results: List[TestResult] = []

    def compare_pair(
        self,
        name_1: str,
        values_1: np.ndarray,
        name_2: str,
        values_2: np.ndarray,
    ) -> Dict[str, TestResult]:
        """Compare a pair of methods with multiple tests."""

        tests = {}

        # Paired t-test
        tests["paired_t"] = paired_t_test(
            values_1, values_2, self.alpha, name_1, name_2
        )

        # Wilcoxon (non-parametric)
        tests["wilcoxon"] = wilcoxon_test(
            values_1, values_2, self.alpha, name_1, name_2
        )

        # Permutation test
        tests["permutation"] = permutation_test(
            values_1, values_2, alpha=self.alpha, method_1=name_1, method_2=name_2
        )

        self.results.extend(tests.values())
        return tests

    def compare_all_pairs(
        self,
        methods: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, TestResult]]:
        """Compare all pairs of methods.

        Args:
            methods: Dict mapping method name to R² values

        Returns:
            Nested dict of comparison results
        """
        names = list(methods.keys())
        all_comparisons = {}

        for i, name_1 in enumerate(names):
            for name_2 in names[i + 1:]:
                key = f"{name_1}_vs_{name_2}"
                all_comparisons[key] = self.compare_pair(
                    name_1, methods[name_1],
                    name_2, methods[name_2],
                )

        return all_comparisons

    def apply_correction(self) -> Dict[str, Any]:
        """Apply multiple comparison correction to all results."""

        p_values = [r.p_value for r in self.results]

        if self.correction == "bonferroni":
            significant, corrected_alpha = bonferroni_correction(
                p_values, self.alpha
            )
            return {
                "method": "bonferroni",
                "corrected_alpha": corrected_alpha,
                "significant": significant,
            }

        elif self.correction == "bh":
            significant, adjusted_p = benjamini_hochberg(p_values, self.alpha)
            return {
                "method": "benjamini_hochberg",
                "adjusted_p_values": adjusted_p,
                "significant": significant,
            }

        else:
            return {
                "method": "none",
                "significant": [r.significant for r in self.results],
            }

    def summary(self) -> Dict[str, Any]:
        """Generate summary of all comparisons."""

        correction_result = self.apply_correction()

        return {
            "n_comparisons": len(self.results),
            "alpha": self.alpha,
            "correction": correction_result,
            "comparisons": [r.to_dict() for r in self.results],
        }
