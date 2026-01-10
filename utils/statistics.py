"""
Statistical Utilities for Nature Methods Publication
=====================================================

Comprehensive statistical testing and metrics for rigorous model comparison.
Includes:
- Parametric tests (t-test, ANOVA)
- Non-parametric tests (Wilcoxon, Mann-Whitney, Kruskal-Wallis)
- Effect sizes (Cohen's d, Hedges' g, eta-squared)
- Confidence intervals
- Multiple comparison corrections (Bonferroni, Holm, FDR)
- Additional regression metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
from scipy import stats
from scipy.stats import (
    ttest_rel,
    ttest_ind,
    wilcoxon,
    mannwhitneyu,
    kruskal,
    friedmanchisquare,
    shapiro,
    levene,
    pearsonr,
    spearmanr,
)


# =============================================================================
# Statistical Result Classes
# =============================================================================

@dataclass
class TestResult:
    """Result from a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    significant: bool = False
    alpha: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "effect_size": float(self.effect_size) if self.effect_size else None,
            "effect_size_name": self.effect_size_name,
            "ci_lower": float(self.ci_lower) if self.ci_lower else None,
            "ci_upper": float(self.ci_upper) if self.ci_upper else None,
            "significant": self.significant,
            "alpha": self.alpha,
        }

    def significance_marker(self) -> str:
        """Return significance marker for figures."""
        if self.p_value < 0.001:
            return "***"
        elif self.p_value < 0.01:
            return "**"
        elif self.p_value < 0.05:
            return "*"
        else:
            return "ns"

    def __str__(self) -> str:
        sig = self.significance_marker()
        effect_str = f", {self.effect_size_name}={self.effect_size:.3f}" if self.effect_size else ""
        return f"{self.test_name}: p={self.p_value:.4f} ({sig}){effect_str}"


@dataclass
class ComparisonResult:
    """Result from comparing two methods/conditions."""

    method_a: str
    method_b: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    mean_diff: float
    ci_diff: Tuple[float, float]
    parametric_test: TestResult
    nonparametric_test: TestResult
    normality_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "mean_a": float(self.mean_a),
            "mean_b": float(self.mean_b),
            "std_a": float(self.std_a),
            "std_b": float(self.std_b),
            "mean_diff": float(self.mean_diff),
            "ci_diff": [float(self.ci_diff[0]), float(self.ci_diff[1])],
            "parametric_test": self.parametric_test.to_dict(),
            "nonparametric_test": self.nonparametric_test.to_dict(),
            "normality_ok": self.normality_ok,
        }

    @property
    def recommended_test(self) -> TestResult:
        """Return the recommended test based on normality."""
        return self.parametric_test if self.normality_ok else self.nonparametric_test


@dataclass
class MetricsResult:
    """Comprehensive metrics for regression evaluation."""

    r2: float
    mae: float
    rmse: float
    mse: float
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    explained_variance: float
    n_samples: int

    # Per-fold or per-channel statistics
    r2_per_fold: Optional[List[float]] = None
    r2_ci: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r2": float(self.r2),
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "mse": float(self.mse),
            "pearson_r": float(self.pearson_r),
            "pearson_p": float(self.pearson_p),
            "spearman_r": float(self.spearman_r),
            "spearman_p": float(self.spearman_p),
            "explained_variance": float(self.explained_variance),
            "n_samples": self.n_samples,
            "r2_per_fold": [float(x) for x in self.r2_per_fold] if self.r2_per_fold else None,
            "r2_ci": [float(self.r2_ci[0]), float(self.r2_ci[1])] if self.r2_ci else None,
        }


# =============================================================================
# Effect Size Calculations
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size for two groups.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Hedges' g (bias-corrected Cohen's d).

    Better for small sample sizes.
    """
    n1, n2 = len(group1), len(group2)
    d = cohens_d(group1, group2)

    # Correction factor for small samples
    correction = 1 - (3 / (4 * (n1 + n2) - 9))

    return d * correction


def cohens_d_paired(diff: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    if np.std(diff, ddof=1) == 0:
        return 0.0
    return np.mean(diff) / np.std(diff, ddof=1)


def effect_size_interpretation(d: float) -> str:
    """Interpret effect size magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


# =============================================================================
# Confidence Intervals
# =============================================================================

def confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Calculate confidence interval for the mean."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)

    # t-critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)

    margin = t_crit * se
    return (mean - margin, mean + margin)


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval."""
    np.random.seed(seed)

    n = len(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return (lower, upper)


def mean_diff_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    confidence: float = 0.95,
    paired: bool = True,
) -> Tuple[float, Tuple[float, float]]:
    """Calculate mean difference and its confidence interval."""
    if paired:
        diff = group1 - group2
        mean_diff = np.mean(diff)
        ci = confidence_interval(diff, confidence)
    else:
        mean_diff = np.mean(group1) - np.mean(group2)
        # Welch's t-test CI
        se = np.sqrt(np.var(group1, ddof=1)/len(group1) + np.var(group2, ddof=1)/len(group2))
        df = len(group1) + len(group2) - 2
        t_crit = stats.t.ppf((1 + confidence) / 2, df)
        margin = t_crit * se
        ci = (mean_diff - margin, mean_diff + margin)

    return mean_diff, ci


# =============================================================================
# Normality Tests
# =============================================================================

def test_normality(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
    """Test if data is normally distributed using Shapiro-Wilk test.

    Returns:
        (is_normal, p_value)
    """
    if len(data) < 3:
        return True, 1.0  # Can't test with < 3 samples

    if len(data) > 5000:
        # Shapiro-Wilk limited to 5000 samples
        data = np.random.choice(data, 5000, replace=False)

    try:
        stat, p = shapiro(data)
        return p > alpha, p
    except Exception:
        return True, 1.0


def test_homogeneity(group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
    """Test homogeneity of variances using Levene's test.

    Returns:
        (variances_equal, p_value)
    """
    try:
        stat, p = levene(group1, group2)
        return p > alpha, p
    except Exception:
        return True, 1.0


# =============================================================================
# Statistical Tests
# =============================================================================

def paired_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Paired t-test with effect size."""
    diff = group1 - group2
    stat, p = ttest_rel(group1, group2, alternative=alternative)

    # Effect size (Cohen's d for paired samples)
    d = cohens_d_paired(diff)

    # CI for mean difference
    ci = confidence_interval(diff, 1 - alpha)

    return TestResult(
        test_name="Paired t-test",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(d),
        effect_size_name="Cohen's d",
        ci_lower=ci[0],
        ci_upper=ci[1],
        significant=p < alpha,
        alpha=alpha,
    )


def wilcoxon_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Wilcoxon signed-rank test (non-parametric paired test)."""
    diff = group1 - group2

    # Remove zeros for Wilcoxon
    diff_nonzero = diff[diff != 0]

    if len(diff_nonzero) < 2:
        return TestResult(
            test_name="Wilcoxon signed-rank",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
        )

    try:
        stat, p = wilcoxon(diff_nonzero, alternative=alternative)
    except Exception:
        return TestResult(
            test_name="Wilcoxon signed-rank",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
        )

    # Rank-biserial correlation as effect size
    n = len(diff_nonzero)
    r = 1 - (2 * stat) / (n * (n + 1))

    return TestResult(
        test_name="Wilcoxon signed-rank",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(r),
        effect_size_name="rank-biserial r",
        significant=p < alpha,
        alpha=alpha,
    )


def independent_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Independent samples t-test with Welch's correction."""
    stat, p = ttest_ind(group1, group2, equal_var=False, alternative=alternative)
    d = cohens_d(group1, group2)

    return TestResult(
        test_name="Welch's t-test",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(d),
        effect_size_name="Cohen's d",
        significant=p < alpha,
        alpha=alpha,
    )


def mann_whitney_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Mann-Whitney U test (non-parametric independent test)."""
    stat, p = mannwhitneyu(group1, group2, alternative=alternative)

    # Effect size: rank-biserial correlation
    n1, n2 = len(group1), len(group2)
    r = 1 - (2 * stat) / (n1 * n2)

    return TestResult(
        test_name="Mann-Whitney U",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(r),
        effect_size_name="rank-biserial r",
        significant=p < alpha,
        alpha=alpha,
    )


# =============================================================================
# Multiple Comparison Corrections
# =============================================================================

def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], float]:
    """Bonferroni correction for multiple comparisons.

    Returns:
        (list of significant flags, corrected alpha)
    """
    n = len(p_values)
    corrected_alpha = alpha / n
    significant = [p < corrected_alpha for p in p_values]
    return significant, corrected_alpha


def holm_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni step-down correction.

    More powerful than Bonferroni while controlling FWER.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    significant = [False] * n

    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        threshold = alpha / (n - i)
        if p < threshold:
            significant[idx] = True
        else:
            break  # All remaining are non-significant

    return significant


def fdr_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """Benjamini-Hochberg FDR correction.

    Controls False Discovery Rate instead of FWER.

    Returns:
        (list of significant flags, adjusted p-values)
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Calculate adjusted p-values
    adjusted_p = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted_p[i] = sorted_p[i]
        else:
            adjusted_p[i] = min(adjusted_p[i + 1], sorted_p[i] * n / (i + 1))

    # Reorder to original
    result_p = np.zeros(n)
    result_p[sorted_indices] = adjusted_p

    significant = [p < alpha for p in result_p]

    return significant, result_p.tolist()


# =============================================================================
# Comprehensive Comparison Functions
# =============================================================================

def compare_methods(
    values_a: np.ndarray,
    values_b: np.ndarray,
    name_a: str = "Method A",
    name_b: str = "Method B",
    paired: bool = True,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Comprehensive comparison of two methods.

    Automatically selects appropriate tests based on data characteristics.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    # Basic statistics
    mean_a, mean_b = np.mean(values_a), np.mean(values_b)
    std_a, std_b = np.std(values_a, ddof=1), np.std(values_b, ddof=1)

    # Mean difference and CI
    mean_diff, ci_diff = mean_diff_ci(values_a, values_b, 1 - alpha, paired)

    # Check normality
    if paired:
        diff = values_a - values_b
        norm_ok, _ = test_normality(diff, alpha)
    else:
        norm_ok_a, _ = test_normality(values_a, alpha)
        norm_ok_b, _ = test_normality(values_b, alpha)
        norm_ok = norm_ok_a and norm_ok_b

    # Run tests
    if paired:
        parametric = paired_ttest(values_a, values_b, alpha)
        nonparametric = wilcoxon_test(values_a, values_b, alpha)
    else:
        parametric = independent_ttest(values_a, values_b, alpha)
        nonparametric = mann_whitney_test(values_a, values_b, alpha)

    return ComparisonResult(
        method_a=name_a,
        method_b=name_b,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        mean_diff=mean_diff,
        ci_diff=ci_diff,
        parametric_test=parametric,
        nonparametric_test=nonparametric,
        normality_ok=norm_ok,
    )


def compare_multiple_methods(
    results_dict: Dict[str, np.ndarray],
    baseline_name: str,
    paired: bool = True,
    alpha: float = 0.05,
    correction: str = "holm",  # "bonferroni", "holm", "fdr", or "none"
) -> Dict[str, ComparisonResult]:
    """Compare multiple methods against a baseline.

    Args:
        results_dict: Dict mapping method names to arrays of results
        baseline_name: Name of the baseline method
        paired: Whether comparisons are paired
        alpha: Significance level
        correction: Multiple comparison correction method

    Returns:
        Dict mapping method names to ComparisonResult
    """
    baseline_values = results_dict[baseline_name]
    comparisons = {}
    p_values = []
    method_names = []

    # Run all comparisons
    for name, values in results_dict.items():
        if name == baseline_name:
            continue

        comp = compare_methods(
            baseline_values, values, baseline_name, name, paired, alpha
        )
        comparisons[name] = comp
        p_values.append(comp.recommended_test.p_value)
        method_names.append(name)

    # Apply correction
    if correction == "bonferroni":
        significant, _ = bonferroni_correction(p_values, alpha)
    elif correction == "holm":
        significant = holm_correction(p_values, alpha)
    elif correction == "fdr":
        significant, _ = fdr_correction(p_values, alpha)
    else:
        significant = [p < alpha for p in p_values]

    # Update significance flags
    for name, sig in zip(method_names, significant):
        comparisons[name].parametric_test.significant = sig
        comparisons[name].nonparametric_test.significant = sig

    return comparisons


# =============================================================================
# Regression Metrics
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    per_fold_r2: Optional[List[float]] = None,
) -> MetricsResult:
    """Compute comprehensive regression metrics."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Basic metrics
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)

    # R² (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    # Explained variance
    var_res = np.var(y_true - y_pred)
    var_tot = np.var(y_true)
    explained_var = 1 - var_res / (var_tot + 1e-10)

    # Correlation metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)

    # CI for R² if per-fold values provided
    r2_ci = None
    if per_fold_r2 and len(per_fold_r2) > 2:
        r2_ci = confidence_interval(np.array(per_fold_r2))

    return MetricsResult(
        r2=float(r2),
        mae=float(mae),
        rmse=float(rmse),
        mse=float(mse),
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        spearman_r=float(spearman_r),
        spearman_p=float(spearman_p),
        explained_variance=float(explained_var),
        n_samples=len(y_true),
        r2_per_fold=per_fold_r2,
        r2_ci=r2_ci,
    )


def compute_fold_statistics(
    fold_values: List[float],
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Compute statistics across CV folds."""
    values = np.array(fold_values)

    ci = confidence_interval(values, confidence)

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "sem": float(stats.sem(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "n_folds": len(values),
    }


# =============================================================================
# Summary Statistics for Publication
# =============================================================================

def format_mean_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    decimals: int = 3,
) -> str:
    """Format mean with CI for publication: 'mean [CI_lower, CI_upper]'."""
    mean = np.mean(values)
    ci = confidence_interval(values, confidence)

    return f"{mean:.{decimals}f} [{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"


def format_mean_std(
    values: np.ndarray,
    decimals: int = 3,
) -> str:
    """Format mean ± std for publication."""
    mean = np.mean(values)
    std = np.std(values, ddof=1)

    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def create_summary_table(
    results: Dict[str, List[float]],
    baseline_name: str,
    metric_name: str = "R²",
) -> Dict[str, Dict[str, Any]]:
    """Create summary table for all methods.

    Returns dict with mean, std, CI, and comparison to baseline.
    """
    table = {}
    baseline_values = np.array(results[baseline_name])

    for name, values in results.items():
        values = np.array(values)
        ci = confidence_interval(values)

        entry = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "formatted": format_mean_ci(values),
        }

        if name != baseline_name:
            comp = compare_methods(baseline_values, values, baseline_name, name)
            entry["vs_baseline"] = {
                "diff": float(np.mean(values) - np.mean(baseline_values)),
                "p_value": comp.recommended_test.p_value,
                "effect_size": comp.recommended_test.effect_size,
                "significant": comp.recommended_test.significant,
                "marker": comp.recommended_test.significance_marker(),
            }

        table[name] = entry

    return table


# =============================================================================
# Utility Functions
# =============================================================================

def check_assumptions(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Check statistical assumptions for parametric tests."""
    diff = group1 - group2

    norm_diff, norm_p = test_normality(diff, alpha)
    norm_g1, norm_g1_p = test_normality(group1, alpha)
    norm_g2, norm_g2_p = test_normality(group2, alpha)
    homo, homo_p = test_homogeneity(group1, group2, alpha)

    return {
        "normality_diff": {"ok": norm_diff, "p": norm_p},
        "normality_group1": {"ok": norm_g1, "p": norm_g1_p},
        "normality_group2": {"ok": norm_g2, "p": norm_g2_p},
        "homogeneity": {"ok": homo, "p": homo_p},
        "parametric_ok": norm_diff and homo,
        "recommendation": "parametric" if norm_diff else "non-parametric",
    }
