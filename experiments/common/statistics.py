"""
Statistical Analysis Utilities for Nature Methods Publication.

Provides:
- Effect size calculations (Cohen's d, eta-squared)
- ANOVA for factorial designs
- Multiple comparison corrections (Bonferroni, FDR)
- Bootstrap confidence intervals
- Power analysis utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


# =============================================================================
# Effect Size Calculations
# =============================================================================


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size between two groups.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large

    Args:
        group1: First group of observations
        group2: Second group of observations

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cohens_d_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def eta_squared(ss_effect: float, ss_total: float) -> float:
    """Calculate eta-squared (proportion of variance explained).

    Args:
        ss_effect: Sum of squares for the effect
        ss_total: Total sum of squares

    Returns:
        Eta-squared value (0 to 1)
    """
    if ss_total == 0:
        return 0.0
    return ss_effect / ss_total


def partial_eta_squared(ss_effect: float, ss_error: float) -> float:
    """Calculate partial eta-squared.

    Args:
        ss_effect: Sum of squares for the effect
        ss_error: Sum of squares for error

    Returns:
        Partial eta-squared value (0 to 1)
    """
    if ss_effect + ss_error == 0:
        return 0.0
    return ss_effect / (ss_effect + ss_error)


# =============================================================================
# ANOVA for Factorial Designs
# =============================================================================


@dataclass
class ANOVAResult:
    """Result from ANOVA analysis."""

    factor: str
    ss: float  # Sum of squares
    df: int  # Degrees of freedom
    ms: float  # Mean square
    f_stat: float
    p_value: float
    eta_sq: float
    partial_eta_sq: float

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if effect is statistically significant."""
        return self.p_value < alpha

    def __repr__(self) -> str:
        sig = "*" if self.is_significant() else ""
        return (
            f"{self.factor}: F({self.df})={self.f_stat:.2f}, "
            f"p={self.p_value:.4f}{sig}, eta²={self.eta_sq:.3f}"
        )


def one_way_anova(groups: List[np.ndarray]) -> ANOVAResult:
    """Perform one-way ANOVA.

    Args:
        groups: List of arrays, one per group

    Returns:
        ANOVAResult with F-statistic, p-value, and effect sizes
    """
    # Overall mean
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    n_total = len(all_data)
    k = len(groups)  # Number of groups

    # Between-group sum of squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)

    # Within-group sum of squares
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

    # Total sum of squares
    ss_total = np.sum((all_data - grand_mean) ** 2)

    # Degrees of freedom
    df_between = k - 1
    df_within = n_total - k

    # Mean squares
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 1e-10

    # F-statistic
    f_stat = ms_between / ms_within

    # P-value
    p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)

    return ANOVAResult(
        factor="Group",
        ss=ss_between,
        df=df_between,
        ms=ms_between,
        f_stat=f_stat,
        p_value=p_value,
        eta_sq=eta_squared(ss_between, ss_total),
        partial_eta_sq=partial_eta_squared(ss_between, ss_within),
    )


@dataclass
class FactorialANOVAResult:
    """Results from factorial ANOVA."""

    main_effects: Dict[str, ANOVAResult]
    interactions: Dict[str, ANOVAResult]
    residual_ss: float
    residual_df: int
    total_ss: float

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["Factorial ANOVA Results:", "=" * 50, "", "Main Effects:"]
        for name, result in self.main_effects.items():
            lines.append(f"  {result}")
        lines.append("")
        lines.append("Interactions:")
        for name, result in self.interactions.items():
            lines.append(f"  {result}")
        return "\n".join(lines)


def two_way_anova(
    data: np.ndarray,
    factor_a: np.ndarray,
    factor_b: np.ndarray,
    factor_a_name: str = "A",
    factor_b_name: str = "B",
) -> FactorialANOVAResult:
    """Perform two-way ANOVA with interaction.

    Args:
        data: Dependent variable values
        factor_a: Factor A levels for each observation
        factor_b: Factor B levels for each observation
        factor_a_name: Name of factor A
        factor_b_name: Name of factor B

    Returns:
        FactorialANOVAResult with main effects and interaction
    """
    # Get unique levels
    levels_a = np.unique(factor_a)
    levels_b = np.unique(factor_b)

    n_total = len(data)
    grand_mean = np.mean(data)

    # Calculate cell means
    cell_means = {}
    cell_counts = {}
    for a in levels_a:
        for b in levels_b:
            mask = (factor_a == a) & (factor_b == b)
            if np.sum(mask) > 0:
                cell_means[(a, b)] = np.mean(data[mask])
                cell_counts[(a, b)] = np.sum(mask)

    # Marginal means
    mean_a = {a: np.mean(data[factor_a == a]) for a in levels_a}
    mean_b = {b: np.mean(data[factor_b == b]) for b in levels_b}

    # Sum of squares for factor A
    ss_a = sum(
        np.sum(factor_a == a) * (mean_a[a] - grand_mean) ** 2 for a in levels_a
    )

    # Sum of squares for factor B
    ss_b = sum(
        np.sum(factor_b == b) * (mean_b[b] - grand_mean) ** 2 for b in levels_b
    )

    # Sum of squares for interaction
    ss_ab = 0
    for a in levels_a:
        for b in levels_b:
            if (a, b) in cell_means:
                expected = mean_a[a] + mean_b[b] - grand_mean
                ss_ab += cell_counts[(a, b)] * (cell_means[(a, b)] - expected) ** 2

    # Total sum of squares
    ss_total = np.sum((data - grand_mean) ** 2)

    # Residual (error) sum of squares
    ss_error = ss_total - ss_a - ss_b - ss_ab

    # Degrees of freedom
    df_a = len(levels_a) - 1
    df_b = len(levels_b) - 1
    df_ab = df_a * df_b
    df_error = n_total - len(levels_a) * len(levels_b)

    # Mean squares
    ms_a = ss_a / df_a if df_a > 0 else 0
    ms_b = ss_b / df_b if df_b > 0 else 0
    ms_ab = ss_ab / df_ab if df_ab > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 1e-10

    # F-statistics
    f_a = ms_a / ms_error
    f_b = ms_b / ms_error
    f_ab = ms_ab / ms_error

    # P-values
    p_a = 1 - stats.f.cdf(f_a, df_a, df_error) if df_a > 0 else 1.0
    p_b = 1 - stats.f.cdf(f_b, df_b, df_error) if df_b > 0 else 1.0
    p_ab = 1 - stats.f.cdf(f_ab, df_ab, df_error) if df_ab > 0 else 1.0

    return FactorialANOVAResult(
        main_effects={
            factor_a_name: ANOVAResult(
                factor=factor_a_name,
                ss=ss_a,
                df=df_a,
                ms=ms_a,
                f_stat=f_a,
                p_value=p_a,
                eta_sq=eta_squared(ss_a, ss_total),
                partial_eta_sq=partial_eta_squared(ss_a, ss_error),
            ),
            factor_b_name: ANOVAResult(
                factor=factor_b_name,
                ss=ss_b,
                df=df_b,
                ms=ms_b,
                f_stat=f_b,
                p_value=p_b,
                eta_sq=eta_squared(ss_b, ss_total),
                partial_eta_sq=partial_eta_squared(ss_b, ss_error),
            ),
        },
        interactions={
            f"{factor_a_name}x{factor_b_name}": ANOVAResult(
                factor=f"{factor_a_name}x{factor_b_name}",
                ss=ss_ab,
                df=df_ab,
                ms=ms_ab,
                f_stat=f_ab,
                p_value=p_ab,
                eta_sq=eta_squared(ss_ab, ss_total),
                partial_eta_sq=partial_eta_squared(ss_ab, ss_error),
            )
        },
        residual_ss=ss_error,
        residual_df=df_error,
        total_ss=ss_total,
    )


def three_way_anova(
    data: np.ndarray,
    factor_a: np.ndarray,
    factor_b: np.ndarray,
    factor_c: np.ndarray,
    names: Tuple[str, str, str] = ("A", "B", "C"),
) -> FactorialANOVAResult:
    """Perform three-way ANOVA for factorial design.

    This is used for Architecture x Loss x Conditioning factorial analysis.

    Args:
        data: Dependent variable (e.g., R² values)
        factor_a: Factor A levels (e.g., architecture)
        factor_b: Factor B levels (e.g., loss)
        factor_c: Factor C levels (e.g., conditioning)
        names: Names of the three factors

    Returns:
        FactorialANOVAResult with main effects and all interactions
    """
    name_a, name_b, name_c = names

    levels_a = np.unique(factor_a)
    levels_b = np.unique(factor_b)
    levels_c = np.unique(factor_c)

    n_total = len(data)
    grand_mean = np.mean(data)

    # Marginal means
    mean_a = {a: np.mean(data[factor_a == a]) for a in levels_a}
    mean_b = {b: np.mean(data[factor_b == b]) for b in levels_b}
    mean_c = {c: np.mean(data[factor_c == c]) for c in levels_c}

    # Main effect SS
    ss_a = sum(np.sum(factor_a == a) * (mean_a[a] - grand_mean) ** 2 for a in levels_a)
    ss_b = sum(np.sum(factor_b == b) * (mean_b[b] - grand_mean) ** 2 for b in levels_b)
    ss_c = sum(np.sum(factor_c == c) * (mean_c[c] - grand_mean) ** 2 for c in levels_c)

    # Two-way marginal means
    mean_ab = {}
    mean_ac = {}
    mean_bc = {}

    for a in levels_a:
        for b in levels_b:
            mask = (factor_a == a) & (factor_b == b)
            if np.sum(mask) > 0:
                mean_ab[(a, b)] = np.mean(data[mask])

    for a in levels_a:
        for c in levels_c:
            mask = (factor_a == a) & (factor_c == c)
            if np.sum(mask) > 0:
                mean_ac[(a, c)] = np.mean(data[mask])

    for b in levels_b:
        for c in levels_c:
            mask = (factor_b == b) & (factor_c == c)
            if np.sum(mask) > 0:
                mean_bc[(b, c)] = np.mean(data[mask])

    # Two-way interaction SS
    ss_ab = 0
    for a in levels_a:
        for b in levels_b:
            if (a, b) in mean_ab:
                mask = (factor_a == a) & (factor_b == b)
                expected = mean_a[a] + mean_b[b] - grand_mean
                ss_ab += np.sum(mask) * (mean_ab[(a, b)] - expected) ** 2

    ss_ac = 0
    for a in levels_a:
        for c in levels_c:
            if (a, c) in mean_ac:
                mask = (factor_a == a) & (factor_c == c)
                expected = mean_a[a] + mean_c[c] - grand_mean
                ss_ac += np.sum(mask) * (mean_ac[(a, c)] - expected) ** 2

    ss_bc = 0
    for b in levels_b:
        for c in levels_c:
            if (b, c) in mean_bc:
                mask = (factor_b == b) & (factor_c == c)
                expected = mean_b[b] + mean_c[c] - grand_mean
                ss_bc += np.sum(mask) * (mean_bc[(b, c)] - expected) ** 2

    # Cell means for three-way interaction
    cell_means = {}
    for a in levels_a:
        for b in levels_b:
            for c in levels_c:
                mask = (factor_a == a) & (factor_b == b) & (factor_c == c)
                if np.sum(mask) > 0:
                    cell_means[(a, b, c)] = np.mean(data[mask])

    # Three-way interaction SS
    ss_abc = 0
    for a in levels_a:
        for b in levels_b:
            for c in levels_c:
                if (a, b, c) in cell_means:
                    mask = (factor_a == a) & (factor_b == b) & (factor_c == c)
                    # Expected from main effects and two-way interactions
                    expected = (
                        mean_a[a]
                        + mean_b[b]
                        + mean_c[c]
                        + (mean_ab.get((a, b), grand_mean) - mean_a[a] - mean_b[b] + grand_mean)
                        + (mean_ac.get((a, c), grand_mean) - mean_a[a] - mean_c[c] + grand_mean)
                        + (mean_bc.get((b, c), grand_mean) - mean_b[b] - mean_c[c] + grand_mean)
                        - 2 * grand_mean
                    )
                    ss_abc += np.sum(mask) * (cell_means[(a, b, c)] - expected) ** 2

    # Total and error SS
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_a - ss_b - ss_c - ss_ab - ss_ac - ss_bc - ss_abc

    # Degrees of freedom
    df_a = len(levels_a) - 1
    df_b = len(levels_b) - 1
    df_c = len(levels_c) - 1
    df_ab = df_a * df_b
    df_ac = df_a * df_c
    df_bc = df_b * df_c
    df_abc = df_a * df_b * df_c
    n_cells = len(levels_a) * len(levels_b) * len(levels_c)
    df_error = max(1, n_total - n_cells)

    # Mean squares and F-statistics
    ms_error = ss_error / df_error if df_error > 0 else 1e-10

    def compute_f_p(ss: float, df: int) -> Tuple[float, float, float]:
        ms = ss / df if df > 0 else 0
        f = ms / ms_error if ms_error > 0 else 0
        p = 1 - stats.f.cdf(f, df, df_error) if df > 0 and df_error > 0 else 1.0
        return ms, f, p

    ms_a, f_a, p_a = compute_f_p(ss_a, df_a)
    ms_b, f_b, p_b = compute_f_p(ss_b, df_b)
    ms_c, f_c, p_c = compute_f_p(ss_c, df_c)
    ms_ab, f_ab, p_ab = compute_f_p(ss_ab, df_ab)
    ms_ac, f_ac, p_ac = compute_f_p(ss_ac, df_ac)
    ms_bc, f_bc, p_bc = compute_f_p(ss_bc, df_bc)
    ms_abc, f_abc, p_abc = compute_f_p(ss_abc, df_abc)

    return FactorialANOVAResult(
        main_effects={
            name_a: ANOVAResult(name_a, ss_a, df_a, ms_a, f_a, p_a,
                               eta_squared(ss_a, ss_total),
                               partial_eta_squared(ss_a, ss_error)),
            name_b: ANOVAResult(name_b, ss_b, df_b, ms_b, f_b, p_b,
                               eta_squared(ss_b, ss_total),
                               partial_eta_squared(ss_b, ss_error)),
            name_c: ANOVAResult(name_c, ss_c, df_c, ms_c, f_c, p_c,
                               eta_squared(ss_c, ss_total),
                               partial_eta_squared(ss_c, ss_error)),
        },
        interactions={
            f"{name_a}x{name_b}": ANOVAResult(
                f"{name_a}x{name_b}", ss_ab, df_ab, ms_ab, f_ab, p_ab,
                eta_squared(ss_ab, ss_total),
                partial_eta_squared(ss_ab, ss_error)),
            f"{name_a}x{name_c}": ANOVAResult(
                f"{name_a}x{name_c}", ss_ac, df_ac, ms_ac, f_ac, p_ac,
                eta_squared(ss_ac, ss_total),
                partial_eta_squared(ss_ac, ss_error)),
            f"{name_b}x{name_c}": ANOVAResult(
                f"{name_b}x{name_c}", ss_bc, df_bc, ms_bc, f_bc, p_bc,
                eta_squared(ss_bc, ss_total),
                partial_eta_squared(ss_bc, ss_error)),
            f"{name_a}x{name_b}x{name_c}": ANOVAResult(
                f"{name_a}x{name_b}x{name_c}", ss_abc, df_abc, ms_abc, f_abc, p_abc,
                eta_squared(ss_abc, ss_total),
                partial_eta_squared(ss_abc, ss_error)),
        },
        residual_ss=ss_error,
        residual_df=df_error,
        total_ss=ss_total,
    )


# =============================================================================
# Multiple Comparison Corrections
# =============================================================================


def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: Array of p-values
        alpha: Significance level

    Returns:
        Dictionary with adjusted p-values and significance flags
    """
    n = len(p_values)
    adjusted_alpha = alpha / n
    adjusted_p = np.minimum(p_values * n, 1.0)
    significant = p_values < adjusted_alpha

    return {
        "raw_p": p_values,
        "adjusted_p": adjusted_p,
        "adjusted_alpha": adjusted_alpha,
        "significant": significant,
        "n_comparisons": n,
    }


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of p-values
        alpha: False discovery rate

    Returns:
        Dictionary with q-values and significance flags
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Calculate q-values (adjusted p-values)
    q_values = np.zeros(n)
    for i, p in enumerate(sorted_p):
        q_values[sorted_idx[i]] = p * n / (i + 1)

    # Ensure monotonicity
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]
    q_values = np.minimum(q_values, 1.0)

    # Determine significance
    significant = q_values < alpha

    return {
        "raw_p": p_values,
        "q_values": q_values,
        "alpha": alpha,
        "significant": significant,
        "n_comparisons": n,
    }


def pairwise_tests(
    groups: Dict[str, np.ndarray],
    correction: str = "bonferroni",
    alpha: float = 0.05,
) -> Dict[str, Dict]:
    """Perform pairwise t-tests between all groups with correction.

    Args:
        groups: Dictionary mapping group names to arrays
        correction: "bonferroni" or "fdr"
        alpha: Significance level

    Returns:
        Dictionary with pairwise comparisons and statistics
    """
    names = list(groups.keys())
    n_groups = len(names)
    n_comparisons = n_groups * (n_groups - 1) // 2

    comparisons = []

    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            name1, name2 = names[i], names[j]
            g1, g2 = groups[name1], groups[name2]

            # Welch's t-test
            t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)

            # Effect size
            d = cohens_d(g1, g2)

            comparisons.append({
                "comparison": f"{name1} vs {name2}",
                "group1": name1,
                "group2": name2,
                "mean1": np.mean(g1),
                "mean2": np.mean(g2),
                "diff": np.mean(g1) - np.mean(g2),
                "t_stat": t_stat,
                "p_value": p_value,
                "cohens_d": d,
                "effect_size": cohens_d_interpretation(d),
            })

    # Apply correction
    p_values = np.array([c["p_value"] for c in comparisons])

    if correction == "bonferroni":
        corrected = bonferroni_correction(p_values, alpha)
    else:
        corrected = fdr_correction(p_values, alpha)

    # Add corrected values to comparisons
    for i, comp in enumerate(comparisons):
        comp["adjusted_p"] = corrected["adjusted_p" if correction == "bonferroni" else "q_values"][i]
        comp["significant"] = corrected["significant"][i]

    return {
        "comparisons": comparisons,
        "correction": correction,
        "alpha": alpha,
        "n_comparisons": n_comparisons,
    }


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: Array of observations
        statistic: "mean", "median", or "std"
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with point estimate, CI bounds, and SE
    """
    rng = np.random.RandomState(seed)
    n = len(data)

    # Compute statistic function
    if statistic == "mean":
        stat_func = np.mean
    elif statistic == "median":
        stat_func = np.median
    elif statistic == "std":
        stat_func = lambda x: np.std(x, ddof=1)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Point estimate
    point_estimate = stat_func(data)

    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = stat_func(resample)

    # CI bounds
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    # Standard error
    se = np.std(bootstrap_stats, ddof=1)

    return {
        "estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci_level,
        "se": se,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_diff_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Bootstrap CI for the difference between two groups.

    Args:
        group1: First group observations
        group2: Second group observations
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        seed: Random seed

    Returns:
        Dictionary with difference estimate and CI
    """
    rng = np.random.RandomState(seed)
    n1, n2 = len(group1), len(group2)

    # Point estimate
    diff = np.mean(group1) - np.mean(group2)

    # Bootstrap
    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample1 = rng.choice(group1, size=n1, replace=True)
        resample2 = rng.choice(group2, size=n2, replace=True)
        bootstrap_diffs[i] = np.mean(resample1) - np.mean(resample2)

    # CI bounds
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    # Check if CI excludes zero (significant)
    significant = not (ci_lower <= 0 <= ci_upper)

    return {
        "diff": diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci_level,
        "significant": significant,
        "n_bootstrap": n_bootstrap,
    }


# =============================================================================
# Power Analysis
# =============================================================================


def required_sample_size(
    effect_size: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True,
) -> int:
    """Calculate required sample size per group for two-sample t-test.

    Args:
        effect_size: Cohen's d effect size to detect
        alpha: Significance level
        power: Desired statistical power
        two_tailed: Whether test is two-tailed

    Returns:
        Required sample size per group
    """
    from scipy.stats import norm

    if two_tailed:
        z_alpha = norm.ppf(1 - alpha / 2)
    else:
        z_alpha = norm.ppf(1 - alpha)

    z_beta = norm.ppf(power)

    # Sample size formula for two-sample t-test
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    return int(np.ceil(n))


def achieved_power(
    n_per_group: int,
    effect_size: float = 0.5,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> float:
    """Calculate achieved power given sample size.

    Args:
        n_per_group: Sample size per group
        effect_size: Cohen's d effect size
        alpha: Significance level
        two_tailed: Whether test is two-tailed

    Returns:
        Achieved statistical power (0 to 1)
    """
    from scipy.stats import norm

    if two_tailed:
        z_alpha = norm.ppf(1 - alpha / 2)
    else:
        z_alpha = norm.ppf(1 - alpha)

    # Calculate z_beta from sample size formula
    z_beta = effect_size * np.sqrt(n_per_group / 2) - z_alpha

    # Convert to power
    power = norm.cdf(z_beta)

    return float(np.clip(power, 0, 1))


# =============================================================================
# Permutation Tests
# =============================================================================


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Perform permutation test for difference in means.

    Args:
        group1: First group observations
        group2: Second group observations
        n_permutations: Number of permutations
        alternative: "two-sided", "greater", or "less"
        seed: Random seed

    Returns:
        Dictionary with observed difference and p-value
    """
    rng = np.random.RandomState(seed)

    # Observed difference
    observed_diff = np.mean(group1) - np.mean(group2)

    # Combined data
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    # Permutation distribution
    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(combined)
        perm_diffs[i] = np.mean(combined[:n1]) - np.mean(combined[n1:])

    # Calculate p-value
    if alternative == "two-sided":
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    elif alternative == "greater":
        p_value = np.mean(perm_diffs >= observed_diff)
    elif alternative == "less":
        p_value = np.mean(perm_diffs <= observed_diff)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "alternative": alternative,
        "null_mean": np.mean(perm_diffs),
        "null_std": np.std(perm_diffs),
    }
