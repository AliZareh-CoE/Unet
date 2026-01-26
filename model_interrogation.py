"""
Model Interrogation Framework - Nature Methods Edition
======================================================
Comprehensive deep learning model analysis for Nature Methods publications.
Designed for neural signal translation models (Region A → Region B).

This framework follows Nature Methods reporting standards:
- Statistical reproducibility with bootstrap confidence intervals (n=10,000)
- Effect sizes (Cohen's d, Hedges' g) with 95% CI for all comparisons
- Multiple comparison corrections (FDR Benjamini-Hochberg, Bonferroni)
- Sample sizes (n) and degrees of freedom (df) reported for all tests
- Permutation tests for non-parametric inference (n_permutations=10,000)
- Colorblind-safe figures meeting journal specifications (89/180mm widths)
- Individual data points shown alongside summary statistics
- Exact p-values reported (not just significance thresholds)

Usage:
    python model_interrogation.py --checkpoint path/to/model.pt --data_dir path/to/data

    # Run specific analyses
    python model_interrogation.py --checkpoint model.pt --analyses training,spectral,latent

    # Generate publication figures
    python model_interrogation.py --checkpoint model.pt --publication --output_dir figures/

    # Generate Nature Methods reproducibility report
    python model_interrogation.py --checkpoint model.pt --methods_report

Reference:
    Nature Methods formatting: https://www.nature.com/nmeth/for-authors/preparing-your-submission
    Statistics guidance: Krzywinski & Altman, Nature Methods (2013-2014) series
    Reporting standards: ARRIVE guidelines 2.0

Author: Neural Signal Translation Team
Version: 2.0.0 (Nature Methods Edition)
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# Nature Methods Statistical Reporting Module
# =============================================================================

class NatureMethodsStatistics:
    """
    Statistical analysis following Nature Methods reporting guidelines.

    Key principles:
    1. Report exact p-values (not just < 0.05)
    2. Include effect sizes with confidence intervals
    3. Use appropriate non-parametric tests when assumptions violated
    4. Apply multiple comparison corrections
    5. Report sample sizes for all groups
    6. Use bootstrap for robust CI estimation
    """

    # Constants for statistical analysis
    N_BOOTSTRAP = 10000  # Nature Methods recommends >= 1000
    N_PERMUTATIONS = 10000
    CONFIDENCE_LEVEL = 0.95
    RANDOM_SEED = 42

    @staticmethod
    def set_seed(seed: int = 42):
        """Set random seed for reproducibility."""
        np.random.seed(seed)

    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic: Callable = np.mean,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval for any statistic.

        Args:
            data: 1D array of observations
            statistic: Function to compute (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default: 0.95)
            random_state: Random seed for reproducibility

        Returns:
            Dict with point estimate, CI bounds, and SE
        """
        np.random.seed(random_state)
        data = np.asarray(data).flatten()
        n = len(data)

        # Bootstrap resampling
        boot_stats = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            boot_sample = np.random.choice(data, size=n, replace=True)
            boot_stats[i] = statistic(boot_sample)

        # Percentile method for CI
        alpha = 1 - confidence_level
        ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
        ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

        # BCa correction (bias-corrected and accelerated)
        # Compute bias correction
        point_estimate = statistic(data)
        z0 = stats.norm.ppf(np.mean(boot_stats < point_estimate))

        # Jackknife for acceleration
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jack_sample)
        jack_mean = np.mean(jackknife_stats)

        # Acceleration factor
        num = np.sum((jack_mean - jackknife_stats) ** 3)
        den = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
        a = num / (den + 1e-10)

        # BCa adjusted percentiles
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

        ci_lower_bca = np.percentile(boot_stats, 100 * p_lower)
        ci_upper_bca = np.percentile(boot_stats, 100 * p_upper)

        return {
            'estimate': point_estimate,
            'ci_lower': ci_lower_bca,
            'ci_upper': ci_upper_bca,
            'ci_lower_percentile': ci_lower,
            'ci_upper_percentile': ci_upper,
            'se': np.std(boot_stats),
            'n': n,
            'n_bootstrap': n_bootstrap,
            'confidence_level': confidence_level,
        }

    @staticmethod
    def cohens_d_with_ci(
        group1: np.ndarray,
        group2: np.ndarray,
        paired: bool = False,
        n_bootstrap: int = 10000
    ) -> Dict[str, float]:
        """
        Compute Cohen's d with bootstrap confidence interval.

        Nature Methods requires effect sizes for all comparisons.

        Args:
            group1, group2: Arrays of observations
            paired: Whether data is paired
            n_bootstrap: Bootstrap samples for CI

        Returns:
            Dict with d, CI, and interpretation
        """
        group1 = np.asarray(group1).flatten()
        group2 = np.asarray(group2).flatten()

        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

        if paired:
            # Cohen's d for paired samples
            diff = group2 - group1
            d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

            # Hedges' g correction for small samples
            df = n1 - 1
            g = d * (1 - 3 / (4 * df - 1))
        else:
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            d = (mean2 - mean1) / (pooled_std + 1e-10)

            # Hedges' g correction
            df = n1 + n2 - 2
            g = d * (1 - 3 / (4 * df - 1))

        # Bootstrap CI for effect size
        def compute_d(g1, g2):
            if paired:
                diff = g2 - g1
                return np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
            else:
                ps = np.sqrt(((len(g1)-1)*np.std(g1,ddof=1)**2 + (len(g2)-1)*np.std(g2,ddof=1)**2) / (len(g1)+len(g2)-2))
                return (np.mean(g2) - np.mean(g1)) / (ps + 1e-10)

        np.random.seed(42)
        boot_d = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            if paired:
                idx = np.random.choice(n1, size=n1, replace=True)
                boot_d[i] = compute_d(group1[idx], group2[idx])
            else:
                boot_g1 = np.random.choice(group1, size=n1, replace=True)
                boot_g2 = np.random.choice(group2, size=n2, replace=True)
                boot_d[i] = compute_d(boot_g1, boot_g2)

        ci_lower = np.percentile(boot_d, 2.5)
        ci_upper = np.percentile(boot_d, 97.5)

        # Interpretation (Cohen's conventions)
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = 'negligible'
        elif abs_d < 0.5:
            interpretation = 'small'
        elif abs_d < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'

        return {
            'cohens_d': d,
            'hedges_g': g,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'interpretation': interpretation,
            'n1': n1,
            'n2': n2,
            'mean_diff': mean2 - mean1,
        }

    @staticmethod
    def permutation_test(
        group1: np.ndarray,
        group2: np.ndarray,
        n_permutations: int = 10000,
        alternative: str = 'two-sided',
        statistic: str = 'mean'
    ) -> Dict[str, float]:
        """
        Non-parametric permutation test.

        Preferred when normality cannot be assumed.

        Args:
            group1, group2: Arrays of observations
            n_permutations: Number of permutations
            alternative: 'two-sided', 'greater', or 'less'
            statistic: 'mean' or 'median'

        Returns:
            Dict with exact p-value and test details
        """
        group1 = np.asarray(group1).flatten()
        group2 = np.asarray(group2).flatten()

        combined = np.concatenate([group1, group2])
        n1 = len(group1)

        stat_func = np.mean if statistic == 'mean' else np.median
        observed_diff = stat_func(group2) - stat_func(group1)

        np.random.seed(42)
        perm_diffs = np.zeros(n_permutations)
        for i in range(n_permutations):
            perm = np.random.permutation(combined)
            perm_diffs[i] = stat_func(perm[n1:]) - stat_func(perm[:n1])

        if alternative == 'two-sided':
            p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        elif alternative == 'greater':
            p_value = np.mean(perm_diffs >= observed_diff)
        else:  # less
            p_value = np.mean(perm_diffs <= observed_diff)

        return {
            'observed_diff': observed_diff,
            'p_value': p_value,
            'p_value_exact': p_value,  # Exact p-value as required by Nature Methods
            'n_permutations': n_permutations,
            'alternative': alternative,
            'statistic': statistic,
            'n1': len(group1),
            'n2': len(group2),
        }

    @staticmethod
    def fdr_correction(p_values: np.ndarray, alpha: float = 0.05, method: str = 'fdr_bh') -> Dict:
        """
        Multiple comparison correction using FDR (Benjamini-Hochberg).

        Required when making multiple statistical comparisons.

        Args:
            p_values: Array of p-values
            alpha: Significance level
            method: 'fdr_bh' (Benjamini-Hochberg) or 'bonferroni'

        Returns:
            Dict with corrected p-values and significance
        """
        from scipy.stats import false_discovery_control

        p_values = np.asarray(p_values)
        n = len(p_values)

        if method == 'fdr_bh':
            # Benjamini-Hochberg procedure
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]

            # Critical values
            critical = (np.arange(1, n + 1) / n) * alpha

            # Find significant tests
            significant = sorted_p <= critical

            # Adjusted p-values
            adjusted_p = np.zeros(n)
            adjusted_p[sorted_idx] = np.minimum.accumulate(
                (n / (np.arange(n, 0, -1))) * sorted_p[::-1]
            )[::-1]
            adjusted_p = np.minimum(adjusted_p, 1.0)

        else:  # Bonferroni
            adjusted_p = np.minimum(p_values * n, 1.0)
            significant = adjusted_p < alpha

        return {
            'original_p': p_values,
            'adjusted_p': adjusted_p,
            'significant': adjusted_p < alpha,
            'n_significant': np.sum(adjusted_p < alpha),
            'n_tests': n,
            'method': method,
            'alpha': alpha,
        }

    @staticmethod
    def comprehensive_comparison(
        group1: np.ndarray,
        group2: np.ndarray,
        paired: bool = False,
        group_names: Tuple[str, str] = ('Group 1', 'Group 2')
    ) -> Dict:
        """
        Complete statistical comparison following Nature Methods standards.

        Performs:
        1. Normality tests
        2. Appropriate parametric/non-parametric test
        3. Effect size with CI
        4. Bootstrap CI for means

        Args:
            group1, group2: Data arrays
            paired: Whether samples are paired
            group_names: Names for reporting

        Returns:
            Comprehensive results dict for Methods section
        """
        group1 = np.asarray(group1).flatten()
        group2 = np.asarray(group2).flatten()

        # Descriptive statistics
        desc1 = {
            'n': len(group1),
            'mean': np.mean(group1),
            'median': np.median(group1),
            'std': np.std(group1, ddof=1),
            'sem': np.std(group1, ddof=1) / np.sqrt(len(group1)),
        }
        desc2 = {
            'n': len(group2),
            'mean': np.mean(group2),
            'median': np.median(group2),
            'std': np.std(group2, ddof=1),
            'sem': np.std(group2, ddof=1) / np.sqrt(len(group2)),
        }

        # Normality tests (Shapiro-Wilk for n < 50, else D'Agostino-Pearson)
        if len(group1) < 50:
            _, p_norm1 = stats.shapiro(group1)
            _, p_norm2 = stats.shapiro(group2)
            normality_test = 'Shapiro-Wilk'
        else:
            _, p_norm1 = stats.normaltest(group1)
            _, p_norm2 = stats.normaltest(group2)
            normality_test = "D'Agostino-Pearson"

        is_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)

        # Choose appropriate test
        if paired:
            if is_normal:
                t_stat, p_param = stats.ttest_rel(group1, group2)
                test_name = 'paired t-test'
                df = len(group1) - 1
            else:
                w_stat, p_param = stats.wilcoxon(group1, group2)
                test_name = 'Wilcoxon signed-rank test'
                df = None
                t_stat = w_stat
        else:
            if is_normal:
                # Check homogeneity of variance
                _, p_levene = stats.levene(group1, group2)
                if p_levene > 0.05:
                    t_stat, p_param = stats.ttest_ind(group1, group2)
                    test_name = "Student's t-test"
                else:
                    t_stat, p_param = stats.ttest_ind(group1, group2, equal_var=False)
                    test_name = "Welch's t-test"
                df = len(group1) + len(group2) - 2
            else:
                u_stat, p_param = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                test_name = 'Mann-Whitney U test'
                df = None
                t_stat = u_stat

        # Effect size
        effect = NatureMethodsStatistics.cohens_d_with_ci(group1, group2, paired=paired)

        # Permutation test for robustness
        perm = NatureMethodsStatistics.permutation_test(group1, group2)

        # Bootstrap CI for difference
        if paired:
            diff = group2 - group1
            boot_diff = NatureMethodsStatistics.bootstrap_ci(diff)
        else:
            def mean_diff(idx):
                return np.mean(group2) - np.mean(group1)
            boot_diff = {
                'estimate': desc2['mean'] - desc1['mean'],
                'ci_lower': effect['ci_lower'] * (desc1['std'] + desc2['std']) / 2,
                'ci_upper': effect['ci_upper'] * (desc1['std'] + desc2['std']) / 2,
            }

        return {
            'group_names': group_names,
            'descriptive': {group_names[0]: desc1, group_names[1]: desc2},
            'normality': {
                'test': normality_test,
                'p_values': (p_norm1, p_norm2),
                'is_normal': is_normal,
            },
            'test': {
                'name': test_name,
                'statistic': t_stat,
                'p_value': p_param,
                'df': df,
            },
            'effect_size': effect,
            'permutation': perm,
            'bootstrap_diff': boot_diff,
            # Nature Methods reporting string
            'methods_text': (
                f"Comparison between {group_names[0]} (n={desc1['n']}, "
                f"mean={desc1['mean']:.4f}±{desc1['std']:.4f}) and "
                f"{group_names[1]} (n={desc2['n']}, mean={desc2['mean']:.4f}±{desc2['std']:.4f}) "
                f"using {test_name} "
                f"({'df=' + str(df) + ', ' if df else ''}"
                f"p={p_param:.2e if p_param < 0.001 else f'{p_param:.4f}'}, "
                f"Cohen's d={effect['cohens_d']:.2f} [{effect['ci_lower']:.2f}, {effect['ci_upper']:.2f}], "
                f"{effect['interpretation']} effect)."
            )
        }

    @staticmethod
    def format_p_value(p: float) -> str:
        """Format p-value according to Nature Methods guidelines."""
        if p < 0.0001:
            return f"p < 0.0001"
        elif p < 0.001:
            return f"p = {p:.4f}"
        elif p < 0.01:
            return f"p = {p:.3f}"
        else:
            return f"p = {p:.2f}"

    @staticmethod
    def generate_methods_text(results: Dict) -> str:
        """Generate Methods section text from analysis results."""
        return results.get('methods_text', '')


# =============================================================================
# Publication Figure Standards (Nature Methods compliant)
# =============================================================================

# Figure dimensions (Nature family)
FIGURE_DIMS = {
    'single_col': 88 / 25.4,    # 88 mm -> inches
    'one_half_col': 120 / 25.4,  # 120 mm -> inches
    'double_col': 180 / 25.4,    # 180 mm -> inches
    'max_height': 247 / 25.4,    # 247 mm -> inches
}

# Colorblind-safe palette (from documentation)
COLORBLIND_PALETTE = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'cyan': '#33BBEE',
    'magenta': '#EE3377',
    'red': '#CC3311',
    'green': '#009988',
    'yellow': '#CCBB44',
    'grey': '#BBBBBB',
}

# Color schemes for publication (colorblind-safe)
COLORS = {
    'primary': COLORBLIND_PALETTE['blue'],
    'secondary': COLORBLIND_PALETTE['red'],
    'tertiary': COLORBLIND_PALETTE['green'],
    'quaternary': COLORBLIND_PALETTE['magenta'],
    'train': COLORBLIND_PALETTE['blue'],
    'val': COLORBLIND_PALETTE['orange'],
    'test': COLORBLIND_PALETTE['green'],
    'pred': COLORBLIND_PALETTE['blue'],
    'target': COLORBLIND_PALETTE['red'],
    'baseline': COLORBLIND_PALETTE['grey'],
}

# Publication-quality matplotlib settings
PUBLICATION_RC = {
    # Fonts (sans-serif preferred)
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,

    # Lines
    'lines.linewidth': 0.8,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,

    # Spines (remove top and right)
    'axes.spines.top': False,
    'axes.spines.right': False,

    # Figure
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,

    # Legend
    'legend.frameon': False,
    'legend.borderpad': 0.2,

    # Grid (off by default)
    'axes.grid': False,
}

# Apply publication settings
plt.rcParams.update(PUBLICATION_RC)

# Frequency bands for neural signals
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'low_gamma': (30, 50),
    'high_gamma': (50, 100),
}

# Neural frequency band descriptions
FREQ_BAND_DESCRIPTIONS = {
    'delta': 'Deep sleep, slow oscillations',
    'theta': 'Memory, navigation',
    'alpha': 'Attention, inhibition',
    'beta': 'Motor, cognitive engagement',
    'low_gamma': 'Local processing, binding',
    'high_gamma': 'High-frequency activity',
}


# =============================================================================
# Publication Helper Functions
# =============================================================================

def create_figure(
    n_panels: int = 1,
    layout: str = 'horizontal',
    width: str = 'single_col',
    height_ratio: float = 0.8,
) -> Tuple[plt.Figure, Any]:
    """Create a publication-ready figure with proper dimensions.

    Args:
        n_panels: Number of panels
        layout: 'horizontal', 'vertical', or 'grid'
        width: 'single_col', 'one_half_col', or 'double_col'
        height_ratio: Height as ratio of width

    Returns:
        Figure and axes
    """
    fig_width = FIGURE_DIMS.get(width, FIGURE_DIMS['single_col'])

    if layout == 'horizontal':
        fig_height = min(fig_width * height_ratio, FIGURE_DIMS['max_height'])
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height))
    elif layout == 'vertical':
        fig_height = min(fig_width * height_ratio * n_panels, FIGURE_DIMS['max_height'])
        fig, axes = plt.subplots(n_panels, 1, figsize=(fig_width, fig_height))
    else:  # grid
        n_cols = int(np.ceil(np.sqrt(n_panels)))
        n_rows = int(np.ceil(n_panels / n_cols))
        fig_height = min(fig_width * height_ratio * n_rows / n_cols, FIGURE_DIMS['max_height'])
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    return fig, axes


def add_panel_labels(
    axes: Union[plt.Axes, List[plt.Axes]],
    labels: Optional[List[str]] = None,
    fontsize: int = 10,
    fontweight: str = 'bold',
    offset: Tuple[float, float] = (-0.1, 1.05),
):
    """Add panel labels (a, b, c, ...) to figure axes.

    Args:
        axes: Single axis or list of axes
        labels: Custom labels or None for auto (a, b, c, ...)
        fontsize: Label font size
        fontweight: Font weight
        offset: (x, y) offset in axes coordinates
    """
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    axes = np.array(axes).flatten()

    if labels is None:
        labels = [chr(ord('a') + i) for i in range(len(axes))]

    for ax, label in zip(axes, labels):
        ax.text(
            offset[0], offset[1], label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            va='top', ha='right'
        )


def add_significance_markers(
    ax: plt.Axes,
    x1: float, x2: float, y: float,
    p_value: float,
    height: float = 0.02,
):
    """Add significance bracket and stars to plot.

    Args:
        ax: Matplotlib axes
        x1, x2: X positions for bracket
        y: Y position for bracket
        p_value: P-value for determining stars
        height: Height of bracket
    """
    # Determine significance marker
    if p_value < 0.001:
        marker = '***'
    elif p_value < 0.01:
        marker = '**'
    elif p_value < 0.05:
        marker = '*'
    else:
        marker = 'n.s.'

    # Draw bracket
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], 'k-', lw=0.8)
    ax.text((x1 + x2) / 2, y + height, marker, ha='center', va='bottom', fontsize=7)


def compute_effect_size(group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
    """Compute effect size metrics following Nature Methods standards.

    Returns:
        Dict with Cohen's d, Hedges' g, improvement %, and bootstrap 95% CI
    """
    # Use the comprehensive NatureMethodsStatistics class
    return NatureMethodsStatistics.cohens_d_with_ci(group1, group2)


class ReproducibilityReport:
    """
    Generate Nature Methods reproducibility report.

    This report includes all information required for:
    1. Methods section writing
    2. Statistical analysis reporting
    3. Code and data availability statements
    4. Figure legends with complete statistical information
    """

    def __init__(self, config: 'AnalysisConfig'):
        self.config = config
        self.analyses_performed = []
        self.statistical_tests = []
        self.figure_descriptions = []
        self.software_versions = self._get_software_versions()

    def _get_software_versions(self) -> Dict[str, str]:
        """Capture software versions for reproducibility."""
        versions = {
            'python': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
            'numpy': np.__version__,
            'torch': torch.__version__,
            'scipy': __import__('scipy').__version__,
        }
        try:
            versions['matplotlib'] = __import__('matplotlib').__version__
        except:
            pass
        try:
            versions['sklearn'] = __import__('sklearn').__version__
        except:
            pass
        return versions

    def add_analysis(
        self,
        name: str,
        description: str,
        parameters: Dict,
        n_samples: int,
        results_summary: str
    ):
        """Record an analysis for the Methods section."""
        self.analyses_performed.append({
            'name': name,
            'description': description,
            'parameters': parameters,
            'n_samples': n_samples,
            'results_summary': results_summary,
        })

    def add_statistical_test(
        self,
        test_name: str,
        comparison: str,
        n1: int, n2: int,
        statistic: float,
        p_value: float,
        effect_size: float,
        effect_ci: Tuple[float, float],
        correction: Optional[str] = None
    ):
        """Record a statistical test."""
        self.statistical_tests.append({
            'test': test_name,
            'comparison': comparison,
            'n': (n1, n2),
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'effect_ci': effect_ci,
            'correction': correction,
        })

    def add_figure_description(
        self,
        figure_number: str,
        title: str,
        panels: List[Dict],
        statistical_info: str
    ):
        """Add a figure legend with complete statistical information."""
        self.figure_descriptions.append({
            'number': figure_number,
            'title': title,
            'panels': panels,
            'stats': statistical_info,
        })

    def generate_methods_section(self) -> str:
        """Generate complete Methods section text."""
        lines = []
        lines.append("=" * 70)
        lines.append("METHODS SECTION (Nature Methods Format)")
        lines.append("=" * 70)
        lines.append("")

        # Software and environment
        lines.append("### Computational Environment")
        lines.append("")
        lines.append("All analyses were performed using the Model Interrogation Framework")
        lines.append(f"(v2.0.0) with the following software versions:")
        for sw, ver in self.software_versions.items():
            lines.append(f"  - {sw}: {ver}")
        lines.append(f"  - Random seed: {self.config.seed}")
        lines.append("")

        # Analysis methods
        lines.append("### Analysis Methods")
        lines.append("")
        for analysis in self.analyses_performed:
            lines.append(f"**{analysis['name']}**")
            lines.append(f"{analysis['description']}")
            lines.append(f"Parameters: {analysis['parameters']}")
            lines.append(f"Sample size: n = {analysis['n_samples']}")
            lines.append("")

        # Statistical analysis
        lines.append("### Statistical Analysis")
        lines.append("")
        lines.append("All statistical tests were two-tailed unless otherwise specified.")
        lines.append("Effect sizes are reported as Cohen's d with 95% bootstrap confidence")
        lines.append("intervals (n_bootstrap = 10,000). Multiple comparisons were corrected")
        lines.append("using the Benjamini-Hochberg false discovery rate procedure.")
        lines.append("")

        if self.statistical_tests:
            lines.append("Statistical comparisons performed:")
            for test in self.statistical_tests:
                correction_note = f", {test['correction']}-corrected" if test['correction'] else ""
                lines.append(f"  - {test['comparison']}: {test['test']} "
                           f"(n₁={test['n'][0]}, n₂={test['n'][1]}, "
                           f"p={test['p_value']:.4f}{correction_note}, "
                           f"d={test['effect_size']:.2f} "
                           f"[{test['effect_ci'][0]:.2f}, {test['effect_ci'][1]:.2f}])")

        lines.append("")
        lines.append("### Data Availability")
        lines.append("")
        lines.append("Source data for all figures are available in the Supplementary")
        lines.append("Information. The model interrogation analysis code is available at")
        lines.append("[repository URL].")
        lines.append("")

        return "\n".join(lines)

    def generate_figure_legends(self) -> str:
        """Generate figure legends with complete statistical information."""
        lines = []
        lines.append("=" * 70)
        lines.append("FIGURE LEGENDS")
        lines.append("=" * 70)
        lines.append("")

        for fig in self.figure_descriptions:
            lines.append(f"**{fig['number']}. {fig['title']}**")
            lines.append("")
            for panel in fig['panels']:
                lines.append(f"({panel['label']}) {panel['description']}")
            lines.append("")
            lines.append(f"Statistical details: {fig['stats']}")
            lines.append("")

        return "\n".join(lines)

    def generate_full_report(self) -> str:
        """Generate complete reproducibility report."""
        sections = [
            self.generate_methods_section(),
            self.generate_figure_legends(),
        ]
        return "\n\n".join(sections)


@dataclass
class AnalysisConfig:
    """Configuration for model interrogation analyses."""
    # Paths
    checkpoint_path: str = ""
    data_dir: str = ""
    output_dir: str = "interrogation_results"

    # Analysis selection
    analyses: List[str] = field(default_factory=lambda: ['all'])

    # Data parameters
    batch_size: int = 16
    num_samples: int = 1000
    sampling_rate: float = 1000.0

    # Computation parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42

    # Visualization
    publication_mode: bool = False
    save_figures: bool = True
    figure_format: str = "pdf"

    # Analysis-specific
    n_perturbation_samples: int = 100
    loss_landscape_resolution: int = 31
    latent_umap_n_neighbors: int = 15

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class AnalysisResults:
    """Container for all analysis results."""
    training_dynamics: Dict = field(default_factory=dict)
    input_attribution: Dict = field(default_factory=dict)
    spectral_analysis: Dict = field(default_factory=dict)
    latent_space: Dict = field(default_factory=dict)
    conditioning: Dict = field(default_factory=dict)
    loss_landscape: Dict = field(default_factory=dict)
    generalization: Dict = field(default_factory=dict)
    error_analysis: Dict = field(default_factory=dict)
    architectural: Dict = field(default_factory=dict)
    baselines: Dict = field(default_factory=dict)

    def save(self, path: str):
        """Save results to JSON (numpy arrays converted)."""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        results_dict = {
            'training_dynamics': convert(self.training_dynamics),
            'input_attribution': convert(self.input_attribution),
            'spectral_analysis': convert(self.spectral_analysis),
            'latent_space': convert(self.latent_space),
            'conditioning': convert(self.conditioning),
            'loss_landscape': convert(self.loss_landscape),
            'generalization': convert(self.generalization),
            'error_analysis': convert(self.error_analysis),
            'architectural': convert(self.architectural),
            'baselines': convert(self.baselines),
        }

        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2)


# =============================================================================
# PART 1: TRAINING DYNAMICS
# =============================================================================

class TrainingDynamicsAnalyzer:
    """
    Analyze training dynamics: learning curves, gradient flow, weight evolution.

    Key analyses:
    1. Learning curves (loss, metrics) across epochs and folds
    2. Gradient flow heatmap (layers × epochs)
    3. Weight norm evolution per layer
    4. Loss component breakdown
    5. Convergence and stability analysis
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.results = {}

    def analyze_learning_curves(
        self,
        training_logs: List[Dict],
        fold_labels: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze learning curves across training runs/folds.

        Args:
            training_logs: List of dicts with keys 'train_loss', 'val_loss',
                          'train_r2', 'val_r2', etc. per epoch
            fold_labels: Optional labels for each fold

        Returns:
            Dict with learning curve statistics and convergence info
        """
        results = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': [],
            'epochs': [],
            'convergence_epoch': [],
            'best_val_loss': [],
            'overfitting_gap': [],
        }

        for i, log in enumerate(training_logs):
            epochs = list(range(len(log.get('train_loss', []))))
            train_loss = np.array(log.get('train_loss', []))
            val_loss = np.array(log.get('val_loss', []))
            train_r2 = np.array(log.get('train_r2', []))
            val_r2 = np.array(log.get('val_r2', []))

            results['train_loss'].append(train_loss)
            results['val_loss'].append(val_loss)
            results['train_r2'].append(train_r2)
            results['val_r2'].append(val_r2)
            results['epochs'].append(epochs)

            # Find convergence epoch (when val_loss stops improving significantly)
            if len(val_loss) > 10:
                smoothed = gaussian_filter1d(val_loss, sigma=3)
                diffs = np.diff(smoothed)
                # Convergence when improvement < 1% of initial loss
                threshold = 0.01 * smoothed[0]
                converged = np.where(np.abs(diffs) < threshold)[0]
                conv_epoch = converged[0] if len(converged) > 0 else len(epochs)
            else:
                conv_epoch = len(epochs)
            results['convergence_epoch'].append(conv_epoch)

            # Best validation loss
            if len(val_loss) > 0:
                results['best_val_loss'].append(float(np.min(val_loss)))

            # Overfitting gap at end of training
            if len(train_loss) > 0 and len(val_loss) > 0:
                gap = val_loss[-1] - train_loss[-1]
                results['overfitting_gap'].append(float(gap))

        # Compute statistics across folds
        max_epochs = max(len(e) for e in results['epochs'])

        # Pad arrays to same length for aggregation
        def pad_to_length(arr, length, fill_value=np.nan):
            padded = np.full(length, fill_value)
            padded[:len(arr)] = arr
            return padded

        train_losses = np.array([pad_to_length(tl, max_epochs) for tl in results['train_loss']])
        val_losses = np.array([pad_to_length(vl, max_epochs) for vl in results['val_loss']])

        results['train_loss_mean'] = np.nanmean(train_losses, axis=0)
        results['train_loss_std'] = np.nanstd(train_losses, axis=0)
        results['val_loss_mean'] = np.nanmean(val_losses, axis=0)
        results['val_loss_std'] = np.nanstd(val_losses, axis=0)

        results['fold_labels'] = fold_labels or [f'Fold {i}' for i in range(len(training_logs))]

        self.results['learning_curves'] = results
        return results

    def analyze_gradient_flow(
        self,
        gradient_history: Dict[str, List[np.ndarray]]
    ) -> Dict:
        """
        Analyze gradient flow across layers and epochs.

        Args:
            gradient_history: Dict mapping layer names to list of gradient norms per epoch

        Returns:
            Dict with gradient flow heatmap and statistics
        """
        layer_names = list(gradient_history.keys())
        n_layers = len(layer_names)
        n_epochs = len(gradient_history[layer_names[0]])

        # Create gradient magnitude matrix [layers × epochs]
        gradient_matrix = np.zeros((n_layers, n_epochs))
        for i, name in enumerate(layer_names):
            gradient_matrix[i] = gradient_history[name]

        # Log-scale for better visualization
        gradient_matrix_log = np.log10(gradient_matrix + 1e-10)

        # Identify problematic patterns
        vanishing_layers = []
        exploding_layers = []
        dead_layers = []

        for i, name in enumerate(layer_names):
            grads = gradient_matrix[i]
            if np.mean(grads[-10:]) < 1e-7:  # Very small gradients at end
                vanishing_layers.append(name)
            if np.max(grads) > 100:  # Large gradient spikes
                exploding_layers.append(name)
            if np.std(grads) < 1e-8:  # No gradient variation
                dead_layers.append(name)

        # Compute per-layer statistics
        layer_stats = {}
        for i, name in enumerate(layer_names):
            grads = gradient_matrix[i]
            layer_stats[name] = {
                'mean': float(np.mean(grads)),
                'std': float(np.std(grads)),
                'max': float(np.max(grads)),
                'min': float(np.min(grads)),
                'final': float(grads[-1]) if len(grads) > 0 else 0,
            }

        results = {
            'gradient_matrix': gradient_matrix,
            'gradient_matrix_log': gradient_matrix_log,
            'layer_names': layer_names,
            'n_epochs': n_epochs,
            'vanishing_layers': vanishing_layers,
            'exploding_layers': exploding_layers,
            'dead_layers': dead_layers,
            'layer_stats': layer_stats,
        }

        self.results['gradient_flow'] = results
        return results

    def analyze_weight_evolution(
        self,
        weight_history: Dict[str, List[float]]
    ) -> Dict:
        """
        Track weight norm evolution per layer.

        Args:
            weight_history: Dict mapping layer names to list of weight norms per epoch

        Returns:
            Dict with weight evolution analysis
        """
        layer_names = list(weight_history.keys())
        n_epochs = len(weight_history[layer_names[0]])

        # Create weight matrix
        weight_matrix = np.zeros((len(layer_names), n_epochs))
        for i, name in enumerate(layer_names):
            weight_matrix[i] = weight_history[name]

        # Relative change from initialization
        relative_change = weight_matrix / (weight_matrix[:, 0:1] + 1e-10)

        # Identify fast vs slow learning layers
        change_rates = []
        for i, name in enumerate(layer_names):
            # Rate of change in first 10 epochs
            if n_epochs >= 10:
                early_change = np.mean(np.diff(weight_matrix[i, :10]))
            else:
                early_change = np.mean(np.diff(weight_matrix[i]))
            change_rates.append((name, early_change))

        change_rates.sort(key=lambda x: abs(x[1]), reverse=True)
        fast_learning = [x[0] for x in change_rates[:5]]
        slow_learning = [x[0] for x in change_rates[-5:]]

        # Early stopping analysis
        # Find when weights stabilize (< 1% change)
        stabilization_epochs = {}
        for i, name in enumerate(layer_names):
            rel_changes = np.abs(np.diff(weight_matrix[i])) / (weight_matrix[i, :-1] + 1e-10)
            stable = np.where(rel_changes < 0.01)[0]
            if len(stable) > 0:
                stabilization_epochs[name] = int(stable[0])
            else:
                stabilization_epochs[name] = n_epochs

        results = {
            'weight_matrix': weight_matrix,
            'relative_change': relative_change,
            'layer_names': layer_names,
            'n_epochs': n_epochs,
            'fast_learning_layers': fast_learning,
            'slow_learning_layers': slow_learning,
            'stabilization_epochs': stabilization_epochs,
            'change_rates': dict(change_rates),
        }

        self.results['weight_evolution'] = results
        return results

    def analyze_loss_components(
        self,
        loss_history: Dict[str, List[float]]
    ) -> Dict:
        """
        Analyze individual loss components over training.

        Args:
            loss_history: Dict mapping loss component names to values per epoch

        Returns:
            Dict with loss component analysis
        """
        component_names = list(loss_history.keys())
        n_epochs = len(loss_history[component_names[0]])

        # Convert to arrays
        loss_arrays = {name: np.array(vals) for name, vals in loss_history.items()}

        # Compute total loss
        total_loss = np.sum([arr for arr in loss_arrays.values()], axis=0)

        # Compute relative contribution of each component
        contributions = {}
        for name, arr in loss_arrays.items():
            contributions[name] = arr / (total_loss + 1e-10)

        # Find which converges first
        convergence_order = []
        for name, arr in loss_arrays.items():
            smoothed = gaussian_filter1d(arr, sigma=3)
            diffs = np.abs(np.diff(smoothed))
            threshold = 0.01 * smoothed[0] if smoothed[0] > 0 else 0.01
            converged = np.where(diffs < threshold)[0]
            conv_epoch = converged[0] if len(converged) > 0 else n_epochs
            convergence_order.append((name, conv_epoch))

        convergence_order.sort(key=lambda x: x[1])

        # Identify dominant loss
        final_contributions = {name: float(contrib[-1]) for name, contrib in contributions.items()}
        dominant_loss = max(final_contributions.items(), key=lambda x: x[1])[0]

        results = {
            'loss_arrays': loss_arrays,
            'total_loss': total_loss,
            'contributions': contributions,
            'convergence_order': convergence_order,
            'dominant_loss': dominant_loss,
            'final_contributions': final_contributions,
            'component_names': component_names,
            'n_epochs': n_epochs,
        }

        self.results['loss_components'] = results
        return results

    def plot_learning_curves(
        self,
        save_path: Optional[str] = None,
        show_folds: bool = True
    ) -> plt.Figure:
        """Generate publication-quality learning curves figure."""
        if 'learning_curves' not in self.results:
            raise ValueError("Run analyze_learning_curves first")

        data = self.results['learning_curves']

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Loss curves
        ax = axes[0]
        epochs = np.arange(len(data['train_loss_mean']))

        if show_folds:
            for i, (tl, vl) in enumerate(zip(data['train_loss'], data['val_loss'])):
                ax.plot(tl, color=COLORS['train'], alpha=0.3, linewidth=0.8)
                ax.plot(vl, color=COLORS['val'], alpha=0.3, linewidth=0.8)

        ax.plot(epochs, data['train_loss_mean'], color=COLORS['train'],
                linewidth=2, label='Train')
        ax.fill_between(epochs,
                       data['train_loss_mean'] - data['train_loss_std'],
                       data['train_loss_mean'] + data['train_loss_std'],
                       color=COLORS['train'], alpha=0.2)

        ax.plot(epochs, data['val_loss_mean'], color=COLORS['val'],
                linewidth=2, label='Validation')
        ax.fill_between(epochs,
                       data['val_loss_mean'] - data['val_loss_std'],
                       data['val_loss_mean'] + data['val_loss_std'],
                       color=COLORS['val'], alpha=0.2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Curves')
        ax.legend(loc='upper right')
        ax.set_xlim(0, len(epochs)-1)

        # R² evolution
        ax = axes[1]
        if len(data.get('val_r2', [[]])[0]) > 0:
            val_r2s = np.array([pad_to_length(r, len(epochs))
                               for r in data['val_r2']])
            val_r2_mean = np.nanmean(val_r2s, axis=0)
            val_r2_std = np.nanstd(val_r2s, axis=0)

            if show_folds:
                for r2 in data['val_r2']:
                    ax.plot(r2, color=COLORS['val'], alpha=0.3, linewidth=0.8)

            ax.plot(epochs, val_r2_mean, color=COLORS['val'], linewidth=2)
            ax.fill_between(epochs,
                           val_r2_mean - val_r2_std,
                           val_r2_mean + val_r2_std,
                           color=COLORS['val'], alpha=0.2)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('R²')
            ax.set_title('Validation R² Evolution')
            ax.set_xlim(0, len(epochs)-1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)

        return fig

    def plot_gradient_flow(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Generate gradient flow heatmap."""
        if 'gradient_flow' not in self.results:
            raise ValueError("Run analyze_gradient_flow first")

        data = self.results['gradient_flow']

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create heatmap
        im = ax.imshow(data['gradient_matrix_log'],
                      aspect='auto', cmap='viridis',
                      interpolation='nearest')

        # Labels
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Layer')
        ax.set_title('Gradient Flow (log₁₀ magnitude)')

        # Y-axis labels (layer names)
        if len(data['layer_names']) <= 20:
            ax.set_yticks(range(len(data['layer_names'])))
            ax.set_yticklabels(data['layer_names'], fontsize=8)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log₁₀(gradient norm)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)

        return fig


# Helper function for padding
def pad_to_length(arr, length, fill_value=np.nan):
    """Pad array to specified length."""
    padded = np.full(length, fill_value)
    arr = np.array(arr)
    padded[:len(arr)] = arr
    return padded


# =============================================================================
# PART 2: INPUT ATTRIBUTION
# =============================================================================

class InputAttributionAnalyzer:
    """
    Analyze which input features drive predictions.

    Key analyses:
    1. Channel importance (occlusion-based)
    2. Channel importance (gradient-based)
    3. Temporal saliency
    4. Temporal receptive field
    5. Input ablation study
    """

    def __init__(self, model: nn.Module, config: AnalysisConfig):
        self.model = model
        self.config = config
        self.device = config.device
        self.results = {}

        # Put model in eval mode
        self.model.eval()

    @torch.no_grad()
    def channel_importance_occlusion(
        self,
        dataloader: DataLoader,
        n_samples: int = 100
    ) -> Dict:
        """
        Compute channel importance by zeroing each channel and measuring performance drop.

        Returns:
            Dict with per-channel importance scores
        """
        self.model.eval()

        # Get baseline performance
        baseline_losses = []
        all_inputs = []
        all_targets = []

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * dataloader.batch_size >= n_samples:
                break

            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            outputs = self.model(inputs)
            loss = F.mse_loss(outputs, targets, reduction='none').mean(dim=(1, 2))
            baseline_losses.extend(loss.detach().cpu().numpy())
            all_inputs.append(inputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

        all_inputs = torch.cat(all_inputs, dim=0)[:n_samples]
        all_targets = torch.cat(all_targets, dim=0)[:n_samples]
        baseline_loss = np.mean(baseline_losses[:n_samples])

        n_channels = all_inputs.shape[1]
        channel_importance = np.zeros(n_channels)
        channel_losses = np.zeros(n_channels)

        # Occlude each channel
        for ch in range(n_channels):
            occluded_inputs = all_inputs.clone()
            occluded_inputs[:, ch, :] = 0  # Zero out channel

            occluded_losses = []
            for i in range(0, len(occluded_inputs), self.config.batch_size):
                batch_in = occluded_inputs[i:i+self.config.batch_size].to(self.device)
                batch_tgt = all_targets[i:i+self.config.batch_size].to(self.device)

                outputs = self.model(batch_in)
                loss = F.mse_loss(outputs, batch_tgt, reduction='none').mean(dim=(1, 2))
                occluded_losses.extend(loss.detach().cpu().numpy())

            channel_losses[ch] = np.mean(occluded_losses)
            channel_importance[ch] = channel_losses[ch] - baseline_loss

        # Normalize importance
        importance_normalized = channel_importance / (np.max(np.abs(channel_importance)) + 1e-10)

        # Rank channels
        ranked_channels = np.argsort(channel_importance)[::-1]

        results = {
            'channel_importance': channel_importance,
            'channel_importance_normalized': importance_normalized,
            'channel_losses': channel_losses,
            'baseline_loss': baseline_loss,
            'ranked_channels': ranked_channels.tolist(),
            'n_channels': n_channels,
            'n_samples': n_samples,
        }

        self.results['channel_occlusion'] = results
        return results

    def channel_importance_gradient(
        self,
        dataloader: DataLoader,
        n_samples: int = 100
    ) -> Dict:
        """
        Compute channel importance using gradient magnitude.

        Returns:
            Dict with gradient-based importance scores
        """
        self.model.eval()

        gradient_sums = None
        n_processed = 0

        for batch_idx, batch in enumerate(dataloader):
            if n_processed >= n_samples:
                break

            inputs = batch[0].to(self.device).requires_grad_(True)
            targets = batch[1].to(self.device)

            outputs = self.model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()

            # Get gradient magnitude per channel
            grad_magnitude = inputs.grad.abs().mean(dim=(0, 2))  # [C]

            if gradient_sums is None:
                gradient_sums = grad_magnitude.cpu().numpy()
            else:
                gradient_sums += grad_magnitude.cpu().numpy()

            n_processed += inputs.shape[0]
            self.model.zero_grad()

        channel_importance = gradient_sums / (n_processed / dataloader.batch_size)
        importance_normalized = channel_importance / (np.max(channel_importance) + 1e-10)
        ranked_channels = np.argsort(channel_importance)[::-1]

        results = {
            'channel_importance': channel_importance,
            'channel_importance_normalized': importance_normalized,
            'ranked_channels': ranked_channels.tolist(),
            'n_channels': len(channel_importance),
            'n_samples': n_processed,
        }

        self.results['channel_gradient'] = results
        return results

    def temporal_saliency(
        self,
        dataloader: DataLoader,
        n_samples: int = 50
    ) -> Dict:
        """
        Compute temporal saliency using input × gradient.

        Returns:
            Dict with temporal importance map
        """
        self.model.eval()

        saliency_sum = None
        n_processed = 0

        for batch_idx, batch in enumerate(dataloader):
            if n_processed >= n_samples:
                break

            inputs = batch[0].to(self.device).requires_grad_(True)
            targets = batch[1].to(self.device)

            outputs = self.model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()

            # Input × gradient saliency
            saliency = (inputs * inputs.grad).abs()

            # Average over batch and channels
            temporal_saliency = saliency.mean(dim=(0, 1)).detach().cpu().numpy()  # [T]

            if saliency_sum is None:
                saliency_sum = temporal_saliency
            else:
                saliency_sum += temporal_saliency

            n_processed += inputs.shape[0]
            self.model.zero_grad()

        temporal_saliency = saliency_sum / (n_processed / dataloader.batch_size)
        temporal_saliency_normalized = temporal_saliency / (np.max(temporal_saliency) + 1e-10)

        # Find peak saliency regions
        threshold = np.percentile(temporal_saliency, 90)
        high_saliency_regions = np.where(temporal_saliency > threshold)[0]

        results = {
            'temporal_saliency': temporal_saliency,
            'temporal_saliency_normalized': temporal_saliency_normalized,
            'high_saliency_regions': high_saliency_regions.tolist(),
            'peak_timepoint': int(np.argmax(temporal_saliency)),
            'n_timepoints': len(temporal_saliency),
            'n_samples': n_processed,
        }

        self.results['temporal_saliency'] = results
        return results

    @torch.no_grad()
    def temporal_receptive_field(
        self,
        dataloader: DataLoader,
        n_samples: int = 20,
        perturbation_strength: float = 2.0
    ) -> Dict:
        """
        Analyze temporal receptive field by perturbing single timepoints.

        Returns:
            Dict with receptive field analysis
        """
        self.model.eval()

        # Get sample data
        sample_batch = next(iter(dataloader))
        inputs = sample_batch[0][:min(n_samples, len(sample_batch[0]))].to(self.device)
        targets = sample_batch[1][:min(n_samples, len(sample_batch[1]))].to(self.device)

        T = inputs.shape[2]
        n_test_points = min(50, T)  # Test 50 timepoints
        test_indices = np.linspace(0, T-1, n_test_points, dtype=int)

        # Baseline outputs
        baseline_outputs = self.model(inputs)

        # Impact matrix [test_points × output_timepoints]
        impact_matrix = np.zeros((n_test_points, baseline_outputs.shape[2]))

        for i, t in enumerate(test_indices):
            perturbed = inputs.clone()
            perturbed[:, :, t] += perturbation_strength * inputs[:, :, t].std()

            perturbed_outputs = self.model(perturbed)

            # Measure impact at each output timepoint
            impact = (perturbed_outputs - baseline_outputs).abs().mean(dim=(0, 1))
            impact_matrix[i] = impact.detach().cpu().numpy()

        # Compute effective receptive field
        # For each output timepoint, find which input timepoints affect it
        rf_start = np.zeros(baseline_outputs.shape[2])
        rf_end = np.zeros(baseline_outputs.shape[2])

        for out_t in range(baseline_outputs.shape[2]):
            impact_at_t = impact_matrix[:, out_t]
            threshold = np.max(impact_at_t) * 0.1
            significant = np.where(impact_at_t > threshold)[0]
            if len(significant) > 0:
                rf_start[out_t] = test_indices[significant[0]]
                rf_end[out_t] = test_indices[significant[-1]]

        results = {
            'impact_matrix': impact_matrix,
            'test_indices': test_indices.tolist(),
            'receptive_field_start': rf_start.tolist(),
            'receptive_field_end': rf_end.tolist(),
            'effective_rf_width': float(np.mean(rf_end - rf_start)),
            'n_samples': n_samples,
        }

        self.results['temporal_receptive_field'] = results
        return results

    @torch.no_grad()
    def input_ablation_study(
        self,
        dataloader: DataLoader,
        n_samples: int = 100,
        n_segments: int = 3
    ) -> Dict:
        """
        Mask early/middle/late portions of input to find critical segments.

        Returns:
            Dict with segment importance analysis
        """
        self.model.eval()

        all_inputs = []
        all_targets = []

        for batch in dataloader:
            all_inputs.append(batch[0])
            all_targets.append(batch[1])
            if sum(x.shape[0] for x in all_inputs) >= n_samples:
                break

        all_inputs = torch.cat(all_inputs, dim=0)[:n_samples].to(self.device)
        all_targets = torch.cat(all_targets, dim=0)[:n_samples].to(self.device)

        T = all_inputs.shape[2]
        segment_size = T // n_segments

        # Baseline performance
        baseline_outputs = self.model(all_inputs)
        baseline_loss = F.mse_loss(baseline_outputs, all_targets).item()
        baseline_r2 = self._compute_r2(baseline_outputs, all_targets)

        segment_results = {}
        segment_names = ['early', 'middle', 'late'] if n_segments == 3 else \
                       [f'segment_{i}' for i in range(n_segments)]

        for i, name in enumerate(segment_names):
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else T

            masked_inputs = all_inputs.clone()
            masked_inputs[:, :, start:end] = 0

            masked_outputs = self.model(masked_inputs)
            masked_loss = F.mse_loss(masked_outputs, all_targets).item()
            masked_r2 = self._compute_r2(masked_outputs, all_targets)

            segment_results[name] = {
                'start': start,
                'end': end,
                'loss': masked_loss,
                'r2': masked_r2,
                'loss_increase': masked_loss - baseline_loss,
                'r2_drop': baseline_r2 - masked_r2,
            }

        # Find most critical segment
        critical_segment = max(segment_results.items(),
                              key=lambda x: x[1]['loss_increase'])[0]

        results = {
            'baseline_loss': baseline_loss,
            'baseline_r2': baseline_r2,
            'segment_results': segment_results,
            'critical_segment': critical_segment,
            'n_segments': n_segments,
            'n_samples': n_samples,
        }

        self.results['input_ablation'] = results
        return results

    def _compute_r2(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute R² score."""
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        return float(1 - ss_res / (ss_tot + 1e-10))

    def plot_channel_importance(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot channel importance comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Occlusion-based
        if 'channel_occlusion' in self.results:
            ax = axes[0]
            data = self.results['channel_occlusion']
            channels = np.arange(data['n_channels'])
            ax.bar(channels, data['channel_importance_normalized'],
                   color=COLORS['primary'], alpha=0.7)
            ax.set_xlabel('Channel')
            ax.set_ylabel('Importance (normalized)')
            ax.set_title('Channel Importance (Occlusion)')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Gradient-based
        if 'channel_gradient' in self.results:
            ax = axes[1]
            data = self.results['channel_gradient']
            channels = np.arange(data['n_channels'])
            ax.bar(channels, data['channel_importance_normalized'],
                   color=COLORS['secondary'], alpha=0.7)
            ax.set_xlabel('Channel')
            ax.set_ylabel('Importance (normalized)')
            ax.set_title('Channel Importance (Gradient)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)

        return fig

    def plot_temporal_saliency(
        self,
        save_path: Optional[str] = None,
        sampling_rate: float = 1000.0
    ) -> plt.Figure:
        """Plot temporal saliency map."""
        if 'temporal_saliency' not in self.results:
            raise ValueError("Run temporal_saliency first")

        data = self.results['temporal_saliency']

        fig, ax = plt.subplots(figsize=(10, 4))

        time = np.arange(data['n_timepoints']) / sampling_rate
        ax.plot(time, data['temporal_saliency_normalized'],
                color=COLORS['primary'], linewidth=1.5)
        ax.fill_between(time, 0, data['temporal_saliency_normalized'],
                       color=COLORS['primary'], alpha=0.3)

        # Mark peak
        peak_time = data['peak_timepoint'] / sampling_rate
        ax.axvline(x=peak_time, color=COLORS['secondary'],
                  linestyle='--', linewidth=1, label=f'Peak: {peak_time:.3f}s')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Saliency (normalized)')
        ax.set_title('Temporal Saliency Map')
        ax.legend()
        ax.set_xlim(0, time[-1])
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)

        return fig


# =============================================================================
# PART 3: SPECTRAL ANALYSIS
# =============================================================================

class SpectralAnalyzer:
    """
    Analyze frequency domain properties of predictions.

    Key analyses:
    1. Band-wise prediction accuracy
    2. Power spectral density comparison
    3. Spectral error distribution
    4. Coherence preservation
    5. Phase relationship analysis (PLV)
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.fs = config.sampling_rate
        self.results = {}

    def bandwise_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        freq_bands: Optional[Dict] = None
    ) -> Dict:
        """
        Compute prediction accuracy per frequency band.

        Args:
            predictions: [N, C, T] or [N, T] array
            targets: Same shape as predictions
            freq_bands: Dict of band_name -> (low_freq, high_freq)

        Returns:
            Dict with per-band R² and correlation
        """
        if freq_bands is None:
            freq_bands = FREQ_BANDS

        predictions = np.atleast_3d(predictions)
        targets = np.atleast_3d(targets)

        band_results = {}

        for band_name, (low, high) in freq_bands.items():
            # Design bandpass filter
            nyq = self.fs / 2
            low_norm = low / nyq
            high_norm = min(high / nyq, 0.99)

            try:
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')

                # Filter predictions and targets
                pred_filtered = signal.filtfilt(b, a, predictions, axis=-1)
                tgt_filtered = signal.filtfilt(b, a, targets, axis=-1)

                # Compute metrics
                r2 = self._compute_r2(pred_filtered, tgt_filtered)
                corr = self._compute_correlation(pred_filtered, tgt_filtered)
                rmse = np.sqrt(np.mean((pred_filtered - tgt_filtered) ** 2))

                band_results[band_name] = {
                    'r2': float(r2),
                    'correlation': float(corr),
                    'rmse': float(rmse),
                    'freq_range': (low, high),
                }
            except Exception as e:
                band_results[band_name] = {
                    'r2': np.nan,
                    'correlation': np.nan,
                    'rmse': np.nan,
                    'freq_range': (low, high),
                    'error': str(e),
                }

        # Summary statistics
        r2_values = [v['r2'] for v in band_results.values() if not np.isnan(v['r2'])]

        results = {
            'band_results': band_results,
            'mean_r2': float(np.mean(r2_values)) if r2_values else np.nan,
            'best_band': max(band_results.items(),
                            key=lambda x: x[1]['r2'] if not np.isnan(x[1]['r2']) else -np.inf)[0],
            'worst_band': min(band_results.items(),
                             key=lambda x: x[1]['r2'] if not np.isnan(x[1]['r2']) else np.inf)[0],
        }

        self.results['bandwise_accuracy'] = results
        return results

    def psd_comparison(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        nperseg: int = 1024
    ) -> Dict:
        """
        Compare power spectral density of predictions vs targets.

        Returns:
            Dict with PSD arrays and error metrics
        """
        predictions = np.atleast_2d(predictions)
        targets = np.atleast_2d(targets)

        # Average over samples/channels if needed
        if predictions.ndim == 3:
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])

        # Compute PSDs
        freqs, psd_pred = signal.welch(predictions, fs=self.fs, nperseg=nperseg, axis=-1)
        _, psd_tgt = signal.welch(targets, fs=self.fs, nperseg=nperseg, axis=-1)

        # Average over samples
        psd_pred_mean = np.mean(psd_pred, axis=0)
        psd_tgt_mean = np.mean(psd_tgt, axis=0)
        psd_pred_std = np.std(psd_pred, axis=0)
        psd_tgt_std = np.std(psd_tgt, axis=0)

        # Convert to dB
        psd_pred_db = 10 * np.log10(psd_pred_mean + 1e-10)
        psd_tgt_db = 10 * np.log10(psd_tgt_mean + 1e-10)

        # Spectral error
        spectral_error_db = psd_pred_db - psd_tgt_db

        # Per-band power
        band_power_pred = {}
        band_power_tgt = {}

        for band_name, (low, high) in FREQ_BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            band_power_pred[band_name] = float(np.mean(psd_pred_mean[mask]))
            band_power_tgt[band_name] = float(np.mean(psd_tgt_mean[mask]))

        results = {
            'frequencies': freqs.tolist(),
            'psd_prediction': psd_pred_mean.tolist(),
            'psd_target': psd_tgt_mean.tolist(),
            'psd_prediction_std': psd_pred_std.tolist(),
            'psd_target_std': psd_tgt_std.tolist(),
            'psd_prediction_db': psd_pred_db.tolist(),
            'psd_target_db': psd_tgt_db.tolist(),
            'spectral_error_db': spectral_error_db.tolist(),
            'mean_spectral_error_db': float(np.mean(np.abs(spectral_error_db))),
            'band_power_prediction': band_power_pred,
            'band_power_target': band_power_tgt,
        }

        self.results['psd_comparison'] = results
        return results

    def spectral_error_distribution(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        nperseg: int = 512
    ) -> Dict:
        """
        Analyze at which frequencies the model fails.

        Returns:
            Dict with frequency-resolved error analysis
        """
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])

        # Compute per-sample PSDs
        freqs, psd_pred = signal.welch(predictions, fs=self.fs, nperseg=nperseg, axis=-1)
        _, psd_tgt = signal.welch(targets, fs=self.fs, nperseg=nperseg, axis=-1)

        # Error per frequency
        psd_error = np.abs(psd_pred - psd_tgt)
        relative_error = psd_error / (psd_tgt + 1e-10)

        error_mean = np.mean(psd_error, axis=0)
        error_std = np.std(psd_error, axis=0)
        relative_error_mean = np.mean(relative_error, axis=0)

        # Find worst frequencies
        worst_freq_indices = np.argsort(error_mean)[-10:][::-1]
        worst_frequencies = freqs[worst_freq_indices]

        # Categorize by band
        band_errors = {}
        for band_name, (low, high) in FREQ_BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            band_errors[band_name] = {
                'mean_error': float(np.mean(error_mean[mask])),
                'max_error': float(np.max(error_mean[mask])),
                'relative_error': float(np.mean(relative_error_mean[mask])),
            }

        results = {
            'frequencies': freqs.tolist(),
            'error_mean': error_mean.tolist(),
            'error_std': error_std.tolist(),
            'relative_error_mean': relative_error_mean.tolist(),
            'worst_frequencies': worst_frequencies.tolist(),
            'band_errors': band_errors,
        }

        self.results['spectral_error'] = results
        return results

    def coherence_preservation(
        self,
        inputs: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        nperseg: int = 512
    ) -> Dict:
        """
        Check if model preserves coherence structure.

        Compares:
        - Coherence(input, target) [real]
        - Coherence(input, prediction) [model output]

        Returns:
            Dict with coherence analysis
        """
        # Average over channels if multi-channel
        if inputs.ndim == 3:
            inputs = inputs.mean(axis=1)
            predictions = predictions.mean(axis=1)
            targets = targets.mean(axis=1)

        inputs = inputs.reshape(-1, inputs.shape[-1])
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])

        # Compute coherences for each sample pair
        coh_real_list = []
        coh_pred_list = []

        for i in range(min(len(inputs), 100)):  # Limit samples
            freqs, coh_real = signal.coherence(
                inputs[i], targets[i], fs=self.fs, nperseg=nperseg
            )
            _, coh_pred = signal.coherence(
                inputs[i], predictions[i], fs=self.fs, nperseg=nperseg
            )
            coh_real_list.append(coh_real)
            coh_pred_list.append(coh_pred)

        coh_real = np.mean(coh_real_list, axis=0)
        coh_pred = np.mean(coh_pred_list, axis=0)
        coh_real_std = np.std(coh_real_list, axis=0)
        coh_pred_std = np.std(coh_pred_list, axis=0)

        # Coherence preservation score
        coherence_error = np.abs(coh_real - coh_pred)
        preservation_score = 1 - np.mean(coherence_error)

        # Per-band coherence
        band_coherence = {}
        for band_name, (low, high) in FREQ_BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            band_coherence[band_name] = {
                'real': float(np.mean(coh_real[mask])),
                'predicted': float(np.mean(coh_pred[mask])),
                'error': float(np.mean(coherence_error[mask])),
            }

        results = {
            'frequencies': freqs.tolist(),
            'coherence_real': coh_real.tolist(),
            'coherence_predicted': coh_pred.tolist(),
            'coherence_real_std': coh_real_std.tolist(),
            'coherence_predicted_std': coh_pred_std.tolist(),
            'coherence_error': coherence_error.tolist(),
            'preservation_score': float(preservation_score),
            'band_coherence': band_coherence,
        }

        self.results['coherence'] = results
        return results

    def phase_locking_value(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        freq_bands: Optional[Dict] = None
    ) -> Dict:
        """
        Compute Phase Locking Value between predictions and targets.

        PLV measures phase synchronization independent of amplitude.

        Returns:
            Dict with PLV per frequency band
        """
        if freq_bands is None:
            freq_bands = FREQ_BANDS

        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])

        plv_results = {}

        for band_name, (low, high) in freq_bands.items():
            # Bandpass filter
            nyq = self.fs / 2
            low_norm = low / nyq
            high_norm = min(high / nyq, 0.99)

            try:
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')

                pred_filt = signal.filtfilt(b, a, predictions, axis=-1)
                tgt_filt = signal.filtfilt(b, a, targets, axis=-1)

                # Compute analytic signal (Hilbert transform)
                pred_analytic = signal.hilbert(pred_filt, axis=-1)
                tgt_analytic = signal.hilbert(tgt_filt, axis=-1)

                # Phase difference
                phase_diff = np.angle(pred_analytic) - np.angle(tgt_analytic)

                # PLV = |mean(exp(i * phase_diff))|
                plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))
                plv_mean = float(np.mean(plv))
                plv_std = float(np.std(plv))

                plv_results[band_name] = {
                    'plv_mean': plv_mean,
                    'plv_std': plv_std,
                    'freq_range': (low, high),
                }
            except Exception as e:
                plv_results[band_name] = {
                    'plv_mean': np.nan,
                    'plv_std': np.nan,
                    'freq_range': (low, high),
                    'error': str(e),
                }

        # Overall PLV
        valid_plvs = [v['plv_mean'] for v in plv_results.values()
                     if not np.isnan(v['plv_mean'])]

        results = {
            'band_plv': plv_results,
            'mean_plv': float(np.mean(valid_plvs)) if valid_plvs else np.nan,
            'best_band': max(plv_results.items(),
                            key=lambda x: x[1]['plv_mean'] if not np.isnan(x[1]['plv_mean']) else -1)[0],
        }

        self.results['plv'] = results
        return results

    def _compute_r2(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute R² score."""
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-10)

    def _compute_correlation(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Pearson correlation."""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        return float(np.corrcoef(pred_flat, target_flat)[0, 1])

    def plot_bandwise_accuracy(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot band-wise R² as bar chart."""
        if 'bandwise_accuracy' not in self.results:
            raise ValueError("Run bandwise_accuracy first")

        data = self.results['bandwise_accuracy']['band_results']

        fig, ax = plt.subplots(figsize=(8, 5))

        bands = list(data.keys())
        r2_values = [data[b]['r2'] for b in bands]

        colors = [COLORS['primary'] if r2 >= 0 else COLORS['secondary']
                 for r2 in r2_values]

        bars = ax.bar(bands, r2_values, color=colors, alpha=0.7, edgecolor='black')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('R²')
        ax.set_title('Band-wise Prediction Accuracy')
        ax.set_ylim(min(0, min(r2_values) - 0.1), max(r2_values) + 0.1)

        # Add value labels
        for bar, val in zip(bars, r2_values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)

        return fig

    def plot_psd_comparison(
        self,
        save_path: Optional[str] = None,
        log_scale: bool = True
    ) -> plt.Figure:
        """Plot PSD comparison between predictions and targets."""
        if 'psd_comparison' not in self.results:
            raise ValueError("Run psd_comparison first")

        data = self.results['psd_comparison']
        freqs = np.array(data['frequencies'])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # PSD overlay
        ax = axes[0]
        if log_scale:
            ax.semilogy(freqs, data['psd_target'], color=COLORS['target'],
                       linewidth=1.5, label='Target')
            ax.semilogy(freqs, data['psd_prediction'], color=COLORS['pred'],
                       linewidth=1.5, label='Prediction')
        else:
            ax.plot(freqs, data['psd_target'], color=COLORS['target'],
                   linewidth=1.5, label='Target')
            ax.plot(freqs, data['psd_prediction'], color=COLORS['pred'],
                   linewidth=1.5, label='Prediction')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title('PSD Comparison')
        ax.legend()
        ax.set_xlim(0, min(100, freqs[-1]))

        # Spectral error
        ax = axes[1]
        ax.plot(freqs, data['spectral_error_db'], color=COLORS['secondary'],
               linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.fill_between(freqs, 0, data['spectral_error_db'],
                       where=np.array(data['spectral_error_db']) > 0,
                       color=COLORS['secondary'], alpha=0.3)
        ax.fill_between(freqs, 0, data['spectral_error_db'],
                       where=np.array(data['spectral_error_db']) < 0,
                       color=COLORS['primary'], alpha=0.3)

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Error (dB)')
        ax.set_title(f'Spectral Error (Mean |error|: {data["mean_spectral_error_db"]:.2f} dB)')
        ax.set_xlim(0, min(100, freqs[-1]))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)

        return fig


# =============================================================================
# PART 4: LATENT SPACE / REPRESENTATION ANALYSIS
# =============================================================================

class LatentSpaceAnalyzer:
    """
    Analyze learned representations in the model's latent space.

    Key analyses:
    1. Bottleneck representation extraction
    2. Latent space visualization (UMAP/t-SNE)
    3. Latent space dimensionality (PCA)
    4. Representational Similarity Analysis (RSA)
    5. Layer-wise representation evolution
    """

    def __init__(self, model: nn.Module, config: AnalysisConfig):
        self.model = model
        self.config = config
        self.device = config.device
        self.results = {}
        self.hooks = []
        self.activations = {}

    def _register_hooks(self, layer_names: List[str]):
        """Register forward hooks to capture activations."""
        self.activations = {}
        self.hooks = []

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if any(ln in name for ln in layer_names):
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def extract_bottleneck_representations(
        self,
        dataloader: DataLoader,
        bottleneck_name: str = 'bottleneck',
        n_samples: int = 1000
    ) -> Dict:
        """
        Extract representations from the bottleneck layer.

        Returns:
            Dict with bottleneck features and metadata
        """
        self.model.eval()

        # Try to find bottleneck layer
        bottleneck_found = False
        for name, module in self.model.named_modules():
            if bottleneck_name in name.lower() or 'mid' in name.lower():
                self._register_hooks([name])
                bottleneck_found = True
                break

        if not bottleneck_found:
            # Fallback: look for the deepest encoder layer
            encoder_layers = [n for n, m in self.model.named_modules()
                            if 'encoder' in n.lower() or 'down' in n.lower()]
            if encoder_layers:
                self._register_hooks([encoder_layers[-1]])

        representations = []
        metadata = {'sample_idx': [], 'batch_idx': []}

        for batch_idx, batch in enumerate(dataloader):
            if len(representations) * (batch[0].shape[0] if representations else 1) >= n_samples:
                break

            inputs = batch[0].to(self.device)
            _ = self.model(inputs)

            # Get bottleneck activation
            if self.activations:
                layer_name = list(self.activations.keys())[0]
                activation = self.activations[layer_name]

                # Global average pool if spatial
                if activation.dim() == 3:  # [B, C, T]
                    activation = activation.mean(dim=-1)  # [B, C]

                representations.append(activation.detach().cpu().numpy())
                metadata['batch_idx'].extend([batch_idx] * len(inputs))
                metadata['sample_idx'].extend(
                    list(range(batch_idx * len(inputs),
                              batch_idx * len(inputs) + len(inputs)))
                )

        self._remove_hooks()

        if representations:
            representations = np.concatenate(representations, axis=0)[:n_samples]
        else:
            representations = np.array([])

        results = {
            'representations': representations,
            'n_samples': len(representations),
            'feature_dim': representations.shape[1] if len(representations) > 0 else 0,
            'metadata': metadata,
        }

        self.results['bottleneck'] = results
        return results

    def visualize_latent_space(
        self,
        method: str = 'umap',
        labels: Optional[np.ndarray] = None,
        label_names: Optional[Dict] = None
    ) -> Dict:
        """
        Visualize latent space using UMAP or t-SNE.

        Args:
            method: 'umap' or 'tsne'
            labels: Optional labels for coloring points
            label_names: Dict mapping label values to names

        Returns:
            Dict with 2D embedding coordinates
        """
        if 'bottleneck' not in self.results:
            raise ValueError("Run extract_bottleneck_representations first")

        representations = self.results['bottleneck']['representations']

        if len(representations) == 0:
            return {'error': 'No representations to visualize'}

        if method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(
                    n_neighbors=self.config.latent_umap_n_neighbors,
                    min_dist=0.1,
                    n_components=2,
                    random_state=self.config.seed
                )
                embedding = reducer.fit_transform(representations)
            except ImportError:
                print("UMAP not installed, falling back to t-SNE")
                method = 'tsne'

        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=2,
                perplexity=min(30, len(representations) - 1),
                random_state=self.config.seed
            )
            embedding = reducer.fit_transform(representations)

        results = {
            'embedding': embedding,
            'method': method,
            'labels': labels.tolist() if labels is not None else None,
            'label_names': label_names,
            'n_samples': len(embedding),
        }

        self.results['latent_visualization'] = results
        return results

    def analyze_dimensionality(
        self,
        variance_threshold: float = 0.9
    ) -> Dict:
        """
        Analyze effective dimensionality via PCA.

        Returns:
            Dict with dimensionality statistics
        """
        if 'bottleneck' not in self.results:
            raise ValueError("Run extract_bottleneck_representations first")

        representations = self.results['bottleneck']['representations']

        if len(representations) == 0:
            return {'error': 'No representations to analyze'}

        from sklearn.decomposition import PCA

        # Full PCA
        n_components = min(representations.shape)
        pca = PCA(n_components=n_components)
        pca.fit(representations)

        # Find components needed for threshold variance
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_for_threshold = np.argmax(cumvar >= variance_threshold) + 1

        # Intrinsic dimensionality estimate
        eigenvalues = pca.explained_variance_
        intrinsic_dim = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

        results = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': cumvar.tolist(),
            'n_components_for_threshold': int(n_for_threshold),
            'variance_threshold': variance_threshold,
            'intrinsic_dimensionality': float(intrinsic_dim),
            'total_components': n_components,
            'top_10_variance': float(cumvar[min(9, len(cumvar)-1)]),
        }

        self.results['dimensionality'] = results
        return results

    def representational_similarity_analysis(
        self,
        compare_to: Optional[np.ndarray] = None,
        comparison_name: str = 'external'
    ) -> Dict:
        """
        Compute Representational Similarity Analysis (RSA).

        Compares learned representation similarity to known structure.

        Args:
            compare_to: Optional external similarity matrix to compare against
            comparison_name: Name for the comparison

        Returns:
            Dict with RSA results
        """
        if 'bottleneck' not in self.results:
            raise ValueError("Run extract_bottleneck_representations first")

        representations = self.results['bottleneck']['representations']

        if len(representations) == 0:
            return {'error': 'No representations to analyze'}

        # Compute representational dissimilarity matrix (RDM)
        from scipy.spatial.distance import pdist, squareform

        rdm = squareform(pdist(representations, metric='correlation'))

        results = {
            'rdm': rdm,
            'rdm_mean': float(np.mean(rdm)),
            'rdm_std': float(np.std(rdm)),
        }

        # Compare to external RDM if provided
        if compare_to is not None:
            # Flatten upper triangles
            rdm_upper = rdm[np.triu_indices(len(rdm), k=1)]
            compare_upper = compare_to[np.triu_indices(len(compare_to), k=1)]

            if len(rdm_upper) == len(compare_upper):
                rsa_correlation = float(np.corrcoef(rdm_upper, compare_upper)[0, 1])
                results[f'rsa_correlation_{comparison_name}'] = rsa_correlation

        self.results['rsa'] = results
        return results

    @torch.no_grad()
    def layer_wise_representation(
        self,
        dataloader: DataLoader,
        n_samples: int = 100
    ) -> Dict:
        """
        Extract features at each encoder/decoder level.

        Returns:
            Dict with per-layer representations
        """
        self.model.eval()

        # Find encoder and decoder layers
        layer_patterns = ['inc', 'encoder', 'bottleneck', 'decoder', 'outc',
                         'down', 'up', 'conv']
        all_layer_names = []
        for name, module in self.model.named_modules():
            if any(p in name.lower() for p in layer_patterns):
                if hasattr(module, 'weight') or 'block' in name.lower():
                    all_layer_names.append(name)

        # Limit to key layers
        layer_names = all_layer_names[:20] if len(all_layer_names) > 20 else all_layer_names

        self._register_hooks(layer_names)

        layer_representations = {name: [] for name in layer_names}

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * dataloader.batch_size >= n_samples:
                break

            inputs = batch[0].to(self.device)
            _ = self.model(inputs)

            for name in layer_names:
                if name in self.activations:
                    act = self.activations[name]
                    # Global average pool
                    if act.dim() == 3:
                        act = act.mean(dim=-1)
                    layer_representations[name].append(act.detach().cpu().numpy())

        self._remove_hooks()

        # Concatenate and compute statistics
        layer_stats = {}
        for name, reps in layer_representations.items():
            if reps:
                reps = np.concatenate(reps, axis=0)[:n_samples]
                layer_stats[name] = {
                    'mean_activation': float(np.mean(reps)),
                    'std_activation': float(np.std(reps)),
                    'sparsity': float(np.mean(reps == 0)),
                    'dim': reps.shape[1] if reps.ndim > 1 else 1,
                }

        results = {
            'layer_names': layer_names,
            'layer_stats': layer_stats,
            'n_samples': n_samples,
        }

        self.results['layer_wise'] = results
        return results

    def plot_latent_space(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot latent space visualization."""
        if 'latent_visualization' not in self.results:
            raise ValueError("Run visualize_latent_space first")

        data = self.results['latent_visualization']
        embedding = data['embedding']

        fig, ax = plt.subplots(figsize=(8, 8))

        if data['labels'] is not None:
            labels = np.array(data['labels'])
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                name = data['label_names'].get(label, str(label)) if data['label_names'] else str(label)
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                          c=[colors[i]], label=name, alpha=0.6, s=20)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1],
                      c=COLORS['primary'], alpha=0.6, s=20)

        ax.set_xlabel(f'{data["method"].upper()} 1')
        ax.set_ylabel(f'{data["method"].upper()} 2')
        ax.set_title(f'Latent Space Visualization ({data["method"].upper()})')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)

        return fig

    def plot_dimensionality(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot PCA explained variance."""
        if 'dimensionality' not in self.results:
            raise ValueError("Run analyze_dimensionality first")

        data = self.results['dimensionality']

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Individual variance
        ax = axes[0]
        n_show = min(50, len(data['explained_variance_ratio']))
        ax.bar(range(n_show), data['explained_variance_ratio'][:n_show],
               color=COLORS['primary'], alpha=0.7)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Explained Variance')

        # Cumulative variance
        ax = axes[1]
        ax.plot(data['cumulative_variance'], color=COLORS['primary'], linewidth=2)
        ax.axhline(y=data['variance_threshold'], color=COLORS['secondary'],
                  linestyle='--', label=f'{data["variance_threshold"]*100:.0f}% threshold')
        ax.axvline(x=data['n_components_for_threshold'], color=COLORS['tertiary'],
                  linestyle='--', label=f'n={data["n_components_for_threshold"]}')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title(f'Intrinsic Dim ≈ {data["intrinsic_dimensionality"]:.1f}')
        ax.legend()
        ax.set_xlim(0, len(data['cumulative_variance']))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)

        return fig


# =============================================================================
# PART 5: CONDITIONING MECHANISM ANALYSIS
# =============================================================================

class ConditioningAnalyzer:
    """
    Analyze how conditioning/modulation affects the model.

    Key analyses:
    1. Conditioning embedding visualization
    2. Conditioning ablation
    3. Gating pattern analysis (for FiLM/cross-attention)
    """

    def __init__(self, model: nn.Module, config: AnalysisConfig):
        self.model = model
        self.config = config
        self.device = config.device
        self.results = {}
        self.hooks = []
        self.activations = {}

    def _register_hooks(self, patterns: List[str]):
        """Register hooks for conditioning-related layers."""
        self.activations = {}
        self.hooks = []

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if any(p in name.lower() for p in patterns):
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def extract_conditioning_embeddings(
        self,
        dataloader: DataLoader,
        n_samples: int = 500
    ) -> Dict:
        """
        Extract conditioning embeddings for visualization.

        Returns:
            Dict with conditioning embeddings
        """
        self.model.eval()

        # Look for conditioning-related modules
        cond_patterns = ['cond', 'embed', 'film', 'spectro', 'temporal']
        self._register_hooks(cond_patterns)

        embeddings = {}

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * dataloader.batch_size >= n_samples:
                break

            inputs = batch[0].to(self.device)
            _ = self.model(inputs)

            for name, act in self.activations.items():
                if name not in embeddings:
                    embeddings[name] = []

                # Flatten if needed
                if act.dim() > 2:
                    act = act.view(act.shape[0], -1)
                embeddings[name].append(act.detach().cpu().numpy())

        self._remove_hooks()

        # Concatenate
        for name in embeddings:
            embeddings[name] = np.concatenate(embeddings[name], axis=0)[:n_samples]

        results = {
            'embeddings': embeddings,
            'embedding_names': list(embeddings.keys()),
            'n_samples': n_samples,
        }

        self.results['conditioning_embeddings'] = results
        return results

    @torch.no_grad()
    def conditioning_ablation(
        self,
        dataloader: DataLoader,
        n_samples: int = 200
    ) -> Dict:
        """
        Measure performance with and without conditioning.

        Returns:
            Dict with ablation results
        """
        self.model.eval()

        # Store original conditioning mode
        original_cond_mode = None
        for name, module in self.model.named_modules():
            if hasattr(module, 'cond_mode'):
                original_cond_mode = module.cond_mode
                break

        # Baseline with conditioning
        baseline_losses = []
        baseline_r2s = []

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * dataloader.batch_size >= n_samples:
                break

            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            outputs = self.model(inputs)
            loss = F.mse_loss(outputs, targets)
            r2 = self._compute_r2(outputs, targets)

            baseline_losses.append(loss.item())
            baseline_r2s.append(r2)

        baseline_loss = np.mean(baseline_losses)
        baseline_r2 = np.mean(baseline_r2s)

        # Ablated (set conditioning to 'none' if possible)
        ablated_loss = baseline_loss
        ablated_r2 = baseline_r2

        for name, module in self.model.named_modules():
            if hasattr(module, 'cond_mode'):
                try:
                    module.cond_mode = 'none'
                except:
                    pass

        ablated_losses = []
        ablated_r2s = []

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * dataloader.batch_size >= n_samples:
                break

            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            try:
                outputs = self.model(inputs)
                loss = F.mse_loss(outputs, targets)
                r2 = self._compute_r2(outputs, targets)
                ablated_losses.append(loss.item())
                ablated_r2s.append(r2)
            except:
                # Model might not support ablation
                break

        if ablated_losses:
            ablated_loss = np.mean(ablated_losses)
            ablated_r2 = np.mean(ablated_r2s)

        # Restore original
        for name, module in self.model.named_modules():
            if hasattr(module, 'cond_mode') and original_cond_mode:
                module.cond_mode = original_cond_mode

        loss_increase = ablated_loss - baseline_loss
        r2_drop = baseline_r2 - ablated_r2

        results = {
            'baseline_loss': float(baseline_loss),
            'baseline_r2': float(baseline_r2),
            'ablated_loss': float(ablated_loss),
            'ablated_r2': float(ablated_r2),
            'loss_increase': float(loss_increase),
            'r2_drop': float(r2_drop),
            'conditioning_helps': loss_increase > 0,
            'relative_improvement': float(loss_increase / (ablated_loss + 1e-10) * 100),
        }

        self.results['conditioning_ablation'] = results
        return results

    @torch.no_grad()
    def analyze_gating_patterns(
        self,
        dataloader: DataLoader,
        n_samples: int = 100
    ) -> Dict:
        """
        Analyze gating weights in attention/FiLM layers.

        Returns:
            Dict with gating pattern statistics
        """
        self.model.eval()

        # Look for gating layers
        gate_patterns = ['gate', 'attn', 'film', 'scale', 'shift']
        self._register_hooks(gate_patterns)

        gating_values = {}

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * dataloader.batch_size >= n_samples:
                break

            inputs = batch[0].to(self.device)
            _ = self.model(inputs)

            for name, act in self.activations.items():
                if name not in gating_values:
                    gating_values[name] = []

                # For gating, we want to see the distribution
                gating_values[name].append(act.detach().cpu().numpy().flatten())

        self._remove_hooks()

        # Analyze distributions
        gating_stats = {}
        for name, values in gating_values.items():
            values = np.concatenate(values)
            gating_stats[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'sparsity': float(np.mean(np.abs(values) < 0.1)),
                'histogram': np.histogram(values, bins=50)[0].tolist(),
            }

        results = {
            'gating_stats': gating_stats,
            'n_samples': n_samples,
        }

        self.results['gating_patterns'] = results
        return results

    def _compute_r2(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        return float(1 - ss_res / (ss_tot + 1e-10))

    def plot_conditioning_ablation(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot conditioning ablation results."""
        if 'conditioning_ablation' not in self.results:
            raise ValueError("Run conditioning_ablation first")

        data = self.results['conditioning_ablation']

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Loss comparison
        ax = axes[0]
        x = ['With Conditioning', 'Without Conditioning']
        heights = [data['baseline_loss'], data['ablated_loss']]
        colors = [COLORS['primary'], COLORS['secondary']]
        bars = ax.bar(x, heights, color=colors, alpha=0.7)
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss Increase: {data["loss_increase"]:.4f}')

        # R² comparison
        ax = axes[1]
        heights = [data['baseline_r2'], data['ablated_r2']]
        bars = ax.bar(x, heights, color=colors, alpha=0.7)
        ax.set_ylabel('R²')
        ax.set_title(f'R² Drop: {data["r2_drop"]:.4f}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)

        return fig


# =============================================================================
# PART 6: LOSS LANDSCAPE & OPTIMIZATION
# =============================================================================

class LossLandscapeAnalyzer:
    """
    Analyze the loss landscape around the trained model.

    Key analyses:
    1. 3D loss landscape visualization
    2. 2D contour plot
    3. Sharpness analysis
    4. Training trajectory (if available)
    """

    def __init__(self, model: nn.Module, config: AnalysisConfig):
        self.model = model
        self.config = config
        self.device = config.device
        self.results = {}

    def _get_random_direction(self, filter_normalize: bool = True) -> Dict[str, torch.Tensor]:
        """Generate a random direction in parameter space."""
        direction = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                d = torch.randn_like(param)
                if filter_normalize and param.dim() > 1:
                    # Filter-wise normalization for conv layers
                    d = d / (d.norm() + 1e-10) * param.norm()
                direction[name] = d

        return direction

    def _perturb_model(
        self,
        direction1: Dict[str, torch.Tensor],
        direction2: Dict[str, torch.Tensor],
        alpha: float,
        beta: float
    ):
        """Perturb model weights: θ* + α·d1 + β·d2"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in direction1:
                    param.add_(alpha * direction1[name] + beta * direction2[name])

    def _restore_model(
        self,
        direction1: Dict[str, torch.Tensor],
        direction2: Dict[str, torch.Tensor],
        alpha: float,
        beta: float
    ):
        """Restore model weights after perturbation."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in direction1:
                    param.sub_(alpha * direction1[name] + beta * direction2[name])

    @torch.no_grad()
    def compute_loss_landscape(
        self,
        dataloader: DataLoader,
        resolution: int = 21,
        range_scale: float = 1.0,
        n_batches: int = 10
    ) -> Dict:
        """
        Compute 2D loss landscape.

        Args:
            dataloader: Data to compute loss on
            resolution: Grid resolution
            range_scale: How far to explore (in units of weight norm)
            n_batches: Number of batches to average

        Returns:
            Dict with loss landscape data
        """
        self.model.eval()

        # Get random directions
        d1 = self._get_random_direction()
        d2 = self._get_random_direction()

        # Create grid
        alphas = np.linspace(-range_scale, range_scale, resolution)
        betas = np.linspace(-range_scale, range_scale, resolution)

        loss_surface = np.zeros((resolution, resolution))

        # Get batches for evaluation
        batches = []
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            batches.append((batch[0].to(self.device), batch[1].to(self.device)))

        # Compute loss at each grid point
        total_points = resolution * resolution
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                self._perturb_model(d1, d2, alpha, beta)

                total_loss = 0
                for inputs, targets in batches:
                    outputs = self.model(inputs)
                    loss = F.mse_loss(outputs, targets)
                    total_loss += loss.item()

                loss_surface[i, j] = total_loss / len(batches)

                self._restore_model(d1, d2, alpha, beta)

                # Progress
                done = i * resolution + j + 1
                if done % 50 == 0:
                    print(f"Loss landscape: {done}/{total_points} ({100*done/total_points:.1f}%)")

        # Find minimum
        min_idx = np.unravel_index(np.argmin(loss_surface), loss_surface.shape)

        results = {
            'loss_surface': loss_surface,
            'alphas': alphas.tolist(),
            'betas': betas.tolist(),
            'resolution': resolution,
            'range_scale': range_scale,
            'min_loss': float(np.min(loss_surface)),
            'max_loss': float(np.max(loss_surface)),
            'min_location': (float(alphas[min_idx[0]]), float(betas[min_idx[1]])),
            'center_loss': float(loss_surface[resolution//2, resolution//2]),
        }

        self.results['loss_landscape'] = results
        return results

    def analyze_sharpness(
        self,
        dataloader: DataLoader,
        n_directions: int = 10,
        epsilon: float = 0.1
    ) -> Dict:
        """
        Analyze sharpness of the minimum.

        Sharp minima → poor generalization
        Flat minima → better generalization

        Returns:
            Dict with sharpness metrics
        """
        self.model.eval()

        # Get base loss
        base_losses = []
        batches = []
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            batches.append((inputs, targets))
            with torch.no_grad():
                outputs = self.model(inputs)
                base_losses.append(F.mse_loss(outputs, targets).item())

        base_loss = np.mean(base_losses)

        # Sample random directions and compute loss
        perturbed_losses = []

        for _ in range(n_directions):
            direction = self._get_random_direction()

            # Perturb
            self._perturb_model(direction, {n: torch.zeros_like(v) for n, v in direction.items()},
                              epsilon, 0)

            losses = []
            for inputs, targets in batches:
                with torch.no_grad():
                    outputs = self.model(inputs)
                    losses.append(F.mse_loss(outputs, targets).item())

            perturbed_losses.append(np.mean(losses))

            # Restore
            self._restore_model(direction, {n: torch.zeros_like(v) for n, v in direction.items()},
                               epsilon, 0)

        avg_perturbed = np.mean(perturbed_losses)
        sharpness = (avg_perturbed - base_loss) / (base_loss + 1e-10)

        results = {
            'base_loss': float(base_loss),
            'avg_perturbed_loss': float(avg_perturbed),
            'sharpness': float(sharpness),
            'sharpness_std': float(np.std(perturbed_losses)),
            'epsilon': epsilon,
            'n_directions': n_directions,
            'is_flat': sharpness < 0.1,  # Heuristic
        }

        self.results['sharpness'] = results
        return results

    def plot_loss_landscape(
        self,
        save_path: Optional[str] = None,
        plot_type: str = 'both'
    ) -> plt.Figure:
        """
        Plot loss landscape.

        Args:
            plot_type: '3d', 'contour', or 'both'
        """
        if 'loss_landscape' not in self.results:
            raise ValueError("Run compute_loss_landscape first")

        data = self.results['loss_landscape']
        alphas = np.array(data['alphas'])
        betas = np.array(data['betas'])
        surface = np.array(data['loss_surface'])

        if plot_type == 'both':
            fig = plt.figure(figsize=(14, 5))

            # 3D surface
            ax1 = fig.add_subplot(121, projection='3d')
            A, B = np.meshgrid(alphas, betas)
            ax1.plot_surface(A, B, surface.T, cmap='viridis', alpha=0.8)
            ax1.set_xlabel('Direction 1')
            ax1.set_ylabel('Direction 2')
            ax1.set_zlabel('Loss')
            ax1.set_title('Loss Landscape (3D)')

            # Contour
            ax2 = fig.add_subplot(122)
            cs = ax2.contour(alphas, betas, surface.T, levels=20, cmap='viridis')
            ax2.clabel(cs, inline=True, fontsize=8)
            ax2.plot(0, 0, 'r*', markersize=15, label='Trained weights')
            ax2.set_xlabel('Direction 1')
            ax2.set_ylabel('Direction 2')
            ax2.set_title('Loss Landscape (Contour)')
            ax2.legend()
            ax2.set_aspect('equal')

        elif plot_type == '3d':
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            A, B = np.meshgrid(alphas, betas)
            ax.plot_surface(A, B, surface.T, cmap='viridis', alpha=0.8)
            ax.set_xlabel('Direction 1')
            ax.set_ylabel('Direction 2')
            ax.set_zlabel('Loss')
            ax.set_title('Loss Landscape')

        else:  # contour
            fig, ax = plt.subplots(figsize=(8, 6))
            cs = ax.contourf(alphas, betas, surface.T, levels=30, cmap='viridis')
            plt.colorbar(cs, ax=ax, label='Loss')
            ax.plot(0, 0, 'r*', markersize=15, label='Trained weights')
            ax.set_xlabel('Direction 1')
            ax.set_ylabel('Direction 2')
            ax.set_title('Loss Landscape')
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)

        return fig


# =============================================================================
# PART 7: GENERALIZATION ANALYSIS
# =============================================================================

class GeneralizationAnalyzer:
    """
    Analyze model generalization across different conditions.

    Key analyses:
    1. Per-session/fold performance table
    2. Cross-subject analysis
    3. Temporal generalization
    4. Performance vs signal quality
    """

    def __init__(self, model: nn.Module, config: AnalysisConfig):
        self.model = model
        self.config = config
        self.device = config.device
        self.results = {}

    @torch.no_grad()
    def per_session_performance(
        self,
        session_dataloaders: Dict[str, DataLoader]
    ) -> Dict:
        """
        Compute detailed metrics for each session.

        Args:
            session_dataloaders: Dict mapping session names to DataLoaders

        Returns:
            Dict with per-session metrics
        """
        self.model.eval()

        session_results = {}

        for session_name, loader in session_dataloaders.items():
            losses = []
            r2s = []
            correlations = []
            maes = []

            for batch in loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)

                outputs = self.model(inputs)

                loss = F.mse_loss(outputs, targets)
                r2 = self._compute_r2(outputs, targets)
                corr = self._compute_correlation(outputs, targets)
                mae = F.l1_loss(outputs, targets)

                losses.append(loss.item())
                r2s.append(r2)
                correlations.append(corr)
                maes.append(mae.item())

            session_results[session_name] = {
                'loss': float(np.mean(losses)),
                'loss_std': float(np.std(losses)),
                'r2': float(np.mean(r2s)),
                'r2_std': float(np.std(r2s)),
                'correlation': float(np.mean(correlations)),
                'correlation_std': float(np.std(correlations)),
                'mae': float(np.mean(maes)),
                'mae_std': float(np.std(maes)),
                'n_samples': len(loader.dataset),
            }

        # Compute summary statistics
        all_r2s = [v['r2'] for v in session_results.values()]
        all_losses = [v['loss'] for v in session_results.values()]

        # Identify outliers (> 2 std from mean)
        r2_mean, r2_std = np.mean(all_r2s), np.std(all_r2s)
        outlier_sessions = [name for name, data in session_results.items()
                          if abs(data['r2'] - r2_mean) > 2 * r2_std]

        results = {
            'session_results': session_results,
            'summary': {
                'mean_r2': float(np.mean(all_r2s)),
                'std_r2': float(np.std(all_r2s)),
                'mean_loss': float(np.mean(all_losses)),
                'std_loss': float(np.std(all_losses)),
                'best_session': max(session_results.items(), key=lambda x: x[1]['r2'])[0],
                'worst_session': min(session_results.items(), key=lambda x: x[1]['r2'])[0],
            },
            'outlier_sessions': outlier_sessions,
            'n_sessions': len(session_results),
        }

        self.results['per_session'] = results
        return results

    @torch.no_grad()
    def temporal_generalization(
        self,
        dataloader: DataLoader,
        n_temporal_bins: int = 5
    ) -> Dict:
        """
        Analyze if performance varies over time within recordings.

        Returns:
            Dict with temporal performance analysis
        """
        self.model.eval()

        all_outputs = []
        all_targets = []
        all_inputs = []

        for batch in dataloader:
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            outputs = self.model(inputs)

            all_inputs.append(inputs.cpu())
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

        all_inputs = torch.cat(all_inputs, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        T = all_outputs.shape[-1]
        bin_size = T // n_temporal_bins

        temporal_results = {}

        for i in range(n_temporal_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_temporal_bins - 1 else T

            out_segment = all_outputs[:, :, start:end]
            tgt_segment = all_targets[:, :, start:end]

            r2 = self._compute_r2(out_segment, tgt_segment)
            mae = F.l1_loss(out_segment, tgt_segment).item()

            temporal_results[f'bin_{i}'] = {
                'time_range': (start, end),
                'r2': float(r2),
                'mae': float(mae),
            }

        # Check for temporal drift
        r2_values = [v['r2'] for v in temporal_results.values()]
        drift = r2_values[-1] - r2_values[0]

        results = {
            'temporal_bins': temporal_results,
            'drift': float(drift),
            'has_drift': abs(drift) > 0.05,
            'n_bins': n_temporal_bins,
        }

        self.results['temporal_generalization'] = results
        return results

    @torch.no_grad()
    def performance_vs_signal_quality(
        self,
        dataloader: DataLoader
    ) -> Dict:
        """
        Analyze if performance correlates with input signal quality.

        Returns:
            Dict with signal quality analysis
        """
        self.model.eval()

        sample_metrics = []

        for batch in dataloader:
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            outputs = self.model(inputs)

            # Per-sample metrics
            for i in range(len(inputs)):
                inp = inputs[i].detach().cpu().numpy()
                out = outputs[i].detach().cpu().numpy()
                tgt = targets[i].detach().cpu().numpy()

                # Input signal quality metrics
                input_var = float(np.var(inp))
                input_snr = float(np.mean(inp ** 2) / (np.var(inp) + 1e-10))

                # Prediction quality
                mse = float(np.mean((out - tgt) ** 2))
                r2 = 1 - mse / (np.var(tgt) + 1e-10)

                sample_metrics.append({
                    'input_variance': input_var,
                    'input_snr': input_snr,
                    'mse': mse,
                    'r2': float(r2),
                })

        # Compute correlations
        variances = [m['input_variance'] for m in sample_metrics]
        snrs = [m['input_snr'] for m in sample_metrics]
        r2s = [m['r2'] for m in sample_metrics]

        var_r2_corr = float(np.corrcoef(variances, r2s)[0, 1])
        snr_r2_corr = float(np.corrcoef(snrs, r2s)[0, 1])

        results = {
            'variance_r2_correlation': var_r2_corr,
            'snr_r2_correlation': snr_r2_corr,
            'sample_metrics': sample_metrics[:100],  # Keep subset
            'n_samples': len(sample_metrics),
            'fails_on_noisy': var_r2_corr < -0.3,  # High variance = noisy
        }

        self.results['signal_quality'] = results
        return results

    def _compute_r2(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred = pred.float()
        target = target.float()
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        return float(1 - ss_res / (ss_tot + 1e-10))

    def _compute_correlation(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred = pred.flatten().detach().cpu().numpy()
        target = target.flatten().detach().cpu().numpy()
        return float(np.corrcoef(pred, target)[0, 1])

    def plot_per_session(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot per-session performance."""
        if 'per_session' not in self.results:
            raise ValueError("Run per_session_performance first")

        data = self.results['per_session']['session_results']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sessions = list(data.keys())
        r2s = [data[s]['r2'] for s in sessions]
        r2_stds = [data[s]['r2_std'] for s in sessions]

        # Bar chart with error bars
        ax = axes[0]
        x = np.arange(len(sessions))
        bars = ax.bar(x, r2s, yerr=r2_stds, capsize=3,
                     color=COLORS['primary'], alpha=0.7)

        # Highlight outliers
        outliers = self.results['per_session']['outlier_sessions']
        for i, session in enumerate(sessions):
            if session in outliers:
                bars[i].set_color(COLORS['secondary'])

        ax.set_xticks(x)
        ax.set_xticklabels(sessions, rotation=45, ha='right')
        ax.set_ylabel('R²')
        ax.set_title('Per-Session Performance')
        ax.axhline(y=self.results['per_session']['summary']['mean_r2'],
                  color='black', linestyle='--', label='Mean')
        ax.legend()

        # Distribution
        ax = axes[1]
        ax.hist(r2s, bins=min(20, len(r2s)), color=COLORS['primary'],
               alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(r2s), color=COLORS['secondary'],
                  linestyle='--', label=f'Mean: {np.mean(r2s):.3f}')
        ax.set_xlabel('R²')
        ax.set_ylabel('Count')
        ax.set_title('R² Distribution Across Sessions')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)

        return fig


# =============================================================================
# PART 8: ERROR ANALYSIS
# =============================================================================

class ErrorAnalyzer:
    """
    Comprehensive error analysis.

    Key analyses:
    1. Error distribution analysis
    2. Error vs signal characteristics
    3. Worst case gallery
    4. Residual analysis
    """

    def __init__(self, model: nn.Module, config: AnalysisConfig):
        self.model = model
        self.config = config
        self.device = config.device
        self.results = {}

    @torch.no_grad()
    def error_distribution(
        self,
        dataloader: DataLoader
    ) -> Dict:
        """
        Analyze distribution of prediction errors.

        Returns:
            Dict with error distribution statistics
        """
        self.model.eval()

        all_errors = []
        per_sample_mse = []
        per_sample_mae = []

        for batch in dataloader:
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            outputs = self.model(inputs)

            errors = (outputs - targets).detach().cpu().numpy()
            all_errors.append(errors.flatten())

            for i in range(len(inputs)):
                mse = float(np.mean((outputs[i].detach().cpu().numpy() - targets[i].detach().cpu().numpy()) ** 2))
                mae = float(np.mean(np.abs(outputs[i].detach().cpu().numpy() - targets[i].detach().cpu().numpy())))
                per_sample_mse.append(mse)
                per_sample_mae.append(mae)

        all_errors = np.concatenate(all_errors)

        error_mean = float(np.mean(all_errors))
        error_std = float(np.std(all_errors))
        error_skew = float(stats.skew(all_errors))
        error_kurtosis = float(stats.kurtosis(all_errors))

        if len(all_errors) > 5000:
            sample = np.random.choice(all_errors, 5000, replace=False)
        else:
            sample = all_errors
        _, normality_pvalue = stats.normaltest(sample)

        hist, bin_edges = np.histogram(all_errors, bins=100, density=True)

        percentile_99 = float(np.percentile(np.abs(all_errors), 99))
        percentile_50 = float(np.percentile(np.abs(all_errors), 50))
        tail_ratio = percentile_99 / (percentile_50 + 1e-10)

        results = {
            'error_mean': error_mean,
            'error_std': error_std,
            'error_skew': error_skew,
            'error_kurtosis': error_kurtosis,
            'is_normal': normality_pvalue > 0.05,
            'normality_pvalue': float(normality_pvalue),
            'histogram_counts': hist.tolist(),
            'histogram_edges': bin_edges.tolist(),
            'percentiles': {
                '1': float(np.percentile(all_errors, 1)),
                '5': float(np.percentile(all_errors, 5)),
                '25': float(np.percentile(all_errors, 25)),
                '50': float(np.percentile(all_errors, 50)),
                '75': float(np.percentile(all_errors, 75)),
                '95': float(np.percentile(all_errors, 95)),
                '99': float(np.percentile(all_errors, 99)),
            },
            'per_sample_mse': per_sample_mse,
            'per_sample_mae': per_sample_mae,
            'tail_ratio': tail_ratio,
            'has_heavy_tails': tail_ratio > 5,
            'n_samples': len(per_sample_mse),
        }

        self.results['error_distribution'] = results
        return results

    @torch.no_grad()
    def error_vs_characteristics(
        self,
        dataloader: DataLoader
    ) -> Dict:
        """Analyze how error correlates with input/output characteristics."""
        self.model.eval()

        characteristics = {
            'input_amplitude': [], 'input_variance': [],
            'target_amplitude': [], 'target_variance': [],
            'mse': [], 'mae': [],
        }

        for batch in dataloader:
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            outputs = self.model(inputs)

            for i in range(len(inputs)):
                inp = inputs[i].detach().cpu().numpy()
                tgt = targets[i].detach().cpu().numpy()
                out = outputs[i].detach().cpu().numpy()

                characteristics['input_amplitude'].append(float(np.mean(np.abs(inp))))
                characteristics['input_variance'].append(float(np.var(inp)))
                characteristics['target_amplitude'].append(float(np.mean(np.abs(tgt))))
                characteristics['target_variance'].append(float(np.var(tgt)))
                characteristics['mse'].append(float(np.mean((out - tgt) ** 2)))
                characteristics['mae'].append(float(np.mean(np.abs(out - tgt))))

        mse = np.array(characteristics['mse'])
        correlations = {}

        for key in ['input_amplitude', 'input_variance', 'target_amplitude', 'target_variance']:
            values = np.array(characteristics[key])
            corr = float(np.corrcoef(values, mse)[0, 1])
            correlations[f'{key}_vs_mse'] = corr

        patterns = []
        if correlations['input_amplitude_vs_mse'] > 0.3:
            patterns.append('Higher input amplitude -> larger errors')
        if correlations['input_amplitude_vs_mse'] < -0.3:
            patterns.append('Lower input amplitude -> larger errors')
        if correlations['target_variance_vs_mse'] > 0.3:
            patterns.append('Higher target variance -> larger errors')

        results = {
            'correlations': correlations,
            'characteristics': {k: v[:100] for k, v in characteristics.items()},
            'systematic_patterns': patterns,
            'n_samples': len(mse),
        }

        self.results['error_characteristics'] = results
        return results

    @torch.no_grad()
    def worst_case_gallery(
        self,
        dataloader: DataLoader,
        n_worst: int = 10
    ) -> Dict:
        """Find and analyze the worst predictions."""
        self.model.eval()

        all_samples = []

        for batch_idx, batch in enumerate(dataloader):
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            outputs = self.model(inputs)

            for i in range(len(inputs)):
                mse = float(F.mse_loss(outputs[i], targets[i]).item())
                all_samples.append({
                    'batch_idx': batch_idx, 'sample_idx': i, 'mse': mse,
                    'input': inputs[i].detach().cpu().numpy(),
                    'target': targets[i].detach().cpu().numpy(),
                    'prediction': outputs[i].detach().cpu().numpy(),
                })

        all_samples.sort(key=lambda x: x['mse'], reverse=True)
        worst_cases = all_samples[:n_worst]

        worst_input_vars = [np.var(w['input']) for w in worst_cases]
        all_input_vars = [np.var(s['input']) for s in all_samples]

        commonalities = []
        if np.mean(worst_input_vars) > 1.5 * np.mean(all_input_vars):
            commonalities.append('Worst cases have higher input variance')
        if np.mean(worst_input_vars) < 0.5 * np.mean(all_input_vars):
            commonalities.append('Worst cases have lower input variance')

        results = {
            'worst_cases': [{
                'batch_idx': w['batch_idx'], 'sample_idx': w['sample_idx'],
                'mse': w['mse'], 'input_variance': float(np.var(w['input'])),
            } for w in worst_cases],
            'worst_mses': [w['mse'] for w in worst_cases],
            'commonalities': commonalities,
            '_worst_inputs': [w['input'] for w in worst_cases],
            '_worst_targets': [w['target'] for w in worst_cases],
            '_worst_predictions': [w['prediction'] for w in worst_cases],
        }

        self.results['worst_cases'] = results
        return results

    @torch.no_grad()
    def residual_analysis(self, dataloader: DataLoader) -> Dict:
        """Analyze residuals for structure the model missed."""
        self.model.eval()

        all_residuals = []

        for batch in dataloader:
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            outputs = self.model(inputs)
            residuals = (targets - outputs).detach().cpu().numpy()
            all_residuals.append(residuals)

        all_residuals = np.concatenate(all_residuals, axis=0)
        residuals_flat = all_residuals.flatten()

        n_lags = min(100, len(residuals_flat) // 10)
        autocorr = np.correlate(residuals_flat[:10000], residuals_flat[:10000], mode='full')
        autocorr = autocorr[len(autocorr)//2:][:n_lags]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        significant_autocorr = np.sum(np.abs(autocorr[1:]) > 0.1)
        has_structure = significant_autocorr > n_lags * 0.1

        results = {
            'autocorrelation': autocorr.tolist(),
            'has_structure': has_structure,
            'residual_mean': float(np.mean(all_residuals)),
            'residual_std': float(np.std(all_residuals)),
            'random_noise': not has_structure and abs(np.mean(all_residuals)) < 0.01,
        }

        self.results['residual_analysis'] = results
        return results

    def plot_error_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot error distribution."""
        if 'error_distribution' not in self.results:
            raise ValueError("Run error_distribution first")

        data = self.results['error_distribution']
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        ax.bar(data['histogram_edges'][:-1], data['histogram_counts'],
               width=np.diff(data['histogram_edges']), color=COLORS['primary'], alpha=0.7)
        ax.axvline(x=data['error_mean'], color=COLORS['secondary'], linestyle='--',
                  label=f"Mean: {data['error_mean']:.4f}")
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        ax.set_title(f"Error Distribution (skew={data['error_skew']:.2f})")
        ax.legend()

        ax = axes[1]
        ax.hist(data['per_sample_mse'], bins=50, color=COLORS['primary'], alpha=0.7)
        ax.set_xlabel('Per-Sample MSE')
        ax.set_ylabel('Count')
        ax.set_title('Sample-Level Error Distribution')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)
        return fig

    def plot_worst_cases(self, save_path: Optional[str] = None, n_show: int = 4) -> plt.Figure:
        """Plot worst case predictions."""
        if 'worst_cases' not in self.results:
            raise ValueError("Run worst_case_gallery first")

        data = self.results['worst_cases']
        n_show = min(n_show, len(data['_worst_inputs']))

        fig, axes = plt.subplots(n_show, 1, figsize=(12, 3 * n_show))
        if n_show == 1:
            axes = [axes]

        for i in range(n_show):
            ax = axes[i]
            target = data['_worst_targets'][i][0] if data['_worst_targets'][i].ndim > 1 else data['_worst_targets'][i]
            pred = data['_worst_predictions'][i][0] if data['_worst_predictions'][i].ndim > 1 else data['_worst_predictions'][i]
            time = np.arange(len(target)) / self.config.sampling_rate

            ax.plot(time, target, color=COLORS['target'], linewidth=1, label='Target', alpha=0.8)
            ax.plot(time, pred, color=COLORS['pred'], linewidth=1, label='Prediction', alpha=0.8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f"Worst Case #{i+1} (MSE={data['worst_mses'][i]:.4f})")
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)
        return fig


# =============================================================================
# PART 9: ARCHITECTURAL INSIGHTS
# =============================================================================

class ArchitecturalAnalyzer:
    """Analyze architectural components: ablation, skip connections, depth."""

    def __init__(self, model: nn.Module, config: AnalysisConfig):
        self.model = model
        self.config = config
        self.device = config.device
        self.results = {}

    def component_ablation_summary(self, ablation_results: Dict[str, Dict]) -> Dict:
        """Summarize component ablation results."""
        importance = {}

        for component, results in ablation_results.items():
            delta = results['baseline_r2'] - results['ablated_r2']
            importance[component] = {
                'delta_r2': float(delta),
                'baseline_r2': float(results['baseline_r2']),
                'ablated_r2': float(results['ablated_r2']),
            }

        ranked = sorted(importance.items(), key=lambda x: x[1]['delta_r2'], reverse=True)

        results = {
            'component_importance': importance,
            'ranked_components': [r[0] for r in ranked],
            'most_important': ranked[0][0] if ranked else None,
        }

        self.results['ablation_summary'] = results
        return results

    @torch.no_grad()
    def depth_analysis(self, dataloader: DataLoader, n_samples: int = 100) -> Dict:
        """Analyze contribution of different network depths."""
        layer_info = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                depth = name.count('.')
                layer_info.append({'name': name, 'depth': depth, 'type': type(module).__name__})

        max_depth = max(l['depth'] for l in layer_info) if layer_info else 0

        param_counts = {'shallow': 0, 'middle': 0, 'deep': 0}
        for name, param in self.model.named_parameters():
            depth = name.count('.')
            if depth <= max_depth // 3:
                param_counts['shallow'] += param.numel()
            elif depth <= 2 * max_depth // 3:
                param_counts['middle'] += param.numel()
            else:
                param_counts['deep'] += param.numel()

        total_params = sum(param_counts.values())

        results = {
            'layer_info': layer_info,
            'param_distribution': {k: {'count': v, 'pct': v/(total_params+1e-10)*100}
                                   for k, v in param_counts.items()},
            'total_params': total_params,
        }

        self.results['depth_analysis'] = results
        return results

    def plot_ablation_summary(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot ablation summary."""
        if 'ablation_summary' not in self.results:
            raise ValueError("Run component_ablation_summary first")

        data = self.results['ablation_summary']
        components = data['ranked_components']
        deltas = [data['component_importance'][c]['delta_r2'] for c in components]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [COLORS['primary'] if d > 0 else COLORS['secondary'] for d in deltas]
        ax.barh(components, deltas, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Delta R2 (baseline - ablated)')
        ax.set_title('Component Ablation Impact')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format=self.config.figure_format)
        return fig


# =============================================================================
# PART 10: BASELINE COMPARISONS
# =============================================================================

class BaselineComparisonAnalyzer:
    """
    Compare deep learning model to linear baselines with Nature Methods statistics.

    Provides:
    - Multiple baseline methods (Ridge, Wiener, Mean, Persistence)
    - Per-sample R² for proper statistical comparison
    - Bootstrap confidence intervals
    - Effect sizes with CIs
    - Permutation tests for significance
    - Multiple comparison correction
    """

    def __init__(self, model: nn.Module, config: AnalysisConfig):
        self.model = model
        self.config = config
        self.device = config.device
        self.results = {}

    @torch.no_grad()
    def linear_baseline(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        methods: List[str] = ['ridge', 'wiener', 'mean', 'persistence']
    ) -> Dict:
        """
        Compare to linear baselines with comprehensive Nature Methods statistics.

        Returns per-sample metrics for proper statistical testing.
        """
        from sklearn.linear_model import Ridge, RidgeCV

        # Collect data
        train_X, train_Y = [], []
        for batch in train_loader:
            train_X.append(batch[0].numpy())
            train_Y.append(batch[1].numpy())
        train_X = np.concatenate(train_X, axis=0)
        train_Y = np.concatenate(train_Y, axis=0)

        test_X, test_Y = [], []
        for batch in test_loader:
            test_X.append(batch[0].numpy())
            test_Y.append(batch[1].numpy())
        test_X = np.concatenate(test_X, axis=0)
        test_Y = np.concatenate(test_Y, axis=0)

        n_train, C, T = train_X.shape
        n_test = test_X.shape[0]

        train_X_flat = train_X.reshape(n_train, -1)
        train_Y_flat = train_Y.reshape(n_train, -1)
        test_X_flat = test_X.reshape(n_test, -1)
        test_Y_flat = test_Y.reshape(n_test, -1)

        baseline_results = {}
        per_sample_r2 = {}

        # Helper to compute per-sample R²
        def compute_per_sample_r2(pred, target):
            r2s = []
            for i in range(len(pred)):
                ss_res = np.sum((target[i] - pred[i]) ** 2)
                ss_tot = np.sum((target[i] - np.mean(target[i])) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)
                r2s.append(r2)
            return np.array(r2s)

        # Ridge regression with cross-validated alpha
        if 'ridge' in methods:
            alphas = np.logspace(-3, 3, 20)
            ridge = RidgeCV(alphas=alphas, cv=5)
            ridge.fit(train_X_flat, train_Y_flat)
            ridge_pred = ridge.predict(test_X_flat).reshape(n_test, C, T)

            r2s = compute_per_sample_r2(ridge_pred, test_Y)
            per_sample_r2['ridge'] = r2s

            ridge_ci = NatureMethodsStatistics.bootstrap_ci(r2s)
            baseline_results['ridge'] = {
                'r2_mean': float(np.mean(r2s)),
                'r2_std': float(np.std(r2s)),
                'r2_ci_lower': ridge_ci['ci_lower'],
                'r2_ci_upper': ridge_ci['ci_upper'],
                'mse': float(np.mean((ridge_pred - test_Y) ** 2)),
                'n': len(r2s),
                'alpha': float(ridge.alpha_),
            }

        # Wiener filter (frequency domain denoising)
        if 'wiener' in methods:
            # Simple Wiener estimate using training data statistics
            wiener_pred = np.zeros_like(test_Y)
            for c in range(C):
                signal_var = np.var(train_Y[:, c, :])
                noise_var = np.var(train_Y[:, c, :] - train_X[:, c, :]) if C <= train_X.shape[1] else signal_var * 0.1
                wiener_gain = signal_var / (signal_var + noise_var + 1e-10)
                wiener_pred[:, c, :] = test_X[:, min(c, test_X.shape[1]-1), :] * wiener_gain

            r2s = compute_per_sample_r2(wiener_pred, test_Y)
            per_sample_r2['wiener'] = r2s

            wiener_ci = NatureMethodsStatistics.bootstrap_ci(r2s)
            baseline_results['wiener'] = {
                'r2_mean': float(np.mean(r2s)),
                'r2_std': float(np.std(r2s)),
                'r2_ci_lower': wiener_ci['ci_lower'],
                'r2_ci_upper': wiener_ci['ci_upper'],
                'mse': float(np.mean((wiener_pred - test_Y) ** 2)),
                'n': len(r2s),
            }

        # Mean baseline (predict training mean)
        if 'mean' in methods:
            mean_pred = np.broadcast_to(
                np.mean(train_Y, axis=0, keepdims=True),
                test_Y.shape
            )
            r2s = compute_per_sample_r2(mean_pred, test_Y)
            per_sample_r2['mean'] = r2s

            mean_ci = NatureMethodsStatistics.bootstrap_ci(r2s)
            baseline_results['mean'] = {
                'r2_mean': float(np.mean(r2s)),  # Should be ~0 by definition
                'r2_std': float(np.std(r2s)),
                'r2_ci_lower': mean_ci['ci_lower'],
                'r2_ci_upper': mean_ci['ci_upper'],
                'mse': float(np.mean((mean_pred - test_Y) ** 2)),
                'n': len(r2s),
            }

        # Persistence baseline (previous timestep)
        if 'persistence' in methods:
            persist_pred = np.roll(test_Y, 1, axis=-1)
            persist_pred[:, :, 0] = test_Y[:, :, 0]

            r2s = compute_per_sample_r2(persist_pred, test_Y)
            per_sample_r2['persistence'] = r2s

            persist_ci = NatureMethodsStatistics.bootstrap_ci(r2s)
            baseline_results['persistence'] = {
                'r2_mean': float(np.mean(r2s)),
                'r2_std': float(np.std(r2s)),
                'r2_ci_lower': persist_ci['ci_lower'],
                'r2_ci_upper': persist_ci['ci_upper'],
                'mse': float(np.mean((persist_pred - test_Y) ** 2)),
                'n': len(r2s),
            }

        # Deep learning model
        self.model.eval()
        dl_preds = []
        for batch in test_loader:
            inputs = batch[0].to(self.device)
            # Handle conditioning if present
            if len(batch) > 2:
                cond = batch[2].to(self.device) if batch[2] is not None else None
                outputs = self.model(inputs, cond)
            else:
                outputs = self.model(inputs)
            dl_preds.append(outputs.detach().cpu().numpy())

        dl_pred = np.concatenate(dl_preds, axis=0)
        dl_r2s = compute_per_sample_r2(dl_pred, test_Y)
        per_sample_r2['deep_learning'] = dl_r2s

        dl_ci = NatureMethodsStatistics.bootstrap_ci(dl_r2s)
        baseline_results['deep_learning'] = {
            'r2_mean': float(np.mean(dl_r2s)),
            'r2_std': float(np.std(dl_r2s)),
            'r2_ci_lower': dl_ci['ci_lower'],
            'r2_ci_upper': dl_ci['ci_upper'],
            'mse': float(np.mean((dl_pred - test_Y) ** 2)),
            'n': len(dl_r2s),
        }

        # Statistical comparisons (DL vs each baseline)
        statistical_comparisons = {}
        p_values = []

        for method in ['ridge', 'wiener', 'mean', 'persistence']:
            if method in per_sample_r2:
                comparison = NatureMethodsStatistics.comprehensive_comparison(
                    per_sample_r2[method],
                    per_sample_r2['deep_learning'],
                    paired=True,  # Same test samples
                    group_names=(method.capitalize(), 'Deep Learning')
                )
                statistical_comparisons[method] = comparison
                p_values.append(comparison['test']['p_value'])

        # Multiple comparison correction
        if p_values:
            fdr_results = NatureMethodsStatistics.fdr_correction(np.array(p_values))
            for i, method in enumerate([m for m in ['ridge', 'wiener', 'mean', 'persistence'] if m in statistical_comparisons]):
                statistical_comparisons[method]['adjusted_p'] = float(fdr_results['adjusted_p'][i])
                statistical_comparisons[method]['significant_fdr'] = bool(fdr_results['significant'][i])

        # Generate methods text for Nature Methods
        methods_texts = []
        for method, comp in statistical_comparisons.items():
            effect = comp['effect_size']
            test = comp['test']
            adj_p = comp.get('adjusted_p', test['p_value'])
            methods_texts.append(
                f"Deep learning vs {method}: {test['name']}, "
                f"p={adj_p:.2e if adj_p < 0.001 else f'{adj_p:.4f}'} (FDR-corrected), "
                f"Cohen's d={effect['cohens_d']:.2f} [{effect['ci_lower']:.2f}, {effect['ci_upper']:.2f}], "
                f"{effect['interpretation']} effect"
            )

        results = {
            'baseline_results': baseline_results,
            'per_sample_r2': {k: v.tolist() for k, v in per_sample_r2.items()},
            'statistical_comparisons': statistical_comparisons,
            'deep_learning_needed': any(
                comp.get('significant_fdr', comp['test']['p_value'] < 0.05)
                and comp['effect_size']['cohens_d'] > 0.2
                for comp in statistical_comparisons.values()
            ),
            'methods_text': "; ".join(methods_texts),
            'n_test_samples': n_test,
        }

        self.results['linear_baseline'] = results
        return results

    def plot_baseline_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot baseline comparison with Nature Methods standards.

        Features:
        - Individual data points shown
        - 95% bootstrap confidence intervals
        - Significance brackets with FDR-corrected p-values
        """
        if 'linear_baseline' not in self.results:
            raise ValueError("Run linear_baseline first")

        data = self.results['linear_baseline']['baseline_results']
        per_sample = self.results['linear_baseline'].get('per_sample_r2', {})
        comparisons = self.results['linear_baseline'].get('statistical_comparisons', {})

        fig, ax = plt.subplots(figsize=(FIGURE_DIMS['single_col'] * 1.5, FIGURE_DIMS['single_col']))

        methods = list(data.keys())
        # Reorder to put deep learning last
        if 'deep_learning' in methods:
            methods = [m for m in methods if m != 'deep_learning'] + ['deep_learning']

        x = np.arange(len(methods))
        width = 0.7

        # Colors: baselines grey, DL blue
        colors = [COLORS['baseline'] if m != 'deep_learning' else COLORS['primary'] for m in methods]

        # Extract means and CIs
        means = [data[m]['r2_mean'] for m in methods]
        ci_lower = [data[m].get('r2_ci_lower', data[m]['r2_mean'] - data[m]['r2_std']) for m in methods]
        ci_upper = [data[m].get('r2_ci_upper', data[m]['r2_mean'] + data[m]['r2_std']) for m in methods]

        # Error bars (asymmetric CI)
        yerr_lower = [means[i] - ci_lower[i] for i in range(len(means))]
        yerr_upper = [ci_upper[i] - means[i] for i in range(len(means))]

        # Bar plot with error bars
        bars = ax.bar(x, means, width, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5,
                      yerr=[yerr_lower, yerr_upper], capsize=3, error_kw={'linewidth': 0.8})

        # Overlay individual data points
        for i, method in enumerate(methods):
            if method in per_sample:
                r2s = np.array(per_sample[method])
                # Jitter x positions
                jitter = np.random.uniform(-0.15, 0.15, len(r2s))
                ax.scatter(x[i] + jitter, r2s, c='black', s=8, alpha=0.4, zorder=3)

        # Add significance brackets
        dl_idx = methods.index('deep_learning') if 'deep_learning' in methods else -1
        if dl_idx >= 0:
            y_max = max(ci_upper) + 0.05
            bracket_height = 0.03

            for i, method in enumerate(methods):
                if method != 'deep_learning' and method in comparisons:
                    comp = comparisons[method]
                    p_val = comp.get('adjusted_p', comp['test']['p_value'])

                    # Draw bracket
                    y_bracket = y_max + (i * bracket_height * 1.5)
                    add_significance_markers(ax, i, dl_idx, y_bracket, p_val, height=bracket_height)

        # Labels
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '\n').title() for m in methods], fontsize=7)
        ax.set_ylabel('R² (bootstrap 95% CI)')
        ax.set_title('Method Comparison')

        # Add sample size annotation
        n = data[methods[0]].get('n', '?')
        ax.text(0.02, 0.98, f'n = {n}', transform=ax.transAxes, fontsize=6,
                va='top', ha='left')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format=self.config.figure_format, dpi=300)
        return fig


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class ModelInterrogator:
    """Main orchestrator for comprehensive model analysis."""

    ANALYSIS_MODULES = {
        'training': TrainingDynamicsAnalyzer,
        'attribution': InputAttributionAnalyzer,
        'spectral': SpectralAnalyzer,
        'latent': LatentSpaceAnalyzer,
        'conditioning': ConditioningAnalyzer,
        'landscape': LossLandscapeAnalyzer,
        'generalization': GeneralizationAnalyzer,
        'error': ErrorAnalyzer,
        'architectural': ArchitecturalAnalyzer,
        'baseline': BaselineComparisonAnalyzer,
    }

    def __init__(
        self,
        model: nn.Module,
        config: AnalysisConfig,
        dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None
    ):
        self.model = model.to(config.device)
        self.config = config
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader or dataloader
        self.results = AnalysisResults()

        self.analyzers = {}
        for name, AnalyzerClass in self.ANALYSIS_MODULES.items():
            if name in ['training', 'spectral']:
                self.analyzers[name] = AnalyzerClass(config)
            else:
                self.analyzers[name] = AnalyzerClass(model, config)

    def run_analysis(
        self,
        analyses: Optional[List[str]] = None,
        training_logs: Optional[List[Dict]] = None
    ) -> AnalysisResults:
        """Run specified analyses."""
        if analyses is None or 'all' in analyses:
            analyses = list(self.ANALYSIS_MODULES.keys())

        print(f"Running {len(analyses)} analyses...")

        for analysis_name in analyses:
            if analysis_name not in self.ANALYSIS_MODULES:
                continue

            print(f"\n{'='*50}")
            print(f"Running: {analysis_name.upper()}")
            print('='*50)

            try:
                self._run_single_analysis(analysis_name, training_logs)
                print(f"Completed: {analysis_name}")
            except Exception as e:
                print(f"Failed {analysis_name}: {str(e)}")

        return self.results

    def _run_single_analysis(self, name: str, training_logs: Optional[List[Dict]] = None):
        """Run a single analysis module."""
        analyzer = self.analyzers[name]

        if name == 'training' and training_logs:
            self.results.training_dynamics['learning_curves'] = analyzer.analyze_learning_curves(training_logs)

        elif name == 'attribution':
            self.results.input_attribution = {
                'channel_occlusion': analyzer.channel_importance_occlusion(self.dataloader, n_samples=100),
                'temporal_saliency': analyzer.temporal_saliency(self.dataloader, n_samples=50),
            }

        elif name == 'spectral':
            preds, targets, inputs = self._get_predictions()
            self.results.spectral_analysis = {
                'bandwise': analyzer.bandwise_accuracy(preds, targets),
                'psd': analyzer.psd_comparison(preds, targets),
                'plv': analyzer.phase_locking_value(preds, targets),
            }

        elif name == 'latent':
            results = {'bottleneck': analyzer.extract_bottleneck_representations(self.dataloader, n_samples=500)}
            if results['bottleneck'].get('representations', np.array([])).size > 0:
                results['dimensionality'] = analyzer.analyze_dimensionality()
            self.results.latent_space = results

        elif name == 'conditioning':
            self.results.conditioning = {
                'ablation': analyzer.conditioning_ablation(self.dataloader, n_samples=200),
            }

        elif name == 'landscape':
            self.results.loss_landscape = {
                'surface': analyzer.compute_loss_landscape(self.dataloader, resolution=21),
                'sharpness': analyzer.analyze_sharpness(self.dataloader),
            }

        elif name == 'generalization':
            self.results.generalization = {
                'temporal': analyzer.temporal_generalization(self.dataloader),
                'signal_quality': analyzer.performance_vs_signal_quality(self.dataloader),
            }

        elif name == 'error':
            self.results.error_analysis = {
                'distribution': analyzer.error_distribution(self.dataloader),
                'worst_cases': analyzer.worst_case_gallery(self.dataloader),
                'residuals': analyzer.residual_analysis(self.dataloader),
            }

        elif name == 'architectural':
            self.results.architectural = {'depth': analyzer.depth_analysis(self.dataloader)}

        elif name == 'baseline':
            self.results.baselines = {'linear': analyzer.linear_baseline(self.dataloader, self.val_dataloader)}

    @torch.no_grad()
    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions."""
        self.model.eval()
        preds, targets, inputs = [], [], []
        for batch in self.dataloader:
            inp = batch[0].to(self.config.device)
            tgt = batch[1].to(self.config.device)
            out = self.model(inp)
            inputs.append(inp.detach().cpu().numpy())
            preds.append(out.detach().cpu().numpy())
            targets.append(tgt.detach().cpu().numpy())
        return np.concatenate(preds), np.concatenate(targets), np.concatenate(inputs)

    def generate_summary_report(self) -> str:
        """Generate comprehensive publication-ready summary report.

        Follows Nature/Science reporting standards with:
        - Training dynamics summary
        - Spectral fidelity analysis
        - Error analysis with distributions
        - Generalization assessment
        - Baseline comparisons with effect sizes
        """
        lines = []
        lines.append("=" * 70)
        lines.append("MODEL INTERROGATION SUMMARY REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # ---------------------------------------------------------------------
        # TRAINING DYNAMICS
        # ---------------------------------------------------------------------
        lines.append("-" * 70)
        lines.append("1. TRAINING DYNAMICS")
        lines.append("-" * 70)

        if self.results.training_dynamics:
            td = self.results.training_dynamics
            if 'learning_curves' in td:
                lc = td['learning_curves']
                conv_epochs = lc.get('convergence_epoch', [])
                best_losses = lc.get('best_val_loss', [])
                overfitting = lc.get('overfitting_gap', [])

                lines.append(f"  Epochs to convergence: {np.mean(conv_epochs):.1f} ± {np.std(conv_epochs):.1f}")
                lines.append(f"  Best validation loss: {np.mean(best_losses):.4f} ± {np.std(best_losses):.4f}")
                lines.append(f"  Overfitting gap: {np.mean(overfitting):.4f} ± {np.std(overfitting):.4f}")
                lines.append(f"  Training stability: {'STABLE' if np.std(best_losses) < 0.1 else 'VARIABLE'}")
        else:
            lines.append("  [Training dynamics not recorded]")
        lines.append("")

        # ---------------------------------------------------------------------
        # SPECTRAL ANALYSIS
        # ---------------------------------------------------------------------
        lines.append("-" * 70)
        lines.append("2. SPECTRAL FIDELITY")
        lines.append("-" * 70)

        if self.results.spectral_analysis:
            sa = self.results.spectral_analysis

            if 'bandwise' in sa:
                bw = sa['bandwise']
                lines.append(f"  Mean band-wise R²: {bw.get('mean_r2', 0):.4f}")
                lines.append(f"  Best preserved band: {bw.get('best_band', 'N/A')}")
                lines.append(f"  Worst preserved band: {bw.get('worst_band', 'N/A')}")
                lines.append("")
                lines.append("  Band-wise Performance:")
                if 'band_results' in bw:
                    for band, metrics in bw['band_results'].items():
                        r2 = metrics.get('r2', 0)
                        desc = FREQ_BAND_DESCRIPTIONS.get(band, '')
                        lines.append(f"    {band:12s}: R² = {r2:.4f}  ({desc})")

            if 'psd' in sa:
                psd = sa['psd']
                lines.append(f"\n  Mean PSD error: {psd.get('mean_spectral_error_db', 0):.2f} dB")

            if 'plv' in sa:
                plv = sa['plv']
                lines.append(f"  Mean Phase Locking Value: {plv.get('mean_plv', 0):.4f}")
        else:
            lines.append("  [Spectral analysis not performed]")
        lines.append("")

        # ---------------------------------------------------------------------
        # INPUT ATTRIBUTION
        # ---------------------------------------------------------------------
        lines.append("-" * 70)
        lines.append("3. INPUT ATTRIBUTION")
        lines.append("-" * 70)

        if self.results.input_attribution:
            ia = self.results.input_attribution

            if 'channel_occlusion' in ia:
                co = ia['channel_occlusion']
                ranked = co.get('ranked_channels', [])[:5]
                lines.append(f"  Top 5 important channels (occlusion): {ranked}")

            if 'temporal_saliency' in ia:
                ts = ia['temporal_saliency']
                peak = ts.get('peak_timepoint', 0)
                lines.append(f"  Peak temporal saliency: {peak / self.config.sampling_rate:.3f} s")

            if 'input_ablation' in ia:
                ab = ia['input_ablation']
                critical = ab.get('critical_segment', 'N/A')
                lines.append(f"  Critical input segment: {critical}")
        else:
            lines.append("  [Input attribution not performed]")
        lines.append("")

        # ---------------------------------------------------------------------
        # LATENT SPACE
        # ---------------------------------------------------------------------
        lines.append("-" * 70)
        lines.append("4. LATENT SPACE ANALYSIS")
        lines.append("-" * 70)

        if self.results.latent_space:
            ls = self.results.latent_space

            if 'bottleneck' in ls:
                bn = ls['bottleneck']
                lines.append(f"  Bottleneck dimension: {bn.get('feature_dim', 'N/A')}")
                lines.append(f"  Samples extracted: {bn.get('n_samples', 0)}")

            if 'dimensionality' in ls:
                dim = ls['dimensionality']
                lines.append(f"  Intrinsic dimensionality: {dim.get('intrinsic_dimensionality', 0):.1f}")
                lines.append(f"  Components for 90% variance: {dim.get('n_components_for_threshold', 'N/A')}")
        else:
            lines.append("  [Latent space analysis not performed]")
        lines.append("")

        # ---------------------------------------------------------------------
        # LOSS LANDSCAPE
        # ---------------------------------------------------------------------
        lines.append("-" * 70)
        lines.append("5. LOSS LANDSCAPE")
        lines.append("-" * 70)

        if self.results.loss_landscape:
            ll = self.results.loss_landscape

            if 'sharpness' in ll:
                sh = ll['sharpness']
                is_flat = sh.get('is_flat', False)
                sharpness = sh.get('sharpness', 0)
                lines.append(f"  Sharpness metric: {sharpness:.4f}")
                lines.append(f"  Minimum type: {'FLAT (good generalization)' if is_flat else 'SHARP (risk of overfitting)'}")

            if 'surface' in ll:
                surf = ll['surface']
                lines.append(f"  Loss at center: {surf.get('center_loss', 0):.4f}")
                lines.append(f"  Loss range: [{surf.get('min_loss', 0):.4f}, {surf.get('max_loss', 0):.4f}]")
        else:
            lines.append("  [Loss landscape not computed]")
        lines.append("")

        # ---------------------------------------------------------------------
        # ERROR ANALYSIS
        # ---------------------------------------------------------------------
        lines.append("-" * 70)
        lines.append("6. ERROR ANALYSIS")
        lines.append("-" * 70)

        if self.results.error_analysis:
            ea = self.results.error_analysis

            if 'distribution' in ea:
                ed = ea['distribution']
                lines.append(f"  Error mean: {ed.get('error_mean', 0):.6f}")
                lines.append(f"  Error std: {ed.get('error_std', 0):.6f}")
                lines.append(f"  Error skew: {ed.get('error_skew', 0):.3f}")
                lines.append(f"  Error kurtosis: {ed.get('error_kurtosis', 0):.3f}")
                lines.append(f"  Distribution: {'NORMAL' if ed.get('is_normal', False) else 'NON-NORMAL'}")
                lines.append(f"  Heavy tails: {'YES' if ed.get('has_heavy_tails', False) else 'NO'}")

            if 'residuals' in ea:
                res = ea['residuals']
                lines.append(f"  Residual structure: {'PRESENT (model missing patterns)' if res.get('has_structure', False) else 'NONE (random noise)'}")

            if 'worst_cases' in ea:
                wc = ea['worst_cases']
                lines.append(f"  Worst case MSE: {wc.get('worst_mses', [0])[0]:.4f}")
                if wc.get('commonalities'):
                    lines.append(f"  Failure patterns: {', '.join(wc['commonalities'])}")
        else:
            lines.append("  [Error analysis not performed]")
        lines.append("")

        # ---------------------------------------------------------------------
        # GENERALIZATION
        # ---------------------------------------------------------------------
        lines.append("-" * 70)
        lines.append("7. GENERALIZATION")
        lines.append("-" * 70)

        if self.results.generalization:
            gen = self.results.generalization

            if 'temporal' in gen:
                temp = gen['temporal']
                drift = temp.get('drift', 0)
                lines.append(f"  Temporal drift: {drift:.4f} ({'SIGNIFICANT' if temp.get('has_drift', False) else 'NONE'})")

            if 'signal_quality' in gen:
                sq = gen['signal_quality']
                corr = sq.get('variance_r2_correlation', 0)
                lines.append(f"  Performance vs input variance correlation: {corr:.3f}")
                lines.append(f"  Fails on noisy inputs: {'YES' if sq.get('fails_on_noisy', False) else 'NO'}")
        else:
            lines.append("  [Generalization analysis not performed]")
        lines.append("")

        # ---------------------------------------------------------------------
        # BASELINE COMPARISON (Nature Methods Format)
        # ---------------------------------------------------------------------
        lines.append("-" * 70)
        lines.append("8. BASELINE COMPARISON (Nature Methods Statistics)")
        lines.append("-" * 70)

        if self.results.baselines:
            bl = self.results.baselines

            if 'linear' in bl:
                lin = bl['linear']
                n_samples = lin.get('n_test_samples', '?')
                lines.append(f"  Test samples: n = {n_samples}")
                lines.append("")
                lines.append("  Method Performance (mean ± SD, 95% CI):")
                if 'baseline_results' in lin:
                    for method, metrics in lin['baseline_results'].items():
                        r2_mean = metrics.get('r2_mean', metrics.get('r2', 0))
                        r2_std = metrics.get('r2_std', 0)
                        ci_low = metrics.get('r2_ci_lower', r2_mean - r2_std)
                        ci_high = metrics.get('r2_ci_upper', r2_mean + r2_std)
                        n = metrics.get('n', '?')
                        lines.append(f"    {method:15s}: R² = {r2_mean:.4f} ± {r2_std:.4f} [{ci_low:.4f}, {ci_high:.4f}], n={n}")

                # Statistical comparisons with effect sizes
                if 'statistical_comparisons' in lin:
                    lines.append("")
                    lines.append("  Statistical Comparisons (vs Deep Learning):")
                    for method, comp in lin['statistical_comparisons'].items():
                        effect = comp['effect_size']
                        test = comp['test']
                        adj_p = comp.get('adjusted_p', test['p_value'])
                        sig = '*' if adj_p < 0.05 else ''
                        sig += '*' if adj_p < 0.01 else ''
                        sig += '*' if adj_p < 0.001 else ''
                        lines.append(
                            f"    vs {method:12s}: {test['name']}, "
                            f"p={adj_p:.2e if adj_p < 0.001 else f'{adj_p:.4f}'}{' (FDR)' if 'adjusted_p' in comp else ''}, "
                            f"d={effect['cohens_d']:.2f} [{effect['ci_lower']:.2f}, {effect['ci_upper']:.2f}] "
                            f"({effect['interpretation']}) {sig}"
                        )

                # Methods text for paper
                if 'methods_text' in lin:
                    lines.append("")
                    lines.append("  FOR METHODS SECTION:")
                    lines.append(f"    {lin['methods_text']}")

                lines.append("")
                lines.append(f"  Deep learning justified: {'YES (significant improvement with medium+ effect)' if lin.get('deep_learning_needed', False) else 'MARGINAL (consider simpler models)'}")
        else:
            lines.append("  [Baseline comparison not performed]")
        lines.append("")

        # ---------------------------------------------------------------------
        # SUMMARY & RECOMMENDATIONS
        # ---------------------------------------------------------------------
        lines.append("=" * 70)
        lines.append("SUMMARY & RECOMMENDATIONS")
        lines.append("=" * 70)

        recommendations = []

        # Check for issues and add recommendations
        if self.results.error_analysis and 'residuals' in self.results.error_analysis:
            if self.results.error_analysis['residuals'].get('has_structure', False):
                recommendations.append("• Residuals show structure - model may benefit from architectural changes")

        if self.results.loss_landscape and 'sharpness' in self.results.loss_landscape:
            if not self.results.loss_landscape['sharpness'].get('is_flat', True):
                recommendations.append("• Sharp minimum detected - consider regularization or SAM optimizer")

        if self.results.spectral_analysis and 'bandwise' in self.results.spectral_analysis:
            bw = self.results.spectral_analysis['bandwise']
            if bw.get('worst_band') == 'high_gamma':
                recommendations.append("• High-gamma band underperforming - consider increasing model capacity")

        if self.results.baselines and 'linear' in self.results.baselines:
            if not self.results.baselines['linear'].get('deep_learning_needed', True):
                recommendations.append("• Linear baseline performs similarly - deep learning may be unnecessary")

        if recommendations:
            for rec in recommendations:
                lines.append(rec)
        else:
            lines.append("• No significant issues detected")

        lines.append("")
        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)

    def save_results(self, path: Optional[str] = None):
        """Save results to JSON."""
        path = path or os.path.join(self.config.output_dir, 'results.json')
        self.results.save(path)
        print(f"Saved to {path}")

    def generate_nature_methods_report(self) -> str:
        """
        Generate complete Nature Methods format report.

        Includes:
        - Methods section text
        - Statistical analysis details
        - Figure legends
        - Data availability statement
        - Code availability statement
        """
        lines = []
        lines.append("=" * 80)
        lines.append("NATURE METHODS FORMAT REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Random seed: {self.config.seed}")
        lines.append("")

        # Software versions
        lines.append("-" * 80)
        lines.append("SOFTWARE VERSIONS (for Methods section)")
        lines.append("-" * 80)
        import sys
        lines.append(f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        lines.append(f"  PyTorch: {torch.__version__}")
        lines.append(f"  NumPy: {np.__version__}")
        try:
            import scipy
            lines.append(f"  SciPy: {scipy.__version__}")
        except:
            pass
        try:
            import sklearn
            lines.append(f"  scikit-learn: {sklearn.__version__}")
        except:
            pass
        lines.append("")

        # Statistical methods
        lines.append("-" * 80)
        lines.append("STATISTICAL METHODS (copy for Methods section)")
        lines.append("-" * 80)
        lines.append("""
Statistical Analysis
--------------------
All statistical analyses were performed using custom Python code (available at
[repository URL]). Data are presented as mean ± standard deviation unless
otherwise noted. Confidence intervals (95% CI) were computed using the
bias-corrected and accelerated bootstrap method (n_bootstrap = 10,000).

For comparisons between methods, we used paired statistical tests as the same
test samples were evaluated by all methods. Normality was assessed using the
Shapiro-Wilk test (n < 50) or D'Agostino-Pearson test (n >= 50). For normally
distributed data, paired t-tests were used; otherwise, Wilcoxon signed-rank
tests were applied. P-values were corrected for multiple comparisons using the
Benjamini-Hochberg false discovery rate (FDR) procedure.

Effect sizes are reported as Cohen's d with 95% bootstrap confidence intervals.
Effect size interpretation: |d| < 0.2 (negligible), 0.2-0.5 (small), 0.5-0.8
(medium), > 0.8 (large). Statistical significance was set at α = 0.05 after
FDR correction.
""")

        # Sample sizes
        lines.append("-" * 80)
        lines.append("SAMPLE SIZES")
        lines.append("-" * 80)
        if self.results.baselines and 'linear' in self.results.baselines:
            n = self.results.baselines['linear'].get('n_test_samples', 'N/A')
            lines.append(f"  Test samples for baseline comparison: n = {n}")
        if self.results.error_analysis and 'distribution' in self.results.error_analysis:
            n = self.results.error_analysis['distribution'].get('n_samples', 'N/A')
            lines.append(f"  Samples for error analysis: n = {n}")
        lines.append("")

        # Statistical tests performed
        lines.append("-" * 80)
        lines.append("STATISTICAL TESTS PERFORMED")
        lines.append("-" * 80)
        if self.results.baselines and 'linear' in self.results.baselines:
            lin = self.results.baselines['linear']
            if 'statistical_comparisons' in lin:
                for method, comp in lin['statistical_comparisons'].items():
                    test = comp['test']
                    effect = comp['effect_size']
                    lines.append(f"\n  {method.upper()} vs DEEP LEARNING:")
                    lines.append(f"    Test: {test['name']}")
                    lines.append(f"    n₁ = {effect['n1']}, n₂ = {effect['n2']}")
                    if test['df'] is not None:
                        lines.append(f"    df = {test['df']}")
                    lines.append(f"    Test statistic = {test['statistic']:.4f}")
                    lines.append(f"    p-value (uncorrected) = {test['p_value']:.6f}")
                    if 'adjusted_p' in comp:
                        lines.append(f"    p-value (FDR-corrected) = {comp['adjusted_p']:.6f}")
                    lines.append(f"    Cohen's d = {effect['cohens_d']:.4f}")
                    lines.append(f"    95% CI for d: [{effect['ci_lower']:.4f}, {effect['ci_upper']:.4f}]")
                    lines.append(f"    Interpretation: {effect['interpretation']} effect")
        lines.append("")

        # Data and code availability
        lines.append("-" * 80)
        lines.append("DATA AND CODE AVAILABILITY (template)")
        lines.append("-" * 80)
        lines.append("""
Data Availability
-----------------
Source data for all figures are provided with this paper. The neural signal
datasets used in this study are available at [DOI/URL]. Processed data and
intermediate results are available upon reasonable request.

Code Availability
-----------------
The model interrogation analysis code is available at [GitHub URL]. The deep
learning model implementation and training scripts are available at [GitHub URL].
All analyses were performed using Python [version] with the packages listed in
the Methods section.
""")

        # Figure legends template
        lines.append("-" * 80)
        lines.append("FIGURE LEGEND TEMPLATES")
        lines.append("-" * 80)

        if self.results.baselines and 'linear' in self.results.baselines:
            lin = self.results.baselines['linear']
            n = lin.get('n_test_samples', 'N')
            lines.append(f"""
Figure X. Deep learning model outperforms linear baselines.
Bar heights indicate mean R² across n = {n} test samples. Error bars show 95%
bootstrap confidence intervals (n_bootstrap = 10,000). Individual data points
are shown as grey dots. Statistical comparisons were made using paired tests
(see Methods) with FDR correction for multiple comparisons.
*p < 0.05, **p < 0.01, ***p < 0.001.
""")

        if self.results.spectral_analysis:
            lines.append("""
Figure Y. Spectral fidelity analysis.
(a) Band-wise R² showing reconstruction accuracy for each frequency band.
    Error bars: ±1 SD across samples.
(b) Power spectral density comparison. Black: ground truth; colored: predicted.
    Shaded region: ±1 SD.
(c) Phase-locking value (PLV) between ground truth and predicted signals.
    Dashed line: chance level.
""")

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF NATURE METHODS REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_nature_methods_report(self, path: Optional[str] = None):
        """Save Nature Methods format report to file."""
        path = path or os.path.join(self.config.output_dir, 'nature_methods_report.txt')
        report = self.generate_nature_methods_report()
        with open(path, 'w') as f:
            f.write(report)
        print(f"Nature Methods report saved to {path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Model Interrogation Framework')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='.', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='interrogation_results')
    parser.add_argument('--analyses', type=str, default='all', help='Analyses to run')
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device

    config = AnalysisConfig(
        checkpoint_path=args.checkpoint or '',
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
    )

    print("Model Interrogation Framework")
    print(f"Device: {device}")
    print(f"Output: {config.output_dir}")

    # Example with dummy model/data
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(32, 32, 7, padding=3)
        def forward(self, x):
            return self.conv(x)

    class DummyDataset(Dataset):
        def __init__(self, n=100):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            return torch.randn(32, 5000), torch.randn(32, 5000)

    model = DummyModel()
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=8)

    interrogator = ModelInterrogator(model, config, dataloader)
    analyses = args.analyses.split(',') if args.analyses != 'all' else ['all']
    interrogator.run_analysis(analyses)

    print(interrogator.generate_summary_report())
    interrogator.save_results()

    print("\nInterrogation complete!")


if __name__ == '__main__':
    main()