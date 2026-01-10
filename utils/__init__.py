"""
Utilities for Nature Methods Publication
========================================

Shared utilities for statistical testing, visualization, and metrics.
"""

from .statistics import (
    # Result classes
    TestResult,
    ComparisonResult,
    MetricsResult,
    # Effect sizes
    cohens_d,
    hedges_g,
    cohens_d_paired,
    effect_size_interpretation,
    # Confidence intervals
    confidence_interval,
    bootstrap_ci,
    mean_diff_ci,
    # Normality tests
    test_normality,
    test_homogeneity,
    # Statistical tests
    paired_ttest,
    wilcoxon_test,
    independent_ttest,
    mann_whitney_test,
    # Multiple comparisons
    bonferroni_correction,
    holm_correction,
    fdr_correction,
    # Comparison functions
    compare_methods,
    compare_multiple_methods,
    # Metrics
    compute_metrics,
    compute_fold_statistics,
    # Formatting
    format_mean_ci,
    format_mean_std,
    create_summary_table,
    check_assumptions,
)

from .plotting import (
    # Style
    NATURE_STYLE,
    COLORS_CATEGORICAL,
    setup_nature_style,
    # Significance
    add_significance_marker,
    add_significance_stars,
    # Plots
    bar_plot_with_significance,
    box_plot_with_points,
    line_plot_with_ci,
    multi_line_plot_with_ci,
    comparison_heatmap,
    effect_size_forest_plot,
    add_stats_table,
    # Figure helpers
    create_figure,
    save_figure,
    add_panel_labels,
)

__all__ = [
    # Statistics - Result classes
    "TestResult",
    "ComparisonResult",
    "MetricsResult",
    # Statistics - Effect sizes
    "cohens_d",
    "hedges_g",
    "cohens_d_paired",
    "effect_size_interpretation",
    # Statistics - Confidence intervals
    "confidence_interval",
    "bootstrap_ci",
    "mean_diff_ci",
    # Statistics - Normality tests
    "test_normality",
    "test_homogeneity",
    # Statistics - Tests
    "paired_ttest",
    "wilcoxon_test",
    "independent_ttest",
    "mann_whitney_test",
    # Statistics - Multiple comparisons
    "bonferroni_correction",
    "holm_correction",
    "fdr_correction",
    # Statistics - Comparison functions
    "compare_methods",
    "compare_multiple_methods",
    # Statistics - Metrics
    "compute_metrics",
    "compute_fold_statistics",
    # Statistics - Formatting
    "format_mean_ci",
    "format_mean_std",
    "create_summary_table",
    "check_assumptions",
    # Plotting - Style
    "NATURE_STYLE",
    "COLORS_CATEGORICAL",
    "setup_nature_style",
    # Plotting - Significance
    "add_significance_marker",
    "add_significance_stars",
    # Plotting - Plots
    "bar_plot_with_significance",
    "box_plot_with_points",
    "line_plot_with_ci",
    "multi_line_plot_with_ci",
    "comparison_heatmap",
    "effect_size_forest_plot",
    "add_stats_table",
    # Plotting - Figure helpers
    "create_figure",
    "save_figure",
    "add_panel_labels",
]
