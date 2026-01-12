"""
Visualization Utilities for Nature Methods Publication
======================================================

Publication-quality plotting utilities with:
- Nature Methods style formatting
- Significance markers and annotations
- Error bars and confidence intervals
- Consistent color schemes
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# =============================================================================
# Nature Methods Style Configuration
# =============================================================================

# Nature Methods figure guidelines:
# - Single column: 88mm (3.46 inches)
# - Double column: 180mm (7.09 inches)
# - Full page: 180mm x 240mm
# - Font: Arial or Helvetica
# - Min font size: 5pt, recommended: 7-8pt
# - Line width: 0.5-1pt

NATURE_STYLE = {
    # Figure sizes (in inches)
    "single_column": (3.46, 2.5),
    "double_column": (7.09, 4.0),
    "full_page": (7.09, 9.45),

    # Fonts
    "font_family": "Arial",
    "font_size": 7,
    "font_size_title": 8,
    "font_size_label": 7,
    "font_size_tick": 6,
    "font_size_legend": 6,
    "font_size_annotation": 6,

    # Lines
    "line_width": 1.0,
    "line_width_thin": 0.5,
    "line_width_thick": 1.5,

    # Markers
    "marker_size": 4,
    "marker_size_small": 3,

    # Colors
    "colors": {
        "primary": "#2171B5",      # Blue
        "secondary": "#CB181D",    # Red
        "tertiary": "#238B45",     # Green
        "quaternary": "#6A51A3",   # Purple
        "neutral": "#525252",      # Gray
        "highlight": "#FD8D3C",    # Orange
        "background": "#F7F7F7",   # Light gray
    },

    # Error bars
    "error_capsize": 2,
    "error_linewidth": 0.5,
}

# Color palette for multiple items
COLORS_CATEGORICAL = [
    "#2171B5",  # Blue
    "#CB181D",  # Red
    "#238B45",  # Green
    "#6A51A3",  # Purple
    "#FD8D3C",  # Orange
    "#41B6C4",  # Cyan
    "#FDAE6B",  # Light orange
    "#807DBA",  # Light purple
]


def setup_nature_style():
    """Apply Nature Methods style to matplotlib."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": NATURE_STYLE["font_size"],
        "axes.titlesize": NATURE_STYLE["font_size_title"],
        "axes.labelsize": NATURE_STYLE["font_size_label"],
        "xtick.labelsize": NATURE_STYLE["font_size_tick"],
        "ytick.labelsize": NATURE_STYLE["font_size_tick"],
        "legend.fontsize": NATURE_STYLE["font_size_legend"],
        "axes.linewidth": NATURE_STYLE["line_width_thin"],
        "xtick.major.width": NATURE_STYLE["line_width_thin"],
        "ytick.major.width": NATURE_STYLE["line_width_thin"],
        "lines.linewidth": NATURE_STYLE["line_width"],
        "lines.markersize": NATURE_STYLE["marker_size"],
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# =============================================================================
# Significance Annotation
# =============================================================================

def add_significance_marker(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    p_value: float,
    height: float = 0.02,
    color: str = "black",
    fontsize: int = 8,
):
    """Add significance bracket and marker between two bars.

    Args:
        ax: Matplotlib axes
        x1, x2: x positions of the two items being compared
        y: y position of the bracket
        p_value: p-value for determining significance marker
        height: Height of the bracket arms
        color: Color of bracket and text
        fontsize: Font size for marker
    """
    # Determine marker
    if p_value < 0.001:
        marker = "***"
    elif p_value < 0.01:
        marker = "**"
    elif p_value < 0.05:
        marker = "*"
    else:
        marker = "ns"

    # Draw bracket
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y],
            color=color, linewidth=NATURE_STYLE["line_width_thin"])

    # Add marker
    ax.text((x1 + x2) / 2, y + height, marker,
            ha="center", va="bottom", color=color, fontsize=fontsize)


def add_significance_stars(
    ax: plt.Axes,
    x: float,
    y: float,
    p_value: float,
    fontsize: int = 8,
    color: str = "black",
):
    """Add significance stars above a bar.

    Args:
        ax: Matplotlib axes
        x: x position
        y: y position (top of bar + offset)
        p_value: p-value
        fontsize: Font size
        color: Text color
    """
    if p_value < 0.001:
        marker = "***"
    elif p_value < 0.01:
        marker = "**"
    elif p_value < 0.05:
        marker = "*"
    else:
        return  # No marker for ns

    ax.text(x, y, marker, ha="center", va="bottom", fontsize=fontsize, color=color)


# =============================================================================
# Bar Plots with Error Bars
# =============================================================================

def bar_plot_with_significance(
    ax: plt.Axes,
    data: Dict[str, Dict[str, float]],
    comparisons: Optional[Dict[str, Dict]] = None,
    baseline_name: Optional[str] = None,
    ylabel: str = "R²",
    colors: Optional[List[str]] = None,
    show_individual_points: bool = True,
    individual_data: Optional[Dict[str, List[float]]] = None,
):
    """Create bar plot with error bars and significance markers.

    Args:
        ax: Matplotlib axes
        data: Dict mapping names to {"mean": x, "std": y, "ci_lower": z, "ci_upper": w}
        comparisons: Dict with comparison results including p-values
        baseline_name: Name of baseline for significance comparisons
        ylabel: Y-axis label
        colors: Optional list of colors
        show_individual_points: Whether to overlay individual data points
        individual_data: Dict mapping names to list of individual values
    """
    names = list(data.keys())
    n = len(names)
    x = np.arange(n)

    if colors is None:
        colors = COLORS_CATEGORICAL[:n]

    means = [data[name]["mean"] for name in names]
    stds = [data[name].get("std", 0) for name in names]

    # Use CI if available, otherwise std
    ci_lower = [data[name].get("ci_lower", data[name]["mean"] - stds[i])
                for i, name in enumerate(names)]
    ci_upper = [data[name].get("ci_upper", data[name]["mean"] + stds[i])
                for i, name in enumerate(names)]

    yerr_lower = [means[i] - ci_lower[i] for i in range(n)]
    yerr_upper = [ci_upper[i] - means[i] for i in range(n)]

    # Create bars
    bars = ax.bar(x, means, color=colors, edgecolor="black",
                  linewidth=NATURE_STYLE["line_width_thin"], alpha=0.8)

    # Add error bars
    ax.errorbar(x, means, yerr=[yerr_lower, yerr_upper],
                fmt="none", color="black",
                capsize=NATURE_STYLE["error_capsize"],
                linewidth=NATURE_STYLE["error_linewidth"],
                capthick=NATURE_STYLE["error_linewidth"])

    # Overlay individual points
    if show_individual_points and individual_data:
        for i, name in enumerate(names):
            if name in individual_data:
                points = individual_data[name]
                jitter = np.random.uniform(-0.15, 0.15, len(points))
                ax.scatter(x[i] + jitter, points, color="black",
                          s=NATURE_STYLE["marker_size_small"]**2,
                          alpha=0.5, zorder=3)

    # Add significance markers
    if comparisons and baseline_name:
        baseline_idx = names.index(baseline_name) if baseline_name in names else 0
        y_max = max(ci_upper) * 1.1

        for i, name in enumerate(names):
            if name in comparisons:
                comp = comparisons[name]
                p_value = comp.get("p_value", 1.0)
                if p_value < 0.05:
                    add_significance_marker(
                        ax, baseline_idx, i, y_max,
                        p_value, height=y_max * 0.05
                    )
                    y_max *= 1.15

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(ylabel)

    return bars


# =============================================================================
# Box Plots
# =============================================================================

def box_plot_with_points(
    ax: plt.Axes,
    data: Dict[str, List[float]],
    ylabel: str = "R²",
    colors: Optional[List[str]] = None,
    show_points: bool = True,
):
    """Create box plot with overlaid individual points.

    Args:
        ax: Matplotlib axes
        data: Dict mapping names to lists of values
        ylabel: Y-axis label
        colors: Optional colors
        show_points: Whether to show individual points
    """
    names = list(data.keys())
    values = [data[name] for name in names]
    n = len(names)

    if colors is None:
        colors = COLORS_CATEGORICAL[:n]

    # Create box plot
    bp = ax.boxplot(values, patch_artist=True, labels=names,
                    widths=0.6, showfliers=False)

    # Color the boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")

    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("black")
            item.set_linewidth(NATURE_STYLE["line_width_thin"])

    # Overlay individual points
    if show_points:
        for i, (name, vals) in enumerate(data.items()):
            jitter = np.random.uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                      color="black", s=NATURE_STYLE["marker_size_small"]**2,
                      alpha=0.5, zorder=3)

    ax.set_ylabel(ylabel)
    ax.set_xticklabels(names, rotation=45, ha="right")


# =============================================================================
# Line Plots with Confidence Bands
# =============================================================================

def line_plot_with_ci(
    ax: plt.Axes,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_ci_lower: np.ndarray,
    y_ci_upper: np.ndarray,
    label: str = "",
    color: str = None,
    alpha_band: float = 0.2,
):
    """Plot line with confidence interval band.

    Args:
        ax: Matplotlib axes
        x: X values
        y_mean: Mean Y values
        y_ci_lower: Lower CI bound
        y_ci_upper: Upper CI bound
        label: Line label
        color: Line color
        alpha_band: Transparency of CI band
    """
    if color is None:
        color = NATURE_STYLE["colors"]["primary"]

    ax.plot(x, y_mean, color=color, label=label,
            linewidth=NATURE_STYLE["line_width"])
    ax.fill_between(x, y_ci_lower, y_ci_upper,
                    color=color, alpha=alpha_band)


def multi_line_plot_with_ci(
    ax: plt.Axes,
    data: Dict[str, Dict[str, np.ndarray]],
    xlabel: str = "Epoch",
    ylabel: str = "R²",
    colors: Optional[Dict[str, str]] = None,
):
    """Plot multiple lines with confidence bands.

    Args:
        ax: Matplotlib axes
        data: Dict mapping names to {"x": x, "mean": y, "ci_lower": l, "ci_upper": u}
        xlabel: X-axis label
        ylabel: Y-axis label
        colors: Optional dict mapping names to colors
    """
    for i, (name, d) in enumerate(data.items()):
        color = colors.get(name) if colors else COLORS_CATEGORICAL[i % len(COLORS_CATEGORICAL)]
        line_plot_with_ci(
            ax,
            d["x"],
            d["mean"],
            d.get("ci_lower", d["mean"]),
            d.get("ci_upper", d["mean"]),
            label=name,
            color=color,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)


# =============================================================================
# Heatmaps
# =============================================================================

def comparison_heatmap(
    ax: plt.Axes,
    p_values: np.ndarray,
    labels: List[str],
    title: str = "Statistical Comparisons",
    cmap: str = "RdYlGn_r",
    annotate: bool = True,
):
    """Create heatmap of p-values for pairwise comparisons.

    Args:
        ax: Matplotlib axes
        p_values: 2D array of p-values
        labels: Labels for rows/columns
        title: Plot title
        cmap: Colormap
        annotate: Whether to annotate cells with significance markers
    """
    im = ax.imshow(p_values, cmap=cmap, vmin=0, vmax=0.1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    if annotate:
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j:
                    p = p_values[i, j]
                    if p < 0.001:
                        text = "***"
                    elif p < 0.01:
                        text = "**"
                    elif p < 0.05:
                        text = "*"
                    else:
                        text = ""

                    ax.text(j, i, text, ha="center", va="center",
                           fontsize=NATURE_STYLE["font_size_annotation"])

    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="p-value")


# =============================================================================
# Effect Size Visualization
# =============================================================================

def effect_size_forest_plot(
    ax: plt.Axes,
    effects: Dict[str, Dict[str, float]],
    baseline_name: str = "Baseline",
    xlabel: str = "Effect Size (Cohen's d)",
):
    """Create forest plot of effect sizes.

    Args:
        ax: Matplotlib axes
        effects: Dict mapping names to {"effect": d, "ci_lower": l, "ci_upper": u}
        baseline_name: Name of baseline (will be shown as reference line)
        xlabel: X-axis label
    """
    names = list(effects.keys())
    n = len(names)
    y_pos = np.arange(n)

    effect_sizes = [effects[name]["effect"] for name in names]
    ci_lower = [effects[name].get("ci_lower", effect_sizes[i] - 0.5)
                for i, name in enumerate(names)]
    ci_upper = [effects[name].get("ci_upper", effect_sizes[i] + 0.5)
                for i, name in enumerate(names)]

    # Plot points with CI
    ax.errorbar(effect_sizes, y_pos,
                xerr=[np.array(effect_sizes) - np.array(ci_lower),
                      np.array(ci_upper) - np.array(effect_sizes)],
                fmt="o", color=NATURE_STYLE["colors"]["primary"],
                capsize=NATURE_STYLE["error_capsize"],
                markersize=NATURE_STYLE["marker_size"])

    # Reference line at 0
    ax.axvline(x=0, color="gray", linestyle="--",
               linewidth=NATURE_STYLE["line_width_thin"])

    # Effect size interpretation lines
    for x, label in [(-0.8, "large"), (-0.5, "medium"), (-0.2, "small"),
                     (0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax.axvline(x=x, color="lightgray", linestyle=":",
                   linewidth=NATURE_STYLE["line_width_thin"], alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()


# =============================================================================
# Summary Statistics Table
# =============================================================================

def add_stats_table(
    ax: plt.Axes,
    data: Dict[str, Dict[str, Any]],
    columns: List[str] = ["mean", "std", "p_value"],
    col_labels: List[str] = ["Mean", "SD", "p-value"],
    loc: str = "bottom",
):
    """Add statistics table below plot.

    Args:
        ax: Matplotlib axes
        data: Dict mapping row names to column values
        columns: Column keys in data
        col_labels: Display labels for columns
        loc: Table location
    """
    rows = list(data.keys())
    cell_text = []

    for row in rows:
        row_data = []
        for col in columns:
            val = data[row].get(col, "")
            if isinstance(val, float):
                if col == "p_value":
                    row_data.append(f"{val:.4f}")
                else:
                    row_data.append(f"{val:.3f}")
            else:
                row_data.append(str(val))
        cell_text.append(row_data)

    table = ax.table(cellText=cell_text,
                     rowLabels=rows,
                     colLabels=col_labels,
                     loc=loc,
                     cellLoc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(NATURE_STYLE["font_size_annotation"])
    table.scale(1, 1.2)


# =============================================================================
# Figure Helpers
# =============================================================================

def create_figure(
    n_rows: int = 1,
    n_cols: int = 1,
    size: str = "double_column",
    height_ratios: Optional[List[float]] = None,
    width_ratios: Optional[List[float]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create figure with Nature Methods sizing.

    Args:
        n_rows: Number of subplot rows
        n_cols: Number of subplot columns
        size: "single_column", "double_column", or "full_page"
        height_ratios: Relative heights of rows
        width_ratios: Relative widths of columns

    Returns:
        (figure, axes array)
    """
    setup_nature_style()

    figsize = NATURE_STYLE[size]

    gridspec_kw = {}
    if height_ratios:
        gridspec_kw["height_ratios"] = height_ratios
    if width_ratios:
        gridspec_kw["width_ratios"] = width_ratios

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             gridspec_kw=gridspec_kw if gridspec_kw else None)

    return fig, axes


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    formats: List[str] = ["pdf", "png"],
    dpi: int = 300,
):
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        path: Base path (without extension)
        formats: List of formats to save
        dpi: DPI for raster formats
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), format=fmt, dpi=dpi,
                    bbox_inches="tight", pad_inches=0.05)

    plt.close(fig)


def add_panel_labels(
    axes: Union[plt.Axes, np.ndarray],
    labels: Optional[List[str]] = None,
    fontsize: int = 10,
    fontweight: str = "bold",
    loc: Tuple[float, float] = (-0.15, 1.05),
):
    """Add panel labels (A, B, C, ...) to subplots.

    Args:
        axes: Single axes or array of axes
        labels: Custom labels (default: A, B, C, ...)
        fontsize: Label font size
        fontweight: Label font weight
        loc: (x, y) position relative to axes
    """
    if isinstance(axes, plt.Axes):
        axes = np.array([axes])

    axes_flat = axes.flatten()

    if labels is None:
        labels = [chr(65 + i) for i in range(len(axes_flat))]  # A, B, C, ...

    for ax, label in zip(axes_flat, labels):
        ax.text(loc[0], loc[1], label, transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight,
                va="top", ha="left")
