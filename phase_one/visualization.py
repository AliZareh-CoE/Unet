"""
Visualization Module for Phase 1
================================

Publication-quality figure generation for Nature Methods.

Figures generated:
    1.1: Classical Baselines Summary
        (A) R² by method (bar chart, sorted)
        (B) Per-frequency breakdown (top 3 methods)
        (C) Example predictions vs ground truth
        (D) PSD comparison (actual vs predicted)

    1.2: Statistical Analysis
        (A) Box plot with individual fold data
        (B) Statistical comparison heatmap
        (C) Effect size forest plot
        (D) Assumption check summary
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from numpy.typing import NDArray

# Matplotlib configuration for publication
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .config import NEURAL_BANDS
from .metrics import Phase1Metrics, StatisticalComparison

# Import shared utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    setup_nature_style,
    NATURE_STYLE as UTILS_NATURE_STYLE,
    COLORS_CATEGORICAL,
    add_significance_marker,
    add_significance_stars,
    box_plot_with_points,
    effect_size_forest_plot,
    comparison_heatmap,
    create_figure,
    save_figure,
    add_panel_labels,
    compare_methods,
    compare_multiple_methods,
    fdr_correction,
    holm_correction,
    check_assumptions,
)


# =============================================================================
# Style Configuration
# =============================================================================

# Nature Methods style (local copy for backward compatibility)
NATURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
}

# Color palette (colorblind-friendly)
COLORS = {
    "primary": "#2E86AB",      # Blue
    "secondary": "#A23B72",    # Magenta
    "accent": "#F18F01",       # Orange
    "success": "#2CA58D",      # Teal
    "neutral": "#6B7280",      # Gray
    "highlight": "#E63946",    # Red
}

# Method colors
METHOD_COLORS = {
    "wiener": "#2E86AB",
    "wiener_mimo": "#1B5583",
    "ridge": "#A23B72",
    "ridge_temporal": "#7D2C5A",
    "ridge_cv": "#5C1F42",
    "var": "#F18F01",
    "var_exogenous": "#C77400",
}


def apply_nature_style():
    """Apply Nature Methods style to matplotlib."""
    plt.rcParams.update(NATURE_STYLE)
    setup_nature_style()  # Also apply shared style


# =============================================================================
# Figure Generation Functions
# =============================================================================

class Phase1Visualizer:
    """Generates publication-quality figures for Phase 1.

    Args:
        output_dir: Directory to save figures
        sample_rate: Sampling rate in Hz (for PSD plots)

    Example:
        >>> viz = Phase1Visualizer(output_dir="figures/")
        >>> viz.plot_main_figure(metrics_list, predictions, targets)
    """

    def __init__(
        self,
        output_dir: Path,
        sample_rate: float = 1000.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        apply_nature_style()

    def plot_main_figure(
        self,
        metrics_list: List[Phase1Metrics],
        predictions: Optional[Dict[str, NDArray]] = None,
        ground_truth: Optional[NDArray] = None,
        input_signal: Optional[NDArray] = None,
        filename: str = "figure_1_1_classical_baselines.pdf",
    ) -> Path:
        """Generate main Figure 1.1: Classical Baselines.

        Four panels:
            (A) R² by method (sorted bar chart with CI)
            (B) Per-frequency R² breakdown (heatmap)
            (C) Example time series (prediction vs truth)
            (D) PSD comparison

        Args:
            metrics_list: List of Phase1Metrics objects
            predictions: Dict of method name -> predictions [N, C, T]
            ground_truth: Ground truth signals [N, C, T]
            input_signal: Input signals [N, C, T]
            filename: Output filename

        Returns:
            Path to saved figure
        """
        # Create figure with 2x2 layout
        fig = plt.figure(figsize=(7.2, 6.0))  # Nature: max 7.2 inches wide
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

        # (A) R² by method
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_r2_bars(ax_a, metrics_list)

        # (B) Per-frequency R² heatmap
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_band_heatmap(ax_b, metrics_list)

        # (C) Example predictions
        ax_c = fig.add_subplot(gs[1, 0])
        if predictions and ground_truth is not None:
            best_method = max(metrics_list, key=lambda m: m.r2_mean).method
            self._plot_example_prediction(
                ax_c,
                predictions.get(best_method),
                ground_truth,
                best_method,
            )
        else:
            ax_c.text(0.5, 0.5, "No predictions available",
                     ha='center', va='center', transform=ax_c.transAxes)
            ax_c.set_title("(C) Example Prediction")

        # (D) PSD comparison
        ax_d = fig.add_subplot(gs[1, 1])
        if predictions and ground_truth is not None:
            best_method = max(metrics_list, key=lambda m: m.r2_mean).method
            self._plot_psd_comparison(
                ax_d,
                predictions.get(best_method),
                ground_truth,
                best_method,
            )
        else:
            ax_d.text(0.5, 0.5, "No predictions available",
                     ha='center', va='center', transform=ax_d.transAxes)
            ax_d.set_title("(D) PSD Comparison")

        # Add panel labels
        for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ['A', 'B', 'C', 'D']):
            ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='top')

        # Save
        output_path = self.output_dir / filename
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

        # Also save PNG for quick viewing
        png_path = output_path.with_suffix('.png')
        fig = plt.figure(figsize=(7.2, 6.0))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_r2_bars(ax_a, metrics_list)
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_band_heatmap(ax_b, metrics_list)
        ax_c = fig.add_subplot(gs[1, 0])
        if predictions and ground_truth is not None:
            best_method = max(metrics_list, key=lambda m: m.r2_mean).method
            self._plot_example_prediction(ax_c, predictions.get(best_method), ground_truth, best_method)
        else:
            ax_c.text(0.5, 0.5, "No predictions available", ha='center', va='center', transform=ax_c.transAxes)
        ax_d = fig.add_subplot(gs[1, 1])
        if predictions and ground_truth is not None:
            self._plot_psd_comparison(ax_d, predictions.get(best_method), ground_truth, best_method)
        else:
            ax_d.text(0.5, 0.5, "No predictions available", ha='center', va='center', transform=ax_d.transAxes)
        for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ['A', 'B', 'C', 'D']):
            ax.text(-0.15, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
        fig.savefig(png_path, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def _plot_r2_bars(self, ax: plt.Axes, metrics_list: List[Phase1Metrics]):
        """Plot R² bar chart sorted by performance."""
        # Sort by R²
        sorted_metrics = sorted(metrics_list, key=lambda m: m.r2_mean, reverse=True)

        methods = [m.method for m in sorted_metrics]
        r2_means = [m.r2_mean for m in sorted_metrics]
        r2_ci_low = [m.r2_ci[0] for m in sorted_metrics]
        r2_ci_high = [m.r2_ci[1] for m in sorted_metrics]

        # Error bars
        yerr_low = [m - l for m, l in zip(r2_means, r2_ci_low)]
        yerr_high = [h - m for m, h in zip(r2_means, r2_ci_high)]

        x = np.arange(len(methods))
        colors = [METHOD_COLORS.get(m, COLORS["neutral"]) for m in methods]

        bars = ax.bar(x, r2_means, color=colors, edgecolor='white', linewidth=0.5)
        ax.errorbar(x, r2_means, yerr=[yerr_low, yerr_high],
                   fmt='none', color='black', capsize=3, linewidth=1)

        # Labels
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=0)
        ax.set_ylabel('R²')
        ax.set_title('(A) R² by Method')
        ax.set_ylim(0, max(r2_means) * 1.15)

        # Add value labels on bars
        for bar, val in zip(bars, r2_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=6)

    def _plot_band_heatmap(self, ax: plt.Axes, metrics_list: List[Phase1Metrics]):
        """Plot per-frequency-band R² as heatmap."""
        # Sort by R²
        sorted_metrics = sorted(metrics_list, key=lambda m: m.r2_mean, reverse=True)

        bands = list(NEURAL_BANDS.keys())
        methods = [m.method for m in sorted_metrics]

        # Build matrix
        data = np.zeros((len(methods), len(bands)))
        for i, m in enumerate(sorted_metrics):
            for j, band in enumerate(bands):
                data[i, j] = m.band_r2.get(band, np.nan)

        # Plot heatmap
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=max(0.5, np.nanmax(data)))

        # Labels
        ax.set_xticks(np.arange(len(bands)))
        ax.set_xticklabels([b.capitalize() for b in bands], rotation=45, ha='right')
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels([m.replace('_', '\n') for m in methods])
        ax.set_title('(B) Per-Band R²')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('R²', fontsize=7)

        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(bands)):
                val = data[i, j]
                if not np.isnan(val):
                    color = 'white' if val > 0.3 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=5, color=color)

    def _plot_example_prediction(
        self,
        ax: plt.Axes,
        y_pred: Optional[NDArray],
        y_true: NDArray,
        method: str,
    ):
        """Plot example prediction vs ground truth."""
        if y_pred is None:
            ax.text(0.5, 0.5, "No prediction data", ha='center', va='center')
            return

        # Select first sample, first channel
        sample_idx = 0
        channel_idx = 0

        pred = y_pred[sample_idx, channel_idx, :]
        true = y_true[sample_idx, channel_idx, :]

        # Time axis
        t = np.arange(len(pred)) / self.sample_rate * 1000  # ms

        # Plot subset for clarity
        t_max = min(500, len(t))  # Show 500ms
        ax.plot(t[:t_max], true[:t_max], label='Ground truth',
               color=COLORS["neutral"], linewidth=1.0, alpha=0.8)
        ax.plot(t[:t_max], pred[:t_max], label=method,
               color=COLORS["primary"], linewidth=0.8)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (a.u.)')
        ax.set_title(f'(C) Example: {method}')
        ax.legend(loc='upper right', frameon=False)

    def _plot_psd_comparison(
        self,
        ax: plt.Axes,
        y_pred: Optional[NDArray],
        y_true: NDArray,
        method: str,
    ):
        """Plot power spectral density comparison."""
        from scipy import signal as scipy_signal

        if y_pred is None:
            ax.text(0.5, 0.5, "No prediction data", ha='center', va='center')
            return

        # Average over all samples and channels
        pred_mean = np.mean(y_pred, axis=(0, 1))
        true_mean = np.mean(y_true, axis=(0, 1))

        nperseg = min(256, len(pred_mean))
        freqs_pred, psd_pred = scipy_signal.welch(pred_mean, fs=self.sample_rate, nperseg=nperseg)
        freqs_true, psd_true = scipy_signal.welch(true_mean, fs=self.sample_rate, nperseg=nperseg)

        # Plot in log scale
        ax.semilogy(freqs_true, psd_true, label='Ground truth',
                   color=COLORS["neutral"], linewidth=1.0)
        ax.semilogy(freqs_pred, psd_pred, label=method,
                   color=COLORS["primary"], linewidth=0.8, linestyle='--')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.set_title('(D) PSD Comparison')
        ax.legend(loc='upper right', frameon=False)
        ax.set_xlim(0, 100)

        # Add frequency band annotations
        for band, (f_low, f_high) in NEURAL_BANDS.items():
            if f_high <= 100:
                ax.axvspan(f_low, f_high, alpha=0.1, color=COLORS["accent"])

    def plot_statistical_summary(
        self,
        metrics_list: List[Phase1Metrics],
        filename: str = "figure_1_1_stats.pdf",
    ) -> Path:
        """Generate statistical summary figure.

        Args:
            metrics_list: List of Phase1Metrics
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(5, 4))

        # Sort by R²
        sorted_metrics = sorted(metrics_list, key=lambda m: m.r2_mean, reverse=True)
        methods = [m.method for m in sorted_metrics]
        r2_means = [m.r2_mean for m in sorted_metrics]

        y_pos = np.arange(len(methods))

        # Horizontal bar chart
        colors = [METHOD_COLORS.get(m, COLORS["neutral"]) for m in methods]
        bars = ax.barh(y_pos, r2_means, color=colors, edgecolor='white')

        # Add CI
        for i, m in enumerate(sorted_metrics):
            ax.plot([m.r2_ci[0], m.r2_ci[1]], [i, i], 'k-', linewidth=2)
            ax.plot(m.r2_ci[0], i, 'k|', markersize=6)
            ax.plot(m.r2_ci[1], i, 'k|', markersize=6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods)
        ax.set_xlabel('R² (95% CI)')
        ax.set_title('Classical Baselines Performance')
        ax.invert_yaxis()

        output_path = self.output_dir / filename
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

        return output_path

    def plot_comprehensive_stats_figure(
        self,
        metrics_list: List[Phase1Metrics],
        comparisons: Optional[List[StatisticalComparison]] = None,
        filename: str = "figure_1_2_statistical_analysis.pdf",
    ) -> Path:
        """Generate comprehensive statistical analysis figure.

        Four panels:
            (A) Box plot with individual fold data points
            (B) Statistical comparison heatmap (p-values)
            (C) Effect size forest plot
            (D) Metrics comparison table

        Args:
            metrics_list: List of Phase1Metrics objects
            comparisons: List of StatisticalComparison objects
            filename: Output filename

        Returns:
            Path to saved figure
        """
        apply_nature_style()

        # Create figure with 2x2 layout
        fig = plt.figure(figsize=(7.2, 6.5))
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.4)

        sorted_metrics = sorted(metrics_list, key=lambda m: m.r2_mean, reverse=True)
        methods = [m.method for m in sorted_metrics]

        # (A) Box plot with individual points
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_boxplot_with_points(ax_a, sorted_metrics)

        # (B) Statistical comparison heatmap
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_comparison_heatmap(ax_b, sorted_metrics, comparisons)

        # (C) Effect size forest plot
        ax_c = fig.add_subplot(gs[1, 0])
        self._plot_effect_sizes(ax_c, sorted_metrics, comparisons)

        # (D) Summary metrics table
        ax_d = fig.add_subplot(gs[1, 1])
        self._plot_metrics_summary_table(ax_d, sorted_metrics)

        # Add panel labels
        for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ['A', 'B', 'C', 'D']):
            ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='top')

        # Save in multiple formats
        output_path = self.output_dir / filename
        fig.savefig(output_path, format='pdf', bbox_inches='tight')

        # Also save PNG
        png_path = output_path.with_suffix('.png')
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def _plot_boxplot_with_points(
        self,
        ax: plt.Axes,
        metrics_list: List[Phase1Metrics],
    ):
        """Plot box plot with individual fold R² values."""
        methods = [m.method for m in metrics_list]
        fold_data = {m.method: m.fold_r2s for m in metrics_list if m.fold_r2s}

        if not fold_data:
            ax.text(0.5, 0.5, "No fold data available",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("(A) R² Distribution Across Folds")
            return

        # Create box plot
        positions = list(range(len(methods)))
        valid_data = [fold_data.get(m, []) for m in methods]

        bp = ax.boxplot([d for d in valid_data if d], positions=[i for i, d in enumerate(valid_data) if d],
                        patch_artist=True, widths=0.6, showfliers=False)

        # Color boxes
        colors = [METHOD_COLORS.get(m, COLORS["neutral"]) for m in methods if fold_data.get(m)]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')

        # Overlay individual points
        for i, m in enumerate(methods):
            if m in fold_data and fold_data[m]:
                points = fold_data[m]
                jitter = np.random.uniform(-0.12, 0.12, len(points))
                ax.scatter([i + j for j in jitter], points, color='black',
                          s=15, alpha=0.6, zorder=3)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=0, fontsize=6)
        ax.set_ylabel('R²')
        ax.set_title('(A) R² Distribution Across Folds')

        # Add mean markers
        for i, m in enumerate(metrics_list):
            ax.plot(i, m.r2_mean, 'r_', markersize=12, markeredgewidth=2)

    def _plot_comparison_heatmap(
        self,
        ax: plt.Axes,
        metrics_list: List[Phase1Metrics],
        comparisons: Optional[List[StatisticalComparison]] = None,
    ):
        """Plot pairwise statistical comparison heatmap."""
        methods = [m.method for m in metrics_list]
        n_methods = len(methods)

        # Build p-value matrix
        p_matrix = np.ones((n_methods, n_methods))

        if comparisons:
            for c in comparisons:
                if c.method1 in methods and c.method2 in methods:
                    i = methods.index(c.method1)
                    j = methods.index(c.method2)
                    p_matrix[i, j] = c.p_adjusted
                    p_matrix[j, i] = c.p_adjusted
        else:
            # Compute from fold data
            for i, m1 in enumerate(metrics_list):
                for j, m2 in enumerate(metrics_list):
                    if i < j and m1.fold_r2s and m2.fold_r2s:
                        comp = compare_methods(
                            np.array(m1.fold_r2s),
                            np.array(m2.fold_r2s),
                            m1.method, m2.method,
                            paired=True
                        )
                        p_matrix[i, j] = comp.parametric_test.p_value
                        p_matrix[j, i] = comp.parametric_test.p_value

        # Plot heatmap
        im = ax.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)

        ax.set_xticks(range(n_methods))
        ax.set_yticks(range(n_methods))
        ax.set_xticklabels([m[:8] for m in methods], rotation=45, ha='right', fontsize=6)
        ax.set_yticklabels([m[:8] for m in methods], fontsize=6)
        ax.set_title('(B) Pairwise p-values')

        # Add significance markers
        for i in range(n_methods):
            for j in range(n_methods):
                if i != j:
                    p = p_matrix[i, j]
                    if p < 0.001:
                        text = "***"
                    elif p < 0.01:
                        text = "**"
                    elif p < 0.05:
                        text = "*"
                    else:
                        text = ""
                    ax.text(j, i, text, ha='center', va='center', fontsize=6, color='white' if p < 0.05 else 'black')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('p-value', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    def _plot_effect_sizes(
        self,
        ax: plt.Axes,
        metrics_list: List[Phase1Metrics],
        comparisons: Optional[List[StatisticalComparison]] = None,
    ):
        """Plot effect size forest plot vs best method."""
        if len(metrics_list) < 2:
            ax.text(0.5, 0.5, "Need multiple methods for comparison",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("(C) Effect Sizes")
            return

        best = metrics_list[0]  # Already sorted
        other_methods = metrics_list[1:]

        effect_data = {}
        for m in other_methods:
            if comparisons:
                # Find comparison
                comp = next((c for c in comparisons
                           if (c.method1 == best.method and c.method2 == m.method) or
                              (c.method2 == best.method and c.method1 == m.method)), None)
                if comp:
                    d = comp.cohens_d if comp.method1 == best.method else -comp.cohens_d
                    effect_data[m.method] = {"effect": d, "interp": comp.effect_size}
            elif best.fold_r2s and m.fold_r2s:
                comp = compare_methods(
                    np.array(best.fold_r2s),
                    np.array(m.fold_r2s),
                    best.method, m.method, paired=True
                )
                d = comp.parametric_test.effect_size or 0
                effect_data[m.method] = {"effect": d, "interp": self._interpret_effect(d)}

        if not effect_data:
            ax.text(0.5, 0.5, "No effect size data available",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("(C) Effect Sizes")
            return

        # Plot forest plot
        methods = list(effect_data.keys())
        effects = [effect_data[m]["effect"] for m in methods]
        y_pos = np.arange(len(methods))

        colors = ['#2E86AB' if e > 0 else '#A23B72' for e in effects]
        ax.barh(y_pos, effects, color=colors, height=0.6, alpha=0.8)

        # Reference lines for effect size interpretation
        for x in [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]:
            ax.axvline(x=x, color='lightgray', linestyle=':', linewidth=0.5, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([m[:10] for m in methods], fontsize=6)
        ax.set_xlabel(f"Cohen's d (vs {best.method})")
        ax.set_title("(C) Effect Sizes vs Best Method")
        ax.invert_yaxis()

        # Add interpretation labels
        ax.text(0.02, 0.98, "← Best wins | Other wins →", transform=ax.transAxes,
               fontsize=5, va='top', ha='left', style='italic')

    def _interpret_effect(self, d: float) -> str:
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

    def _plot_metrics_summary_table(
        self,
        ax: plt.Axes,
        metrics_list: List[Phase1Metrics],
    ):
        """Plot comprehensive metrics summary table."""
        ax.axis('off')

        # Table data
        columns = ['Method', 'R²', '95% CI', 'Pearson', 'MAE', 'PSD Err']
        cell_data = []

        for m in metrics_list[:6]:  # Top 6 methods
            ci_str = f"[{m.r2_ci[0]:.3f}, {m.r2_ci[1]:.3f}]"
            row = [
                m.method[:12],
                f"{m.r2_mean:.4f}",
                ci_str,
                f"{m.pearson_mean:.4f}",
                f"{m.mae_mean:.4f}",
                f"{m.psd_error_db:.2f}"
            ]
            cell_data.append(row)

        table = ax.table(
            cellText=cell_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
        )

        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1.2, 1.4)

        # Style header row
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#E6E6E6')
            table[(0, i)].set_text_props(weight='bold')

        ax.set_title('(D) Performance Summary', pad=10)


def create_summary_table(metrics_list: List[Phase1Metrics]) -> str:
    """Create a LaTeX-ready summary table.

    Args:
        metrics_list: List of Phase1Metrics

    Returns:
        LaTeX table string
    """
    sorted_metrics = sorted(metrics_list, key=lambda m: m.r2_mean, reverse=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Phase 1: Classical Baselines Performance}",
        r"\label{tab:classical_baselines}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & R² & 95\% CI & Pearson & PSD Error (dB) \\",
        r"\midrule",
    ]

    for m in sorted_metrics:
        ci_str = f"[{m.r2_ci[0]:.3f}, {m.r2_ci[1]:.3f}]"
        line = f"{m.method} & {m.r2_mean:.4f} & {ci_str} & {m.pearson_mean:.4f} & {m.psd_error_db:.2f} \\\\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)
