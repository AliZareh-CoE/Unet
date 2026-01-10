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
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Matplotlib configuration for publication
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .config import NEURAL_BANDS
from .metrics import Phase1Metrics


# =============================================================================
# Style Configuration
# =============================================================================

# Nature Methods style
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
