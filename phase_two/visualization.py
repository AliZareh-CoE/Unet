"""
Visualization Module for Phase 2
================================

Publication-quality figure generation for neural architecture comparison.

Figures generated:
    2.1: Architecture Comparison
        (A) R² by architecture (bar chart with error bars)
        (B) Training curves (loss over epochs)
        (C) R² progression over epochs
        (D) Parameter efficiency (R² vs params)

    2.2: Statistical Analysis
        (A) Box plot with individual fold data
        (B) Statistical comparison heatmap (p-values)
        (C) Effect size forest plot
        (D) Summary metrics table
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

import numpy as np

# Matplotlib configuration for publication
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared statistical utilities
from utils import (
    setup_nature_style,
    COLORS_CATEGORICAL,
    compare_methods,
    compare_multiple_methods,
    holm_correction,
    fdr_correction,
    check_assumptions,
    format_mean_ci,
    confidence_interval,
)


# =============================================================================
# Style Configuration
# =============================================================================

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
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#2CA58D",
    "neutral": "#6B7280",
    "highlight": "#E63946",
}

# Architecture colors
ARCH_COLORS = {
    "linear": "#6B7280",
    "simplecnn": "#9CA3AF",
    "wavenet": "#2E86AB",
    "fnet": "#1B5583",
    "vit": "#A23B72",
    "condunet": "#2CA58D",
}


def apply_nature_style():
    """Apply Nature Methods style to matplotlib."""
    plt.rcParams.update(NATURE_STYLE)
    setup_nature_style()  # Also apply shared style


# =============================================================================
# Phase 2 Visualizer
# =============================================================================

class Phase2Visualizer:
    """Generates publication-quality figures for Phase 2.

    Args:
        output_dir: Directory to save figures
        dpi: Figure resolution

    Example:
        >>> viz = Phase2Visualizer(output_dir="figures/")
        >>> viz.plot_main_figure(result)
    """

    def __init__(
        self,
        output_dir: Path,
        dpi: int = 300,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        apply_nature_style()

    def plot_main_figure(
        self,
        result: Any,  # Phase2Result
        filename: str = "figure_2_1_architecture_comparison.pdf",
        show_gate: bool = True,
    ) -> Path:
        """Generate main Figure 2.1: Architecture Comparison.

        Four panels:
            (A) R² by architecture (sorted bar chart with CI)
            (B) Learning curves (validation loss)
            (C) R² progression over epochs
            (D) Parameter efficiency scatter

        Args:
            result: Phase2Result object
            filename: Output filename
            show_gate: Whether to show gate threshold line

        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(7.2, 6.0))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

        # (A) R² by architecture
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_r2_comparison(ax_a, result, show_gate)

        # (B) Learning curves
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_learning_curves(ax_b, result)

        # (C) R² progression
        ax_c = fig.add_subplot(gs[1, 0])
        self._plot_r2_progression(ax_c, result)

        # (D) Parameter efficiency
        ax_d = fig.add_subplot(gs[1, 1])
        self._plot_parameter_efficiency(ax_d, result)

        # Add panel labels
        for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ['A', 'B', 'C', 'D']):
            ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='top')

        # Save
        output_path = self.output_dir / filename
        fig.savefig(output_path, format=filename.split('.')[-1],
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def _plot_r2_comparison(self, ax: plt.Axes, result: Any, show_gate: bool):
        """Plot R² bar chart sorted by performance."""
        aggregated = result.aggregated

        # Sort by R²
        sorted_archs = sorted(aggregated.keys(),
                             key=lambda k: aggregated[k]["r2_mean"],
                             reverse=True)

        archs = sorted_archs
        r2_means = [aggregated[a]["r2_mean"] for a in archs]
        r2_stds = [aggregated[a]["r2_std"] for a in archs]

        x = np.arange(len(archs))
        colors = [ARCH_COLORS.get(a, COLORS["neutral"]) for a in archs]

        bars = ax.bar(x, r2_means, color=colors, edgecolor='white', linewidth=0.5)
        ax.errorbar(x, r2_means, yerr=r2_stds,
                   fmt='none', color='black', capsize=3, linewidth=1)

        # Gate threshold line
        if show_gate:
            gate = result.classical_baseline_r2 + 0.10
            ax.axhline(y=gate, color=COLORS["highlight"], linestyle='--',
                      linewidth=1, label=f'Gate: {gate:.3f}')
            ax.axhline(y=result.classical_baseline_r2, color=COLORS["neutral"],
                      linestyle=':', linewidth=1, label=f'Classical: {result.classical_baseline_r2:.3f}')

        ax.set_xticks(x)
        ax.set_xticklabels([a.replace('_', '\n') for a in archs], rotation=45, ha='right')
        ax.set_ylabel('R²')
        ax.set_title('(A) R² by Architecture')
        ax.set_ylim(0, max(r2_means) * 1.15)
        ax.legend(loc='upper right', frameon=False, fontsize=6)

        # Mark best
        best_idx = archs.index(result.best_architecture)
        bars[best_idx].set_edgecolor(COLORS["highlight"])
        bars[best_idx].set_linewidth(2)

    def _plot_learning_curves(self, ax: plt.Axes, result: Any):
        """Plot validation loss curves."""
        # Group results by architecture, use first fold
        arch_results = {}
        for r in result.results:
            if r.architecture not in arch_results:
                arch_results[r.architecture] = r

        for arch, r in arch_results.items():
            color = ARCH_COLORS.get(arch, COLORS["neutral"])
            epochs = np.arange(1, len(r.val_losses) + 1)
            ax.plot(epochs, r.val_losses, label=arch, color=color, linewidth=1)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('(B) Learning Curves')
        ax.legend(loc='upper right', frameon=False, fontsize=5, ncol=2)

    def _plot_r2_progression(self, ax: plt.Axes, result: Any):
        """Plot R² over training epochs."""
        arch_results = {}
        for r in result.results:
            if r.architecture not in arch_results:
                arch_results[r.architecture] = r

        for arch, r in arch_results.items():
            color = ARCH_COLORS.get(arch, COLORS["neutral"])
            epochs = np.arange(1, len(r.val_r2s) + 1)
            ax.plot(epochs, r.val_r2s, label=arch, color=color, linewidth=1)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation R²')
        ax.set_title('(C) R² Progression')
        ax.legend(loc='lower right', frameon=False, fontsize=5, ncol=2)

    def _plot_parameter_efficiency(self, ax: plt.Axes, result: Any):
        """Plot R² vs number of parameters."""
        aggregated = result.aggregated

        for r in result.results:
            arch = r.architecture
            if arch in aggregated:
                r2 = aggregated[arch]["r2_mean"]
                params = r.n_parameters / 1e6  # Millions
                color = ARCH_COLORS.get(arch, COLORS["neutral"])
                ax.scatter(params, r2, c=color, s=50, label=arch, edgecolor='white', linewidth=0.5)

        # Only show one label per architecture
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                 loc='lower right', frameon=False, fontsize=5, ncol=2)

        ax.set_xlabel('Parameters (M)')
        ax.set_ylabel('R²')
        ax.set_title('(D) Parameter Efficiency')

    def plot_training_summary(
        self,
        result: Any,
        filename: str = "figure_2_1_training_summary.pdf",
    ) -> Path:
        """Plot training time and epochs summary."""
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

        aggregated = result.aggregated
        sorted_archs = sorted(aggregated.keys(),
                             key=lambda k: aggregated[k]["r2_mean"],
                             reverse=True)

        # Gather training times
        arch_times = {}
        arch_epochs = {}
        for r in result.results:
            if r.architecture not in arch_times:
                arch_times[r.architecture] = []
                arch_epochs[r.architecture] = []
            arch_times[r.architecture].append(r.total_time / 60)  # minutes
            arch_epochs[r.architecture].append(r.epochs_trained)

        # (A) Training time
        ax = axes[0]
        times = [np.mean(arch_times.get(a, [0])) for a in sorted_archs]
        colors = [ARCH_COLORS.get(a, COLORS["neutral"]) for a in sorted_archs]
        ax.barh(np.arange(len(sorted_archs)), times, color=colors)
        ax.set_yticks(np.arange(len(sorted_archs)))
        ax.set_yticklabels(sorted_archs)
        ax.set_xlabel('Training Time (min)')
        ax.set_title('Training Time')
        ax.invert_yaxis()

        # (B) Epochs trained
        ax = axes[1]
        epochs = [np.mean(arch_epochs.get(a, [0])) for a in sorted_archs]
        ax.barh(np.arange(len(sorted_archs)), epochs, color=colors)
        ax.set_yticks(np.arange(len(sorted_archs)))
        ax.set_yticklabels(sorted_archs)
        ax.set_xlabel('Epochs Trained')
        ax.set_title('Epochs (Early Stopping)')
        ax.invert_yaxis()

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(output_path, format=filename.split('.')[-1],
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def plot_comprehensive_stats_figure(
        self,
        result: Any,
        filename: str = "figure_2_2_statistical_analysis.pdf",
    ) -> Path:
        """Generate comprehensive statistical analysis figure.

        Four panels:
            (A) Box plot with individual fold data points
            (B) Statistical comparison heatmap (p-values)
            (C) Effect size forest plot (vs best architecture)
            (D) Summary metrics table

        Args:
            result: Phase2Result object
            filename: Output filename

        Returns:
            Path to saved figure
        """
        apply_nature_style()

        fig = plt.figure(figsize=(7.2, 6.5))
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.4)

        aggregated = result.aggregated
        sorted_archs = sorted(aggregated.keys(),
                             key=lambda k: aggregated[k]["r2_mean"],
                             reverse=True)

        # (A) Box plot with individual fold data
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_boxplot_with_points(ax_a, result, sorted_archs)

        # (B) Statistical comparison heatmap
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_comparison_heatmap(ax_b, result, sorted_archs)

        # (C) Effect size forest plot
        ax_c = fig.add_subplot(gs[1, 0])
        self._plot_effect_sizes(ax_c, result, sorted_archs)

        # (D) Summary metrics table
        ax_d = fig.add_subplot(gs[1, 1])
        self._plot_summary_table(ax_d, result, sorted_archs)

        # Add panel labels
        for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ['A', 'B', 'C', 'D']):
            ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='top')

        # Save in multiple formats
        output_path = self.output_dir / filename
        fig.savefig(output_path, format='pdf', bbox_inches='tight')

        png_path = output_path.with_suffix('.png')
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def _plot_boxplot_with_points(
        self,
        ax: plt.Axes,
        result: Any,
        sorted_archs: List[str],
    ):
        """Plot box plot with individual fold R² values."""
        aggregated = result.aggregated

        fold_data = []
        for arch in sorted_archs:
            r2s = aggregated[arch].get("fold_r2s", [])
            fold_data.append(r2s if r2s else [aggregated[arch]["r2_mean"]])

        # Create box plot
        bp = ax.boxplot(fold_data, patch_artist=True, widths=0.6, showfliers=False)

        # Color boxes
        for i, (patch, arch) in enumerate(zip(bp["boxes"], sorted_archs)):
            color = ARCH_COLORS.get(arch, COLORS["neutral"])
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')

        # Overlay individual points
        for i, (arch, data) in enumerate(zip(sorted_archs, fold_data)):
            if len(data) > 1:
                jitter = np.random.uniform(-0.12, 0.12, len(data))
                ax.scatter([i + 1 + j for j in jitter], data, color='black',
                          s=15, alpha=0.6, zorder=3)

        ax.set_xticklabels([a[:8] for a in sorted_archs], rotation=45, ha='right', fontsize=6)
        ax.set_ylabel('R²')
        ax.set_title('(A) R² Distribution Across Folds')

        # Add mean markers
        for i, arch in enumerate(sorted_archs):
            mean = aggregated[arch]["r2_mean"]
            ax.plot(i + 1, mean, 'r_', markersize=10, markeredgewidth=2)

    def _plot_comparison_heatmap(
        self,
        ax: plt.Axes,
        result: Any,
        sorted_archs: List[str],
    ):
        """Plot pairwise statistical comparison heatmap."""
        aggregated = result.aggregated
        n_archs = len(sorted_archs)

        # Build p-value matrix
        p_matrix = np.ones((n_archs, n_archs))

        for i, arch1 in enumerate(sorted_archs):
            for j, arch2 in enumerate(sorted_archs):
                if i < j:
                    r2s_1 = aggregated[arch1].get("fold_r2s", [])
                    r2s_2 = aggregated[arch2].get("fold_r2s", [])

                    if len(r2s_1) >= 2 and len(r2s_2) >= 2:
                        comp = compare_methods(
                            np.array(r2s_1), np.array(r2s_2),
                            arch1, arch2, paired=True
                        )
                        p_matrix[i, j] = comp.parametric_test.p_value
                        p_matrix[j, i] = comp.parametric_test.p_value

        # Plot heatmap
        im = ax.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)

        ax.set_xticks(range(n_archs))
        ax.set_yticks(range(n_archs))
        ax.set_xticklabels([a[:6] for a in sorted_archs], rotation=45, ha='right', fontsize=5)
        ax.set_yticklabels([a[:6] for a in sorted_archs], fontsize=5)
        ax.set_title('(B) Pairwise p-values')

        # Add significance markers
        for i in range(n_archs):
            for j in range(n_archs):
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
                    ax.text(j, i, text, ha='center', va='center', fontsize=5,
                           color='white' if p < 0.05 else 'black')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('p-value', fontsize=6)
        cbar.ax.tick_params(labelsize=5)

    def _plot_effect_sizes(
        self,
        ax: plt.Axes,
        result: Any,
        sorted_archs: List[str],
    ):
        """Plot effect size forest plot vs best architecture."""
        aggregated = result.aggregated

        if len(sorted_archs) < 2:
            ax.text(0.5, 0.5, "Need multiple architectures",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("(C) Effect Sizes")
            return

        best_arch = sorted_archs[0]
        best_r2s = aggregated[best_arch].get("fold_r2s", [])
        other_archs = sorted_archs[1:]

        effect_data = {}
        for arch in other_archs:
            r2s = aggregated[arch].get("fold_r2s", [])
            if len(best_r2s) >= 2 and len(r2s) >= 2:
                comp = compare_methods(
                    np.array(best_r2s), np.array(r2s),
                    best_arch, arch, paired=True
                )
                d = comp.parametric_test.effect_size or 0
                effect_data[arch] = d

        if not effect_data:
            ax.text(0.5, 0.5, "Insufficient fold data",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("(C) Effect Sizes")
            return

        # Plot forest plot
        archs = list(effect_data.keys())
        effects = [effect_data[a] for a in archs]
        y_pos = np.arange(len(archs))

        colors = [COLORS["primary"] if e > 0 else COLORS["secondary"] for e in effects]
        ax.barh(y_pos, effects, color=colors, height=0.6, alpha=0.8)

        # Reference lines
        for x in [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]:
            ax.axvline(x=x, color='lightgray', linestyle=':', linewidth=0.5, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([a[:10] for a in archs], fontsize=6)
        ax.set_xlabel(f"Cohen's d (vs {best_arch})")
        ax.set_title("(C) Effect Sizes vs Best")
        ax.invert_yaxis()

    def _plot_summary_table(
        self,
        ax: plt.Axes,
        result: Any,
        sorted_archs: List[str],
    ):
        """Plot comprehensive summary table."""
        ax.axis('off')

        aggregated = result.aggregated
        gate_threshold = result.classical_baseline_r2 + 0.10

        # Get param counts
        arch_params = {}
        for r in result.results:
            if r.architecture not in arch_params:
                arch_params[r.architecture] = r.n_parameters

        # Table data
        columns = ['Arch', 'R² Mean', '± Std', 'Params', 'Gate']
        cell_data = []

        for arch in sorted_archs[:6]:  # Top 6
            stats = aggregated[arch]
            params = arch_params.get(arch, 0)
            passed = "Pass" if stats["r2_mean"] >= gate_threshold else "Fail"

            # Compute CI if fold data available
            fold_r2s = stats.get("fold_r2s", [])
            if len(fold_r2s) >= 2:
                ci = confidence_interval(np.array(fold_r2s))
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            else:
                ci_str = f"± {stats['r2_std']:.3f}"

            row = [
                arch[:10],
                f"{stats['r2_mean']:.4f}",
                ci_str,
                f"{params/1e6:.1f}M",
                passed,
            ]
            cell_data.append(row)

        table = ax.table(
            cellText=cell_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
        )

        table.auto_set_font_size(False)
        table.set_fontsize(5)
        table.scale(1.2, 1.4)

        # Style header row
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#E6E6E6')
            table[(0, i)].set_text_props(weight='bold')

        ax.set_title('(D) Performance Summary', pad=10)

    def plot_all(self, result: Any, format: str = "pdf") -> List[Path]:
        """Generate all Phase 2 figures.

        Args:
            result: Phase2Result object
            format: Output format (pdf, png, svg)

        Returns:
            List of paths to saved figures
        """
        paths = []

        paths.append(self.plot_main_figure(
            result,
            filename=f"figure_2_1_architecture_comparison.{format}",
        ))

        paths.append(self.plot_training_summary(
            result,
            filename=f"figure_2_1_training_summary.{format}",
        ))

        paths.append(self.plot_comprehensive_stats_figure(
            result,
            filename=f"figure_2_2_statistical_analysis.{format}",
        ))

        return paths


def create_summary_table(result: Any) -> str:
    """Create a LaTeX-ready summary table.

    Args:
        result: Phase2Result object

    Returns:
        LaTeX table string
    """
    aggregated = result.aggregated
    sorted_archs = sorted(aggregated.keys(),
                         key=lambda k: aggregated[k]["r2_mean"],
                         reverse=True)

    # Get parameter counts
    arch_params = {}
    for r in result.results:
        if r.architecture not in arch_params:
            arch_params[r.architecture] = r.n_parameters

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Phase 2: Neural Architecture Comparison}",
        r"\label{tab:architecture_comparison}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Architecture & R² Mean & R² Std & Parameters & Gate \\",
        r"\midrule",
    ]

    gate_threshold = result.classical_baseline_r2 + 0.10
    for arch in sorted_archs:
        stats = aggregated[arch]
        params = arch_params.get(arch, 0)
        passed = "Pass" if stats["r2_mean"] >= gate_threshold else "Fail"
        line = f"{arch} & {stats['r2_mean']:.4f} & {stats['r2_std']:.4f} & {params:,} & {passed} \\\\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)
