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
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Matplotlib configuration for publication
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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
    "performer": "#7D2C5A",
    "mamba": "#F18F01",
    "condunet": "#2CA58D",
}


def apply_nature_style():
    """Apply Nature Methods style to matplotlib."""
    plt.rcParams.update(NATURE_STYLE)


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
