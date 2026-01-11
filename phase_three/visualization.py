"""
Visualization Module for Phase 3
================================

Publication-quality figure generation for CondUNet ablation studies.

Figures generated:
    3.1: Ablation Results
        (A) Attention ablation: R² by type
        (B) Loss ablation: R² by loss function
        (C) Waterfall: contribution of each component
        (D) Final optimal configuration summary

    3.2: Statistical Analysis
        (A) Box plots with fold data by study
        (B) Effect sizes for ablations
        (C) Statistical significance table
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

# Add project root for shared utilities
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    setup_nature_style,
    COLORS_CATEGORICAL,
    compare_methods,
    holm_correction,
    fdr_correction,
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
    "baseline": "#1B5583",
}

# Study colors
STUDY_COLORS = {
    "attention": "#2E86AB",
    "conditioning": "#A23B72",
    "loss": "#F18F01",
    "capacity": "#2CA58D",
    "bidirectional": "#6B7280",
    "augmentation": "#7C3AED",
}

# Variant colors for each study
VARIANT_COLORS = {
    "attention": {
        "none": "#E5E7EB",
        "basic": "#9CA3AF",
        "cross_freq": "#60A5FA",
        "cross_freq_v2": "#2563EB",
    },
    "conditioning": {
        "none": "#E5E7EB",
        "odor": "#F472B6",
        "spectro_temporal": "#DB2777",
    },
    "loss": {
        "l1": "#FCD34D",
        "huber": "#FBBF24",
        "wavelet": "#F59E0B",
        "combined": "#D97706",
    },
    "capacity": {
        "small": "#6EE7B7",
        "medium": "#34D399",
        "large": "#10B981",
    },
    "bidirectional": {
        "unidirectional": "#9CA3AF",
        "bidirectional": "#4B5563",
    },
    "augmentation": {
        "none": "#C4B5FD",
        "full": "#8B5CF6",
    },
}


def apply_nature_style():
    """Apply Nature Methods style to matplotlib."""
    plt.rcParams.update(NATURE_STYLE)
    setup_nature_style()  # Also apply shared style


# =============================================================================
# Phase 3 Visualizer
# =============================================================================

class Phase3Visualizer:
    """Generates publication-quality figures for Phase 3.

    Args:
        output_dir: Directory to save figures
        dpi: Figure resolution

    Example:
        >>> viz = Phase3Visualizer(output_dir="figures/")
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
        result: Any,  # Phase3Result
        filename: str = "figure_3_1_ablation_results.pdf",
    ) -> Path:
        """Generate main Figure 3.1: Ablation Results.

        Four panels:
            (A) Attention ablation: R² by type
            (B) Loss ablation: R² by loss function
            (C) Waterfall: contribution of each component
            (D) Optimal configuration summary

        Args:
            result: Phase3Result object
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(7.2, 6.0))
        gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

        # (A) Attention ablation
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_study_bars(ax_a, result, "attention", "(A) Attention Mechanism")

        # (B) Loss function ablation
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_study_bars(ax_b, result, "loss", "(B) Loss Function")

        # (C) Waterfall chart
        ax_c = fig.add_subplot(gs[1, 0])
        self._plot_waterfall(ax_c, result)

        # (D) Optimal configuration
        ax_d = fig.add_subplot(gs[1, 1])
        self._plot_optimal_config(ax_d, result)

        # Add panel labels
        for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ['A', 'B', 'C', 'D']):
            ax.text(-0.15, 1.08, label, transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='top')

        # Save
        output_path = self.output_dir / filename
        fig.savefig(output_path, format=filename.split('.')[-1],
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def _plot_study_bars(
        self,
        ax: plt.Axes,
        result: Any,
        study: str,
        title: str,
    ):
        """Plot bar chart for a single ablation study."""
        if study not in result.aggregated:
            ax.text(0.5, 0.5, f"No data for {study}", ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(title)
            return

        variants = result.aggregated[study]
        sorted_variants = sorted(
            variants.items(), key=lambda x: x[1]["r2_mean"], reverse=True
        )

        names = [v[0] for v in sorted_variants]
        r2_means = [v[1]["r2_mean"] for v in sorted_variants]
        r2_stds = [v[1]["r2_std"] for v in sorted_variants]

        x = np.arange(len(names))
        colors = [VARIANT_COLORS.get(study, {}).get(n, COLORS["neutral"]) for n in names]

        bars = ax.bar(x, r2_means, color=colors, edgecolor='white', linewidth=0.5)
        ax.errorbar(x, r2_means, yerr=r2_stds,
                   fmt='none', color='black', capsize=3, linewidth=1)

        # Baseline line
        ax.axhline(y=result.baseline_r2, color=COLORS["baseline"],
                  linestyle='--', linewidth=1, label=f'Baseline: {result.baseline_r2:.3f}')

        # Labels
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('_', '\n') for n in names], rotation=0, ha='center')
        ax.set_ylabel('R²')
        ax.set_title(title)
        ax.legend(loc='upper right', frameon=False, fontsize=6)

        # Highlight best
        best_idx = 0  # Already sorted
        bars[best_idx].set_edgecolor(COLORS["highlight"])
        bars[best_idx].set_linewidth(2)

    def _plot_waterfall(self, ax: plt.Axes, result: Any):
        """Plot waterfall chart showing component contributions."""
        # Calculate contribution of each ablation
        contributions = []
        labels = []

        baseline = result.baseline_r2

        for study, variants in result.aggregated.items():
            if study == "baseline":
                continue

            # Find worst variant (component removed)
            worst_r2 = min(v["r2_mean"] for v in variants.values())
            # Contribution = baseline - worst (how much we lose without this component)
            contribution = baseline - worst_r2
            contributions.append(contribution)
            labels.append(study.replace('_', '\n'))

        # Sort by contribution
        sorted_pairs = sorted(zip(contributions, labels), reverse=True)
        contributions = [p[0] for p in sorted_pairs]
        labels = [p[1] for p in sorted_pairs]

        # Plot waterfall
        x = np.arange(len(labels))
        colors = [STUDY_COLORS.get(l.replace('\n', '_'), COLORS["primary"]) for l in labels]

        # Cumulative positions for waterfall effect
        bars = ax.bar(x, contributions, color=colors, edgecolor='white', linewidth=0.5)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, contributions)):
            sign = '+' if val >= 0 else ''
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{sign}{val:.3f}', ha='center', va='bottom', fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('ΔR² (contribution)')
        ax.set_title('(C) Component Contributions')
        ax.axhline(y=0, color='black', linewidth=0.5)

    def _plot_optimal_config(self, ax: plt.Axes, result: Any):
        """Plot summary of optimal configuration."""
        # Create table-like visualization
        ax.axis('off')

        # Title
        ax.text(0.5, 0.95, '(D) Optimal Configuration', ha='center', va='top',
               fontsize=9, fontweight='bold', transform=ax.transAxes)

        # Configuration items
        y_start = 0.80
        y_step = 0.12

        items = [
            ("Baseline R²", f"{result.baseline_r2:.4f}"),
        ]

        for study, variant in result.optimal_config.items():
            study_name = study.replace('_', ' ').title()
            items.append((study_name, variant))

        # Draw items
        for i, (key, value) in enumerate(items):
            y = y_start - i * y_step
            ax.text(0.1, y, f"{key}:", ha='left', va='center',
                   fontsize=8, fontweight='bold', transform=ax.transAxes)
            ax.text(0.6, y, value, ha='left', va='center',
                   fontsize=8, transform=ax.transAxes)

        # Draw box
        ax.add_patch(plt.Rectangle((0.05, y_start - len(items) * y_step),
                                   0.90, len(items) * y_step + 0.1,
                                   fill=False, edgecolor=COLORS["neutral"],
                                   linewidth=1, transform=ax.transAxes))

    def plot_all_studies(
        self,
        result: Any,
        filename: str = "figure_3_2_all_ablations.pdf",
    ) -> Path:
        """Plot bar charts for all ablation studies.

        Args:
            result: Phase3Result object
            filename: Output filename

        Returns:
            Path to saved figure
        """
        studies = [s for s in result.aggregated.keys() if s != "baseline"]
        n_studies = len(studies)

        if n_studies == 0:
            print("No ablation studies to plot")
            return None

        # Create grid
        n_cols = min(3, n_studies)
        n_rows = (n_studies + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_studies == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, study in enumerate(studies):
            self._plot_study_bars(axes[i], result, study, study.replace('_', ' ').title())

        # Hide unused axes
        for i in range(n_studies, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(output_path, format=filename.split('.')[-1],
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def plot_learning_curves(
        self,
        result: Any,
        filename: str = "figure_3_3_learning_curves.pdf",
    ) -> Path:
        """Plot learning curves for selected ablations.

        Args:
            result: Phase3Result object
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

        # Group results by study
        study_results = {}
        for r in result.results:
            key = f"{r.study}_{r.variant}"
            if key not in study_results:
                study_results[key] = r

        # (A) Validation loss curves
        ax = axes[0]
        for key, r in study_results.items():
            if r.study in ["attention", "loss"]:  # Focus on key studies
                color = STUDY_COLORS.get(r.study, COLORS["neutral"])
                epochs = np.arange(1, len(r.val_losses) + 1)
                ax.plot(epochs, r.val_losses, label=f"{r.study}: {r.variant}",
                       color=color, alpha=0.7, linewidth=1)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss')
        ax.legend(loc='upper right', frameon=False, fontsize=5, ncol=2)

        # (B) R² progression
        ax = axes[1]
        for key, r in study_results.items():
            if r.study in ["attention", "loss"]:
                color = STUDY_COLORS.get(r.study, COLORS["neutral"])
                epochs = np.arange(1, len(r.val_r2s) + 1)
                ax.plot(epochs, r.val_r2s, label=f"{r.study}: {r.variant}",
                       color=color, alpha=0.7, linewidth=1)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation R²')
        ax.set_title('R² Progression')
        ax.legend(loc='lower right', frameon=False, fontsize=5, ncol=2)

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(output_path, format=filename.split('.')[-1],
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def plot_comprehensive_stats_figure(
        self,
        result: Any,
        filename: str = "figure_3_4_statistical_analysis.pdf",
    ) -> Path:
        """Generate comprehensive statistical analysis figure for ablations.

        Three panels:
            (A) Box plots by ablation study showing fold distributions
            (B) Effect sizes for each ablation (vs baseline)
            (C) Statistical comparison summary table

        Args:
            result: Phase3Result object
            filename: Output filename

        Returns:
            Path to saved figure
        """
        apply_nature_style()

        fig = plt.figure(figsize=(7.2, 4.5))
        gs = GridSpec(1, 3, figure=fig, wspace=0.4)

        # (A) Box plots by study
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_ablation_boxplots(ax_a, result)

        # (B) Effect sizes
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_ablation_effect_sizes(ax_b, result)

        # (C) Statistical table
        ax_c = fig.add_subplot(gs[0, 2])
        self._plot_statistical_table(ax_c, result)

        # Add panel labels
        for ax, label in zip([ax_a, ax_b, ax_c], ['A', 'B', 'C']):
            ax.text(-0.15, 1.08, label, transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='top')

        # Save
        output_path = self.output_dir / filename
        fig.savefig(output_path, format='pdf', bbox_inches='tight')

        png_path = output_path.with_suffix('.png')
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def _plot_ablation_boxplots(self, ax: plt.Axes, result: Any):
        """Plot box plots for ablation studies with fold data."""
        # Collect fold data per ablation
        ablation_data = {}

        for r in result.results:
            key = f"{r.study}:{r.variant}"
            if key not in ablation_data:
                ablation_data[key] = []
            ablation_data[key].append(r.best_r2)

        if not ablation_data:
            ax.text(0.5, 0.5, "No fold data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(A) R² by Ablation')
            return

        # Sort by mean R²
        sorted_keys = sorted(ablation_data.keys(), key=lambda k: np.mean(ablation_data[k]), reverse=True)[:8]

        fold_data = [ablation_data[k] for k in sorted_keys]
        labels = [k.split(':')[1][:8] for k in sorted_keys]

        bp = ax.boxplot(fold_data, patch_artist=True, widths=0.6, showfliers=False)

        for i, patch in enumerate(bp["boxes"]):
            study = sorted_keys[i].split(':')[0]
            color = STUDY_COLORS.get(study, COLORS["neutral"])
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay points
        for i, data in enumerate(fold_data):
            jitter = np.random.uniform(-0.1, 0.1, len(data))
            ax.scatter([i + 1 + j for j in jitter], data, color='black', s=10, alpha=0.5, zorder=3)

        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=5)
        ax.set_ylabel('R²')
        ax.set_title('(A) R² by Ablation')
        ax.axhline(y=result.baseline_r2, color=COLORS["baseline"], linestyle='--', linewidth=1)

    def _plot_ablation_effect_sizes(self, ax: plt.Axes, result: Any):
        """Plot effect sizes for ablations vs baseline."""
        # Get baseline fold data
        baseline_data = []
        for r in result.results:
            if r.study == "baseline":
                baseline_data.append(r.best_r2)

        if len(baseline_data) < 2:
            ax.text(0.5, 0.5, "Insufficient baseline data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("(B) Effect Sizes")
            return

        baseline_arr = np.array(baseline_data)

        # Compute effect sizes for each ablation
        effect_data = {}
        for r in result.results:
            if r.study == "baseline":
                continue

            key = f"{r.study}:{r.variant}"
            if key not in effect_data:
                effect_data[key] = []
            effect_data[key].append(r.best_r2)

        effects = []
        labels = []
        for key, values in effect_data.items():
            if len(values) >= 2:
                comp = compare_methods(baseline_arr, np.array(values), "baseline", key, paired=True)
                d = comp.parametric_test.effect_size or 0
                effects.append(d)
                labels.append(key.split(':')[1][:8])

        if not effects:
            ax.text(0.5, 0.5, "No effect size data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("(B) Effect Sizes")
            return

        y_pos = np.arange(len(labels))
        colors = [COLORS["primary"] if e > 0 else COLORS["secondary"] for e in effects]

        ax.barh(y_pos, effects, color=colors, height=0.6, alpha=0.8)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

        for x in [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]:
            ax.axvline(x=x, color='lightgray', linestyle=':', linewidth=0.5, alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=5)
        ax.set_xlabel("Cohen's d (vs baseline)")
        ax.set_title("(B) Effect Sizes")
        ax.invert_yaxis()

    def _plot_statistical_table(self, ax: plt.Axes, result: Any):
        """Plot statistical comparison summary table."""
        ax.axis('off')

        # Get baseline data
        baseline_data = [r.best_r2 for r in result.results if r.study == "baseline"]

        if len(baseline_data) < 2:
            ax.text(0.5, 0.5, "Need baseline folds for stats", ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(C) Statistical Comparison')
            return

        baseline_arr = np.array(baseline_data)

        # Compute stats for each ablation study
        table_data = []
        for study, variants in result.aggregated.items():
            if study == "baseline":
                continue

            for variant, stats in variants.items():
                # Get fold data
                variant_data = [r.best_r2 for r in result.results
                               if r.study == study and r.variant == variant]

                if len(variant_data) >= 2:
                    comp = compare_methods(baseline_arr, np.array(variant_data),
                                         "baseline", variant, paired=True)
                    p_val = comp.parametric_test.p_value
                    d = comp.parametric_test.effect_size or 0

                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

                    table_data.append([
                        f"{variant[:8]}",
                        f"{stats['r2_mean']:.3f}",
                        f"{p_val:.3f}",
                        sig
                    ])

        if not table_data:
            ax.text(0.5, 0.5, "No statistical data", ha='center', va='center', transform=ax.transAxes)
            return

        columns = ['Variant', 'R² Mean', 'p-value', 'Sig']
        table = ax.table(
            cellText=table_data[:8],  # Limit rows
            colLabels=columns,
            loc='center',
            cellLoc='center',
        )

        table.auto_set_font_size(False)
        table.set_fontsize(5)
        table.scale(1.2, 1.3)

        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#E6E6E6')
            table[(0, i)].set_text_props(weight='bold')

        ax.set_title('(C) Statistical Comparison', pad=10)

    def plot_all(self, result: Any, format: str = "pdf") -> List[Path]:
        """Generate all Phase 3 figures.

        Args:
            result: Phase3Result object
            format: Output format (pdf, png, svg)

        Returns:
            List of paths to saved figures
        """
        paths = []

        paths.append(self.plot_main_figure(
            result,
            filename=f"figure_3_1_ablation_results.{format}",
        ))

        paths.append(self.plot_all_studies(
            result,
            filename=f"figure_3_2_all_ablations.{format}",
        ))

        paths.append(self.plot_learning_curves(
            result,
            filename=f"figure_3_3_learning_curves.{format}",
        ))

        paths.append(self.plot_comprehensive_stats_figure(
            result,
            filename=f"figure_3_4_statistical_analysis.{format}",
        ))

        # Filter out None values
        paths = [p for p in paths if p is not None]

        return paths


def create_ablation_table(result: Any) -> str:
    """Create a LaTeX-ready summary table.

    Args:
        result: Phase3Result object

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Phase 3: CondUNet Ablation Results}",
        r"\label{tab:ablation_results}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Study & Variant & R² Mean & R² Std & $\Delta$ Baseline \\",
        r"\midrule",
    ]

    for study, variants in result.aggregated.items():
        if study == "baseline":
            continue

        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{study.replace('_', ' ').title()}}}}} \\\\")

        sorted_variants = sorted(
            variants.items(), key=lambda x: x[1]["r2_mean"], reverse=True
        )

        for variant, stats in sorted_variants:
            delta = stats["r2_mean"] - result.baseline_r2
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            line = f"  & {variant} & {stats['r2_mean']:.4f} & {stats['r2_std']:.4f} & {delta_str} \\\\"
            lines.append(line)

    lines.extend([
        r"\midrule",
        f"\\multicolumn{{2}}{{l}}{{Baseline}} & {result.baseline_r2:.4f} & - & - \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)
