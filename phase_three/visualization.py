"""
Visualization Module for Phase 3: CondUNet Ablation Studies
============================================================

Publication-quality figure generation following Nature Methods guidelines.

Supports both protocols:
- ADDITIVE: Incremental component analysis (waterfall chart)
- SUBTRACTIVE: Traditional ablation (component removal)

Figures generated:
    3.1: Main Ablation Results
    3.2: Statistical Analysis
    3.3: Learning Curves
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
import matplotlib.patches as mpatches

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

from .config import INCREMENTAL_STAGES


# =============================================================================
# Style Configuration (Nature Methods compliant)
# =============================================================================

NATURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.titlesize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 0.8,
    "lines.markersize": 3,
    "patch.linewidth": 0.5,
}

# Colorblind-friendly palette (based on Wong 2011, Nature Methods)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
    "cyan": "#56B4E9",
    "red": "#D55E00",
    "black": "#000000",
    "gray": "#999999",
}

# Stage colors for incremental analysis
STAGE_COLORS = [
    "#E5E7EB",  # Stage 0: baseline (light gray)
    "#0072B2",  # Stage 1: modern_conv (blue)
    "#E69F00",  # Stage 2: attention (orange)
    "#009E73",  # Stage 3: conditioning (green)
    "#CC79A7",  # Stage 4: wavelet_loss (pink)
    "#56B4E9",  # Stage 5: augmentation (cyan)
    "#D55E00",  # Stage 6: bidirectional (red)
]

# Component names for display
STAGE_LABELS = {
    0: "Baseline",
    1: "+ Modern Conv",
    2: "+ Attention",
    3: "+ Conditioning",
    4: "+ Wavelet Loss",
    5: "+ Augmentation",
    6: "+ Bidirectional",
}


def apply_nature_style():
    """Apply Nature Methods style to matplotlib."""
    plt.rcParams.update(NATURE_STYLE)
    try:
        setup_nature_style()
    except Exception:
        pass  # Use our style if shared utils fail


# =============================================================================
# Phase 3 Visualizer
# =============================================================================

class Phase3Visualizer:
    """Generates publication-quality figures for Phase 3 ablation studies.

    Supports both ADDITIVE and SUBTRACTIVE protocols.

    Args:
        output_dir: Directory to save figures
        dpi: Figure resolution (default 300 for Nature Methods)

    Example:
        >>> viz = Phase3Visualizer(output_dir="figures/")
        >>> viz.plot_all(result)
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

    def _detect_protocol(self, result: Any) -> str:
        """Detect whether result is from additive or subtractive protocol."""
        studies = list(result.aggregated.keys())
        if any(s.startswith("stage") for s in studies):
            return "additive"
        return "subtractive"

    def plot_all(self, result: Any, format: str = "pdf") -> List[Path]:
        """Generate all Phase 3 figures.

        Automatically detects protocol and generates appropriate figures.

        Args:
            result: Phase3Result object
            format: Output format (pdf, png, svg)

        Returns:
            List of paths to saved figures
        """
        paths = []
        protocol = self._detect_protocol(result)

        if protocol == "additive":
            paths.extend(self._plot_additive_figures(result, format))
        else:
            paths.extend(self._plot_subtractive_figures(result, format))

        # Filter out None values
        paths = [p for p in paths if p is not None]
        return paths

    # =========================================================================
    # ADDITIVE PROTOCOL FIGURES
    # =========================================================================

    def _plot_additive_figures(self, result: Any, format: str) -> List[Path]:
        """Generate figures for additive (build-up) protocol."""
        paths = []

        # Main figure: Incremental analysis
        paths.append(self.plot_incremental_analysis(
            result,
            filename=f"figure_3_1_incremental_analysis.{format}",
        ))

        # Statistical comparison
        paths.append(self.plot_incremental_stats(
            result,
            filename=f"figure_3_2_statistical_analysis.{format}",
        ))

        return paths

    def plot_incremental_analysis(
        self,
        result: Any,
        filename: str = "figure_3_1_incremental_analysis.pdf",
    ) -> Path:
        """Generate main figure for additive protocol: Incremental Component Analysis.

        Nature Methods style 2-column figure (180mm width = 7.08 inches).

        Panels:
            (A) Progressive R² improvement (line + bar chart)
            (B) Component contribution waterfall
            (C) Parameter efficiency
            (D) Summary table

        Args:
            result: Phase3Result object
            filename: Output filename

        Returns:
            Path to saved figure
        """
        apply_nature_style()

        # Nature Methods 2-column figure: 180mm = 7.08 inches
        fig = plt.figure(figsize=(7.08, 5.5))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
                     left=0.08, right=0.98, top=0.95, bottom=0.08)

        # Extract stage data
        stage_data = self._extract_stage_data(result)

        if not stage_data:
            fig.text(0.5, 0.5, "No stage data available", ha='center', va='center')
            output_path = self.output_dir / filename
            fig.savefig(output_path, format=filename.split('.')[-1], dpi=self.dpi)
            plt.close(fig)
            return output_path

        # (A) Progressive R² improvement
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_progressive_r2(ax_a, stage_data)

        # (B) Component contribution waterfall
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_contribution_waterfall(ax_b, stage_data)

        # (C) Fold distribution box plots
        ax_c = fig.add_subplot(gs[1, 0])
        self._plot_stage_boxplots(ax_c, stage_data, result)

        # (D) Summary table
        ax_d = fig.add_subplot(gs[1, 1])
        self._plot_summary_table(ax_d, stage_data, result)

        # Add panel labels (Nature Methods style: bold, 8pt)
        for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ['a', 'b', 'c', 'd']):
            ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                   fontsize=8, fontweight='bold', va='top')

        # Save
        output_path = self.output_dir / filename
        fig.savefig(output_path, format=filename.split('.')[-1],
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {output_path}")
        return output_path

    def _extract_stage_data(self, result: Any) -> List[Dict]:
        """Extract stage data from result in order."""
        stage_data = []

        for stage_idx in range(len(INCREMENTAL_STAGES)):
            stage_name = f"stage{stage_idx}"
            stage_info = INCREMENTAL_STAGES[stage_idx]
            variant_name = stage_info["name"]

            if stage_name in result.aggregated:
                if variant_name in result.aggregated[stage_name]:
                    stats = result.aggregated[stage_name][variant_name]
                    stage_data.append({
                        "stage": stage_idx,
                        "name": variant_name,
                        "label": STAGE_LABELS.get(stage_idx, f"Stage {stage_idx}"),
                        "description": stage_info["description"],
                        "r2_mean": stats["r2_mean"],
                        "r2_std": stats["r2_std"],
                        "fold_r2s": stats.get("fold_r2s", []),
                        "color": STAGE_COLORS[stage_idx % len(STAGE_COLORS)],
                    })

        return stage_data

    def _plot_progressive_r2(self, ax: plt.Axes, stage_data: List[Dict]):
        """Plot progressive R² improvement as line chart with bars."""
        stages = [d["stage"] for d in stage_data]
        r2_means = [d["r2_mean"] for d in stage_data]
        r2_stds = [d["r2_std"] for d in stage_data]
        colors = [d["color"] for d in stage_data]
        labels = [d["label"] for d in stage_data]

        x = np.arange(len(stages))

        # Bar chart
        bars = ax.bar(x, r2_means, color=colors, edgecolor='black', linewidth=0.3,
                     alpha=0.85, width=0.7)

        # Error bars
        ax.errorbar(x, r2_means, yerr=r2_stds, fmt='none', color='black',
                   capsize=2, capthick=0.5, linewidth=0.5)

        # Connecting line to show progression
        ax.plot(x, r2_means, 'k-', linewidth=1, marker='o', markersize=3,
               markerfacecolor='white', markeredgecolor='black', markeredgewidth=0.5)

        # Delta annotations
        for i in range(1, len(r2_means)):
            delta = r2_means[i] - r2_means[i-1]
            if delta > 0:
                ax.annotate(f'+{delta:.3f}',
                           xy=(i, r2_means[i] + r2_stds[i] + 0.01),
                           ha='center', va='bottom', fontsize=5,
                           color=COLORS["green"])
            elif delta < 0:
                ax.annotate(f'{delta:.3f}',
                           xy=(i, r2_means[i] + r2_stds[i] + 0.01),
                           ha='center', va='bottom', fontsize=5,
                           color=COLORS["red"])

        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=5)
        ax.set_ylabel('R²', fontsize=7)
        ax.set_title('Progressive Component Addition', fontsize=8, fontweight='bold')

        # Set y-axis limits with padding
        y_min = min(r2_means) - max(r2_stds) - 0.05
        y_max = max(r2_means) + max(r2_stds) + 0.08
        ax.set_ylim(max(0, y_min), min(1, y_max))

        # Add baseline reference line
        ax.axhline(y=r2_means[0], color=COLORS["gray"], linestyle='--',
                  linewidth=0.5, alpha=0.7)

    def _plot_contribution_waterfall(self, ax: plt.Axes, stage_data: List[Dict]):
        """Plot waterfall chart showing incremental contribution of each component."""
        if len(stage_data) < 2:
            ax.text(0.5, 0.5, "Need ≥2 stages", ha='center', va='center',
                   transform=ax.transAxes, fontsize=7)
            ax.set_title('Component Contributions', fontsize=8, fontweight='bold')
            return

        # Calculate incremental contributions
        contributions = []
        labels = []
        colors = []

        for i in range(1, len(stage_data)):
            delta = stage_data[i]["r2_mean"] - stage_data[i-1]["r2_mean"]
            contributions.append(delta)
            labels.append(stage_data[i]["label"].replace("+ ", ""))
            colors.append(stage_data[i]["color"])

        # Sort by contribution (largest first)
        sorted_idx = np.argsort(contributions)[::-1]
        contributions = [contributions[i] for i in sorted_idx]
        labels = [labels[i] for i in sorted_idx]
        colors = [colors[i] for i in sorted_idx]

        x = np.arange(len(contributions))

        # Plot bars
        bar_colors = [COLORS["green"] if c >= 0 else COLORS["red"] for c in contributions]
        bars = ax.bar(x, contributions, color=bar_colors, edgecolor='black',
                     linewidth=0.3, alpha=0.85)

        # Value annotations
        for i, (bar, val) in enumerate(zip(bars, contributions)):
            y_pos = bar.get_height()
            if val >= 0:
                ax.text(bar.get_x() + bar.get_width()/2, y_pos + 0.002,
                       f'+{val:.3f}', ha='center', va='bottom', fontsize=5,
                       fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, y_pos - 0.002,
                       f'{val:.3f}', ha='center', va='top', fontsize=5,
                       fontweight='bold')

        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=5)
        ax.set_ylabel('ΔR²', fontsize=7)
        ax.set_title('Component Contributions', fontsize=8, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.5)

        # Total improvement annotation
        total = sum(contributions)
        ax.text(0.98, 0.95, f'Total: +{total:.3f}',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=6, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=COLORS["gray"], linewidth=0.5))

    def _plot_stage_boxplots(self, ax: plt.Axes, stage_data: List[Dict], result: Any):
        """Plot box plots showing R² distribution across folds for each stage."""
        # Collect fold data
        fold_data = []
        labels = []
        colors = []

        for sd in stage_data:
            stage_name = f"stage{sd['stage']}"
            variant_name = sd["name"]

            # Get fold R² values from results
            r2s = [r.best_r2 for r in result.results
                  if r.study == stage_name and r.variant == variant_name]

            if r2s:
                fold_data.append(r2s)
                labels.append(sd["label"])
                colors.append(sd["color"])

        if not fold_data:
            ax.text(0.5, 0.5, "No fold data", ha='center', va='center',
                   transform=ax.transAxes, fontsize=7)
            ax.set_title('R² Distribution Across Folds', fontsize=8, fontweight='bold')
            return

        # Create box plot
        bp = ax.boxplot(fold_data, patch_artist=True, widths=0.6,
                       showfliers=False, notch=False)

        # Style boxes
        for i, (patch, color) in enumerate(zip(bp["boxes"], colors)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.5)

        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='black', linewidth=0.5)

        # Overlay individual data points
        for i, (data, color) in enumerate(zip(fold_data, colors)):
            jitter = np.random.uniform(-0.15, 0.15, len(data))
            ax.scatter([i + 1 + j for j in jitter], data,
                      c='black', s=8, alpha=0.6, zorder=3, linewidth=0)

        # Styling
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=5)
        ax.set_ylabel('R²', fontsize=7)
        ax.set_title('R² Distribution Across Folds', fontsize=8, fontweight='bold')

    def _plot_summary_table(self, ax: plt.Axes, stage_data: List[Dict], result: Any):
        """Plot summary table with key metrics."""
        ax.axis('off')

        if not stage_data:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                   transform=ax.transAxes, fontsize=7)
            return

        # Table data
        columns = ['Stage', 'R² Mean', '± Std', 'Δ Prev']
        cell_data = []
        cell_colors = []

        prev_r2 = None
        for sd in stage_data:
            if prev_r2 is not None:
                delta = sd["r2_mean"] - prev_r2
                delta_str = f'+{delta:.3f}' if delta >= 0 else f'{delta:.3f}'
            else:
                delta_str = '—'

            row = [
                sd["label"][:12],
                f'{sd["r2_mean"]:.4f}',
                f'{sd["r2_std"]:.4f}',
                delta_str,
            ]
            cell_data.append(row)
            cell_colors.append([sd["color"], 'white', 'white', 'white'])
            prev_r2 = sd["r2_mean"]

        # Create table
        table = ax.table(
            cellText=cell_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            cellColours=cell_colors,
        )

        table.auto_set_font_size(False)
        table.set_fontsize(5)
        table.scale(1.0, 1.3)

        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4B5563')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        # Total improvement
        if len(stage_data) >= 2:
            total_gain = stage_data[-1]["r2_mean"] - stage_data[0]["r2_mean"]
            ax.text(0.5, -0.05, f'Total Improvement: +{total_gain:.4f}',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=7, fontweight='bold')

    def plot_incremental_stats(
        self,
        result: Any,
        filename: str = "figure_3_2_statistical_analysis.pdf",
    ) -> Path:
        """Generate statistical analysis figure for additive protocol.

        Args:
            result: Phase3Result object
            filename: Output filename

        Returns:
            Path to saved figure
        """
        apply_nature_style()

        fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.0))

        stage_data = self._extract_stage_data(result)

        # (A) Effect sizes vs baseline
        self._plot_effect_sizes_vs_baseline(axes[0], stage_data, result)

        # (B) Pairwise comparison matrix
        self._plot_pairwise_comparison(axes[1], stage_data, result)

        # Panel labels
        for ax, label in zip(axes, ['a', 'b']):
            ax.text(-0.12, 1.08, label, transform=ax.transAxes,
                   fontsize=8, fontweight='bold', va='top')

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(output_path, format=filename.split('.')[-1],
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {output_path}")
        return output_path

    def _plot_effect_sizes_vs_baseline(self, ax: plt.Axes, stage_data: List[Dict], result: Any):
        """Plot effect sizes of each stage vs baseline (stage 0)."""
        if len(stage_data) < 2:
            ax.text(0.5, 0.5, "Need ≥2 stages", ha='center', va='center',
                   transform=ax.transAxes, fontsize=7)
            ax.set_title('Effect Sizes vs Baseline', fontsize=8, fontweight='bold')
            return

        # Get baseline fold data
        baseline_name = f"stage0"
        baseline_variant = stage_data[0]["name"]
        baseline_r2s = [r.best_r2 for r in result.results
                       if r.study == baseline_name and r.variant == baseline_variant]

        if len(baseline_r2s) < 2:
            ax.text(0.5, 0.5, "Insufficient baseline data", ha='center', va='center',
                   transform=ax.transAxes, fontsize=7)
            ax.set_title('Effect Sizes vs Baseline', fontsize=8, fontweight='bold')
            return

        baseline_arr = np.array(baseline_r2s)

        # Compute effect sizes for each stage
        labels = []
        effects = []
        colors = []

        for sd in stage_data[1:]:  # Skip baseline
            stage_name = f"stage{sd['stage']}"
            variant_name = sd["name"]

            stage_r2s = [r.best_r2 for r in result.results
                        if r.study == stage_name and r.variant == variant_name]

            if len(stage_r2s) >= 2:
                try:
                    comp = compare_methods(baseline_arr, np.array(stage_r2s),
                                          "baseline", stage_name, paired=True)
                    d = comp.parametric_test.effect_size or 0
                    effects.append(d)
                    labels.append(sd["label"].replace("+ ", ""))
                    colors.append(sd["color"])
                except Exception:
                    pass

        if not effects:
            ax.text(0.5, 0.5, "Could not compute effect sizes", ha='center', va='center',
                   transform=ax.transAxes, fontsize=7)
            ax.set_title('Effect Sizes vs Baseline', fontsize=8, fontweight='bold')
            return

        # Plot horizontal bar chart
        y_pos = np.arange(len(labels))
        bar_colors = [COLORS["green"] if e > 0 else COLORS["red"] for e in effects]

        ax.barh(y_pos, effects, color=bar_colors, height=0.6, alpha=0.85,
               edgecolor='black', linewidth=0.3)

        # Reference lines for effect size interpretation
        for x_val, label in [(-0.8, 'large'), (-0.5, 'med'), (-0.2, 'small'),
                             (0.2, 'small'), (0.5, 'med'), (0.8, 'large')]:
            ax.axvline(x=x_val, color=COLORS["gray"], linestyle=':', linewidth=0.3, alpha=0.5)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Cohen's d (vs Baseline)", fontsize=7)
        ax.set_title('Effect Sizes vs Baseline', fontsize=8, fontweight='bold')
        ax.invert_yaxis()

    def _plot_pairwise_comparison(self, ax: plt.Axes, stage_data: List[Dict], result: Any):
        """Plot pairwise statistical comparison heatmap."""
        n_stages = len(stage_data)

        if n_stages < 2:
            ax.text(0.5, 0.5, "Need ≥2 stages", ha='center', va='center',
                   transform=ax.transAxes, fontsize=7)
            ax.set_title('Pairwise Comparisons', fontsize=8, fontweight='bold')
            return

        # Build significance matrix
        sig_matrix = np.zeros((n_stages, n_stages))

        for i, sd_i in enumerate(stage_data):
            stage_i = f"stage{sd_i['stage']}"
            r2s_i = [r.best_r2 for r in result.results
                    if r.study == stage_i and r.variant == sd_i["name"]]

            for j, sd_j in enumerate(stage_data):
                if i == j:
                    continue

                stage_j = f"stage{sd_j['stage']}"
                r2s_j = [r.best_r2 for r in result.results
                        if r.study == stage_j and r.variant == sd_j["name"]]

                if len(r2s_i) >= 2 and len(r2s_j) >= 2:
                    try:
                        comp = compare_methods(np.array(r2s_i), np.array(r2s_j),
                                              stage_i, stage_j, paired=True)
                        sig_matrix[i, j] = comp.parametric_test.p_value
                    except Exception:
                        sig_matrix[i, j] = 1.0

        # Plot heatmap
        im = ax.imshow(sig_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1, aspect='equal')

        # Add significance markers
        for i in range(n_stages):
            for j in range(n_stages):
                if i != j:
                    p = sig_matrix[i, j]
                    if p < 0.001:
                        text = "***"
                    elif p < 0.01:
                        text = "**"
                    elif p < 0.05:
                        text = "*"
                    else:
                        text = ""

                    text_color = 'white' if p < 0.05 else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                           fontsize=6, color=text_color, fontweight='bold')

        # Labels
        labels = [sd["label"][:8] for sd in stage_data]
        ax.set_xticks(range(n_stages))
        ax.set_yticks(range(n_stages))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
        ax.set_title('Pairwise p-values', fontsize=8, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('p-value', fontsize=6)
        cbar.ax.tick_params(labelsize=5)

    # =========================================================================
    # SUBTRACTIVE PROTOCOL FIGURES (legacy support)
    # =========================================================================

    def _plot_subtractive_figures(self, result: Any, format: str) -> List[Path]:
        """Generate figures for subtractive (traditional) protocol."""
        paths = []

        paths.append(self.plot_main_figure_subtractive(
            result, filename=f"figure_3_1_ablation_results.{format}"))

        paths.append(self.plot_all_studies(
            result, filename=f"figure_3_2_all_ablations.{format}"))

        return paths

    def plot_main_figure_subtractive(
        self,
        result: Any,
        filename: str = "figure_3_1_ablation_results.pdf",
    ) -> Path:
        """Generate main figure for subtractive protocol."""
        apply_nature_style()

        fig = plt.figure(figsize=(7.08, 5.5))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

        studies = [s for s in result.aggregated.keys() if s != "baseline"]

        # Plot up to 4 studies
        axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(min(4, len(studies)))]

        for ax, study in zip(axes, studies[:4]):
            self._plot_study_bars(ax, result, study, study.replace('_', ' ').title())

        # Panel labels
        for ax, label in zip(axes, ['a', 'b', 'c', 'd']):
            ax.text(-0.12, 1.08, label, transform=ax.transAxes,
                   fontsize=8, fontweight='bold', va='top')

        output_path = self.output_dir / filename
        fig.savefig(output_path, format=filename.split('.')[-1],
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def _plot_study_bars(self, ax: plt.Axes, result: Any, study: str, title: str):
        """Plot bar chart for a single ablation study."""
        if study not in result.aggregated:
            ax.text(0.5, 0.5, f"No data for {study}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=7)
            ax.set_title(title, fontsize=8, fontweight='bold')
            return

        variants = result.aggregated[study]
        sorted_variants = sorted(variants.items(), key=lambda x: x[1]["r2_mean"], reverse=True)

        names = [v[0] for v in sorted_variants]
        r2_means = [v[1]["r2_mean"] for v in sorted_variants]
        r2_stds = [v[1]["r2_std"] for v in sorted_variants]

        x = np.arange(len(names))
        colors = [STAGE_COLORS[i % len(STAGE_COLORS)] for i in range(len(names))]

        bars = ax.bar(x, r2_means, color=colors, edgecolor='black', linewidth=0.3)
        ax.errorbar(x, r2_means, yerr=r2_stds, fmt='none', color='black',
                   capsize=2, linewidth=0.5)

        # Baseline line
        ax.axhline(y=result.baseline_r2, color=COLORS["gray"], linestyle='--',
                  linewidth=0.5, label=f'Baseline: {result.baseline_r2:.3f}')

        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('_', '\n')[:8] for n in names], fontsize=5)
        ax.set_ylabel('R²', fontsize=7)
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.legend(loc='upper right', frameon=False, fontsize=5)

    def plot_all_studies(self, result: Any, filename: str) -> Path:
        """Plot all ablation studies in a grid."""
        studies = [s for s in result.aggregated.keys() if s not in ["baseline", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6"]]

        if not studies:
            return None

        n_studies = len(studies)
        n_cols = min(3, n_studies)
        n_rows = (n_studies + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
        if n_studies == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()

        for i, study in enumerate(studies):
            self._plot_study_bars(axes[i], result, study, study.replace('_', ' ').title())

        for i in range(n_studies, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(output_path, format=filename.split('.')[-1],
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return output_path


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def create_ablation_table(result: Any) -> str:
    """Create a LaTeX-ready summary table for ablation results.

    Args:
        result: Phase3Result object

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Incremental Component Analysis Results}",
        r"\label{tab:ablation_results}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Stage & Component & R² Mean & R² Std & $\Delta$ Previous \\",
        r"\midrule",
    ]

    prev_r2 = None
    for stage_idx in range(len(INCREMENTAL_STAGES)):
        stage_name = f"stage{stage_idx}"
        stage_info = INCREMENTAL_STAGES[stage_idx]
        variant_name = stage_info["name"]

        if stage_name in result.aggregated:
            if variant_name in result.aggregated[stage_name]:
                stats = result.aggregated[stage_name][variant_name]
                r2_mean = stats["r2_mean"]
                r2_std = stats["r2_std"]

                if prev_r2 is not None:
                    delta = r2_mean - prev_r2
                    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                else:
                    delta_str = "---"

                line = f"{stage_idx} & {stage_info['description'][:40]} & {r2_mean:.4f} & {r2_std:.4f} & {delta_str} \\\\"
                lines.append(line)
                prev_r2 = r2_mean

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)
