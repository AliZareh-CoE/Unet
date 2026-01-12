"""
Visualization Module for Phase 4
================================

Publication-quality figure generation for inter vs intra session study.

Figures generated:
    4.1: Intra vs Inter Session
        (A) Olfactory: Intra vs Inter session R²
        (B) PFC: Intra vs Inter session R²
        (C) DANDI: Intra vs Inter session R²
        (D) Generalization gap across datasets

    4.2: Session/Subject Analysis
        (A) Per-session performance breakdown
        (B) Easy vs hard sessions identified
        (C) DANDI per-subject (18 subjects)
        (D) What predicts session difficulty?
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root for shared utilities
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    setup_nature_style,
    compare_methods,
    holm_correction,
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
}

# Colors
COLORS = {
    "intra": "#2E86AB",  # Blue for intra-session
    "inter": "#E63946",  # Red for inter-session
    "gap": "#6B7280",    # Gray for gap
    "olfactory": "#2CA58D",
    "pfc": "#F18F01",
    "dandi": "#7C3AED",
    "easy": "#10B981",
    "hard": "#EF4444",
}

DATASET_NAMES = {
    "olfactory": "Olfactory (OB→PCx)",
    "pfc": "PFC (PFC→CA1)",
    "dandi": "DANDI (AMY→HPC)",
}


def apply_nature_style():
    """Apply Nature Methods style."""
    plt.rcParams.update(NATURE_STYLE)
    setup_nature_style()  # Also apply shared style


# =============================================================================
# Phase 4 Visualizer
# =============================================================================

class Phase4Visualizer:
    """Generates publication-quality figures for Phase 4.

    Args:
        output_dir: Directory to save figures
        dpi: Figure resolution
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

    def plot_figure_4_1(
        self,
        result: Any,
        filename: str = "figure_4_1_intra_inter_session.pdf",
    ) -> Path:
        """Generate Figure 4.1: Intra vs Inter Session.

        Four panels:
            (A) Olfactory comparison
            (B) PFC comparison
            (C) DANDI comparison
            (D) Generalization gap summary

        Args:
            result: Phase4Result object
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(7.2, 6.0))
        gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

        datasets = ["olfactory", "pfc", "dandi"]
        panels = ["A", "B", "C"]

        # (A-C) Individual dataset comparisons
        for i, (dataset, panel) in enumerate(zip(datasets, panels)):
            row = i // 2
            col = i % 2
            if i == 2:  # DANDI goes in bottom left
                row, col = 1, 0

            ax = fig.add_subplot(gs[row, col])
            self._plot_dataset_comparison(ax, result, dataset, panel)

        # (D) Generalization gap summary
        ax_d = fig.add_subplot(gs[1, 1])
        self._plot_generalization_gaps(ax_d, result)

        # Save
        output_path = self.output_dir / filename
        fig.savefig(output_path, format=filename.split('.')[-1],
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def _plot_dataset_comparison(
        self,
        ax: plt.Axes,
        result: Any,
        dataset: str,
        panel: str,
    ):
        """Plot intra vs inter comparison for a single dataset."""
        if dataset not in result.summary:
            ax.text(0.5, 0.5, f"No data for {dataset}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"({panel}) {DATASET_NAMES.get(dataset, dataset)}")
            return

        modes = result.summary[dataset]

        # Get values
        intra = modes.get("intra", {})
        inter = modes.get("inter", {})

        intra_mean = intra.get("test_r2_mean", 0)
        intra_std = intra.get("test_r2_std", 0)
        inter_mean = inter.get("test_r2_mean", 0)
        inter_std = inter.get("test_r2_std", 0)

        # Bar plot
        x = [0, 1]
        means = [intra_mean, inter_mean]
        stds = [intra_std, inter_std]
        colors = [COLORS["intra"], COLORS["inter"]]
        labels = ["Intra-session", "Inter-session"]

        bars = ax.bar(x, means, color=colors, edgecolor='white', linewidth=0.5)
        ax.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=4)

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Test R²')
        ax.set_title(f"({panel}) {DATASET_NAMES.get(dataset, dataset)}")
        ax.set_ylim(bottom=0)

        # Add gap annotation
        gap = result.generalization_gaps.get(dataset, 0)
        mid_y = (intra_mean + inter_mean) / 2
        ax.annotate('', xy=(1, inter_mean), xytext=(0, intra_mean),
                   arrowprops=dict(arrowstyle='<->', color=COLORS["gap"], lw=1.5))
        ax.text(0.5, mid_y, f'Δ={gap:.3f}', ha='center', va='bottom',
               fontsize=7, color=COLORS["gap"])

    def _plot_generalization_gaps(self, ax: plt.Axes, result: Any):
        """Plot generalization gaps across datasets."""
        datasets = list(result.generalization_gaps.keys())
        gaps = [result.generalization_gaps[d] for d in datasets]

        # Sort by gap
        sorted_pairs = sorted(zip(gaps, datasets), reverse=True)
        gaps = [p[0] for p in sorted_pairs]
        datasets = [p[1] for p in sorted_pairs]

        x = np.arange(len(datasets))
        colors = [COLORS.get(d, COLORS["gap"]) for d in datasets]

        bars = ax.bar(x, gaps, color=colors, edgecolor='white', linewidth=0.5)

        # Add value labels
        for bar, gap in zip(bars, gaps):
            y_pos = bar.get_height() + 0.005 if gap >= 0 else bar.get_height() - 0.02
            va = 'bottom' if gap >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{gap:.3f}', ha='center', va=va, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_NAMES.get(d, d).split('(')[0].strip() for d in datasets])
        ax.set_ylabel('Generalization Gap (Intra - Inter)')
        ax.set_title('(D) Generalization Gap Summary')
        ax.axhline(y=0, color='black', linewidth=0.5)

    def plot_figure_4_2(
        self,
        result: Any,
        filename: str = "figure_4_2_session_analysis.pdf",
    ) -> Path:
        """Generate Figure 4.2: Session/Subject Analysis.

        Four panels:
            (A) Per-session performance breakdown
            (B) Easy vs hard sessions
            (C) DANDI per-subject
            (D) Session difficulty predictors

        Args:
            result: Phase4Result object
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(7.2, 6.0))
        gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

        # (A) Per-session breakdown (olfactory example)
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_session_breakdown(ax_a, result, "olfactory")

        # (B) Easy vs hard sessions
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_easy_hard(ax_b, result)

        # (C) DANDI per-subject
        ax_c = fig.add_subplot(gs[1, 0])
        self._plot_session_breakdown(ax_c, result, "dandi")

        # (D) Session difficulty histogram
        ax_d = fig.add_subplot(gs[1, 1])
        self._plot_difficulty_distribution(ax_d, result)

        # Save
        output_path = self.output_dir / filename
        fig.savefig(output_path, format=filename.split('.')[-1],
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def _plot_session_breakdown(
        self,
        ax: plt.Axes,
        result: Any,
        dataset: str,
    ):
        """Plot per-session performance breakdown."""
        # Collect session stats from inter-session results
        session_r2s = {}

        for r in result.results:
            if r.dataset == dataset and r.split_mode == "inter":
                for session, stats in r.session_stats.items():
                    if session not in session_r2s:
                        session_r2s[session] = []
                    session_r2s[session].append(stats["r2"])

        if not session_r2s:
            ax.text(0.5, 0.5, f"No session data for {dataset}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Per-Session R² ({dataset.title()})")
            return

        # Average across runs
        sessions = sorted(session_r2s.keys())
        means = [np.mean(session_r2s[s]) for s in sessions]
        stds = [np.std(session_r2s[s]) for s in sessions]

        # Sort by performance
        sorted_data = sorted(zip(means, stds, sessions), reverse=True)
        means = [d[0] for d in sorted_data]
        stds = [d[1] for d in sorted_data]
        sessions = [d[2] for d in sorted_data]

        x = np.arange(len(sessions))
        color = COLORS.get(dataset, COLORS["gap"])

        ax.bar(x, means, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=2)

        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}" for s in sessions], rotation=45, ha='right', fontsize=6)
        ax.set_ylabel('Test R²')
        ax.set_xlabel('Session')
        ax.set_title(f"Per-Session R² ({dataset.title()})")

    def _plot_easy_hard(self, ax: plt.Axes, result: Any):
        """Plot easy vs hard session comparison."""
        datasets = list(result.easy_hard_sessions.keys())

        if not datasets:
            ax.text(0.5, 0.5, "No easy/hard data available",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("(B) Easy vs Hard Sessions")
            return

        # Compute average R² for easy vs hard sessions
        easy_r2s = []
        hard_r2s = []

        for dataset in datasets:
            easy, hard = result.easy_hard_sessions[dataset]

            # Get session stats
            for r in result.results:
                if r.dataset == dataset:
                    for session, stats in r.session_stats.items():
                        if session in easy:
                            easy_r2s.append(stats["r2"])
                        elif session in hard:
                            hard_r2s.append(stats["r2"])

        if easy_r2s and hard_r2s:
            x = [0, 1]
            means = [np.mean(easy_r2s), np.mean(hard_r2s)]
            stds = [np.std(easy_r2s), np.std(hard_r2s)]
            colors = [COLORS["easy"], COLORS["hard"]]

            bars = ax.bar(x, means, color=colors, edgecolor='white', linewidth=0.5)
            ax.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=4)

            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(["Easy Sessions", "Hard Sessions"])
            ax.set_ylabel('Test R²')
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center',
                   transform=ax.transAxes)

        ax.set_title("(B) Easy vs Hard Sessions")

    def _plot_difficulty_distribution(self, ax: plt.Axes, result: Any):
        """Plot distribution of session difficulties."""
        all_r2s = []

        for r in result.results:
            if r.split_mode == "inter":
                for stats in r.session_stats.values():
                    all_r2s.append(stats["r2"])

        if not all_r2s:
            ax.text(0.5, 0.5, "No session data available",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("(D) Session Difficulty Distribution")
            return

        ax.hist(all_r2s, bins=20, color=COLORS["gap"], alpha=0.7, edgecolor='white')
        ax.axvline(np.median(all_r2s), color=COLORS["intra"], linestyle='--',
                  label=f'Median: {np.median(all_r2s):.3f}')
        ax.axvline(np.mean(all_r2s), color=COLORS["inter"], linestyle='-.',
                  label=f'Mean: {np.mean(all_r2s):.3f}')

        ax.set_xlabel('Session R²')
        ax.set_ylabel('Count')
        ax.set_title("(D) Session Difficulty Distribution")
        ax.legend(loc='upper left', frameon=False, fontsize=6)

    def plot_figure_4_3(
        self,
        result: Any,
        filename: str = "figure_4_3_statistical_analysis.pdf",
    ) -> Path:
        """Generate Figure 4.3: Statistical Analysis of Intra vs Inter.

        Three panels:
            (A) Box plots: Intra vs Inter for each dataset with fold data
            (B) Effect sizes for the generalization gap
            (C) Statistical significance table

        Args:
            result: Phase4Result object
            filename: Output filename

        Returns:
            Path to saved figure
        """
        apply_nature_style()

        fig = plt.figure(figsize=(7.2, 4.5))
        gs = GridSpec(1, 3, figure=fig, wspace=0.4)

        # (A) Box plots
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_intra_inter_boxplots(ax_a, result)

        # (B) Effect sizes
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_generalization_effect_sizes(ax_b, result)

        # (C) Statistical table
        ax_c = fig.add_subplot(gs[0, 2])
        self._plot_statistical_table(ax_c, result)

        # Add panel labels
        for ax, label in zip([ax_a, ax_b, ax_c], ['A', 'B', 'C']):
            ax.text(-0.15, 1.08, label, transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='top')

        output_path = self.output_dir / filename
        fig.savefig(output_path, format='pdf', bbox_inches='tight')

        png_path = output_path.with_suffix('.png')
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def _plot_intra_inter_boxplots(self, ax: plt.Axes, result: Any):
        """Plot box plots comparing intra vs inter for each dataset."""
        # Collect fold data
        data = {}
        for r in result.results:
            key = f"{r.dataset}:{r.split_mode}"
            if key not in data:
                data[key] = []
            data[key].append(r.test_r2)

        datasets = list(set(r.dataset for r in result.results))
        n_datasets = len(datasets)

        positions = []
        fold_data = []
        colors = []
        labels = []

        for i, dataset in enumerate(datasets):
            intra_key = f"{dataset}:intra"
            inter_key = f"{dataset}:inter"

            if intra_key in data:
                positions.append(i * 3)
                fold_data.append(data[intra_key])
                colors.append(COLORS["intra"])
                labels.append(f"{dataset[:3]} intra")

            if inter_key in data:
                positions.append(i * 3 + 1)
                fold_data.append(data[inter_key])
                colors.append(COLORS["inter"])
                labels.append(f"{dataset[:3]} inter")

        if fold_data:
            bp = ax.boxplot(fold_data, positions=positions, patch_artist=True, widths=0.7, showfliers=False)

            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Overlay points
            for pos, fd in zip(positions, fold_data):
                jitter = np.random.uniform(-0.15, 0.15, len(fd))
                ax.scatter([pos + j for j in jitter], fd, color='black', s=15, alpha=0.5, zorder=3)

            ax.set_xticks([i * 3 + 0.5 for i in range(n_datasets)])
            ax.set_xticklabels([d[:3].upper() for d in datasets], fontsize=7)

        ax.set_ylabel('Test R²')
        ax.set_title('(A) Intra vs Inter by Dataset')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS["intra"], alpha=0.7, label='Intra'),
            Patch(facecolor=COLORS["inter"], alpha=0.7, label='Inter'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=6, frameon=False)

    def _plot_generalization_effect_sizes(self, ax: plt.Axes, result: Any):
        """Plot effect sizes for generalization gap."""
        # Collect fold data for statistical comparison
        effect_data = {}

        for dataset in set(r.dataset for r in result.results):
            intra_r2s = [r.test_r2 for r in result.results if r.dataset == dataset and r.split_mode == "intra"]
            inter_r2s = [r.test_r2 for r in result.results if r.dataset == dataset and r.split_mode == "inter"]

            if len(intra_r2s) >= 2 and len(inter_r2s) >= 2:
                comp = compare_methods(
                    np.array(intra_r2s), np.array(inter_r2s),
                    "intra", "inter", paired=True
                )
                d = comp.parametric_test.effect_size or 0
                effect_data[dataset] = d

        if not effect_data:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("(B) Effect Sizes")
            return

        datasets = list(effect_data.keys())
        effects = [effect_data[d] for d in datasets]
        y_pos = np.arange(len(datasets))

        colors = [COLORS.get(d, COLORS["gap"]) for d in datasets]
        ax.barh(y_pos, effects, color=colors, height=0.6, alpha=0.8)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        for x in [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]:
            ax.axvline(x=x, color='lightgray', linestyle=':', linewidth=0.5, alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([d[:8] for d in datasets], fontsize=6)
        ax.set_xlabel("Cohen's d (Intra vs Inter)")
        ax.set_title("(B) Effect Sizes")
        ax.invert_yaxis()

    def _plot_statistical_table(self, ax: plt.Axes, result: Any):
        """Plot statistical comparison table."""
        ax.axis('off')

        table_data = []
        for dataset in set(r.dataset for r in result.results):
            intra_r2s = [r.test_r2 for r in result.results if r.dataset == dataset and r.split_mode == "intra"]
            inter_r2s = [r.test_r2 for r in result.results if r.dataset == dataset and r.split_mode == "inter"]

            if len(intra_r2s) >= 2 and len(inter_r2s) >= 2:
                comp = compare_methods(
                    np.array(intra_r2s), np.array(inter_r2s),
                    "intra", "inter", paired=True
                )
                p_val = comp.parametric_test.p_value
                d = comp.parametric_test.effect_size or 0
                gap = result.generalization_gaps.get(dataset, 0)

                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

                table_data.append([
                    dataset[:8],
                    f"{gap:.3f}",
                    f"{p_val:.3f}",
                    f"{d:.2f}",
                    sig
                ])

        if not table_data:
            ax.text(0.5, 0.5, "No statistical data", ha='center', va='center', transform=ax.transAxes)
            return

        columns = ['Dataset', 'Gap', 'p-value', "Cohen's d", 'Sig']
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
        )

        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1.2, 1.4)

        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#E6E6E6')
            table[(0, i)].set_text_props(weight='bold')

        ax.set_title('(C) Statistical Comparison', pad=10)

    def plot_all(self, result: Any, format: str = "pdf") -> List[Path]:
        """Generate all Phase 4 figures.

        Args:
            result: Phase4Result object
            format: Output format

        Returns:
            List of paths to saved figures
        """
        paths = []

        paths.append(self.plot_figure_4_1(
            result, filename=f"figure_4_1_intra_inter_session.{format}"
        ))

        paths.append(self.plot_figure_4_2(
            result, filename=f"figure_4_2_session_analysis.{format}"
        ))

        paths.append(self.plot_figure_4_3(
            result, filename=f"figure_4_3_statistical_analysis.{format}"
        ))

        return [p for p in paths if p is not None]


def create_summary_table(result: Any) -> str:
    """Create LaTeX summary table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Phase 4: Inter vs Intra Session Results}",
        r"\label{tab:phase4_results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & Intra R² & Inter R² & Gap \\",
        r"\midrule",
    ]

    for dataset, modes in result.summary.items():
        intra = modes.get("intra", {})
        inter = modes.get("inter", {})

        intra_str = f"{intra.get('test_r2_mean', 0):.4f} $\\pm$ {intra.get('test_r2_std', 0):.4f}"
        inter_str = f"{inter.get('test_r2_mean', 0):.4f} $\\pm$ {inter.get('test_r2_std', 0):.4f}"
        gap = result.generalization_gaps.get(dataset, 0)

        name = DATASET_NAMES.get(dataset, dataset)
        lines.append(f"{name} & {intra_str} & {inter_str} & {gap:.4f} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)
