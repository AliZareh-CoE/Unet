"""
Nature Methods Publication Figures
==================================

Generate publication-quality figures for the dual-track HPO results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed, plotting disabled")

from .phase1_screening import analyze_phase1, rank_approaches, compute_improvements_over_baseline
from .phase2_deep_dive import analyze_phase2, pairwise_comparison, extract_track_b_metrics
from .confidence_intervals import bootstrap_ci


# Nature Methods style settings
NATURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": (7, 5),
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1,
}

# Color palette
COLORS = {
    "baseline": "#7f7f7f",  # Gray
    "cpc": "#1f77b4",       # Blue
    "vqvae": "#ff7f0e",     # Orange
    "vib": "#2ca02c",       # Green
    "freq_disentangled": "#d62728",  # Red
    "cycle_latent": "#9467bd",       # Purple
    "phase_preserving": "#1f77b4",
    "adaptive_gated": "#ff7f0e",
    "wavelet": "#2ca02c",
    "spectral_loss": "#d62728",
    "iterative": "#9467bd",
}


def apply_nature_style():
    """Apply Nature Methods style to matplotlib."""
    if HAS_MATPLOTLIB:
        plt.rcParams.update(NATURE_STYLE)


def plot_phase1_comparison(
    track: str = "a",
    storage_dir: str = "artifacts",
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Generate Phase 1 comparison bar plot.

    Args:
        track: "a" or "b"
        storage_dir: Artifacts directory
        output_path: Save path (optional)

    Returns:
        Figure or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    apply_nature_style()

    results = analyze_phase1(track, storage_dir)
    if not results:
        print(f"No Phase 1 results for Track {track}")
        return None

    ranking = rank_approaches(results)
    approaches = [a for a, _ in ranking]
    values = [v for _, v in ranking]

    # Get CIs
    ci_lowers = [results[a]["ci_lower"] for a in approaches]
    ci_uppers = [results[a]["ci_upper"] for a in approaches]
    errors = [[v - l for v, l in zip(values, ci_lowers)],
              [u - v for u, v in zip(ci_uppers, values)]]

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 3))

    colors = [COLORS.get(a, "#333333") for a in approaches]
    bars = ax.barh(range(len(approaches)), values, xerr=errors, capsize=2,
                   color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(approaches)))
    ax.set_yticklabels(approaches)
    ax.invert_yaxis()

    track_name = "Conditioning" if track == "a" else "SpectralShift"
    metric_name = "R²" if track == "a" else "Composite Score"
    ax.set_xlabel(f"Best {metric_name}")
    ax.set_title(f"Phase 1: {track_name} Approaches")

    # Add baseline reference line
    if "baseline" in results:
        ax.axvline(results["baseline"]["best_value"], color="gray",
                   linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {output_path}")

    return fig


def plot_phase2_boxplot(
    track: str = "a",
    storage_dir: str = "artifacts",
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Generate Phase 2 box plot comparison.

    Args:
        track: "a" or "b"
        storage_dir: Artifacts directory
        output_path: Save path

    Returns:
        Figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    apply_nature_style()

    results = analyze_phase2(track, storage_dir)
    if not results:
        print(f"No Phase 2 results for Track {track}")
        return None

    approaches = list(results.keys())
    data = [results[a]["values"] for a in approaches]

    fig, ax = plt.subplots(figsize=(4, 3))

    bp = ax.boxplot(data, labels=approaches, patch_artist=True)

    # Color boxes
    for i, (patch, approach) in enumerate(zip(bp["boxes"], approaches)):
        patch.set_facecolor(COLORS.get(approach, "#333333"))
        patch.set_alpha(0.7)

    track_name = "Conditioning" if track == "a" else "SpectralShift"
    metric_name = "R²" if track == "a" else "Composite Score"
    ax.set_ylabel(metric_name)
    ax.set_title(f"Phase 2: {track_name} (96 trials each)")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {output_path}")

    return fig


def plot_improvement_over_baseline(
    track: str = "a",
    storage_dir: str = "artifacts",
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot improvements over baseline with CIs.

    Args:
        track: "a" or "b"
        storage_dir: Artifacts directory
        output_path: Save path

    Returns:
        Figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    apply_nature_style()

    results = analyze_phase1(track, storage_dir)
    if not results:
        return None

    improvements = compute_improvements_over_baseline(results)
    if not improvements:
        return None

    # Sort by improvement
    sorted_items = sorted(improvements.items(), key=lambda x: -x[1]["mean_improvement"])
    approaches = [a for a, _ in sorted_items]
    means = [improvements[a]["mean_improvement"] for a in approaches]
    ci_lowers = [improvements[a]["ci_lower"] for a in approaches]
    ci_uppers = [improvements[a]["ci_upper"] for a in approaches]
    errors = [[m - l for m, l in zip(means, ci_lowers)],
              [u - m for u, m in zip(ci_uppers, means)]]

    fig, ax = plt.subplots(figsize=(4, 3))

    colors = [COLORS.get(a, "#333333") for a in approaches]

    # Determine significance
    for i, (approach, imp) in enumerate(sorted_items):
        edgecolor = "black" if imp["significant"] else "none"
        ax.barh(i, imp["mean_improvement"],
                xerr=[[means[i] - ci_lowers[i]], [ci_uppers[i] - means[i]]],
                color=colors[i], edgecolor=edgecolor, linewidth=1.5, capsize=2)

    ax.set_yticks(range(len(approaches)))
    ax.set_yticklabels(approaches)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)

    track_name = "Conditioning" if track == "a" else "SpectralShift"
    metric_name = "R²" if track == "a" else "Score"
    ax.set_xlabel(f"{metric_name} Improvement vs Baseline")
    ax.set_title(f"Track {track.upper()}: Improvements over Baseline")

    # Add significance legend
    sig_patch = mpatches.Patch(facecolor="white", edgecolor="black", linewidth=1.5,
                               label="p < 0.05")
    ax.legend(handles=[sig_patch], loc="lower right", frameon=False)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {output_path}")

    return fig


def plot_track_b_pareto(
    storage_dir: str = "artifacts",
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot Track B Pareto front (PSD improvement vs R² preservation).

    Args:
        storage_dir: Artifacts directory
        output_path: Save path

    Returns:
        Figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    apply_nature_style()

    results = analyze_phase2("b", storage_dir)
    if not results:
        return None

    fig, ax = plt.subplots(figsize=(4, 3.5))

    for approach, data in results.items():
        trials = data.get("trials", [])

        psd_imps = []
        r2_changes = []

        for trial in trials:
            attrs = trial.get("user_attrs", {})
            if "psd_improvement_db" in attrs and "r2_change" in attrs:
                psd_imps.append(attrs["psd_improvement_db"])
                r2_changes.append(attrs["r2_change"])

        if psd_imps:
            color = COLORS.get(approach, "#333333")
            ax.scatter(r2_changes, psd_imps, c=color, alpha=0.5, s=20, label=approach)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Highlight good region
    ax.fill_between([-0.02, 0.02], [0, 0], [10, 10], alpha=0.1, color="green")

    ax.set_xlabel("R² Change (vs Stage 1)")
    ax.set_ylabel("PSD Improvement (dB)")
    ax.set_title("Track B: PSD vs R² Trade-off")
    ax.legend(loc="upper left", frameon=False, fontsize=6)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {output_path}")

    return fig


def plot_hyperparameter_importance(
    track: str = "a",
    approach: str = "cpc",
    storage_dir: str = "artifacts",
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot hyperparameter importance.

    Args:
        track: "a" or "b"
        approach: Approach name
        storage_dir: Artifacts directory
        output_path: Save path

    Returns:
        Figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    from .phase2_deep_dive import analyze_hyperparameter_importance

    apply_nature_style()

    importances = analyze_hyperparameter_importance(track, storage_dir, approach)
    if not importances:
        return None

    # Sort by importance
    sorted_items = sorted(importances.items(), key=lambda x: -x[1])[:10]  # Top 10
    params = [p for p, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(4, 3))

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(params)))
    ax.barh(range(len(params)), values, color=colors)

    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"HP Importance: {approach}")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {output_path}")

    return fig


def generate_all_figures(
    storage_dir: str = "artifacts",
    output_dir: str = "artifacts/figures",
) -> None:
    """Generate all Nature Methods figures.

    Args:
        storage_dir: Artifacts directory
        output_dir: Output directory
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for figure generation")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating Nature Methods figures...")

    # Track A figures
    print("\nTrack A (Conditioning):")
    plot_phase1_comparison("a", storage_dir, output_path / "track_a_phase1.pdf")
    plot_phase2_boxplot("a", storage_dir, output_path / "track_a_phase2_boxplot.pdf")
    plot_improvement_over_baseline("a", storage_dir, output_path / "track_a_improvements.pdf")

    # Track B figures
    print("\nTrack B (SpectralShift):")
    plot_phase1_comparison("b", storage_dir, output_path / "track_b_phase1.pdf")
    plot_phase2_boxplot("b", storage_dir, output_path / "track_b_phase2_boxplot.pdf")
    plot_improvement_over_baseline("b", storage_dir, output_path / "track_b_improvements.pdf")
    plot_track_b_pareto(storage_dir, output_path / "track_b_pareto.pdf")

    print(f"\nAll figures saved to: {output_path}")


def generate_summary_table(
    storage_dir: str = "artifacts",
    output_path: Optional[str] = None,
) -> str:
    """Generate summary table for Nature Methods.

    Args:
        storage_dir: Artifacts directory
        output_path: Save path for CSV

    Returns:
        Table as string
    """
    lines = []

    # Header
    lines.append("Track,Approach,Phase,Best,Mean,95% CI Lower,95% CI Upper,Significant")

    for track in ["a", "b"]:
        track_name = "Conditioning" if track == "a" else "SpectralShift"

        # Phase 1
        phase1 = analyze_phase1(track, storage_dir)
        improvements1 = compute_improvements_over_baseline(phase1) if phase1 else {}

        for approach, data in (phase1 or {}).items():
            sig = "Yes" if improvements1.get(approach, {}).get("significant", False) else "No"
            if approach == "baseline":
                sig = "-"
            lines.append(
                f"{track_name},{approach},1,"
                f"{data.get('best_value', 'NA'):.4f},"
                f"{data.get('mean_value', 'NA'):.4f},"
                f"{data.get('ci_lower', 'NA'):.4f},"
                f"{data.get('ci_upper', 'NA'):.4f},"
                f"{sig}"
            )

        # Phase 2
        phase2 = analyze_phase2(track, storage_dir)
        for approach, data in (phase2 or {}).items():
            ci = data.get("mean_ci", (None, None))
            lines.append(
                f"{track_name},{approach},2,"
                f"{data.get('best_value', 'NA'):.4f},"
                f"{data.get('mean', 'NA'):.4f},"
                f"{ci[0]:.4f if ci[0] else 'NA'},"
                f"{ci[1]:.4f if ci[1] else 'NA'},"
                "-"
            )

    table = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(table)
        print(f"Saved: {output_path}")

    return table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Nature Methods figures")
    parser.add_argument("--storage-dir", type=str, default="artifacts",
                       help="Artifacts directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/figures",
                       help="Output directory")
    parser.add_argument("--table-only", action="store_true",
                       help="Only generate summary table")

    args = parser.parse_args()

    if args.table_only:
        table = generate_summary_table(
            args.storage_dir,
            f"{args.output_dir}/summary_table.csv"
        )
        print("\nSummary Table:")
        print(table)
    else:
        generate_all_figures(args.storage_dir, args.output_dir)
        generate_summary_table(
            args.storage_dir,
            f"{args.output_dir}/summary_table.csv"
        )
