"""
Phase 2 Deep Dive Analysis
==========================

Detailed analysis of Phase 2 results with confidence intervals,
hyperparameter importance, and Nature Methods reporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
    from optuna.importance import get_param_importances
except ImportError:
    optuna = None
    get_param_importances = None

from .confidence_intervals import (
    bootstrap_ci,
    paired_bootstrap_test,
    format_ci_string,
    compute_effect_size,
    summarize_with_ci,
)
from .phase1_screening import load_study_results, TRACK_A_APPROACHES, TRACK_B_APPROACHES


def analyze_phase2(
    track: str = "a",
    storage_dir: str = "artifacts",
    approaches: Optional[List[str]] = None,
    n_bootstrap: int = 10000,
) -> Dict[str, Dict]:
    """Analyze Phase 2 deep dive results.

    Args:
        track: "a" for conditioning, "b" for spectral
        storage_dir: Base artifacts directory
        approaches: Specific approaches to analyze (None = all)
        n_bootstrap: Bootstrap iterations

    Returns:
        Dictionary mapping approach names to detailed results
    """
    if track.lower() == "a":
        storage_path = f"{storage_dir}/optuna_conditioning/phase2.db"
        default_approaches = ["cpc", "vqvae", "vib"]
        prefix = "conditioning"
    else:
        storage_path = f"{storage_dir}/optuna_spectral/phase2.db"
        default_approaches = ["phase_preserving", "adaptive_gated", "wavelet"]
        prefix = "spectral"

    if approaches is None:
        approaches = default_approaches

    results = {}

    for approach in approaches:
        study_name = f"{prefix}_{approach}_phase2"
        study_results = load_study_results(storage_path, study_name)

        if study_results is None:
            print(f"No Phase 2 results for {approach}")
            continue

        # Detailed statistics
        values = np.array(study_results["values"])

        # Bootstrap CIs for multiple statistics
        mean_ci = bootstrap_ci(values, np.mean, n_bootstrap)
        std_ci = bootstrap_ci(values, np.std, n_bootstrap)
        median_ci = bootstrap_ci(values, np.median, n_bootstrap)

        # Top 10% analysis
        top_10_pct = np.percentile(values, 90)
        top_values = values[values >= top_10_pct]

        results[approach] = {
            **study_results,
            "mean": mean_ci[0],
            "mean_ci": (mean_ci[1], mean_ci[2]),
            "mean_ci_string": format_ci_string(*mean_ci),
            "std": std_ci[0],
            "median": median_ci[0],
            "median_ci": (median_ci[1], median_ci[2]),
            "top_10_pct_threshold": top_10_pct,
            "top_10_pct_mean": np.mean(top_values) if len(top_values) > 0 else None,
            "top_10_pct_count": len(top_values),
        }

    return results


def compare_best_configs(
    results: Dict[str, Dict],
) -> Dict[str, Any]:
    """Compare best configurations across approaches.

    Args:
        results: Results from analyze_phase2

    Returns:
        Comparison dictionary
    """
    comparison = {
        "best_overall": None,
        "best_value": -float("inf"),
        "rankings": [],
        "configs": {},
    }

    for approach, data in results.items():
        best_val = data.get("best_value", -float("inf"))
        comparison["rankings"].append((approach, best_val))
        comparison["configs"][approach] = data.get("best_params", {})

        if best_val > comparison["best_value"]:
            comparison["best_value"] = best_val
            comparison["best_overall"] = approach

    # Sort rankings
    comparison["rankings"].sort(key=lambda x: x[1], reverse=True)

    return comparison


def analyze_hyperparameter_importance(
    track: str = "a",
    storage_dir: str = "artifacts",
    approach: str = "cpc",
) -> Optional[Dict[str, float]]:
    """Compute hyperparameter importance for an approach.

    Args:
        track: "a" or "b"
        storage_dir: Artifacts directory
        approach: Approach name

    Returns:
        Dictionary mapping param names to importance scores
    """
    if optuna is None or get_param_importances is None:
        print("Optuna not available for importance analysis")
        return None

    if track.lower() == "a":
        storage_path = f"{storage_dir}/optuna_conditioning/phase2.db"
        prefix = "conditioning"
    else:
        storage_path = f"{storage_dir}/optuna_spectral/phase2.db"
        prefix = "spectral"

    study_name = f"{prefix}_{approach}_phase2"

    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{storage_path}",
        )
        importances = get_param_importances(study)
        return dict(importances)
    except Exception as e:
        print(f"Could not compute importance for {approach}: {e}")
        return None


def pairwise_comparison(
    results: Dict[str, Dict],
    n_bootstrap: int = 10000,
) -> Dict[Tuple[str, str], Dict]:
    """Perform pairwise statistical comparisons.

    Args:
        results: Results from analyze_phase2
        n_bootstrap: Bootstrap iterations

    Returns:
        Dictionary mapping (approach_a, approach_b) to comparison results
    """
    comparisons = {}
    approaches = list(results.keys())

    for i, approach_a in enumerate(approaches):
        for approach_b in approaches[i+1:]:
            values_a = np.array(results[approach_a]["values"])
            values_b = np.array(results[approach_b]["values"])

            # Use the smaller sample size for paired comparison
            n = min(len(values_a), len(values_b))
            values_a = values_a[:n]
            values_b = values_b[:n]

            diff, p_value = paired_bootstrap_test(
                values_a, values_b, n_bootstrap=n_bootstrap
            )

            effect = compute_effect_size(values_a, values_b)

            comparisons[(approach_a, approach_b)] = {
                "difference": diff,
                "p_value": p_value,
                "significant_05": p_value < 0.05,
                "significant_01": p_value < 0.01,
                "winner": approach_a if diff > 0 else approach_b,
                **effect,
            }

    return comparisons


def extract_track_b_metrics(
    results: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Extract Track B specific metrics (PSD improvement, R² preservation).

    Args:
        results: Results from analyze_phase2

    Returns:
        Dictionary with Track B metrics
    """
    metrics = {}

    for approach, data in results.items():
        trials = data.get("trials", [])

        psd_improvements = []
        r2_changes = []

        for trial in trials:
            user_attrs = trial.get("user_attrs", {})
            if "psd_improvement_db" in user_attrs:
                psd_improvements.append(user_attrs["psd_improvement_db"])
            if "r2_change" in user_attrs:
                r2_changes.append(user_attrs["r2_change"])

        if psd_improvements:
            metrics[approach] = {
                **summarize_with_ci(psd_improvements, "psd_improvement_db"),
                **summarize_with_ci(r2_changes, "r2_change"),
            }

    return metrics


def generate_phase2_report(
    track: str = "a",
    storage_dir: str = "artifacts",
    output_dir: Optional[str] = None,
) -> str:
    """Generate Phase 2 deep dive report.

    Args:
        track: "a" or "b"
        storage_dir: Artifacts directory
        output_dir: Output directory

    Returns:
        Report text
    """
    results = analyze_phase2(track, storage_dir)

    if not results:
        return f"No Phase 2 results found for Track {track.upper()}"

    track_name = "Conditioning (A)" if track.lower() == "a" else "SpectralShift (B)"

    lines = [
        f"Phase 2 Deep Dive Report: Track {track_name}",
        "=" * 70,
        "",
    ]

    # Summary table
    lines.append("Approach Results (96 trials each):")
    lines.append("-" * 70)
    lines.append(f"{'Approach':<20} {'Best':<10} {'Mean (95% CI)':<30} {'Top 10%':<10}")
    lines.append("-" * 70)

    for approach, data in sorted(results.items(), key=lambda x: -x[1].get("best_value", 0)):
        best = data.get("best_value", 0)
        mean_ci = data.get("mean_ci_string", "N/A")
        top10 = data.get("top_10_pct_mean", 0)
        lines.append(f"{approach:<20} {best:<10.4f} {mean_ci:<30} {top10:.4f}")

    lines.append("")

    # Pairwise comparisons
    comparisons = pairwise_comparison(results)
    if comparisons:
        lines.append("Pairwise Statistical Comparisons:")
        lines.append("-" * 70)
        lines.append(f"{'Comparison':<30} {'Diff':<10} {'p-value':<10} {'Cohen d':<10} {'Sig?':<5}")
        lines.append("-" * 70)

        for (a, b), comp in sorted(comparisons.items(), key=lambda x: x[1]["p_value"]):
            sig = "**" if comp["significant_01"] else ("*" if comp["significant_05"] else "")
            lines.append(
                f"{a} vs {b:<15} {comp['difference']:+.4f}    "
                f"{comp['p_value']:.4f}     {comp['cohens_d']:+.3f}     {sig}"
            )

    lines.append("")

    # Best overall
    comparison = compare_best_configs(results)
    lines.append(f"Best Overall: {comparison['best_overall']} ({comparison['best_value']:.4f})")
    lines.append("")

    # Best configuration
    best_approach = comparison["best_overall"]
    if best_approach and best_approach in comparison["configs"]:
        lines.append(f"Best Configuration ({best_approach}):")
        lines.append("-" * 70)
        for param, value in comparison["configs"][best_approach].items():
            lines.append(f"  {param}: {value}")

    lines.append("")

    # Track B specific metrics
    if track.lower() == "b":
        track_b_metrics = extract_track_b_metrics(results)
        if track_b_metrics:
            lines.append("Track B Specific Metrics:")
            lines.append("-" * 70)
            lines.append(f"{'Approach':<20} {'PSD Improvement (dB)':<25} {'R² Change':<20}")
            lines.append("-" * 70)

            for approach, metrics in track_b_metrics.items():
                psd = metrics.get("psd_improvement_db_ci_string", "N/A")
                r2 = metrics.get("r2_change_ci_string", "N/A")
                lines.append(f"{approach:<20} {psd:<25} {r2:<20}")

    report = "\n".join(lines)

    # Save if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / f"phase2_report_track_{track}.txt", "w") as f:
            f.write(report)

        # Save detailed JSON
        json_results = {}
        for approach, data in results.items():
            json_results[approach] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else (
                    list(v) if isinstance(v, np.ndarray) else v
                )
                for k, v in data.items()
                if k != "trials"
            }

        with open(output_path / f"phase2_results_track_{track}.json", "w") as f:
            json.dump(json_results, f, indent=2, default=str)

    return report


def export_for_nature_methods(
    storage_dir: str = "artifacts",
    output_dir: str = "artifacts/nature_methods",
) -> None:
    """Export results in Nature Methods format.

    Args:
        storage_dir: Artifacts directory
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Analyze both tracks
    for track in ["a", "b"]:
        track_name = "conditioning" if track == "a" else "spectral"

        # Phase 1
        from .phase1_screening import analyze_phase1, compute_improvements_over_baseline
        phase1 = analyze_phase1(track, storage_dir)

        if phase1:
            improvements = compute_improvements_over_baseline(phase1)

            with open(output_path / f"{track_name}_phase1.json", "w") as f:
                json.dump({
                    "results": {k: {
                        "best": v.get("best_value"),
                        "mean": v.get("mean_value"),
                        "ci": v.get("ci_string"),
                    } for k, v in phase1.items()},
                    "improvements": improvements,
                }, f, indent=2, default=str)

        # Phase 2
        phase2 = analyze_phase2(track, storage_dir)

        if phase2:
            comparisons = pairwise_comparison(phase2)
            best_configs = compare_best_configs(phase2)

            with open(output_path / f"{track_name}_phase2.json", "w") as f:
                json.dump({
                    "results": {k: {
                        "best": v.get("best_value"),
                        "mean_ci": v.get("mean_ci_string"),
                        "top_10_pct": v.get("top_10_pct_mean"),
                    } for k, v in phase2.items()},
                    "pairwise_comparisons": {
                        f"{a}_vs_{b}": {
                            "p_value": c["p_value"],
                            "cohens_d": c["cohens_d"],
                            "significant": c["significant_05"],
                        }
                        for (a, b), c in comparisons.items()
                    },
                    "best_overall": best_configs["best_overall"],
                    "best_config": best_configs["configs"].get(best_configs["best_overall"]),
                }, f, indent=2, default=str)

    print(f"Nature Methods export saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 Deep Dive Analysis")
    parser.add_argument("--track", type=str, default="both", choices=["a", "b", "both"],
                       help="Track to analyze")
    parser.add_argument("--storage-dir", type=str, default="artifacts",
                       help="Artifacts directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/analysis",
                       help="Output directory")
    parser.add_argument("--export-nature", action="store_true",
                       help="Export in Nature Methods format")

    args = parser.parse_args()

    if args.export_nature:
        export_for_nature_methods(args.storage_dir, args.output_dir)
    else:
        tracks = ["a", "b"] if args.track == "both" else [args.track]

        for track in tracks:
            print(f"\n{'='*70}")
            print(f"  Phase 2 Analysis: Track {track.upper()}")
            print(f"{'='*70}\n")

            report = generate_phase2_report(
                track=track,
                storage_dir=args.storage_dir,
                output_dir=args.output_dir,
            )
            print(report)
