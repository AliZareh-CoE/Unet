"""
Phase 1 Screening Analysis
==========================

Analyze Phase 1 (coarse screening) results and rank approaches for Phase 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
except ImportError:
    optuna = None

from .confidence_intervals import bootstrap_ci, format_ci_string, compute_effect_size


# Track approach lists
TRACK_A_APPROACHES = ["baseline", "cpc", "vqvae", "freq_disentangled", "cycle_latent", "vib"]
TRACK_B_APPROACHES = ["baseline", "phase_preserving", "adaptive_gated", "wavelet", "spectral_loss", "iterative"]


def load_study_results(
    storage_path: str,
    study_name: str,
) -> Optional[Dict]:
    """Load study results from Optuna storage.

    Args:
        storage_path: Path to SQLite storage
        study_name: Name of the study

    Returns:
        Dictionary with study results or None if not found
    """
    if optuna is None:
        print("Optuna not installed")
        return None

    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{storage_path}",
        )
    except Exception as e:
        print(f"Could not load study {study_name}: {e}")
        return None

    # Extract trial data
    trials = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trials.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            })

    if not trials:
        return None

    values = [t["value"] for t in trials]

    return {
        "study_name": study_name,
        "n_trials": len(trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "mean_value": np.mean(values),
        "std_value": np.std(values),
        "trials": trials,
        "values": values,
    }


def analyze_phase1(
    track: str = "a",
    storage_dir: str = "artifacts",
    n_bootstrap: int = 10000,
) -> Dict[str, Dict]:
    """Analyze Phase 1 results for all approaches.

    Args:
        track: "a" for conditioning, "b" for spectral
        storage_dir: Base artifacts directory
        n_bootstrap: Bootstrap iterations for CI

    Returns:
        Dictionary mapping approach names to results
    """
    if track.lower() == "a":
        storage_path = f"{storage_dir}/optuna_conditioning/phase1.db"
        approaches = TRACK_A_APPROACHES
        prefix = "conditioning"
    else:
        storage_path = f"{storage_dir}/optuna_spectral/phase1.db"
        approaches = TRACK_B_APPROACHES
        prefix = "spectral"

    results = {}

    for approach in approaches:
        study_name = f"{prefix}_{approach}_phase1"
        study_results = load_study_results(storage_path, study_name)

        if study_results is None:
            print(f"No results for {approach}")
            continue

        # Compute confidence intervals
        values = np.array(study_results["values"])
        mean, ci_lower, ci_upper = bootstrap_ci(values, np.mean, n_bootstrap)

        results[approach] = {
            **study_results,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_string": format_ci_string(mean, ci_lower, ci_upper),
        }

    return results


def rank_approaches(
    results: Dict[str, Dict],
    metric: str = "best_value",
    higher_is_better: bool = True,
) -> List[Tuple[str, float]]:
    """Rank approaches by a metric.

    Args:
        results: Results from analyze_phase1
        metric: Metric to rank by
        higher_is_better: Whether higher values are better

    Returns:
        List of (approach, value) sorted by rank
    """
    items = []
    for approach, data in results.items():
        if metric in data:
            items.append((approach, data[metric]))

    # Sort
    items.sort(key=lambda x: x[1], reverse=higher_is_better)

    return items


def compute_improvements_over_baseline(
    results: Dict[str, Dict],
    n_bootstrap: int = 10000,
) -> Dict[str, Dict]:
    """Compute improvements relative to baseline with CIs.

    Args:
        results: Results from analyze_phase1
        n_bootstrap: Bootstrap iterations

    Returns:
        Dictionary of improvements for each approach
    """
    if "baseline" not in results:
        print("No baseline results found")
        return {}

    baseline_values = np.array(results["baseline"]["values"])
    improvements = {}

    for approach, data in results.items():
        if approach == "baseline":
            continue

        approach_values = np.array(data["values"])

        # Compute effect size
        effect = compute_effect_size(approach_values, baseline_values)

        # Bootstrap CI for difference
        from .confidence_intervals import bootstrap_ci_difference
        diff, diff_ci_lower, diff_ci_upper = bootstrap_ci_difference(
            approach_values, baseline_values, n_bootstrap
        )

        improvements[approach] = {
            "mean_improvement": diff,
            "ci_lower": diff_ci_lower,
            "ci_upper": diff_ci_upper,
            "ci_string": format_ci_string(diff, diff_ci_lower, diff_ci_upper),
            "significant": diff_ci_lower > 0 or diff_ci_upper < 0,  # CI doesn't include 0
            **effect,
        }

    return improvements


def select_top_k(
    results: Dict[str, Dict],
    k: int = 3,
    exclude_baseline: bool = True,
) -> List[str]:
    """Select top k approaches for Phase 2.

    Args:
        results: Results from analyze_phase1
        k: Number of approaches to select
        exclude_baseline: Whether to exclude baseline from selection

    Returns:
        List of top k approach names
    """
    ranking = rank_approaches(results, metric="best_value", higher_is_better=True)

    if exclude_baseline:
        ranking = [(a, v) for a, v in ranking if a != "baseline"]

    return [approach for approach, _ in ranking[:k]]


def generate_phase1_report(
    track: str = "a",
    storage_dir: str = "artifacts",
    output_dir: Optional[str] = None,
) -> str:
    """Generate Phase 1 screening report.

    Args:
        track: "a" or "b"
        storage_dir: Artifacts directory
        output_dir: Output directory (optional)

    Returns:
        Report text
    """
    results = analyze_phase1(track, storage_dir)

    if not results:
        return f"No Phase 1 results found for Track {track.upper()}"

    track_name = "Conditioning (A)" if track.lower() == "a" else "SpectralShift (B)"

    lines = [
        f"Phase 1 Screening Report: Track {track_name}",
        "=" * 60,
        "",
    ]

    # Summary table
    lines.append("Approach Results (sorted by best value):")
    lines.append("-" * 60)
    lines.append(f"{'Approach':<20} {'Best':<10} {'Mean':<10} {'95% CI':<25}")
    lines.append("-" * 60)

    ranking = rank_approaches(results)
    for approach, best_val in ranking:
        data = results[approach]
        lines.append(
            f"{approach:<20} {best_val:<10.4f} {data['mean_value']:<10.4f} {data['ci_string']:<25}"
        )

    lines.append("")

    # Improvements over baseline
    improvements = compute_improvements_over_baseline(results)
    if improvements:
        lines.append("Improvements over Baseline:")
        lines.append("-" * 60)
        lines.append(f"{'Approach':<20} {'Improvement':<12} {'95% CI':<25} {'Sig?':<5}")
        lines.append("-" * 60)

        for approach, imp in sorted(improvements.items(), key=lambda x: -x[1]["mean_improvement"]):
            sig = "*" if imp["significant"] else ""
            lines.append(
                f"{approach:<20} {imp['mean_improvement']:+.4f}      {imp['ci_string']:<25} {sig}"
            )

    lines.append("")

    # Top 3 recommendation
    top3 = select_top_k(results, k=3)
    lines.append("Recommended for Phase 2:")
    lines.append("-" * 60)
    for i, approach in enumerate(top3, 1):
        lines.append(f"  {i}. {approach}")

    report = "\n".join(lines)

    # Save if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / f"phase1_report_track_{track}.txt", "w") as f:
            f.write(report)

        # Also save JSON
        with open(output_path / f"phase1_results_track_{track}.json", "w") as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for approach, data in results.items():
                json_results[approach] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in data.items()
                    if k != "trials"  # Skip trial list for brevity
                }
            json.dump(json_results, f, indent=2, default=str)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 Screening Analysis")
    parser.add_argument("--track", type=str, default="both", choices=["a", "b", "both"],
                       help="Track to analyze")
    parser.add_argument("--storage-dir", type=str, default="artifacts",
                       help="Artifacts directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/analysis",
                       help="Output directory")

    args = parser.parse_args()

    tracks = ["a", "b"] if args.track == "both" else [args.track]

    for track in tracks:
        print(f"\n{'='*60}")
        print(f"  Analyzing Track {track.upper()}")
        print(f"{'='*60}\n")

        report = generate_phase1_report(
            track=track,
            storage_dir=args.storage_dir,
            output_dir=args.output_dir,
        )
        print(report)
