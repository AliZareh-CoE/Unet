"""
Reporting utilities for Nature Methods publication.

Includes confidence interval computation, export functions, and visualization.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
except ImportError:
    optuna = None


def compute_bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: List of metric values
        confidence: Confidence level (default: 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        (mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)

    if n == 0:
        return 0.0, 0.0, 0.0

    if n == 1:
        return float(values[0]), float(values[0]), float(values[0])

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return float(np.mean(values)), float(ci_lower), float(ci_upper)


def compute_improvement_over_baseline(
    approach_values: List[float],
    baseline_values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
) -> Dict[str, float]:
    """Compute improvement over baseline with confidence intervals.

    Args:
        approach_values: Metric values for the approach
        baseline_values: Metric values for baseline
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with improvement statistics
    """
    rng = np.random.RandomState(42)

    approach = np.array(approach_values)
    baseline = np.array(baseline_values)

    # Absolute improvement
    abs_improvement = np.mean(approach) - np.mean(baseline)

    # Relative improvement
    baseline_mean = np.mean(baseline)
    rel_improvement = (abs_improvement / baseline_mean * 100) if baseline_mean != 0 else 0

    # Bootstrap CI for improvement
    bootstrap_diffs = []
    n_approach = len(approach)
    n_baseline = len(baseline)

    for _ in range(n_bootstrap):
        sample_approach = rng.choice(approach, size=n_approach, replace=True)
        sample_baseline = rng.choice(baseline, size=n_baseline, replace=True)
        bootstrap_diffs.append(np.mean(sample_approach) - np.mean(sample_baseline))

    bootstrap_diffs = np.array(bootstrap_diffs)
    alpha = 1 - confidence

    return {
        "abs_improvement": abs_improvement,
        "rel_improvement_pct": rel_improvement,
        "ci_lower": float(np.percentile(bootstrap_diffs, alpha/2 * 100)),
        "ci_upper": float(np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)),
        "significant": not (
            np.percentile(bootstrap_diffs, alpha/2 * 100) <= 0 <=
            np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
        ),
    }


def export_trials_csv(
    study: "optuna.Study",
    output_path: Path,
    approach_key: str = "approach",
) -> None:
    """Export all trials to CSV for analysis.

    Args:
        study: Optuna study
        output_path: Path to output CSV file
        approach_key: Key for approach identifier in params/attrs
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all parameter names and user attribute names
    all_params = set()
    all_attrs = set()

    for trial in study.trials:
        all_params.update(trial.params.keys())
        all_attrs.update(trial.user_attrs.keys())

    # Sort for consistent ordering
    param_names = sorted(all_params)
    attr_names = sorted(all_attrs)

    # Write CSV
    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "trial_number", "state", "value", "datetime_start", "datetime_complete"
        ] + [f"param_{p}" for p in param_names] + [f"attr_{a}" for a in attr_names]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial in study.trials:
            row = {
                "trial_number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else "",
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else "",
            }

            for p in param_names:
                row[f"param_{p}"] = trial.params.get(p, "")

            for a in attr_names:
                val = trial.user_attrs.get(a, "")
                # Handle JSON-serialized values
                if isinstance(val, str) and val.startswith("["):
                    row[f"attr_{a}"] = val  # Keep as JSON string
                else:
                    row[f"attr_{a}"] = val

            writer.writerow(row)

    print(f"Exported {len(study.trials)} trials to {output_path}")


def export_best_configs(
    study: "optuna.Study",
    output_path: Path,
    n: int = 10,
) -> None:
    """Export top N configurations as JSON.

    Args:
        study: Optuna study
        output_path: Path to output JSON file
        n: Number of top configs to export
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get completed trials sorted by value
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]

    # Sort by value (assuming maximization)
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:n]

    configs = []
    for trial in sorted_trials:
        config = {
            "trial_number": trial.number,
            "value": trial.value,
            "params": dict(trial.params),
            "user_attrs": {
                k: v for k, v in trial.user_attrs.items()
                if not k.startswith("_")  # Skip internal attrs
            },
        }

        # Try to get full config if stored
        full_config = trial.user_attrs.get("full_config")
        if full_config:
            try:
                config["full_config"] = json.loads(full_config)
            except json.JSONDecodeError:
                pass

        configs.append(config)

    with open(output_path, "w") as f:
        json.dump(configs, f, indent=2, default=str)

    print(f"Exported top {len(configs)} configs to {output_path}")


def export_study_results(
    study: "optuna.Study",
    output_dir: Path,
    study_name: str = "study",
) -> Dict[str, Path]:
    """Export comprehensive study results for publication.

    Args:
        study: Optuna study
        output_dir: Directory for output files
        study_name: Name prefix for files

    Returns:
        Dictionary mapping file type to path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    # 1. All trials CSV
    csv_path = output_dir / f"{study_name}_trials.csv"
    export_trials_csv(study, csv_path)
    exported["trials_csv"] = csv_path

    # 2. Best configs JSON
    configs_path = output_dir / f"{study_name}_best_configs.json"
    export_best_configs(study, configs_path, n=10)
    exported["best_configs"] = configs_path

    # 3. Summary statistics
    summary = compute_study_summary(study)
    summary_path = output_dir / f"{study_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    exported["summary"] = summary_path

    # 4. Hyperparameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        importance_path = output_dir / f"{study_name}_importance.json"
        with open(importance_path, "w") as f:
            json.dump(importance, f, indent=2)
        exported["importance"] = importance_path
    except Exception as e:
        print(f"Could not compute importance: {e}")

    print(f"Study results exported to {output_dir}")
    return exported


def compute_study_summary(study: "optuna.Study") -> Dict[str, Any]:
    """Compute summary statistics for a study.

    Args:
        study: Optuna study

    Returns:
        Summary dictionary
    """
    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

    values = [t.value for t in completed if t.value is not None]

    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(trials),
        "n_completed": len(completed),
        "n_pruned": len(pruned),
        "n_failed": len(failed),
        "timestamp": datetime.now().isoformat(),
    }

    if values:
        mean, ci_lower, ci_upper = compute_bootstrap_ci(values)
        summary["best_value"] = max(values) if study.direction.name == "MAXIMIZE" else min(values)
        summary["mean_value"] = mean
        summary["ci_95_lower"] = ci_lower
        summary["ci_95_upper"] = ci_upper
        summary["std_value"] = float(np.std(values))

    if study.best_trial:
        summary["best_trial_number"] = study.best_trial.number
        summary["best_params"] = dict(study.best_trial.params)

    # Per-approach breakdown (if applicable)
    approach_values = {}
    for trial in completed:
        if trial.value is None:
            continue
        approach = trial.params.get("approach") or trial.params.get("conditioning_type") or trial.params.get("spectral_method")
        if approach:
            if approach not in approach_values:
                approach_values[approach] = []
            approach_values[approach].append(trial.value)

    if approach_values:
        summary["per_approach"] = {}
        for approach, vals in approach_values.items():
            mean, ci_lower, ci_upper = compute_bootstrap_ci(vals)
            summary["per_approach"][approach] = {
                "n_trials": len(vals),
                "mean": mean,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "best": max(vals) if study.direction.name == "MAXIMIZE" else min(vals),
            }

    return summary


def generate_comparison_table(
    summaries: Dict[str, Dict[str, Any]],
    baseline_key: str = "baseline",
) -> str:
    """Generate a comparison table in markdown format.

    Args:
        summaries: Dictionary of approach -> summary dict
        baseline_key: Key for baseline approach

    Returns:
        Markdown table string
    """
    if baseline_key not in summaries:
        baseline_mean = 0
    else:
        baseline_mean = summaries[baseline_key].get("mean", 0)

    rows = []
    for approach, summary in sorted(summaries.items()):
        mean = summary.get("mean", 0)
        ci_lower = summary.get("ci_95_lower", 0)
        ci_upper = summary.get("ci_95_upper", 0)
        n = summary.get("n_trials", 0)

        improvement = ((mean - baseline_mean) / baseline_mean * 100) if baseline_mean else 0

        rows.append({
            "approach": approach,
            "n": n,
            "mean": f"{mean:.4f}",
            "ci": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
            "improvement": f"{improvement:+.1f}%" if approach != baseline_key else "-",
        })

    # Generate markdown
    header = "| Approach | N | Mean | 95% CI | Improvement |"
    separator = "|----------|---|------|--------|-------------|"
    body = "\n".join(
        f"| {r['approach']} | {r['n']} | {r['mean']} | {r['ci']} | {r['improvement']} |"
        for r in rows
    )

    return f"{header}\n{separator}\n{body}"
