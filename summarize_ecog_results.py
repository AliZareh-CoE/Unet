#!/usr/bin/env python3
"""Summarize ECoG LOSO fold results into comprehensive tables.

Reads the per-fold JSON files from fold_results/ directories and produces
tables of R², correlation, and delta for both validation and test metrics.

Usage:
    python summarize_ecog_results.py                              # default: results/ECoG/
    python summarize_ecog_results.py /path/to/results/ECoG        # explicit path
    python summarize_ecog_results.py --csv results_summary.csv    # also save CSV
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def find_fold_jsons(experiment_dir: Path):
    """Find all fold result JSON files in a fold_results/ subdirectory."""
    fold_dir = experiment_dir / "fold_results"
    if not fold_dir.exists():
        return []
    return sorted(fold_dir.glob("fold_*_results.json"))


def parse_fold_json(path: Path) -> dict:
    """Parse a single fold result JSON and extract key metrics."""
    with open(path) as f:
        data = json.load(f)

    # Extract test metrics (held-out subject)
    test_r2 = data.get("test_avg_r2")
    test_corr = data.get("test_avg_corr")
    test_delta = data.get("test_avg_delta")

    # Extract validation metrics (from model selection)
    val_r2 = data.get("best_val_r2")
    val_corr = data.get("best_val_corr")
    val_loss = data.get("best_val_loss")

    # Per-session test results
    per_session = data.get("per_session_test_results", [])

    # Parse fold/seed from filename: fold_<idx>_<subject>_seed<n>_results.json
    name = path.stem  # e.g. fold_0_sub-bp_seed0_results
    parts = name.split("_")
    fold_idx = int(parts[1]) if len(parts) > 1 else 0

    # Extract subject name (everything between fold_<idx>_ and _seed<n>)
    # Handle subject names that may contain underscores
    seed_part = [i for i, p in enumerate(parts) if p.startswith("seed")]
    if seed_part:
        subject = "_".join(parts[2:seed_part[0]])
        seed_idx = int(parts[seed_part[0]].replace("seed", ""))
    else:
        subject = "_".join(parts[2:-1])
        seed_idx = 0

    return {
        "file": path.name,
        "fold_idx": fold_idx,
        "subject": subject,
        "seed_idx": seed_idx,
        "test_r2": test_r2,
        "test_corr": test_corr,
        "test_delta": test_delta,
        "val_r2": val_r2,
        "val_corr": val_corr,
        "val_loss": val_loss,
        "per_session_test": per_session,
        "epochs_trained": data.get("epochs_trained", 0),
        "n_parameters": data.get("n_parameters", 0),
        "architecture": data.get("architecture", "?"),
        "best_epoch": data.get("best_epoch", 0),
        "total_time": data.get("total_time", 0),
    }


def fmt(val, width=8):
    """Format a metric value."""
    if val is None:
        return " " * (width - 3) + "N/A"
    return f"{val:>{width}.4f}"


def fmt_pct(val, width=8):
    """Format as percentage."""
    if val is None:
        return " " * (width - 3) + "N/A"
    return f"{val * 100:>{width}.2f}%"


def mean_of(values):
    """Mean ignoring None."""
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def std_of(values):
    """Std dev ignoring None."""
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return None
    m = sum(valid) / len(valid)
    return (sum((v - m) ** 2 for v in valid) / (len(valid) - 1)) ** 0.5


def summarize_experiment(experiment_name: str, run_dir: Path) -> dict:
    """Summarize all folds for one experiment run (one source→target pair)."""
    jsons = find_fold_jsons(run_dir)
    if not jsons:
        return None

    folds = [parse_fold_json(j) for j in jsons]

    # Group by fold (aggregate seeds)
    by_fold = defaultdict(list)
    for f in folds:
        by_fold[f["fold_idx"]].append(f)

    fold_summaries = []
    for fold_idx in sorted(by_fold.keys()):
        seeds = by_fold[fold_idx]
        subject = seeds[0]["subject"]

        fold_summaries.append({
            "fold_idx": fold_idx,
            "subject": subject,
            "n_seeds": len(seeds),
            # Test metrics (mean across seeds)
            "test_r2": mean_of([s["test_r2"] for s in seeds]),
            "test_corr": mean_of([s["test_corr"] for s in seeds]),
            "test_delta": mean_of([s["test_delta"] for s in seeds]),
            # Validation metrics (mean across seeds)
            "val_r2": mean_of([s["val_r2"] for s in seeds]),
            "val_corr": mean_of([s["val_corr"] for s in seeds]),
            "val_loss": mean_of([s["val_loss"] for s in seeds]),
            # Per-seed details
            "seed_test_r2s": [s["test_r2"] for s in seeds],
            "seed_test_corrs": [s["test_corr"] for s in seeds],
            "seed_val_r2s": [s["val_r2"] for s in seeds],
            "seed_val_corrs": [s["val_corr"] for s in seeds],
            # Meta
            "epochs": mean_of([s["epochs_trained"] for s in seeds]),
            "best_epoch": mean_of([s["best_epoch"] for s in seeds]),
            "architecture": seeds[0]["architecture"],
        })

    return {
        "experiment": experiment_name,
        "run_dir": str(run_dir),
        "n_folds": len(fold_summaries),
        "n_total_jsons": len(folds),
        "folds": fold_summaries,
        # Aggregate across folds
        "mean_test_r2": mean_of([f["test_r2"] for f in fold_summaries]),
        "std_test_r2": std_of([f["test_r2"] for f in fold_summaries]),
        "mean_test_corr": mean_of([f["test_corr"] for f in fold_summaries]),
        "std_test_corr": std_of([f["test_corr"] for f in fold_summaries]),
        "mean_test_delta": mean_of([f["test_delta"] for f in fold_summaries]),
        "std_test_delta": std_of([f["test_delta"] for f in fold_summaries]),
        "mean_val_r2": mean_of([f["val_r2"] for f in fold_summaries]),
        "std_val_r2": std_of([f["val_r2"] for f in fold_summaries]),
        "mean_val_corr": mean_of([f["val_corr"] for f in fold_summaries]),
        "std_val_corr": std_of([f["val_corr"] for f in fold_summaries]),
    }


def print_run_table(summary: dict):
    """Print a detailed table for one source→target run."""
    exp = summary["experiment"]
    print(f"\n{'─' * 100}")
    print(f"  {exp}  ({summary['n_folds']} folds, {summary['n_total_jsons']} seed runs)")
    print(f"{'─' * 100}")

    # Header
    hdr = (f"  {'Fold':>4}  {'Subject':<15}"
           f"  {'Val R²':>8}  {'Val Corr':>8}"
           f"  {'Test R²':>8}  {'Test Corr':>8}  {'Test Δ':>8}"
           f"  {'Seeds':>5}")
    print(hdr)
    print(f"  {'─' * 4}  {'─' * 15}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 5}")

    for f in summary["folds"]:
        print(f"  {f['fold_idx']:>4}  {f['subject']:<15}"
              f"  {fmt(f['val_r2'])}  {fmt(f['val_corr'])}"
              f"  {fmt(f['test_r2'])}  {fmt(f['test_corr'])}  {fmt(f['test_delta'])}"
              f"  {f['n_seeds']:>5}")

    # Mean ± std
    print(f"  {'─' * 4}  {'─' * 15}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 5}")
    print(f"  {'Mean':>4}  {'':15}"
          f"  {fmt(summary['mean_val_r2'])}  {fmt(summary['mean_val_corr'])}"
          f"  {fmt(summary['mean_test_r2'])}  {fmt(summary['mean_test_corr'])}  {fmt(summary['mean_test_delta'])}")
    print(f"  {'±Std':>4}  {'':15}"
          f"  {fmt(summary['std_val_r2'])}  {fmt(summary['std_val_corr'])}"
          f"  {fmt(summary['std_test_r2'])}  {fmt(summary['std_test_corr'])}  {fmt(summary['std_test_delta'])}")


def print_master_table(all_summaries: list):
    """Print master summary table across all experiments and region pairs."""
    print("\n")
    print("=" * 120)
    print("  MASTER SUMMARY — ALL ECoG EXPERIMENTS")
    print("=" * 120)

    hdr = (f"  {'Experiment':<18} {'Direction':<25}"
           f" {'Folds':>5}"
           f" {'Val R²':>12}  {'Val Corr':>12}"
           f" {'Test R²':>12}  {'Test Corr':>12}  {'Test Δ':>12}")
    print(hdr)
    print(f"  {'─' * 18} {'─' * 25} {'─' * 5} {'─' * 12}  {'─' * 12} {'─' * 12}  {'─' * 12}  {'─' * 12}")

    # Group by experiment
    by_experiment = defaultdict(list)
    for s in all_summaries:
        parts = s["experiment"].split("/")
        exp_name = parts[0] if parts else s["experiment"]
        by_experiment[exp_name].append(s)

    for exp_name in sorted(by_experiment.keys()):
        runs = by_experiment[exp_name]
        for s in sorted(runs, key=lambda x: x["experiment"]):
            # Parse direction from experiment name
            parts = s["experiment"].split("/")
            direction = parts[1].replace("_to_", " → ") if len(parts) > 1 else "?"

            def pm(mean, std):
                if mean is None:
                    return "         N/A"
                if std is not None:
                    return f"{mean:>6.4f}±{std:.4f}"
                return f"{mean:>6.4f}      "

            print(f"  {exp_name:<18} {direction:<25}"
                  f" {s['n_folds']:>5}"
                  f" {pm(s['mean_val_r2'], s['std_val_r2']):>12}  {pm(s['mean_val_corr'], s['std_val_corr']):>12}"
                  f" {pm(s['mean_test_r2'], s['std_test_r2']):>12}  {pm(s['mean_test_corr'], s['std_test_corr']):>12}"
                  f"  {pm(s['mean_test_delta'], s['std_test_delta']):>12}")

        # Separator between experiments
        print()

    # Overall summary
    all_test_r2 = [s["mean_test_r2"] for s in all_summaries if s["mean_test_r2"] is not None]
    all_test_corr = [s["mean_test_corr"] for s in all_summaries if s["mean_test_corr"] is not None]
    if all_test_r2:
        print(f"  {'─' * 116}")
        print(f"  Overall across {len(all_summaries)} runs: "
              f"Test R² = {mean_of(all_test_r2):.4f} (range [{min(all_test_r2):.4f}, {max(all_test_r2):.4f}]), "
              f"Test Corr = {mean_of(all_test_corr):.4f} (range [{min(all_test_corr):.4f}, {max(all_test_corr):.4f}])")


def print_direction_comparison(all_summaries: list):
    """Print table comparing forward vs reverse directions."""
    print("\n")
    print("=" * 100)
    print("  BIDIRECTIONAL COMPARISON (A→B vs B→A)")
    print("=" * 100)

    # Find pairs
    pairs = {}
    for s in all_summaries:
        parts = s["experiment"].split("/")
        if len(parts) < 2:
            continue
        exp_name = parts[0]
        direction = parts[1]  # e.g. "frontal_to_temporal"

        dir_parts = direction.split("_to_")
        if len(dir_parts) != 2:
            continue
        src, tgt = dir_parts
        key = tuple(sorted([src, tgt]))
        pair_key = (exp_name, key)

        if pair_key not in pairs:
            pairs[pair_key] = {}
        pairs[pair_key][(src, tgt)] = s

    if not pairs:
        print("  No bidirectional pairs found yet.")
        return

    print(f"  {'Experiment':<18} {'A → B':<20} {'Test R²':>8} {'Test Corr':>9}"
          f"  {'B → A':<20} {'Test R²':>8} {'Test Corr':>9}  {'Better?':>10}")
    print(f"  {'─' * 18} {'─' * 20} {'─' * 8} {'─' * 9}  {'─' * 20} {'─' * 8} {'─' * 9}  {'─' * 10}")

    for (exp_name, region_pair), directions in sorted(pairs.items()):
        keys = sorted(directions.keys())
        if len(keys) < 2:
            # Only one direction available so far
            d = keys[0]
            s = directions[d]
            print(f"  {exp_name:<18} {d[0]:>8} → {d[1]:<9} {fmt(s['mean_test_r2'])} {fmt(s['mean_test_corr'],9)}"
                  f"  {'(pending)':>20} {'':>8} {'':>9}  {'':>10}")
            continue

        d1, d2 = keys[0], keys[1]
        s1, s2 = directions[d1], directions[d2]

        # Determine which direction is better
        r2_1 = s1["mean_test_r2"] or -999
        r2_2 = s2["mean_test_r2"] or -999
        if r2_1 > r2_2:
            better = f"{d1[0]}→{d1[1]}"
        elif r2_2 > r2_1:
            better = f"{d2[0]}→{d2[1]}"
        else:
            better = "tie"

        print(f"  {exp_name:<18} {d1[0]:>8} → {d1[1]:<9} {fmt(s1['mean_test_r2'])} {fmt(s1['mean_test_corr'],9)}"
              f"  {d2[0]:>8} → {d2[1]:<9} {fmt(s2['mean_test_r2'])} {fmt(s2['mean_test_corr'],9)}"
              f"  {better:>10}")


def save_csv(all_summaries: list, csv_path: Path):
    """Save all results to a CSV file."""
    rows = []
    for s in all_summaries:
        parts = s["experiment"].split("/")
        exp_name = parts[0] if parts else s["experiment"]
        direction = parts[1] if len(parts) > 1 else "?"
        dir_parts = direction.split("_to_")
        source = dir_parts[0] if len(dir_parts) == 2 else "?"
        target = dir_parts[1] if len(dir_parts) == 2 else "?"

        # Master row
        rows.append({
            "experiment": exp_name,
            "source": source,
            "target": target,
            "direction": direction.replace("_to_", " → "),
            "n_folds": s["n_folds"],
            "mean_val_r2": s["mean_val_r2"],
            "std_val_r2": s["std_val_r2"],
            "mean_val_corr": s["mean_val_corr"],
            "std_val_corr": s["std_val_corr"],
            "mean_test_r2": s["mean_test_r2"],
            "std_test_r2": s["std_test_r2"],
            "mean_test_corr": s["mean_test_corr"],
            "std_test_corr": s["std_test_corr"],
            "mean_test_delta": s["mean_test_delta"],
            "std_test_delta": s["std_test_delta"],
        })

    # Write CSV manually (no pandas dependency)
    if not rows:
        print("  No data to save.")
        return

    headers = list(rows[0].keys())
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            vals = []
            for h in headers:
                v = row[h]
                if v is None:
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

    print(f"\n  CSV saved to: {csv_path}")


def save_per_fold_csv(all_summaries: list, csv_path: Path):
    """Save per-fold breakdown to CSV."""
    rows = []
    for s in all_summaries:
        parts = s["experiment"].split("/")
        exp_name = parts[0] if parts else s["experiment"]
        direction = parts[1] if len(parts) > 1 else "?"

        for fold in s["folds"]:
            rows.append({
                "experiment": exp_name,
                "direction": direction.replace("_to_", " → "),
                "fold": fold["fold_idx"],
                "subject": fold["subject"],
                "n_seeds": fold["n_seeds"],
                "val_r2": fold["val_r2"],
                "val_corr": fold["val_corr"],
                "test_r2": fold["test_r2"],
                "test_corr": fold["test_corr"],
                "test_delta": fold["test_delta"],
            })

    if not rows:
        return

    headers = list(rows[0].keys())
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            vals = []
            for h in headers:
                v = row[h]
                if v is None:
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

    print(f"  Per-fold CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize ECoG LOSO fold results")
    parser.add_argument("results_dir", nargs="?", default="results/ECoG",
                        help="Path to ECoG results root (default: results/ECoG)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save summary CSV to this path")
    parser.add_argument("--per-fold-csv", type=str, default=None,
                        help="Save per-fold CSV to this path")
    parser.add_argument("--no-details", action="store_true",
                        help="Skip per-fold detail tables, only show master summary")
    args = parser.parse_args()

    root = Path(args.results_dir)
    if not root.exists():
        print(f"ERROR: Results directory not found: {root}")
        print(f"  Run ECoG LOSO experiments first, or specify the correct path.")
        sys.exit(1)

    # Discover all experiment/direction combos
    # Structure: results/ECoG/<experiment>/<source_to_target>/fold_results/*.json
    all_summaries = []
    n_found = 0
    n_empty = 0

    experiments = sorted([d for d in root.iterdir() if d.is_dir()])
    if not experiments:
        print(f"ERROR: No experiment directories found in {root}")
        sys.exit(1)

    print(f"Scanning: {root}")
    print(f"Experiments found: {[e.name for e in experiments]}")

    for exp_dir in experiments:
        direction_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
        for dir_dir in direction_dirs:
            if dir_dir.name == "fold_results":
                # Single-level: results/ECoG/<experiment>/fold_results/
                name = exp_dir.name
                summary = summarize_experiment(name, exp_dir)
            else:
                # Two-level: results/ECoG/<experiment>/<direction>/fold_results/
                name = f"{exp_dir.name}/{dir_dir.name}"
                summary = summarize_experiment(name, dir_dir)

            if summary is not None:
                all_summaries.append(summary)
                n_found += 1
            else:
                n_empty += 1

    if not all_summaries:
        print(f"\nNo fold result JSONs found anywhere under {root}")
        print("  Expected structure: results/ECoG/<experiment>/<src_to_tgt>/fold_results/fold_*_results.json")
        sys.exit(1)

    print(f"\nFound {n_found} completed runs ({n_empty} directories with no results yet)")

    # Print detailed per-run tables
    if not args.no_details:
        for s in all_summaries:
            print_run_table(s)

    # Print master summary
    print_master_table(all_summaries)

    # Print bidirectional comparison
    print_direction_comparison(all_summaries)

    # Save CSV if requested
    csv_path = args.csv or str(root / "ecog_summary.csv")
    save_csv(all_summaries, Path(csv_path))

    per_fold_path = args.per_fold_csv or str(root / "ecog_per_fold.csv")
    save_per_fold_csv(all_summaries, Path(per_fold_path))

    print(f"\nDone.")


if __name__ == "__main__":
    main()
