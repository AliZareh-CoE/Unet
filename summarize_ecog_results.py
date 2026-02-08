#!/usr/bin/env python3
"""Summarize ECoG LOSO fold results into comprehensive tables.

Reads the per-fold JSON files from fold_results/ directories and produces
tables of R², correlation, and delta for both validation and test metrics,
plus baseline (natural) correlations and statistical significance tests.

Usage:
    python summarize_ecog_results.py                              # default: results/ECoG/
    python summarize_ecog_results.py /path/to/results/ECoG        # explicit path
    python summarize_ecog_results.py --csv results_summary.csv    # also save CSV
    python summarize_ecog_results.py --no-details                 # master tables only
"""

import json
import math
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# =============================================================================
# Statistical helpers (no scipy dependency — pure Python fallbacks + scipy when available)
# =============================================================================

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _t_cdf(t_val, df):
    """Approximate two-tailed p-value from t-distribution using the Beta incomplete function.
    Falls back to a rough normal approximation for df > 100."""
    if df <= 0:
        return 1.0
    if df > 100:
        # For large df, t ≈ normal
        # Use complementary error function approximation
        z = abs(t_val)
        p_one_tail = 0.5 * math.erfc(z / math.sqrt(2))
        return 2 * p_one_tail
    # For moderate df, use a rough approximation
    x = df / (df + t_val ** 2)
    # Regularized incomplete beta function approximation (crude but sufficient for reporting)
    # For more accuracy, scipy is preferred
    z = abs(t_val)
    p_one_tail = 0.5 * math.erfc(z / math.sqrt(2))  # normal approx
    return 2 * p_one_tail


def _ttest_1samp(values, popmean=0.0):
    """One-sample t-test: are values significantly different from popmean?"""
    if HAS_SCIPY:
        stat, p = sp_stats.ttest_1samp(values, popmean)
        return float(stat), float(p)
    n = len(values)
    if n < 2:
        return 0.0, 1.0
    m = sum(values) / n
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    if s == 0:
        return float('inf') if m != popmean else 0.0, 0.0 if m != popmean else 1.0
    t = (m - popmean) / (s / math.sqrt(n))
    p = _t_cdf(t, n - 1)
    return t, p


def _ttest_rel(a, b):
    """Paired t-test: are paired differences significantly non-zero?"""
    if HAS_SCIPY:
        stat, p = sp_stats.ttest_rel(a, b)
        return float(stat), float(p)
    diffs = [x - y for x, y in zip(a, b)]
    return _ttest_1samp(diffs, 0.0)


def _wilcoxon(a, b):
    """Wilcoxon signed-rank test on paired differences."""
    if HAS_SCIPY:
        diffs = [x - y for x, y in zip(a, b)]
        nonzero = [d for d in diffs if d != 0]
        if len(nonzero) < 6:
            return None, None  # Too few samples
        try:
            stat, p = sp_stats.wilcoxon(nonzero)
            return float(stat), float(p)
        except Exception:
            return None, None
    return None, None  # Need scipy for Wilcoxon


def _mannwhitneyu(a, b):
    """Mann-Whitney U test (unpaired)."""
    if HAS_SCIPY:
        try:
            stat, p = sp_stats.mannwhitneyu(a, b, alternative='two-sided')
            return float(stat), float(p)
        except Exception:
            return None, None
    return None, None


def _cohens_d(a, b):
    """Cohen's d effect size for paired samples."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return None
    m = sum(diffs) / n
    s = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    if s == 0:
        return float('inf') if m != 0 else 0.0
    return m / s


def _cohens_d_1samp(values, popmean=0.0):
    """Cohen's d for one-sample test."""
    n = len(values)
    if n < 2:
        return None
    m = sum(values) / n
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    if s == 0:
        return float('inf') if m != popmean else 0.0
    return (m - popmean) / s


def significance_stars(p):
    """Convert p-value to significance stars."""
    if p is None:
        return "   "
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "** "
    if p < 0.05:
        return "*  "
    return "n.s."


def fmt_p(p):
    """Format p-value for display."""
    if p is None:
        return "    N/A"
    if p < 0.001:
        return f"  <.001"
    return f"  {p:.3f}"


# =============================================================================
# Natural baseline computation from raw data
# =============================================================================

def _per_channel_normalize(x: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel independently. x: (C, T)."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std[std < 1e-8] = 1e-8
    return (x - mean) / std


def _compute_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mean Pearson correlation across channels. x, y: (C, T)."""
    x_norm = _per_channel_normalize(x)
    y_norm = _per_channel_normalize(y)
    n_ch = min(x_norm.shape[0], y_norm.shape[0])
    corrs = []
    for c in range(n_ch):
        corr = np.corrcoef(x_norm[c], y_norm[c])[0, 1]
        if not np.isnan(corr):
            corrs.append(corr)
    return float(np.mean(corrs)) if corrs else 0.0


def _compute_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mean R2 across channels. x, y: (C, T)."""
    x_norm = _per_channel_normalize(x)
    y_norm = _per_channel_normalize(y)
    n_ch = min(x_norm.shape[0], y_norm.shape[0])
    r2s = []
    for c in range(n_ch):
        ss_res = np.sum((y_norm[c] - x_norm[c]) ** 2)
        ss_tot = np.sum((y_norm[c] - y_norm[c].mean()) ** 2)
        if ss_tot > 1e-8:
            r2s.append(1 - ss_res / ss_tot)
    return float(np.mean(r2s)) if r2s else 0.0


def _compute_windowed_baseline(
    source: np.ndarray,
    target: np.ndarray,
    window_size: int = 5000,
    stride: int = 2500,
) -> dict:
    """Compute natural baseline metrics from raw source/target signals.

    Args:
        source: (C, T) source signals
        target: (C, T) target signals
        window_size: Window size in samples
        stride: Stride between windows

    Returns:
        Dict with corr_mean, corr_std, r2_mean, r2_std, or None if too short
    """
    source = np.nan_to_num(source.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    n_ch = min(source.shape[0], target.shape[0])
    n_samples = min(source.shape[1], target.shape[1])
    source = source[:n_ch, :n_samples]
    target = target[:n_ch, :n_samples]

    n_windows = (n_samples - window_size) // stride + 1
    if n_windows < 1:
        return None

    window_corrs = []
    window_r2s = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window_corrs.append(_compute_correlation(source[:, start:end], target[:, start:end]))
        window_r2s.append(_compute_r2(source[:, start:end], target[:, start:end]))

    return {
        "corr_mean": float(np.mean(window_corrs)),
        "corr_std": float(np.std(window_corrs)),
        "r2_mean": float(np.mean(window_r2s)),
        "r2_std": float(np.std(window_r2s)),
        "n_windows": int(n_windows),
        "n_channels": n_ch,
    }


def compute_natural_baselines(experiment_name: str) -> dict:
    """Compute proper per-channel natural baselines from raw data.

    Detects dataset type and direction from experiment_name, loads raw data
    per subject, and returns per-subject natural correlations.

    Args:
        experiment_name: e.g. "boran_mtl/hippocampus_to_entorhinal_cortex"

    Returns:
        Dict of {subject_id: {"corr": float, "r2": float}} or empty dict on failure
    """
    parts = experiment_name.split("/")
    if len(parts) < 2:
        return {}

    dataset_name = parts[0]
    direction = parts[1]
    dir_parts = direction.split("_to_")
    if len(dir_parts) != 2:
        return {}
    source_region, target_region = dir_parts

    baselines = {}

    try:
        if dataset_name.startswith("boran"):
            from data import list_boran_subjects, load_boran_subject, BORAN_SAMPLING_RATE_HZ
            subjects = list_boran_subjects()
            for subj_id in subjects:
                try:
                    data = load_boran_subject(
                        subj_id,
                        source_region=source_region,
                        target_region=target_region,
                        zscore=False,
                        min_channels=4,
                    )
                    if data is None:
                        continue
                    result = _compute_windowed_baseline(
                        data["source"], data["target"],
                        window_size=5000, stride=2500,
                    )
                    if result:
                        baselines[subj_id] = {
                            "corr": result["corr_mean"],
                            "r2": result["r2_mean"],
                            "corr_std": result["corr_std"],
                            "r2_std": result["r2_std"],
                        }
                except Exception as e:
                    print(f"  Warning: baseline for {subj_id}: {e}")

        elif dataset_name.startswith("ecog") or dataset_name in (
            "motor_imagery", "memory", "fix_cross",
        ):
            from data import (
                _ECOG_DATA_DIR, _enumerate_ecog_recordings,
                _get_ecog_region_channels, _parse_ecog_srate,
            )
            # Try to find the experiment NPZ
            ecog_experiment = dataset_name.replace("ecog_", "") if dataset_name.startswith("ecog_") else dataset_name
            npz_path = _ECOG_DATA_DIR / f"{ecog_experiment}.npz"
            if npz_path.exists():
                alldat = np.load(npz_path, allow_pickle=True)["dat"]
                recordings = _enumerate_ecog_recordings(alldat)
                for row, col, rec_id in recordings:
                    dat = alldat[row, col]
                    if dat is None or not isinstance(dat, dict):
                        continue
                    V = dat.get("V")
                    if V is None or V.size == 0:
                        continue
                    try:
                        src_chs, tgt_chs = _get_ecog_region_channels(dat, source_region, target_region)
                    except Exception:
                        continue
                    if len(src_chs) < 4 or len(tgt_chs) < 4:
                        continue
                    srate = _parse_ecog_srate(dat)
                    source = V[:, src_chs].T.astype(np.float64)
                    target = V[:, tgt_chs].T.astype(np.float64)
                    label = f"s{row:02d}_b{col:02d}"
                    result = _compute_windowed_baseline(
                        source, target,
                        window_size=int(5 * srate), stride=int(2.5 * srate),
                    )
                    if result:
                        baselines[label] = {
                            "corr": result["corr_mean"],
                            "r2": result["r2_mean"],
                            "corr_std": result["corr_std"],
                            "r2_std": result["r2_std"],
                        }

        elif dataset_name.startswith("dandi"):
            from data import list_dandi_subjects, load_dandi_subject
            subjects = list_dandi_subjects()
            for subj_id in subjects:
                try:
                    data = load_dandi_subject(subj_id, source_region, target_region)
                    if data is None:
                        continue
                    result = _compute_windowed_baseline(
                        data["source"], data["target"],
                        window_size=5000, stride=2500,
                    )
                    if result:
                        baselines[subj_id] = {
                            "corr": result["corr_mean"],
                            "r2": result["r2_mean"],
                            "corr_std": result["corr_std"],
                            "r2_std": result["r2_std"],
                        }
                except Exception as e:
                    print(f"  Warning: baseline for {subj_id}: {e}")

    except ImportError as e:
        print(f"  Warning: Cannot compute natural baselines for {dataset_name}: {e}")
        return {}
    except Exception as e:
        print(f"  Warning: Failed computing natural baselines: {e}")
        return {}

    if baselines:
        print(f"  Computed {len(baselines)} natural baselines from raw data ({dataset_name}: {source_region} -> {target_region})")
    return baselines


# =============================================================================
# Core parsing
# =============================================================================

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

    # Per-session test results (contains baseline_corr)
    per_session = data.get("per_session_test_results", [])

    # Extract baseline correlations from per-session test results
    baseline_corrs = [r.get("baseline_corr", 0) for r in per_session if r.get("baseline_corr") is not None]
    baseline_corr = sum(baseline_corrs) / len(baseline_corrs) if baseline_corrs else None

    # Also extract per-session test corrs and r2s
    per_session_test_corrs = {r["session"]: r.get("corr") for r in per_session if "session" in r}
    per_session_test_r2s = {r["session"]: r.get("r2") for r in per_session if "session" in r}
    per_session_baseline_corrs = {r["session"]: r.get("baseline_corr", 0) for r in per_session if "session" in r}

    # Parse fold/seed from filename: fold_<idx>_<subject>_seed<n>_results.json
    name = path.stem
    parts = name.split("_")
    fold_idx = int(parts[1]) if len(parts) > 1 else 0

    # Extract subject name (between fold_<idx>_ and _seed<n>)
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
        "baseline_corr": baseline_corr,
        "per_session_test": per_session,
        "per_session_test_corrs": per_session_test_corrs,
        "per_session_test_r2s": per_session_test_r2s,
        "per_session_baseline_corrs": per_session_baseline_corrs,
        "epochs_trained": data.get("epochs_trained", 0),
        "n_parameters": data.get("n_parameters", 0),
        "architecture": data.get("architecture", "?"),
        "best_epoch": data.get("best_epoch", 0),
        "total_time": data.get("total_time", 0),
    }


# =============================================================================
# Formatting helpers
# =============================================================================

def fmt(val, width=8):
    if val is None:
        return " " * (width - 3) + "N/A"
    return f"{val:>{width}.4f}"


def mean_of(values):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def std_of(values):
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return None
    m = sum(valid) / len(valid)
    return (sum((v - m) ** 2 for v in valid) / (len(valid) - 1)) ** 0.5


def pm(mean, std):
    """Format mean ± std."""
    if mean is None:
        return "         N/A"
    if std is not None:
        return f"{mean:>6.4f}±{std:.4f}"
    return f"{mean:>6.4f}      "


# =============================================================================
# Experiment summarization
# =============================================================================

def summarize_experiment(experiment_name: str, run_dir: Path, use_natural_baselines: bool = True) -> dict:
    """Summarize all folds for one experiment run (one source->target pair).

    Args:
        experiment_name: e.g. "boran_mtl/hippocampus_to_entorhinal_cortex"
        run_dir: Path to the run directory containing fold_results/
        use_natural_baselines: If True, compute proper per-channel baselines
            from raw data instead of using (potentially inflated) fold JSON values.
    """
    jsons = find_fold_jsons(run_dir)
    if not jsons:
        return None

    folds = [parse_fold_json(j) for j in jsons]

    # Compute proper natural baselines from raw data
    # This avoids the inflated mean-signal baselines that older fold JSONs may contain
    natural_baselines = {}
    if use_natural_baselines:
        natural_baselines = compute_natural_baselines(experiment_name)

    # Group by fold (aggregate seeds)
    by_fold = defaultdict(list)
    for f in folds:
        by_fold[f["fold_idx"]].append(f)

    fold_summaries = []
    for fold_idx in sorted(by_fold.keys()):
        seeds = by_fold[fold_idx]
        subject = seeds[0]["subject"]

        # Use natural baseline from raw data if available, else fall back to fold JSON
        if subject in natural_baselines:
            baseline_corr = natural_baselines[subject]["corr"]
            baseline_source = "raw_data"
        else:
            baseline_corr = mean_of([s["baseline_corr"] for s in seeds])
            baseline_source = "fold_json"

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
            # Baseline (natural correlation) — from raw data when available
            "baseline_corr": baseline_corr,
            "baseline_source": baseline_source,
            # Per-seed details
            "seed_test_r2s": [s["test_r2"] for s in seeds],
            "seed_test_corrs": [s["test_corr"] for s in seeds],
            "seed_val_r2s": [s["val_r2"] for s in seeds],
            "seed_val_corrs": [s["val_corr"] for s in seeds],
            "seed_baseline_corrs": [s["baseline_corr"] for s in seeds],
            # Meta
            "epochs": mean_of([s["epochs_trained"] for s in seeds]),
            "best_epoch": mean_of([s["best_epoch"] for s in seeds]),
            "architecture": seeds[0]["architecture"],
        })

    # Collect all baseline corrs (one per fold)
    all_baseline = [f["baseline_corr"] for f in fold_summaries if f["baseline_corr"] is not None]
    all_test_corrs = [f["test_corr"] for f in fold_summaries if f["test_corr"] is not None]
    all_test_r2s = [f["test_r2"] for f in fold_summaries if f["test_r2"] is not None]

    # Recompute delta (test_corr - baseline) using the proper natural baselines
    all_test_deltas = []
    for f in fold_summaries:
        if f["test_corr"] is not None and f["baseline_corr"] is not None:
            delta = f["test_corr"] - f["baseline_corr"]
            f["test_delta"] = delta  # Update fold-level delta too
            all_test_deltas.append(delta)

    # Track which baseline source was used
    baseline_sources = set(f.get("baseline_source", "fold_json") for f in fold_summaries)
    baseline_note = " (from raw data)" if "raw_data" in baseline_sources else " (from fold JSON)"

    return {
        "experiment": experiment_name,
        "run_dir": str(run_dir),
        "n_folds": len(fold_summaries),
        "n_total_jsons": len(folds),
        "folds": fold_summaries,
        "baseline_note": baseline_note,
        # Aggregate across folds
        "mean_test_r2": mean_of(all_test_r2s),
        "std_test_r2": std_of(all_test_r2s),
        "mean_test_corr": mean_of(all_test_corrs),
        "std_test_corr": std_of(all_test_corrs),
        "mean_test_delta": mean_of(all_test_deltas),
        "std_test_delta": std_of(all_test_deltas),
        "mean_val_r2": mean_of([f["val_r2"] for f in fold_summaries]),
        "std_val_r2": std_of([f["val_r2"] for f in fold_summaries]),
        "mean_val_corr": mean_of([f["val_corr"] for f in fold_summaries]),
        "std_val_corr": std_of([f["val_corr"] for f in fold_summaries]),
        # Baseline (natural) correlations
        "mean_baseline_corr": mean_of(all_baseline),
        "std_baseline_corr": std_of(all_baseline),
        # Raw lists for statistical tests
        "_fold_test_r2s": all_test_r2s,
        "_fold_test_corrs": all_test_corrs,
        "_fold_test_deltas": all_test_deltas,
        "_fold_baseline_corrs": all_baseline,
        "_fold_val_r2s": [f["val_r2"] for f in fold_summaries if f["val_r2"] is not None],
        "_fold_val_corrs": [f["val_corr"] for f in fold_summaries if f["val_corr"] is not None],
    }


# =============================================================================
# Display: per-run detail table
# =============================================================================

def print_run_table(summary: dict):
    """Print a detailed table for one source→target run with baseline & stats."""
    exp = summary["experiment"]
    print(f"\n{'─' * 115}")
    print(f"  {exp}  ({summary['n_folds']} folds, {summary['n_total_jsons']} seed runs)")
    print(f"{'─' * 115}")

    hdr = (f"  {'Fold':>4}  {'Subject':<15}"
           f"  {'Baseline':>8}"
           f"  {'Val R²':>8}  {'Val Corr':>8}"
           f"  {'Test R²':>8}  {'Test Corr':>8}  {'Test Δ':>8}"
           f"  {'Seeds':>5}")
    print(hdr)
    sep = f"  {'─'*4}  {'─'*15}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*5}"
    print(sep)

    for f in summary["folds"]:
        print(f"  {f['fold_idx']:>4}  {f['subject']:<15}"
              f"  {fmt(f['baseline_corr'])}"
              f"  {fmt(f['val_r2'])}  {fmt(f['val_corr'])}"
              f"  {fmt(f['test_r2'])}  {fmt(f['test_corr'])}  {fmt(f['test_delta'])}"
              f"  {f['n_seeds']:>5}")

    print(sep)
    print(f"  {'Mean':>4}  {'':15}"
          f"  {fmt(summary['mean_baseline_corr'])}"
          f"  {fmt(summary['mean_val_r2'])}  {fmt(summary['mean_val_corr'])}"
          f"  {fmt(summary['mean_test_r2'])}  {fmt(summary['mean_test_corr'])}  {fmt(summary['mean_test_delta'])}")
    print(f"  {'±Std':>4}  {'':15}"
          f"  {fmt(summary['std_baseline_corr'])}"
          f"  {fmt(summary['std_val_r2'])}  {fmt(summary['std_val_corr'])}"
          f"  {fmt(summary['std_test_r2'])}  {fmt(summary['std_test_corr'])}  {fmt(summary['std_test_delta'])}")

    # Per-run statistical tests
    _print_run_stats(summary)


def _print_run_stats(summary: dict):
    """Print statistical tests for a single run."""
    test_r2s = summary["_fold_test_r2s"]
    test_corrs = summary["_fold_test_corrs"]
    baselines = summary["_fold_baseline_corrs"]

    if len(test_r2s) < 3:
        print(f"\n  (Too few folds for statistical tests)")
        return

    print(f"\n  Statistical Tests:")

    # 1. One-sample t-test: Is test R² > 0?
    t, p = _ttest_1samp(test_r2s, 0.0)
    d = _cohens_d_1samp(test_r2s, 0.0)
    print(f"    Test R² > 0:           t={t:>7.3f}, p={fmt_p(p)}, d={fmt(d)} {significance_stars(p)}")

    # 2. One-sample t-test: Is test Corr > 0?
    t, p = _ttest_1samp(test_corrs, 0.0)
    d = _cohens_d_1samp(test_corrs, 0.0)
    print(f"    Test Corr > 0:         t={t:>7.3f}, p={fmt_p(p)}, d={fmt(d)} {significance_stars(p)}")

    # 3. Paired t-test: Is test Corr > baseline Corr?
    if len(baselines) == len(test_corrs) and len(baselines) >= 3:
        t, p = _ttest_rel(test_corrs, baselines)
        d = _cohens_d(test_corrs, baselines)
        print(f"    Test Corr > Baseline:  t={t:>7.3f}, p={fmt_p(p)}, d={fmt(d)} {significance_stars(p)}")

        # Wilcoxon (non-parametric alternative)
        w, p_w = _wilcoxon(test_corrs, baselines)
        if w is not None:
            print(f"    (Wilcoxon signed-rank):W={w:>7.1f}, p={fmt_p(p_w)}        {significance_stars(p_w)}")


# =============================================================================
# Display: master summary
# =============================================================================

def print_master_table(all_summaries: list):
    print("\n")
    print("=" * 140)
    print("  MASTER SUMMARY — ALL ECoG EXPERIMENTS")
    print("=" * 140)

    hdr = (f"  {'Experiment':<18} {'Direction':<22}"
           f" {'N':>3}"
           f" {'Baseline':>12}"
           f" {'Val R²':>12}  {'Val Corr':>12}"
           f" {'Test R²':>12}  {'Test Corr':>12}  {'Test Δ':>12}"
           f" {'p(R²>0)':>8}")
    print(hdr)
    print(f"  {'─'*18} {'─'*22} {'─'*3} {'─'*12} {'─'*12}  {'─'*12} {'─'*12}  {'─'*12}  {'─'*12} {'─'*8}")

    by_experiment = defaultdict(list)
    for s in all_summaries:
        parts = s["experiment"].split("/")
        exp_name = parts[0] if parts else s["experiment"]
        by_experiment[exp_name].append(s)

    for exp_name in sorted(by_experiment.keys()):
        runs = by_experiment[exp_name]
        for s in sorted(runs, key=lambda x: x["experiment"]):
            parts = s["experiment"].split("/")
            direction = parts[1].replace("_to_", " → ") if len(parts) > 1 else "?"

            # Compute p-value for R² > 0
            r2s = s["_fold_test_r2s"]
            p_r2 = None
            if len(r2s) >= 3:
                _, p_r2 = _ttest_1samp(r2s, 0.0)

            print(f"  {exp_name:<18} {direction:<22}"
                  f" {s['n_folds']:>3}"
                  f" {pm(s['mean_baseline_corr'], s['std_baseline_corr']):>12}"
                  f" {pm(s['mean_val_r2'], s['std_val_r2']):>12}  {pm(s['mean_val_corr'], s['std_val_corr']):>12}"
                  f" {pm(s['mean_test_r2'], s['std_test_r2']):>12}  {pm(s['mean_test_corr'], s['std_test_corr']):>12}"
                  f"  {pm(s['mean_test_delta'], s['std_test_delta']):>12}"
                  f" {fmt_p(p_r2):>8} {significance_stars(p_r2)}")

        print()

    # Overall summary
    all_test_r2 = [s["mean_test_r2"] for s in all_summaries if s["mean_test_r2"] is not None]
    all_test_corr = [s["mean_test_corr"] for s in all_summaries if s["mean_test_corr"] is not None]
    all_baseline = [s["mean_baseline_corr"] for s in all_summaries if s["mean_baseline_corr"] is not None]
    if all_test_r2:
        print(f"  {'─' * 136}")
        print(f"  Overall across {len(all_summaries)} runs:")
        print(f"    Test R²:       {mean_of(all_test_r2):.4f} (range [{min(all_test_r2):.4f}, {max(all_test_r2):.4f}])")
        print(f"    Test Corr:     {mean_of(all_test_corr):.4f} (range [{min(all_test_corr):.4f}, {max(all_test_corr):.4f}])")
        if all_baseline:
            print(f"    Baseline Corr: {mean_of(all_baseline):.4f} (range [{min(all_baseline):.4f}, {max(all_baseline):.4f}])")


# =============================================================================
# Display: natural (baseline) correlation table
# =============================================================================

def print_baseline_table(all_summaries: list):
    """Print table of natural/baseline correlations between regions."""
    # Determine baseline source
    sources = set()
    for s in all_summaries:
        sources.add(s.get("baseline_note", " (from fold JSON)"))
    source_note = ", ".join(sorted(sources))

    print("\n")
    print("=" * 110)
    print("  NATURAL (BASELINE) CORRELATIONS — Per-Channel Raw Source vs Target")
    print(f"  (Before any model — inherent similarity between brain regions){source_note}")
    print("=" * 110)

    hdr = (f"  {'Experiment':<18} {'Direction':<22}"
           f" {'N':>3}"
           f" {'Baseline Corr':>14}"
           f" {'Model Corr':>14}"
           f" {'Improvement':>14}"
           f" {'p(Model>Base)':>14}")
    print(hdr)
    print(f"  {'─'*18} {'─'*22} {'─'*3} {'─'*14} {'─'*14} {'─'*14} {'─'*14}")

    by_experiment = defaultdict(list)
    for s in all_summaries:
        parts = s["experiment"].split("/")
        exp_name = parts[0] if parts else s["experiment"]
        by_experiment[exp_name].append(s)

    for exp_name in sorted(by_experiment.keys()):
        runs = by_experiment[exp_name]
        for s in sorted(runs, key=lambda x: x["experiment"]):
            parts = s["experiment"].split("/")
            direction = parts[1].replace("_to_", " → ") if len(parts) > 1 else "?"

            baselines = s["_fold_baseline_corrs"]
            test_corrs = s["_fold_test_corrs"]
            p_val = None
            if len(baselines) >= 3 and len(test_corrs) >= 3 and len(baselines) == len(test_corrs):
                _, p_val = _ttest_rel(test_corrs, baselines)

            improvement = None
            if s["mean_test_corr"] is not None and s["mean_baseline_corr"] is not None:
                improvement = s["mean_test_corr"] - s["mean_baseline_corr"]

            print(f"  {exp_name:<18} {direction:<22}"
                  f" {s['n_folds']:>3}"
                  f" {pm(s['mean_baseline_corr'], s['std_baseline_corr']):>14}"
                  f" {pm(s['mean_test_corr'], s['std_test_corr']):>14}"
                  f" {fmt(improvement, 14)}"
                  f" {fmt_p(p_val):>10} {significance_stars(p_val)}")

        print()

    # Build region-pair baseline matrix
    _print_baseline_matrix(all_summaries)


def _print_baseline_matrix(all_summaries: list):
    """Print a matrix of natural baseline correlations between region pairs."""
    # Collect all baselines keyed by (experiment, source, target)
    region_baselines = defaultdict(list)  # (experiment, src, tgt) -> [baseline values]
    all_regions = set()

    for s in all_summaries:
        parts = s["experiment"].split("/")
        if len(parts) < 2:
            continue
        exp_name = parts[0]
        direction = parts[1]
        dir_parts = direction.split("_to_")
        if len(dir_parts) != 2:
            continue
        src, tgt = dir_parts

        all_regions.add(src)
        all_regions.add(tgt)
        for b in s["_fold_baseline_corrs"]:
            region_baselines[(exp_name, src, tgt)].append(b)

    if not region_baselines:
        return

    # Group by experiment
    experiments = sorted(set(k[0] for k in region_baselines.keys()))
    regions = sorted(all_regions)

    for exp in experiments:
        print(f"\n  Baseline Correlation Matrix — {exp}")
        print(f"  (rows=source, cols=target, values=mean natural corr)")

        # Header
        col_w = max(10, max(len(r) for r in regions) + 2)
        hdr = f"  {'Source↓ Target→':<16}" + "".join(f"{r:>{col_w}}" for r in regions)
        print(hdr)
        print(f"  {'─'*16}" + "─" * (col_w * len(regions)))

        for src in regions:
            row = f"  {src:<16}"
            for tgt in regions:
                if src == tgt:
                    row += f"{'—':>{col_w}}"
                else:
                    vals = region_baselines.get((exp, src, tgt), [])
                    if vals:
                        m = sum(vals) / len(vals)
                        row += f"{m:>{col_w}.4f}"
                    else:
                        row += f"{'':>{col_w}}"
            print(row)


# =============================================================================
# Display: bidirectional comparison with stats
# =============================================================================

def print_direction_comparison(all_summaries: list):
    print("\n")
    print("=" * 130)
    print("  BIDIRECTIONAL COMPARISON (A→B vs B→A) WITH STATISTICAL TESTS")
    print("=" * 130)

    pairs = {}
    for s in all_summaries:
        parts = s["experiment"].split("/")
        if len(parts) < 2:
            continue
        exp_name = parts[0]
        direction = parts[1]
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

    print(f"  {'Experiment':<18}"
          f" {'A → B':<18} {'R²':>7} {'Corr':>7}"
          f"  {'B → A':<18} {'R²':>7} {'Corr':>7}"
          f"  {'Δ R²':>7} {'p-val':>7} {'':>4}"
          f"  {'Δ Corr':>7} {'p-val':>7} {'':>4}")
    print(f"  {'─'*18}"
          f" {'─'*18} {'─'*7} {'─'*7}"
          f"  {'─'*18} {'─'*7} {'─'*7}"
          f"  {'─'*7} {'─'*7} {'─'*4}"
          f"  {'─'*7} {'─'*7} {'─'*4}")

    for (exp_name, region_pair), directions in sorted(pairs.items()):
        keys = sorted(directions.keys())
        if len(keys) < 2:
            d = keys[0]
            s = directions[d]
            lbl = f"{d[0]} → {d[1]}"
            print(f"  {exp_name:<18}"
                  f" {lbl:<18} {fmt(s['mean_test_r2'],7)} {fmt(s['mean_test_corr'],7)}"
                  f"  {'(pending)':<18} {'':>7} {'':>7}"
                  f"  {'':>7} {'':>7} {'':>4}"
                  f"  {'':>7} {'':>7} {'':>4}")
            continue

        d1, d2 = keys[0], keys[1]
        s1, s2 = directions[d1], directions[d2]
        lbl1 = f"{d1[0]} → {d1[1]}"
        lbl2 = f"{d2[0]} → {d2[1]}"

        # Statistical test on per-fold R² values
        r2_1 = s1["_fold_test_r2s"]
        r2_2 = s2["_fold_test_r2s"]
        corr_1 = s1["_fold_test_corrs"]
        corr_2 = s2["_fold_test_corrs"]

        delta_r2 = None
        if s1["mean_test_r2"] is not None and s2["mean_test_r2"] is not None:
            delta_r2 = s1["mean_test_r2"] - s2["mean_test_r2"]
        delta_corr = None
        if s1["mean_test_corr"] is not None and s2["mean_test_corr"] is not None:
            delta_corr = s1["mean_test_corr"] - s2["mean_test_corr"]

        # Use Mann-Whitney U since folds may not be paired across directions
        # (different subjects might be excluded in different directions)
        p_r2 = None
        p_corr = None
        if len(r2_1) >= 3 and len(r2_2) >= 3:
            # Check if same subjects (paired) or different (unpaired)
            subjects_1 = {f["subject"] for f in s1["folds"]}
            subjects_2 = {f["subject"] for f in s2["folds"]}
            if subjects_1 == subjects_2 and len(r2_1) == len(r2_2):
                # Paired comparison (same subjects in both directions)
                _, p_r2 = _ttest_rel(r2_1, r2_2)
                _, p_corr = _ttest_rel(corr_1, corr_2)
            else:
                # Unpaired comparison
                _, p_r2 = _mannwhitneyu(r2_1, r2_2)
                _, p_corr = _mannwhitneyu(corr_1, corr_2)

        print(f"  {exp_name:<18}"
              f" {lbl1:<18} {fmt(s1['mean_test_r2'],7)} {fmt(s1['mean_test_corr'],7)}"
              f"  {lbl2:<18} {fmt(s2['mean_test_r2'],7)} {fmt(s2['mean_test_corr'],7)}"
              f"  {fmt(delta_r2,7)} {fmt_p(p_r2):>7} {significance_stars(p_r2)}"
              f"  {fmt(delta_corr,7)} {fmt_p(p_corr):>7} {significance_stars(p_corr)}")


# =============================================================================
# Display: global statistical summary
# =============================================================================

def print_global_stats(all_summaries: list):
    """Print global statistical tests across all runs."""
    print("\n")
    print("=" * 90)
    print("  GLOBAL STATISTICAL SUMMARY")
    print("=" * 90)

    # Collect all per-fold test R² and corr across ALL runs
    all_fold_r2 = []
    all_fold_corr = []
    all_fold_baseline = []
    all_fold_delta = []

    for s in all_summaries:
        all_fold_r2.extend(s["_fold_test_r2s"])
        all_fold_corr.extend(s["_fold_test_corrs"])
        all_fold_baseline.extend(s["_fold_baseline_corrs"])
        all_fold_delta.extend(s["_fold_test_deltas"])

    n = len(all_fold_r2)
    print(f"\n  Total fold-level observations: {n} (across {len(all_summaries)} direction-runs)")

    if n < 3:
        print("  Too few observations for global statistics.")
        return

    # Test R² > 0 globally
    t, p = _ttest_1samp(all_fold_r2, 0.0)
    print(f"\n  Global Test R² > 0:")
    print(f"    Mean ± Std: {mean_of(all_fold_r2):.4f} ± {std_of(all_fold_r2):.4f}")
    print(f"    One-sample t-test: t={t:.3f}, p={fmt_p(p)} {significance_stars(p)}")

    # Test Corr > 0 globally
    t, p = _ttest_1samp(all_fold_corr, 0.0)
    print(f"\n  Global Test Corr > 0:")
    print(f"    Mean ± Std: {mean_of(all_fold_corr):.4f} ± {std_of(all_fold_corr):.4f}")
    print(f"    One-sample t-test: t={t:.3f}, p={fmt_p(p)} {significance_stars(p)}")

    # Test Corr > Baseline globally (paired)
    if len(all_fold_baseline) == len(all_fold_corr) and len(all_fold_baseline) >= 3:
        t, p = _ttest_rel(all_fold_corr, all_fold_baseline)
        d = _cohens_d(all_fold_corr, all_fold_baseline)
        w, pw = _wilcoxon(all_fold_corr, all_fold_baseline)
        print(f"\n  Global Test Corr > Baseline Corr:")
        print(f"    Baseline Mean ± Std: {mean_of(all_fold_baseline):.4f} ± {std_of(all_fold_baseline):.4f}")
        print(f"    Model Mean ± Std:    {mean_of(all_fold_corr):.4f} ± {std_of(all_fold_corr):.4f}")
        print(f"    Paired t-test: t={t:.3f}, p={fmt_p(p)}, Cohen's d={fmt(d)} {significance_stars(p)}")
        if w is not None:
            print(f"    Wilcoxon signed-rank: W={w:.1f}, p={fmt_p(pw)} {significance_stars(pw)}")

    # Test Delta > 0 (improvement over baseline)
    if len(all_fold_delta) >= 3:
        t, p = _ttest_1samp(all_fold_delta, 0.0)
        print(f"\n  Global Test Δ (Corr - Baseline) > 0:")
        print(f"    Mean ± Std: {mean_of(all_fold_delta):.4f} ± {std_of(all_fold_delta):.4f}")
        print(f"    One-sample t-test: t={t:.3f}, p={fmt_p(p)} {significance_stars(p)}")

    print(f"\n  Significance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. p≥0.05")
    if not HAS_SCIPY:
        print(f"  NOTE: scipy not found — using normal approximation for p-values.")
        print(f"         Install scipy for exact t-distribution and Wilcoxon tests: pip install scipy")


# =============================================================================
# CSV export
# =============================================================================

def _write_csv(rows: list, csv_path: Path):
    """Write list of dicts to CSV."""
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
    print(f"  Saved: {csv_path}")


def save_csv(all_summaries: list, csv_path: Path):
    rows = []
    for s in all_summaries:
        parts = s["experiment"].split("/")
        exp_name = parts[0] if parts else s["experiment"]
        direction = parts[1] if len(parts) > 1 else "?"
        dir_parts = direction.split("_to_")
        source = dir_parts[0] if len(dir_parts) == 2 else "?"
        target = dir_parts[1] if len(dir_parts) == 2 else "?"

        # p-value for R² > 0
        p_r2 = None
        if len(s["_fold_test_r2s"]) >= 3:
            _, p_r2 = _ttest_1samp(s["_fold_test_r2s"], 0.0)

        # p-value for corr > baseline
        p_corr_vs_base = None
        if (len(s["_fold_test_corrs"]) >= 3 and
                len(s["_fold_baseline_corrs"]) == len(s["_fold_test_corrs"])):
            _, p_corr_vs_base = _ttest_rel(s["_fold_test_corrs"], s["_fold_baseline_corrs"])

        rows.append({
            "experiment": exp_name,
            "source": source,
            "target": target,
            "direction": direction.replace("_to_", " → "),
            "n_folds": s["n_folds"],
            "mean_baseline_corr": s["mean_baseline_corr"],
            "std_baseline_corr": s["std_baseline_corr"],
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
            "p_r2_gt_0": p_r2,
            "p_corr_gt_baseline": p_corr_vs_base,
        })

    _write_csv(rows, csv_path)


def save_per_fold_csv(all_summaries: list, csv_path: Path):
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
                "baseline_corr": fold["baseline_corr"],
                "val_r2": fold["val_r2"],
                "val_corr": fold["val_corr"],
                "test_r2": fold["test_r2"],
                "test_corr": fold["test_corr"],
                "test_delta": fold["test_delta"],
            })

    _write_csv(rows, csv_path)


# =============================================================================
# Main
# =============================================================================

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
    parser.add_argument("--no-recompute-baselines", action="store_true",
                        help="Use baseline_corr from fold JSONs instead of recomputing from raw data")
    args = parser.parse_args()

    root = Path(args.results_dir)
    if not root.exists():
        print(f"ERROR: Results directory not found: {root}")
        print(f"  Run ECoG LOSO experiments first, or specify the correct path.")
        sys.exit(1)

    # Discover all experiment/direction combos
    all_summaries = []
    n_found = 0
    n_empty = 0

    experiments = sorted([d for d in root.iterdir() if d.is_dir()])
    if not experiments:
        print(f"ERROR: No experiment directories found in {root}")
        sys.exit(1)

    print(f"Scanning: {root}")
    print(f"Experiments found: {[e.name for e in experiments]}")

    use_natural = not args.no_recompute_baselines
    if use_natural:
        print("Computing natural baselines from raw data (use --no-recompute-baselines to skip)")

    for exp_dir in experiments:
        direction_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
        for dir_dir in direction_dirs:
            if dir_dir.name == "fold_results":
                name = exp_dir.name
                summary = summarize_experiment(name, exp_dir, use_natural_baselines=use_natural)
            else:
                name = f"{exp_dir.name}/{dir_dir.name}"
                summary = summarize_experiment(name, dir_dir, use_natural_baselines=use_natural)

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
    print(f"scipy available: {HAS_SCIPY}")

    # 1. Detailed per-run tables (with per-run stats)
    if not args.no_details:
        for s in all_summaries:
            print_run_table(s)

    # 2. Master summary table
    print_master_table(all_summaries)

    # 3. Natural (baseline) correlation table
    print_baseline_table(all_summaries)

    # 4. Bidirectional comparison with stats
    print_direction_comparison(all_summaries)

    # 5. Global statistical summary
    print_global_stats(all_summaries)

    # 6. Save CSVs
    csv_path = args.csv or str(root / "ecog_summary.csv")
    save_csv(all_summaries, Path(csv_path))

    per_fold_path = args.per_fold_csv or str(root / "ecog_per_fold.csv")
    save_per_fold_csv(all_summaries, Path(per_fold_path))

    print(f"\nDone.")


if __name__ == "__main__":
    main()
