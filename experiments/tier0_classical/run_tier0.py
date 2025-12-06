#!/usr/bin/env python3
"""
Tier 0: Classical Floor (GATE)
==============================

Establishes the minimum performance bar that neural methods must beat.
Uses 5-fold CV for robust estimation.

GATE CONDITION: Neural methods in Tier 1+ must beat best classical by 0.10 R²

Usage:
    python experiments/tier0_classical/run_tier0.py
    python experiments/tier0_classical/run_tier0.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# CRITICAL: Fix Python path for imports
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.common.cross_validation import create_data_splits
from experiments.common.config_registry import (
    ClassicalResult,
    Tier0Result,
    get_registry,
)
from experiments.study5_classical.baselines import create_baseline, BASELINE_REGISTRY


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier0_classical"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# All baselines to evaluate - these are deterministic, no seeds needed
BASELINES = list(BASELINE_REGISTRY.keys())

# Fast baselines for dry-run
FAST_BASELINES = ["ridge", "wiener"]


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        y_pred: Predicted values [N, C, T] or [N, D]
        y_true: True values [N, C, T] or [N, D]

    Returns:
        Dictionary of metrics
    """
    # Flatten if needed
    if y_pred.ndim == 3:
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        y_true = y_true.reshape(y_true.shape[0], -1)

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # Pearson correlation (vectorized across dimensions)
    # Limit dimensions for speed, then compute all correlations at once
    n_dims = min(y_pred.shape[1], 100)
    y_pred_sub = y_pred[:, :n_dims]
    y_true_sub = y_true[:, :n_dims]

    # Vectorized Pearson: (x - mean(x)) @ (y - mean(y)) / (std(x) * std(y) * n)
    y_pred_centered = y_pred_sub - y_pred_sub.mean(axis=0, keepdims=True)
    y_true_centered = y_true_sub - y_true_sub.mean(axis=0, keepdims=True)

    y_pred_std = y_pred_sub.std(axis=0) + 1e-8
    y_true_std = y_true_sub.std(axis=0) + 1e-8

    correlations = np.sum(y_pred_centered * y_true_centered, axis=0) / (
        y_pred_std * y_true_std * y_pred_sub.shape[0]
    )
    valid_corrs = correlations[~np.isnan(correlations)]
    pearson = float(np.mean(valid_corrs)) if len(valid_corrs) > 0 else 0.0

    return {
        "r2": float(r2),
        "mae": float(mae),
        "pearson": float(pearson),
    }


# =============================================================================
# Baseline Evaluation with CV
# =============================================================================

def evaluate_baseline_cv(
    baseline_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> ClassicalResult:
    """Evaluate a baseline with 5-fold cross-validation.

    Args:
        baseline_name: Name of baseline method
        X: Input data [N, C, T]
        y: Target data [N, C, T]
        n_folds: Number of CV folds
        seed: Random seed for splits

    Returns:
        ClassicalResult with mean/std metrics across folds
    """
    n_samples = X.shape[0]
    splits = create_data_splits(
        n_samples=n_samples,
        n_folds=n_folds,
        holdout_fraction=0.0,  # No holdout for Tier 0
        seed=seed,
    )

    fold_r2s = []
    fold_maes = []

    for split in splits.cv_splits:
        # Get fold data
        X_train = X[split.train_indices]
        y_train = y[split.train_indices]
        X_val = X[split.val_indices]
        y_val = y[split.val_indices]

        # Create and fit baseline
        try:
            baseline = create_baseline(baseline_name)
            baseline.fit(X_train, y_train)
            y_pred = baseline.predict(X_val)

            metrics = compute_metrics(y_pred, y_val)
            fold_r2s.append(metrics["r2"])
            fold_maes.append(metrics["mae"])
        except Exception as e:
            print(f"    Fold {split.fold_idx} failed: {e}")
            fold_r2s.append(np.nan)
            fold_maes.append(np.nan)

    # Aggregate across folds
    valid_r2s = [r for r in fold_r2s if not np.isnan(r)]
    valid_maes = [m for m in fold_maes if not np.isnan(m)]

    return ClassicalResult(
        method=baseline_name,
        r2_mean=np.mean(valid_r2s) if valid_r2s else np.nan,
        r2_std=np.std(valid_r2s, ddof=1) if len(valid_r2s) > 1 else 0.0,
        mae_mean=np.mean(valid_maes) if valid_maes else np.nan,
        mae_std=np.std(valid_maes, ddof=1) if len(valid_maes) > 1 else 0.0,
    )


def _load_checkpoint() -> Dict[str, ClassicalResult]:
    """Load checkpoint for intra-tier resume."""
    checkpoint_file = ARTIFACTS_DIR / "tier0_checkpoint.json"
    if not checkpoint_file.exists():
        return {}

    with open(checkpoint_file) as f:
        data = json.load(f)

    results = {}
    for name, r in data.items():
        results[name] = ClassicalResult(
            method=r["method"],
            r2_mean=r["r2_mean"],
            r2_std=r["r2_std"],
            mae_mean=r["mae_mean"],
            mae_std=r["mae_std"],
        )
    return results


def _save_checkpoint(results: Dict[str, ClassicalResult]) -> None:
    """Save checkpoint for intra-tier resume."""
    checkpoint_file = ARTIFACTS_DIR / "tier0_checkpoint.json"
    data = {}
    for name, r in results.items():
        data[name] = {
            "method": r.method,
            "r2_mean": r.r2_mean,
            "r2_std": r.r2_std,
            "mae_mean": r.mae_mean,
            "mae_std": r.mae_std,
        }
    with open(checkpoint_file, "w") as f:
        json.dump(data, f, indent=2)


def _clear_checkpoint() -> None:
    """Clear checkpoint after successful tier completion."""
    checkpoint_file = ARTIFACTS_DIR / "tier0_checkpoint.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()


def run_tier0(
    X: np.ndarray,
    y: np.ndarray,
    baselines: Optional[List[str]] = None,
    n_folds: int = 5,
    seed: int = 42,
    register: bool = True,
    resume: bool = True,
) -> Tier0Result:
    """Run Tier 0: Classical Floor evaluation.

    Args:
        X: Input data [N, C, T]
        y: Target data [N, C, T]
        baselines: List of baseline names (default: all)
        n_folds: Number of CV folds
        seed: Random seed
        register: Whether to register results with ConfigRegistry
        resume: Whether to resume from checkpoint (skip completed experiments)

    Returns:
        Tier0Result with all baseline results
    """
    baselines = baselines or BASELINES

    # Load checkpoint for intra-tier resume
    completed_results = {}
    if resume:
        completed_results = _load_checkpoint()
        if completed_results:
            print(f"\n🔄 Resuming from checkpoint: {len(completed_results)} baselines already done")

    results = []

    print(f"\nEvaluating {len(baselines)} classical baselines with {n_folds}-fold CV...")

    for baseline_name in baselines:
        # Skip if already completed
        if baseline_name in completed_results:
            result = completed_results[baseline_name]
            print(f"  {baseline_name}... SKIPPED (already done) R² = {result.r2_mean:.4f} ± {result.r2_std:.4f}")
            results.append(result)
            continue

        print(f"  {baseline_name}...", end=" ", flush=True)
        start = time.time()

        result = evaluate_baseline_cv(
            baseline_name=baseline_name,
            X=X,
            y=y,
            n_folds=n_folds,
            seed=seed,
        )
        results.append(result)

        elapsed = time.time() - start
        print(f"R² = {result.r2_mean:.4f} ± {result.r2_std:.4f} ({elapsed:.1f}s)")

        # Save checkpoint after each baseline
        completed_results[baseline_name] = result
        _save_checkpoint(completed_results)

    # Find best
    valid_results = [r for r in results if not np.isnan(r.r2_mean)]
    if not valid_results:
        raise RuntimeError("All baselines failed!")

    best = max(valid_results, key=lambda r: r.r2_mean)

    tier0_result = Tier0Result(
        results=results,
        best_method=best.method,
        best_r2=best.r2_mean,
    )

    # Register with global registry
    if register:
        registry = get_registry(ARTIFACTS_DIR)
        registry.register_tier0(tier0_result)

    # Clear checkpoint on successful completion
    _clear_checkpoint()

    return tier0_result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tier 0: Classical Floor")
    parser.add_argument("--dry-run", action="store_true", help="Use small synthetic data")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    print("=" * 60)
    print("TIER 0: Classical Floor")
    print("=" * 60)
    print("Purpose: Establish minimum bar for neural methods")
    print("Gate: Neural must beat classical by R² >= 0.10")
    print()

    # Load data
    print("Loading data...")

    if args.dry_run:
        # Small synthetic data
        N = 100
        C_in, C_out, T = 32, 32, 300
        X = np.random.randn(N, C_in, T).astype(np.float32)
        y = np.random.randn(N, C_out, T).astype(np.float32)
        baselines = FAST_BASELINES
        print(f"  [DRY-RUN] Using synthetic data: {X.shape}")
    else:
        try:
            from data import prepare_data
            data = prepare_data()

            # Combine train and val for CV (holdout is separate)
            X_train = data["train"]["ob"].numpy()
            y_train = data["train"]["hp"].numpy()
            X_val = data["val"]["ob"].numpy()
            y_val = data["val"]["hp"].numpy()

            X = np.concatenate([X_train, X_val], axis=0)
            y = np.concatenate([y_train, y_val], axis=0)
            baselines = BASELINES
            print(f"  Loaded real data: {X.shape}")
        except Exception as e:
            print(f"  Warning: Could not load data ({e}), using synthetic")
            N = 200
            C_in, C_out, T = 32, 32, 500
            X = np.random.randn(N, C_in, T).astype(np.float32)
            y = np.random.randn(N, C_out, T).astype(np.float32)
            baselines = BASELINES

    # Run
    result = run_tier0(
        X=X,
        y=y,
        baselines=baselines,
        n_folds=args.folds,
        seed=args.seed,
    )

    # Summary
    print("\n" + "=" * 60)
    print("TIER 0 RESULTS: Classical Floor")
    print("=" * 60)

    sorted_results = sorted(
        [r for r in result.results if not np.isnan(r.r2_mean)],
        key=lambda r: r.r2_mean,
        reverse=True,
    )

    for r in sorted_results:
        marker = " <-- BEST" if r.method == result.best_method else ""
        print(f"  {r.method:20s}: R² = {r.r2_mean:.4f} ± {r.r2_std:.4f}{marker}")

    print()
    print(f"CLASSICAL FLOOR: {result.best_r2:.4f} ({result.best_method})")
    print(f"GATE THRESHOLD: Neural must achieve R² >= {result.best_r2 + 0.10:.4f}")

    # Save
    output_file = args.output or (ARTIFACTS_DIR / f"tier0_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_file = Path(output_file)

    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return result


if __name__ == "__main__":
    main()
