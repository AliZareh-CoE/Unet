#!/usr/bin/env python3
"""
Study 5: Classical Baselines (STAGE 1)
======================================

Evaluates classical signal processing baselines (no HPO needed).
Establishes floor performance for comparison with deep learning.

IMPORTANT: Run this FIRST before neural network studies to establish
the baseline that deep learning methods must beat.

Usage:
    # Run all baselines
    python experiments/study5_classical/run_study5.py

    # Specific baseline
    python experiments/study5_classical/run_study5.py --baseline wiener

    # Dry-run
    python experiments/study5_classical/run_study5.py --dry-run
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

from experiments.study5_classical.baselines import (
    create_wiener_filter,
    create_ridge_regression,
    create_cca_baseline,
    create_kalman_filter,
    BASELINE_REGISTRY,
)


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "study5_classical"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# All baselines to evaluate
BASELINES = [
    "wiener", "wiener_mimo",
    "ridge", "ridge_temporal", "ridge_cv",
    "cca", "cca_regularized", "cca_temporal",
    "kalman", "kalman_extended",
]


# =============================================================================
# Evaluation Metrics
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

    # Correlation (average across dimensions)
    correlations = []
    for d in range(y_pred.shape[1]):
        corr = np.corrcoef(y_pred[:, d], y_true[:, d])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    mean_corr = np.mean(correlations) if correlations else 0.0

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # MSE
    mse = np.mean((y_true - y_pred) ** 2)

    return {
        "r2": float(r2),
        "correlation": float(mean_corr),
        "mae": float(mae),
        "mse": float(mse),
    }


# =============================================================================
# Baseline Evaluation
# =============================================================================

def evaluate_baseline(
    baseline_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate a single baseline.

    Args:
        baseline_name: Name of baseline
        X_train, y_train: Training data
        X_test, y_test: Test data
        **kwargs: Additional arguments for baseline

    Returns:
        Results dictionary
    """
    print(f"  Evaluating {baseline_name}...")

    # Create baseline
    from experiments.study5_classical.baselines import create_baseline
    baseline = create_baseline(baseline_name, **kwargs)

    # Time fitting
    start_time = time.time()
    baseline.fit(X_train, y_train)
    fit_time = time.time() - start_time

    # Predict
    start_time = time.time()
    y_pred_train = baseline.predict(X_train)
    y_pred_test = baseline.predict(X_test)
    pred_time = time.time() - start_time

    # Compute metrics
    train_metrics = compute_metrics(y_pred_train, y_train)
    test_metrics = compute_metrics(y_pred_test, y_test)

    return {
        "baseline": baseline_name,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "fit_time_seconds": fit_time,
        "predict_time_seconds": pred_time,
        "kwargs": kwargs,
    }


def run_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baselines: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run all classical baselines.

    Args:
        X_train, y_train: Training data [N, C, T]
        X_test, y_test: Test data [N, C, T]
        baselines: Optional list of baselines to run

    Returns:
        Dictionary of results for each baseline
    """
    baselines = baselines or BASELINES
    results = {}

    for baseline_name in baselines:
        try:
            result = evaluate_baseline(
                baseline_name,
                X_train, y_train,
                X_test, y_test,
            )
            results[baseline_name] = result

            print(f"    Train R²: {result['train_metrics']['r2']:.4f}")
            print(f"    Test R²:  {result['test_metrics']['r2']:.4f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            results[baseline_name] = {"error": str(e)}

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Study 5: Classical Baselines")
    parser.add_argument("--baseline", type=str, choices=BASELINES, help="Single baseline")
    parser.add_argument("--dry-run", action="store_true", help="Use small synthetic data")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    print("="*60)
    print("Study 5: Classical Baselines")
    print("="*60)

    # Load data
    print("\nLoading data...")

    if args.dry_run:
        # Minimal synthetic data for fast dry-run testing
        N_train, N_test = 50, 20
        C_in, C_out, T = 32, 32, 300  # Shorter time series

        X_train = np.random.randn(N_train, C_in, T).astype(np.float32)
        y_train = np.random.randn(N_train, C_out, T).astype(np.float32)
        X_test = np.random.randn(N_test, C_in, T).astype(np.float32)
        y_test = np.random.randn(N_test, C_out, T).astype(np.float32)

        print(f"  Using minimal synthetic data for dry-run: {N_train} train, {N_test} test samples")
    else:
        try:
            from data import prepare_data
            data = prepare_data()

            # Convert to numpy
            X_train = data["train"]["ob"].numpy()
            y_train = data["train"]["hp"].numpy()
            X_test = data["val"]["ob"].numpy()
            y_test = data["val"]["hp"].numpy()

            print(f"  Loaded real data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        except Exception as e:
            print(f"  Warning: Could not load data ({e}), using synthetic")
            N_train, N_test = 100, 30
            C_in, C_out, T = 32, 32, 500

            X_train = np.random.randn(N_train, C_in, T).astype(np.float32)
            y_train = np.random.randn(N_train, C_out, T).astype(np.float32)
            X_test = np.random.randn(N_test, C_in, T).astype(np.float32)
            y_test = np.random.randn(N_test, C_out, T).astype(np.float32)

    # Run baselines
    print("\nEvaluating baselines...")

    if args.baseline:
        baselines_to_run = [args.baseline]
    elif args.dry_run:
        # Only run 2 fast baselines in dry-run mode for speed
        baselines_to_run = ["ridge", "wiener"]
        print(f"  Dry-run: Only testing {baselines_to_run}")
    else:
        baselines_to_run = BASELINES

    results = run_all_baselines(
        X_train, y_train,
        X_test, y_test,
        baselines=baselines_to_run,
    )

    # Summary
    print("\n" + "="*60)
    print("Summary: Test R² Scores")
    print("="*60)

    sorted_results = sorted(
        [(name, r.get("test_metrics", {}).get("r2", float("-inf")))
         for name, r in results.items() if "error" not in r],
        key=lambda x: x[1],
        reverse=True,
    )

    for name, r2 in sorted_results:
        print(f"  {name:20s}: {r2:.4f}")

    # Save results
    output_file = args.output or (ARTIFACTS_DIR / f"baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_file = Path(output_file)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
