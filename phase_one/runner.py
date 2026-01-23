#!/usr/bin/env python3
"""
Phase 1 Runner: Classical Baselines Evaluation
===============================================

Main execution script for Phase 1 of the Nature Methods study.
Evaluates 7 classical signal processing baselines on the olfactory dataset.

3-Fold × 3-Seed Strategy:
- 3-fold session-based cross-validation (9 sessions → 3 groups of 3)
- 3 random seeds per fold for variance estimation
- Total: 9 training runs per baseline method
- Proper statistical comparisons with CI, SEM, and effect sizes

Usage:
    python -m phase_one.runner --dataset olfactory
    python -m phase_one.runner --dry-run  # Quick test with synthetic data
    python -m phase_one.runner --3fold-3seed  # Full 3-fold 3-seed evaluation

Output:
    - JSON results file with all metrics
    - Per-session metrics JSON
    - PDF figures for publication
    - Console summary
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from phase_one.config import (
    Phase1Config,
    BASELINE_METHODS,
    FAST_BASELINES,
    GATE_THRESHOLD,
    get_gate_threshold,
    NEURAL_BANDS,
)
from phase_one.baselines import create_baseline, list_baselines, BASELINE_REGISTRY
from phase_one.metrics import (
    MetricsCalculator,
    Phase1Metrics,
    StatisticalComparison,
    compute_statistical_comparisons,
    format_metrics_table,
    rank_methods,
)
from phase_one.visualization import Phase1Visualizer, create_summary_table

# Import shared statistical utilities
from utils import (
    compare_methods,
    compare_multiple_methods,
    holm_correction,
    fdr_correction,
    check_assumptions,
    format_mean_ci,
    TestResult,
    ComparisonResult,
)


# =============================================================================
# 3-Fold × 3-Seed Strategy Configuration
# =============================================================================

N_FOLDS = 3   # Number of cross-validation folds
N_SEEDS = 3   # Number of seeds per fold
BASE_SEED = 42  # Base random seed


# =============================================================================
# Checkpoint/Resume Support
# =============================================================================

def get_checkpoint_path(output_dir: Path) -> Path:
    """Get the checkpoint file path for incremental saving."""
    return output_dir / "phase1_checkpoint.pkl"


def save_checkpoint(
    checkpoint_path: Path,
    all_metrics: List,
    all_predictions: Dict,
    all_models: Dict,
    method_fold_r2s: Dict,
    completed_methods: set,
) -> None:
    """Save checkpoint after each method for resume capability."""
    checkpoint = {
        "all_metrics": [m.to_dict() if hasattr(m, 'to_dict') else m for m in all_metrics],
        "all_predictions": {},  # Don't save predictions (large arrays)
        "all_models": {},  # Don't save models (handled separately)
        "method_fold_r2s": method_fold_r2s,
        "completed_methods": list(completed_methods),
        "timestamp": datetime.now().isoformat(),
    }

    # Save atomically
    temp_path = checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    temp_path.rename(checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Load checkpoint if it exists."""
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"\n*** RESUMING FROM CHECKPOINT ***")
        print(f"    Found {len(checkpoint['completed_methods'])} completed methods")
        print(f"    Checkpoint from: {checkpoint['timestamp']}")
        return checkpoint
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None


def reconstruct_from_checkpoint(checkpoint: Dict[str, Any]) -> Tuple[List, Dict, Dict, Dict, set]:
    """Reconstruct state from checkpoint."""
    # Reconstruct Phase1Metrics objects
    all_metrics = []
    for m_dict in checkpoint["all_metrics"]:
        if isinstance(m_dict, dict):
            metrics = Phase1Metrics(
                method=m_dict["method"],
                r2_mean=m_dict["r2_mean"],
                r2_std=m_dict["r2_std"],
                r2_ci=tuple(m_dict["r2_ci"]),
                mae_mean=m_dict["mae_mean"],
                mae_std=m_dict["mae_std"],
                pearson_mean=m_dict["pearson_mean"],
                spearman_mean=m_dict["spearman_mean"],
                band_r2=m_dict.get("band_r2", {}),
                psd_error_db=m_dict.get("psd_error_db", 0.0),
                fold_r2s=m_dict.get("fold_r2s", []),
                plv_mean=m_dict.get("plv_mean", 0.0),
                plv_std=m_dict.get("plv_std", 0.0),
                pli_mean=m_dict.get("pli_mean", 0.0),
                wpli_mean=m_dict.get("wpli_mean", 0.0),
                coherence_mean=m_dict.get("coherence_mean", 0.0),
                coherence_std=m_dict.get("coherence_std", 0.0),
                envelope_corr_mean=m_dict.get("envelope_corr_mean", 0.0),
                mutual_info_mean=m_dict.get("mutual_info_mean", 0.0),
                reconstruction_snr_db=m_dict.get("reconstruction_snr_db", 0.0),
                fold_plv=m_dict.get("fold_plv", []),
                fold_coherence=m_dict.get("fold_coherence", []),
            )
            all_metrics.append(metrics)

    method_fold_r2s = checkpoint["method_fold_r2s"]
    completed_methods = set(checkpoint["completed_methods"])

    return all_metrics, {}, {}, method_fold_r2s, completed_methods


# =============================================================================
# 3-Fold × 3-Seed Result Dataclasses (Aligned with Ablation Study)
# =============================================================================

@dataclass
class FoldSeedResult:
    """Result from a single fold+seed combination.

    IMPORTANT: Proper separation of val vs test metrics (NO data leakage):
    - val_* metrics: From 70/30 trial-wise split on training sessions
    - test_* metrics: From held-out test sessions (PRIMARY metric)
    """
    fold_idx: int
    seed: int
    test_sessions: List[str]
    train_sessions: List[str]

    # Validation metrics (from 70/30 split on training sessions)
    val_r2: float = 0.0
    val_mae: float = 0.0
    val_pearson: float = 0.0

    # TEST metrics (from held-out sessions - PRIMARY, NO leakage)
    test_r2: float = 0.0
    test_mae: float = 0.0
    test_pearson: float = 0.0

    # Per-session TEST R² (CRITICAL for session-level analysis)
    per_session_test_r2: Dict[str, float] = field(default_factory=dict)
    per_session_test_mae: Dict[str, float] = field(default_factory=dict)
    per_session_test_corr: Dict[str, float] = field(default_factory=dict)

    # DSP metrics (from test)
    plv: float = 0.0
    coherence: float = 0.0
    reconstruction_snr_db: float = 0.0
    envelope_corr: float = 0.0

    # Band R² (from test)
    band_r2: Dict[str, float] = field(default_factory=dict)

    # Timing
    total_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_idx": self.fold_idx,
            "seed": self.seed,
            "test_sessions": self.test_sessions,
            "train_sessions": self.train_sessions,
            # Validation metrics
            "val_r2": self.val_r2,
            "val_mae": self.val_mae,
            "val_pearson": self.val_pearson,
            # TEST metrics (PRIMARY)
            "test_r2": self.test_r2,
            "test_mae": self.test_mae,
            "test_pearson": self.test_pearson,
            # Per-session TEST metrics
            "per_session_test_r2": self.per_session_test_r2,
            "per_session_test_mae": self.per_session_test_mae,
            "per_session_test_corr": self.per_session_test_corr,
            # DSP metrics
            "plv": self.plv,
            "coherence": self.coherence,
            "reconstruction_snr_db": self.reconstruction_snr_db,
            "envelope_corr": self.envelope_corr,
            "band_r2": self.band_r2,
            "total_time": self.total_time,
        }


@dataclass
class MethodResult:
    """Aggregated results for one baseline method across all folds and seeds.

    IMPORTANT: Uses TEST metrics (from held-out sessions) as PRIMARY metrics.
    Val metrics are only for reference. This mirrors AblationResult design.
    """
    method: str
    fold_seed_results: List[FoldSeedResult]

    # PRIMARY statistics - TEST metrics (from held-out sessions, NO leakage)
    mean_r2: float = 0.0  # Mean TEST R² across all fold+seed runs
    std_r2: float = 0.0
    sem_r2: float = 0.0
    median_r2: float = 0.0
    min_r2: float = 0.0
    max_r2: float = 0.0

    # Confidence intervals (95%) for TEST R²
    ci_lower_r2: float = 0.0
    ci_upper_r2: float = 0.0

    # TEST correlation statistics
    mean_pearson: float = 0.0
    std_pearson: float = 0.0

    # TEST MAE statistics
    mean_mae: float = 0.0
    std_mae: float = 0.0

    # Validation statistics (for reference - from 70/30 split)
    mean_val_r2: float = 0.0
    std_val_r2: float = 0.0

    # DSP metrics aggregated (from test)
    mean_plv: float = 0.0
    mean_coherence: float = 0.0
    mean_snr: float = 0.0

    # Per-session TEST R² (each session evaluated once per fold, 3 seeds)
    all_session_r2s: Dict[str, float] = field(default_factory=dict)  # Mean TEST R² per session
    session_r2_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-fold TEST statistics (for paired tests - average seeds within fold)
    fold_r2_values: List[float] = field(default_factory=list)  # Mean TEST R² per fold
    fold_pearson_values: List[float] = field(default_factory=list)

    # Raw TEST values for statistical tests (all 9 = 3 folds × 3 seeds)
    all_r2_values: List[float] = field(default_factory=list)
    all_pearson_values: List[float] = field(default_factory=list)

    # Timing
    total_time: float = 0.0

    def compute_statistics(self) -> None:
        """Compute comprehensive aggregate statistics from fold+seed results.

        PRIMARY metrics are from TEST (held-out sessions) - NO data leakage.
        Val metrics are kept for reference.
        """
        if not self.fold_seed_results:
            return

        # Extract all TEST R² values (3 folds × 3 seeds = 9 values) - PRIMARY
        self.all_r2_values = [r.test_r2 for r in self.fold_seed_results]
        self.all_pearson_values = [r.test_pearson for r in self.fold_seed_results]

        # Also track validation metrics
        val_r2_values = [r.val_r2 for r in self.fold_seed_results]

        test_r2_arr = np.array(self.all_r2_values)
        test_pearson_arr = np.array(self.all_pearson_values)
        test_mae_arr = np.array([r.test_mae for r in self.fold_seed_results])
        val_r2_arr = np.array(val_r2_values)

        # PRIMARY R² statistics (from TEST - held-out sessions)
        self.mean_r2 = float(np.mean(test_r2_arr))
        self.std_r2 = float(np.std(test_r2_arr, ddof=1)) if len(test_r2_arr) > 1 else 0.0
        self.sem_r2 = float(self.std_r2 / np.sqrt(len(test_r2_arr))) if len(test_r2_arr) > 1 else 0.0
        self.median_r2 = float(np.median(test_r2_arr))
        self.min_r2 = float(np.min(test_r2_arr))
        self.max_r2 = float(np.max(test_r2_arr))

        # Confidence interval for TEST R² (95%)
        if len(test_r2_arr) >= 2:
            from scipy import stats as scipy_stats
            t_crit = scipy_stats.t.ppf(0.975, len(test_r2_arr) - 1)
            margin = t_crit * self.sem_r2
            self.ci_lower_r2 = self.mean_r2 - margin
            self.ci_upper_r2 = self.mean_r2 + margin

        # TEST correlation statistics
        self.mean_pearson = float(np.mean(test_pearson_arr))
        self.std_pearson = float(np.std(test_pearson_arr, ddof=1)) if len(test_pearson_arr) > 1 else 0.0

        # TEST MAE statistics
        self.mean_mae = float(np.mean(test_mae_arr))
        self.std_mae = float(np.std(test_mae_arr, ddof=1)) if len(test_mae_arr) > 1 else 0.0

        # Validation statistics (for reference)
        self.mean_val_r2 = float(np.mean(val_r2_arr))
        self.std_val_r2 = float(np.std(val_r2_arr, ddof=1)) if len(val_r2_arr) > 1 else 0.0

        # DSP metrics (from test)
        self.mean_plv = float(np.mean([r.plv for r in self.fold_seed_results]))
        self.mean_coherence = float(np.mean([r.coherence for r in self.fold_seed_results]))
        self.mean_snr = float(np.mean([r.reconstruction_snr_db for r in self.fold_seed_results]))

        # Compute per-fold mean TEST R² (for paired statistical tests)
        # Each fold has 3 seeds, so we average the seeds within each fold
        fold_groups = {}
        for r in self.fold_seed_results:
            if r.fold_idx not in fold_groups:
                fold_groups[r.fold_idx] = []
            fold_groups[r.fold_idx].append(r.test_r2)  # Use TEST R²

        self.fold_r2_values = [float(np.mean(fold_groups[i])) for i in sorted(fold_groups.keys())]

        fold_pearson_groups = {}
        for r in self.fold_seed_results:
            if r.fold_idx not in fold_pearson_groups:
                fold_pearson_groups[r.fold_idx] = []
            fold_pearson_groups[r.fold_idx].append(r.test_pearson)  # Use TEST pearson
        self.fold_pearson_values = [float(np.mean(fold_pearson_groups[i])) for i in sorted(fold_pearson_groups.keys())]

        # Aggregate per-session TEST R²s across all fold+seed runs
        # Each session is tested in exactly one fold, but with 3 seeds
        session_r2_lists: Dict[str, List[float]] = {}

        for r in self.fold_seed_results:
            for session, r2 in r.per_session_test_r2.items():  # Use TEST R²
                if session not in session_r2_lists:
                    session_r2_lists[session] = []
                session_r2_lists[session].append(r2)

        # Store per-session TEST statistics (mean/std across 3 seeds for that session)
        for session, r2_list in session_r2_lists.items():
            self.all_session_r2s[session] = float(np.mean(r2_list))
            self.session_r2_stats[session] = {
                "test_r2_mean": float(np.mean(r2_list)),
                "test_r2_std": float(np.std(r2_list, ddof=1)) if len(r2_list) > 1 else 0.0,
                "n_seeds": len(r2_list),
                "values": r2_list,  # All 3 seed values for this session
            }

        # Total time
        self.total_time = sum(r.total_time for r in self.fold_seed_results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "fold_seed_results": [r.to_dict() for r in self.fold_seed_results],
            # Primary R² statistics
            "mean_r2": self.mean_r2,
            "std_r2": self.std_r2,
            "sem_r2": self.sem_r2,
            "median_r2": self.median_r2,
            "min_r2": self.min_r2,
            "max_r2": self.max_r2,
            "ci_lower_r2": self.ci_lower_r2,
            "ci_upper_r2": self.ci_upper_r2,
            # Correlation statistics
            "mean_pearson": self.mean_pearson,
            "std_pearson": self.std_pearson,
            # MAE statistics
            "mean_mae": self.mean_mae,
            "std_mae": self.std_mae,
            # DSP metrics
            "mean_plv": self.mean_plv,
            "mean_coherence": self.mean_coherence,
            "mean_snr": self.mean_snr,
            # Per-session data
            "all_session_r2s": self.all_session_r2s,
            "session_r2_stats": self.session_r2_stats,
            # Per-fold values (for paired tests)
            "fold_r2_values": self.fold_r2_values,
            "fold_pearson_values": self.fold_pearson_values,
            # Raw values (all 9 runs)
            "all_r2_values": self.all_r2_values,
            "all_pearson_values": self.all_pearson_values,
            # Timing
            "total_time": self.total_time,
        }


@dataclass
class Phase1Result3Fold3Seed:
    """Complete Phase 1 results using the 3-fold × 3-seed strategy.

    This is the main result class that aligns with the ablation study design.
    """
    method_results: Dict[str, MethodResult]
    best_method: str
    best_r2: float
    gate_threshold: float

    # Metadata
    n_folds: int = N_FOLDS
    n_seeds: int = N_SEEDS
    total_runs: int = N_FOLDS * N_SEEDS

    # Statistical comparisons
    statistical_comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Config and timing
    config: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_results": {k: v.to_dict() for k, v in self.method_results.items()},
            "best_method": self.best_method,
            "best_r2": self.best_r2,
            "gate_threshold": self.gate_threshold,
            "n_folds": self.n_folds,
            "n_seeds": self.n_seeds,
            "total_runs": self.total_runs,
            "statistical_comparisons": self.statistical_comparisons,
            "config": self.config,
            "timestamp": self.timestamp,
            "total_time": self.total_time,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def save_per_session_metrics(self, path: Path) -> None:
        """Save per-session metrics to a separate JSON file for easy access."""
        per_session = {
            "metadata": {
                "timestamp": self.timestamp,
                "n_folds": self.n_folds,
                "n_seeds": self.n_seeds,
                "best_method": self.best_method,
            },
            "methods": {},
        }

        for method_name, result in self.method_results.items():
            per_session["methods"][method_name] = {
                "mean_r2": result.mean_r2,
                "session_r2_stats": result.session_r2_stats,
            }

        with open(path, 'w') as f:
            json.dump(per_session, f, indent=2)


# =============================================================================
# Result Dataclass (Legacy - kept for backward compatibility)
# =============================================================================

@dataclass
class Phase1Result:
    """Complete results from Phase 1 evaluation.

    Attributes:
        metrics: List of metrics for each baseline
        best_method: Name of best performing method
        best_r2: R² of best method
        gate_threshold: Minimum R² for neural methods to pass
        statistical_comparisons: Pairwise method comparisons
        comprehensive_stats: Enhanced statistical analysis with both parametric/non-parametric tests
        assumption_checks: Results of normality and homogeneity tests
        models: Dictionary of trained models (method_name -> model object)
        config: Configuration used
        timestamp: Evaluation timestamp
    """

    metrics: List[Phase1Metrics]
    best_method: str
    best_r2: float
    gate_threshold: float
    statistical_comparisons: List[StatisticalComparison] = field(default_factory=list)
    comprehensive_stats: Optional[Dict[str, Any]] = None
    assumption_checks: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Any]] = None  # Trained models for each method
    config: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "best_method": self.best_method,
            "best_r2": self.best_r2,
            "gate_threshold": self.gate_threshold,
            "statistical_comparisons": [
                {
                    "method1": c.method1,
                    "method2": c.method2,
                    "mean_diff": c.mean_diff,
                    "t_stat": c.t_stat,
                    "p_value": c.p_value,
                    "p_adjusted": c.p_adjusted,
                    "cohens_d": c.cohens_d,
                    "effect_size": c.effect_size,
                    "significant": c.significant,
                }
                for c in self.statistical_comparisons
            ],
            "comprehensive_stats": self.comprehensive_stats,
            "assumption_checks": self.assumption_checks,
            "has_models": self.models is not None,
            "config": self.config,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def save_models(self, path: Path) -> None:
        """Save trained models to pickle file.

        Args:
            path: Path to save models (will add .pkl extension if not present)
        """
        if self.models is None:
            print("No models to save")
            return

        model_path = Path(path)
        if model_path.suffix != '.pkl':
            model_path = model_path.with_suffix('.pkl')

        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(self.models, f)

        print(f"Models saved to: {model_path}")

    @staticmethod
    def load_models(path: Path) -> Dict[str, Any]:
        """Load trained models from pickle file.

        Args:
            path: Path to models pickle file

        Returns:
            Dictionary of method_name -> model object
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def load(cls, path: Path) -> "Phase1Result":
        """Load results from JSON file.

        Args:
            path: Path to JSON results file

        Returns:
            Phase1Result object

        Example:
            >>> result = Phase1Result.load("results/phase1/phase1_results_20240101.json")
            >>> print(result.best_method, result.best_r2)
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct Phase1Metrics objects
        metrics = []
        for m in data["metrics"]:
            metrics.append(Phase1Metrics(
                method=m["method"],
                r2_mean=m["r2_mean"],
                r2_std=m["r2_std"],
                r2_ci=tuple(m["r2_ci"]),
                mae_mean=m["mae_mean"],
                mae_std=m["mae_std"],
                pearson_mean=m["pearson_mean"],
                pearson_std=m["pearson_std"],
                psd_error_db=m.get("psd_error_db", 0.0),
                band_r2=m.get("band_r2", {}),
                fold_r2s=m.get("fold_r2s", []),
                fold_maes=m.get("fold_maes", []),
                fold_pearsons=m.get("fold_pearsons", []),
                n_folds=m.get("n_folds", 5),
                n_samples=m.get("n_samples", 0),
            ))

        # Reconstruct StatisticalComparison objects
        comparisons = []
        for c in data.get("statistical_comparisons", []):
            comparisons.append(StatisticalComparison(
                method1=c["method1"],
                method2=c["method2"],
                mean_diff=c["mean_diff"],
                t_stat=c["t_stat"],
                p_value=c["p_value"],
                p_adjusted=c["p_adjusted"],
                cohens_d=c["cohens_d"],
                effect_size=c["effect_size"],
                significant=c["significant"],
            ))

        return cls(
            metrics=metrics,
            best_method=data["best_method"],
            best_r2=data["best_r2"],
            gate_threshold=data["gate_threshold"],
            statistical_comparisons=comparisons,
            config=data.get("config"),
            timestamp=data.get("timestamp", ""),
        )


# =============================================================================
# Data Loading
# =============================================================================

def load_olfactory_data() -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, Optional[Dict[str, NDArray]]]:
    """Load olfactory dataset (OB -> PCx translation).

    Returns:
        X: Input signals (OB) [N, C, T]
        y: Target signals (PCx) [N, C, T]
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices (held out)
        val_idx_per_session: Dict mapping session name -> validation indices for that session
    """
    from data import prepare_data

    print("Loading olfactory dataset (cross-subject: 3 held-out sessions, no test set)...")
    data = prepare_data(
        split_by_session=True,
        n_test_sessions=0,   # No separate test set (matches Phase 3)
        n_val_sessions=3,    # 3 sessions held out for validation (matches Phase 3)
        no_test_set=True,    # All held-out sessions for validation
        force_recreate_splits=True,  # Override any cached splits to ensure correct config
        separate_val_sessions=True,  # Get per-session val indices
    )

    ob = data["ob"]    # [N, C, T] - Olfactory Bulb
    pcx = data["pcx"]  # [N, C, T] - Piriform Cortex

    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    val_idx_per_session = data.get("val_idx_per_session", None)

    # Log session split info if available
    if "split_info" in data:
        split_info = data["split_info"]
        print(f"  Train sessions: {split_info.get('train_sessions', 'N/A')}")
        print(f"  Val sessions: {split_info.get('val_sessions', 'N/A')}")
        print(f"  Test sessions: {split_info.get('test_sessions', 'N/A')}")

    print(f"  Loaded: OB {ob.shape} -> PCx {pcx.shape}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    if val_idx_per_session:
        for sess_name, sess_idx in val_idx_per_session.items():
            print(f"    Val session {sess_name}: {len(sess_idx)} samples")

    return ob, pcx, train_idx, val_idx, test_idx, val_idx_per_session


def create_synthetic_data(
    n_samples: int = 200,
    n_channels: int = 32,
    time_points: int = 1000,
    seed: int = 42,
) -> Tuple[NDArray, NDArray]:
    """Create synthetic data for testing.

    Generates correlated signals to simulate neural translation task.

    Args:
        n_samples: Number of samples
        n_channels: Number of channels
        time_points: Time points per sample
        seed: Random seed

    Returns:
        X: Input signals [N, C, T]
        y: Target signals [N, C, T]
    """
    np.random.seed(seed)

    # Generate correlated signals
    X = np.random.randn(n_samples, n_channels, time_points).astype(np.float32)

    # Target is filtered/transformed version of input + noise
    y = np.zeros_like(X)
    for i in range(n_samples):
        for c in range(n_channels):
            # Simple smoothing + delay + noise
            kernel = np.hanning(50) / np.hanning(50).sum()
            y[i, c, :] = np.convolve(X[i, c, :], kernel, mode='same')
            y[i, c, :] += 0.3 * np.random.randn(time_points)

    return X, y


# =============================================================================
# Cross-Validation
# =============================================================================

def create_cv_splits(
    n_samples: int,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[NDArray, NDArray]]:
    """Create K-fold cross-validation splits.

    Args:
        n_samples: Total number of samples
        n_folds: Number of folds
        seed: Random seed

    Returns:
        List of (train_indices, val_indices) tuples
    """
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // n_folds

    splits = []
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n_samples

        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        splits.append((train_idx, val_idx))

    return splits


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_baseline(
    baseline_name: str,
    X: NDArray,
    y: NDArray,
    config: Phase1Config,
    return_model: bool = True,
) -> Tuple[Phase1Metrics, Optional[NDArray], Optional[Any]]:
    """Evaluate a single baseline with cross-validation.

    Args:
        baseline_name: Name of baseline method
        X: Input signals [N, C, T]
        y: Target signals [N, C, T]
        config: Configuration object
        return_model: Whether to return a trained model on full data

    Returns:
        Phase1Metrics object, optional predictions on last fold, and optional trained model
    """
    n_samples = X.shape[0]
    splits = create_cv_splits(n_samples, config.n_folds, config.seed)

    metrics_calc = MetricsCalculator(
        sample_rate=config.sample_rate,
        n_bootstrap=config.n_bootstrap,
        ci_level=config.ci_level,
    )

    fold_results = []
    last_predictions = None

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        try:
            # Create and fit model
            model = create_baseline(baseline_name)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Compute metrics (including DSP metrics)
            metrics = metrics_calc.compute(y_pred, y_val)
            fold_results.append(metrics)

            # Save last fold predictions
            if fold_idx == len(splits) - 1:
                last_predictions = (y_pred, y_val)

        except Exception as e:
            print(f"    Warning: Fold {fold_idx} failed for {baseline_name}: {e}")
            fold_results.append({"r2": np.nan, "mae": np.nan, "pearson": np.nan})

    # Aggregate across folds - basic metrics
    fold_r2s = [r.get("r2", np.nan) for r in fold_results]
    fold_maes = [r.get("mae", np.nan) for r in fold_results]
    fold_pearsons = [r.get("pearson", np.nan) for r in fold_results]
    fold_spearmans = [r.get("spearman", np.nan) for r in fold_results]

    valid_r2s = np.array([r for r in fold_r2s if not np.isnan(r)])
    valid_maes = np.array([m for m in fold_maes if not np.isnan(m)])
    valid_pearsons = np.array([p for p in fold_pearsons if not np.isnan(p)])
    valid_spearmans = np.array([s for s in fold_spearmans if not np.isnan(s)])

    # Compute bootstrap CI
    ci = metrics_calc.bootstrap_ci(np.array(fold_r2s))

    # Aggregate band R² (average across folds)
    band_r2 = {}
    for band in NEURAL_BANDS.keys():
        band_vals = [r.get(f"r2_{band}", np.nan) for r in fold_results]
        valid_band = [v for v in band_vals if not np.isnan(v)]
        band_r2[band] = float(np.mean(valid_band)) if valid_band else np.nan

    # PSD error
    psd_errors = [r.get("psd_error_db", np.nan) for r in fold_results]
    valid_psd = [p for p in psd_errors if not np.isnan(p)]
    psd_error_db = float(np.mean(valid_psd)) if valid_psd else np.nan

    # Aggregate DSP metrics across folds
    def aggregate_metric(key):
        vals = [r.get(key, np.nan) for r in fold_results]
        valid = [v for v in vals if not np.isnan(v)]
        return (float(np.mean(valid)) if valid else 0.0,
                float(np.std(valid)) if len(valid) > 1 else 0.0,
                vals)

    plv_mean, plv_std, fold_plvs = aggregate_metric("plv")
    pli_mean, pli_std, fold_plis = aggregate_metric("pli")
    wpli_mean, wpli_std, _ = aggregate_metric("wpli")
    coherence_mean, coherence_std, fold_coherences = aggregate_metric("coherence")
    recon_snr, _, fold_snrs = aggregate_metric("reconstruction_snr_db")
    cross_corr_max, _, _ = aggregate_metric("cross_corr_max")
    cross_corr_lag = int(np.nanmean([r.get("cross_corr_lag", 0) for r in fold_results]))
    env_corr_mean, env_corr_std, fold_env_corrs = aggregate_metric("envelope_corr")
    mi_mean, _, _ = aggregate_metric("mutual_info")
    nmi_mean, _, _ = aggregate_metric("normalized_mi")
    psd_corr_mean, _, _ = aggregate_metric("psd_correlation")

    # Aggregate coherence bands
    coherence_bands = {}
    for band in NEURAL_BANDS.keys():
        band_vals = []
        for r in fold_results:
            coh_bands = r.get("coherence_bands", {})
            if band in coh_bands and not np.isnan(coh_bands[band]):
                band_vals.append(coh_bands[band])
        coherence_bands[band] = float(np.mean(band_vals)) if band_vals else 0.0

    # Aggregate band power errors
    band_power_errors = {}
    for band in NEURAL_BANDS.keys():
        band_vals = []
        for r in fold_results:
            bp_errors = r.get("band_power_errors", {})
            if band in bp_errors and not np.isnan(bp_errors[band]):
                band_vals.append(bp_errors[band])
        band_power_errors[band] = float(np.mean(band_vals)) if band_vals else 0.0

    result = Phase1Metrics(
        method=baseline_name,
        r2_mean=float(np.mean(valid_r2s)) if len(valid_r2s) > 0 else np.nan,
        r2_std=float(np.std(valid_r2s, ddof=1)) if len(valid_r2s) > 1 else 0.0,
        r2_ci=ci,
        mae_mean=float(np.mean(valid_maes)) if len(valid_maes) > 0 else np.nan,
        mae_std=float(np.std(valid_maes, ddof=1)) if len(valid_maes) > 1 else 0.0,
        pearson_mean=float(np.mean(valid_pearsons)) if len(valid_pearsons) > 0 else np.nan,
        pearson_std=float(np.std(valid_pearsons, ddof=1)) if len(valid_pearsons) > 1 else 0.0,
        spearman_mean=float(np.mean(valid_spearmans)) if len(valid_spearmans) > 0 else np.nan,
        spearman_std=float(np.std(valid_spearmans, ddof=1)) if len(valid_spearmans) > 1 else 0.0,
        psd_error_db=psd_error_db,
        band_r2=band_r2,
        fold_r2s=fold_r2s,
        fold_maes=fold_maes,
        fold_pearsons=fold_pearsons,
        n_folds=config.n_folds,
        n_samples=n_samples,
        # DSP metrics
        plv_mean=plv_mean,
        plv_std=plv_std,
        pli_mean=pli_mean,
        pli_std=pli_std,
        wpli_mean=wpli_mean,
        wpli_std=wpli_std,
        coherence_mean=coherence_mean,
        coherence_std=coherence_std,
        coherence_bands=coherence_bands,
        reconstruction_snr_db=recon_snr,
        cross_corr_max=cross_corr_max,
        cross_corr_lag=cross_corr_lag,
        envelope_corr_mean=env_corr_mean,
        envelope_corr_std=env_corr_std,
        mutual_info_mean=mi_mean,
        normalized_mi_mean=nmi_mean,
        psd_correlation=psd_corr_mean,
        band_power_errors=band_power_errors,
        # Fold-level DSP data
        fold_plvs=fold_plvs,
        fold_plis=fold_plis,
        fold_coherences=fold_coherences,
        fold_envelope_corrs=fold_env_corrs,
        fold_reconstruction_snrs=fold_snrs,
    )

    # Train final model on full data if requested
    final_model = None
    if return_model:
        try:
            final_model = create_baseline(baseline_name)
            final_model.fit(X, y)
        except Exception as e:
            print(f"    Warning: Could not train final model for {baseline_name}: {e}")

    return result, last_predictions, final_model


def evaluate_baseline_cross_subject(
    baseline_name: str,
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
    config: Phase1Config,
    return_model: bool = True,
    val_idx_per_session: Optional[Dict[str, NDArray]] = None,
    X_full: Optional[NDArray] = None,
    y_full: Optional[NDArray] = None,
) -> Tuple[Phase1Metrics, Optional[NDArray], Optional[Any], Optional[Dict[str, float]]]:
    """Evaluate a single baseline with cross-subject split (no CV mixing).

    This is the proper cross-subject evaluation:
    - Train ONLY on X_train/y_train (train sessions)
    - Evaluate ONLY on X_val/y_val (held-out sessions)
    - No K-fold CV that would mix sessions

    Args:
        baseline_name: Name of baseline method
        X_train: Training input signals [N_train, C, T]
        y_train: Training target signals [N_train, C, T]
        X_val: Validation input signals [N_val, C, T] (held-out sessions)
        y_val: Validation target signals [N_val, C, T] (held-out sessions)
        config: Configuration object
        return_model: Whether to return the trained model
        val_idx_per_session: Dict mapping session name -> indices (into full array)
        X_full: Full input array (needed for per-session metrics)
        y_full: Full target array (needed for per-session metrics)

    Returns:
        Phase1Metrics object, predictions on val set, trained model, and per-session R² dict
    """
    metrics_calc = MetricsCalculator(
        sample_rate=config.sample_rate,
        n_bootstrap=config.n_bootstrap,
        ci_level=config.ci_level,
    )

    try:
        # Create and fit model on training data ONLY
        model = create_baseline(baseline_name)
        model.fit(X_train, y_train)

        # Evaluate on held-out validation sessions
        y_pred = model.predict(X_val)

        # Compute metrics on held-out data
        metrics = metrics_calc.compute(y_pred, y_val)

    except Exception as e:
        print(f"    Warning: Evaluation failed for {baseline_name}: {e}")
        metrics = {"r2": np.nan, "mae": np.nan, "pearson": np.nan}
        y_pred = None
        model = None

    # Compute per-session R² if session info is available
    per_session_r2 = {}
    if val_idx_per_session is not None and X_full is not None and y_full is not None and model is not None:
        from sklearn.metrics import r2_score
        for sess_name, sess_idx in val_idx_per_session.items():
            X_sess = X_full[sess_idx]
            y_sess = y_full[sess_idx]
            try:
                y_pred_sess = model.predict(X_sess)
                # Flatten for r2_score
                r2_sess = r2_score(y_sess.flatten(), y_pred_sess.flatten())
                per_session_r2[sess_name] = float(r2_sess)
            except Exception as e:
                print(f"    Warning: Per-session R² failed for {sess_name}: {e}")
                per_session_r2[sess_name] = np.nan

    # Extract metrics (single evaluation, no folds)
    r2 = metrics.get("r2", np.nan)
    mae = metrics.get("mae", np.nan)
    pearson = metrics.get("pearson", np.nan)
    spearman = metrics.get("spearman", np.nan)

    # Band R²
    band_r2 = {}
    for band in NEURAL_BANDS.keys():
        band_r2[band] = metrics.get(f"r2_{band}", np.nan)

    # PSD error
    psd_error_db = metrics.get("psd_error_db", np.nan)

    # DSP metrics
    plv = metrics.get("plv", 0.0)
    pli = metrics.get("pli", 0.0)
    wpli = metrics.get("wpli", 0.0)
    coherence = metrics.get("coherence", 0.0)
    recon_snr = metrics.get("reconstruction_snr_db", 0.0)
    env_corr = metrics.get("envelope_corr", 0.0)
    mi = metrics.get("mutual_info", 0.0)
    nmi = metrics.get("normalized_mi", 0.0)
    psd_corr = metrics.get("psd_correlation", 0.0)

    # Coherence bands
    coherence_bands = {}
    for band in NEURAL_BANDS.keys():
        coh_bands = metrics.get("coherence_bands", {})
        coherence_bands[band] = coh_bands.get(band, 0.0) if isinstance(coh_bands, dict) else 0.0

    # Band power errors
    band_power_errors = {}
    for band in NEURAL_BANDS.keys():
        bp_errors = metrics.get("band_power_errors", {})
        band_power_errors[band] = bp_errors.get(band, 0.0) if isinstance(bp_errors, dict) else 0.0

    # For cross-subject, we have a single evaluation (no folds)
    # Store as single-element list for compatibility
    result = Phase1Metrics(
        method=baseline_name,
        r2_mean=float(r2) if not np.isnan(r2) else np.nan,
        r2_std=0.0,  # Single evaluation, no std
        r2_ci=(float(r2), float(r2)),  # Point estimate
        mae_mean=float(mae) if not np.isnan(mae) else np.nan,
        mae_std=0.0,
        pearson_mean=float(pearson) if not np.isnan(pearson) else np.nan,
        pearson_std=0.0,
        spearman_mean=float(spearman) if not np.isnan(spearman) else np.nan,
        spearman_std=0.0,
        psd_error_db=float(psd_error_db) if not np.isnan(psd_error_db) else np.nan,
        band_r2=band_r2,
        fold_r2s=[r2],  # Single value for compatibility
        fold_maes=[mae],
        fold_pearsons=[pearson],
        n_folds=1,  # Cross-subject = single split
        n_samples=X_val.shape[0],
        # DSP metrics
        plv_mean=plv,
        plv_std=0.0,
        pli_mean=pli,
        pli_std=0.0,
        wpli_mean=wpli,
        wpli_std=0.0,
        coherence_mean=coherence,
        coherence_std=0.0,
        coherence_bands=coherence_bands,
        reconstruction_snr_db=recon_snr,
        envelope_corr_mean=env_corr,
        envelope_corr_std=0.0,
        mutual_info_mean=mi,
        normalized_mi_mean=nmi,
        psd_correlation=psd_corr,
        band_power_errors=band_power_errors,
        # Single fold data
        fold_plvs=[plv],
        fold_plis=[pli],
        fold_coherences=[coherence],
        fold_envelope_corrs=[env_corr],
        fold_reconstruction_snrs=[recon_snr],
    )

    predictions = (y_pred, y_val) if y_pred is not None else None

    return result, predictions, model, per_session_r2


def run_session_cv(
    config: Phase1Config,
    n_folds: int = 3,
) -> None:
    """Run K-fold session-based cross-validation to investigate variance.

    With 9 sessions, splits into 3 folds of 3 sessions each.
    Each fold: train on 6 sessions, evaluate on 3 held-out sessions.

    Args:
        config: Configuration object
        n_folds: Number of folds (default 3 for 9 sessions)
    """
    from data import prepare_data, load_session_ids, ODOR_CSV_PATH
    from sklearn.metrics import r2_score

    print("\n" + "=" * 70)
    print("SESSION-BASED CROSS-VALIDATION INVESTIGATION")
    print("=" * 70)

    # Load data without splitting - we'll do our own splits
    print("Loading full dataset...")
    data = prepare_data(
        split_by_session=False,
        force_recreate_splits=True,
    )

    X = data["ob"]
    y = data["pcx"]
    odors = data["odors"]

    # Get session IDs
    session_ids, session_to_idx, idx_to_session = load_session_ids(
        ODOR_CSV_PATH, num_trials=len(odors)
    )

    unique_sessions = np.unique(session_ids)
    n_sessions = len(unique_sessions)
    print(f"Total sessions: {n_sessions}")
    print(f"Sessions: {[idx_to_session[s] for s in unique_sessions]}")

    # Shuffle sessions
    rng = np.random.default_rng(config.seed)
    shuffled_sessions = unique_sessions.copy()
    rng.shuffle(shuffled_sessions)

    # Split into folds
    sessions_per_fold = n_sessions // n_folds
    print(f"\nSplitting {n_sessions} sessions into {n_folds} folds ({sessions_per_fold} sessions each)")

    fold_sessions = []
    for i in range(n_folds):
        start = i * sessions_per_fold
        end = start + sessions_per_fold if i < n_folds - 1 else n_sessions
        fold_sessions.append(shuffled_sessions[start:end])
        sess_names = [idx_to_session[s] for s in fold_sessions[i]]
        print(f"  Fold {i+1}: {sess_names}")

    # Results storage
    all_fold_results = {method: [] for method in config.baselines}
    all_session_results = {method: {} for method in config.baselines}

    print("\n" + "-" * 70)

    # Run each fold
    for fold_idx in range(n_folds):
        val_session_set = set(fold_sessions[fold_idx].tolist())
        train_session_set = set(shuffled_sessions.tolist()) - val_session_set

        # Create indices
        all_indices = np.arange(len(session_ids))
        train_idx = all_indices[np.isin(session_ids, list(train_session_set))]
        val_idx = all_indices[np.isin(session_ids, list(val_session_set))]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        val_sess_names = [idx_to_session[s] for s in fold_sessions[fold_idx]]
        print(f"\nFOLD {fold_idx + 1}: Val sessions = {val_sess_names}")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        # Evaluate each baseline
        for baseline_name in config.baselines:
            try:
                model = create_baseline(baseline_name)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                # Overall R²
                r2 = r2_score(y_val.flatten(), y_pred.flatten())
                all_fold_results[baseline_name].append(r2)

                # Per-session R² within this fold
                for sess_id in fold_sessions[fold_idx]:
                    sess_name = idx_to_session[sess_id]
                    sess_mask = session_ids[val_idx] == sess_id
                    y_sess = y_val[sess_mask]
                    y_pred_sess = y_pred[sess_mask]
                    r2_sess = r2_score(y_sess.flatten(), y_pred_sess.flatten())

                    if sess_name not in all_session_results[baseline_name]:
                        all_session_results[baseline_name][sess_name] = r2_sess

                print(f"  {baseline_name}: R² = {r2:.4f}")

            except Exception as e:
                print(f"  {baseline_name}: FAILED - {e}")
                all_fold_results[baseline_name].append(np.nan)

    # Summary
    print("\n" + "=" * 70)
    print("SESSION-CV RESULTS SUMMARY")
    print("=" * 70)

    print("\nPer-Fold R² Values:")
    print("-" * 70)
    header = f"{'Method':<18}" + "".join(f"{'Fold '+str(i+1):>12}" for i in range(n_folds)) + f"{'Mean':>12}{'Std':>12}"
    print(header)
    print("-" * len(header))

    for method in config.baselines:
        fold_vals = all_fold_results[method]
        vals_str = "".join(f"{v:>12.4f}" for v in fold_vals)
        mean_val = np.nanmean(fold_vals)
        std_val = np.nanstd(fold_vals)
        print(f"{method:<18}{vals_str}{mean_val:>12.4f}{std_val:>12.4f}")

    print("\nPer-Session R² Values:")
    print("-" * 70)
    all_sessions_sorted = sorted(all_session_results[config.baselines[0]].keys())
    header = f"{'Method':<18}" + "".join(f"{s:>12}" for s in all_sessions_sorted) + f"{'Mean':>12}"
    print(header)
    print("-" * len(header))

    for method in config.baselines:
        sess_vals = [all_session_results[method].get(s, np.nan) for s in all_sessions_sorted]
        vals_str = "".join(f"{v:>12.4f}" for v in sess_vals)
        mean_val = np.nanmean(sess_vals)
        print(f"{method:<18}{vals_str}{mean_val:>12.4f}")

    # Show which sessions have highest/lowest R²
    print("\nSession Difficulty Analysis (using first method):")
    method = config.baselines[0]
    sess_r2 = all_session_results[method]
    sorted_sessions = sorted(sess_r2.items(), key=lambda x: x[1], reverse=True)
    print(f"  Easiest session: {sorted_sessions[0][0]} (R² = {sorted_sessions[0][1]:.4f})")
    print(f"  Hardest session: {sorted_sessions[-1][0]} (R² = {sorted_sessions[-1][1]:.4f})")
    print(f"  Range: {sorted_sessions[0][1] - sorted_sessions[-1][1]:.4f}")

    print("\n" + "=" * 70)


# =============================================================================
# 3-Fold × 3-Seed Evaluation (Aligned with Ablation Study)
# =============================================================================

def get_3fold_session_splits(all_sessions: List[str]) -> List[Dict[str, Any]]:
    """Create 3-fold CV splits where each fold holds out 3 sessions.

    For 9 sessions [s0, s1, s2, s3, s4, s5, s6, s7, s8]:
    - Fold 0: test=[s0,s1,s2], train=[s3,s4,s5,s6,s7,s8]
    - Fold 1: test=[s3,s4,s5], train=[s0,s1,s2,s6,s7,s8]
    - Fold 2: test=[s6,s7,s8], train=[s0,s1,s2,s3,s4,s5]

    This ensures every session is tested exactly once.
    """
    n_sessions = len(all_sessions)
    if n_sessions < 6:
        raise ValueError(f"Need at least 6 sessions for 3-fold CV, got {n_sessions}")

    # Calculate sessions per fold (handle non-divisible cases)
    sessions_per_fold = n_sessions // 3
    remainder = n_sessions % 3

    splits = []
    start_idx = 0

    for fold_idx in range(3):
        # Handle remainder by adding extra session to first folds
        n_test = sessions_per_fold + (1 if fold_idx < remainder else 0)
        end_idx = start_idx + n_test

        test_sessions = all_sessions[start_idx:end_idx]
        train_sessions = all_sessions[:start_idx] + all_sessions[end_idx:]

        splits.append({
            "fold_idx": fold_idx,
            "test_sessions": test_sessions,
            "train_sessions": train_sessions,
        })

        start_idx = end_idx

    return splits


def run_single_fold_seed(
    baseline_name: str,
    X_train_all: NDArray,
    y_train_all: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    fold_idx: int,
    seed: int,
    test_sessions: List[str],
    train_sessions: List[str],
    session_to_test_indices: Dict[str, NDArray],
    config: Phase1Config,
) -> FoldSeedResult:
    """Run evaluation for a single fold+seed combination.

    PROPER DATA SPLIT (NO DATA LEAKAGE):
    1. X_train_all (from 6 training sessions) → 70/30 trial-wise split → TRAIN / VAL
    2. Model is fit on TRAIN only
    3. Evaluate on VAL (validation metric)
    4. Evaluate on TEST (PRIMARY metric - from held-out sessions)

    Args:
        baseline_name: Name of the baseline method
        X_train_all: All training input [N_train, C, T] from 6 training sessions
        y_train_all: All training target [N_train, C, T]
        X_test: Test input (held-out sessions) [N_test, C, T]
        y_test: Test target [N_test, C, T]
        fold_idx: Fold index (0, 1, or 2)
        seed: Random seed
        test_sessions: List of session names in test set
        train_sessions: List of session names in training set
        session_to_test_indices: Dict mapping session name -> indices in test set
        config: Configuration object

    Returns:
        FoldSeedResult with val and test metrics
    """
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split

    np.random.seed(seed)

    start_time = time.time()

    metrics_calc = MetricsCalculator(
        sample_rate=config.sample_rate,
        n_bootstrap=100,  # Reduced for speed
        ci_level=config.ci_level,
    )

    try:
        # STEP 1: 70/30 trial-wise split on training data
        n_train_total = len(X_train_all)
        indices = np.arange(n_train_total)
        train_idx, val_idx = train_test_split(
            indices, test_size=0.3, random_state=seed
        )

        X_train = X_train_all[train_idx]
        y_train = y_train_all[train_idx]
        X_val = X_train_all[val_idx]
        y_val = y_train_all[val_idx]

        # STEP 2: Fit model on TRAIN only (70%)
        model = create_baseline(baseline_name)
        model.fit(X_train, y_train)

        # STEP 3: Evaluate on VALIDATION (30% of training sessions)
        y_val_pred = model.predict(X_val)
        val_r2 = float(r2_score(y_val.flatten(), y_val_pred.flatten()))
        val_mae = float(mean_absolute_error(y_val.flatten(), y_val_pred.flatten()))
        val_pearson = float(np.corrcoef(y_val.flatten(), y_val_pred.flatten())[0, 1])

        # STEP 4: Evaluate on TEST (held-out sessions) - PRIMARY METRIC
        y_test_pred = model.predict(X_test)
        test_r2 = float(r2_score(y_test.flatten(), y_test_pred.flatten()))
        test_mae = float(mean_absolute_error(y_test.flatten(), y_test_pred.flatten()))
        test_pearson = float(np.corrcoef(y_test.flatten(), y_test_pred.flatten())[0, 1])

        # Compute per-session TEST metrics
        per_session_test_r2 = {}
        per_session_test_mae = {}
        per_session_test_corr = {}

        for session_name, indices in session_to_test_indices.items():
            y_sess_true = y_test[indices]
            y_sess_pred = y_test_pred[indices]

            per_session_test_r2[session_name] = float(r2_score(
                y_sess_true.flatten(), y_sess_pred.flatten()
            ))
            per_session_test_mae[session_name] = float(mean_absolute_error(
                y_sess_true.flatten(), y_sess_pred.flatten()
            ))
            per_session_test_corr[session_name] = float(np.corrcoef(
                y_sess_true.flatten(), y_sess_pred.flatten()
            )[0, 1])

        # Compute DSP metrics (on TEST predictions)
        dsp_metrics = metrics_calc.compute(y_test_pred, y_test)
        plv = dsp_metrics.get("plv", 0.0)
        coherence = dsp_metrics.get("coherence", 0.0)
        reconstruction_snr = dsp_metrics.get("reconstruction_snr_db", 0.0)
        envelope_corr = dsp_metrics.get("envelope_corr", 0.0)

        # Band R² (from test)
        band_r2 = {}
        for band in NEURAL_BANDS.keys():
            band_r2[band] = dsp_metrics.get(f"r2_{band}", 0.0)

    except Exception as e:
        print(f"    Warning: Fold {fold_idx} seed {seed} failed for {baseline_name}: {e}")
        val_r2, val_mae, val_pearson = np.nan, np.nan, np.nan
        test_r2, test_mae, test_pearson = np.nan, np.nan, np.nan
        per_session_test_r2, per_session_test_mae, per_session_test_corr = {}, {}, {}
        plv, coherence, reconstruction_snr, envelope_corr = 0.0, 0.0, 0.0, 0.0
        band_r2 = {}

    elapsed = time.time() - start_time

    return FoldSeedResult(
        fold_idx=fold_idx,
        seed=seed,
        test_sessions=test_sessions,
        train_sessions=train_sessions,
        # Validation metrics (from 70/30 split)
        val_r2=val_r2,
        val_mae=val_mae,
        val_pearson=val_pearson,
        # TEST metrics (PRIMARY - from held-out sessions)
        test_r2=test_r2,
        test_mae=test_mae,
        test_pearson=test_pearson,
        # Per-session TEST metrics
        per_session_test_r2=per_session_test_r2,
        per_session_test_mae=per_session_test_mae,
        per_session_test_corr=per_session_test_corr,
        # DSP metrics
        plv=plv,
        coherence=coherence,
        reconstruction_snr_db=reconstruction_snr,
        envelope_corr=envelope_corr,
        band_r2=band_r2,
        total_time=elapsed,
    )


def run_3fold_3seed_evaluation(
    config: Phase1Config,
    n_folds: int = N_FOLDS,
    n_seeds: int = N_SEEDS,
    base_seed: int = BASE_SEED,
) -> Phase1Result3Fold3Seed:
    """Run complete 3-fold × 3-seed Phase 1 evaluation.

    PROPER DATA SPLIT (NO DATA LEAKAGE):
    For each fold:
    - 3 sessions → TEST (held out, never seen during training)
    - 6 sessions → 70/30 trial-wise split:
      - ~70% trials → TRAIN (fit model)
      - ~30% trials → VAL (validation metric)

    Classical baselines are fit once on TRAIN, evaluated on VAL and TEST.
    TEST R² is the PRIMARY metric (true generalization, NO leakage).

    Args:
        config: Configuration object
        n_folds: Number of cross-validation folds (default: 3)
        n_seeds: Number of seeds per fold (default: 3)
        base_seed: Base random seed

    Returns:
        Phase1Result3Fold3Seed with all results
    """
    from data import prepare_data, load_session_ids, ODOR_CSV_PATH

    print("\n" + "=" * 70)
    print("PHASE 1: Classical Baselines Evaluation")
    print("Strategy: 3-Fold × 3-Seed Cross-Validation")
    print("=" * 70)
    print(f"  Folds: {n_folds}")
    print(f"  Seeds per fold: {n_seeds}")
    print(f"  Total runs per method: {n_folds * n_seeds}")
    print(f"  Methods: {', '.join(config.baselines)}")
    print()

    # CRITICAL: Load raw data WITHOUT normalization - we compute per-fold normalization
    # to avoid data leakage (each fold's normalization uses only its training sessions)
    print("Loading raw dataset (no normalization - computed per fold)...")
    from data import load_signals, load_odor_labels, extract_window, compute_normalization, normalize, DATA_PATH

    signals = load_signals(DATA_PATH)
    num_trials = signals.shape[0]
    odors, vocab = load_odor_labels(ODOR_CSV_PATH, num_trials)
    windowed = extract_window(signals)  # Raw unnormalized data

    # Get session IDs
    session_ids, session_to_idx, idx_to_session = load_session_ids(
        ODOR_CSV_PATH, num_trials=num_trials
    )

    unique_sessions = np.unique(session_ids)
    all_sessions = [idx_to_session[s] for s in unique_sessions]
    print(f"  Total sessions: {len(all_sessions)}")
    print(f"  Sessions: {all_sessions}")

    # Create 3-fold session splits
    fold_splits = get_3fold_session_splits(all_sessions)
    print(f"\n3-Fold Session Splits:")
    for split in fold_splits:
        print(f"  Fold {split['fold_idx']}: test={split['test_sessions']}")
    print()

    # Run evaluation for each baseline
    method_results: Dict[str, MethodResult] = {}
    total_start_time = time.time()

    for method_idx, baseline_name in enumerate(config.baselines):
        print(f"\n{'#'*70}")
        print(f"# [{method_idx + 1}/{len(config.baselines)}] {baseline_name}")
        print(f"{'#'*70}")

        fold_seed_results = []

        for fold_split in fold_splits:
            fold_idx = fold_split["fold_idx"]
            test_sessions = fold_split["test_sessions"]
            train_sessions = fold_split["train_sessions"]

            # Get indices for train and test
            test_session_ids = set(session_to_idx[s] for s in test_sessions)
            train_session_ids = set(session_to_idx[s] for s in train_sessions)

            all_indices = np.arange(len(session_ids))
            train_idx = all_indices[np.isin(session_ids, list(train_session_ids))]
            test_idx = all_indices[np.isin(session_ids, list(test_session_ids))]

            # CRITICAL FIX: Compute normalization from TRAINING sessions only (no leakage)
            # This ensures test sessions don't influence normalization statistics
            fold_norm_stats = compute_normalization(windowed, train_idx)
            normalized = normalize(windowed, fold_norm_stats)
            X_fold = normalized[:, 0]  # OB
            y_fold = normalized[:, 1]  # PCx

            X_train, y_train = X_fold[train_idx], y_fold[train_idx]
            X_test, y_test = X_fold[test_idx], y_fold[test_idx]

            # Create session-to-indices mapping for per-session metrics
            # (indices relative to X_test)
            session_to_test_indices = {}
            test_session_ids_arr = session_ids[test_idx]
            for sess_name in test_sessions:
                sess_id = session_to_idx[sess_name]
                local_mask = test_session_ids_arr == sess_id
                session_to_test_indices[sess_name] = np.where(local_mask)[0]

            print(f"\n  Fold {fold_idx}: test={test_sessions}, train={len(train_idx)} samples (70/30 split)")

            # Run with multiple seeds
            for seed_offset in range(n_seeds):
                seed = base_seed + fold_idx * n_seeds + seed_offset
                print(f"    Seed {seed}...", end=" ", flush=True)

                result = run_single_fold_seed(
                    baseline_name=baseline_name,
                    X_train_all=X_train,  # Will be split 70/30 internally
                    y_train_all=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    fold_idx=fold_idx,
                    seed=seed,
                    test_sessions=test_sessions,
                    train_sessions=train_sessions,
                    session_to_test_indices=session_to_test_indices,
                    config=config,
                )

                fold_seed_results.append(result)
                print(f"TEST R² = {result.test_r2:.4f} (val R² = {result.val_r2:.4f})")

        # Create method result and compute statistics
        method_result = MethodResult(
            method=baseline_name,
            fold_seed_results=fold_seed_results,
        )
        method_result.compute_statistics()
        method_results[baseline_name] = method_result

        # Print method summary
        print(f"\n  {baseline_name} Summary:")
        print(f"    TEST R² = {method_result.mean_r2:.4f} ± {method_result.std_r2:.4f} (PRIMARY)")
        print(f"    Val R² = {method_result.mean_val_r2:.4f} ± {method_result.std_val_r2:.4f} (reference)")
        print(f"    95% CI = [{method_result.ci_lower_r2:.4f}, {method_result.ci_upper_r2:.4f}]")
        print(f"    Per-session TEST R²: {method_result.all_session_r2s}")

    total_time = time.time() - total_start_time

    # Find best method
    best_method_result = max(method_results.values(), key=lambda r: r.mean_r2)
    best_method = best_method_result.method
    best_r2 = best_method_result.mean_r2
    gate_threshold = get_gate_threshold(best_r2)

    # Compute statistical comparisons between methods
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISONS (vs best method)")
    print("=" * 70)

    statistical_comparisons = compute_method_comparisons(method_results, best_method)

    # Create result
    result = Phase1Result3Fold3Seed(
        method_results=method_results,
        best_method=best_method,
        best_r2=best_r2,
        gate_threshold=gate_threshold,
        n_folds=n_folds,
        n_seeds=n_seeds,
        total_runs=n_folds * n_seeds,
        statistical_comparisons=statistical_comparisons,
        config=config.to_dict(),
        total_time=total_time,
    )

    # Print final summary
    print_3fold_3seed_summary(result)

    return result


def compute_method_comparisons(
    method_results: Dict[str, MethodResult],
    best_method: str,
) -> Dict[str, Dict[str, Any]]:
    """Compute statistical comparisons between methods.

    Uses paired t-tests on fold-level means (3 values per method).
    """
    from scipy import stats as scipy_stats

    best = method_results[best_method]
    best_fold_r2s = np.array(best.fold_r2_values)

    comparisons = {}

    for method_name, result in method_results.items():
        if method_name == best_method:
            continue

        other_fold_r2s = np.array(result.fold_r2_values)

        if len(other_fold_r2s) < 2 or len(best_fold_r2s) < 2:
            continue

        # Mean difference
        mean_diff = result.mean_r2 - best.mean_r2
        diff = other_fold_r2s - best_fold_r2s

        # Cohen's d (paired)
        if np.std(diff, ddof=1) > 0:
            cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        else:
            cohens_d = 0.0

        # Effect size interpretation
        d_abs = abs(cohens_d)
        if d_abs < 0.2:
            effect_interp = "negligible"
        elif d_abs < 0.5:
            effect_interp = "small"
        elif d_abs < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"

        # Paired t-test (on fold means)
        try:
            t_stat, t_pvalue = scipy_stats.ttest_rel(other_fold_r2s, best_fold_r2s)
        except Exception:
            t_stat, t_pvalue = 0.0, 1.0

        # Wilcoxon signed-rank test (non-parametric)
        try:
            diff_nonzero = diff[diff != 0]
            if len(diff_nonzero) >= 2:
                w_stat, w_pvalue = scipy_stats.wilcoxon(diff_nonzero)
            else:
                w_stat, w_pvalue = 0.0, 1.0
        except Exception:
            w_stat, w_pvalue = 0.0, 1.0

        # Significance markers
        if t_pvalue < 0.001:
            sig_marker = "***"
        elif t_pvalue < 0.01:
            sig_marker = "**"
        elif t_pvalue < 0.05:
            sig_marker = "*"
        else:
            sig_marker = "ns"

        comparisons[method_name] = {
            "mean_diff": float(mean_diff),
            "cohens_d": float(cohens_d),
            "effect_interpretation": effect_interp,
            "t_statistic": float(t_stat),
            "t_pvalue": float(t_pvalue),
            "wilcoxon_pvalue": float(w_pvalue),
            "significance_marker": sig_marker,
            "significant_005": t_pvalue < 0.05,
        }

        print(f"  {method_name} vs {best_method}: d={cohens_d:.2f} ({effect_interp}), p={t_pvalue:.4f} {sig_marker}")

    return comparisons


def print_3fold_3seed_summary(result: Phase1Result3Fold3Seed) -> None:
    """Print comprehensive summary of 3-fold × 3-seed results.

    NOTE: All R² values shown are TEST R² from held-out sessions (NO data leakage).
    """
    print("\n" + "=" * 85)
    print("PHASE 1 RESULTS SUMMARY (3-Fold × 3-Seed)")
    print("=" * 85)
    print("NOTE: All R² values are from TEST (held-out sessions) - NO data leakage")
    print("      Model was fit on 70% of training sessions, val on 30%, test held out")
    print()

    # Sort by mean TEST R²
    sorted_methods = sorted(
        result.method_results.items(),
        key=lambda x: x[1].mean_r2,
        reverse=True
    )

    print(f"{'Method':<18} {'TEST R² (mean±std)':<20} {'95% CI':<22} {'MAE':>8} {'Pearson':>8}")
    print("-" * 85)

    for method_name, method_result in sorted_methods:
        r2_str = f"{method_result.mean_r2:.4f}±{method_result.std_r2:.4f}"
        ci_str = f"[{method_result.ci_lower_r2:.4f}, {method_result.ci_upper_r2:.4f}]"
        mae_str = f"{method_result.mean_mae:.4f}"
        pearson_str = f"{method_result.mean_pearson:.4f}"
        print(f"{method_name:<18} {r2_str:<20} {ci_str:<22} {mae_str:>8} {pearson_str:>8}")

    # Per-session TEST summary
    print("\n" + "-" * 85)
    print("PER-SESSION TEST R² (from held-out sessions, mean across 3 seeds):")
    print("-" * 85)

    # Get all sessions
    all_sessions = set()
    for method_result in result.method_results.values():
        all_sessions.update(method_result.all_session_r2s.keys())
    all_sessions = sorted(all_sessions)

    header = f"{'Method':<18}" + "".join(f"{s:>12}" for s in all_sessions)
    print(header)
    print("-" * len(header))

    for method_name, method_result in sorted_methods:
        sess_vals = [method_result.all_session_r2s.get(s, np.nan) for s in all_sessions]
        sess_str = "".join(f"{v:>12.4f}" for v in sess_vals)
        print(f"{method_name:<18}{sess_str}")

    # Gate threshold
    print("\n" + "=" * 85)
    print(f"CLASSICAL FLOOR: TEST R² = {result.best_r2:.4f} ({result.best_method})")
    print(f"GATE THRESHOLD: Neural methods must achieve TEST R² >= {result.gate_threshold:.4f}")
    print(f"Total evaluation time: {result.total_time / 60:.1f} minutes")
    print("=" * 85)


# =============================================================================
# Main Runner (Legacy - kept for backward compatibility)
# =============================================================================

def run_phase1(
    config: Phase1Config,
    X_train: NDArray,
    y_train: NDArray,
    X_val: Optional[NDArray] = None,
    y_val: Optional[NDArray] = None,
    val_idx_per_session: Optional[Dict[str, NDArray]] = None,
    X_full: Optional[NDArray] = None,
    y_full: Optional[NDArray] = None,
) -> Phase1Result:
    """Run complete Phase 1 evaluation.

    Args:
        config: Configuration object
        X_train: Training input signals [N_train, C, T]
        y_train: Training target signals [N_train, C, T]
        X_val: Validation input signals [N_val, C, T] (held-out sessions)
        y_val: Validation target signals [N_val, C, T] (held-out sessions)
        val_idx_per_session: Dict mapping session name -> indices (for per-session metrics)
        X_full: Full input array (needed for per-session metrics)
        y_full: Full target array (needed for per-session metrics)

    Returns:
        Phase1Result with all metrics and analysis

    Cross-subject evaluation mode (when X_val/y_val provided):
        - Train on X_train/y_train (train sessions only)
        - Evaluate on X_val/y_val (held-out sessions)
        - No K-fold CV - true cross-subject generalization

    Legacy mode (when X_val/y_val are None):
        - Use K-fold CV on X_train/y_train
    """
    cross_subject_mode = X_val is not None and y_val is not None

    print("\n" + "=" * 70)
    print("PHASE 1: Classical Baselines Evaluation")
    print("=" * 70)
    print(f"Dataset: {config.dataset}")
    if cross_subject_mode:
        print(f"Mode: CROSS-SUBJECT (train on {X_train.shape[0]} samples, eval on {X_val.shape[0]} held-out samples)")
        if val_idx_per_session:
            print(f"Per-session evaluation: {list(val_idx_per_session.keys())}")
    else:
        print(f"Mode: K-fold CV on {X_train.shape[0]} samples ({config.n_folds} folds)")
    print(f"Methods: {', '.join(config.baselines)}")
    print()

    # Setup checkpoint for resume capability
    checkpoint_path = get_checkpoint_path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load checkpoint for resume
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is not None:
        all_metrics, all_predictions, all_models, method_fold_r2s, completed_methods = reconstruct_from_checkpoint(checkpoint)
    else:
        all_metrics = []
        all_predictions = {}
        all_models = {}
        method_fold_r2s = {}
        completed_methods = set()

    skipped_methods = 0
    all_per_session_r2 = {}  # Store per-session R² for each method

    # Evaluate each baseline
    for i, baseline_name in enumerate(config.baselines):
        # Check if already completed
        if baseline_name in completed_methods:
            skipped_methods += 1
            print(f"[{i+1}/{len(config.baselines)}] {baseline_name} - SKIPPED (already completed)")
            continue

        print(f"[{i+1}/{len(config.baselines)}] Evaluating {baseline_name}...")
        start_time = time.time()

        if cross_subject_mode:
            # Cross-subject: train on train sessions, eval on held-out val sessions
            result, predictions, model, per_session_r2 = evaluate_baseline_cross_subject(
                baseline_name, X_train, y_train, X_val, y_val, config, return_model=True,
                val_idx_per_session=val_idx_per_session, X_full=X_full, y_full=y_full
            )
            if per_session_r2:
                all_per_session_r2[baseline_name] = per_session_r2
        else:
            # Legacy K-fold CV mode
            result, predictions, model = evaluate_baseline(baseline_name, X_train, y_train, config, return_model=True)
        all_metrics.append(result)

        if predictions is not None:
            all_predictions[baseline_name] = predictions[0]  # y_pred

        if model is not None:
            all_models[baseline_name] = model

        method_fold_r2s[baseline_name] = result.fold_r2s

        elapsed = time.time() - start_time
        ci_str = f"[{result.r2_ci[0]:.4f}, {result.r2_ci[1]:.4f}]"
        print(f"    R² = {result.r2_mean:.4f} ± {result.r2_std:.4f} {ci_str} ({elapsed:.1f}s)")

        # Print per-session R² if available
        if cross_subject_mode and baseline_name in all_per_session_r2:
            per_sess = all_per_session_r2[baseline_name]
            sess_str = ", ".join([f"{s}: {r2:.4f}" for s, r2 in sorted(per_sess.items())])
            print(f"    Per-session R²: {sess_str}")

        # Print key DSP metrics
        print(f"    PLV = {result.plv_mean:.4f}, Coherence = {result.coherence_mean:.4f}, SNR = {result.reconstruction_snr_db:.1f} dB")

        # Save checkpoint after each method
        completed_methods.add(baseline_name)
        save_checkpoint(checkpoint_path, all_metrics, all_predictions, all_models, method_fold_r2s, completed_methods)
        print(f"    Checkpoint saved ({len(completed_methods)}/{len(config.baselines)} methods completed)")

    if skipped_methods > 0:
        print(f"\n*** Resumed run: skipped {skipped_methods} already-completed methods ***")

    # Find best method
    valid_metrics = [m for m in all_metrics if not np.isnan(m.r2_mean)]
    if not valid_metrics:
        raise RuntimeError("All baselines failed!")

    best = max(valid_metrics, key=lambda m: m.r2_mean)
    gate_threshold = get_gate_threshold(best.r2_mean)

    # Statistical comparisons (legacy)
    comparisons = compute_statistical_comparisons(method_fold_r2s)

    # Comprehensive statistical analysis using new utilities
    comprehensive_stats = {}
    assumption_checks = {}

    if len(method_fold_r2s) >= 2:
        # Compare all methods against the best method
        print("\nComputing comprehensive statistical analysis...")

        for method_name, fold_values in method_fold_r2s.items():
            if method_name == best.method:
                continue

            values_best = np.array(method_fold_r2s[best.method])
            values_other = np.array(fold_values)

            # Skip if not enough data
            if len(values_best) < 2 or len(values_other) < 2:
                continue

            # Comprehensive comparison (parametric + non-parametric)
            comp_result = compare_methods(
                values_best, values_other,
                name_a=best.method, name_b=method_name,
                paired=True, alpha=0.05
            )

            comprehensive_stats[method_name] = comp_result.to_dict()

            # Check assumptions
            assumptions = check_assumptions(values_best, values_other, alpha=0.05)
            assumption_checks[method_name] = assumptions

        # Apply multiple comparison correction (Holm method)
        if comprehensive_stats:
            p_values = [comprehensive_stats[m]["parametric_test"]["p_value"]
                       for m in comprehensive_stats]
            significant_holm = holm_correction(p_values, alpha=0.05)

            for i, method_name in enumerate(comprehensive_stats.keys()):
                comprehensive_stats[method_name]["significant_holm"] = significant_holm[i]

            # Also apply FDR correction
            significant_fdr, adjusted_p = fdr_correction(p_values, alpha=0.05)
            for i, method_name in enumerate(comprehensive_stats.keys()):
                comprehensive_stats[method_name]["significant_fdr"] = significant_fdr[i]
                comprehensive_stats[method_name]["p_fdr_adjusted"] = adjusted_p[i]

    # Create result
    result = Phase1Result(
        metrics=all_metrics,
        best_method=best.method,
        best_r2=best.r2_mean,
        gate_threshold=gate_threshold,
        statistical_comparisons=comparisons,
        comprehensive_stats=comprehensive_stats,
        assumption_checks=assumption_checks,
        models=all_models if all_models else None,
        config=config.to_dict(),
    )

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 1 RESULTS SUMMARY")
    print("=" * 70)
    print(format_metrics_table(all_metrics))

    # DSP Metrics Summary
    print("\nDSP Metrics Summary:")
    print("-" * 90)
    print(f"{'Method':<18}{'PLV':>8}{'PLI':>8}{'Coh':>8}{'Env.Corr':>10}{'SNR(dB)':>10}{'NMI':>8}")
    print("-" * 90)
    for m in sorted(all_metrics, key=lambda x: -x.r2_mean):
        print(f"{m.method:<18}{m.plv_mean:>8.4f}{m.pli_mean:>8.4f}{m.coherence_mean:>8.4f}"
              f"{m.envelope_corr_mean:>10.4f}{m.reconstruction_snr_db:>10.2f}{m.normalized_mi_mean:>8.4f}")

    # Per-band breakdown for top 3
    print("\nPer-Band R² (Top 3 Methods):")
    print("-" * 70)
    print(f"{'Method':<18}" + "".join(f"{b.capitalize():>10}" for b in NEURAL_BANDS.keys()))
    print("-" * 70)

    ranked = rank_methods(all_metrics)[:3]
    for _, m in ranked:
        bands = "".join(f"{m.band_r2.get(b, np.nan):>10.4f}" for b in NEURAL_BANDS.keys())
        print(f"{m.method:<18}{bands}")

    # Per-Session R² Summary (cross-subject mode only)
    if all_per_session_r2:
        session_names = sorted(list(next(iter(all_per_session_r2.values())).keys()))
        print("\n" + "=" * 70)
        print("PER-SESSION R² VALUES (Cross-Subject Evaluation)")
        print("=" * 70)
        # Header
        header = f"{'Method':<18}" + "".join(f"{s:>12}" for s in session_names) + f"{'Mean':>12}"
        print(header)
        print("-" * len(header))
        # Rows (sorted by overall R²)
        for m in sorted(all_metrics, key=lambda x: -x.r2_mean):
            if m.method in all_per_session_r2:
                per_sess = all_per_session_r2[m.method]
                sess_vals = [per_sess.get(s, np.nan) for s in session_names]
                sess_str = "".join(f"{v:>12.4f}" for v in sess_vals)
                mean_val = np.nanmean(sess_vals)
                print(f"{m.method:<18}{sess_str}{mean_val:>12.4f}")

    # Statistical summary
    sig_comparisons = [c for c in comparisons if c.significant]
    if sig_comparisons:
        print(f"\nSignificant Differences ({len(sig_comparisons)} comparisons, p < 0.05 Bonferroni):")
        for c in sorted(sig_comparisons, key=lambda x: -abs(x.cohens_d))[:5]:
            print(f"  {c.method1} vs {c.method2}: d={c.cohens_d:.2f} ({c.effect_size}), p_adj={c.p_adjusted:.4f}")

    # Comprehensive stats summary
    if comprehensive_stats:
        print("\nComprehensive Statistical Analysis (vs best method):")
        print("-" * 70)
        print("{:<15}{:<12}{:<12}{:<12}{:<8}{:<8}".format("Method", "t-test p", "Wilcoxon p", "Cohen's d", "Holm", "FDR"))
        print("-" * 70)
        for method_name, stats in comprehensive_stats.items():
            t_p = stats["parametric_test"]["p_value"]
            w_p = stats["nonparametric_test"]["p_value"]
            d = stats["parametric_test"]["effect_size"] or 0
            holm_sig = "Yes" if stats.get("significant_holm", False) else "No"
            fdr_sig = "Yes" if stats.get("significant_fdr", False) else "No"
            print(f"{method_name:<15}{t_p:<12.4f}{w_p:<12.4f}{d:<12.3f}{holm_sig:<8}{fdr_sig:<8}")

        # Assumption check summary
        print("\nAssumption Checks (Normality of differences):")
        for method_name, checks in assumption_checks.items():
            norm_ok = "OK" if checks["normality_diff"]["ok"] else "VIOLATED"
            rec = checks["recommendation"]
            print(f"  {method_name}: {norm_ok} (recommended: {rec})")

    # Gate threshold
    print("\n" + "=" * 70)
    print(f"CLASSICAL FLOOR: R² = {best.r2_mean:.4f} ({best.method})")
    print(f"GATE THRESHOLD: Neural methods must achieve R² >= {gate_threshold:.4f}")
    print("=" * 70)

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for Phase 1."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Classical Baselines Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m phase_one.runner --dataset olfactory
    python -m phase_one.runner --dry-run  # Quick test
    python -m phase_one.runner --output results/phase1/
    python -m phase_one.runner --3fold-3seed  # Full 3-fold × 3-seed evaluation (aligned with ablation)
    python -m phase_one.runner --3fold-3seed --fast  # Quick 3-fold × 3-seed with fast baselines only
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="olfactory",
        choices=["olfactory", "pfc", "dandi"],
        help="Dataset to evaluate on (default: olfactory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic data for quick testing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase1",
        help="Output directory for results",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Only run fast baselines (ridge, wiener)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--load-results",
        type=str,
        default=None,
        help="Load previous results and regenerate figures (path to JSON)",
    )
    parser.add_argument(
        "--figure-format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg", "eps"],
        help="Output format for figures",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
        help="DPI for raster figures (png)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint",
    )
    parser.add_argument(
        "--within-subject",
        action="store_true",
        help="Use within-subject K-fold CV (old mode) instead of cross-subject session-based split",
    )
    parser.add_argument(
        "--session-cv",
        action="store_true",
        help="Run 3-fold session-based CV (9 sessions split into 3 groups of 3)",
    )
    parser.add_argument(
        "--3fold-3seed",
        dest="fold3_seed3",
        action="store_true",
        help="Run 3-fold × 3-seed evaluation (aligned with ablation study strategy)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of seeds per fold (for --3fold-3seed mode)",
    )

    args = parser.parse_args()

    # Handle result loading mode (figures only)
    if args.load_results:
        print(f"Loading results from: {args.load_results}")
        result = Phase1Result.load(Path(args.load_results))

        print(f"Loaded Phase 1 results:")
        print(f"  Best method: {result.best_method} (R² = {result.best_r2:.4f})")
        print(f"  Gate threshold: {result.gate_threshold:.4f}")
        print(f"  Methods evaluated: {len(result.metrics)}")

        # Generate figures
        output_dir = Path(args.output)
        viz = Phase1Visualizer(
            output_dir=output_dir / "figures",
            sample_rate=result.config.get("sample_rate", 1000.0) if result.config else 1000.0,
        )

        fig_path = viz.plot_main_figure(
            metrics_list=result.metrics,
            predictions=None,  # No predictions when loading
            ground_truth=None,
            filename=f"figure_1_1_classical_baselines.{args.figure_format}",
        )
        print(f"\nFigure regenerated: {fig_path}")

        # Stats figure (simple)
        stats_path = viz.plot_statistical_summary(
            result.metrics,
            filename=f"figure_1_1_stats.{args.figure_format}",
        )
        print(f"Stats figure: {stats_path}")

        # Comprehensive statistical analysis figure
        comprehensive_path = viz.plot_comprehensive_stats_figure(
            result.metrics,
            result.statistical_comparisons,
            filename=f"figure_1_2_statistical_analysis.{args.figure_format}",
        )
        print(f"Comprehensive stats figure: {comprehensive_path}")

        # LaTeX table
        latex_table = create_summary_table(result.metrics)
        latex_path = output_dir / "table_classical_baselines.tex"
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table: {latex_path}")

        print("\nFigure regeneration complete!")
        return result

    # Create configuration
    baselines = FAST_BASELINES if args.fast else BASELINE_METHODS
    config = Phase1Config(
        dataset=args.dataset,
        n_folds=args.folds,
        seed=args.seed,
        output_dir=Path(args.output),
        baselines=baselines,
    )

    # Handle session-CV mode (investigation mode)
    if args.session_cv:
        run_session_cv(config, n_folds=3)
        print("\nSession-CV investigation complete!")
        return None

    # Handle 3-fold × 3-seed mode (aligned with ablation study)
    if args.fold3_seed3:
        print("\n[3-FOLD × 3-SEED MODE]")
        result = run_3fold_3seed_evaluation(
            config=config,
            n_folds=N_FOLDS,
            n_seeds=args.n_seeds,
            base_seed=args.seed,
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = config.output_dir / f"phase1_3fold3seed_{timestamp}.json"
        result.save(results_file)
        print(f"\nResults saved to: {results_file}")

        # Save per-session metrics
        per_session_file = config.output_dir / f"phase1_per_session_metrics_{timestamp}.json"
        result.save_per_session_metrics(per_session_file)
        print(f"Per-session metrics saved to: {per_session_file}")

        print("\nPhase 1 (3-fold × 3-seed) complete!")
        return result

    # Load data
    val_idx_per_session = None  # For per-session R² reporting
    X_full, y_full = None, None

    if args.dry_run:
        print("[DRY-RUN] Using synthetic data")
        X, y = create_synthetic_data(n_samples=100, time_points=500)
        ground_truth = y
        # For dry-run, just use all data with internal CV
        X_train, y_train = X, y
        X_val, y_val = None, None
    elif args.within_subject:
        # Within-subject mode: random stratified splits (old evaluation mode)
        from data import prepare_data
        print("Loading olfactory dataset (WITHIN-SUBJECT: random stratified splits)...")
        data = prepare_data(
            split_by_session=False,  # Random splits, NOT session-based
            force_recreate_splits=True,  # Force recreate to match current data
        )

        X = data["ob"]
        y = data["pcx"]
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]

        # Combine train+val for K-fold CV (like the old code)
        cv_idx = np.concatenate([train_idx, val_idx])
        X_train = X[cv_idx]
        y_train = y[cv_idx]
        X_val, y_val = None, None  # Will use K-fold CV
        ground_truth = y_train
        print(f"  Within-subject split: {len(cv_idx)} samples for K-fold CV")
    else:
        try:
            X, y, train_idx, val_idx, test_idx, val_idx_per_session = load_olfactory_data()
            # Cross-subject evaluation: train on train sessions, eval on held-out val sessions
            # NO mixing of sessions - this ensures true cross-subject generalization
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            X_full, y_full = X, y  # Keep full arrays for per-session metrics
            ground_truth = y_val
            print(f"  Cross-subject split: Train {X_train.shape[0]} samples, Val {X_val.shape[0]} samples (held-out sessions)")
        except Exception as e:
            print(f"Warning: Could not load data ({e}). Using synthetic data.")
            X, y = create_synthetic_data()
            X_train, y_train = X, y
            X_val, y_val = None, None
            ground_truth = y

    # Handle --fresh flag: delete existing checkpoint
    if args.fresh:
        checkpoint_path = get_checkpoint_path(config.output_dir)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("Deleted existing checkpoint (--fresh mode)")

    # Run evaluation
    # For cross-subject: train on train sessions, eval on held-out val sessions
    result = run_phase1(
        config, X_train, y_train, X_val, y_val,
        val_idx_per_session=val_idx_per_session,
        X_full=X_full, y_full=y_full
    )

    # Clean up checkpoint after successful completion
    checkpoint_path = get_checkpoint_path(config.output_dir)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint cleaned up after successful completion")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = config.output_dir / f"phase1_results_{timestamp}.json"
    result.save(results_file)
    print(f"\nResults saved to: {results_file}")

    # Save models separately (they can be large)
    if result.models:
        models_file = config.output_dir / f"phase1_models_{timestamp}.pkl"
        result.save_models(models_file)

    # Generate figures
    if not args.no_figures:
        print("\nGenerating figures...")
        viz = Phase1Visualizer(
            output_dir=config.output_dir / "figures",
            sample_rate=config.sample_rate,
        )

        # Collect predictions for visualization
        predictions = {}
        for m in result.metrics:
            # Re-run best method to get predictions for visualization
            if m.method == result.best_method:
                model = create_baseline(m.method)
                model.fit(X, y)
                predictions[m.method] = model.predict(X)

        fig_path = viz.plot_main_figure(
            metrics_list=result.metrics,
            predictions=predictions if predictions else None,
            ground_truth=ground_truth,
        )
        print(f"Figure saved to: {fig_path}")

        # Comprehensive statistical analysis figure
        comprehensive_path = viz.plot_comprehensive_stats_figure(
            result.metrics,
            result.statistical_comparisons,
        )
        print(f"Comprehensive stats figure: {comprehensive_path}")

        # LaTeX table
        latex_table = create_summary_table(result.metrics)
        latex_path = config.output_dir / "table_classical_baselines.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {latex_path}")

    print("\nPhase 1 complete!")
    return result


if __name__ == "__main__":
    main()
