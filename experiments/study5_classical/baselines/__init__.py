"""
Classical Baselines for Study 5
===============================

Classical signal processing baselines for comparison:
- Wiener Filter: Optimal linear filter in frequency domain
- Ridge Regression: L2-regularized linear mapping
- Kalman Filter: State-space temporal model
"""

from __future__ import annotations

from .wiener_filter import (
    WienerFilter,
    MultiChannelWienerFilter,
    create_wiener_filter,
)
from .ridge_regression import (
    RidgeRegression,
    TemporalRidgeRegression,
    MultiOutputRidgeCV,
    create_ridge_regression,
)
from .kalman_filter import (
    KalmanFilter,
    ExtendedKalmanFilter,
    create_kalman_filter,
)

__all__ = [
    # Wiener filter
    "WienerFilter",
    "MultiChannelWienerFilter",
    "create_wiener_filter",
    # Ridge regression
    "RidgeRegression",
    "TemporalRidgeRegression",
    "MultiOutputRidgeCV",
    "create_ridge_regression",
    # Kalman filter
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "create_kalman_filter",
    # Registry
    "BASELINE_REGISTRY",
    "create_baseline",
]

# Baseline registry for easy access
BASELINE_REGISTRY = {
    "wiener": create_wiener_filter,
    "wiener_mimo": lambda **kw: create_wiener_filter("mimo", **kw),
    "ridge": create_ridge_regression,
    "ridge_temporal": lambda **kw: create_ridge_regression("temporal", **kw),
    "ridge_cv": lambda **kw: create_ridge_regression("cv", **kw),
    "kalman": create_kalman_filter,
    "kalman_extended": lambda **kw: create_kalman_filter("extended", **kw),
}


def create_baseline(name: str, **kwargs):
    """Create classical baseline by name.

    Args:
        name: Baseline name (see BASELINE_REGISTRY keys)
        **kwargs: Additional arguments

    Returns:
        Baseline instance
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINE_REGISTRY.keys())}")

    return BASELINE_REGISTRY[name](**kwargs)
