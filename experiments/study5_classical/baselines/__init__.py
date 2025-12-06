"""
Classical Baselines for Study 5
===============================

Classical signal processing baselines for comparison:
- Wiener Filter: Optimal linear filter in frequency domain
- Ridge Regression: L2-regularized linear mapping
- Kalman Filter: State-space temporal model
- VAR: Vector Autoregressive model
- SVR: Support Vector Regression
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
# Kalman filter removed - not suitable for this signal translation task
# (produces garbage R² values due to state-space model mismatch)
from .var_model import (
    VARModel,
    VARExogenous,
    create_var_model,
)
# SVR removed - O(n²) complexity unsuitable for high-dimensional neural data (32x5000)

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
    # Kalman filter - REMOVED (not suitable for this task)
    # VAR
    "VARModel",
    "VARExogenous",
    "create_var_model",
    # SVR - REMOVED (O(n²) complexity unsuitable for 32x5000 data)
    # Registry
    "BASELINE_REGISTRY",
    "create_baseline",
]

# Baseline registry for easy access
BASELINE_REGISTRY = {
    # Wiener filter variants
    "wiener": create_wiener_filter,
    "wiener_mimo": lambda **kw: create_wiener_filter("mimo", **kw),
    # Ridge regression variants
    "ridge": create_ridge_regression,
    "ridge_temporal": lambda **kw: create_ridge_regression("temporal", **kw),
    "ridge_cv": lambda **kw: create_ridge_regression("cv", **kw),
    # Kalman filter - REMOVED (not suitable for signal translation)
    # VAR variants
    "var": create_var_model,
    "var_exogenous": lambda **kw: create_var_model("exogenous", **kw),
    # SVR - REMOVED (O(n²) complexity, unsuitable for high-dimensional data)
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
