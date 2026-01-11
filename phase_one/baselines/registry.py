"""
Baseline Registry for Phase 1
=============================

Central registry for creating and accessing classical baseline models.
Provides factory functions and documentation for all available methods.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Union

from .wiener import WienerFilter, WienerMIMO
from .ridge import RidgeBaseline, RidgeTemporal, RidgeCV
from .var import VARBaseline, VARExogenous


# Type alias for baseline models
BaselineModel = Union[WienerFilter, WienerMIMO, RidgeBaseline, RidgeTemporal, RidgeCV, VARBaseline, VARExogenous]


# =============================================================================
# Baseline Registry
# =============================================================================

BASELINE_REGISTRY: Dict[str, Callable[..., BaselineModel]] = {
    # Wiener filter variants
    "wiener": lambda **kw: WienerFilter(**kw),
    "wiener_mimo": lambda **kw: WienerMIMO(**kw),

    # Ridge regression variants
    "ridge": lambda **kw: RidgeBaseline(**kw),
    "ridge_temporal": lambda **kw: RidgeTemporal(**kw),
    "ridge_cv": lambda **kw: RidgeCV(**kw),

    # VAR variants
    "var": lambda **kw: VARBaseline(**kw),
    "var_exogenous": lambda **kw: VARExogenous(**kw),
}


# =============================================================================
# Baseline Descriptions (for documentation and reporting)
# =============================================================================

BASELINE_DESCRIPTIONS: Dict[str, str] = {
    "wiener": "Single-channel Wiener filter (optimal linear filter in frequency domain)",
    "wiener_mimo": "Multi-input multi-output Wiener filter (joint channel optimization)",
    "ridge": "L2-regularized linear regression (pointwise mapping)",
    "ridge_temporal": "Ridge regression with temporal context windows",
    "ridge_cv": "Ridge regression with cross-validated regularization",
    "var": "Vector Autoregressive model (temporal dynamics)",
    "var_exogenous": "VAR with exogenous inputs (VARX model)",
}


# =============================================================================
# Default Hyperparameters
# =============================================================================

DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "wiener": {
        "regularization": 1e-6,
        "smooth_window": 5,
    },
    "wiener_mimo": {
        "regularization": 1e-5,
        "smooth_window": 5,
    },
    "ridge": {
        "alpha": 1.0,
        "fit_intercept": True,
    },
    "ridge_temporal": {
        "alpha": 1.0,
        "window_size": 10,
        "stride": 1,
    },
    "ridge_cv": {
        "alphas": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "cv": 5,
    },
    "var": {
        "order": 5,
        "regularization": 1e-4,
    },
    "var_exogenous": {
        "order": 5,
        "input_order": 3,
        "regularization": 1e-4,
    },
}


# =============================================================================
# Factory Functions
# =============================================================================

def create_baseline(name: str, **kwargs) -> BaselineModel:
    """Create a baseline model by name.

    Args:
        name: Baseline name (see list_baselines() for options)
        **kwargs: Override default hyperparameters

    Returns:
        Instantiated baseline model

    Raises:
        ValueError: If baseline name is unknown

    Example:
        >>> model = create_baseline("ridge", alpha=0.5)
        >>> model = create_baseline("wiener_mimo")
    """
    if name not in BASELINE_REGISTRY:
        available = list(BASELINE_REGISTRY.keys())
        raise ValueError(f"Unknown baseline: '{name}'. Available: {available}")

    # Merge default hyperparameters with provided kwargs
    params = DEFAULT_HYPERPARAMS.get(name, {}).copy()
    params.update(kwargs)

    return BASELINE_REGISTRY[name](**params)


def list_baselines() -> List[str]:
    """List all available baseline methods.

    Returns:
        List of baseline names
    """
    return list(BASELINE_REGISTRY.keys())


def get_baseline_info(name: str) -> Dict[str, Any]:
    """Get information about a baseline method.

    Args:
        name: Baseline name

    Returns:
        Dictionary with description, default parameters, etc.

    Raises:
        ValueError: If baseline name is unknown
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: '{name}'")

    return {
        "name": name,
        "description": BASELINE_DESCRIPTIONS.get(name, "No description available"),
        "default_params": DEFAULT_HYPERPARAMS.get(name, {}),
    }


def print_baselines_summary() -> None:
    """Print a formatted summary of all available baselines."""
    print("\n" + "=" * 70)
    print("PHASE 1: Classical Baselines for Neural Signal Translation")
    print("=" * 70)

    print("\n{:<18} {:<52}".format("Method", "Description"))
    print("-" * 70)

    for name in BASELINE_REGISTRY.keys():
        desc = BASELINE_DESCRIPTIONS.get(name, "")
        # Truncate description if too long
        if len(desc) > 50:
            desc = desc[:47] + "..."
        print(f"{name:<18} {desc:<52}")

    print("-" * 70)
    print(f"Total: {len(BASELINE_REGISTRY)} methods")
    print()
