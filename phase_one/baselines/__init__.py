"""
Classical Baselines for Phase 1
===============================

Implements 7 classical signal processing methods for neural signal translation:

1. Wiener Filter - Optimal linear filter in frequency domain
2. Wiener MIMO - Multi-input multi-output Wiener filter
3. Ridge Regression - L2-regularized linear mapping (pointwise)
4. Ridge Temporal - Ridge with temporal context windows
5. Ridge CV - Ridge with cross-validated regularization
6. VAR - Vector Autoregressive model
7. VAR Exogenous - VAR with exogenous inputs (VARX)

All methods follow sklearn-like API: fit(X, y) -> predict(X) -> y_pred
"""

from .wiener import WienerFilter, WienerMIMO
from .ridge import RidgeBaseline, RidgeTemporal, RidgeCV
from .var import VARBaseline, VARExogenous
from .registry import BASELINE_REGISTRY, create_baseline, list_baselines

__all__ = [
    # Wiener variants
    "WienerFilter",
    "WienerMIMO",
    # Ridge variants
    "RidgeBaseline",
    "RidgeTemporal",
    "RidgeCV",
    # VAR variants
    "VARBaseline",
    "VARExogenous",
    # Registry
    "BASELINE_REGISTRY",
    "create_baseline",
    "list_baselines",
]
