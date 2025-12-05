"""
Hyperparameter Search Spaces for Track B: SpectralShift Experiments
===================================================================

Defines search spaces for each spectral correction approach in Track B.
"""

from typing import Any, Dict

# =============================================================================
# Common Hyperparameters (SpectralShift modules only - U-Net is frozen)
# =============================================================================

COMMON_SPECTRAL_HP_SPACE = {
    # Learning rate for spectral module only
    "spectral_lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    "spectral_lr_decay": {"type": "float", "low": 0.9, "high": 0.99},
}

# =============================================================================
# Baseline: Current AdaptiveSpectralShift
# =============================================================================

BASELINE_SPECTRAL_HP_SPACE = {
    **COMMON_SPECTRAL_HP_SPACE,
    "spectral_method": {"type": "fixed", "value": "baseline"},

    # Current adaptive spectral shift parameters
    "spectral_shift_max_db": {"type": "float", "low": 6.0, "high": 15.0},
    "spectral_shift_hidden_dim": {"type": "categorical", "choices": [64, 128, 256]},
    "spectral_shift_per_channel": {"type": "categorical", "choices": [True, False]},
}

# =============================================================================
# Phase-Preserving Soft Correction
# =============================================================================

PHASE_PRESERVING_HP_SPACE = {
    **COMMON_SPECTRAL_HP_SPACE,
    "spectral_method": {"type": "fixed", "value": "phase_preserving"},

    # Phase-preserving-specific parameters
    "target_bands": {
        "type": "categorical",
        "choices": [["beta"], ["gamma"], ["beta", "gamma"], ["alpha", "beta", "gamma"]],
    },
    "max_correction_db": {"type": "float", "low": 3.0, "high": 12.0},
    "softness": {"type": "float", "low": 0.2, "high": 0.8},
    "per_channel": {"type": "categorical", "choices": [True, False]},
    "transition_width_hz": {"type": "float", "low": 1.0, "high": 4.0},
}

# =============================================================================
# Adaptive Gated Spectral Correction
# =============================================================================

ADAPTIVE_GATED_HP_SPACE = {
    **COMMON_SPECTRAL_HP_SPACE,
    "spectral_method": {"type": "fixed", "value": "adaptive_gated"},

    # Adaptive gated-specific parameters
    "gating_hidden_dim": {"type": "categorical", "choices": [32, 64, 128]},
    "n_bands": {"type": "categorical", "choices": [5, 8, 10, 16]},
    "gate_temperature": {"type": "float", "low": 0.5, "high": 2.0},
    "max_correction_db": {"type": "float", "low": 6.0, "high": 12.0},
    "use_target_guidance": {"type": "categorical", "choices": [True, False]},
}

# =============================================================================
# Wavelet-Based Local Spectral Correction
# =============================================================================

WAVELET_CORRECTION_HP_SPACE = {
    **COMMON_SPECTRAL_HP_SPACE,
    "spectral_method": {"type": "fixed", "value": "wavelet"},

    # Wavelet-specific parameters
    "wavelet_omega0": {"type": "categorical", "choices": [3.0, 5.0, 7.0]},
    "n_scales": {"type": "categorical", "choices": [8, 16, 24]},
    "correction_kernel_size": {"type": "categorical", "choices": [1, 3, 5]},
    "n_correction_layers": {"type": "categorical", "choices": [1, 2, 3]},
}

# =============================================================================
# Spectral Loss Instead of Block
# =============================================================================

SPECTRAL_LOSS_HP_SPACE = {
    **COMMON_SPECTRAL_HP_SPACE,
    "spectral_method": {"type": "fixed", "value": "spectral_loss"},

    # Spectral loss-specific parameters (affects training, not inference)
    "spectral_scales": {
        "type": "categorical",
        "choices": [[512, 1024], [256, 512, 1024], [256, 512, 1024, 2048]],
    },
    "use_log_psd": {"type": "categorical", "choices": [True, False]},
    "delta_weight": {"type": "float", "low": 0.5, "high": 2.0},
    "theta_weight": {"type": "float", "low": 0.5, "high": 2.0},
    "alpha_weight": {"type": "float", "low": 0.5, "high": 2.0},
    "beta_weight": {"type": "float", "low": 1.0, "high": 4.0},
    "gamma_weight": {"type": "float", "low": 1.0, "high": 5.0},
    "spectral_loss_weight": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
}

# =============================================================================
# Iterative Refinement with R² Feedback
# =============================================================================

ITERATIVE_REFINEMENT_HP_SPACE = {
    **COMMON_SPECTRAL_HP_SPACE,
    "spectral_method": {"type": "fixed", "value": "iterative"},

    # Iterative refinement-specific parameters
    "n_iterations": {"type": "categorical", "choices": [3, 5, 7, 10]},
    "step_size": {"type": "float", "low": 0.05, "high": 0.2},
    "r2_tolerance": {"type": "float", "low": 0.005, "high": 0.02},
    "use_momentum": {"type": "categorical", "choices": [True, False]},
    "n_bands": {"type": "categorical", "choices": [4, 8, 12]},
    "use_adaptive": {"type": "categorical", "choices": [True, False]},
}

# =============================================================================
# Master Dictionary
# =============================================================================

TRACK_B_HP_SPACES: Dict[str, Dict[str, Any]] = {
    "baseline": BASELINE_SPECTRAL_HP_SPACE,
    "phase_preserving": PHASE_PRESERVING_HP_SPACE,
    "adaptive_gated": ADAPTIVE_GATED_HP_SPACE,
    "wavelet": WAVELET_CORRECTION_HP_SPACE,
    "spectral_loss": SPECTRAL_LOSS_HP_SPACE,
    "iterative": ITERATIVE_REFINEMENT_HP_SPACE,
}

# Approach names for iteration
TRACK_B_APPROACHES = list(TRACK_B_HP_SPACES.keys())


def get_hp_space(approach: str) -> Dict[str, Any]:
    """Get hyperparameter space for an approach.

    Args:
        approach: Approach name

    Returns:
        Search space dictionary
    """
    if approach not in TRACK_B_HP_SPACES:
        raise ValueError(f"Unknown approach: {approach}. Valid: {TRACK_B_APPROACHES}")
    return TRACK_B_HP_SPACES[approach]
