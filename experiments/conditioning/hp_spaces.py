"""
Hyperparameter Search Spaces for Track A: Auto-Conditioning Experiments
========================================================================

Defines search spaces for each conditioning approach in Track A.
"""

from typing import Any, Dict

# =============================================================================
# Common Hyperparameters (shared across all approaches)
# =============================================================================

COMMON_HP_SPACE = {
    "learning_rate": {"type": "float", "low": 5e-5, "high": 5e-3, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    "dropout": {"type": "float", "low": 0.0, "high": 0.3},
    "n_downsample": {"type": "categorical", "choices": [3, 4]},
    "weight_l1": {"type": "float", "low": 0.5, "high": 5.0, "log": True},
    "weight_wavelet": {"type": "float", "low": 0.5, "high": 5.0, "log": True},
}

# =============================================================================
# Baseline: Current cross_attn_gated + odor embedding
# =============================================================================

BASELINE_HP_SPACE = {
    **COMMON_HP_SPACE,
    "conditioning_type": {"type": "fixed", "value": "baseline"},
}

# =============================================================================
# CPC Embeddings
# =============================================================================

CPC_HP_SPACE = {
    **COMMON_HP_SPACE,
    "conditioning_type": {"type": "fixed", "value": "cpc"},

    # CPC-specific parameters
    "cpc_hidden_dim": {"type": "categorical", "choices": [128, 256, 512]},
    "cpc_context_dim": {"type": "categorical", "choices": [64, 128, 256]},
    "cpc_n_steps_ahead": {"type": "categorical", "choices": [4, 8, 12]},
    "cpc_loss_weight": {"type": "float", "low": 0.01, "high": 1.0, "log": True},
    "cpc_use_odor_concat": {"type": "categorical", "choices": [True, False]},
}

# =============================================================================
# VQ-VAE Discrete Codes
# =============================================================================

VQVAE_HP_SPACE = {
    **COMMON_HP_SPACE,
    "conditioning_type": {"type": "fixed", "value": "vqvae"},

    # VQ-VAE-specific parameters
    "vq_codebook_size": {"type": "categorical", "choices": [128, 256, 512, 1024]},
    "vq_code_dim": {"type": "categorical", "choices": [32, 64, 128]},
    "vq_commitment_cost": {"type": "float", "low": 0.1, "high": 0.5},
    "vq_loss_weight": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
    "vq_use_odor_residual": {"type": "categorical", "choices": [True, False]},
}

# =============================================================================
# Frequency-Aware Disentangled Conditioning
# =============================================================================

FREQ_DISENTANGLED_HP_SPACE = {
    **COMMON_HP_SPACE,
    "conditioning_type": {"type": "fixed", "value": "freq_disentangled"},

    # Frequency-disentangled-specific parameters
    "low_freq_cutoff": {"type": "categorical", "choices": [4.0, 8.0, 12.0]},
    "high_freq_cutoff": {"type": "categorical", "choices": [20.0, 30.0, 40.0]},
    "stream_balance": {"type": "float", "low": 0.3, "high": 0.7},
    "use_cross_stream_attention": {"type": "categorical", "choices": [True, False]},
}

# =============================================================================
# Cycle-Consistent Latent Discovery
# =============================================================================

CYCLE_LATENT_HP_SPACE = {
    **COMMON_HP_SPACE,
    "conditioning_type": {"type": "fixed", "value": "cycle_latent"},

    # Cycle-latent-specific parameters
    "latent_dim": {"type": "categorical", "choices": [32, 64, 128]},
    "cycle_loss_weight": {"type": "float", "low": 0.1, "high": 1.0},
    "encoder_depth": {"type": "categorical", "choices": [2, 3, 4]},
}

# =============================================================================
# Variational Information Bottleneck (VIB)
# =============================================================================

VIB_HP_SPACE = {
    **COMMON_HP_SPACE,
    "conditioning_type": {"type": "fixed", "value": "vib"},

    # VIB-specific parameters
    "vib_latent_dim": {"type": "categorical", "choices": [16, 32, 64, 128]},
    "vib_beta": {"type": "float", "low": 0.001, "high": 1.0, "log": True},
    "vib_encoder_type": {"type": "categorical", "choices": ["conv", "transformer"]},
}

# =============================================================================
# Master Dictionary
# =============================================================================

TRACK_A_HP_SPACES: Dict[str, Dict[str, Any]] = {
    "baseline": BASELINE_HP_SPACE,
    "cpc": CPC_HP_SPACE,
    "vqvae": VQVAE_HP_SPACE,
    "freq_disentangled": FREQ_DISENTANGLED_HP_SPACE,
    "cycle_latent": CYCLE_LATENT_HP_SPACE,
    "vib": VIB_HP_SPACE,
}

# Approach names for iteration
TRACK_A_APPROACHES = list(TRACK_A_HP_SPACES.keys())


def get_hp_space(approach: str) -> Dict[str, Any]:
    """Get hyperparameter space for an approach.

    Args:
        approach: Approach name

    Returns:
        Search space dictionary
    """
    if approach not in TRACK_A_HP_SPACES:
        raise ValueError(f"Unknown approach: {approach}. Valid: {TRACK_A_APPROACHES}")
    return TRACK_A_HP_SPACES[approach]
