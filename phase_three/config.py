"""
Configuration for Phase 3: CondUNet Ablation Studies
=====================================================

Supports three ablation protocols:
1. SUBTRACTIVE (traditional): Start with full model, remove components
2. ADDITIVE (build-up): Start with simple baseline, add components incrementally
3. GREEDY_FORWARD (recommended): Test groups sequentially, winner becomes default

The GREEDY_FORWARD protocol is recommended based on literature review:
- Sequential Forward Selection (SFS) - well-established in ML
- Coordinate-wise optimization - supported by JMLR and NeurIPS papers
- nnU-Net (Nature Methods): "one split of training data" for ablation
- Efficient: 33 runs vs 5^11 = 48M for grid search

Literature support:
- nnU-Net (Nature Methods): Single split for ablation, full CV for final
- Binary Coordinate Ascent (ScienceDirect): 5-37x efficiency gain
- JMLR: Greedy block selection (Gauss-Southwell rule)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch

# =============================================================================
# Ablation Protocol Selection
# =============================================================================

ABLATION_PROTOCOLS = ["additive", "subtractive", "greedy_forward", "stage2_greedy"]

# =============================================================================
# ADDITIVE PROTOCOL: Incremental Component Analysis
# =============================================================================
# Start with simple baseline, add one component at a time
# Based on: nnU-Net (Nature Methods), Transformer paper methodology

# Ordered list of components to add incrementally
# Each stage adds ONE component on top of previous stage
INCREMENTAL_STAGES: List[Dict[str, Any]] = [
    {
        "stage": 0,
        "name": "simple_baseline",
        "description": "Plain U-Net: standard conv, no attention, no conditioning, L1 loss",
        "config": {
            "conv_type": "standard",
            "attention_type": "none",
            "cond_mode": "none",
            "use_odor_embedding": False,
            "loss_type": "l1",
            "use_augmentation": False,
            "bidirectional": False,
            "base_channels": 64,
            # Cross-session evaluation (main result)
            "split_by_session": True,
            "n_test_sessions": 4,
            "n_val_sessions": 1,
        },
    },
    {
        "stage": 1,
        "name": "modern_conv",
        "description": "+ Modern convolutions (dilated depthwise separable + SE)",
        "adds": "conv_type",
        "config": {
            "conv_type": "modern",
            "attention_type": "none",
            "cond_mode": "none",
            "use_odor_embedding": False,
            "loss_type": "l1",
            "use_augmentation": False,
            "bidirectional": False,
            "base_channels": 64,
            # Cross-session evaluation (main result)
            "split_by_session": True,
            "n_test_sessions": 4,
            "n_val_sessions": 1,
        },
    },
    {
        "stage": 2,
        "name": "attention",
        "description": "+ Cross-frequency attention mechanism",
        "adds": "attention_type",
        "config": {
            "conv_type": "modern",
            "attention_type": "cross_freq_v2",
            "cond_mode": "none",
            "use_odor_embedding": False,
            "loss_type": "l1",
            "use_augmentation": False,
            "bidirectional": False,
            "base_channels": 64,
            # Cross-session evaluation (main result)
            "split_by_session": True,
            "n_test_sessions": 4,
            "n_val_sessions": 1,
        },
    },
    {
        "stage": 3,
        "name": "conditioning",
        "description": "+ Spectro-temporal auto-conditioning",
        "adds": "cond_mode",
        "config": {
            "conv_type": "modern",
            "attention_type": "cross_freq_v2",
            "cond_mode": "cross_attn_gated",
            "use_odor_embedding": True,
            "loss_type": "l1",
            "use_augmentation": False,
            "bidirectional": False,
            "base_channels": 64,
            # Cross-session evaluation (main result)
            "split_by_session": True,
            "n_test_sessions": 4,
            "n_val_sessions": 1,
        },
    },
    {
        "stage": 4,
        "name": "wavelet_loss",
        "description": "+ Wavelet multi-scale loss",
        "adds": "loss_type",
        "config": {
            "conv_type": "modern",
            "attention_type": "cross_freq_v2",
            "cond_mode": "cross_attn_gated",
            "use_odor_embedding": True,
            "loss_type": "l1_wavelet",
            "use_augmentation": False,
            "bidirectional": False,
            "base_channels": 64,
            # Cross-session evaluation (main result)
            "split_by_session": True,
            "n_test_sessions": 4,
            "n_val_sessions": 1,
        },
    },
    {
        "stage": 5,
        "name": "augmentation",
        "description": "+ Data augmentation suite",
        "adds": "use_augmentation",
        "config": {
            "conv_type": "modern",
            "attention_type": "cross_freq_v2",
            "cond_mode": "cross_attn_gated",
            "use_odor_embedding": True,
            "loss_type": "l1_wavelet",
            "use_augmentation": True,
            "bidirectional": False,
            "base_channels": 64,
            # Cross-session evaluation (main result)
            "split_by_session": True,
            "n_test_sessions": 4,
            "n_val_sessions": 1,
        },
    },
    {
        "stage": 6,
        "name": "bidirectional",
        "description": "+ Bidirectional training (cycle consistency)",
        "adds": "bidirectional",
        "config": {
            "conv_type": "modern",
            "attention_type": "cross_freq_v2",
            "cond_mode": "cross_attn_gated",
            "use_odor_embedding": True,
            "loss_type": "l1_wavelet",
            "use_augmentation": True,
            "bidirectional": True,
            "base_channels": 64,
            # Cross-session evaluation (main result)
            "split_by_session": True,
            "n_test_sessions": 4,
            "n_val_sessions": 1,
        },
    },
]

# Component order for reference (matches stage order)
COMPONENT_ORDER = [
    "conv_type",        # Stage 1: standard -> modern
    "attention_type",   # Stage 2: none -> cross_freq_v2
    "cond_mode",        # Stage 3: none -> cross_attn_gated
    "loss_type",        # Stage 4: l1 -> l1_wavelet
    "use_augmentation", # Stage 5: False -> True
    "bidirectional",    # Stage 6: False -> True
]

# =============================================================================
# GREEDY FORWARD SELECTION: Coordinate-wise Optimization (RECOMMENDED)
# =============================================================================
# Test variants in each group sequentially. Pick winner (best R²).
# Winner becomes default for subsequent groups.
#
# Based on:
# - nnU-Net (Nature Methods): "one split of training data" for ablation
# - Binary Coordinate Ascent: Sequential forward selection methodology
# - JMLR: Greedy block selection (Gauss-Southwell rule)
#
# Total runs: 33 (vs 48M for full grid search)
# Single split: 8 sessions train, 4 sessions test (no CV during ablation)

# Default configuration (simple baseline to start)
GREEDY_DEFAULTS: Dict[str, Any] = {
    "conv_type": "standard",
    "n_downsample": 2,
    "base_channels": 64,
    "attention_type": "none",
    "n_heads": 4,
    "cond_mode": "none",
    "use_odor_embedding": False,
    "loss_type": "l1",
    "use_augmentation": False,
    "aug_strength": "none",
    "bidirectional": False,
    "cycle_lambda": 0.5,
    # New architectural choices
    "norm_type": "batch",           # Normalization type
    "skip_type": "add",             # Skip connection type
    "activation": "relu",           # Activation function
    "dropout": 0.0,                 # Dropout rate
    # New training choices
    "optimizer": "adamw",           # Optimizer
    "lr_schedule": "cosine",        # Learning rate schedule
    "weight_decay": 1e-4,           # Weight decay
    # Session adaptation approaches (Group 18)
    # Multiple methods based on domain adaptation literature:
    "use_session_stats": False,     # Statistics-based conditioning (FiLM style)
    "session_emb_dim": 32,          # Session statistics embedding dimension
    "session_use_spectral": False,  # Include spectral features in session stats
    "use_adaptive_scaling": False,  # Session-adaptive output scaling (AdaIN style)
    "use_cov_augment": False,       # Covariance expansion augmentation
    "cov_augment_prob": 0.5,        # Probability of applying cov augmentation
    "use_session_embedding": False, # Learnable session embedding (lookup table → FiLM)
    "use_adabn": False,             # AdaBN: use batch stats at inference (requires norm_type="batch")
}

# =============================================================================
# Ablation groups - ORDERED BY IMPORTANCE for greedy forward selection
# Most critical architectural choices FIRST, then training refinements
# =============================================================================
ABLATION_GROUPS: List[Dict[str, Any]] = [
    # =========================================================================
    # PHASE 1: CRITICAL ARCHITECTURAL FOUNDATIONS (Groups 1-6)
    # These affect EVERYTHING - test first to establish strong baseline
    # =========================================================================
    # GROUP 1: Normalization Type (THE most critical component!)
    # Literature: BatchNorm (2015), LayerNorm (2016), GroupNorm (2018), RMSNorm (2019)
    {
        "group_id": 1,
        "name": "normalization",
        "description": "Normalization layer type - critical for training stability",
        "parameter": "norm_type",
        "variants": [
            {"value": "batch", "name": "batch_norm", "desc": "Batch Normalization - standard, batch-dependent"},
            {"value": "layer", "name": "layer_norm", "desc": "Layer Normalization - batch-independent, good for attention"},
            {"value": "instance", "name": "instance_norm", "desc": "Instance Normalization - per-sample, style transfer origin"},
            {"value": "group", "name": "group_norm", "desc": "Group Normalization - compromise, groups of channels"},
            {"value": "rms", "name": "rms_norm", "desc": "RMS Normalization - simpler, used in LLaMA/modern transformers"},
            {"value": "none", "name": "no_norm", "desc": "No normalization - baseline"},
        ],
        "conditional_on": None,
    },
    # GROUP 2: Activation Function
    # Literature: ReLU (2010), GELU (2016), SiLU/Swish (2017), Mish (2019)
    {
        "group_id": 7,
        "name": "activation",
        "description": "Activation function - affects gradient flow and expressivity",
        "parameter": "activation",
        "variants": [
            {"value": "relu", "name": "relu", "desc": "ReLU - simple, sparse activations"},
            {"value": "leaky_relu", "name": "leaky_relu", "desc": "Leaky ReLU - no dead neurons"},
            {"value": "gelu", "name": "gelu", "desc": "GELU - smooth, used in BERT/GPT/ViT"},
            {"value": "silu", "name": "silu_swish", "desc": "SiLU/Swish - self-gated, used in EfficientNet"},
            {"value": "mish", "name": "mish", "desc": "Mish - smooth, non-monotonic"},
        ],
        "conditional_on": None,
    },
    # GROUP 3: Convolution Type
    {
        "group_id": 3,
        "name": "conv_type",
        "description": "Convolution architecture",
        "parameter": "conv_type",
        "variants": [
            {"value": "standard", "name": "standard_conv", "desc": "Standard convolutions"},
            {"value": "modern", "name": "modern_conv", "desc": "Dilated depthwise-separable + SE"},
        ],
        "conditional_on": None,
    },
    # GROUP 4: Skip Connection Type (core U-Net design)
    # Literature: ResNet (2015), DenseNet (2017), Attention U-Net (2018)
    {
        "group_id": 6,
        "name": "skip_connection",
        "description": "Skip connection type in U-Net architecture",
        "parameter": "skip_type",
        "variants": [
            {"value": "add", "name": "residual_add", "desc": "Additive residual (ResNet style)"},
            {"value": "concat", "name": "concat", "desc": "Concatenation (original U-Net)"},
            {"value": "attention", "name": "attention_gate", "desc": "Attention-gated skip (Attention U-Net)"},
            {"value": "dense", "name": "dense", "desc": "Dense connections (all previous layers)"},
        ],
        "conditional_on": None,
    },
    # GROUP 5: Network Depth
    {
        "group_id": 4,
        "name": "depth",
        "description": "Encoder/decoder depth",
        "parameter": "n_downsample",
        "variants": [
            {"value": 2, "name": "shallow", "desc": "2 downsample levels"},
            {"value": 3, "name": "medium", "desc": "3 downsample levels"},
            {"value": 4, "name": "deep", "desc": "4 downsample levels"},
        ],
        "conditional_on": None,
    },
    # GROUP 6: Network Width
    {
        "group_id": 5,
        "name": "width",
        "description": "Base channel count",
        "parameter": "base_channels",
        "variants": [
            {"value": 32, "name": "narrow", "desc": "32 base channels"},
            {"value": 64, "name": "medium", "desc": "64 base channels"},
            {"value": 128, "name": "wide", "desc": "128 base channels"},
        ],
        "conditional_on": None,
    },
    # =========================================================================
    # PHASE 2: ATTENTION & CONDITIONING (Groups 7-9)
    # Optional enhancements to the base architecture
    # =========================================================================
    # GROUP 7: Attention Type
    {
        "group_id": 8,
        "name": "attention",
        "description": "Attention mechanism",
        "parameter": "attention_type",
        "variants": [
            {"value": "none", "name": "no_attention", "desc": "No attention"},
            {"value": "basic", "name": "basic_attention", "desc": "Basic self-attention"},
            {"value": "cross_freq_v2", "name": "cross_freq", "desc": "Cross-frequency attention v2"},
        ],
        "conditional_on": None,
    },
    # GROUP 8: Attention Heads (conditional on attention != none)
    {
        "group_id": 9,
        "name": "attention_heads",
        "description": "Number of attention heads",
        "parameter": "n_heads",
        "variants": [
            {"value": 2, "name": "2_heads", "desc": "2 attention heads"},
            {"value": 4, "name": "4_heads", "desc": "4 attention heads"},
            {"value": 8, "name": "8_heads", "desc": "8 attention heads"},
        ],
        "conditional_on": None,  # Run unconditionally - greedy will handle appropriately
    },
    # GROUP 9: Conditioning Mode
    # NOTE: train.py only supports 'none' and 'cross_attn_gated' currently
    {
        "group_id": 10,
        "name": "conditioning",
        "description": "Conditioning mechanism",
        "parameter": "cond_mode",
        "variants": [
            {"value": "none", "name": "no_conditioning", "desc": "No conditioning"},
            {"value": "cross_attn_gated", "name": "gated", "desc": "Gated cross-attention conditioning"},
        ],
        "conditional_on": None,
    },
    # =========================================================================
    # PHASE 3: LOSS & REGULARIZATION (Groups 10-13)
    # How we train and prevent overfitting
    # =========================================================================
    # GROUP 10: Loss Function
    {
        "group_id": 12,
        "name": "loss",
        "description": "Training loss function",
        "parameter": "loss_type",
        "variants": [
            {"value": "l1", "name": "l1", "desc": "L1 loss"},
            {"value": "huber", "name": "huber", "desc": "Huber loss"},
            {"value": "l1_wavelet", "name": "l1_wavelet", "desc": "L1 + wavelet multi-scale"},
            {"value": "huber_wavelet", "name": "huber_wavelet", "desc": "Huber + wavelet multi-scale"},
        ],
        "conditional_on": None,
    },
    # GROUP 11: Data Augmentation
    {
        "group_id": 2,
        "name": "augmentation",
        "description": "Data augmentation strategy",
        "parameter": "use_augmentation",
        "variants": [
            {"value": False, "name": "no_aug", "desc": "No augmentation", "aug_strength": "none"},
            {"value": True, "name": "light_aug", "desc": "Light augmentation", "aug_strength": "light"},
            {"value": True, "name": "medium_aug", "desc": "Medium augmentation", "aug_strength": "medium"},
            {"value": True, "name": "heavy_aug", "desc": "Heavy augmentation", "aug_strength": "heavy"},
        ],
        "conditional_on": None,
    },
    # GROUP 12: Dropout Strategy
    # Literature: Dropout (2014), Spatial Dropout (2015), DropBlock (2018)
    {
        "group_id": 15,
        "name": "dropout",
        "description": "Dropout regularization strategy",
        "parameter": "dropout",
        "variants": [
            {"value": 0.0, "name": "no_dropout", "desc": "No dropout"},
            {"value": 0.1, "name": "light_dropout", "desc": "Light dropout (p=0.1)"},
            {"value": 0.2, "name": "medium_dropout", "desc": "Medium dropout (p=0.2)"},
            {"value": 0.3, "name": "heavy_dropout", "desc": "Heavy dropout (p=0.3)"},
            {"value": 0.5, "name": "aggressive_dropout", "desc": "Aggressive dropout (p=0.5)"},
        ],
        "conditional_on": None,
    },
    # GROUP 13: Weight Decay
    # Literature: L2 regularization, decoupled weight decay (AdamW paper)
    {
        "group_id": 16,
        "name": "weight_decay",
        "description": "Weight decay (L2 regularization) strength",
        "parameter": "weight_decay",
        "variants": [
            {"value": 0.0, "name": "no_wd", "desc": "No weight decay"},
            {"value": 1e-5, "name": "very_light_wd", "desc": "Very light (1e-5)"},
            {"value": 1e-4, "name": "light_wd", "desc": "Light (1e-4) - common default"},
            {"value": 1e-3, "name": "medium_wd", "desc": "Medium (1e-3)"},
            {"value": 1e-2, "name": "heavy_wd", "desc": "Heavy (1e-2) - strong regularization"},
        ],
        "conditional_on": None,
    },
    # =========================================================================
    # PHASE 4: TRAINING DYNAMICS (Groups 14-17)
    # Optimizer and learning rate strategies
    # =========================================================================
    # GROUP 14: Optimizer
    # Literature: Adam (2014), AdamW (2017), Lion (2023), Shampoo (2015)
    {
        "group_id": 13,
        "name": "optimizer",
        "description": "Optimization algorithm",
        "parameter": "optimizer",
        "variants": [
            {"value": "adamw", "name": "adamw", "desc": "AdamW - decoupled weight decay, standard"},
            {"value": "adam", "name": "adam", "desc": "Adam - original adaptive moments"},
            {"value": "sgd", "name": "sgd_momentum", "desc": "SGD + Momentum - classic, often better generalization"},
            {"value": "lion", "name": "lion", "desc": "Lion - Google 2023, memory efficient"},
            {"value": "adafactor", "name": "adafactor", "desc": "Adafactor - memory efficient, used in T5"},
        ],
        "conditional_on": None,
    },
    # GROUP 15: Learning Rate Schedule
    # Literature: Step decay, Cosine annealing (2016), Warmup (2017), OneCycle (2018)
    {
        "group_id": 14,
        "name": "lr_schedule",
        "description": "Learning rate scheduling strategy",
        "parameter": "lr_schedule",
        "variants": [
            {"value": "cosine", "name": "cosine", "desc": "Cosine annealing - smooth decay"},
            {"value": "cosine_warmup", "name": "cosine_warmup", "desc": "Cosine with linear warmup"},
            {"value": "step", "name": "step_decay", "desc": "Step decay - reduce at milestones"},
            {"value": "plateau", "name": "reduce_on_plateau", "desc": "Reduce on plateau - adaptive"},
            {"value": "onecycle", "name": "one_cycle", "desc": "OneCycleLR - super-convergence"},
            {"value": "constant", "name": "constant", "desc": "Constant LR - baseline"},
        ],
        "conditional_on": None,
    },
    # GROUP 16: Bidirectional Training
    {
        "group_id": 17,
        "name": "bidirectional",
        "description": "Bidirectional training with cycle consistency",
        "parameter": "bidirectional",
        "variants": [
            {"value": False, "name": "unidirectional", "desc": "Forward only"},
            {"value": True, "name": "bidirectional", "desc": "Bidirectional with cycle loss"},
        ],
        "conditional_on": None,
    },
    # GROUP 17: Cycle Lambda (conditional on bidirectional=True)
    {
        "group_id": 18,
        "name": "cycle_lambda",
        "description": "Cycle consistency loss weight",
        "parameter": "cycle_lambda",
        "variants": [
            {"value": 0.1, "name": "weak_cycle", "desc": "λ=0.1 (weak)"},
            {"value": 0.5, "name": "medium_cycle", "desc": "λ=0.5 (medium)"},
            {"value": 1.0, "name": "strong_cycle", "desc": "λ=1.0 (strong)"},
        ],
        "conditional_on": None,  # Run unconditionally - greedy will handle appropriately
    },
]

# =============================================================================
# STAGE 2 GREEDY: Fine-tuning after Stage 1
# =============================================================================
# After Stage 1 greedy forward selection identifies optimal architecture/training choices,
# Stage 2 explores scaling and fine-tuning parameters that interact with each other:
# - Width (model capacity)
# - Augmentation strength (regularization)
# - Depth (model complexity)
# - Bidirectional (confirm always on)
# - Batch size (training dynamics)
#
# Total: 5 + 3 + 4 + 1 + 4 = 17 runs

STAGE2_GREEDY_GROUPS: List[Dict[str, Any]] = [
    # GROUP 1: WIDTH - model capacity scaling
    {
        "group_id": 1,
        "name": "width",
        "description": "Model width (base_channels) - primary capacity control",
        "parameter": "base_channels",
        "variants": [
            {"value": 128, "name": "width_128", "desc": "128 base channels - smaller model"},
            {"value": 192, "name": "width_192", "desc": "192 base channels - medium-small"},
            {"value": 256, "name": "width_256", "desc": "256 base channels - medium"},
            {"value": 384, "name": "width_384", "desc": "384 base channels - medium-large"},
            {"value": 512, "name": "width_512", "desc": "512 base channels - large model"},
        ],
        "conditional_on": None,
    },

    # GROUP 2: AUGMENTATION STRENGTH - regularization
    {
        "group_id": 2,
        "name": "augmentation",
        "description": "Data augmentation strength - regularization and generalization",
        "parameter": "aug_strength",
        "variants": [
            {"value": "medium", "name": "medium_aug", "desc": "Medium augmentation (noise=0.05, shift=25)"},
            {"value": "strong", "name": "strong_aug", "desc": "Strong augmentation (noise=0.1, shift=50)"},
            {"value": "very_strong", "name": "very_strong_aug", "desc": "Very strong augmentation (noise=0.15, shift=75)"},
        ],
        "conditional_on": None,
    },

    # GROUP 3: DEPTH - model complexity
    {
        "group_id": 3,
        "name": "depth",
        "description": "Model depth (n_downsample) - complexity vs efficiency trade-off",
        "parameter": "n_downsample",
        "variants": [
            {"value": 2, "name": "depth_2", "desc": "2 downsample blocks - shallow"},
            {"value": 3, "name": "depth_3", "desc": "3 downsample blocks - medium"},
            {"value": 4, "name": "depth_4", "desc": "4 downsample blocks - deep"},
            {"value": 5, "name": "depth_5", "desc": "5 downsample blocks - very deep"},
        ],
        "conditional_on": None,
    },

    # GROUP 4: BIDIRECTIONAL - confirm winner
    {
        "group_id": 4,
        "name": "bidirectional",
        "description": "Bidirectional processing - confirm always on is optimal",
        "parameter": "bidirectional",
        "variants": [
            {"value": True, "name": "bidirectional_on", "desc": "Bidirectional processing enabled"},
        ],
        "conditional_on": None,
    },

    # GROUP 5: BATCH SIZE - training dynamics
    {
        "group_id": 5,
        "name": "batch_size",
        "description": "Batch size - affects gradient noise and memory usage",
        "parameter": "batch_size",
        "variants": [
            {"value": 8, "name": "batch_8", "desc": "Batch size 8 - more gradient noise"},
            {"value": 16, "name": "batch_16", "desc": "Batch size 16 - balanced"},
            {"value": 32, "name": "batch_32", "desc": "Batch size 32 - less noise"},
            {"value": 64, "name": "batch_64", "desc": "Batch size 64 - smooth gradients"},
        ],
        "conditional_on": None,
    },

    # GROUP 6: LOSS TYPE - L1 vs wavelet-enhanced
    {
        "group_id": 6,
        "name": "loss_type",
        "description": "Loss function - L1 alone vs L1+wavelet for multi-scale learning",
        "parameter": "loss_type",
        "variants": [
            {"value": "l1", "name": "l1_only", "desc": "Pure L1 loss - simple, robust"},
            {"value": "l1_wavelet", "name": "l1_wavelet", "desc": "L1 + wavelet loss - multi-scale frequency learning"},
        ],
        "conditional_on": None,
    },
]

# Stage 2 defaults - starting point for Stage 2 (should load from Stage 1 results)
# These are placeholder defaults; actual values come from Stage 1 optimal config
STAGE2_GREEDY_DEFAULTS: Dict[str, Any] = {
    # These will be overridden by Stage 1 winners when running Stage 2
    "conv_type": "standard",
    "n_downsample": 2,
    "base_channels": 64,
    "attention_type": "basic",
    "n_heads": 4,
    "cond_mode": "film",
    "use_odor_embedding": True,
    "loss_type": "l1",
    "use_augmentation": True,
    "aug_strength": "medium",
    "bidirectional": True,
    "cycle_lambda": 0.5,
    "norm_type": "batch",
    "skip_type": "add",
    "activation": "relu",
    "dropout": 0.0,
    "optimizer": "adamw",
    "lr_schedule": "cosine",
    "weight_decay": 1e-4,
    "use_session_stats": False,
    "session_emb_dim": 32,
    "session_use_spectral": False,
    "use_adaptive_scaling": False,
    "use_cov_augment": False,
    "cov_augment_prob": 0.5,
    "use_session_embedding": False,
    "use_adabn": False,
    # Stage 2 specific - batch_size is special (handled by TrainingConfig)
    "batch_size": 16,
}

def _count_stage2_runs() -> int:
    """Count total runs for Stage 2 greedy selection."""
    total = 0
    for group in STAGE2_GREEDY_GROUPS:
        total += len(group["variants"])
    return total

STAGE2_TOTAL_RUNS = _count_stage2_runs()  # 17 runs

# Calculate total runs
def _count_greedy_runs() -> int:
    """Count total runs for greedy forward selection."""
    total = 0
    for group in ABLATION_GROUPS:
        # Conditional groups might be skipped, but count them anyway for worst case
        total += len(group["variants"])
    return total

GREEDY_TOTAL_RUNS = _count_greedy_runs()  # 33 runs

# =============================================================================
# SUBTRACTIVE PROTOCOL: Traditional Ablation (legacy)
# =============================================================================
# Start with full model, remove components one at a time

ATTENTION_VARIANTS: Dict[str, str] = {
    "none": "none",
    "basic": "basic",
    "cross_freq": "cross_freq",
    "cross_freq_v2": "cross_freq_v2",
}

CONDITIONING_VARIANTS: Dict[str, Dict[str, Any]] = {
    "none": {"cond_mode": "none", "use_odor_embedding": False},
    "odor": {"cond_mode": "film", "use_odor_embedding": True},
    "spectro_temporal": {"cond_mode": "cross_attn_gated", "use_odor_embedding": True},
}

LOSS_VARIANTS: Dict[str, Dict[str, Any]] = {
    "l1": {"loss_type": "l1", "loss_weights": None},
    "l1_wavelet": {"loss_type": "l1_wavelet", "loss_weights": None},
    "huber": {"loss_type": "huber", "loss_weights": None},
    "huber_wavelet": {"loss_type": "huber_wavelet", "loss_weights": None},
}

CAPACITY_VARIANTS: Dict[str, int] = {
    "small": 32,
    "medium": 64,
    "large": 128,
}

BIDIRECTIONAL_VARIANTS: Dict[str, bool] = {
    "unidirectional": False,
    "bidirectional": True,
}

AUGMENTATION_VARIANTS: Dict[str, bool] = {
    "none": False,
    "full": True,
}

CONV_TYPE_VARIANTS: Dict[str, str] = {
    "standard": "standard",
    "modern": "modern",
}

# For subtractive: which variant to use when "ablating" (removing) each component
ABLATION_VARIANTS_SIMPLE: Dict[str, str] = {
    "attention": "none",
    "conditioning": "none",
    "loss": "l1",
    "capacity": "small",
    "bidirectional": "unidirectional",
    "augmentation": "none",
    "conv_type": "standard",
}

ABLATION_STUDIES: Dict[str, Dict[str, Any]] = {
    "attention": {
        "description": "Attention mechanism comparison",
        "variants": list(ATTENTION_VARIANTS.keys()),
        "parameter": "attention_type",
        "n_variants": 4,
    },
    "conditioning": {
        "description": "Conditioning strategy comparison",
        "variants": list(CONDITIONING_VARIANTS.keys()),
        "parameter": "conditioning",
        "n_variants": 3,
    },
    "loss": {
        "description": "Loss function comparison",
        "variants": list(LOSS_VARIANTS.keys()),
        "parameter": "loss",
        "n_variants": 4,
    },
    "capacity": {
        "description": "Model capacity comparison",
        "variants": list(CAPACITY_VARIANTS.keys()),
        "parameter": "base_channels",
        "n_variants": 3,
    },
    "bidirectional": {
        "description": "Training direction comparison",
        "variants": list(BIDIRECTIONAL_VARIANTS.keys()),
        "parameter": "bidirectional",
        "n_variants": 2,
    },
    "augmentation": {
        "description": "Data augmentation comparison",
        "variants": list(AUGMENTATION_VARIANTS.keys()),
        "parameter": "use_augmentation",
        "n_variants": 2,
    },
    "conv_type": {
        "description": "Convolution type comparison",
        "variants": list(CONV_TYPE_VARIANTS.keys()),
        "parameter": "conv_type",
        "n_variants": 2,
    },
}


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment.

    Works with additive, subtractive, and greedy_forward protocols.
    """

    study: str
    variant: str

    # CondUNet configuration
    attention_type: str = "cross_freq_v2"
    cond_mode: str = "cross_attn_gated"
    use_odor_embedding: bool = True
    base_channels: int = 64
    n_downsample: int = 2
    conv_type: str = "modern"
    n_heads: int = 4

    # Loss configuration
    loss_type: str = "l1_wavelet"
    loss_weights: Optional[Dict[str, float]] = None

    # Training options
    use_augmentation: bool = True
    bidirectional: bool = True
    cycle_lambda: float = 0.5

    # Augmentation parameters
    aug_noise_std: float = 0.1
    aug_time_shift: int = 50
    aug_strength: str = "medium"  # none, light, medium, heavy

    # NEW: Architectural choices (Nature Methods level ablation)
    norm_type: str = "batch"        # batch, layer, instance, group, rms, none
    skip_type: str = "add"          # add, concat, attention, dense
    activation: str = "relu"        # relu, leaky_relu, gelu, silu, mish
    dropout: float = 0.0            # Dropout rate

    # NEW: Training choices (Nature Methods level ablation)
    optimizer: str = "adamw"        # adamw, adam, sgd, lion, adafactor
    lr_schedule: str = "cosine"     # cosine, cosine_warmup, step, plateau, onecycle, constant
    weight_decay: float = 1e-4      # L2 regularization strength

    # Cross-session evaluation (for Nature Methods main result)
    split_by_session: bool = True  # True = cross-session, False = intra-session
    n_test_sessions: int = 4  # Sessions held out for testing
    n_val_sessions: int = 1  # Sessions held out for validation

    # Session adaptation methods (Group 18)
    # Multiple approaches based on domain adaptation literature:
    use_session_stats: bool = False  # Statistics-based conditioning (FiLM style)
    session_emb_dim: int = 32  # Session statistics embedding dimension
    session_use_spectral: bool = False  # Include spectral features in session stats
    use_adaptive_scaling: bool = False  # Session-adaptive output scaling (AdaIN style)
    use_cov_augment: bool = False  # Covariance expansion augmentation
    cov_augment_prob: float = 0.5  # Probability of applying cov augmentation
    use_session_embedding: bool = False  # Learnable session embedding (lookup table → FiLM)
    use_adabn: bool = False  # AdaBN: use batch stats at inference (requires norm_type="batch")

    # Stage info for additive protocol
    stage: int = -1  # -1 = subtractive protocol

    # Group info for greedy_forward protocol
    group_id: int = -1  # -1 = not greedy_forward

    def __post_init__(self):
        """Apply ablation variant modifications for subtractive protocol."""
        if self.stage >= 0:
            # Additive protocol: config already set from INCREMENTAL_STAGES
            return

        # Subtractive protocol: modify based on study/variant
        if self.study == "attention":
            self.attention_type = ATTENTION_VARIANTS[self.variant]
        elif self.study == "conditioning":
            cond_cfg = CONDITIONING_VARIANTS[self.variant]
            self.cond_mode = cond_cfg["cond_mode"]
            self.use_odor_embedding = cond_cfg["use_odor_embedding"]
        elif self.study == "loss":
            loss_cfg = LOSS_VARIANTS[self.variant]
            self.loss_type = loss_cfg["loss_type"]
            self.loss_weights = loss_cfg["loss_weights"]
        elif self.study == "capacity":
            self.base_channels = CAPACITY_VARIANTS[self.variant]
        elif self.study == "bidirectional":
            self.bidirectional = BIDIRECTIONAL_VARIANTS[self.variant]
        elif self.study == "augmentation":
            self.use_augmentation = AUGMENTATION_VARIANTS[self.variant]
        elif self.study == "conv_type":
            self.conv_type = CONV_TYPE_VARIANTS[self.variant]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study": self.study,
            "variant": self.variant,
            "stage": self.stage,
            "group_id": self.group_id,
            "attention_type": self.attention_type,
            "cond_mode": self.cond_mode,
            "use_odor_embedding": self.use_odor_embedding,
            "base_channels": self.base_channels,
            "n_downsample": self.n_downsample,
            "n_heads": self.n_heads,
            "conv_type": self.conv_type,
            "loss_type": self.loss_type,
            "loss_weights": self.loss_weights,
            "use_augmentation": self.use_augmentation,
            "aug_strength": self.aug_strength,
            "bidirectional": self.bidirectional,
            "cycle_lambda": self.cycle_lambda,
            # NEW: Architectural choices
            "norm_type": self.norm_type,
            "skip_type": self.skip_type,
            "activation": self.activation,
            "dropout": self.dropout,
            # NEW: Training choices
            "optimizer": self.optimizer,
            "lr_schedule": self.lr_schedule,
            "weight_decay": self.weight_decay,
            # Cross-session evaluation
            "split_by_session": self.split_by_session,
            "n_test_sessions": self.n_test_sessions,
            "n_val_sessions": self.n_val_sessions,
            # Session adaptation methods
            "use_session_stats": self.use_session_stats,
            "session_emb_dim": self.session_emb_dim,
            "session_use_spectral": self.session_use_spectral,
            "use_adaptive_scaling": self.use_adaptive_scaling,
            "use_cov_augment": self.use_cov_augment,
            "cov_augment_prob": self.cov_augment_prob,
            "use_session_embedding": self.use_session_embedding,
            "use_adabn": self.use_adabn,
        }

    @property
    def name(self) -> str:
        """Experiment name for logging."""
        if self.group_id >= 0:
            return f"g{self.group_id}_{self.variant}"
        if self.stage >= 0:
            return f"stage{self.stage}_{self.variant}"
        return f"{self.study}_{self.variant}"

    @classmethod
    def from_greedy_group(
        cls,
        group: Dict[str, Any],
        variant: Dict[str, Any],
        current_config: Dict[str, Any],
    ) -> "AblationConfig":
        """Create config from greedy forward selection group.

        Args:
            group: Group definition from ABLATION_GROUPS
            variant: Variant definition within the group
            current_config: Current best configuration (winner of previous groups)

        Returns:
            AblationConfig with the variant applied on top of current best
        """
        # Start with current best config
        config = current_config.copy()

        # Apply the variant's parameter(s)
        # Support both single-parameter and multi-parameter variants
        if "config" in variant:
            # Multi-parameter variant (like Group 18 session adaptation)
            # The variant has a "config" dict with multiple param: value pairs
            for param, value in variant["config"].items():
                config[param] = value
        else:
            # Single-parameter variant (traditional)
            param = group["parameter"]
            config[param] = variant["value"]

        # Handle special cases for augmentation variants
        if "aug_strength" in variant:
            config["aug_strength"] = variant["aug_strength"]

        return cls(
            study=f"group{group['group_id']}_{group['name']}",
            variant=variant["name"],
            group_id=group["group_id"],
            conv_type=config.get("conv_type", "standard"),
            attention_type=config.get("attention_type", "none"),
            cond_mode=config.get("cond_mode", "none"),
            use_odor_embedding=config.get("use_odor_embedding", False),
            base_channels=config.get("base_channels", 64),
            n_downsample=config.get("n_downsample", 2),
            n_heads=config.get("n_heads", 4),
            loss_type=config.get("loss_type", "l1"),
            use_augmentation=config.get("use_augmentation", False),
            aug_strength=config.get("aug_strength", "none"),
            bidirectional=config.get("bidirectional", False),
            cycle_lambda=config.get("cycle_lambda", 0.5),
            # NEW: Architectural choices
            norm_type=config.get("norm_type", "batch"),
            skip_type=config.get("skip_type", "add"),
            activation=config.get("activation", "relu"),
            dropout=config.get("dropout", 0.0),
            # NEW: Training choices
            optimizer=config.get("optimizer", "adamw"),
            lr_schedule=config.get("lr_schedule", "cosine"),
            weight_decay=config.get("weight_decay", 1e-4),
            # Multi-session validation: 6 train, 3 val, NO test
            # Winner = best mean R² across all 3 validation sessions
            split_by_session=True,
            n_test_sessions=0,  # No separate test set
            n_val_sessions=3,   # All held-out sessions for validation
            # Session adaptation methods
            use_session_stats=config.get("use_session_stats", False),
            session_emb_dim=config.get("session_emb_dim", 32),
            session_use_spectral=config.get("session_use_spectral", False),
            use_adaptive_scaling=config.get("use_adaptive_scaling", False),
            use_cov_augment=config.get("use_cov_augment", False),
            cov_augment_prob=config.get("cov_augment_prob", 0.5),
            use_session_embedding=config.get("use_session_embedding", False),
            use_adabn=config.get("use_adabn", False),
        )

    @classmethod
    def from_stage(cls, stage_idx: int) -> "AblationConfig":
        """Create config from incremental stage definition."""
        if stage_idx < 0 or stage_idx >= len(INCREMENTAL_STAGES):
            raise ValueError(f"Invalid stage index: {stage_idx}")

        stage = INCREMENTAL_STAGES[stage_idx]
        cfg = stage["config"]

        return cls(
            study=f"stage{stage_idx}",
            variant=stage["name"],
            stage=stage_idx,
            conv_type=cfg["conv_type"],
            attention_type=cfg["attention_type"],
            cond_mode=cfg["cond_mode"],
            use_odor_embedding=cfg["use_odor_embedding"],
            loss_type=cfg["loss_type"],
            use_augmentation=cfg["use_augmentation"],
            bidirectional=cfg["bidirectional"],
            base_channels=cfg["base_channels"],
            # Cross-session evaluation parameters
            split_by_session=cfg.get("split_by_session", True),
            n_test_sessions=cfg.get("n_test_sessions", 4),
            n_val_sessions=cfg.get("n_val_sessions", 1),
        )


@dataclass
class TrainingConfig:
    """Training hyperparameters for ablation experiments."""

    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Learning rate schedule
    lr_scheduler: str = "cosine"
    lr_min: float = 1e-6

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Gradient clipping
    grad_clip: float = 1.0

    # Mixed precision
    use_amp: bool = True

    # Robustness settings
    checkpoint_every: int = 10
    max_nan_recovery: int = 100
    log_every: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "lr_scheduler": self.lr_scheduler,
            "lr_min": self.lr_min,
            "patience": self.patience,
            "grad_clip": self.grad_clip,
            "use_amp": self.use_amp,
            "checkpoint_every": self.checkpoint_every,
            "max_nan_recovery": self.max_nan_recovery,
        }


@dataclass
class Phase3Config:
    """Configuration for Phase 3 ablation studies.

    Supports four protocols:
    - ADDITIVE (build-up): Start simple, add components incrementally
    - SUBTRACTIVE (traditional): Start full, remove components
    - GREEDY_FORWARD (recommended): Test groups sequentially, winner propagates
    - STAGE2_GREEDY: Fine-tuning after Stage 1 (width, aug, depth, batch_size)
    """

    # Dataset (Phase 3 uses olfactory only)
    dataset: str = "olfactory"
    sample_rate: float = 1000.0

    # Ablation protocol: "additive", "subtractive", "greedy_forward", or "stage2_greedy"
    protocol: str = "greedy_forward"

    # For subtractive protocol: which studies to run
    studies: List[str] = field(default_factory=lambda: list(ABLATION_STUDIES.keys()))

    # For additive protocol: which stages to run (default: all)
    stages: List[int] = field(default_factory=lambda: list(range(len(INCREMENTAL_STAGES))))

    # For greedy_forward protocol: which groups to run (default: all)
    groups: List[int] = field(default_factory=lambda: list(range(1, len(ABLATION_GROUPS) + 1)))

    # For stage2_greedy protocol: which Stage 2 groups to run (default: all)
    stage2_groups: List[int] = field(default_factory=lambda: list(range(1, len(STAGE2_GREEDY_GROUPS) + 1)))

    # For stage2_greedy: path to Stage 1 results (to load optimal config as baseline)
    stage1_results_path: Optional[Path] = None

    # For stage2_greedy: manually specified Stage 1 optimal config (overrides stage1_results_path)
    stage1_optimal_config: Optional[Dict[str, Any]] = None

    # Cross-validation settings
    # For greedy_forward: use single split (no CV) following nnU-Net methodology
    n_folds: int = 1  # 1 = single split (recommended for ablation)
    cv_seed: int = 42

    # Training config
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Output
    output_dir: Path = field(default_factory=lambda: Path("results/phase3"))

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_workers: int = 4

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/phase3"))

    # Logging
    use_wandb: bool = False
    wandb_project: str = "neural-signal-translation"
    verbose: int = 1
    log_dir: Path = field(default_factory=lambda: Path("logs/phase3"))

    # Phase 2 baseline (best architecture R²)
    phase2_best_r2: Optional[float] = None

    def __post_init__(self):
        """Validate configuration."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.protocol not in ABLATION_PROTOCOLS:
            raise ValueError(
                f"Unknown protocol: {self.protocol}. "
                f"Use 'additive', 'subtractive', 'greedy_forward', or 'stage2_greedy'"
            )

        # Validate studies (for subtractive protocol)
        for study in self.studies:
            if study not in ABLATION_STUDIES:
                raise ValueError(f"Unknown ablation study: {study}")

        # Validate stages (for additive protocol)
        for stage in self.stages:
            if stage < 0 or stage >= len(INCREMENTAL_STAGES):
                raise ValueError(f"Invalid stage: {stage}. Valid: 0-{len(INCREMENTAL_STAGES)-1}")

        # Validate groups (for greedy_forward protocol)
        for group_id in self.groups:
            if group_id < 1 or group_id > len(ABLATION_GROUPS):
                raise ValueError(f"Invalid group: {group_id}. Valid: 1-{len(ABLATION_GROUPS)}")

        # Validate stage2_groups (for stage2_greedy protocol)
        for group_id in self.stage2_groups:
            if group_id < 1 or group_id > len(STAGE2_GREEDY_GROUPS):
                raise ValueError(f"Invalid stage2 group: {group_id}. Valid: 1-{len(STAGE2_GREEDY_GROUPS)}")

    def get_ablation_configs(self) -> List[AblationConfig]:
        """Generate ablation configurations based on selected protocol.

        Note: For greedy_forward/stage2_greedy, this returns ALL possible configs for planning.
        The actual execution should use get_greedy_groups()/get_stage2_groups() and run sequentially.
        """
        if self.protocol == "additive":
            return self._get_additive_configs()
        elif self.protocol == "greedy_forward":
            return self._get_greedy_configs()
        elif self.protocol == "stage2_greedy":
            return self._get_stage2_greedy_configs()
        else:
            return self._get_subtractive_configs()

    def _get_additive_configs(self) -> List[AblationConfig]:
        """Generate configs for additive (build-up) protocol."""
        configs = []
        for stage_idx in self.stages:
            configs.append(AblationConfig.from_stage(stage_idx))
        return configs

    def _get_subtractive_configs(self) -> List[AblationConfig]:
        """Generate configs for subtractive (traditional) protocol."""
        configs = []
        for study in self.studies:
            variant = ABLATION_VARIANTS_SIMPLE[study]
            configs.append(AblationConfig(study=study, variant=variant))
        return configs

    def _get_greedy_configs(self) -> List[AblationConfig]:
        """Generate all configs for greedy_forward protocol (for planning only).

        Note: In actual execution, configs are generated dynamically as winners
        are determined. This method returns all possible configs assuming
        no conditional skipping and all defaults.
        """
        configs = []
        current_config = GREEDY_DEFAULTS.copy()

        for group in ABLATION_GROUPS:
            if group["group_id"] not in self.groups:
                continue

            for variant in group["variants"]:
                config = AblationConfig.from_greedy_group(group, variant, current_config)
                configs.append(config)

        return configs

    def get_greedy_groups(self) -> List[Dict[str, Any]]:
        """Get ablation groups for greedy_forward protocol, sorted by group_id."""
        groups = [g for g in ABLATION_GROUPS if g["group_id"] in self.groups]
        return sorted(groups, key=lambda g: g["group_id"])

    def _get_stage2_greedy_configs(self) -> List[AblationConfig]:
        """Generate configs for Stage 2 greedy selection protocol."""
        configs = []

        # Start with Stage 1 optimal config or defaults
        current_config = self.get_stage2_baseline()

        for group in STAGE2_GREEDY_GROUPS:
            if group["group_id"] not in self.stage2_groups:
                continue

            for variant in group["variants"]:
                config = AblationConfig.from_greedy_group(group, variant, current_config)
                configs.append(config)

        return configs

    def get_stage2_groups(self) -> List[Dict[str, Any]]:
        """Get ablation groups for stage2_greedy protocol, sorted by group_id."""
        groups = [g for g in STAGE2_GREEDY_GROUPS if g["group_id"] in self.stage2_groups]
        return sorted(groups, key=lambda g: g["group_id"])

    def get_stage2_baseline(self) -> Dict[str, Any]:
        """Get the baseline config for Stage 2 (from Stage 1 results or defaults).

        Priority:
        1. stage1_optimal_config (if provided)
        2. Load from stage1_results_path (if provided)
        3. STAGE2_GREEDY_DEFAULTS (fallback)
        """
        if self.stage1_optimal_config is not None:
            # Merge with defaults to ensure all keys present
            baseline = STAGE2_GREEDY_DEFAULTS.copy()
            baseline.update(self.stage1_optimal_config)
            return baseline

        if self.stage1_results_path is not None:
            import json
            results_path = Path(self.stage1_results_path)
            if results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                if "optimal_config" in data:
                    baseline = STAGE2_GREEDY_DEFAULTS.copy()
                    baseline.update(data["optimal_config"])
                    return baseline

        return STAGE2_GREEDY_DEFAULTS.copy()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sample_rate": self.sample_rate,
            "protocol": self.protocol,
            "studies": self.studies,
            "stages": self.stages,
            "groups": self.groups,
            "stage2_groups": self.stage2_groups,
            "stage1_results_path": str(self.stage1_results_path) if self.stage1_results_path else None,
            "n_folds": self.n_folds,
            "cv_seed": self.cv_seed,
            "training": self.training.to_dict(),
            "output_dir": str(self.output_dir),
            "device": self.device,
            "phase2_best_r2": self.phase2_best_r2,
        }

    @property
    def total_runs(self) -> int:
        """Total number of training runs."""
        if self.protocol == "additive":
            # All stages × n_folds
            return len(self.stages) * self.n_folds
        elif self.protocol == "greedy_forward":
            # Sum of variants in selected groups (single split, no CV)
            total = 0
            for group in ABLATION_GROUPS:
                if group["group_id"] in self.groups:
                    total += len(group["variants"])
            return total
        elif self.protocol == "stage2_greedy":
            # Sum of variants in selected Stage 2 groups (single split, no CV)
            total = 0
            for group in STAGE2_GREEDY_GROUPS:
                if group["group_id"] in self.stage2_groups:
                    total += len(group["variants"])
            return total
        else:
            # (1 baseline + n_studies ablations) × n_folds
            return (1 + len(self.studies)) * self.n_folds


def get_baseline_config() -> AblationConfig:
    """Get the baseline configuration.

    For ADDITIVE protocol: Stage 0 (simple baseline)
    For SUBTRACTIVE protocol: Full model (all features enabled)
    """
    # Return full model config (Stage 6 = final stage with all components)
    return AblationConfig.from_stage(len(INCREMENTAL_STAGES) - 1)


def get_simple_baseline_config() -> AblationConfig:
    """Get the simple baseline configuration (Stage 0)."""
    return AblationConfig.from_stage(0)


def check_conditional(condition: Optional[Dict[str, Any]], current_config: Dict[str, Any]) -> bool:
    """Check if a conditional group should be executed.

    Args:
        condition: Conditional specification from group definition
        current_config: Current best configuration

    Returns:
        True if the group should be executed, False if skipped
    """
    if condition is None:
        return True

    param = condition["parameter"]
    current_value = current_config.get(param)

    if "equal" in condition:
        return current_value == condition["equal"]
    elif "not_equal" in condition:
        return current_value != condition["not_equal"]

    return True


def print_protocol_summary(protocol: str = "greedy_forward"):
    """Print summary of the ablation protocol."""
    print("\n" + "=" * 70)
    print(f"ABLATION PROTOCOL: {protocol.upper()}")
    print("=" * 70)

    if protocol == "greedy_forward":
        print("\nGREEDY FORWARD SELECTION (Coordinate-wise Optimization)")
        print("Based on: nnU-Net (Nature Methods), Sequential Forward Selection")
        print("-" * 70)
        print("Methodology:")
        print("  1. Test all variants in each group")
        print("  2. Pick winner (best R²)")
        print("  3. Winner becomes default for subsequent groups")
        print("  4. Conditional groups are skipped if condition not met")
        print("-" * 70)
        print(f"{'Group':<6}{'Name':<18}{'Variants':<8}{'Conditional':<20}")
        print("-" * 70)
        for group in ABLATION_GROUPS:
            cond = group["conditional_on"]
            cond_str = "None"
            if cond:
                if "equal" in cond:
                    cond_str = f"{cond['parameter']}={cond['equal']}"
                elif "not_equal" in cond:
                    cond_str = f"{cond['parameter']}≠{cond['not_equal']}"
            print(f"{group['group_id']:<6}{group['name']:<18}{len(group['variants']):<8}{cond_str:<20}")
        print("-" * 70)
        print(f"Total runs (worst case): {GREEDY_TOTAL_RUNS}")
        print("Single split: No cross-validation during ablation (per nnU-Net)")
    elif protocol == "stage2_greedy":
        print("\nSTAGE 2 GREEDY SELECTION (Fine-tuning after Stage 1)")
        print("Explores scaling/regularization after architecture is fixed")
        print("-" * 70)
        print("Methodology:")
        print("  1. Load optimal config from Stage 1 results")
        print("  2. Test width, augmentation, depth, batch_size variants")
        print("  3. Winner propagates to next group")
        print("-" * 70)
        print(f"{'Group':<6}{'Name':<18}{'Variants':<8}{'Parameter':<20}")
        print("-" * 70)
        for group in STAGE2_GREEDY_GROUPS:
            print(f"{group['group_id']:<6}{group['name']:<18}{len(group['variants']):<8}{group['parameter']:<20}")
        print("-" * 70)
        print(f"Total runs: {STAGE2_TOTAL_RUNS}")
        print("Single split: No cross-validation during ablation (per nnU-Net)")
    elif protocol == "additive":
        print("\nINCREMENTAL COMPONENT ANALYSIS (Build-Up Approach)")
        print("Based on: nnU-Net (Nature Methods), Transformer paper methodology")
        print("-" * 70)
        print(f"{'Stage':<8}{'Name':<20}{'Description':<42}")
        print("-" * 70)
        for stage in INCREMENTAL_STAGES:
            print(f"{stage['stage']:<8}{stage['name']:<20}{stage['description']:<42}")
        print("-" * 70)
        print("\nEach stage adds ONE component on top of previous stage.")
        print("This shows the incremental contribution of each component.")
    else:
        print("\nTRADITIONAL ABLATION (Subtractive Approach)")
        print("-" * 70)
        print("Start with full model, remove one component at a time.")
        print(f"{'Study':<15}{'Ablated Variant':<20}{'Description':<35}")
        print("-" * 70)
        for study, info in ABLATION_STUDIES.items():
            variant = ABLATION_VARIANTS_SIMPLE[study]
            print(f"{study:<15}{variant:<20}{info['description']:<35}")
        print("-" * 70)

    print("=" * 70 + "\n")
