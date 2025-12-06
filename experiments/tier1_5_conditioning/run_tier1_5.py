#!/usr/bin/env python3
"""
Tier 1.5: Conditioning Deep Dive
=================================

Tests different conditioning mechanisms to find the best way to incorporate
auxiliary information (e.g., odor identity, behavioral state) into the model.

Conditioning Mechanisms Tested:
    1. none: No conditioning (baseline)
    2. concat: Simple concatenation
    3. film: Feature-wise Linear Modulation
    4. cross_attn: Cross-attention conditioning
    5. cross_attn_gated: Gated cross-attention
    6. adaptive_norm: Adaptive instance normalization

This tier uses the winning architecture from Tier 1 screening.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fix imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.common.cross_validation import create_data_splits
from experiments.common.config_registry import (
    get_registry,
    Tier1_5Result,
    ConditioningResult,
)


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier1_5_conditioning"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Conditioning mechanisms to test
CONDITIONING_MECHANISMS = [
    "none",           # No conditioning (baseline)
    "concat",         # Simple concatenation
    "film",           # Feature-wise Linear Modulation
    "cross_attn",     # Cross-attention
    "cross_attn_gated",  # Gated cross-attention
    "adaptive_norm",  # Adaptive instance normalization
]

# Fast subset for dry-run
FAST_CONDITIONINGS = ["none", "concat", "cross_attn"]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConditioningRunResult:
    """Result from a single conditioning run."""
    conditioning: str
    seed: int
    fold: int
    r2: float
    mae: float
    training_time: float


# =============================================================================
# Model Creation with Conditioning
# =============================================================================

def create_conditioned_model(
    arch_name: str,
    conditioning: str,
    in_channels: int,
    out_channels: int,
    cond_dim: int = 64,
    device: torch.device = None,
) -> nn.Module:
    """Create model with specified conditioning mechanism.

    Args:
        arch_name: Base architecture name
        conditioning: Conditioning mechanism type
        in_channels: Input channels
        out_channels: Output channels
        cond_dim: Conditioning embedding dimension
        device: Device to place model on

    Returns:
        Model instance
    """
    from models import create_architecture, UNet1DConditioned

    # Variant mapping
    VARIANT_MAP = {
        "linear": "simple",
        "cnn": "basic",
        "wavenet": "standard",
        "fnet": "standard",
        "vit": "standard",
        "performer": "standard",
        "mamba": "standard",
    }

    if arch_name == "unet":
        # UNet has built-in conditioning support
        model = UNet1DConditioned(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=64,
            conditioning_type=conditioning if conditioning != "none" else None,
            cond_dim=cond_dim,
        )
    elif arch_name in VARIANT_MAP:
        variant = VARIANT_MAP[arch_name]
        # For other architectures, wrap with conditioning
        base_model = create_architecture(
            arch_name,
            variant=variant,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        if conditioning == "none":
            model = base_model
        else:
            model = ConditioningWrapper(
                base_model,
                conditioning_type=conditioning,
                cond_dim=cond_dim,
                hidden_dim=out_channels,
            )
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    return model.to(device) if device else model


class ConditioningWrapper(nn.Module):
    """Wrapper to add conditioning to any base model."""

    def __init__(
        self,
        base_model: nn.Module,
        conditioning_type: str,
        cond_dim: int = 64,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.base_model = base_model
        self.conditioning_type = conditioning_type
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        if conditioning_type == "concat":
            # Project conditioning to spatial dimension
            self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        elif conditioning_type == "film":
            # FiLM: gamma and beta per channel
            self.film_gamma = nn.Linear(cond_dim, hidden_dim)
            self.film_beta = nn.Linear(cond_dim, hidden_dim)
        elif conditioning_type in ["cross_attn", "cross_attn_gated"]:
            # Cross-attention
            self.cond_proj = nn.Linear(cond_dim, hidden_dim)
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            if conditioning_type == "cross_attn_gated":
                self.gate = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Sigmoid(),
                )
        elif conditioning_type == "adaptive_norm":
            # Adaptive instance normalization
            self.norm = nn.InstanceNorm1d(hidden_dim, affine=False)
            self.style_gamma = nn.Linear(cond_dim, hidden_dim)
            self.style_beta = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with conditioning.

        Args:
            x: Input tensor [B, C, T]
            cond: Conditioning tensor [B, cond_dim] or None

        Returns:
            Output tensor [B, C, T]
        """
        # Get base model output
        if hasattr(self.base_model, 'forward'):
            # Check if base model accepts conditioning
            try:
                out = self.base_model(x, cond)
            except TypeError:
                out = self.base_model(x)
        else:
            out = self.base_model(x)

        if cond is None or self.conditioning_type == "none":
            return out

        B, C, T = out.shape

        if self.conditioning_type == "concat":
            # Add conditioning as bias
            cond_proj = self.cond_proj(cond)  # [B, C]
            out = out + cond_proj.unsqueeze(-1)

        elif self.conditioning_type == "film":
            # FiLM modulation
            gamma = self.film_gamma(cond).unsqueeze(-1)  # [B, C, 1]
            beta = self.film_beta(cond).unsqueeze(-1)    # [B, C, 1]
            out = gamma * out + beta

        elif self.conditioning_type == "cross_attn":
            # Cross-attention
            cond_proj = self.cond_proj(cond).unsqueeze(1)  # [B, 1, C]
            out_flat = out.transpose(1, 2)  # [B, T, C]
            attn_out, _ = self.attn(out_flat, cond_proj, cond_proj)
            out = out + attn_out.transpose(1, 2)

        elif self.conditioning_type == "cross_attn_gated":
            # Gated cross-attention
            cond_proj = self.cond_proj(cond).unsqueeze(1)  # [B, 1, C]
            out_flat = out.transpose(1, 2)  # [B, T, C]
            attn_out, _ = self.attn(out_flat, cond_proj, cond_proj)

            # Compute gate
            gate_input = torch.cat([out_flat, attn_out], dim=-1)
            gate = self.gate(gate_input)  # [B, T, C]

            out = out + (gate * attn_out).transpose(1, 2)

        elif self.conditioning_type == "adaptive_norm":
            # Adaptive instance normalization
            out_norm = self.norm(out)
            gamma = self.style_gamma(cond).unsqueeze(-1)  # [B, C, 1]
            beta = self.style_beta(cond).unsqueeze(-1)    # [B, C, 1]
            out = gamma * out_norm + beta

        return out


# =============================================================================
# Training
# =============================================================================

def train_with_conditioning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> Tuple[float, float]:
    """Train model with conditioning and return validation metrics.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of epochs
        device: Device to use
        lr: Learning rate

    Returns:
        Tuple of (r2, mae)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.HuberLoss()

    model.train()
    for epoch in range(n_epochs):
        for batch in train_loader:
            if len(batch) == 3:
                X, y, cond = batch
                X, y, cond = X.to(device), y.to(device), cond.to(device)
            else:
                X, y = batch
                X, y = X.to(device), y.to(device)
                cond = None

            optimizer.zero_grad()

            # Forward pass
            if cond is not None:
                try:
                    y_pred = model(X, cond)
                except TypeError:
                    y_pred = model(X)
            else:
                y_pred = model(X)

            loss = criterion(y_pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

    # Validation
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                X, y, cond = batch
                X, y, cond = X.to(device), y.to(device), cond.to(device)
            else:
                X, y = batch
                X, y = X.to(device), y.to(device)
                cond = None

            if cond is not None:
                try:
                    y_pred = model(X, cond)
                except TypeError:
                    y_pred = model(X)
            else:
                y_pred = model(X)

            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # Flatten for metrics
    preds_flat = preds.reshape(preds.shape[0], -1)
    targets_flat = targets.reshape(targets.shape[0], -1)

    # R²
    ss_res = np.sum((targets_flat - preds_flat) ** 2)
    ss_tot = np.sum((targets_flat - np.mean(targets_flat, axis=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # MAE
    mae = np.mean(np.abs(targets_flat - preds_flat))

    return float(r2), float(mae)


# =============================================================================
# Main Tier Function
# =============================================================================

def run_tier1_5(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    architecture: Optional[str] = None,
    conditionings: Optional[List[str]] = None,
    cond_labels: Optional[torch.Tensor] = None,
    n_seeds: int = 3,
    n_folds: int = 3,
    n_epochs: int = 50,
    batch_size: int = 32,
    register: bool = True,
) -> Tier1_5Result:
    """Run Tier 1.5: Conditioning Deep Dive.

    Args:
        X: Input data [N, C, T]
        y: Target data [N, C, T]
        device: Device to use
        architecture: Architecture to use (from Tier 1 winner, or specified)
        conditionings: List of conditioning mechanisms to test
        cond_labels: Conditioning labels [N] (e.g., odor IDs)
        n_seeds: Number of random seeds
        n_folds: Number of CV folds
        n_epochs: Training epochs
        batch_size: Batch size
        register: Whether to register results

    Returns:
        Tier1_5Result with all conditioning comparisons
    """
    # Get architecture from registry if not specified
    if architecture is None:
        registry = get_registry()
        if registry.tier1 is not None:
            architecture = registry.tier1.top_architectures[0]
        else:
            architecture = "unet"  # Default

    conditionings = conditionings or CONDITIONING_MECHANISMS

    print(f"\nTesting {len(conditionings)} conditioning mechanisms")
    print(f"Architecture: {architecture}")
    print(f"Seeds: {n_seeds}, Folds: {n_folds}, Epochs: {n_epochs}")

    N = X.shape[0]
    in_channels = X.shape[1]
    out_channels = y.shape[1]

    # Create conditioning embeddings if labels provided
    if cond_labels is not None:
        n_classes = int(cond_labels.max().item()) + 1
        cond_dim = 64
        # One-hot encode and project
        cond_embeddings = torch.zeros(N, cond_dim)
        for i in range(N):
            cond_embeddings[i, int(cond_labels[i]) % cond_dim] = 1.0
    else:
        # Random conditioning for testing
        cond_dim = 64
        cond_embeddings = torch.randn(N, cond_dim)

    # Create CV splits
    splits = create_data_splits(
        n_samples=N,
        n_folds=n_folds,
        holdout_fraction=0.0,
        seed=42,
    )

    all_results: List[ConditioningRunResult] = []

    for cond_type in conditionings:
        print(f"\n  Conditioning: {cond_type}")
        cond_r2s = []

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            for split in splits.cv_splits:
                # Prepare data
                X_train = X[split.train_indices]
                y_train = y[split.train_indices]
                cond_train = cond_embeddings[split.train_indices]

                X_val = X[split.val_indices]
                y_val = y[split.val_indices]
                cond_val = cond_embeddings[split.val_indices]

                train_dataset = TensorDataset(X_train, y_train, cond_train)
                val_dataset = TensorDataset(X_val, y_val, cond_val)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                # Create model
                model = None
                try:
                    model = create_conditioned_model(
                        arch_name=architecture,
                        conditioning=cond_type,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        cond_dim=cond_dim,
                        device=device,
                    )

                    start_time = time.time()
                    r2, mae = train_with_conditioning(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        n_epochs=n_epochs,
                        device=device,
                    )
                    elapsed = time.time() - start_time

                    all_results.append(ConditioningRunResult(
                        conditioning=cond_type,
                        seed=seed,
                        fold=split.fold_idx,
                        r2=r2,
                        mae=mae,
                        training_time=elapsed,
                    ))
                    cond_r2s.append(r2)

                except Exception as e:
                    print(f"    [seed={seed}, fold={split.fold_idx}] Error: {e}")
                    all_results.append(ConditioningRunResult(
                        conditioning=cond_type,
                        seed=seed,
                        fold=split.fold_idx,
                        r2=float("-inf"),
                        mae=float("inf"),
                        training_time=0,
                    ))

                finally:
                    if model is not None:
                        del model
                    gc.collect()
                    torch.cuda.empty_cache()

        if cond_r2s:
            mean_r2 = np.mean([r for r in cond_r2s if r > float("-inf")])
            print(f"    Mean R²: {mean_r2:.4f}")

    # Aggregate results
    cond_results = {}
    for cond_type in conditionings:
        cond_runs = [r for r in all_results if r.conditioning == cond_type]
        valid_r2s = [r.r2 for r in cond_runs if r.r2 > float("-inf")]
        valid_maes = [r.mae for r in cond_runs if r.mae < float("inf")]

        if valid_r2s:
            cond_results[cond_type] = ConditioningResult(
                conditioning=cond_type,
                r2_mean=float(np.mean(valid_r2s)),
                r2_std=float(np.std(valid_r2s, ddof=1)) if len(valid_r2s) > 1 else 0.0,
                mae_mean=float(np.mean(valid_maes)) if valid_maes else float("inf"),
                mae_std=float(np.std(valid_maes, ddof=1)) if len(valid_maes) > 1 else 0.0,
            )

    # Find best
    best_cond = max(cond_results.values(), key=lambda x: x.r2_mean)

    result = Tier1_5Result(
        architecture=architecture,
        results=cond_results,
        best_conditioning=best_cond.conditioning,
        best_r2_mean=best_cond.r2_mean,
        best_r2_std=best_cond.r2_std,
    )

    # Register
    if register:
        registry = get_registry(ARTIFACTS_DIR)
        registry.register_tier1_5(result)

    # Print summary
    print("\n" + "=" * 60)
    print("TIER 1.5 RESULTS: Conditioning Deep Dive")
    print("=" * 60)

    sorted_results = sorted(cond_results.values(), key=lambda x: x.r2_mean, reverse=True)
    for r in sorted_results:
        marker = " <-- BEST" if r.conditioning == best_cond.conditioning else ""
        print(f"  {r.conditioning:20s}: R² = {r.r2_mean:.4f} ± {r.r2_std:.4f}{marker}")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tier 1.5: Conditioning Deep Dive")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TIER 1.5: Conditioning Deep Dive")
    print("=" * 60)

    # Load data
    if args.dry_run:
        N, C, T = 100, 32, 500
        X = torch.randn(N, C, T)
        y = torch.randn(N, C, T)
        cond_labels = torch.randint(0, 5, (N,))
        conditionings = FAST_CONDITIONINGS
        n_epochs = 5
        n_seeds = 2
        n_folds = 2
    else:
        from data import prepare_data
        data = prepare_data()
        ob = data["ob"]
        pcx = data["pcx"]
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]

        idx = np.concatenate([train_idx, val_idx])
        X = torch.from_numpy(ob[idx]).float()
        y = torch.from_numpy(pcx[idx]).float()
        cond_labels = None  # Use random conditioning
        conditionings = CONDITIONING_MECHANISMS
        n_epochs = args.epochs
        n_seeds = args.seeds
        n_folds = args.folds

    run_tier1_5(
        X=X,
        y=y,
        device=device,
        architecture=args.arch,
        conditionings=conditionings,
        cond_labels=cond_labels,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_epochs=n_epochs,
    )


if __name__ == "__main__":
    main()
