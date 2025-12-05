#!/usr/bin/env python3
"""
Study 7: Component Ablation Analysis
=====================================

Systematic ablation study to justify each architectural design choice.
Essential for Nature Methods - reviewers will ask "why attention? why this norm?"

Ablation Categories:
1. Attention Type: cross_freq_v2 vs self-attention vs no_attention vs linear
2. Normalization: instance vs batch vs layer vs group
3. Conditioning: cross_attn_gated vs film vs concat vs add
4. Skip Connections: with vs without
5. Squeeze-Excitation: with vs without
6. Conv Type: modern (GELU+GroupNorm) vs standard (ReLU+BatchNorm)
7. Encoder Depth: 3 vs 4 vs 5 downsampling layers

Each ablation: Start from best config, change ONE component, measure impact.

Usage:
    # Full ablation study
    python experiments/study7_ablation/run_study7.py --seeds 5

    # Single category
    python experiments/study7_ablation/run_study7.py --category attention --seeds 3

    # Dry-run
    python experiments/study7_ablation/run_study7.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# =============================================================================
# Path Setup
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "study7_ablation"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Ablation Configuration
# =============================================================================

# Best baseline configuration (from Study 1 results)
BASELINE_CONFIG = {
    "architecture": "unet",
    "in_channels": 32,
    "out_channels": 32,
    "base_channels": 64,
    "n_downsample": 4,
    "use_attention": True,
    "attention_type": "cross_freq_v2",
    "norm_type": "instance",
    "cond_mode": "cross_attn_gated",
    "use_se": True,
    "conv_type": "modern",
    "dropout": 0.1,
    "n_odors": 7,
    "emb_dim": 128,
}

# Ablation variants for each component
ABLATION_VARIANTS = {
    "attention": {
        "description": "Attention mechanism comparison",
        "component": "attention_type",
        "variants": [
            {"name": "cross_freq_v2", "config": {"use_attention": True, "attention_type": "cross_freq_v2"}},
            {"name": "self_attention", "config": {"use_attention": True, "attention_type": "self"}},
            {"name": "linear_attention", "config": {"use_attention": True, "attention_type": "linear"}},
            {"name": "no_attention", "config": {"use_attention": False, "attention_type": None}},
        ],
        "justification": "Cross-frequency attention captures spectral relationships between OB and PCx signals",
    },
    "normalization": {
        "description": "Normalization layer comparison",
        "component": "norm_type",
        "variants": [
            {"name": "instance_norm", "config": {"norm_type": "instance"}},
            {"name": "batch_norm", "config": {"norm_type": "batch"}},
            {"name": "layer_norm", "config": {"norm_type": "layer"}},
            {"name": "group_norm", "config": {"norm_type": "group"}},
        ],
        "justification": "Instance norm preserves per-sample statistics important for neural signals",
    },
    "conditioning": {
        "description": "Conditioning mechanism comparison",
        "component": "cond_mode",
        "variants": [
            {"name": "cross_attn_gated", "config": {"cond_mode": "cross_attn_gated"}},
            {"name": "film", "config": {"cond_mode": "film"}},
            {"name": "concat", "config": {"cond_mode": "concat"}},
            {"name": "add", "config": {"cond_mode": "add"}},
        ],
        "justification": "Gated cross-attention allows dynamic, odor-specific modulation of features",
    },
    "skip_connections": {
        "description": "Skip connection ablation",
        "component": "use_skip",
        "variants": [
            {"name": "with_skip", "config": {"use_skip": True}},
            {"name": "no_skip", "config": {"use_skip": False}},
        ],
        "justification": "Skip connections preserve fine temporal details during encoding",
    },
    "squeeze_excitation": {
        "description": "Squeeze-and-Excitation module ablation",
        "component": "use_se",
        "variants": [
            {"name": "with_se", "config": {"use_se": True}},
            {"name": "no_se", "config": {"use_se": False}},
        ],
        "justification": "SE blocks enable channel-wise attention for frequency band importance",
    },
    "conv_type": {
        "description": "Convolution block type comparison",
        "component": "conv_type",
        "variants": [
            {"name": "modern", "config": {"conv_type": "modern"}},  # GELU + GroupNorm
            {"name": "standard", "config": {"conv_type": "standard"}},  # ReLU + BatchNorm
            {"name": "residual", "config": {"conv_type": "residual"}},
        ],
        "justification": "Modern blocks with GELU activation provide smoother gradients",
    },
    "encoder_depth": {
        "description": "Encoder depth comparison",
        "component": "n_downsample",
        "variants": [
            {"name": "depth_3", "config": {"n_downsample": 3}},
            {"name": "depth_4", "config": {"n_downsample": 4}},
            {"name": "depth_5", "config": {"n_downsample": 5}},
        ],
        "justification": "Depth 4 balances receptive field size with computational efficiency",
    },
}

# Categories to run in full ablation
ABLATION_CATEGORIES = list(ABLATION_VARIANTS.keys())


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class AblationResult:
    """Result of a single ablation experiment."""
    category: str
    variant_name: str
    config: Dict[str, Any]
    seed: int
    metrics: Dict[str, float]
    training_time: float


@dataclass
class CategoryResults:
    """Results for an entire ablation category."""
    category: str
    description: str
    justification: str
    results: List[AblationResult] = field(default_factory=list)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get mean/std metrics per variant."""
        from collections import defaultdict

        variant_metrics = defaultdict(lambda: defaultdict(list))

        for r in self.results:
            for metric, value in r.metrics.items():
                variant_metrics[r.variant_name][metric].append(value)

        summary = {}
        for variant, metrics in variant_metrics.items():
            summary[variant] = {
                f"{m}_mean": np.mean(vals) for m, vals in metrics.items()
            }
            summary[variant].update({
                f"{m}_std": np.std(vals) for m, vals in metrics.items()
            })

        return summary


# =============================================================================
# Model Creation
# =============================================================================

def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create model from configuration."""
    try:
        from models import CondUNet1D

        # Merge with baseline
        full_config = {**BASELINE_CONFIG, **config}

        model = CondUNet1D(
            in_channels=full_config["in_channels"],
            out_channels=full_config["out_channels"],
            base=full_config["base_channels"],
            n_odors=full_config["n_odors"],
            emb_dim=full_config["emb_dim"],
            dropout=full_config["dropout"],
            use_attention=full_config["use_attention"],
            attention_type=full_config.get("attention_type", "cross_freq_v2"),
            norm_type=full_config["norm_type"],
            cond_mode=full_config["cond_mode"],
            use_spectral_shift=False,
            n_downsample=full_config["n_downsample"],
            conv_type=full_config["conv_type"],
            use_se=full_config["use_se"],
        ).to(device)

        return model

    except ImportError:
        # Fallback for testing
        print("Warning: Using dummy model for testing")
        return nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
        ).to(device)


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 80,
    patience: int = 8,
    lr: float = 1e-3,
) -> Tuple[Dict[str, float], float]:
    """Train model and return validation metrics.

    Returns:
        (metrics_dict, training_time_seconds)
    """
    import time
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    criterion = nn.L1Loss()

    best_val_loss = float("inf")
    best_metrics = {}
    patience_counter = 0

    start_time = time.time()

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            # Handle different batch formats
            if len(batch) == 3:
                x, y, odor = batch
            elif len(batch) == 2:
                x, y = batch
                odor = torch.zeros(x.size(0), dtype=torch.long, device=device)
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")

            x, y = x.to(device), y.to(device)
            if hasattr(model, 'n_odors'):
                odor = odor.to(device)

            optimizer.zero_grad()

            # Forward pass
            if hasattr(model, 'n_odors') and model.n_odors > 0:
                y_pred = model(x, odor)
            else:
                y_pred = model(x)

            loss = criterion(y_pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x, y, odor = batch
                elif len(batch) == 2:
                    x, y = batch
                    odor = torch.zeros(x.size(0), dtype=torch.long, device=device)
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)} elements")

                x, y = x.to(device), y.to(device)
                if hasattr(model, 'n_odors'):
                    odor = odor.to(device)

                if hasattr(model, 'n_odors') and model.n_odors > 0:
                    y_pred = model(x, odor)
                else:
                    y_pred = model(x)

                val_preds.append(y_pred.cpu())
                val_targets.append(y.cpu())

        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)

        # Compute metrics
        val_loss = criterion(val_preds, val_targets).item()

        # R² score
        ss_res = ((val_targets - val_preds) ** 2).sum()
        ss_tot = ((val_targets - val_targets.mean()) ** 2).sum()
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()

        # Pearson correlation
        pred_flat = val_preds.flatten()
        target_flat = val_targets.flatten()
        pred_centered = pred_flat - pred_flat.mean()
        target_centered = target_flat - target_flat.mean()
        corr = (pred_centered * target_centered).sum() / (
            pred_centered.norm() * target_centered.norm() + 1e-8
        )
        corr = corr.item()

        current_metrics = {
            "val_loss": val_loss,
            "r2": r2,
            "pearson_corr": corr,
        }

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = current_metrics
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    training_time = time.time() - start_time

    return best_metrics, training_time


# =============================================================================
# Ablation Runner
# =============================================================================

def run_single_ablation(
    category: str,
    variant: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    seed: int,
    n_epochs: int = 80,
    patience: int = 8,
) -> AblationResult:
    """Run a single ablation experiment."""

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create model with variant config
    config = variant["config"]
    model = create_model(config, device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    {variant['name']}: {n_params:,} params")

    # Train and evaluate
    metrics, training_time = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=n_epochs,
        patience=patience,
    )

    # Add parameter count to metrics
    metrics["n_params"] = n_params

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return AblationResult(
        category=category,
        variant_name=variant["name"],
        config=config,
        seed=seed,
        metrics=metrics,
        training_time=training_time,
    )


def run_category_ablation(
    category: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_seeds: int = 5,
    n_epochs: int = 80,
    patience: int = 8,
    dry_run: bool = False,
) -> CategoryResults:
    """Run ablation for an entire category across multiple seeds."""

    ablation_info = ABLATION_VARIANTS[category]

    print(f"\n{'='*60}")
    print(f"Ablation Category: {category}")
    print(f"Description: {ablation_info['description']}")
    print(f"{'='*60}")

    category_results = CategoryResults(
        category=category,
        description=ablation_info["description"],
        justification=ablation_info["justification"],
    )

    n_seeds_actual = 1 if dry_run else n_seeds
    n_epochs_actual = 3 if dry_run else n_epochs

    for variant in ablation_info["variants"]:
        print(f"\n  Variant: {variant['name']}")

        for seed in range(n_seeds_actual):
            print(f"    Seed {seed + 1}/{n_seeds_actual}...", end=" ")

            result = run_single_ablation(
                category=category,
                variant=variant,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                seed=42 + seed,
                n_epochs=n_epochs_actual,
                patience=patience,
            )

            category_results.results.append(result)
            print(f"R²={result.metrics['r2']:.4f}")

    return category_results


def run_full_ablation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    categories: Optional[List[str]] = None,
    n_seeds: int = 5,
    n_epochs: int = 80,
    patience: int = 8,
    dry_run: bool = False,
) -> Dict[str, CategoryResults]:
    """Run full ablation study across all categories."""

    categories = categories or ABLATION_CATEGORIES
    all_results = {}

    for category in categories:
        results = run_category_ablation(
            category=category,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            n_seeds=n_seeds,
            n_epochs=n_epochs,
            patience=patience,
            dry_run=dry_run,
        )
        all_results[category] = results

    return all_results


# =============================================================================
# Reporting
# =============================================================================

def print_ablation_summary(all_results: Dict[str, CategoryResults]) -> None:
    """Print formatted ablation summary."""

    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)

    for category, results in all_results.items():
        print(f"\n{category.upper()}: {results.description}")
        print("-" * 50)

        summary = results.get_summary()

        # Sort by R² mean
        sorted_variants = sorted(
            summary.items(),
            key=lambda x: x[1].get("r2_mean", 0),
            reverse=True
        )

        for rank, (variant, metrics) in enumerate(sorted_variants, 1):
            r2_mean = metrics.get("r2_mean", 0)
            r2_std = metrics.get("r2_std", 0)
            print(f"  {rank}. {variant:20s}: R² = {r2_mean:.4f} ± {r2_std:.4f}")

        print(f"\n  Justification: {results.justification}")

    print("\n" + "=" * 70)


def save_ablation_results(
    all_results: Dict[str, CategoryResults],
    output_dir: Path,
) -> Path:
    """Save ablation results to JSON."""

    output = {
        "timestamp": datetime.now().isoformat(),
        "baseline_config": BASELINE_CONFIG,
        "categories": {},
    }

    for category, results in all_results.items():
        output["categories"][category] = {
            "description": results.description,
            "justification": results.justification,
            "summary": results.get_summary(),
            "raw_results": [
                {
                    "variant": r.variant_name,
                    "seed": r.seed,
                    "metrics": r.metrics,
                    "training_time": r.training_time,
                }
                for r in results.results
            ],
        }

    output_path = output_dir / f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    return output_path


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Study 7: Component Ablation Analysis")
    parser.add_argument("--category", type=str, choices=ABLATION_CATEGORIES,
                        help="Single category to ablate")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per variant")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--dry-run", action="store_true", help="Quick test run")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    try:
        from data import prepare_data, create_dataloaders
        data = prepare_data()
        loaders = create_dataloaders(data, batch_size=16)
        train_loader = loaders["train"]
        val_loader = loaders["val"]
    except Exception as e:
        print(f"Warning: Using dummy data for testing (reason: {type(e).__name__}: {e})")
        from torch.utils.data import TensorDataset, DataLoader
        n_samples = 50 if args.dry_run else 500
        X = torch.randn(n_samples, 32, 1000)
        y = torch.randn(n_samples, 32, 1000)
        odor = torch.randint(0, 7, (n_samples,))
        dataset = TensorDataset(X, y, odor)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16)

    # Run ablation
    categories = [args.category] if args.category else None

    all_results = run_full_ablation(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        categories=categories,
        n_seeds=args.seeds,
        n_epochs=args.epochs,
        patience=args.patience,
        dry_run=args.dry_run,
    )

    # Print and save results
    print_ablation_summary(all_results)
    save_ablation_results(all_results, ARTIFACTS_DIR)

    print("\n" + "=" * 60)
    print("Study 7 Complete: Component Ablation Analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()
