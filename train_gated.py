"""Training script for GatedTranslator.

Learns when brain regions are communicating vs when translation is noise.

Usage:
    # Train on olfactory data
    python train_gated.py --dataset olfactory --epochs 50

    # Train on DANDI data
    python train_gated.py --dataset dandi --epochs 50

    # Adjust sparsity
    python train_gated.py --sparsity-target 0.2 --sparsity-weight 0.05
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import (
    CondUNet1D,
    GatedTranslator,
    create_gated_translator,
)
from data import (
    prepare_data,
    prepare_dandi_data,
    create_dataloaders,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GatedTranslator")
    parser.add_argument("--dataset", type=str, default="olfactory",
                        choices=["olfactory", "dandi"],
                        help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--sparsity-weight", type=float, default=0.01,
                        help="Weight for sparsity penalty")
    parser.add_argument("--sparsity-target", type=float, default=0.3,
                        help="Target gate activation rate (0-1)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to save/load checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N batches")
    return parser.parse_args()


def load_data(dataset: str, batch_size: int) -> Tuple:
    """Load dataset and create dataloaders."""
    print(f"Loading {dataset} dataset...")

    if dataset == "olfactory":
        data = prepare_data()
        train_loader, val_loader, _ = create_dataloaders(
            X=data["X"],
            y=data["y"],
            train_idx=data["train_idx"],
            val_idx=data["val_idx"],
            test_idx=data["test_idx"],
            batch_size=batch_size,
        )
        in_channels = data["X"].shape[1]
        out_channels = data["y"].shape[1]

    elif dataset == "dandi":
        data = prepare_dandi_data(
            source_region="amygdala",
            target_region="hippocampus",
            window_size=5000,
            stride=2500,
        )
        train_loader = data["train_loader"]
        val_loader = data["val_loader"]
        in_channels = data["n_source_channels"]
        out_channels = data["n_target_channels"]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"  In channels: {in_channels}, Out channels: {out_channels}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    return train_loader, val_loader, in_channels, out_channels


def create_model(
    in_channels: int,
    out_channels: int,
    sparsity_weight: float,
    sparsity_target: float,
    device: torch.device,
) -> GatedTranslator:
    """Create GatedTranslator with CondUNet base."""

    # Base translator
    base_model = CondUNet1D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
    )

    # Wrap with gating
    model = create_gated_translator(
        translator=base_model,
        in_channels=in_channels,
        out_channels=out_channels,
        sparsity_weight=sparsity_weight,
        sparsity_target=sparsity_target,
        gate_hidden_channels=64,
        gate_kernel_size=31,
        gate_num_layers=3,
        baseline_hidden_channels=64,
        baseline_kernel_size=15,
    )

    return model.to(device)


def train_epoch(
    model: GatedTranslator,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int = 10,
) -> Dict[str, float]:
    """Train one epoch."""
    model.train()
    model.enable_logging(max_history=50)

    total_loss = 0.0
    total_recon = 0.0
    total_sparsity = 0.0
    total_gate_mean = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
            else:
                x, y = batch[0], batch[1]
        else:
            x, y = batch["source"], batch["target"]

        x = x.to(device)
        y = y.to(device)

        # Forward with components
        y_pred, components = model(x, return_components=True)
        gate = components["gate"]

        # Compute loss
        loss, loss_dict = model.compute_loss(y_pred, y, gate)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        total_recon += loss_dict["reconstruction"].item()
        total_sparsity += loss_dict["sparsity"].item()
        total_gate_mean += loss_dict["gate_mean"].item()
        n_batches += 1

        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "gate": f"{loss_dict['gate_mean'].item():.3f}",
            })

    model.disable_logging()

    return {
        "loss": total_loss / n_batches,
        "reconstruction": total_recon / n_batches,
        "sparsity": total_sparsity / n_batches,
        "gate_mean": total_gate_mean / n_batches,
    }


@torch.no_grad()
def validate(
    model: GatedTranslator,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    model.enable_logging(max_history=100)

    total_loss = 0.0
    total_recon = 0.0
    total_gate_mean = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Validating"):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
            else:
                x, y = batch[0], batch[1]
        else:
            x, y = batch["source"], batch["target"]

        x = x.to(device)
        y = y.to(device)

        y_pred, components = model(x, return_components=True)
        gate = components["gate"]

        loss, loss_dict = model.compute_loss(y_pred, y, gate)

        total_loss += loss.item()
        total_recon += loss_dict["reconstruction"].item()
        total_gate_mean += loss_dict["gate_mean"].item()
        n_batches += 1

    # Analyze sparsity
    sparsity = model.analyze_sparsity()
    model.disable_logging()

    return {
        "loss": total_loss / n_batches,
        "reconstruction": total_recon / n_batches,
        "gate_mean": total_gate_mean / n_batches,
        **sparsity,
    }


def main():
    args = parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, in_channels, out_channels = load_data(
        args.dataset, args.batch_size
    )

    # Create model
    model = create_model(
        in_channels=in_channels,
        out_channels=out_channels,
        sparsity_weight=args.sparsity_weight,
        sparsity_target=args.sparsity_target,
        device=device,
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Sparsity target: {args.sparsity_target} (weight: {args.sparsity_weight})")

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float("inf")

    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.log_interval
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Recon: {train_metrics['reconstruction']:.4f}, "
              f"Gate: {train_metrics['gate_mean']:.3f}")

        # Validate
        val_metrics = validate(model, val_loader, device)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Recon: {val_metrics['reconstruction']:.4f}, "
              f"Gate: {val_metrics['gate_mean']:.3f}")
        print(f"        Sparse enough: {val_metrics.get('is_sparse_enough', 'N/A')}, "
              f"ON frac (>0.5): {val_metrics.get('sparsity_on_frac', 0):.3f}")

        scheduler.step()

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            if args.checkpoint:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "args": vars(args),
                }, args.checkpoint)
                print(f"        Saved checkpoint to {args.checkpoint}")

    print(f"\n{'='*60}")
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
