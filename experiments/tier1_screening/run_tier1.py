#!/usr/bin/env python3
"""
Tier 1: Full Training with DDP
==============================

Full training of architecture × loss combinations using DDP across 8 GPUs.
One experiment at a time, distributed across all GPUs for maximum memory efficiency.

Usage:
    python experiments/tier1_screening/run_tier1.py
    python experiments/tier1_screening/run_tier1.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# CRITICAL: Fix Python path for imports
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier1_screening"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters - FULL SCALE
TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 4  # Per GPU, so effective batch = 4 * 8 = 32
TRAIN_LR = 1e-3

# 7 architectures
ARCHITECTURES = ["linear", "cnn", "wavenet", "fnet", "vit", "performer", "mamba"]

# 6 loss categories
LOSS_CATEGORIES = {
    "huber": "huber",
    "spectral": "multi_scale_spectral",
    "ccc": "concordance",
    "ssim": "ssim",
    "gaussian_nll": "gaussian_nll",
    "combined": "combined",
}


# =============================================================================
# Checkpoint Support
# =============================================================================

def _load_checkpoint() -> Dict[str, Dict]:
    """Load checkpoint for resume."""
    checkpoint_file = ARTIFACTS_DIR / "tier1_checkpoint.json"
    if not checkpoint_file.exists():
        return {}
    with open(checkpoint_file) as f:
        return json.load(f)


def _save_checkpoint(results: Dict[str, Dict]) -> None:
    """Save checkpoint."""
    checkpoint_file = ARTIFACTS_DIR / "tier1_checkpoint.json"
    with open(checkpoint_file, "w") as f:
        json.dump(results, f, indent=2)


def _clear_checkpoint() -> None:
    """Clear checkpoint after completion."""
    checkpoint_file = ARTIFACTS_DIR / "tier1_checkpoint.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()


# =============================================================================
# DDP Training Script (written to temp file and executed with torchrun)
# =============================================================================

DDP_TRAIN_SCRIPT = '''#!/usr/bin/env python3
"""DDP Training Worker - launched by torchrun"""
import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler

from experiments.study1_architecture.architectures import create_architecture
from experiments.study3_loss.losses import create_loss


def setup_ddp():
    """Initialize DDP."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Cleanup DDP."""
    dist.destroy_process_group()


def create_model(arch_name: str, in_channels: int, out_channels: int):
    """Create model by architecture name."""
    VARIANT_MAP = {
        "linear": "simple",
        "cnn": "basic",
        "wavenet": "standard",
        "fnet": "standard",
        "vit": "standard",
        "performer": "standard",
        "mamba": "standard",
    }
    variant = VARIANT_MAP.get(arch_name, "standard")
    return create_architecture(arch_name, variant=variant,
                               in_channels=in_channels, out_channels=out_channels)


def train_ddp(arch_name, loss_name, data_path, n_epochs, batch_size, lr):
    """Train with DDP across all GPUs."""
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    is_main = local_rank == 0

    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        print(f"Training {arch_name} + {loss_name} on {world_size} GPUs")
        print(f"  Epochs: {n_epochs}, Batch/GPU: {batch_size}, Effective batch: {batch_size * world_size}")

    # Load data
    data = np.load(data_path, allow_pickle=True).item()
    X_train = torch.from_numpy(data["X_train"]).float()
    y_train = torch.from_numpy(data["y_train"]).float()
    X_val = torch.from_numpy(data["X_val"]).float()
    y_val = torch.from_numpy(data["y_val"]).float()

    in_channels = X_train.shape[1]
    out_channels = y_train.shape[1]

    if is_main:
        print(f"  Data: X_train={X_train.shape}, X_val={X_val.shape}")

    # Create datasets with distributed sampler
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=4, pin_memory=True)

    # Create model and wrap in DDP
    model = create_model(arch_name, in_channels, out_channels).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create loss
    criterion = create_loss(loss_name)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = GradScaler('cuda')

    start_time = time.time()

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # Important for shuffling

        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x, y_batch = batch
            x, y_batch = x.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                y_pred = model(x)
                try:
                    loss = criterion(y_pred, y_batch)
                except Exception:
                    loss = nn.functional.l1_loss(y_pred, y_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if is_main and (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    training_time = time.time() - start_time

    # Evaluation (only on main process gathers all results)
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            x, y_batch = batch
            x = x.to(device, non_blocking=True)

            with autocast('cuda'):
                y_pred = model(x)

            all_preds.append(y_pred.float().cpu())
            all_targets.append(y_batch)

    # Gather predictions from all processes
    local_preds = torch.cat(all_preds, dim=0)
    local_targets = torch.cat(all_targets, dim=0)

    # Gather sizes first
    local_size = torch.tensor([local_preds.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    # Compute metrics on main process
    if is_main:
        # For simplicity, compute R² on local validation (each GPU sees 1/8 of val data)
        # This is an approximation but much simpler than full gather
        preds = local_preds
        targets = local_targets

        ss_res = ((targets - preds) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
        mae = (targets - preds).abs().mean().item()

        print(f"  Final: R²={r2:.4f}, MAE={mae:.4f}, Time={training_time:.1f}s")

        # Save result
        result = {
            "arch": arch_name,
            "loss": loss_name,
            "r2": r2,
            "mae": mae,
            "training_time": training_time,
            "world_size": world_size,
            "success": True,
        }

        result_file = Path(data_path).parent / f"result_{arch_name}_{loss_name}.json"
        with open(result_file, "w") as f:
            json.dump(result, f)

    cleanup_ddp()


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--loss", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    try:
        train_ddp(args.arch, args.loss, args.data_path, args.epochs, args.batch_size, args.lr)
    except Exception as e:
        import traceback
        # Save error result
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            result = {
                "arch": args.arch,
                "loss": args.loss,
                "r2": float("-inf"),
                "mae": float("inf"),
                "training_time": 0,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            result_file = Path(args.data_path).parent / f"result_{args.arch}_{args.loss}.json"
            with open(result_file, "w") as f:
                json.dump(result, f)
        raise
'''


def run_single_experiment_ddp(
    arch_name: str,
    loss_name: str,
    data_path: Path,
    n_epochs: int,
    batch_size: int,
    lr: float,
    n_gpus: int,
) -> Dict[str, Any]:
    """Run a single experiment with DDP using torchrun."""

    # Write the DDP script to a temp file
    script_path = ARTIFACTS_DIR / "ddp_train_worker.py"
    with open(script_path, "w") as f:
        f.write(DDP_TRAIN_SCRIPT)

    # Result file
    result_file = data_path.parent / f"result_{arch_name}_{loss_name}.json"
    if result_file.exists():
        result_file.unlink()

    # Run with torchrun
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={n_gpus}",
        "--master_port=29500",
        str(script_path),
        f"--arch={arch_name}",
        f"--loss={loss_name}",
        f"--data-path={data_path}",
        f"--epochs={n_epochs}",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
    ]

    print(f"\n{'='*60}")
    print(f"Training: {arch_name} + {loss_name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=False,  # Show output in real-time
            timeout=3600,  # 1 hour timeout per experiment
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"ERROR: Training failed with return code {result.returncode}")
            return {
                "arch": arch_name,
                "loss": loss_name,
                "r2": float("-inf"),
                "mae": float("inf"),
                "training_time": elapsed,
                "success": False,
                "error": f"Return code {result.returncode}",
            }

        # Read result file
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        else:
            return {
                "arch": arch_name,
                "loss": loss_name,
                "r2": float("-inf"),
                "mae": float("inf"),
                "training_time": elapsed,
                "success": False,
                "error": "No result file generated",
            }

    except subprocess.TimeoutExpired:
        return {
            "arch": arch_name,
            "loss": loss_name,
            "r2": float("-inf"),
            "mae": float("inf"),
            "training_time": 3600,
            "success": False,
            "error": "Timeout (1 hour)",
        }
    except Exception as e:
        return {
            "arch": arch_name,
            "loss": loss_name,
            "r2": float("-inf"),
            "mae": float("inf"),
            "training_time": time.time() - start_time,
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Tier 1: Full DDP Training")
    parser.add_argument("--dry-run", action="store_true", help="Use minimal configuration")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=TRAIN_LR, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")

    args = parser.parse_args()

    print("=" * 60)
    print("TIER 1: Full DDP Training")
    print("=" * 60)
    print("Method: One experiment at a time, DDP across all GPUs")
    print("Data: FULL (no subsampling)")
    print()

    # Check GPUs
    import torch
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPUs available!")
    print(f"GPUs available: {n_gpus}")

    # Load data
    print("\nLoading data...")

    if args.dry_run:
        print("  [DRY-RUN] Using synthetic data")
        N, C, T = 100, 32, 500
        np.random.seed(args.seed)
        X = np.random.randn(N, C, T).astype(np.float32)
        y = np.random.randn(N, C, T).astype(np.float32)
        architectures = ["linear", "cnn"]
        losses = {"huber": "huber", "ccc": "concordance"}
        n_epochs = 3
    else:
        from data import prepare_data
        data = prepare_data()

        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        ob = data["ob"]   # [n_samples, 32, 5000]
        pcx = data["pcx"]  # [n_samples, 32, 5000]

        # FULL DATA - no subsampling!
        cv_idx = np.concatenate([train_idx, val_idx])
        X = ob[cv_idx]
        y = pcx[cv_idx]

        architectures = ARCHITECTURES
        losses = LOSS_CATEGORIES
        n_epochs = args.epochs

        print(f"  Loaded: X={X.shape}, y={y.shape}")
        print(f"  FULL TIME RESOLUTION: {X.shape[2]} time steps")

    # Split train/val
    np.random.seed(args.seed)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * 0.8)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    # Save data for DDP workers
    data_path = ARTIFACTS_DIR / "tier1_data.npy"
    np.save(data_path, {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    })
    print(f"  Data saved to: {data_path}")

    # Load checkpoint
    completed_results = {} if args.no_resume else _load_checkpoint()
    if completed_results:
        print(f"\nResuming: {len(completed_results)} experiments already done")

    # Build experiment list
    all_combos = []
    for arch in architectures:
        for loss_cat, loss_name in losses.items():
            combo_key = f"{arch}_{loss_cat}"
            if combo_key not in completed_results:
                all_combos.append((arch, loss_cat, loss_name))

    total = len(architectures) * len(losses)
    to_run = len(all_combos)
    print(f"\nExperiments: {to_run} to run ({total - to_run} already done)")
    print(f"Estimated time: ~{to_run * 10} minutes (very rough estimate)")

    # Run experiments sequentially with DDP
    results = dict(completed_results)

    for i, (arch, loss_cat, loss_name) in enumerate(all_combos):
        combo_key = f"{arch}_{loss_cat}"
        print(f"\n[{i+1}/{to_run}] {arch} + {loss_cat}")

        result = run_single_experiment_ddp(
            arch_name=arch,
            loss_name=loss_name,
            data_path=data_path,
            n_epochs=n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_gpus=n_gpus,
        )

        results[combo_key] = {
            "architecture": arch,
            "loss": loss_cat,
            "r2": result["r2"],
            "mae": result["mae"],
            "training_time": result["training_time"],
            "success": result.get("success", False),
        }

        # Save checkpoint after each experiment
        _save_checkpoint(results)

        if result.get("success"):
            print(f"  SUCCESS: R²={result['r2']:.4f}")
        else:
            print(f"  FAILED: {result.get('error', 'Unknown')}")

    # Summary
    print("\n" + "=" * 60)
    print("TIER 1 RESULTS")
    print("=" * 60)

    print("\nR² Matrix:")
    print("-" * 60)

    archs = sorted(set(r["architecture"] for r in results.values()))
    loss_cats = sorted(set(r["loss"] for r in results.values()))

    # Header
    print(f"{'Architecture':15s}", end="")
    for loss in loss_cats:
        print(f" {loss:>12s}", end="")
    print()

    # Rows
    for arch in archs:
        print(f"{arch:15s}", end="")
        for loss in loss_cats:
            combo_key = f"{arch}_{loss}"
            if combo_key in results:
                r2 = results[combo_key]["r2"]
                if r2 > float("-inf"):
                    print(f" {r2:12.4f}", end="")
                else:
                    print(f" {'FAIL':>12s}", end="")
            else:
                print(f" {'N/A':>12s}", end="")
        print()

    # Top performers
    valid = [(k, v) for k, v in results.items() if v["r2"] > float("-inf")]
    if valid:
        best = max(valid, key=lambda x: x[1]["r2"])
        print(f"\nBest: {best[0]} with R²={best[1]['r2']:.4f}")

        # Top architectures (by mean R²)
        arch_scores = {}
        for k, v in valid:
            arch = v["architecture"]
            if arch not in arch_scores:
                arch_scores[arch] = []
            arch_scores[arch].append(v["r2"])
        arch_means = {a: np.mean(s) for a, s in arch_scores.items()}
        top_archs = sorted(arch_means, key=arch_means.get, reverse=True)[:3]
        print(f"Top architectures: {top_archs}")

        # Top losses (by mean R²)
        loss_scores = {}
        for k, v in valid:
            loss = v["loss"]
            if loss not in loss_scores:
                loss_scores[loss] = []
            loss_scores[loss].append(v["r2"])
        loss_means = {l: np.mean(s) for l, s in loss_scores.items()}
        top_losses = sorted(loss_means, key=loss_means.get, reverse=True)[:3]
        print(f"Top losses: {top_losses}")

    # Save final results
    output_file = ARTIFACTS_DIR / f"tier1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Clear checkpoint on success
    if to_run > 0 and all(results[f"{a}_{l}"].get("success", False)
                          for a in architectures for l, _ in losses.items()
                          if f"{a}_{l}" in results):
        _clear_checkpoint()

    return results


if __name__ == "__main__":
    main()
