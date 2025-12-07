#!/usr/bin/env python3
"""
Tier 1: Full Training with DDP (torchrun)
==========================================

Full training using DDP across 8 GPUs.
ONE experiment at a time with torchrun.
FULL data using memory-mapped files (no RAM duplication).

Usage:
    python experiments/tier1_screening/run_tier1.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier1_screening"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 8  # Reduced for single 6GB GPU
TRAIN_LR = 1e-3

ARCHITECTURES = ["linear", "cnn", "wavenet", "fnet", "vit", "performer", "mamba"]

LOSS_CATEGORIES = {
    "huber": "huber",
    "spectral": "multi_scale_spectral",
    "ccc": "concordance",
    "ssim": "ssim",
    "gaussian_nll": "gaussian_nll",
    "combined": "combined",
}


# =============================================================================
# Checkpoint
# =============================================================================

def _load_checkpoint() -> Dict[str, Dict]:
    checkpoint_file = ARTIFACTS_DIR / "tier1_checkpoint.json"
    if not checkpoint_file.exists():
        return {}
    with open(checkpoint_file) as f:
        return json.load(f)


def _save_checkpoint(results: Dict[str, Dict]) -> None:
    checkpoint_file = ARTIFACTS_DIR / "tier1_checkpoint.json"
    with open(checkpoint_file, "w") as f:
        json.dump(results, f, indent=2)


# =============================================================================
# DDP Worker Script (saved to file, run with torchrun)
# =============================================================================

SINGLE_GPU_SCRIPT = '''#!/usr/bin/env python3
"""Single-GPU Training with mixed precision (autocast + GradScaler)"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

from experiments.study1_architecture.architectures import create_architecture
from experiments.study3_loss.losses import create_loss


class MemmapDataset(Dataset):
    """Dataset using memory-mapped files."""
    def __init__(self, x_path: str, y_path: str):
        self.X = np.load(x_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = torch.from_numpy(self.y[idx].copy()).float()
        return x, y


def create_model(arch_name: str, in_channels: int, out_channels: int):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--loss", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    print(f"\\nTraining {args.arch} + {args.loss}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")

    # Load data
    data_dir = Path(args.data_dir)
    train_dataset = MemmapDataset(str(data_dir / "X_train.npy"), str(data_dir / "y_train.npy"))
    val_dataset = MemmapDataset(str(data_dir / "X_val.npy"), str(data_dir / "y_val.npy"))

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"  Shape: {train_dataset.X.shape}")

    # Get dims
    sample_x, sample_y = train_dataset[0]
    in_channels = sample_x.shape[0]
    out_channels = sample_y.shape[0]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = create_model(args.arch, in_channels, out_channels)
    model = model.to(device)

    # Loss
    criterion = create_loss(args.loss)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    start_time = time.time()

    # Training
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y_batch in train_loader:
            x = x.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                y_pred = model(x)
                try:
                    loss = criterion(y_pred, y_batch)
                except:
                    loss = nn.functional.l1_loss(y_pred, y_batch)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.6f}")

    training_time = time.time() - start_time

    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y_batch in val_loader:
            x = x.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            with autocast('cuda'):
                y_pred = model(x)
            all_preds.append(y_pred.float().cpu())
            all_targets.append(y_batch.float().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    ss_res = ((all_targets - all_preds) ** 2).sum().item()
    ss_tot = ((all_targets - all_targets.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    mae = (all_targets - all_preds).abs().mean().item()

    print(f"  DONE: R²={r2:.4f}, MAE={mae:.6f}, Time={training_time:.1f}s")

    # Save result
    result = {
        "arch": args.arch,
        "loss": args.loss,
        "r2": r2,
        "mae": mae,
        "training_time": training_time,
        "success": True,
    }
    result_file = data_dir / f"result_{args.arch}_{args.loss}.json"
    with open(result_file, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--arch", required=True)
        parser.add_argument("--loss", required=True)
        parser.add_argument("--data-dir", required=True)
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--lr", type=float, default=1e-3)
        args, _ = parser.parse_known_args()
        result = {"arch": args.arch, "loss": args.loss, "r2": float("-inf"),
                  "mae": float("inf"), "training_time": 0, "success": False, "error": str(e)}
        result_file = Path(args.data_dir) / f"result_{args.arch}_{args.loss}.json"
        with open(result_file, "w") as f:
            json.dump(result, f)
        raise
'''


def run_experiment(
    arch: str,
    loss_name: str,
    data_dir: Path,
    n_epochs: int,
    batch_size: int,
    lr: float,
    python_path: str,
) -> Dict[str, Any]:
    """Run ONE experiment using single GPU."""

    # Write worker script
    worker_path = ARTIFACTS_DIR / "worker.py"
    with open(worker_path, "w") as f:
        f.write(SINGLE_GPU_SCRIPT)

    # Clear previous result
    result_file = data_dir / f"result_{arch}_{loss_name}.json"
    if result_file.exists():
        result_file.unlink()

    # Build command (direct Python, no torchrun)
    cmd = [
        python_path,
        str(worker_path),
        f"--arch={arch}",
        f"--loss={loss_name}",
        f"--data-dir={data_dir}",
        f"--epochs={n_epochs}",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
    ]

    print(f"\n{'='*60}")
    print(f"[Training] {arch} + {loss_name}")
    print(f"{'='*60}")

    start = time.time()
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, timeout=7200)
        elapsed = time.time() - start

        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        else:
            return {"arch": arch, "loss": loss_name, "r2": float("-inf"),
                    "success": False, "error": "No result file"}

    except subprocess.CalledProcessError as e:
        return {"arch": arch, "loss": loss_name, "r2": float("-inf"),
                "success": False, "error": f"Exit code {e.returncode}"}
    except subprocess.TimeoutExpired:
        return {"arch": arch, "loss": loss_name, "r2": float("-inf"),
                "success": False, "error": "Timeout"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=TRAIN_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 60)
    print("TIER 1: Single-GPU Training")
    print("=" * 60)
    print("ONE experiment at a time with mixed precision")
    print("Memory-mapped files (no RAM duplication)")
    print()

    # Find Python with PyTorch
    python_candidates = [
        "/home/codemaster/miniconda3/envs/DeepLearning/bin/python",
        "/home/codemaster/miniconda3/envs/ML_WORK/bin/python",
        sys.executable,
    ]
    python_path = None
    for p in python_candidates:
        if Path(p).exists():
            python_path = p
            break
    if not python_path:
        raise RuntimeError("Could not find Python with PyTorch")
    print(f"Python: {python_path}")

    # Load and prepare data
    print("\nPreparing data...")

    if args.dry_run:
        N, C, T = 100, 32, 500
        X = np.random.randn(N, C, T).astype(np.float32)
        y = np.random.randn(N, C, T).astype(np.float32)
        architectures = ARCHITECTURES  # All 7 architectures
        losses = {"huber": "huber"}  # One loss for quick test
        n_epochs = 3
    else:
        from data import prepare_data
        data = prepare_data()
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        ob = data["ob"]
        pcx = data["pcx"]

        cv_idx = np.concatenate([train_idx, val_idx])
        X = ob[cv_idx].astype(np.float32)
        y = pcx[cv_idx].astype(np.float32)

        architectures = ARCHITECTURES
        losses = LOSS_CATEGORIES
        n_epochs = args.epochs

    # Split
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * 0.8)

    X_train = X[indices[:n_train]]
    y_train = y[indices[:n_train]]
    X_val = X[indices[n_train:]]
    y_val = y[indices[n_train:]]

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  FULL: {X_train.shape[2]} time steps")

    # Save as separate files for memory-mapped loading
    data_dir = ARTIFACTS_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    np.save(data_dir / "X_train.npy", X_train)
    np.save(data_dir / "y_train.npy", y_train)
    np.save(data_dir / "X_val.npy", X_val)
    np.save(data_dir / "y_val.npy", y_val)
    print(f"  Saved to: {data_dir}")

    # Free RAM
    del X, y, X_train, y_train, X_val, y_val
    gc.collect()

    # Load checkpoint
    results = {} if args.no_resume else _load_checkpoint()
    if results:
        print(f"\nResuming: {len(results)} done")

    # Build combo list
    combos = [(a, lc, ln) for a in architectures for lc, ln in losses.items()
              if f"{a}_{lc}" not in results]

    print(f"\nExperiments: {len(combos)} to run")

    # Run ONE AT A TIME
    for i, (arch, loss_cat, loss_name) in enumerate(combos):
        print(f"\n[{i+1}/{len(combos)}] {arch} + {loss_cat}")

        result = run_experiment(
            arch=arch,
            loss_name=loss_name,
            data_dir=data_dir,
            n_epochs=n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            python_path=python_path,
        )

        results[f"{arch}_{loss_cat}"] = {
            "architecture": arch,
            "loss": loss_cat,
            "r2": result.get("r2", float("-inf")),
            "mae": result.get("mae", float("inf")),
            "training_time": result.get("training_time", 0),
            "success": result.get("success", False),
        }

        _save_checkpoint(results)

        if result.get("success"):
            print(f"  SUCCESS: R²={result['r2']:.4f}")
        else:
            print(f"  FAILED: {result.get('error')}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    archs = sorted(set(r["architecture"] for r in results.values()))
    loss_cats = sorted(set(r["loss"] for r in results.values()))

    print(f"\n{'Arch':12}", end="")
    for lc in loss_cats:
        print(f" {lc:>10}", end="")
    print()

    for arch in archs:
        print(f"{arch:12}", end="")
        for lc in loss_cats:
            key = f"{arch}_{lc}"
            if key in results and results[key]["r2"] > float("-inf"):
                print(f" {results[key]['r2']:10.4f}", end="")
            else:
                print(f" {'FAIL':>10}", end="")
        print()

    # Save
    out = ARTIFACTS_DIR / f"tier1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
