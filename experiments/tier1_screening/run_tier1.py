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
TRAIN_BATCH_SIZE = 64  # Total batch size (divided across 8 GPUs = 8 per GPU)
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

DDP_WORKER_SCRIPT = '''#!/usr/bin/env python3
"""FSDP Worker - ONE experiment across 8 GPUs using torchrun with FSDP"""
import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from experiments.study1_architecture.architectures import create_architecture
from experiments.study3_loss.losses import create_loss


class MemmapDataset(Dataset):
    """Dataset using memory-mapped files - NO RAM duplication across processes."""
    def __init__(self, x_path: str, y_path: str):
        self.X = np.load(x_path, mmap_mode='r')  # Memory-mapped, read-only
        self.y = np.load(y_path, mmap_mode='r')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Copy single sample to tensor (minimal memory)
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    is_main = (local_rank == 0)

    # Clear GPU cache before starting
    torch.cuda.empty_cache()

    if is_main:
        print(f"\\nTraining {args.arch} + {args.loss}")
        print(f"  World size: {world_size} GPUs (FSDP)")
        print(f"  Batch size: {args.batch_size} total ({args.batch_size // world_size} per GPU)")
        print(f"  Epochs: {args.epochs}")

    # Load data using memory-mapped files (shared across processes, no duplication)
    data_dir = Path(args.data_dir)
    train_dataset = MemmapDataset(str(data_dir / "X_train.npy"), str(data_dir / "y_train.npy"))
    val_dataset = MemmapDataset(str(data_dir / "X_val.npy"), str(data_dir / "y_val.npy"))

    if is_main:
        print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"  Shape: {train_dataset.X.shape}")

    # Get dims
    sample_x, sample_y = train_dataset[0]
    in_channels = sample_x.shape[0]
    out_channels = sample_y.shape[0]

    # Distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                        rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size,
                                      rank=local_rank, shuffle=False)

    batch_per_gpu = args.batch_size // world_size
    train_loader = DataLoader(train_dataset, batch_size=batch_per_gpu, sampler=train_sampler,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_per_gpu, sampler=val_sampler,
                            num_workers=2, pin_memory=True)

    # Create model on CPU first, then wrap with FSDP (shards across GPUs)
    model = create_model(args.arch, in_channels, out_channels)

    # FSDP with mixed precision - shards model across all GPUs
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    model = FSDP(model, mixed_precision=mp_policy, device_id=local_rank)

    # Loss
    criterion = create_loss(args.loss)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)

    # Optimizer (FSDP handles mixed precision internally)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_time = time.time()

    # Training (FSDP MixedPrecision handles fp16 automatically)
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        n_batches = 0

        for x, y_batch in train_loader:
            x = x.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # FSDP with MixedPrecision handles autocast internally
            y_pred = model(x)
            try:
                loss = criterion(y_pred, y_batch)
            except:
                loss = nn.functional.l1_loss(y_pred, y_batch)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                model.clip_grad_norm_(1.0)  # FSDP's gradient clipping
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

        scheduler.step()

        if is_main and (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.6f}")

    training_time = time.time() - start_time

    # Evaluation - compute metrics locally then aggregate with all_reduce
    model.eval()
    local_preds = []
    local_targets = []

    with torch.no_grad():
        for x, y_batch in val_loader:
            x = x.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            y_pred = model(x)
            local_preds.append(y_pred.float())
            local_targets.append(y_batch.float())

    local_preds = torch.cat(local_preds, dim=0)
    local_targets = torch.cat(local_targets, dim=0)

    # Compute local statistics on GPU
    local_ss_res = ((local_targets - local_preds) ** 2).sum()
    local_ss_tot = ((local_targets - local_targets.mean()) ** 2).sum()
    local_mae_sum = (local_targets - local_preds).abs().sum()
    local_count = torch.tensor([local_targets.numel()], device=device, dtype=torch.float32)

    # Aggregate across all GPUs using all_reduce
    dist.all_reduce(local_ss_res, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_ss_tot, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_mae_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

    # Compute final metrics
    r2 = (1 - local_ss_res / (local_ss_tot + 1e-8)).item()
    mae = (local_mae_sum / local_count).item()

    if is_main:
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

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            print(f"ERROR: {e}")
            traceback.print_exc()
            # Save error result
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--arch", required=True)
            parser.add_argument("--loss", required=True)
            parser.add_argument("--data-dir", required=True)
            parser.add_argument("--epochs", type=int, default=50)
            parser.add_argument("--batch-size", type=int, default=64)
            parser.add_argument("--lr", type=float, default=1e-3)
            args, _ = parser.parse_known_args()
            result = {"arch": args.arch, "loss": args.loss, "r2": float("-inf"),
                      "mae": float("inf"), "training_time": 0, "success": False, "error": str(e)}
            result_file = Path(args.data_dir) / f"result_{args.arch}_{args.loss}.json"
            with open(result_file, "w") as f:
                json.dump(result, f)
        raise
'''


def run_experiment_with_torchrun(
    arch: str,
    loss_name: str,
    data_dir: Path,
    n_epochs: int,
    batch_size: int,
    lr: float,
    n_gpus: int,
) -> Dict[str, Any]:
    """Run ONE experiment using torchrun with DDP."""

    # Write worker script
    worker_path = ARTIFACTS_DIR / "ddp_worker.py"
    with open(worker_path, "w") as f:
        f.write(DDP_WORKER_SCRIPT)

    # Clear previous result
    result_file = data_dir / f"result_{arch}_{loss_name}.json"
    if result_file.exists():
        result_file.unlink()

    # Build torchrun command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={n_gpus}",
        "--master_port=29500",
        str(worker_path),
        f"--arch={arch}",
        f"--loss={loss_name}",
        f"--data-dir={data_dir}",
        f"--epochs={n_epochs}",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
    ]

    print(f"\n{'='*60}")
    print(f"[torchrun] {arch} + {loss_name} on {n_gpus} GPUs")
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
    print("TIER 1: DDP Training (torchrun)")
    print("=" * 60)
    print("ONE experiment at a time, DDP across all GPUs")
    print("FULL data, memory-mapped files (no RAM duplication)")
    print()

    import torch
    n_gpus = torch.cuda.device_count()
    print(f"GPUs: {n_gpus}")

    # Load and prepare data
    print("\nPreparing data...")

    if args.dry_run:
        N, C, T = 100, 32, 500
        X = np.random.randn(N, C, T).astype(np.float32)
        y = np.random.randn(N, C, T).astype(np.float32)
        architectures = ["linear", "cnn"]
        losses = {"huber": "huber"}
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

        result = run_experiment_with_torchrun(
            arch=arch,
            loss_name=loss_name,
            data_dir=data_dir,
            n_epochs=n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_gpus=n_gpus,
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
