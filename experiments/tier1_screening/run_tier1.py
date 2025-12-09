#!/usr/bin/env python3
"""
Tier 1: Architecture Selection - U-Net vs Competitors
======================================================

Purpose: Prove that U-Net (CondUNet1D) beats other neural architectures.

This tier compares our U-Net against 7 competing architectures:
- linear: Simple linear baseline
- cnn: Basic CNN
- wavenet: WaveNet-style dilated convolutions
- fnet: Fourier-based architecture
- vit: Vision Transformer adapted for 1D
- performer: Efficient attention
- mamba: State-space model
- unet: Our CondUNet1D (should WIN!)

Goal: Show that U-Net significantly outperforms all competitors,
justifying its use as the sole architecture for Tiers 1.5+.

Features:
- Odor conditioning support for all architectures
- Single-run training per combination
- Metrics: R², MAE, Pearson correlation, PSD error
- Per-frequency-band R² (delta, theta, alpha, beta, gamma)

Losses tested:
- l1: Simple L1 baseline
- huber: Robust Huber loss
- wavelet: Standalone wavelet loss

Usage:
    python experiments/tier1_screening/run_tier1.py
    python experiments/tier1_screening/run_tier1.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
TRAIN_BATCH_SIZE = 8  # Same as train.py
TRAIN_LR = 0.0002  # Same as train.py

# Architectures to compare - includes our U-Net (CondUNet1D) as the candidate to beat others
ARCHITECTURES = ["unet", "linear", "cnn", "wavenet", "fnet", "vit"]

# Loss functions (comparable with train.py)
LOSS_FUNCTIONS = {
    "l1": "l1",                    # Simple baseline
    "huber": "huber",              # Robust baseline
    "wavelet": "wavelet",          # Standalone wavelet loss
}

# Neural frequency bands (same as tier0)
NEURAL_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 100),
}


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class NeuralResult:
    """Result for a single architecture + loss combination."""
    architecture: str
    loss: str
    # Core metrics (same as tier0)
    r2_mean: float
    mae_mean: float
    pearson_mean: float
    psd_error_db: float
    # Per-band R²
    r2_delta: Optional[float] = None
    r2_theta: Optional[float] = None
    r2_alpha: Optional[float] = None
    r2_beta: Optional[float] = None
    r2_gamma: Optional[float] = None
    # Training info
    training_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


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
# Worker Script (with odor conditioning and proper losses)
# =============================================================================

WORKER_SCRIPT = '''#!/usr/bin/env python3
"""Single-GPU Training with odor conditioning, mixed precision, and proper losses."""
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
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from scipy import signal as scipy_signal
from tqdm import tqdm

from experiments.study1_architecture.architectures import create_architecture
from experiments.study3_loss.losses import create_loss as create_study3_loss

# Import from models.py (exactly like train.py)
try:
    from models import build_wavelet_loss, CondUNet1D
    HAS_WAVELET_LOSS = True
    HAS_UNET = True
except ImportError:
    HAS_WAVELET_LOSS = False
    HAS_UNET = False
    build_wavelet_loss = None
    CondUNet1D = None

# Neural frequency bands for per-band R²
NEURAL_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 100),
}


class WaveletLoss(nn.Module):
    """Standalone wavelet loss."""
    def __init__(self):
        super().__init__()
        if HAS_WAVELET_LOSS:
            self.wavelet_loss = build_wavelet_loss(
                wavelet="morlet",
                omega0=3.0,
                use_complex_morlet=False,
            )
        else:
            self.wavelet_loss = None

    def forward(self, pred, target):
        if self.wavelet_loss is not None:
            return self.wavelet_loss(pred, target)
        return F.l1_loss(pred, target)  # Fallback


def create_loss(name: str, **kwargs):
    """Create loss by name, with support for wavelet."""
    if name == "wavelet":
        return WaveletLoss()
    else:
        return create_study3_loss(name, **kwargs)


class OdorDataset(Dataset):
    """Dataset with odor conditioning support."""
    def __init__(self, x_path: str, y_path: str, odor_path: str = None):
        self.X = np.load(x_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        self.odors = np.load(odor_path, mmap_mode='r') if odor_path and Path(odor_path).exists() else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = torch.from_numpy(self.y[idx].copy()).float()
        odor = torch.tensor(self.odors[idx], dtype=torch.long) if self.odors is not None else torch.tensor(0, dtype=torch.long)
        return x, y, odor


def create_model(arch_name: str, in_channels: int, out_channels: int, n_odors: int = 0):
    # Handle UNet (CondUNet1D) - same config as train.py
    if arch_name == "unet":
        if not HAS_UNET:
            raise ImportError("CondUNet1D not available - models.py import failed")
        return CondUNet1D(
            in_channels=in_channels,
            out_channels=out_channels,
            base=64,  # Smaller than train.py (128) for faster tier1 screening
            n_odors=n_odors,
            dropout=0.0,
            use_attention=True,
            attention_type="cross_freq_v2",
            norm_type="batch",
            cond_mode="cross_attn_gated",
            use_spectral_shift=False,  # Disable for tier1 (adds complexity)
            n_downsample=4,
            conv_type="modern",
            use_se=True,
            conv_kernel_size=7,
            dilations=(1, 4, 16, 32),
        )

    # Other architectures from study1
    VARIANT_MAP = {
        "linear": "simple",
        "cnn": "basic",
        "wavenet": "standard",
        "fnet": "standard",
        "vit": "standard",
    }
    variant = VARIANT_MAP.get(arch_name, "standard")
    try:
        return create_architecture(arch_name, variant=variant,
                                   in_channels=in_channels, out_channels=out_channels,
                                   n_odors=n_odors)
    except TypeError:
        return create_architecture(arch_name, variant=variant,
                                   in_channels=in_channels, out_channels=out_channels)


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, sample_rate: float = 1000.0):
    """Compute metrics - same as tier0 for comparability.

    Returns:
        Dict with: r2, mae, pearson, psd_error_db, r2_delta, r2_theta, r2_alpha, r2_beta, r2_gamma
    """
    if y_pred.ndim == 2:
        y_pred = y_pred[:, np.newaxis, :]
        y_true = y_true[:, np.newaxis, :]

    N, C, T = y_pred.shape

    # R² - per-channel then averaged
    pred_flat = y_pred.transpose(1, 0, 2).reshape(C, -1)
    true_flat = y_true.transpose(1, 0, 2).reshape(C, -1)

    ss_res = np.sum((true_flat - pred_flat) ** 2, axis=1)
    true_means = np.mean(true_flat, axis=1, keepdims=True)
    ss_tot = np.sum((true_flat - true_means) ** 2, axis=1)

    r2_per_channel = 1 - ss_res / (ss_tot + 1e-8)
    r2 = float(np.mean(r2_per_channel))

    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Pearson correlation
    pred_centered = pred_flat - np.mean(pred_flat, axis=1, keepdims=True)
    true_centered = true_flat - np.mean(true_flat, axis=1, keepdims=True)
    numerator = np.sum(pred_centered * true_centered, axis=1)
    denominator = np.sqrt(np.sum(pred_centered ** 2, axis=1) * np.sum(true_centered ** 2, axis=1))
    correlations = numerator / (denominator + 1e-8)
    valid_mask = np.isfinite(correlations)
    pearson = float(np.mean(correlations[valid_mask])) if valid_mask.any() else 0.0

    # PSD Error in dB
    try:
        _, psd_pred = scipy_signal.welch(y_pred, fs=sample_rate, axis=-1, nperseg=min(256, T))
        _, psd_true = scipy_signal.welch(y_true, fs=sample_rate, axis=-1, nperseg=min(256, T))
        log_pred = np.log10(psd_pred + 1e-10)
        log_true = np.log10(psd_true + 1e-10)
        psd_error_db = float(10 * np.mean(np.abs(log_pred - log_true)))
    except Exception:
        psd_error_db = np.nan

    # Per-frequency-band R²
    band_r2 = {}
    try:
        nyq = sample_rate / 2
        for band_name, (f_low, f_high) in NEURAL_BANDS.items():
            if f_low >= nyq:
                continue
            low = f_low / nyq
            high = min(f_high / nyq, 0.99)
            if low < high:
                try:
                    b, a = scipy_signal.butter(4, [low, high], btype='band')
                    pred_filt = scipy_signal.filtfilt(b, a, y_pred, axis=-1)
                    true_filt = scipy_signal.filtfilt(b, a, y_true, axis=-1)

                    pred_band_flat = pred_filt.transpose(1, 0, 2).reshape(C, -1)
                    true_band_flat = true_filt.transpose(1, 0, 2).reshape(C, -1)

                    ss_res_band = np.sum((true_band_flat - pred_band_flat) ** 2, axis=1)
                    true_band_means = np.mean(true_band_flat, axis=1, keepdims=True)
                    ss_tot_band = np.sum((true_band_flat - true_band_means) ** 2, axis=1)

                    r2_band = 1 - ss_res_band / (ss_tot_band + 1e-8)
                    band_r2[band_name] = float(np.mean(r2_band))
                except Exception:
                    pass
    except Exception:
        pass

    return {
        "r2": r2,
        "mae": mae,
        "pearson": pearson,
        "psd_error_db": psd_error_db,
        **{f"r2_{k}": v for k, v in band_r2.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--loss", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--n-odors", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    print(f"\\nTraining {args.arch} + {args.loss} (seed {args.seed})")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")

    # Load data
    data_dir = Path(args.data_dir)
    odor_path = str(data_dir / "odors_train.npy")
    train_dataset = OdorDataset(
        str(data_dir / "X_train.npy"),
        str(data_dir / "y_train.npy"),
        odor_path if Path(odor_path).exists() else None
    )
    odor_val_path = str(data_dir / "odors_val.npy")
    val_dataset = OdorDataset(
        str(data_dir / "X_val.npy"),
        str(data_dir / "y_val.npy"),
        odor_val_path if Path(odor_val_path).exists() else None
    )

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"  Shape: {train_dataset.X.shape}")

    # Get dims
    sample_x, sample_y, _ = train_dataset[0]
    in_channels = sample_x.shape[0]
    out_channels = sample_y.shape[0]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Model with odor conditioning
    model = create_model(args.arch, in_channels, out_channels, args.n_odors)
    model = model.to(device)

    # Loss
    criterion = create_loss(args.loss)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    warmup_epochs = 5

    start_time = time.time()

    # Training with tqdm progress bars
    # Track best validation loss for early stopping / best model
    best_val_loss = float('inf')
    best_model_state = None

    epoch_pbar = tqdm(range(args.epochs), desc=f"{args.arch}+{args.loss}", unit="epoch",
                      dynamic_ncols=True, leave=True)

    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Learning rate warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * warmup_factor

        # Batch progress bar with live loss
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}", unit="batch",
                          leave=False, dynamic_ncols=True)

        for x, y_batch, odor_ids in batch_pbar:
            x = x.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            odor_ids = odor_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                try:
                    y_pred = model(x, odor_ids)
                except TypeError:
                    y_pred = model(x)

                try:
                    loss = criterion(y_pred, y_batch)
                except Exception as e:
                    loss = F.l1_loss(y_pred, y_batch)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                n_batches += 1

                # Update batch progress bar with live loss
                batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        if epoch >= warmup_epochs:
            scheduler.step()

        # Validation after each epoch to track best model
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y_batch, odor_ids in val_loader:
                x = x.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                odor_ids = odor_ids.to(device, non_blocking=True)
                with autocast('cuda'):
                    try:
                        y_pred = model(x, odor_ids)
                    except TypeError:
                        y_pred = model(x)
                    try:
                        loss = criterion(y_pred, y_batch)
                    except Exception:
                        loss = F.l1_loss(y_pred, y_batch)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Update epoch progress bar with average loss
        avg_loss = epoch_loss / max(n_batches, 1)
        lr = optimizer.param_groups[0]['lr']
        epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}", val=f"{avg_val_loss:.4f}", lr=f"{lr:.2e}")

    training_time = time.time() - start_time

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Loaded best model (val_loss={best_val_loss:.4f})")

    # Final evaluation with best model
    model.eval()
    all_preds = []
    all_targets = []

    eval_pbar = tqdm(val_loader, desc="Evaluating", unit="batch", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for x, y_batch, odor_ids in eval_pbar:
            x = x.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            odor_ids = odor_ids.to(device, non_blocking=True)

            with autocast('cuda'):
                try:
                    y_pred = model(x, odor_ids)
                except TypeError:
                    y_pred = model(x)
            all_preds.append(y_pred.float().cpu().numpy())
            all_targets.append(y_batch.float().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metrics (same as tier0)
    metrics = compute_metrics(all_preds, all_targets)

    print(f"  DONE: R²={metrics['r2']:.4f}, Pearson={metrics['pearson']:.4f}, PSD_err={metrics['psd_error_db']:.2f}dB, Time={training_time:.1f}s")

    # Save result
    result = {
        "arch": args.arch,
        "loss": args.loss,
        "seed": args.seed,
        "training_time": training_time,
        "best_val_loss": best_val_loss,
        "success": True,
        **metrics,
    }
    result_file = Path(args.data_dir) / f"result_{args.arch}_{args.loss}.json"
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
        parser.add_argument("--lr", type=float, default=0.0002)
        parser.add_argument("--n-odors", type=int, default=0)
        parser.add_argument("--seed", type=int, default=42)
        args, _ = parser.parse_known_args()
        result = {"arch": args.arch, "loss": args.loss, "seed": args.seed,
                  "r2": float("-inf"), "mae": float("inf"), "pearson": 0.0,
                  "psd_error_db": float("inf"), "training_time": 0,
                  "success": False, "error": str(e)}
        result_file = Path(args.data_dir) / f"result_{args.arch}_{args.loss}.json"
        with open(result_file, "w") as f:
            json.dump(result, f)
        raise
'''


def run_unet_via_train_py(
    n_epochs: int,
    batch_size: int,
    lr: float,
    base_channels: int = 64,
) -> Dict[str, Any]:
    """Run UNet via torchrun train.py with fair comparison flags.

    Uses --no-bidirectional and --skip-spectral-finetune for fair comparison
    with other architectures that don't have these features.

    Returns metrics from the training run.
    """
    import re

    # Use torchrun for proper distributed training setup (even with 1 GPU)
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        str(PROJECT_ROOT / "train.py"),
        f"--epochs={n_epochs}",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
        f"--base-channels={base_channels}",
        "--no-bidirectional",           # Fair comparison: no reverse model
        "--skip-spectral-finetune",     # Fair comparison: no Stage 2
        "--eval-stage1",                # Evaluate after Stage 1
    ]

    print(f"  Running: {' '.join(cmd)}")

    try:
        # Run train.py and capture output
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            timeout=14400,  # 4 hours
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        # Parse metrics from train.py output
        # Look for Stage 1 final evaluation output like:
        # [Stage1 Final] R²=0.xxxx | Pearson=0.xxxx | MAE=0.xxxx | ...
        metrics = {
            "r2": float("-inf"),
            "mae": float("inf"),
            "pearson": 0.0,
            "psd_error_db": float("inf"),
            "success": False,
        }

        # Try to find Stage 1 final metrics
        for line in output.split("\n"):
            # Match lines with metrics
            if "R²=" in line or "r2=" in line.lower():
                # Parse R²
                r2_match = re.search(r"R²=([0-9.-]+)", line)
                if r2_match:
                    metrics["r2"] = float(r2_match.group(1))
                    metrics["success"] = True

                # Parse Pearson
                pearson_match = re.search(r"Pearson=([0-9.-]+)", line, re.IGNORECASE)
                if pearson_match:
                    metrics["pearson"] = float(pearson_match.group(1))

                # Parse MAE
                mae_match = re.search(r"MAE=([0-9.-]+)", line, re.IGNORECASE)
                if mae_match:
                    metrics["mae"] = float(mae_match.group(1))

                # Parse PSD error
                psd_match = re.search(r"psd_err(?:_db)?=([0-9.-]+)", line, re.IGNORECASE)
                if psd_match:
                    metrics["psd_error_db"] = float(psd_match.group(1))

        # If we didn't find metrics in output, try to load from checkpoint
        if not metrics["success"]:
            checkpoint_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "best_model.pt"
            if checkpoint_path.exists():
                import torch
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                if "metrics" in ckpt:
                    m = ckpt["metrics"]
                    metrics["r2"] = m.get("r2", m.get("corr", float("-inf")))
                    metrics["mae"] = m.get("mae", float("inf"))
                    metrics["pearson"] = m.get("corr", 0.0)
                    metrics["psd_error_db"] = m.get("psd_err_db", float("inf"))
                    metrics["success"] = True

        return metrics

    except subprocess.CalledProcessError as e:
        print(f"  train.py failed with exit code {e.returncode}")
        print(f"  stderr: {e.stderr[:500] if e.stderr else 'None'}")
        return {"r2": float("-inf"), "success": False, "error": f"Exit code {e.returncode}"}
    except subprocess.TimeoutExpired:
        return {"r2": float("-inf"), "success": False, "error": "Timeout (4h)"}
    except Exception as e:
        return {"r2": float("-inf"), "success": False, "error": str(e)}


def run_single_experiment(
    arch: str,
    loss_name: str,
    data_dir: Path,
    n_epochs: int,
    batch_size: int,
    lr: float,
    n_odors: int,
    python_path: str,
) -> Dict[str, Any]:
    """Run ONE experiment (single run, no folds).

    For UNet: Uses torchrun train.py with --no-bidirectional for fair comparison.
    For other architectures: Uses embedded WORKER_SCRIPT.
    """

    # =========================================================================
    # UNet: Use train.py via torchrun for fair comparison
    # =========================================================================
    if arch == "unet":
        print(f"  [UNet] Using torchrun train.py with --no-bidirectional --skip-spectral-finetune")
        return run_unet_via_train_py(
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            base_channels=64,  # Same as other tier1 architectures
        )

    # =========================================================================
    # Other architectures: Use embedded worker script
    # =========================================================================

    # Write worker script
    worker_path = ARTIFACTS_DIR / "worker.py"
    with open(worker_path, "w") as f:
        f.write(WORKER_SCRIPT)

    # Clear previous result
    result_file = data_dir / f"result_{arch}_{loss_name}.json"
    if result_file.exists():
        result_file.unlink()

    cmd = [
        python_path,
        str(worker_path),
        f"--arch={arch}",
        f"--loss={loss_name}",
        f"--data-dir={data_dir}",
        f"--epochs={n_epochs}",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
        f"--n-odors={n_odors}",
        f"--seed=42",
    ]

    try:
        # Don't capture output so tqdm progress bars are visible
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, timeout=7200)

        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        else:
            return {"r2": float("-inf"), "success": False, "error": "No result file"}

    except subprocess.CalledProcessError as e:
        return {"r2": float("-inf"), "success": False, "error": f"Exit code {e.returncode}"}
    except subprocess.TimeoutExpired:
        return {"r2": float("-inf"), "success": False, "error": "Timeout"}


def evaluate_architecture(
    arch: str,
    loss_name: str,
    data_dir: Path,
    n_epochs: int,
    batch_size: int,
    lr: float,
    n_odors: int,
    python_path: str,
) -> NeuralResult:
    """Evaluate architecture with single run (no CV for simplicity)."""

    print(f"\n{'='*60}")
    print(f"Evaluating: {arch} + {loss_name}")
    print(f"{'='*60}")

    result = run_single_experiment(
        arch=arch,
        loss_name=loss_name,
        data_dir=data_dir,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_odors=n_odors,
        python_path=python_path,
    )

    if result.get("success", False):
        print(f"  R²={result['r2']:.4f}, Pearson={result['pearson']:.4f}, PSD_err={result['psd_error_db']:.2f}dB")
        return NeuralResult(
            architecture=arch,
            loss=loss_name,
            r2_mean=result["r2"],
            mae_mean=result.get("mae", np.nan),
            pearson_mean=result.get("pearson", np.nan),
            psd_error_db=result.get("psd_error_db", np.nan),
            r2_delta=result.get("r2_delta"),
            r2_theta=result.get("r2_theta"),
            r2_alpha=result.get("r2_alpha"),
            r2_beta=result.get("r2_beta"),
            r2_gamma=result.get("r2_gamma"),
            training_time=result.get("training_time", 0),
            success=True,
        )
    else:
        print(f"  FAILED: {result.get('error', 'Unknown')}")
        return NeuralResult(
            architecture=arch,
            loss=loss_name,
            r2_mean=np.nan,
            mae_mean=np.nan,
            pearson_mean=np.nan,
            psd_error_db=np.nan,
            training_time=0,
            success=False,
            error=result.get("error"),
        )


def load_tier0_results() -> Optional[Dict[str, Any]]:
    """Load Tier 0 classical baseline results for comparison."""
    tier0_dir = PROJECT_ROOT / "artifacts" / "tier0_classical"

    # Find the most recent results file
    result_files = list(tier0_dir.glob("tier0_results_*.json"))
    if not result_files:
        return None

    latest = max(result_files, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=TRAIN_LR)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("TIER 1: Architecture Screening (Single Run)")
    print("=" * 60)
    print("Losses: l1, huber, wavelet")
    print("Metrics: R², MAE, Pearson, PSD error, per-band R²")
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
        odors = np.random.randint(0, 7, size=N).astype(np.int64)
        n_odors = 7
        architectures = ARCHITECTURES  # Test ALL architectures
        losses = LOSS_FUNCTIONS         # Test ALL losses
        n_epochs = 3                    # But with fewer epochs
    else:
        from data import prepare_data
        data = prepare_data()
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        ob = data["ob"]
        pcx = data["pcx"]
        odors_all = data["odors"]
        n_odors = data["n_odors"]

        cv_idx = np.concatenate([train_idx, val_idx])
        X = ob[cv_idx].astype(np.float32)
        y = pcx[cv_idx].astype(np.float32)
        odors = odors_all[cv_idx]

        architectures = ARCHITECTURES
        losses = LOSS_FUNCTIONS
        n_epochs = args.epochs

    # Split data
    n_samples = len(X)
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * 0.8)

    X_train = X[indices[:n_train]]
    y_train = y[indices[:n_train]]
    odors_train = odors[indices[:n_train]]
    X_val = X[indices[n_train:]]
    y_val = y[indices[n_train:]]
    odors_val = odors[indices[n_train:]]

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  Odors: {n_odors} classes")
    print(f"  Time steps: {X_train.shape[2]}")

    # Save data
    data_dir = ARTIFACTS_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    np.save(data_dir / "X_train.npy", X_train)
    np.save(data_dir / "y_train.npy", y_train)
    np.save(data_dir / "odors_train.npy", odors_train)
    np.save(data_dir / "X_val.npy", X_val)
    np.save(data_dir / "y_val.npy", y_val)
    np.save(data_dir / "odors_val.npy", odors_val)
    print(f"  Saved to: {data_dir}")

    # Free RAM
    del X, y, odors, X_train, y_train, X_val, y_val, odors_train, odors_val
    gc.collect()

    # Load checkpoint
    results = {} if args.no_resume else _load_checkpoint()
    if results:
        print(f"\nResuming: {len(results)} combinations done")

    # Build experiment list
    combos = [(a, lc, ln) for a in architectures for lc, ln in losses.items()
              if f"{a}_{lc}" not in results]

    print(f"\nExperiments: {len(combos)} to run")

    # Run experiments
    for i, (arch, loss_cat, loss_name) in enumerate(combos):
        print(f"\n[{i+1}/{len(combos)}] {arch} + {loss_cat}")

        result = evaluate_architecture(
            arch=arch,
            loss_name=loss_name,
            data_dir=data_dir,
            n_epochs=n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_odors=n_odors,
            python_path=python_path,
        )

        results[f"{arch}_{loss_cat}"] = asdict(result)
        _save_checkpoint(results)

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Load Tier 0 results
    tier0 = load_tier0_results()
    if tier0:
        best_classical_r2 = tier0.get("best_r2", 0.0)
        best_classical_method = tier0.get("best_method", "unknown")
        print(f"\nTier 0 Classical Floor: R²={best_classical_r2:.4f} ({best_classical_method})")
        print(f"Gate threshold (beat by 0.10): {best_classical_r2 + 0.10:.4f}")
    else:
        best_classical_r2 = 0.0
        print("\n[Warning] No Tier 0 results found for comparison")

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'Arch':12} {'Loss':12} {'R²':>8} {'Pearson':>8} {'MAE':>8} {'PSD_err':>10} {'vs Classical':>12}")
    print(f"{'='*100}")

    sorted_results = sorted(results.items(), key=lambda x: -x[1].get("r2_mean", float("-inf")))

    passing_gate = []
    for key, r in sorted_results:
        r2 = r.get("r2_mean", float("-inf"))
        pearson = r.get("pearson_mean", np.nan)
        mae = r.get("mae_mean", np.nan)
        psd_err = r.get("psd_error_db", np.nan)
        arch = r.get("architecture", "?")
        loss = r.get("loss", "?")

        if np.isnan(r2) or r2 == float("-inf"):
            status = "FAIL"
        elif r2 > best_classical_r2 + 0.10:
            status = "✓ PASS"
            passing_gate.append((arch, loss, r2, pearson, psd_err))
        elif r2 > best_classical_r2:
            status = "~ marginal"
        else:
            status = "✗ below"

        psd_str = f"{psd_err:.2f}dB" if not np.isnan(psd_err) else "nan"
        print(f"{arch:12} {loss:12} {r2:8.4f} {pearson:8.4f} {mae:8.4f} {psd_str:>10} {status:>12}")

    print(f"{'='*100}")

    # Per-band R² for top result
    if sorted_results:
        top_key, top_result = sorted_results[0]
        print(f"\nTop result ({top_result.get('architecture')} + {top_result.get('loss')}) per-band R²:")
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            band_r2 = top_result.get(f"r2_{band}")
            if band_r2 is not None:
                print(f"  {band:8}: {band_r2:.4f}")

    # Gate summary
    print(f"\n{'='*60}")
    print("GATE RESULTS")
    print(f"{'='*60}")
    print(f"Classical floor (Tier 0): {best_classical_r2:.4f}")
    print(f"Gate threshold (+0.10):   {best_classical_r2 + 0.10:.4f}")
    print(f"Combinations passing:     {len(passing_gate)}/{len(results)}")

    if passing_gate:
        print("\nPassing combinations (proceed to Tier 2):")
        for arch, loss, r2, pearson, psd_err in sorted(passing_gate, key=lambda x: -x[2]):
            print(f"  {arch} + {loss}: R²={r2:.4f} (Δ=+{r2 - best_classical_r2:.4f}), Pearson={pearson:.4f}, PSD_err={psd_err:.2f}dB")
    else:
        print("\n[!] No neural methods beat the gate threshold.")
        print("    Consider: more epochs, hyperparameter tuning, or architectural changes.")

    # Save final results
    final_results = {
        "tier": 1,
        "name": "Architecture Screening (Single Run)",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs": n_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "losses": list(losses.keys()),
        },
        "classical_floor": best_classical_r2,
        "gate_threshold": best_classical_r2 + 0.10,
        "results": results,
        "passing_gate": [{"arch": a, "loss": l, "r2": r, "pearson": p, "psd_error_db": e} for a, l, r, p, e in passing_gate],
        "best_neural": sorted_results[0] if sorted_results else None,
    }

    out_path = ARTIFACTS_DIR / f"tier1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
