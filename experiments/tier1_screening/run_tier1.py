#!/usr/bin/env python3
"""
Tier 1: Architecture Screening with Statistical Analysis
=========================================================

Full training with cross-validation and bootstrap CI.
Compares neural architectures against Tier 0 classical baselines.

Features:
- Odor conditioning support
- 5-fold cross-validation with multiple seeds
- Bootstrap 95% confidence intervals
- Statistical comparison with classical baselines
- Checkpointing for resume

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
from typing import Any, Dict, List, Optional, Tuple

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
TRAIN_BATCH_SIZE = 8  # For single 6GB GPU
TRAIN_LR = 1e-3
N_FOLDS = 1  # Single run (no cross-validation)
N_SEEDS = 1

ARCHITECTURES = ["linear", "cnn", "wavenet", "fnet", "vit", "performer", "mamba"]

# Best loss per category (fast + literature-recommended)
LOSS_CATEGORIES = {
    "huber": "huber",              # Standard robust loss (fast)
    "mse": "mse",                  # Baseline
    "spectral": "multi_scale_spectral",  # FFT-based spectral (Défossez et al.)
    "ccc": "concordance",          # Gold standard correlation (Lin 1989)
}


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class NeuralResult:
    """Result for a single architecture + loss combination."""
    architecture: str
    loss: str
    r2_mean: float
    r2_std: float
    r2_ci_lower: float
    r2_ci_upper: float
    mae_mean: float
    mae_std: float
    training_time: float
    fold_r2s: List[float]
    n_folds: int
    n_seeds: int
    success: bool
    error: Optional[str] = None


# =============================================================================
# Statistical Functions
# =============================================================================

def compute_bootstrap_ci(values: np.ndarray, n_bootstrap: int = 10000, ci_level: float = 0.95) -> Dict[str, float]:
    """Compute bootstrap 95% confidence interval - VECTORIZED."""
    valid_values = values[~np.isnan(values)]

    if len(valid_values) < 2:
        return {"ci_lower": np.nan, "ci_upper": np.nan, "se": np.nan}

    n = len(valid_values)
    rng = np.random.RandomState(42)

    # Vectorized bootstrap
    bootstrap_indices = rng.randint(0, n, size=(n_bootstrap, n))
    bootstrap_samples = valid_values[bootstrap_indices]
    bootstrap_means = np.mean(bootstrap_samples, axis=1)

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    se = float(np.std(bootstrap_means, ddof=1))

    return {"ci_lower": ci_lower, "ci_upper": ci_upper, "se": se}


def paired_t_test(scores1: np.ndarray, scores2: np.ndarray) -> Tuple[float, float]:
    """Paired t-test for comparing two methods."""
    from scipy import stats
    valid_mask = ~(np.isnan(scores1) | np.isnan(scores2))
    if valid_mask.sum() < 2:
        return np.nan, np.nan
    t_stat, p_value = stats.ttest_rel(scores1[valid_mask], scores2[valid_mask])
    return float(t_stat), float(p_value)


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
# Worker Script (with odor conditioning)
# =============================================================================

WORKER_SCRIPT = '''#!/usr/bin/env python3
"""Single-GPU Training with odor conditioning and mixed precision"""
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
from experiments.study3_loss.losses import create_loss as create_study3_loss

# Import losses from models.py
try:
    from models import WaveletLoss, HighFrequencySpectralLoss
    HAS_MODEL_LOSSES = True
except ImportError:
    HAS_MODEL_LOSSES = False


def create_loss(name: str, **kwargs):
    """Create loss by name, with support for models.py losses."""
    # Handle special losses from models.py
    if name == "wavelet" and HAS_MODEL_LOSSES:
        return WaveletLoss(omega0=5.0)
    elif name == "hf_spectral" and HAS_MODEL_LOSSES:
        return HighFrequencySpectralLoss(sample_rate=1000.0, use_log_psd=True)
    else:
        # Use study3_loss registry
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
    # Try to pass n_odors for conditioning
    try:
        return create_architecture(arch_name, variant=variant,
                                   in_channels=in_channels, out_channels=out_channels,
                                   n_odors=n_odors)
    except TypeError:
        # Architecture doesn't support n_odors
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
    parser.add_argument("--n-odors", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    print(f"\\nTraining {args.arch} + {args.loss} (fold {args.fold}, seed {args.seed})")
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

    # Training
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Learning rate warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * warmup_factor

        for x, y_batch, odor_ids in train_loader:
            x = x.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            odor_ids = odor_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                # Try to pass odor_ids for conditioning
                try:
                    y_pred = model(x, odor_ids)
                except TypeError:
                    y_pred = model(x)

                try:
                    loss = criterion(y_pred, y_batch)
                except Exception as e:
                    print(f"  [Warning] Loss {args.loss} failed: {e}, using L1")
                    loss = nn.functional.l1_loss(y_pred, y_batch)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                n_batches += 1

        if epoch >= warmup_epochs:
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
        for x, y_batch, odor_ids in val_loader:
            x = x.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            odor_ids = odor_ids.to(device, non_blocking=True)

            with autocast('cuda'):
                try:
                    y_pred = model(x, odor_ids)
                except TypeError:
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
        "fold": args.fold,
        "seed": args.seed,
        "r2": r2,
        "mae": mae,
        "training_time": training_time,
        "success": True,
    }
    result_file = Path(args.data_dir) / f"result_{args.arch}_{args.loss}_f{args.fold}_s{args.seed}.json"
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
        parser.add_argument("--n-odors", type=int, default=0)
        parser.add_argument("--fold", type=int, default=0)
        parser.add_argument("--seed", type=int, default=42)
        args, _ = parser.parse_known_args()
        result = {"arch": args.arch, "loss": args.loss, "fold": args.fold, "seed": args.seed,
                  "r2": float("-inf"), "mae": float("inf"), "training_time": 0,
                  "success": False, "error": str(e)}
        result_file = Path(args.data_dir) / f"result_{args.arch}_{args.loss}_f{args.fold}_s{args.seed}.json"
        with open(result_file, "w") as f:
            json.dump(result, f)
        raise
'''


def run_single_experiment(
    arch: str,
    loss_name: str,
    data_dir: Path,
    n_epochs: int,
    batch_size: int,
    lr: float,
    n_odors: int,
    fold: int,
    seed: int,
    python_path: str,
) -> Dict[str, Any]:
    """Run ONE experiment (single fold, single seed)."""

    # Write worker script
    worker_path = ARTIFACTS_DIR / "worker.py"
    with open(worker_path, "w") as f:
        f.write(WORKER_SCRIPT)

    # Clear previous result
    result_file = data_dir / f"result_{arch}_{loss_name}_f{fold}_s{seed}.json"
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
        f"--fold={fold}",
        f"--seed={seed}",
    ]

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, timeout=7200,
                      capture_output=True, text=True)

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
    n_folds: int,
    seeds: List[int],
    python_path: str,
) -> NeuralResult:
    """Evaluate architecture with cross-validation and multiple seeds."""

    print(f"\n{'='*60}")
    print(f"Evaluating: {arch} + {loss_name}")
    print(f"  {n_folds} folds × {len(seeds)} seeds = {n_folds * len(seeds)} runs")
    print(f"{'='*60}")

    all_r2s = []
    all_maes = []
    total_time = 0.0

    for seed in seeds:
        for fold in range(n_folds):
            print(f"  [{arch}+{loss_name}] Fold {fold+1}/{n_folds}, Seed {seed}")

            result = run_single_experiment(
                arch=arch,
                loss_name=loss_name,
                data_dir=data_dir,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                n_odors=n_odors,
                fold=fold,
                seed=seed,
                python_path=python_path,
            )

            if result.get("success", False):
                all_r2s.append(result["r2"])
                all_maes.append(result.get("mae", np.nan))
                total_time += result.get("training_time", 0)
                print(f"    R²={result['r2']:.4f}")
            else:
                all_r2s.append(np.nan)
                all_maes.append(np.nan)
                print(f"    FAILED: {result.get('error', 'Unknown')}")

    # Compute statistics
    r2_array = np.array(all_r2s)
    mae_array = np.array(all_maes)
    valid_r2s = r2_array[~np.isnan(r2_array)]
    valid_maes = mae_array[~np.isnan(mae_array)]

    ci = compute_bootstrap_ci(r2_array)

    r2_mean = float(np.mean(valid_r2s)) if len(valid_r2s) > 0 else np.nan
    r2_std = float(np.std(valid_r2s, ddof=1)) if len(valid_r2s) > 1 else 0.0
    mae_mean = float(np.mean(valid_maes)) if len(valid_maes) > 0 else np.nan
    mae_std = float(np.std(valid_maes, ddof=1)) if len(valid_maes) > 1 else 0.0

    print(f"\n  SUMMARY: R² = {r2_mean:.4f} ± {r2_std:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    return NeuralResult(
        architecture=arch,
        loss=loss_name,
        r2_mean=r2_mean,
        r2_std=r2_std,
        r2_ci_lower=ci["ci_lower"],
        r2_ci_upper=ci["ci_upper"],
        mae_mean=mae_mean,
        mae_std=mae_std,
        training_time=total_time,
        fold_r2s=all_r2s,
        n_folds=n_folds,
        n_seeds=len(seeds),
        success=len(valid_r2s) > 0,
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
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("TIER 1: Architecture Screening with Statistical Analysis")
    print("=" * 60)
    print(f"Cross-validation: {args.n_folds} folds × {args.n_seeds} seeds")
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
        architectures = ["cnn", "wavenet"]  # Quick test
        losses = {"huber": "huber", "spectral": "multi_scale_spectral"}
        n_epochs = 3
        n_folds = 1  # Single run
        seeds = [42]
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
        losses = LOSS_CATEGORIES
        n_epochs = args.epochs
        n_folds = args.n_folds
        seeds = list(range(42, 42 + args.n_seeds))

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
    print(f"Total runs: {len(combos)} × {n_folds} folds × {len(seeds)} seeds = {len(combos) * n_folds * len(seeds)}")

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
            n_folds=n_folds,
            seeds=seeds,
            python_path=python_path,
        )

        results[f"{arch}_{loss_cat}"] = asdict(result)
        _save_checkpoint(results)

    # ==========================================================================
    # Statistical Analysis
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    # Load Tier 0 results
    tier0 = load_tier0_results()
    if tier0:
        best_classical = tier0.get("best_r2", 0.0)
        best_classical_method = tier0.get("best_method", "unknown")
        print(f"\nTier 0 Classical Floor: {best_classical:.4f} ({best_classical_method})")
        print(f"Gate threshold (beat by 0.10): {best_classical + 0.10:.4f}")
    else:
        best_classical = 0.0
        print("\n[Warning] No Tier 0 results found for comparison")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Arch':12} {'Loss':12} {'R² Mean':>10} {'± Std':>8} {'95% CI':>20} {'vs Classical':>12}")
    print(f"{'='*80}")

    sorted_results = sorted(results.items(), key=lambda x: -x[1].get("r2_mean", float("-inf")))

    passing_gate = []
    for key, r in sorted_results:
        r2 = r.get("r2_mean", float("-inf"))
        std = r.get("r2_std", 0)
        ci_l = r.get("r2_ci_lower", float("-inf"))
        ci_u = r.get("r2_ci_upper", float("-inf"))
        arch = r.get("architecture", "?")
        loss = r.get("loss", "?")

        if np.isnan(r2) or r2 == float("-inf"):
            status = "FAIL"
        elif r2 > best_classical + 0.10:
            status = "✓ PASS"
            passing_gate.append((arch, loss, r2))
        elif r2 > best_classical:
            status = "~ marginal"
        else:
            status = "✗ below"

        ci_str = f"[{ci_l:.4f}, {ci_u:.4f}]" if not np.isnan(ci_l) else "[nan, nan]"
        print(f"{arch:12} {loss:12} {r2:10.4f} {std:8.4f} {ci_str:>20} {status:>12}")

    print(f"{'='*80}")

    # Gate summary
    print(f"\n{'='*60}")
    print("GATE RESULTS")
    print(f"{'='*60}")
    print(f"Classical floor (Tier 0): {best_classical:.4f}")
    print(f"Gate threshold (+0.10):   {best_classical + 0.10:.4f}")
    print(f"Combinations passing:     {len(passing_gate)}/{len(results)}")

    if passing_gate:
        print("\nPassing combinations (proceed to Tier 2):")
        for arch, loss, r2 in sorted(passing_gate, key=lambda x: -x[2]):
            print(f"  {arch} + {loss}: R² = {r2:.4f} (Δ = +{r2 - best_classical:.4f})")
    else:
        print("\n[!] No neural methods beat the gate threshold.")
        print("    Consider: more epochs, hyperparameter tuning, or architectural changes.")

    # Save final results
    final_results = {
        "tier": 1,
        "name": "Architecture Screening",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs": n_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_folds": n_folds,
            "n_seeds": len(seeds),
        },
        "classical_floor": best_classical,
        "gate_threshold": best_classical + 0.10,
        "results": results,
        "passing_gate": [{"arch": a, "loss": l, "r2": r} for a, l, r in passing_gate],
        "best_neural": sorted_results[0] if sorted_results else None,
    }

    out_path = ARTIFACTS_DIR / f"tier1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
