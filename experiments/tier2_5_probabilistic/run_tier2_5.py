#!/usr/bin/env python3
"""
Tier 2.5: Probabilistic Loss Ablation
======================================

Tests probabilistic losses that model signal distributions. These are added
ON TOP of the base loss (huber_wavelet by default) to see if probabilistic
modeling of neural signal properties improves translation quality.

Probabilistic losses tested:
    - none: Baseline (no probabilistic loss)
    - gaussian_nll: Gaussian negative log-likelihood (learned variance)
    - laplacian_nll: Laplacian NLL (robust to outliers)
    - rayleigh: Signal envelope (amplitude) distribution
    - von_mises: Phase distribution (circular statistics)
    - kl_divergence: Distribution matching via soft histograms
    - cauchy_nll: Heavy-tailed (very robust to outliers)
    - student_t_nll: Controllable heavy tails (df=4)
    - gumbel: Peak/extreme value distribution
    - gamma: Positive-valued signals
    - log_normal: Multiplicative processes
    - mixture: Gaussian mixture (multi-modal)

Architecture: CondUNet1D (fixed from tier 1/2)
Base loss: huber_wavelet (fixed, Huber + wavelet)

This tier uses train.py via torchrun for fair comparison with other tiers.

References:
    - PyTorch GaussianNLLLoss: https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
    - Circular statistics: https://www.jneurosci.org/content/36/3/860
    - KL Divergence: https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fix imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.common.config_registry import (
    get_registry,
    Tier2_5Result,
    ProbabilisticLossResult,
)


def cleanup_gpu_memory():
    """Force cleanup of GPU memory between runs.

    This is CRITICAL for multi-GPU training to prevent memory leaks
    and zombie processes from previous runs causing crashes.
    """
    # Force Python garbage collection
    gc.collect()

    # Clear CUDA cache if torch is available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset peak memory stats
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
    except ImportError:
        pass

    # Kill any zombie torchrun/python processes that might be holding GPU memory
    try:
        # Find and kill any orphaned torchrun processes
        subprocess.run(
            ["pkill", "-9", "-f", "torch.distributed.run"],
            capture_output=True,
            timeout=5,
        )
        # Also kill any orphaned train.py processes
        subprocess.run(
            ["pkill", "-9", "-f", "train.py"],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass  # Ignore errors - processes might not exist

    # Wait for GPU memory to be fully released
    time.sleep(5)

    # Force another garbage collection after sleep
    gc.collect()


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier2_5_probabilistic"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Probabilistic losses to test (added ON TOP of base loss)
# These map to --prob-loss argument in train.py
PROBABILISTIC_LOSSES = [
    "none",           # Baseline (no probabilistic loss)
    "gaussian_nll",   # Gaussian negative log-likelihood
    "laplacian_nll",  # Laplacian NLL (robust to outliers)
    "rayleigh",       # Signal envelope distribution
    "von_mises",      # Phase distribution (circular statistics)
    "kl_divergence",  # Distribution matching
    "cauchy_nll",     # Heavy-tailed (very robust)
    "student_t_nll",  # Controllable heavy tails
    "gumbel",         # Peak/extreme value distribution
    "gamma",          # Positive-valued signals
    "log_normal",     # Multiplicative processes
    "mixture",        # Gaussian mixture (multi-modal)
]

# Dry-run uses a subset for faster testing
FAST_PROB_LOSSES = ["none", "gaussian_nll", "laplacian_nll", "rayleigh", "von_mises"]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ProbLossRunResult:
    """Result from a single probabilistic loss run."""
    prob_loss: str
    # Test set metrics (primary for publication)
    r2: float
    mae: float
    pearson: float
    psd_error_db: float
    # Validation set metrics (for training diagnostics)
    val_r2: float = 0.0
    val_mae: float = 0.0
    val_pearson: float = 0.0
    val_psd_error_db: float = 0.0
    training_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# Run via train.py
# =============================================================================

def run_prob_loss_via_train_py(
    prob_loss: str,
    n_epochs: int,
    batch_size: int,
    lr: float,
    base_channels: int = 64,
    seed: int = 42,
    prob_loss_weight: float = 1.0,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run UNet with specific probabilistic loss via torchrun train.py.

    Uses --no-bidirectional and --skip-spectral-finetune for fair comparison.
    The base loss is huber_wavelet (default), probabilistic loss is added on top.

    Args:
        prob_loss: Probabilistic loss type (none, gaussian_nll, rayleigh, etc.)
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        base_channels: UNet base channels
        seed: Random seed for reproducibility
        prob_loss_weight: Weight for probabilistic loss
        debug: If True, enable verbose debugging for distributed training

    Returns:
        Dict with r2, mae, pearson, psd_error_db, success, error
    """
    # Use torchrun for distributed training (8x A100)
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        str(PROJECT_ROOT / "train.py"),
        f"--epochs={n_epochs}",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
        f"--base-channels={base_channels}",
        f"--seed={seed}",
        "--fsdp",
        "--fsdp-strategy=grad_op",
        "--no-bidirectional",
        "--skip-spectral-finetune",
        "--eval-stage1",
        f"--prob-loss={prob_loss}",
        f"--prob-loss-weight={prob_loss_weight}",
    ]

    # Set up environment with debugging options
    env = os.environ.copy()
    if debug:
        env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        env["NCCL_DEBUG"] = "WARN"
        env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
        print(f"  [DEBUG MODE] TORCH_DISTRIBUTED_DEBUG=DETAIL, NCCL_DEBUG=WARN")

    print(f"  Running: {' '.join(cmd)}")

    try:
        start_time = time.time()

        # Run train.py and capture output
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            timeout=14400,  # 4 hours
            capture_output=True,
            text=True,
            env=env,
        )

        elapsed = time.time() - start_time
        output = result.stdout + result.stderr

        # Parse metrics from train.py output
        metrics = {
            # Test set metrics (primary for publication)
            "r2": float("-inf"),
            "mae": float("inf"),
            "pearson": 0.0,
            "psd_error_db": float("inf"),
            # Validation set metrics
            "val_r2": float("-inf"),
            "val_mae": float("inf"),
            "val_pearson": 0.0,
            "val_psd_error_db": float("inf"),
            "training_time": elapsed,
            "success": False,
        }

        # Try to find Stage 1 final metrics (machine-parseable format)
        for line in output.split("\n"):
            # VALIDATION metrics: STAGE1_VAL_*
            if "STAGE1_VAL_R2=" in line:
                match = re.search(r"STAGE1_VAL_R2=([0-9.-]+)", line)
                if match:
                    metrics["val_r2"] = float(match.group(1))
            if "STAGE1_VAL_CORR=" in line:
                match = re.search(r"STAGE1_VAL_CORR=([0-9.-]+)", line)
                if match:
                    metrics["val_pearson"] = float(match.group(1))
            if "STAGE1_VAL_MAE=" in line:
                match = re.search(r"STAGE1_VAL_MAE=([0-9.-]+)", line)
                if match:
                    metrics["val_mae"] = float(match.group(1))
            if "STAGE1_VAL_PSD_ERR_DB=" in line:
                match = re.search(r"STAGE1_VAL_PSD_ERR_DB=([0-9.-]+)", line)
                if match:
                    metrics["val_psd_error_db"] = float(match.group(1))

            # TEST metrics: STAGE1_RESULT_* (primary for publication)
            if "STAGE1_RESULT_R2=" in line:
                r2_match = re.search(r"STAGE1_RESULT_R2=([0-9.-]+)", line)
                if r2_match:
                    metrics["r2"] = float(r2_match.group(1))
                    metrics["success"] = True
            if "STAGE1_RESULT_CORR=" in line:
                corr_match = re.search(r"STAGE1_RESULT_CORR=([0-9.-]+)", line)
                if corr_match:
                    metrics["pearson"] = float(corr_match.group(1))
            if "STAGE1_RESULT_MAE=" in line:
                mae_match = re.search(r"STAGE1_RESULT_MAE=([0-9.-]+)", line)
                if mae_match:
                    metrics["mae"] = float(mae_match.group(1))
            if "STAGE1_RESULT_PSD_ERR_DB=" in line:
                psd_match = re.search(r"STAGE1_RESULT_PSD_ERR_DB=([0-9.-]+)", line)
                if psd_match:
                    metrics["psd_error_db"] = float(psd_match.group(1))

        # If we didn't find metrics in output, try to load from checkpoint
        if not metrics["success"]:
            checkpoint_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "best_model.pt"
            if checkpoint_path.exists():
                import torch
                try:
                    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    if "metrics" in ckpt:
                        m = ckpt["metrics"]
                        metrics["r2"] = m.get("r2", m.get("corr", float("-inf")))
                        metrics["mae"] = m.get("mae", float("inf"))
                        metrics["pearson"] = m.get("corr", 0.0)
                        metrics["psd_error_db"] = m.get("psd_err_db", float("inf"))
                        metrics["success"] = True
                    elif "best_val_loss" in ckpt:
                        # Fallback: at least we have loss
                        metrics["success"] = True
                        metrics["r2"] = 0.0
                except Exception as e:
                    print(f"  Warning: Could not load checkpoint: {e}")

        # If still no success, show output for debugging
        if not metrics["success"]:
            print("\n" + "="*60)
            print("FAILED TO PARSE METRICS - FULL OUTPUT:")
            print("="*60)
            print(output[-3000:] if len(output) > 3000 else output)
            print("="*60)
            metrics["error"] = f"Could not parse metrics. Last output: {output[-500:]}"

        return metrics

    except subprocess.CalledProcessError as e:
        print("\n" + "="*60)
        print(f"train.py FAILED with exit code {e.returncode}")
        print("="*60)
        if e.stdout:
            print("STDOUT:")
            print(e.stdout[-2000:] if len(e.stdout) > 2000 else e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr[-2000:] if len(e.stderr) > 2000 else e.stderr)
        print("="*60)
        error_msg = f"Exit code {e.returncode}"
        if e.stderr:
            error_msg += f": {e.stderr[-200:]}"
        return {
            "r2": float("-inf"),
            "mae": float("inf"),
            "pearson": 0.0,
            "psd_error_db": float("inf"),
            "training_time": 0.0,
            "success": False,
            "error": error_msg,
        }
    except subprocess.TimeoutExpired:
        return {
            "r2": float("-inf"),
            "mae": float("inf"),
            "pearson": 0.0,
            "psd_error_db": float("inf"),
            "training_time": 0.0,
            "success": False,
            "error": "Timeout (4h)",
        }
    except Exception as e:
        return {
            "r2": float("-inf"),
            "mae": float("inf"),
            "pearson": 0.0,
            "psd_error_db": float("inf"),
            "training_time": 0.0,
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Main Tier Function
# =============================================================================

def run_tier2_5(
    prob_losses: List[str],
    n_epochs: int = 20,
    batch_size: int = 8,
    lr: float = 0.0002,
    base_channels: int = 64,
    seed: int = 42,
    prob_loss_weight: float = 1.0,
    debug: bool = False,
    previous_results: Optional[List[Dict]] = None,
) -> Tier2_5Result:
    """Run Tier 2.5: Test different probabilistic losses.

    Uses train.py via torchrun for each probabilistic loss.

    Args:
        prob_losses: List of probabilistic losses to test
        n_epochs: Training epochs per loss
        batch_size: Batch size
        lr: Learning rate
        base_channels: UNet base channels
        seed: Random seed for reproducibility
        prob_loss_weight: Weight for probabilistic loss
        debug: If True, enable verbose debugging
        previous_results: Optional list of successful results from previous run

    Returns:
        Tier2_5Result with comparison of all probabilistic losses
    """
    print(f"\nTesting {len(prob_losses)} probabilistic losses")
    print(f"Using train.py with --no-bidirectional --skip-spectral-finetune")
    print(f"Base loss: huber_wavelet, Prob loss weight: {prob_loss_weight}")
    print(f"Epochs: {n_epochs}, Batch: {batch_size}, LR: {lr}, Seed: {seed}")
    print()

    # Start with previous successful results if provided
    all_results: List[ProbLossRunResult] = []
    if previous_results:
        for r in previous_results:
            all_results.append(ProbLossRunResult(
                prob_loss=r["prob_loss"],
                r2=r.get("r2", float("-inf")),
                mae=r.get("mae", float("inf")),
                pearson=r.get("pearson", 0.0),
                psd_error_db=r.get("psd_error_db", float("inf")),
                val_r2=r.get("val_r2", float("-inf")),
                val_mae=r.get("val_mae", float("inf")),
                val_pearson=r.get("val_pearson", 0.0),
                val_psd_error_db=r.get("val_psd_error_db", float("inf")),
                training_time=r.get("training_time", 0.0),
                success=True,
            ))
        print(f"Loaded {len(previous_results)} successful results from previous run")

    for i, prob_loss in enumerate(prob_losses):
        # CRITICAL: Clean up GPU memory before each run
        print(f"\n  Cleaning up GPU memory before run...")
        cleanup_gpu_memory()

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(prob_losses)}] Probabilistic Loss: {prob_loss}")
        print(f"{'='*60}")

        metrics = run_prob_loss_via_train_py(
            prob_loss=prob_loss,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            base_channels=base_channels,
            seed=seed,
            prob_loss_weight=prob_loss_weight,
            debug=debug,
        )

        if metrics.get("success", False):
            result = ProbLossRunResult(
                prob_loss=prob_loss,
                r2=metrics["r2"],
                mae=metrics.get("mae", float("inf")),
                pearson=metrics.get("pearson", 0.0),
                psd_error_db=metrics.get("psd_error_db", float("inf")),
                val_r2=metrics.get("val_r2", float("-inf")),
                val_mae=metrics.get("val_mae", float("inf")),
                val_pearson=metrics.get("val_pearson", 0.0),
                val_psd_error_db=metrics.get("val_psd_error_db", float("inf")),
                training_time=metrics.get("training_time", 0.0),
                success=True,
            )
            print(f"  SUCCESS: Val R²={result.val_r2:.4f}, Test R²={result.r2:.4f}, Test Pearson={result.pearson:.4f}")
        else:
            result = ProbLossRunResult(
                prob_loss=prob_loss,
                r2=float("-inf"),
                mae=float("inf"),
                pearson=0.0,
                psd_error_db=float("inf"),
                training_time=0.0,
                success=False,
                error=metrics.get("error", "Unknown error"),
            )
            print(f"  FAILED: {metrics.get('error', 'Unknown error')}")

        all_results.append(result)

        # Save intermediate checkpoint
        checkpoint = {
            "completed": [asdict(r) for r in all_results],
            "remaining": prob_losses[i+1:],
        }
        with open(ARTIFACTS_DIR / "tier2_5_checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)

    # Build results dict (for registry)
    prob_results = {}
    for result in all_results:
        if result.success:
            prob_results[result.prob_loss] = ProbabilisticLossResult(
                base_loss="huber_wavelet",
                prob_loss=result.prob_loss,
                r2_mean=result.r2,
                r2_std=0.0,  # Single run
            )

    # Find best
    if prob_results:
        best = max(prob_results.values(), key=lambda x: x.r2_mean)
        best_combination = f"huber_wavelet+{best.prob_loss}" if best.prob_loss != "none" else "huber_wavelet"
    else:
        best = ProbabilisticLossResult("huber_wavelet", "none", 0.0, 0.0)
        best_combination = "huber_wavelet"

    # Create final result
    final_result = Tier2_5Result(
        architecture="CondUNet1D",
        results=prob_results,
        best_combination=best_combination,
        best_r2_mean=best.r2_mean,
        best_r2_std=best.r2_std,
    )

    # Register results
    try:
        registry = get_registry(ARTIFACTS_DIR)
        registry.register_tier2_5(final_result)
    except Exception as e:
        print(f"  Warning: Could not register results: {e}")

    # Print summary
    print("\n" + "=" * 100)
    print("TIER 2.5 RESULTS: Probabilistic Loss Ablation")
    print("=" * 100)
    print(f"{'Prob Loss':<20} {'Val R²':>10} {'Val r':>10} {'Test R²':>10} {'Test r':>10} {'Test MAE':>10} {'Time':>10}")
    print("-" * 100)

    sorted_results = sorted(all_results, key=lambda x: x.r2 if x.success else float("-inf"), reverse=True)
    for r in sorted_results:
        if r.success:
            marker = " <-- BEST" if r.prob_loss == best.prob_loss else ""
            print(f"{r.prob_loss:<20} {r.val_r2:>10.4f} {r.val_pearson:>10.4f} {r.r2:>10.4f} {r.pearson:>10.4f} {r.mae:>10.4f} {r.training_time:>9.1f}s{marker}")
        else:
            print(f"{r.prob_loss:<20} {'FAILED':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10}  [{r.error}]")

    print("=" * 100)
    print(f"\nBest probabilistic loss: {best.prob_loss} (Test R²={best.r2_mean:.4f})")
    print(f"Best combination: {best_combination}")

    # Save final results
    final_output = {
        "tier": "2.5",
        "name": "Probabilistic Loss Ablation",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "base_channels": base_channels,
            "seed": seed,
            "prob_loss_weight": prob_loss_weight,
            "base_loss": "huber_wavelet",
            "prob_losses": prob_losses,
        },
        "results": [asdict(r) for r in all_results],
        "best_prob_loss": best.prob_loss,
        "best_combination": best_combination,
        "best_r2": best.r2_mean,
    }

    out_path = ARTIFACTS_DIR / f"tier2_5_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved: {out_path}")

    return final_result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tier 2.5: Probabilistic Loss Ablation")
    parser.add_argument("--dry-run", action="store_true", help="Quick test with fewer epochs and losses")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per loss (default: 20)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--base-channels", type=int, default=64, help="UNet base channels")
    parser.add_argument("--losses", type=str, nargs="+", default=None,
                        help="Specific probabilistic losses to test (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--prob-loss-weight", type=float, default=1.0,
                        help="Weight for probabilistic loss (default: 1.0)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debugging for distributed training errors")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Only rerun failed experiments from previous run")

    args = parser.parse_args()

    print("=" * 60)
    print("TIER 2.5: Probabilistic Loss Ablation")
    print("=" * 60)
    print()
    print("Tests probabilistic losses ON TOP of huber_wavelet base loss.")
    print("Uses train.py via torchrun with --no-bidirectional --skip-spectral-finetune")
    print()
    print("Probabilistic losses available:")
    for i, loss in enumerate(PROBABILISTIC_LOSSES, 1):
        print(f"  {i:2d}. {loss}")
    print()

    # Handle --retry-failed: load checkpoint and only run failed ones
    if args.retry_failed:
        checkpoint_path = ARTIFACTS_DIR / "tier2_5_checkpoint.json"
        if not checkpoint_path.exists():
            print("ERROR: No checkpoint found. Run without --retry-failed first.")
            return

        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        # Find failed experiments
        completed = checkpoint.get("completed", [])
        failed_losses = [r["prob_loss"] for r in completed if not r.get("success", False)]
        successful_results = [r for r in completed if r.get("success", False)]

        if not failed_losses:
            print("No failed experiments to retry!")
            print(f"All {len(completed)} experiments completed successfully.")
            return

        print(f"[RETRY-FAILED] Found {len(failed_losses)} failed experiments:")
        for loss in failed_losses:
            print(f"  - {loss}")
        print(f"\nKeeping {len(successful_results)} successful results from previous run.\n")

        prob_losses = failed_losses

        run_tier2_5(
            prob_losses=prob_losses,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            base_channels=args.base_channels,
            seed=args.seed,
            prob_loss_weight=args.prob_loss_weight,
            debug=args.debug,
            previous_results=successful_results,
        )
        return

    # Select probabilistic losses
    if args.losses:
        prob_losses = args.losses
    elif args.dry_run:
        prob_losses = FAST_PROB_LOSSES
        args.epochs = 3  # Fewer epochs for dry run
        print(f"[DRY-RUN] Testing {len(prob_losses)} probabilistic losses with {args.epochs} epochs each\n")
    else:
        prob_losses = PROBABILISTIC_LOSSES

    run_tier2_5(
        prob_losses=prob_losses,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_channels=args.base_channels,
        seed=args.seed,
        prob_loss_weight=args.prob_loss_weight,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
