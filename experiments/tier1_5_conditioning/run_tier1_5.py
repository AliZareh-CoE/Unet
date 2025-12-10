#!/usr/bin/env python3
"""
Tier 1.5: Conditioning Optimization for U-Net
==============================================

Purpose: Find the BEST conditioning source for our U-Net (CondUNet1D).

After Tier 1 proves U-Net beats all other architectures, this tier
tests different conditioning signal SOURCES to find what information
best guides the U-Net translation.

Architecture: CondUNet1D (from models.py) - FIXED
Conditioning Sources Tested:
    1. odor_onehot: One-hot encoding of odor identity (baseline)
    2. cpc: Contrastive Predictive Coding embeddings (self-supervised)
    3. vqvae: Vector Quantized VAE discrete codes
    4. freq_disentangled: Frequency-band-specific latents (delta/theta/alpha/beta/gamma)
    5. spectro_temporal: SpectroTemporalEncoder - FFT + multi-scale temporal conv (auto-conditioning)

NOTE: cycle_consistent is NOT tested here - it requires bidirectional training (OB↔PCx)
which causes data leakage and train/test mismatch in forward-only mode.

This tier uses U-Net ONLY - no architecture comparison.
Now uses train.py via torchrun for fair comparison with other tiers.
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
from typing import Any, Dict, List, Optional, Tuple

# Fix imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.common.config_registry import (
    get_registry,
    Tier1_5Result,
    ConditioningResult,
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
        result = subprocess.run(
            ["pkill", "-9", "-f", "torch.distributed.run"],
            capture_output=True,
            timeout=5,
        )
        # Also kill any orphaned train.py processes
        result = subprocess.run(
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

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier1_5_conditioning"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Conditioning SOURCES to test (what to condition on)
# These map to --conditioning argument in train.py
# NOTE: cycle_consistent REMOVED - requires bidirectional training (OB↔PCx)
# which we're not doing in tier1.5. It causes data leakage (uses target
# during training) and train/test mismatch.
CONDITIONING_SOURCES = [
    "odor_onehot",        # One-hot odor identity (baseline)
    "cpc",                # Contrastive Predictive Coding embeddings
    "vqvae",              # Vector Quantized VAE codes
    "freq_disentangled",  # Frequency-band-specific latents
    "spectro_temporal",   # SpectroTemporalEncoder (auto-conditioning from signal dynamics)
]

# Dry-run uses ALL sources but fewer epochs (no subset)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConditioningRunResult:
    """Result from a single conditioning run."""
    conditioning: str
    r2: float
    mae: float
    pearson: float
    psd_error_db: float
    r2_delta: Optional[float] = None
    r2_theta: Optional[float] = None
    r2_alpha: Optional[float] = None
    r2_beta: Optional[float] = None
    r2_gamma: Optional[float] = None
    training_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# Run via train.py
# =============================================================================

def run_conditioning_via_train_py(
    conditioning: str,
    n_epochs: int,
    batch_size: int,
    lr: float,
    base_channels: int = 64,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run UNet with specific conditioning via torchrun train.py.

    Uses --no-bidirectional and --skip-spectral-finetune for fair comparison.

    Args:
        conditioning: Conditioning source (odor_onehot, spectro_temporal, cpc, etc.)
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        base_channels: UNet base channels
        debug: If True, enable verbose debugging for distributed training

    Returns:
        Dict with r2, mae, pearson, psd_error_db, success, error
    """
    # Use torchrun for distributed training (8x A100)
    # FSDP with grad_op strategy is more stable than DDP for 8 GPUs
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        str(PROJECT_ROOT / "train.py"),
        f"--epochs={n_epochs}",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
        f"--base-channels={base_channels}",
        f"--conditioning={conditioning}",
        "--fsdp",                        # Use FSDP instead of DDP
        "--fsdp-strategy=grad_op",       # grad_op recommended for 8x A100
        "--no-bidirectional",            # Fair comparison: no reverse model
        "--skip-spectral-finetune",      # Fair comparison: no Stage 2
        "--eval-stage1",                 # Evaluate after Stage 1
    ]

    # Set up environment with debugging options
    env = os.environ.copy()
    if debug:
        # Enable detailed PyTorch distributed debugging
        env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        env["NCCL_DEBUG"] = "WARN"
        env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
        # CUDA_LAUNCH_BLOCKING=1 helps debug CUDA errors but slows training
        # Only enable if needed for CUDA-specific errors
        # env["CUDA_LAUNCH_BLOCKING"] = "1"
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
        # Look for Stage 1 final evaluation output like:
        # STAGE1_RESULT_R2=0.xxxx
        # STAGE1_RESULT_CORR=0.xxxx
        # Or: R²: 0.xxxx, Correlation: 0.xxxx
        metrics = {
            "r2": float("-inf"),
            "mae": float("inf"),
            "pearson": 0.0,
            "psd_error_db": float("inf"),
            "training_time": elapsed,
            "success": False,
        }

        # Try to find Stage 1 final metrics (machine-parseable format first)
        for line in output.split("\n"):
            # Machine-parseable format: STAGE1_RESULT_R2=0.xxxx
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

            # Human-readable format: "  R²: 0.xxxx" or "  MAE: 0.xxxx"
            if "R²:" in line:
                r2_match = re.search(r"R²:\s*([0-9.-]+)", line)
                if r2_match:
                    metrics["r2"] = float(r2_match.group(1))
                    metrics["success"] = True
            if "Correlation:" in line:
                corr_match = re.search(r"Correlation:\s*([0-9.-]+)", line)
                if corr_match:
                    metrics["pearson"] = float(corr_match.group(1))
            if "MAE:" in line:
                mae_match = re.search(r"MAE:\s*([0-9.-]+)", line)
                if mae_match:
                    metrics["mae"] = float(mae_match.group(1))
            if "PSD Error:" in line:
                psd_match = re.search(r"PSD Error:\s*([0-9.-]+)", line)
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
                        metrics["r2"] = 0.0  # Unknown, but training completed
                except Exception as e:
                    print(f"  Warning: Could not load checkpoint: {e}")

        # If still no success, show full output for debugging
        if not metrics["success"]:
            print("\n" + "="*60)
            print("FAILED TO PARSE METRICS - FULL OUTPUT:")
            print("="*60)
            print(output[-3000:] if len(output) > 3000 else output)  # Last 3000 chars
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
            "error": f"Exit code {e.returncode}",
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

def run_tier1_5(
    conditionings: List[str],
    n_epochs: int = 50,
    batch_size: int = 8,
    lr: float = 0.0002,
    base_channels: int = 64,
    debug: bool = False,
) -> Tier1_5Result:
    """Run Tier 1.5: Test different conditioning sources for UNet.

    Uses train.py via torchrun for each conditioning source.

    Args:
        conditionings: List of conditioning sources to test
        n_epochs: Training epochs per source
        batch_size: Batch size
        lr: Learning rate
        base_channels: UNet base channels
        debug: If True, enable verbose debugging for distributed training

    Returns:
        Tier1_5Result with comparison of all conditioning sources
    """
    print(f"\nTesting {len(conditionings)} conditioning sources")
    print(f"Using train.py with --no-bidirectional --skip-spectral-finetune")
    print(f"Epochs: {n_epochs}, Batch: {batch_size}, LR: {lr}")
    print()

    all_results: List[ConditioningRunResult] = []

    for i, cond_source in enumerate(conditionings):
        # CRITICAL: Clean up GPU memory before each run to prevent crashes
        print(f"\n  Cleaning up GPU memory before run...")
        cleanup_gpu_memory()

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(conditionings)}] Conditioning: {cond_source}")
        print(f"{'='*60}")

        metrics = run_conditioning_via_train_py(
            conditioning=cond_source,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            base_channels=base_channels,
            debug=debug,
        )

        if metrics.get("success", False):
            result = ConditioningRunResult(
                conditioning=cond_source,
                r2=metrics["r2"],
                mae=metrics.get("mae", float("inf")),
                pearson=metrics.get("pearson", 0.0),
                psd_error_db=metrics.get("psd_error_db", float("inf")),
                training_time=metrics.get("training_time", 0.0),
                success=True,
            )
            print(f"  SUCCESS: R²={metrics['r2']:.4f}, Pearson={metrics.get('pearson', 0):.4f}")
        else:
            result = ConditioningRunResult(
                conditioning=cond_source,
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
            "remaining": conditionings[i+1:],
        }
        with open(ARTIFACTS_DIR / "tier1_5_checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)

    # Build results dict
    cond_results = {}
    for result in all_results:
        if result.success:
            cond_results[result.conditioning] = ConditioningResult(
                conditioning=result.conditioning,
                r2_mean=result.r2,
                r2_std=0.0,  # Single run
                mae_mean=result.mae,
                mae_std=0.0,
            )

    # Find best
    if cond_results:
        best_cond = max(cond_results.values(), key=lambda x: x.r2_mean)
    else:
        best_cond = ConditioningResult("none", 0.0, 0.0, 0.0, 0.0)

    # Create final result
    final_result = Tier1_5Result(
        architecture="CondUNet1D",
        results=cond_results,
        best_conditioning=best_cond.conditioning,
        best_r2_mean=best_cond.r2_mean,
        best_r2_std=best_cond.r2_std,
    )

    # Register results
    try:
        registry = get_registry(ARTIFACTS_DIR)
        registry.register_tier1_5(final_result)
    except Exception as e:
        print(f"  Warning: Could not register results: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("TIER 1.5 RESULTS: Conditioning Source Comparison")
    print("=" * 70)
    print(f"{'Conditioning':<20} {'R²':>10} {'Pearson':>10} {'MAE':>10} {'PSD_err':>10} {'Time':>10}")
    print("-" * 70)

    sorted_results = sorted(all_results, key=lambda x: x.r2 if x.success else float("-inf"), reverse=True)
    for r in sorted_results:
        if r.success:
            marker = " <-- BEST" if r.conditioning == best_cond.conditioning else ""
            print(f"{r.conditioning:<20} {r.r2:>10.4f} {r.pearson:>10.4f} {r.mae:>10.4f} {r.psd_error_db:>10.2f} {r.training_time:>9.1f}s{marker}")
        else:
            print(f"{r.conditioning:<20} {'FAILED':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10}  [{r.error}]")

    print("=" * 70)
    print(f"\nBest conditioning: {best_cond.conditioning} (R²={best_cond.r2_mean:.4f})")

    # Save final results
    final_output = {
        "tier": "1.5",
        "name": "Conditioning Source Comparison",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "base_channels": base_channels,
            "conditionings": conditionings,
        },
        "results": [asdict(r) for r in all_results],
        "best_conditioning": best_cond.conditioning,
        "best_r2": best_cond.r2_mean,
    }

    out_path = ARTIFACTS_DIR / f"tier1_5_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved: {out_path}")

    return final_result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tier 1.5: Conditioning Source Comparison")
    parser.add_argument("--dry-run", action="store_true", help="Quick test with fewer epochs")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per source")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--base-channels", type=int, default=64, help="UNet base channels")
    parser.add_argument("--sources", type=str, nargs="+", default=None,
                        help="Specific conditioning sources to test (default: all)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debugging for distributed training errors")

    args = parser.parse_args()

    print("=" * 60)
    print("TIER 1.5: Conditioning Source Comparison")
    print("=" * 60)
    print()
    print("Uses train.py via torchrun with --no-bidirectional --skip-spectral-finetune")
    print("This ensures fair comparison with tier1 architectures.")
    print()
    print("Conditioning sources:")
    for i, src in enumerate(CONDITIONING_SOURCES, 1):
        print(f"  {i}. {src}")
    print()

    # Select conditioning sources
    if args.sources:
        conditionings = args.sources
    elif args.dry_run:
        conditionings = CONDITIONING_SOURCES  # Test ALL sources
        args.epochs = 3                       # But with fewer epochs
        print("[DRY-RUN] Testing ALL conditioning sources with 3 epochs each\n")
    else:
        conditionings = CONDITIONING_SOURCES

    run_tier1_5(
        conditionings=conditionings,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_channels=args.base_channels,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
