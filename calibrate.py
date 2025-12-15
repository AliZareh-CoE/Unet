"""Stage 2: Post-hoc calibration for neural signal translation.

This script applies spectral bias correction and envelope matching to a trained UNet model.
Run after training Stage 1 with train.py.

Usage:
    python calibrate.py --checkpoint artifacts/checkpoints/best_model.pt

    # With custom test sessions:
    python calibrate.py --checkpoint artifacts/checkpoints/best_model.pt --test-sessions 170614 170609
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

# Local imports
from models import (
    CondUNet1D,
    build_wavelet_loss,
    pearson_batch,
    psd_error_db_torch,
    psd_diff_db_torch,
    hilbert_torch,
)

from data import (
    prepare_data,
    create_dataloaders,
    create_single_session_dataloader,
    SAMPLING_RATE_HZ,
)

# Import spectral shift and envelope matching
try:
    from models import SpectralShift, OptimalSpectralBias, EnvelopeHistogramMatching
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False
    print("WARNING: SpectralShift not available")

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = Path("artifacts")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CALIBRATION_DIR = OUTPUT_DIR / "calibration"

# Default config (should match train.py defaults)
DEFAULT_CONFIG = {
    "in_channels": 32,
    "out_channels": 32,
    "base_channels": 64,
    "depth": 4,
    "use_attention": True,
    "attention_heads": 4,
    "dropout": 0.1,
    "n_odors": 7,
    "per_channel_norm": True,
    "sampling_rate": SAMPLING_RATE_HZ,
    "use_envelope_matching": True,
    "spectral_shift_compute_bias": True,
}


def per_channel_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-channel normalization: (B, C, T) -> normalized per channel."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=eps)
    return (x - mean) / std


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    reverse_model: Optional[nn.Module] = None,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, Any]:
    """Load model checkpoint and return config."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load reverse model if available
    if reverse_model is not None and "reverse_model_state_dict" in checkpoint:
        reverse_model.load_state_dict(checkpoint["reverse_model_state_dict"])

    # Extract config
    config = checkpoint.get("config", DEFAULT_CONFIG.copy())

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    return config


def compute_spectral_bias(
    model: nn.Module,
    spectral_shift: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict[str, Any],
) -> None:
    """Compute optimal spectral bias from UNet outputs vs targets."""
    model.eval()

    all_unet_outputs = []
    all_targets = []
    all_odor_ids = []

    with torch.no_grad():
        for ob_batch, pcx_batch, odor_batch in tqdm(loader, desc="Computing spectral bias"):
            ob_batch = ob_batch.to(device)
            pcx_batch = pcx_batch.to(device)
            odor_batch = odor_batch.to(device)

            # Apply per-channel normalization if enabled
            if config.get("per_channel_norm", True):
                ob_batch = per_channel_normalize(ob_batch)
                pcx_batch = per_channel_normalize(pcx_batch)

            # Forward pass through UNet
            unet_out = model(ob_batch, odor_batch)

            all_unet_outputs.append(unet_out.cpu())
            all_targets.append(pcx_batch.cpu())
            all_odor_ids.append(odor_batch.cpu())

    # Concatenate all batches
    all_unet_outputs = torch.cat(all_unet_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_odor_ids = torch.cat(all_odor_ids, dim=0)

    print(f"Collected {len(all_unet_outputs)} samples for bias computation")

    # Set optimal bias
    spectral_shift.set_bias_from_data(
        all_unet_outputs.to(device),
        all_targets.to(device),
        all_odor_ids.to(device),
    )


def compute_envelope_matching(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    n_odors: int = 7,
) -> EnvelopeHistogramMatching:
    """Compute envelope histogram matching from target data."""
    envelope_matcher = EnvelopeHistogramMatching(n_odors=n_odors).to(device)

    all_targets = []
    all_odor_ids = []

    with torch.no_grad():
        for ob_batch, pcx_batch, odor_batch in tqdm(loader, desc="Computing envelope stats"):
            pcx_batch = pcx_batch.to(device)
            odor_batch = odor_batch.to(device)

            if config.get("per_channel_norm", True):
                pcx_batch = per_channel_normalize(pcx_batch)

            all_targets.append(pcx_batch.cpu())
            all_odor_ids.append(odor_batch.cpu())

    all_targets = torch.cat(all_targets, dim=0)
    all_odor_ids = torch.cat(all_odor_ids, dim=0)

    # Fit envelope matcher
    envelope_matcher.fit(all_targets.to(device), all_odor_ids.to(device))

    return envelope_matcher


def evaluate_session(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    spectral_shift: Optional[nn.Module] = None,
    envelope_matcher: Optional[nn.Module] = None,
    session_name: str = "test",
) -> Dict[str, float]:
    """Evaluate model on a single session."""
    model.eval()
    if spectral_shift is not None:
        spectral_shift.eval()
    if envelope_matcher is not None:
        envelope_matcher.eval()

    all_preds = []
    all_targets = []
    all_baselines = []

    with torch.no_grad():
        for ob_batch, pcx_batch, odor_batch in loader:
            ob_batch = ob_batch.to(device)
            pcx_batch = pcx_batch.to(device)
            odor_batch = odor_batch.to(device)

            if config.get("per_channel_norm", True):
                ob_batch = per_channel_normalize(ob_batch)
                pcx_batch = per_channel_normalize(pcx_batch)

            # Forward pass
            pred = model(ob_batch, odor_batch)

            # Apply spectral shift if available
            if spectral_shift is not None:
                pred = spectral_shift(pred, odor_ids=odor_batch)

            # Apply envelope matching if available
            if envelope_matcher is not None:
                pred = envelope_matcher(pred, odor_ids=odor_batch)

            all_preds.append(pred.cpu())
            all_targets.append(pcx_batch.cpu())
            all_baselines.append(ob_batch.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_baselines = torch.cat(all_baselines, dim=0)

    # Compute metrics
    corr = pearson_batch(all_preds, all_targets).mean().item()

    # Baseline correlation
    baseline_corr = pearson_batch(all_baselines, all_targets).mean().item()

    # R²
    ss_res = ((all_preds - all_targets) ** 2).sum()
    ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()

    # PSD metrics
    sampling_rate = config.get("sampling_rate", SAMPLING_RATE_HZ)
    psd_err = psd_error_db_torch(all_preds, all_targets, fs=sampling_rate).item()
    psd_bias = psd_diff_db_torch(all_preds, all_targets, fs=sampling_rate).item()

    return {
        "session": session_name,
        "n_trials": len(all_preds),
        "corr": corr,
        "baseline_corr": baseline_corr,
        "delta_corr": corr - baseline_corr,
        "r2": r2,
        "psd_err_db": psd_err,
        "psd_bias_db": psd_bias,
    }


def save_calibrated_model(
    model: nn.Module,
    spectral_shift: nn.Module,
    envelope_matcher: Optional[nn.Module],
    config: Dict[str, Any],
    output_path: Path,
    reverse_model: Optional[nn.Module] = None,
    spectral_shift_rev: Optional[nn.Module] = None,
    envelope_matcher_rev: Optional[nn.Module] = None,
):
    """Save calibrated model with all components."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "spectral_shift_fwd_state_dict": spectral_shift.state_dict(),
        "config": config,
    }

    if envelope_matcher is not None:
        checkpoint["envelope_matcher_fwd_state_dict"] = envelope_matcher.state_dict()

    if reverse_model is not None:
        checkpoint["reverse_model_state_dict"] = reverse_model.state_dict()

    if spectral_shift_rev is not None:
        checkpoint["spectral_shift_rev_state_dict"] = spectral_shift_rev.state_dict()

    if envelope_matcher_rev is not None:
        checkpoint["envelope_matcher_rev_state_dict"] = envelope_matcher_rev.state_dict()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Saved calibrated model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Post-hoc calibration")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Stage 1 checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for calibrated model (default: artifacts/checkpoints/calibrated_model.pt)")
    parser.add_argument("--test-sessions", type=str, nargs="+", default=None,
                        help="Explicit session names for test evaluation")
    parser.add_argument("--val-sessions", type=str, nargs="+", default=None,
                        help="Explicit session names for validation")
    parser.add_argument("--n-test-sessions", type=int, default=4,
                        help="Number of sessions to use for testing (default: 4)")
    parser.add_argument("--n-val-sessions", type=int, default=1,
                        help="Number of sessions to use for validation (default: 1)")
    parser.add_argument("--no-envelope", action="store_true",
                        help="Skip envelope histogram matching")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--force-recreate-splits", action="store_true",
                        help="Force recreate train/val/test splits")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = Path(args.output) if args.output else CHECKPOINT_DIR / "calibrated_model.pt"

    print("\n" + "=" * 70)
    print("STAGE 2: POST-HOC CALIBRATION")
    print("=" * 70)

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", DEFAULT_CONFIG.copy())

    # Create model
    print("\nCreating model...")
    model = CondUNet1D(
        in_channels=config.get("in_channels", 32),
        out_channels=config.get("out_channels", 32),
        base_channels=config.get("base_channels", 64),
        depth=config.get("depth", 4),
        use_attention=config.get("use_attention", True),
        attention_heads=config.get("attention_heads", 4),
        dropout=config.get("dropout", 0.1),
        n_conditions=config.get("n_odors", 7),
        cond_mode=config.get("cond_mode", "film"),
    ).to(device)

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    print(f"  Model loaded from: {checkpoint_path}")
    print(f"  Config: depth={config.get('depth')}, base_channels={config.get('base_channels')}")

    # Load data with session-based splits
    print("\nLoading data...")
    data = prepare_data(
        split_by_session=True,
        n_test_sessions=args.n_test_sessions,
        n_val_sessions=args.n_val_sessions,
        test_sessions=args.test_sessions,
        val_sessions=args.val_sessions,
        force_recreate_splits=args.force_recreate_splits,
    )

    # Create dataloaders
    loaders = create_dataloaders(
        data,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Create SpectralShift
    if not SPECTRAL_AVAILABLE:
        raise ImportError("SpectralShift not available. Cannot run calibration.")

    n_odors = config.get("n_odors", 7)
    spectral_shift = OptimalSpectralBias(
        n_channels=config.get("out_channels", 32),
        n_odors=n_odors,
        fs=config.get("sampling_rate", SAMPLING_RATE_HZ),
    ).to(device)

    # =========================================================================
    # COMPUTE OPTIMAL SPECTRAL BIAS
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPUTING OPTIMAL SPECTRAL BIAS FROM TRAIN DATA")
    print("=" * 70)

    compute_spectral_bias(model, spectral_shift, loaders["train"], device, config)

    # Freeze spectral shift
    for param in spectral_shift.parameters():
        param.requires_grad = False

    # =========================================================================
    # COMPUTE ENVELOPE MATCHING
    # =========================================================================
    envelope_matcher = None
    if not args.no_envelope:
        print("\n" + "=" * 70)
        print("COMPUTING ENVELOPE HISTOGRAM MATCHING FROM TRAIN DATA")
        print("=" * 70)

        envelope_matcher = compute_envelope_matching(
            loaders["train"], device, config, n_odors=n_odors
        )

    # =========================================================================
    # PER-SESSION TEST EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PER-SESSION TEST EVALUATION")
    print("=" * 70)

    # Get test session info
    split_info = data.get("split_info", {})
    test_sessions = split_info.get("test_sessions", ["test"])
    session_ids = data.get("session_ids")
    idx_to_session = data.get("idx_to_session", {})
    test_idx = data["test_idx"]

    # Create session name to int ID mapping
    session_to_id = {name: idx for idx, name in idx_to_session.items()}

    results = []

    for session_name in test_sessions:
        # Find indices for this session
        session_id = session_to_id.get(session_name)
        if session_id is None:
            print(f"  Warning: Session {session_name} not found in mapping")
            continue

        # Get indices for this session that are in test set
        session_mask = session_ids[test_idx] == session_id
        session_indices = test_idx[session_mask]

        if len(session_indices) == 0:
            print(f"  Warning: No test samples for session {session_name}")
            continue

        # Create dataloader for this session
        session_loader = create_single_session_dataloader(
            data, session_name, session_indices,
            batch_size=args.batch_size,
            num_workers=4,
        )

        # Evaluate
        metrics = evaluate_session(
            model, session_loader, device, config,
            spectral_shift=spectral_shift,
            envelope_matcher=envelope_matcher,
            session_name=session_name,
        )
        results.append(metrics)

        print(f"\n  Session {session_name}: {metrics['n_trials']} trials")
        print(f"    Correlation: {metrics['corr']:.4f} (baseline: {metrics['baseline_corr']:.4f}, Δ={metrics['delta_corr']:+.4f})")
        print(f"    R²: {metrics['r2']:.4f}")
        print(f"    PSD Error: {metrics['psd_err_db']:.2f} dB")

    # Print summary table
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY: PER-SESSION TEST RESULTS")
        print("=" * 70)
        print(f"{'Session':<15} {'Trials':>8} {'Corr':>8} {'Baseline':>10} {'Δ Corr':>10} {'R²':>8} {'PSD_err':>10}")
        print("-" * 70)

        for r in results:
            print(f"{r['session']:<15} {r['n_trials']:>8} {r['corr']:>8.4f} {r['baseline_corr']:>10.4f} {r['delta_corr']:>+10.4f} {r['r2']:>8.4f} {r['psd_err_db']:>10.2f}dB")

        # Compute aggregate stats
        avg_corr = np.mean([r['corr'] for r in results])
        avg_baseline = np.mean([r['baseline_corr'] for r in results])
        avg_delta = np.mean([r['delta_corr'] for r in results])
        avg_r2 = np.mean([r['r2'] for r in results])
        avg_psd = np.mean([r['psd_err_db'] for r in results])
        total_trials = sum(r['n_trials'] for r in results)

        print("-" * 70)
        print(f"{'AVERAGE':<15} {total_trials:>8} {avg_corr:>8.4f} {avg_baseline:>10.4f} {avg_delta:>+10.4f} {avg_r2:>8.4f} {avg_psd:>10.2f}dB")
        print("=" * 70)

        # Machine-parseable output
        print(f"\nRESULT_CORR={avg_corr:.4f}")
        print(f"RESULT_BASELINE_CORR={avg_baseline:.4f}")
        print(f"RESULT_DELTA_CORR={avg_delta:.4f}")
        print(f"RESULT_R2={avg_r2:.4f}")
        print(f"RESULT_PSD_ERR={avg_psd:.2f}")

    # =========================================================================
    # SAVE CALIBRATED MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING CALIBRATED MODEL")
    print("=" * 70)

    save_calibrated_model(
        model, spectral_shift, envelope_matcher, config, output_path
    )

    # Save results JSON
    results_path = CALIBRATION_DIR / "calibration_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "per_session_results": results,
            "aggregate": {
                "avg_corr": avg_corr,
                "avg_baseline_corr": avg_baseline,
                "avg_delta_corr": avg_delta,
                "avg_r2": avg_r2,
                "avg_psd_err_db": avg_psd,
                "total_trials": total_trials,
                "n_sessions": len(results),
            },
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        }, f, indent=2)
    print(f"Saved results to: {results_path}")

    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
