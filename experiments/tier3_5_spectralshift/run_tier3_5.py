#!/usr/bin/env python3
"""
Tier 3.5: SpectralShift Variants
=================================

Tests different approaches to correct PSD mismatch between predicted and target signals.

Approaches:
    1. none: No correction (baseline)
    2. hard_fft: Direct FFT magnitude replacement
    3. soft_01 to soft_05: Soft interpolation with different beta values
    4. adaptive_gated: Learned gating for frequency-dependent correction
    5. wavelet: Wavelet-domain correction
    6. band_specific: Per-frequency-band correction

KEY METRIC: R² preservation while PSD improves
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.common.cross_validation import create_data_splits
from experiments.common.config_registry import (
    get_registry,
    Tier3_5Result,
    SpectralShiftResult,
)


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier3_5_spectralshift"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SPECTRAL_METHODS = [
    "none",
    "hard_fft",
    "soft_01",
    "soft_02",
    "soft_03",
    "soft_05",
    "adaptive_gated",
    "wavelet",
    "band_specific",
]

FAST_METHODS = ["none", "hard_fft", "soft_03"]


# =============================================================================
# SpectralShift Implementations
# =============================================================================

class SpectralShiftNone(nn.Module):
    """No spectral correction (baseline)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        return pred


class SpectralShiftHardFFT(nn.Module):
    """Hard FFT magnitude replacement.

    Replaces predicted magnitude with target magnitude, keeps predicted phase.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)

        # Get magnitude and phase
        pred_mag = torch.abs(pred_fft)
        pred_phase = torch.angle(pred_fft)
        target_mag = torch.abs(target_fft)

        # Replace magnitude
        corrected_fft = target_mag * torch.exp(1j * pred_phase)

        # Inverse FFT
        corrected = torch.fft.irfft(corrected_fft, n=pred.shape[-1], dim=-1)

        return corrected


class SpectralShiftSoft(nn.Module):
    """Soft spectral correction with interpolation.

    Interpolates between predicted and target magnitudes.
    """

    def __init__(self, beta: float = 0.3):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)

        pred_mag = torch.abs(pred_fft)
        pred_phase = torch.angle(pred_fft)
        target_mag = torch.abs(target_fft)

        # Soft interpolation
        corrected_mag = (1 - self.beta) * pred_mag + self.beta * target_mag

        corrected_fft = corrected_mag * torch.exp(1j * pred_phase)
        corrected = torch.fft.irfft(corrected_fft, n=pred.shape[-1], dim=-1)

        return corrected


class SpectralShiftAdaptiveGated(nn.Module):
    """Adaptive gated spectral correction.

    Learns frequency-dependent gating for correction.
    """

    def __init__(self, n_freq: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(n_freq * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_freq),
            nn.Sigmoid(),
        )
        self.n_freq = n_freq

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, T = pred.shape

        pred_fft = torch.fft.rfft(pred, n=self.n_freq * 2, dim=-1)
        target_fft = torch.fft.rfft(target, n=self.n_freq * 2, dim=-1)

        pred_mag = torch.abs(pred_fft)
        pred_phase = torch.angle(pred_fft)
        target_mag = torch.abs(target_fft)

        # Compute gate based on magnitude difference
        gate_input = torch.cat([
            pred_mag.reshape(B, -1),
            target_mag.reshape(B, -1),
        ], dim=-1)

        # Pad/truncate to expected size
        expected_size = self.n_freq * 2 * 2
        if gate_input.shape[-1] < expected_size:
            gate_input = F.pad(gate_input, (0, expected_size - gate_input.shape[-1]))
        else:
            gate_input = gate_input[..., :expected_size]

        gate = self.gate_net(gate_input)  # [B, n_freq]
        gate = gate.unsqueeze(1).expand(-1, C, -1)  # [B, C, n_freq]

        # Pad gate to match FFT size
        if gate.shape[-1] < pred_mag.shape[-1]:
            gate = F.pad(gate, (0, pred_mag.shape[-1] - gate.shape[-1]))
        else:
            gate = gate[..., :pred_mag.shape[-1]]

        # Apply gated correction
        corrected_mag = (1 - gate) * pred_mag + gate * target_mag
        corrected_fft = corrected_mag * torch.exp(1j * pred_phase)
        corrected = torch.fft.irfft(corrected_fft, n=T, dim=-1)

        return corrected


class SpectralShiftWavelet(nn.Module):
    """Wavelet-domain spectral correction.

    Uses multi-scale decomposition for correction.
    """

    def __init__(self, n_scales: int = 4):
        super().__init__()
        self.n_scales = n_scales

    def _wavelet_decompose(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Simple Haar wavelet decomposition."""
        coeffs = []
        current = x

        for _ in range(self.n_scales):
            if current.shape[-1] < 2:
                break

            # Downsample
            low = (current[..., 0::2] + current[..., 1::2]) / 2
            high = (current[..., 0::2] - current[..., 1::2]) / 2

            coeffs.append(high)
            current = low

        coeffs.append(current)
        return coeffs

    def _wavelet_reconstruct(self, coeffs: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct from wavelet coefficients."""
        current = coeffs[-1]

        for high in reversed(coeffs[:-1]):
            # Upsample
            T = high.shape[-1] * 2
            low_up = torch.zeros(*current.shape[:-1], T, device=current.device)
            low_up[..., 0::2] = current
            low_up[..., 1::2] = current

            high_up = torch.zeros(*high.shape[:-1], T, device=high.device)
            high_up[..., 0::2] = high
            high_up[..., 1::2] = -high

            current = low_up + high_up

        return current

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Decompose
        pred_coeffs = self._wavelet_decompose(pred)
        target_coeffs = self._wavelet_decompose(target)

        # Correct magnitudes at each scale
        corrected_coeffs = []
        for p, t in zip(pred_coeffs, target_coeffs):
            # Match standard deviation (simple magnitude matching)
            p_std = torch.std(p, dim=-1, keepdim=True) + 1e-8
            t_std = torch.std(t, dim=-1, keepdim=True) + 1e-8
            corrected = p * (t_std / p_std)
            corrected_coeffs.append(corrected)

        # Reconstruct
        corrected = self._wavelet_reconstruct(corrected_coeffs)

        # Truncate to original length
        return corrected[..., :pred.shape[-1]]


class SpectralShiftBandSpecific(nn.Module):
    """Per-frequency-band correction.

    Applies different correction strengths to different frequency bands.
    """

    def __init__(self, sample_rate: float = 1000.0):
        super().__init__()
        self.sample_rate = sample_rate

        # Define bands (Hz)
        self.bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 30),
            "gamma": (30, 100),
        }

        # Learnable correction strength per band
        self.band_weights = nn.Parameter(torch.ones(len(self.bands)) * 0.3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, T = pred.shape

        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)

        pred_mag = torch.abs(pred_fft)
        pred_phase = torch.angle(pred_fft)
        target_mag = torch.abs(target_fft)

        # Frequency bins
        n_freq = pred_fft.shape[-1]
        freqs = torch.fft.rfftfreq(T, 1 / self.sample_rate).to(pred.device)

        # Create band mask
        corrected_mag = pred_mag.clone()

        for i, (band_name, (low, high)) in enumerate(self.bands.items()):
            mask = (freqs >= low) & (freqs < high)
            beta = torch.sigmoid(self.band_weights[i])  # 0-1

            corrected_mag[..., mask] = (
                (1 - beta) * pred_mag[..., mask] +
                beta * target_mag[..., mask]
            )

        corrected_fft = corrected_mag * torch.exp(1j * pred_phase)
        corrected = torch.fft.irfft(corrected_fft, n=T, dim=-1)

        return corrected


SPECTRAL_SHIFT_REGISTRY = {
    "none": SpectralShiftNone,
    "hard_fft": SpectralShiftHardFFT,
    "soft_01": lambda: SpectralShiftSoft(beta=0.1),
    "soft_02": lambda: SpectralShiftSoft(beta=0.2),
    "soft_03": lambda: SpectralShiftSoft(beta=0.3),
    "soft_05": lambda: SpectralShiftSoft(beta=0.5),
    "adaptive_gated": SpectralShiftAdaptiveGated,
    "wavelet": SpectralShiftWavelet,
    "band_specific": SpectralShiftBandSpecific,
}


# =============================================================================
# Metrics
# =============================================================================

def compute_psd_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute PSD error in dB."""
    pred_psd = np.abs(np.fft.rfft(pred, axis=-1)) ** 2
    target_psd = np.abs(np.fft.rfft(target, axis=-1)) ** 2

    # Avoid log of zero
    pred_psd = np.maximum(pred_psd, 1e-10)
    target_psd = np.maximum(target_psd, 1e-10)

    # Log PSD difference (dB)
    log_diff = 10 * np.log10(pred_psd) - 10 * np.log10(target_psd)
    psd_error = np.mean(np.abs(log_diff))

    return float(psd_error)


def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute R² score."""
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat, axis=0)) ** 2)

    return float(1 - ss_res / (ss_tot + 1e-8))


# =============================================================================
# Main Tier Function
# =============================================================================

def run_tier3_5(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    base_predictions: Optional[torch.Tensor] = None,
    methods: Optional[List[str]] = None,
    n_seeds: int = 3,
    n_folds: int = 3,
    n_epochs: int = 20,
    register: bool = True,
) -> Tier3_5Result:
    """Run Tier 3.5: SpectralShift Variants.

    Args:
        X: Input data
        y: Target data
        device: Device
        base_predictions: Pre-computed predictions from winning model (optional)
        methods: SpectralShift methods to test
        n_seeds: Number of seeds for learnable methods
        n_folds: Number of CV folds
        n_epochs: Epochs for learnable methods
        register: Whether to register results

    Returns:
        Tier3_5Result
    """
    methods = methods or SPECTRAL_METHODS

    print(f"\nTesting {len(methods)} SpectralShift methods")

    N = X.shape[0]
    splits = create_data_splits(n_samples=N, n_folds=n_folds, holdout_fraction=0.0, seed=42)

    # If no base predictions, we need to generate them using the winning model
    if base_predictions is None:
        print("  Generating base predictions from model...")
        # Use a simple model for demo
        from models import UNet1DConditioned

        model = UNet1DConditioned(
            in_channels=X.shape[1],
            out_channels=y.shape[1],
            base_channels=32,
        ).to(device)

        # Quick training
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

        model.train()
        for epoch in range(20):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = F.mse_loss(pred, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            base_predictions = model(X.to(device)).cpu()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    all_results = {}

    for method in methods:
        print(f"\n  Method: {method}...", end=" ", flush=True)

        r2s = []
        psd_errors = []

        for seed in range(n_seeds):
            torch.manual_seed(seed)

            for split in splits.cv_splits:
                pred_val = base_predictions[split.val_indices]
                y_val = y[split.val_indices]

                try:
                    # Create SpectralShift module
                    if method in SPECTRAL_SHIFT_REGISTRY:
                        ss_module = SPECTRAL_SHIFT_REGISTRY[method]()
                        if callable(ss_module) and not isinstance(ss_module, nn.Module):
                            ss_module = ss_module()
                        ss_module = ss_module.to(device)
                    else:
                        continue

                    # For learnable methods, train briefly
                    if method in ["adaptive_gated", "band_specific"]:
                        optimizer = torch.optim.Adam(ss_module.parameters(), lr=1e-3)

                        for _ in range(n_epochs):
                            pred_batch = pred_val.to(device)
                            y_batch = y_val.to(device)

                            optimizer.zero_grad()
                            corrected = ss_module(pred_batch, y_batch)
                            loss = F.mse_loss(corrected, y_batch)
                            loss.backward()
                            optimizer.step()

                    # Evaluate
                    ss_module.eval()
                    with torch.no_grad():
                        corrected = ss_module(
                            pred_val.to(device),
                            y_val.to(device)
                        ).cpu().numpy()

                    y_val_np = y_val.numpy()

                    r2 = compute_r2(corrected, y_val_np)
                    psd_err = compute_psd_error(corrected, y_val_np)

                    r2s.append(r2)
                    psd_errors.append(psd_err)

                except Exception as e:
                    print(f"Error: {e}")

                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

        if r2s:
            all_results[method] = SpectralShiftResult(
                method=method,
                r2_mean=float(np.mean(r2s)),
                r2_std=float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
                psd_error_mean=float(np.mean(psd_errors)),
                psd_error_std=float(np.std(psd_errors, ddof=1)) if len(psd_errors) > 1 else 0.0,
            )
            print(f"R² = {all_results[method].r2_mean:.4f}, PSD err = {all_results[method].psd_error_mean:.2f} dB")

    # Find best (highest R² with PSD error < 3dB)
    valid_results = [r for r in all_results.values() if r.psd_error_mean < 3.0]
    if valid_results:
        best = max(valid_results, key=lambda x: x.r2_mean)
    else:
        best = max(all_results.values(), key=lambda x: x.r2_mean)

    result = Tier3_5Result(
        results=all_results,
        best_method=best.method,
        best_r2_mean=best.r2_mean,
        best_psd_error=best.psd_error_mean,
    )

    if register:
        registry = get_registry(ARTIFACTS_DIR)
        registry.register_tier3_5(result)

    # Print summary
    print("\n" + "=" * 60)
    print("TIER 3.5 RESULTS: SpectralShift Variants")
    print("=" * 60)
    print(f"{'Method':<20} {'R²':>12} {'PSD Error (dB)':>15}")
    print("-" * 50)

    sorted_results = sorted(all_results.values(), key=lambda x: x.r2_mean, reverse=True)
    for r in sorted_results:
        marker = " <-- BEST" if r.method == best.method else ""
        print(f"  {r.method:<18} {r.r2_mean:>10.4f} {r.psd_error_mean:>13.2f}{marker}")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tier 3.5: SpectralShift Variants")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TIER 3.5: SpectralShift Variants")
    print("=" * 60)

    if args.dry_run:
        N, C, T = 100, 32, 500
        X = torch.randn(N, C, T)
        y = torch.randn(N, C, T)
        methods = FAST_METHODS
        n_seeds, n_folds = 2, 2
    else:
        from data import prepare_data
        data = prepare_data()
        idx = np.concatenate([data["train_idx"], data["val_idx"]])
        X = torch.from_numpy(data["ob"][idx]).float()
        y = torch.from_numpy(data["pcx"][idx]).float()
        methods = SPECTRAL_METHODS
        n_seeds, n_folds = 3, 3

    run_tier3_5(
        X=X, y=y, device=device,
        methods=methods,
        n_seeds=n_seeds, n_folds=n_folds,
    )


if __name__ == "__main__":
    main()
