"""
Multi-Scale Spectral Loss
=========================

Instead of post-hoc spectral correction, add spectral matching as a
LOSS FUNCTION during training. This guides the U-Net to produce correct
PSD naturally, without needing a separate SpectralShift block.

Key insight: If the model learns to produce correct spectra during
training, no post-hoc correction is needed, eliminating the R²
degradation problem entirely.

Search parameters:
- spectral_scales: FFT sizes for multi-scale matching
- use_log_psd: Whether to match in log space
- beta_weight, gamma_weight: Per-band weights
- spectral_loss_weight: Overall loss weight
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Neural frequency bands with default weights
BAND_DEFAULTS = {
    "delta": {"range": (1, 4), "weight": 1.0},
    "theta": {"range": (4, 8), "weight": 1.0},
    "alpha": {"range": (8, 12), "weight": 1.0},
    "beta": {"range": (12, 30), "weight": 2.0},   # Higher weight - often problematic
    "gamma": {"range": (30, 100), "weight": 3.0},  # Highest weight - most problematic
}


class MultiScaleSpectralLoss(nn.Module):
    """Multi-scale spectral loss for training-time PSD guidance.

    This loss computes PSD matching at multiple FFT scales and with
    per-band weighting. It can be added to the training loss to guide
    the model to produce correct spectra without post-hoc correction.

    Args:
        scales: List of FFT sizes for multi-scale matching
        use_log_psd: Whether to compute loss in log space (dB)
        band_weights: Per-band weighting (higher = more emphasis)
        sample_rate: Sampling rate in Hz
        eps: Small value for numerical stability
    """

    def __init__(
        self,
        scales: Optional[List[int]] = None,
        use_log_psd: bool = True,
        band_weights: Optional[Dict[str, float]] = None,
        sample_rate: float = 1000.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.scales = scales or [256, 512, 1024, 2048]
        self.use_log_psd = use_log_psd
        self.sample_rate = sample_rate
        self.eps = eps

        # Band weights
        self.band_weights = band_weights or {
            band: info["weight"] for band, info in BAND_DEFAULTS.items()
        }

        # Store band ranges
        self.band_ranges = {
            band: info["range"] for band, info in BAND_DEFAULTS.items()
        }

    def _compute_psd(
        self,
        signal: torch.Tensor,
        n_fft: int,
    ) -> torch.Tensor:
        """Compute power spectral density.

        Args:
            signal: Input signal [B, C, T]
            n_fft: FFT size

        Returns:
            PSD [B, C, n_fft//2+1]
        """
        B, C, T = signal.shape

        # Pad if necessary
        if T < n_fft:
            signal = F.pad(signal, (0, n_fft - T))

        # Apply window to reduce spectral leakage
        window = torch.hann_window(n_fft, device=signal.device, dtype=signal.dtype)

        # Reshape for windowed FFT
        # Take overlapping segments
        hop = n_fft // 2
        n_segments = max(1, (signal.shape[-1] - n_fft) // hop + 1)

        segments = []
        for i in range(n_segments):
            start = i * hop
            end = start + n_fft
            if end <= signal.shape[-1]:
                seg = signal[..., start:end] * window
                segments.append(seg)

        if not segments:
            # Signal too short, use zero-padded version
            seg = F.pad(signal, (0, n_fft - T)) * window[:T if T < n_fft else n_fft]
            segments = [seg]

        # Stack and compute FFT
        segments = torch.stack(segments, dim=-2)  # [B, C, n_segments, n_fft]
        fft = torch.fft.rfft(segments, dim=-1)
        psd = (fft.abs() ** 2).mean(dim=-2)  # Average over segments

        return psd

    def _get_band_mask(
        self,
        n_freqs: int,
        band_name: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Get frequency mask for a band.

        Args:
            n_freqs: Number of frequency bins
            band_name: Name of the band
            device: Target device

        Returns:
            Boolean mask [n_freqs]
        """
        f_low, f_high = self.band_ranges[band_name]
        freqs = torch.fft.rfftfreq(
            (n_freqs - 1) * 2,
            d=1/self.sample_rate,
        ).to(device)
        return (freqs >= f_low) & (freqs < f_high)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-scale spectral loss.

        Args:
            pred: Predicted signal [B, C, T]
            target: Target signal [B, C, T]

        Returns:
            (loss, details): Loss value and per-band breakdown
        """
        device = pred.device
        total_loss = torch.tensor(0.0, device=device)
        details = {}

        for scale in self.scales:
            pred_psd = self._compute_psd(pred, scale)
            target_psd = self._compute_psd(target, scale)

            if self.use_log_psd:
                pred_psd = torch.log(pred_psd + self.eps)
                target_psd = torch.log(target_psd + self.eps)

            n_freqs = pred_psd.shape[-1]

            # Per-band weighted loss
            scale_loss = torch.tensor(0.0, device=device)
            total_weight = 0.0

            for band_name, weight in self.band_weights.items():
                if band_name not in self.band_ranges:
                    continue

                mask = self._get_band_mask(n_freqs, band_name, device)
                if mask.sum() == 0:
                    continue

                band_pred = pred_psd[..., mask]
                band_target = target_psd[..., mask]

                band_loss = F.l1_loss(band_pred, band_target)
                scale_loss = scale_loss + weight * band_loss
                total_weight += weight

                # Record for analysis
                key = f"scale_{scale}_{band_name}"
                details[key] = band_loss.item()

            if total_weight > 0:
                scale_loss = scale_loss / total_weight

            total_loss = total_loss + scale_loss
            details[f"scale_{scale}_total"] = scale_loss.item()

        # Average over scales
        total_loss = total_loss / len(self.scales)
        details["total"] = total_loss.item()

        return total_loss, details


class SpectralMatchingLoss(nn.Module):
    """Combined loss for training with spectral matching.

    This wrapper combines standard losses (L1, wavelet) with spectral
    matching loss for end-to-end training without post-hoc correction.

    Args:
        l1_weight: Weight for L1 loss
        wavelet_weight: Weight for wavelet loss
        spectral_weight: Weight for spectral loss
        spectral_config: Configuration for MultiScaleSpectralLoss
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        wavelet_weight: float = 1.0,
        spectral_weight: float = 1.0,
        spectral_config: Optional[Dict] = None,
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.wavelet_weight = wavelet_weight
        self.spectral_weight = spectral_weight

        # Create spectral loss
        config = spectral_config or {}
        self.spectral_loss = MultiScaleSpectralLoss(
            scales=config.get("spectral_scales", [512, 1024, 2048]),
            use_log_psd=config.get("use_log_psd", True),
            band_weights={
                "delta": config.get("delta_weight", 1.0),
                "theta": config.get("theta_weight", 1.0),
                "alpha": config.get("alpha_weight", 1.0),
                "beta": config.get("beta_weight", 2.0),
                "gamma": config.get("gamma_weight", 3.0),
            },
            sample_rate=config.get("sample_rate", 1000.0),
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        wavelet_loss_fn: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            pred: Predicted signal [B, C, T]
            target: Target signal [B, C, T]
            wavelet_loss_fn: Optional wavelet loss function

        Returns:
            (total_loss, details)
        """
        details = {}

        # L1 loss
        l1 = F.l1_loss(pred, target)
        details["l1"] = l1.item()

        total_loss = self.l1_weight * l1

        # Wavelet loss (if provided)
        if wavelet_loss_fn is not None and self.wavelet_weight > 0:
            wavelet = wavelet_loss_fn(pred, target)
            details["wavelet"] = wavelet.item()
            total_loss = total_loss + self.wavelet_weight * wavelet

        # Spectral loss
        if self.spectral_weight > 0:
            spectral, spectral_details = self.spectral_loss(pred, target)
            details["spectral"] = spectral.item()
            details.update({f"spectral_{k}": v for k, v in spectral_details.items()})
            total_loss = total_loss + self.spectral_weight * spectral

        details["total"] = total_loss.item()

        return total_loss, details


class IdentitySpectralModule(nn.Module):
    """Identity module for 'spectral loss only' experiments.

    When using spectral loss during training, no post-hoc correction
    is needed. This module passes signals through unchanged.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return x


def create_spectral_loss_module(
    config: Dict,
    sample_rate: float = 1000.0,
) -> Tuple[nn.Module, MultiScaleSpectralLoss]:
    """Factory function for spectral loss experiment.

    Returns:
        (spectral_module, loss_fn): Identity module and spectral loss
    """
    loss_fn = MultiScaleSpectralLoss(
        scales=config.get("spectral_scales", [512, 1024, 2048]),
        use_log_psd=config.get("use_log_psd", True),
        band_weights={
            "delta": config.get("delta_weight", 1.0),
            "theta": config.get("theta_weight", 1.0),
            "alpha": config.get("alpha_weight", 1.0),
            "beta": config.get("beta_weight", 2.0),
            "gamma": config.get("gamma_weight", 3.0),
        },
        sample_rate=sample_rate,
    )

    return IdentitySpectralModule(), loss_fn
