"""
Phase-Preserving Soft Spectral Correction
==========================================

Key insight: Modify MAGNITUDE only, preserve PHASE exactly.

The phase of a signal carries critical temporal information. Traditional
spectral correction methods often distort phase when matching PSDs, which
degrades R² and correlation. This module applies soft power-law correction
to magnitude while keeping phase unchanged.

Correction formula:
    corrected_magnitude = magnitude * (target_psd / pred_psd)^(softness * beta)

Where:
    - softness ∈ [0, 1]: 0 = no correction, 1 = full correction
    - beta < 1: Soft correction to avoid over-correction
    - Phase is strictly preserved

Search parameters:
    - target_bands: Which frequency bands to correct (beta, gamma, or both)
    - max_correction_db: Maximum correction magnitude in dB
    - softness: How aggressively to apply correction
    - per_channel: Whether to learn per-channel corrections
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# Neural frequency bands
NEURAL_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 100),
}


class PhasePreservingSoftCorrection(nn.Module):
    """Soft spectral correction that strictly preserves phase.

    This module applies power-law magnitude correction in targeted frequency
    bands while keeping the phase spectrum unchanged. The correction strength
    is controlled by learnable per-band parameters and a softness factor.

    Args:
        n_channels: Number of input channels
        target_bands: List of band names to correct (e.g., ["beta", "gamma"])
        max_correction_db: Maximum correction in dB (clamps extreme corrections)
        softness: Base softness factor [0, 1] (can be learned)
        per_channel: Learn separate corrections per channel
        sample_rate: Sampling rate in Hz
        transition_width_hz: Soft transition width at band edges
        learn_softness: Whether softness is learnable
    """

    def __init__(
        self,
        n_channels: int = 32,
        target_bands: Optional[List[str]] = None,
        max_correction_db: float = 6.0,
        softness: float = 0.5,
        per_channel: bool = False,
        sample_rate: float = 1000.0,
        transition_width_hz: float = 2.0,
        learn_softness: bool = True,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.target_bands = target_bands or ["beta", "gamma"]
        self.max_correction_db = max_correction_db
        self.sample_rate = sample_rate
        self.transition_width_hz = transition_width_hz
        self.per_channel = per_channel

        # Convert max correction to linear scale
        self.max_correction_linear = 10 ** (max_correction_db / 20)

        # Get band frequency ranges
        self.band_ranges = [NEURAL_BANDS[b] for b in self.target_bands]

        # Learnable parameters
        n_bands = len(self.target_bands)

        if per_channel:
            # [n_channels, n_bands] - separate correction per channel and band
            self.log_corrections = nn.Parameter(torch.zeros(n_channels, n_bands))
        else:
            # [n_bands] - shared correction across channels
            self.log_corrections = nn.Parameter(torch.zeros(n_bands))

        # Learnable softness (sigmoid-transformed to stay in [0, 1])
        if learn_softness:
            # Initialize to given softness via inverse sigmoid
            init_logit = math.log(softness / (1 - softness + 1e-8))
            self.softness_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.register_buffer("softness_logit", torch.tensor(
                math.log(softness / (1 - softness + 1e-8))
            ))

    @property
    def softness(self) -> torch.Tensor:
        """Current softness value in [0, 1]."""
        return torch.sigmoid(self.softness_logit)

    def _build_band_masks(
        self,
        n_freqs: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build soft frequency band masks.

        Args:
            n_freqs: Number of frequency bins (rfft output length)
            device: Target device

        Returns:
            Band masks [n_bands, n_freqs] with soft transitions
        """
        # Frequency bins
        freqs = torch.fft.rfftfreq(
            (n_freqs - 1) * 2,  # Original signal length
            d=1/self.sample_rate,
        ).to(device)

        masks = []
        for f_low, f_high in self.band_ranges:
            # Soft transitions using sigmoid
            # Low edge: sigmoid((f - f_low) / width)
            # High edge: sigmoid((f_high - f) / width)
            low_edge = torch.sigmoid(
                (freqs - f_low) / self.transition_width_hz
            )
            high_edge = torch.sigmoid(
                (f_high - freqs) / self.transition_width_hz
            )
            mask = low_edge * high_edge
            masks.append(mask)

        return torch.stack(masks, dim=0)  # [n_bands, n_freqs]

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply phase-preserving soft spectral correction.

        Args:
            x: Input signal [B, C, T]
            target: Target signal for PSD matching (optional, used for adaptive correction)
            odor_ids: Odor condition IDs (not used in this module)

        Returns:
            Corrected signal [B, C, T]
        """
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        # Compute FFT
        x_fft = torch.fft.rfft(x, dim=-1)  # [B, C, T//2+1]

        # Separate magnitude and phase
        magnitude = x_fft.abs()
        phase = x_fft.angle()  # PRESERVE THIS!

        # Build band masks
        n_freqs = x_fft.shape[-1]
        band_masks = self._build_band_masks(n_freqs, device)  # [n_bands, n_freqs]

        # Get correction factors
        corrections = torch.exp(self.log_corrections)  # [C, n_bands] or [n_bands]

        # Clamp corrections to max_correction_db
        corrections = torch.clamp(
            corrections,
            1 / self.max_correction_linear,
            self.max_correction_linear,
        )

        # Apply soft correction
        softness = self.softness.to(dtype)
        corrected_magnitude = magnitude.clone()

        for band_idx in range(len(self.target_bands)):
            mask = band_masks[band_idx].to(dtype)  # [n_freqs]

            if self.per_channel:
                corr = corrections[:, band_idx]  # [C]
                corr = corr.view(1, C, 1)  # [1, C, 1]
            else:
                corr = corrections[band_idx]  # scalar
                corr = corr.view(1, 1, 1)  # [1, 1, 1]

            # Soft blending: 1 + (correction - 1) * softness
            effective_corr = 1 + (corr - 1) * softness

            # Apply only in band (with soft transitions)
            # correction_map = 1 + (effective_corr - 1) * mask
            mask = mask.view(1, 1, -1)  # [1, 1, n_freqs]
            correction_map = 1 + (effective_corr - 1) * mask

            corrected_magnitude = corrected_magnitude * correction_map

        # Reconstruct with ORIGINAL phase
        corrected_fft = corrected_magnitude * torch.exp(1j * phase)

        # Inverse FFT
        corrected = torch.fft.irfft(corrected_fft, n=T, dim=-1)

        return corrected.to(dtype)

    def get_correction_summary(self) -> Dict[str, float]:
        """Get current correction parameters for logging."""
        corrections = torch.exp(self.log_corrections).detach().cpu()
        corrections_db = 20 * torch.log10(corrections + 1e-8)

        summary = {
            "softness": self.softness.item(),
        }

        if self.per_channel:
            for band_idx, band_name in enumerate(self.target_bands):
                summary[f"{band_name}_mean_db"] = corrections_db[:, band_idx].mean().item()
                summary[f"{band_name}_std_db"] = corrections_db[:, band_idx].std().item()
        else:
            for band_idx, band_name in enumerate(self.target_bands):
                summary[f"{band_name}_db"] = corrections_db[band_idx].item()

        return summary


class AdaptivePhasePreservingCorrection(nn.Module):
    """Phase-preserving correction with target-adaptive strength.

    Instead of fixed correction factors, this module adapts the correction
    strength based on the PSD mismatch between prediction and target.

    The correction is: magnitude * (target_psd / pred_psd)^beta
    where beta = softness * base_strength

    Args:
        n_channels: Number of input channels
        target_bands: Frequency bands to correct
        softness: Base softness factor
        max_correction_db: Maximum correction in dB
        sample_rate: Sampling rate in Hz
    """

    def __init__(
        self,
        n_channels: int = 32,
        target_bands: Optional[List[str]] = None,
        softness: float = 0.3,
        max_correction_db: float = 9.0,
        sample_rate: float = 1000.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.target_bands = target_bands or ["beta", "gamma"]
        self.max_correction_db = max_correction_db
        self.sample_rate = sample_rate
        self.eps = eps

        # Learnable per-band softness (beta exponent)
        n_bands = len(self.target_bands)
        init_logit = math.log(softness / (1 - softness + 1e-8))
        self.softness_logits = nn.Parameter(torch.full((n_bands,), init_logit))

        # Get band ranges
        self.band_ranges = [NEURAL_BANDS[b] for b in self.target_bands]

    @property
    def softness_values(self) -> torch.Tensor:
        """Current softness values per band."""
        return torch.sigmoid(self.softness_logits)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply adaptive phase-preserving correction.

        Args:
            x: Predicted signal [B, C, T]
            target: Target signal [B, C, T]
            odor_ids: Not used

        Returns:
            Corrected signal [B, C, T]
        """
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        # Compute FFTs
        x_fft = torch.fft.rfft(x, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)

        # Separate magnitude and phase
        x_mag = x_fft.abs()
        x_phase = x_fft.angle()  # PRESERVE THIS
        target_mag = target_fft.abs()

        # Compute PSDs (magnitude squared, smoothed)
        x_psd = x_mag ** 2 + self.eps
        target_psd = target_mag ** 2 + self.eps

        # Build frequency bins
        n_freqs = x_fft.shape[-1]
        freqs = torch.fft.rfftfreq((n_freqs - 1) * 2, d=1/self.sample_rate).to(device)

        # Apply correction per band
        corrected_mag = x_mag.clone()
        softness = self.softness_values.to(dtype)

        for band_idx, (f_low, f_high) in enumerate(self.band_ranges):
            # Band mask
            mask = ((freqs >= f_low) & (freqs < f_high)).float()
            mask = mask.view(1, 1, -1)  # [1, 1, n_freqs]

            # PSD ratio (clamped)
            max_ratio = 10 ** (self.max_correction_db / 10)
            psd_ratio = torch.clamp(target_psd / x_psd, 1/max_ratio, max_ratio)

            # Correction: (target_psd / pred_psd)^(softness/2)
            # Note: /2 because we're correcting magnitude, not power
            beta = softness[band_idx] / 2
            correction = psd_ratio ** beta

            # Apply only in band
            corrected_mag = corrected_mag * (1 + (correction - 1) * mask)

        # Reconstruct with original phase
        corrected_fft = corrected_mag * torch.exp(1j * x_phase)
        corrected = torch.fft.irfft(corrected_fft, n=T, dim=-1)

        return corrected.to(dtype)


def create_phase_preserving_module(
    config: Dict,
    n_channels: int = 32,
    sample_rate: float = 1000.0,
) -> nn.Module:
    """Factory function to create phase-preserving module from config.

    Args:
        config: Configuration dictionary with search parameters
        n_channels: Number of channels
        sample_rate: Sampling rate

    Returns:
        Configured PhasePreservingSoftCorrection module
    """
    # Parse target bands
    target_bands_raw = config.get("target_bands", ["beta", "gamma"])
    if isinstance(target_bands_raw, str):
        target_bands = [target_bands_raw]
    else:
        target_bands = list(target_bands_raw)

    return PhasePreservingSoftCorrection(
        n_channels=n_channels,
        target_bands=target_bands,
        max_correction_db=config.get("max_correction_db", 6.0),
        softness=config.get("softness", 0.5),
        per_channel=config.get("per_channel", False),
        sample_rate=sample_rate,
        transition_width_hz=config.get("transition_width_hz", 2.0),
        learn_softness=config.get("learn_softness", True),
    )
