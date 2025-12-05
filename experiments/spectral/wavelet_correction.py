"""
Wavelet-Based Local Spectral Correction
=======================================

Uses Continuous Wavelet Transform (CWT) instead of FFT to enable
time-localized spectral correction. This prevents global ringing
artifacts that occur with FFT-based methods.

Key insight: Neural signals have non-stationary frequency content.
FFT assumes stationarity, but CWT can correct spectral content
in specific time windows without affecting others.

Search parameters:
- wavelet_omega0: Central frequency of Morlet wavelet
- correction_kernel_size: Local correction kernel size
- n_correction_layers: Number of time-frequency correction layers
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Neural frequency bands for wavelet analysis
WAVELET_SCALES = {
    "delta": 100,    # ~1-4 Hz at 1000Hz sample rate
    "theta": 50,     # ~4-8 Hz
    "alpha": 30,     # ~8-12 Hz
    "beta": 15,      # ~12-30 Hz
    "gamma": 6,      # ~30-100 Hz
}


class MorletWavelet(nn.Module):
    """Morlet wavelet for CWT computation.

    The Morlet wavelet is a complex sinusoid modulated by a Gaussian:
        psi(t) = exp(-t^2 / 2) * exp(i * omega0 * t)

    Args:
        omega0: Central frequency parameter (default: 5.0)
    """

    def __init__(self, omega0: float = 5.0):
        super().__init__()
        self.omega0 = omega0

    def forward(
        self,
        t: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Generate scaled Morlet wavelet.

        Args:
            t: Time points tensor
            scale: Wavelet scale (larger = lower frequency)

        Returns:
            Complex wavelet values
        """
        t_scaled = t / scale
        gaussian = torch.exp(-t_scaled ** 2 / 2)
        oscillation = torch.exp(1j * self.omega0 * t_scaled)
        # Normalization
        norm = 1 / (math.pi ** 0.25 * math.sqrt(scale))
        return norm * gaussian * oscillation


class WaveletLocalSpectralCorrection(nn.Module):
    """CWT-based time-localized spectral correction.

    This module:
    1. Computes CWT of input signal
    2. Applies learnable correction in time-frequency domain
    3. Reconstructs signal via inverse CWT

    The key advantage over FFT is that corrections are local in time,
    preventing global ringing artifacts.

    Args:
        n_channels: Number of input channels
        n_scales: Number of wavelet scales (frequency bands)
        omega0: Morlet wavelet central frequency
        correction_kernel_size: Kernel size for local correction
        n_correction_layers: Number of correction conv layers
        sample_rate: Sampling rate in Hz
        scales: Custom scale values (if None, auto-generated)
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_scales: int = 16,
        omega0: float = 5.0,
        correction_kernel_size: int = 3,
        n_correction_layers: int = 2,
        sample_rate: float = 1000.0,
        scales: Optional[List[float]] = None,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_scales = n_scales
        self.omega0 = omega0
        self.sample_rate = sample_rate

        # Generate scales for neural frequency bands
        if scales is None:
            # Log-spaced scales from ~100Hz to ~1Hz
            self.scales = torch.logspace(
                math.log10(2),  # ~100Hz
                math.log10(200),  # ~1Hz
                n_scales,
            )
        else:
            self.scales = torch.tensor(scales)

        self.register_buffer("scale_values", self.scales)

        # Morlet wavelet generator
        self.wavelet = MorletWavelet(omega0)

        # Time-frequency correction network
        # Uses depthwise separable convolutions for efficiency
        correction_layers = []
        for i in range(n_correction_layers):
            correction_layers.extend([
                # Depthwise conv over time-frequency
                nn.Conv2d(
                    n_channels, n_channels,
                    kernel_size=correction_kernel_size,
                    padding=correction_kernel_size // 2,
                    groups=n_channels,  # Depthwise
                ),
                nn.GELU(),
            ])

        self.correction_net = nn.Sequential(*correction_layers)

        # Learnable blending factor (how much correction to apply)
        self.blend_logit = nn.Parameter(torch.tensor(0.0))

        # Per-scale correction factors
        self.log_scale_corrections = nn.Parameter(torch.zeros(n_scales))

    @property
    def blend(self) -> torch.Tensor:
        """Blending factor in [0, 1]."""
        return torch.sigmoid(self.blend_logit)

    def _compute_cwt(
        self,
        x: torch.Tensor,
        max_wavelet_len: int = 512,
    ) -> torch.Tensor:
        """Compute Continuous Wavelet Transform.

        Args:
            x: Input signal [B, C, T]
            max_wavelet_len: Maximum wavelet length

        Returns:
            CWT coefficients [B, C, n_scales, T]
        """
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        # Pre-compute wavelet for each scale
        cwt_coeffs = []

        for scale in self.scale_values:
            scale = scale.item()

            # Wavelet length based on scale
            wavelet_len = min(int(6 * scale), max_wavelet_len)
            if wavelet_len % 2 == 0:
                wavelet_len += 1

            # Time points centered at 0
            t = torch.linspace(
                -wavelet_len // 2,
                wavelet_len // 2,
                wavelet_len,
                device=device,
            )

            # Generate wavelet
            wavelet = self.wavelet(t, scale).to(dtype)

            # Convolve signal with wavelet (complex convolution)
            # Separate real and imaginary parts
            wavelet_real = wavelet.real.view(1, 1, -1)
            wavelet_imag = wavelet.imag.view(1, 1, -1)

            # Pad signal
            pad = wavelet_len // 2
            x_padded = F.pad(x, (pad, pad), mode='reflect')

            # Convolve
            conv_real = F.conv1d(
                x_padded,
                wavelet_real.expand(C, 1, -1),
                groups=C,
            )
            conv_imag = F.conv1d(
                x_padded,
                wavelet_imag.expand(C, 1, -1),
                groups=C,
            )

            # Trim to original length
            conv_real = conv_real[..., :T]
            conv_imag = conv_imag[..., :T]

            # Store magnitude (we'll preserve phase separately)
            cwt_mag = torch.sqrt(conv_real ** 2 + conv_imag ** 2)
            cwt_coeffs.append(cwt_mag)

        return torch.stack(cwt_coeffs, dim=2)  # [B, C, n_scales, T]

    def _inverse_cwt(
        self,
        cwt_coeffs: torch.Tensor,
        original: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate inverse CWT via weighted sum.

        Note: True inverse CWT is complex. We use a simplified
        reconstruction that preserves the original signal's phase
        while applying magnitude corrections.

        Args:
            cwt_coeffs: Corrected CWT magnitudes [B, C, n_scales, T]
            original: Original signal [B, C, T]

        Returns:
            Reconstructed signal [B, C, T]
        """
        B, C, n_scales, T = cwt_coeffs.shape

        # Compute original CWT for phase reference
        original_cwt = self._compute_cwt(original)

        # Ratio of corrected to original (avoid division by zero)
        ratio = cwt_coeffs / (original_cwt + 1e-8)

        # Weighted reconstruction across scales
        # Each scale contributes based on its correction ratio
        weights = F.softmax(torch.ones(n_scales, device=original.device), dim=0)
        weights = weights.view(1, 1, n_scales, 1)

        # Mean correction factor across scales
        correction_factor = (ratio * weights).sum(dim=2)  # [B, C, T]

        # Apply to original signal
        reconstructed = original * correction_factor

        return reconstructed

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply wavelet-based local spectral correction.

        Args:
            x: Input signal [B, C, T]
            target: Target signal (optional)
            odor_ids: Not used

        Returns:
            Corrected signal [B, C, T]
        """
        B, C, T = x.shape

        # Compute CWT
        cwt_coeffs = self._compute_cwt(x)  # [B, C, n_scales, T]

        # Apply per-scale correction factors
        scale_corr = torch.exp(self.log_scale_corrections)
        scale_corr = scale_corr.view(1, 1, -1, 1)
        cwt_scaled = cwt_coeffs * scale_corr

        # Apply learned local correction
        cwt_corrected = self.correction_net(cwt_scaled)

        # Blend with original
        blend = self.blend
        cwt_blended = blend * cwt_corrected + (1 - blend) * cwt_coeffs

        # Reconstruct
        corrected = self._inverse_cwt(cwt_blended, x)

        return corrected


def create_wavelet_correction_module(
    config: Dict,
    n_channels: int = 32,
    sample_rate: float = 1000.0,
) -> nn.Module:
    """Factory function to create wavelet correction module from config.

    Args:
        config: Configuration dictionary
        n_channels: Number of channels
        sample_rate: Sampling rate

    Returns:
        Configured WaveletLocalSpectralCorrection module
    """
    return WaveletLocalSpectralCorrection(
        n_channels=n_channels,
        n_scales=config.get("n_scales", 16),
        omega0=config.get("wavelet_omega0", 5.0),
        correction_kernel_size=config.get("correction_kernel_size", 3),
        n_correction_layers=config.get("n_correction_layers", 2),
        sample_rate=sample_rate,
    )
