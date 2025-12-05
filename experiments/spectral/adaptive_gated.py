"""
Adaptive Gated Spectral Correction
==================================

Learn a gating network that decides PER SAMPLE how much spectral correction
to apply in each frequency band. This allows the model to adaptively correct
only where it helps, avoiding over-correction.

Key features:
- Signal analyzer extracts features from input
- Gate predictor outputs per-band correction gates in [0, 1]
- Correction = gate * learned_correction + (1 - gate) * identity
- Bands where correction hurts R² can be automatically gated out

Search parameters:
- gating_hidden_dim: Size of gating network
- n_bands: Number of frequency bands
- gate_temperature: Controls gate sharpness
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default frequency bands (Hz)
DEFAULT_BAND_EDGES = [1, 4, 8, 12, 20, 30, 50, 80, 100]


class AdaptiveGatedSpectralCorrection(nn.Module):
    """Learned per-sample, per-band spectral correction with gating.

    The gating network analyzes the input signal and decides how much
    correction to apply in each frequency band. This allows automatic
    suppression of corrections that hurt R².

    Args:
        n_channels: Number of input channels
        n_bands: Number of frequency bands
        hidden_dim: Hidden dimension for gating network
        sample_rate: Sampling rate in Hz
        band_edges: Frequency edges defining bands
        gate_temperature: Temperature for gate sharpness (lower = sharper)
        max_correction_db: Maximum correction magnitude
        use_target_guidance: Use target signal to guide gating (training only)
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_bands: int = 8,
        hidden_dim: int = 64,
        sample_rate: float = 1000.0,
        band_edges: Optional[List[float]] = None,
        gate_temperature: float = 1.0,
        max_correction_db: float = 9.0,
        use_target_guidance: bool = False,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.sample_rate = sample_rate
        self.gate_temperature = gate_temperature
        self.max_correction_db = max_correction_db
        self.use_target_guidance = use_target_guidance

        # Define frequency bands
        if band_edges is None:
            # Create uniform bands
            band_edges = self._create_uniform_bands(n_bands, 1.0, 100.0)
        self.band_edges = band_edges
        self.n_bands = len(band_edges) - 1

        # Signal analyzer: extract features for gating decision
        self.signal_analyzer = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, kernel_size=15, stride=5, padding=7),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=3, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Gate predictor: how much correction per band
        gate_input_dim = hidden_dim
        if use_target_guidance:
            gate_input_dim *= 2  # Concat target features

        self.gate_predictor = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_bands),
        )

        # Base correction factors (learnable)
        self.log_corrections = nn.Parameter(torch.zeros(self.n_bands))

        # Per-channel scaling (optional)
        self.channel_scale = nn.Parameter(torch.ones(n_channels))

    def _create_uniform_bands(
        self,
        n_bands: int,
        f_min: float,
        f_max: float,
    ) -> List[float]:
        """Create uniform frequency bands on log scale."""
        log_min = math.log10(f_min)
        log_max = math.log10(f_max)
        log_edges = torch.linspace(log_min, log_max, n_bands + 1)
        return (10 ** log_edges).tolist()

    def _build_band_masks(
        self,
        n_freqs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build frequency band masks with soft transitions."""
        freqs = torch.fft.rfftfreq(
            (n_freqs - 1) * 2,
            d=1/self.sample_rate,
        ).to(device)

        masks = []
        for i in range(len(self.band_edges) - 1):
            f_low = self.band_edges[i]
            f_high = self.band_edges[i + 1]

            # Soft band mask
            transition = (f_high - f_low) * 0.1  # 10% transition
            low_edge = torch.sigmoid((freqs - f_low) / (transition + 0.1))
            high_edge = torch.sigmoid((f_high - freqs) / (transition + 0.1))
            mask = low_edge * high_edge
            masks.append(mask)

        return torch.stack(masks, dim=0).to(dtype)  # [n_bands, n_freqs]

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply adaptive gated spectral correction.

        Args:
            x: Input signal [B, C, T]
            target: Target signal (optional, for training guidance)
            odor_ids: Not used

        Returns:
            Corrected signal [B, C, T]
        """
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        # Analyze input signal
        features = self.signal_analyzer(x)  # [B, hidden_dim]

        # Optionally include target information
        if self.use_target_guidance and target is not None and self.training:
            target_features = self.signal_analyzer(target)
            features = torch.cat([features, target_features], dim=-1)

        # Predict gates
        gate_logits = self.gate_predictor(features)  # [B, n_bands]
        gates = torch.sigmoid(gate_logits / self.gate_temperature)  # [B, n_bands]

        # Get base corrections
        corrections = torch.exp(self.log_corrections)  # [n_bands]
        max_corr = 10 ** (self.max_correction_db / 20)
        corrections = torch.clamp(corrections, 1/max_corr, max_corr)

        # Effective correction = gates * corrections
        effective = gates * corrections.view(1, -1)  # [B, n_bands]

        # Apply in frequency domain
        x_fft = torch.fft.rfft(x, dim=-1)
        magnitude = x_fft.abs()
        phase = x_fft.angle()

        n_freqs = x_fft.shape[-1]
        band_masks = self._build_band_masks(n_freqs, device, dtype)  # [n_bands, n_freqs]

        # Apply per-band corrections
        corrected_magnitude = magnitude.clone()

        for band_idx in range(self.n_bands):
            mask = band_masks[band_idx].view(1, 1, -1)  # [1, 1, n_freqs]
            corr = effective[:, band_idx].view(B, 1, 1)  # [B, 1, 1]

            # Gated correction: original + gate * (correction - 1)
            correction_factor = 1 + (corr - 1) * mask
            corrected_magnitude = corrected_magnitude * correction_factor

        # Apply channel scaling
        channel_scale = self.channel_scale.view(1, C, 1)
        corrected_magnitude = corrected_magnitude * channel_scale

        # Reconstruct with original phase
        corrected_fft = corrected_magnitude * torch.exp(1j * phase)
        corrected = torch.fft.irfft(corrected_fft, n=T, dim=-1)

        return corrected.to(dtype)

    def get_gate_statistics(self, x: torch.Tensor) -> Dict[str, float]:
        """Get gate activation statistics for analysis."""
        with torch.no_grad():
            features = self.signal_analyzer(x)
            gate_logits = self.gate_predictor(features)
            gates = torch.sigmoid(gate_logits / self.gate_temperature)

            stats = {
                "mean_gate": gates.mean().item(),
                "std_gate": gates.std().item(),
            }

            for i in range(self.n_bands):
                f_low = self.band_edges[i]
                f_high = self.band_edges[i + 1]
                stats[f"gate_{f_low:.0f}_{f_high:.0f}Hz"] = gates[:, i].mean().item()

            return stats


def create_adaptive_gated_module(
    config: Dict,
    n_channels: int = 32,
    sample_rate: float = 1000.0,
) -> nn.Module:
    """Factory function to create adaptive gated module from config.

    Args:
        config: Configuration dictionary
        n_channels: Number of channels
        sample_rate: Sampling rate

    Returns:
        Configured AdaptiveGatedSpectralCorrection module
    """
    return AdaptiveGatedSpectralCorrection(
        n_channels=n_channels,
        n_bands=config.get("n_bands", 8),
        hidden_dim=config.get("gating_hidden_dim", 64),
        sample_rate=sample_rate,
        gate_temperature=config.get("gate_temperature", 1.0),
        max_correction_db=config.get("max_correction_db", 9.0),
        use_target_guidance=config.get("use_target_guidance", False),
    )
