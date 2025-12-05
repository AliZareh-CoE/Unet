"""
Multi-Scale FFT Loss for Neural Signal Translation
==================================================

Frequency-domain losses at multiple scales for better
spectral fidelity in neural signal translation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """Basic spectral loss in frequency domain.

    Computes loss on magnitude spectrum.

    Args:
        n_fft: FFT size
        log_scale: If True, compute loss in log scale (dB)
        weight: Loss weight
    """

    def __init__(
        self,
        n_fft: int = 1024,
        log_scale: bool = True,
        weight: float = 1.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.log_scale = log_scale
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spectral loss.

        Args:
            pred: Predicted signal [B, C, T]
            target: Target signal [B, C, T]

        Returns:
            Spectral loss
        """
        # Compute FFT
        pred_fft = torch.fft.rfft(pred, n=self.n_fft, dim=-1)
        target_fft = torch.fft.rfft(target, n=self.n_fft, dim=-1)

        # Magnitude
        pred_mag = pred_fft.abs()
        target_mag = target_fft.abs()

        if self.log_scale:
            pred_mag = torch.log(pred_mag + 1e-8)
            target_mag = torch.log(target_mag + 1e-8)

        loss = F.l1_loss(pred_mag, target_mag)

        return self.weight * loss


class MultiScaleSpectralLoss(nn.Module):
    """Multi-scale spectral loss.

    Computes spectral loss at multiple FFT sizes to capture
    different frequency resolutions.

    Args:
        n_ffts: List of FFT sizes (e.g., [256, 512, 1024, 2048])
        log_scale: If True, use log magnitude
        mag_weight: Weight for magnitude loss
        phase_weight: Weight for phase loss (0 to disable)
        weight: Overall loss weight
    """

    def __init__(
        self,
        n_ffts: List[int] = [256, 512, 1024, 2048],
        log_scale: bool = True,
        mag_weight: float = 1.0,
        phase_weight: float = 0.0,
        weight: float = 1.0,
    ):
        super().__init__()
        self.n_ffts = n_ffts
        self.log_scale = log_scale
        self.mag_weight = mag_weight
        self.phase_weight = phase_weight
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-scale spectral loss.

        Args:
            pred: Predicted signal [B, C, T]
            target: Target signal [B, C, T]

        Returns:
            Multi-scale spectral loss
        """
        total_loss = 0

        for n_fft in self.n_ffts:
            # Compute FFT
            pred_fft = torch.fft.rfft(pred, n=n_fft, dim=-1)
            target_fft = torch.fft.rfft(target, n=n_fft, dim=-1)

            # Magnitude loss
            pred_mag = pred_fft.abs()
            target_mag = target_fft.abs()

            if self.log_scale:
                pred_mag = torch.log(pred_mag + 1e-8)
                target_mag = torch.log(target_mag + 1e-8)

            mag_loss = F.l1_loss(pred_mag, target_mag)
            total_loss = total_loss + self.mag_weight * mag_loss

            # Phase loss (optional)
            if self.phase_weight > 0:
                pred_phase = pred_fft.angle()
                target_phase = target_fft.angle()

                # Circular phase difference
                phase_diff = torch.remainder(pred_phase - target_phase + torch.pi, 2 * torch.pi) - torch.pi
                phase_loss = phase_diff.abs().mean()
                total_loss = total_loss + self.phase_weight * phase_loss

        # Average over scales
        total_loss = total_loss / len(self.n_ffts)

        return self.weight * total_loss


class STFTLoss(nn.Module):
    """Short-Time Fourier Transform loss.

    Computes loss on spectrogram (time-frequency representation).

    Args:
        n_fft: FFT size
        hop_length: Hop length between frames
        win_length: Window length
        log_scale: Use log magnitude
        weight: Loss weight
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: Optional[int] = None,
        log_scale: bool = True,
        weight: float = 1.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.log_scale = log_scale
        self.weight = weight

        # Register Hann window
        self.register_buffer('window', torch.hann_window(self.win_length))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute STFT loss.

        Args:
            pred: Predicted signal [B, C, T]
            target: Target signal [B, C, T]

        Returns:
            STFT loss
        """
        B, C, T = pred.shape

        # Flatten batch and channel for STFT
        pred_flat = pred.reshape(B * C, T)
        target_flat = target.reshape(B * C, T)

        # Compute STFT
        pred_stft = torch.stft(
            pred_flat, self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        target_stft = torch.stft(
            target_flat, self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        # Magnitude
        pred_mag = pred_stft.abs()
        target_mag = target_stft.abs()

        if self.log_scale:
            pred_mag = torch.log(pred_mag + 1e-8)
            target_mag = torch.log(target_mag + 1e-8)

        loss = F.l1_loss(pred_mag, target_mag)

        return self.weight * loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss.

    Computes STFT loss at multiple time-frequency resolutions.

    Args:
        resolutions: List of (n_fft, hop_length) tuples
        log_scale: Use log magnitude
        weight: Overall loss weight
    """

    def __init__(
        self,
        resolutions: List[Tuple[int, int]] = [(512, 128), (1024, 256), (2048, 512)],
        log_scale: bool = True,
        weight: float = 1.0,
    ):
        super().__init__()
        self.resolutions = resolutions
        self.log_scale = log_scale
        self.weight = weight

        # Create STFT losses for each resolution
        self.stft_losses = nn.ModuleList([
            STFTLoss(n_fft=n_fft, hop_length=hop, log_scale=log_scale, weight=1.0)
            for n_fft, hop in resolutions
        ])

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-resolution STFT loss."""
        total_loss = 0

        for stft_loss in self.stft_losses:
            total_loss = total_loss + stft_loss(pred, target)

        return self.weight * total_loss / len(self.stft_losses)


class BandSpecificLoss(nn.Module):
    """Spectral loss with band-specific weights.

    Allows emphasizing certain frequency bands (e.g., gamma for neural signals).

    Args:
        n_fft: FFT size
        sample_rate: Sampling rate in Hz
        band_weights: Dict mapping band name to weight
        log_scale: Use log magnitude
        weight: Overall loss weight
    """

    def __init__(
        self,
        n_fft: int = 1024,
        sample_rate: float = 1000.0,
        band_weights: Optional[dict] = None,
        log_scale: bool = True,
        weight: float = 1.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.log_scale = log_scale
        self.weight = weight

        # Default band weights (emphasize gamma for neural signals)
        self.band_weights = band_weights or {
            "delta": 0.5,
            "theta": 1.0,
            "alpha": 1.0,
            "beta": 1.5,
            "gamma": 2.0,
        }

        # Neural frequency bands
        self.bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 30),
            "gamma": (30, 100),
        }

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute band-specific spectral loss."""
        # Compute FFT
        pred_fft = torch.fft.rfft(pred, n=self.n_fft, dim=-1)
        target_fft = torch.fft.rfft(target, n=self.n_fft, dim=-1)

        # Get frequencies
        freqs = torch.fft.rfftfreq(self.n_fft, d=1/self.sample_rate)
        freqs = freqs.to(pred.device)

        # Magnitude
        pred_mag = pred_fft.abs()
        target_mag = target_fft.abs()

        if self.log_scale:
            pred_mag = torch.log(pred_mag + 1e-8)
            target_mag = torch.log(target_mag + 1e-8)

        total_loss = 0

        for band_name, (f_low, f_high) in self.bands.items():
            mask = (freqs >= f_low) & (freqs < f_high)
            if mask.sum() == 0:
                continue

            band_weight = self.band_weights.get(band_name, 1.0)
            band_loss = F.l1_loss(pred_mag[..., mask], target_mag[..., mask])
            total_loss = total_loss + band_weight * band_loss

        return self.weight * total_loss


class CombinedSpectralTimeLoss(nn.Module):
    """Combined time-domain and spectral loss.

    Args:
        time_loss: Time-domain loss type ('l1', 'mse', 'huber')
        time_weight: Weight for time-domain loss
        spectral_weight: Weight for spectral loss
        n_ffts: FFT sizes for multi-scale spectral loss
        weight: Overall loss weight
    """

    def __init__(
        self,
        time_loss: str = "l1",
        time_weight: float = 1.0,
        spectral_weight: float = 1.0,
        n_ffts: List[int] = [512, 1024, 2048],
        weight: float = 1.0,
    ):
        super().__init__()
        self.time_weight = time_weight
        self.spectral_weight = spectral_weight
        self.weight = weight

        if time_loss == "l1":
            self.time_loss_fn = nn.L1Loss()
        elif time_loss == "mse":
            self.time_loss_fn = nn.MSELoss()
        elif time_loss == "huber":
            self.time_loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown time loss: {time_loss}")

        self.spectral_loss_fn = MultiScaleSpectralLoss(n_ffts=n_ffts, weight=1.0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss."""
        time_loss = self.time_loss_fn(pred, target)
        spectral_loss = self.spectral_loss_fn(pred, target)

        return self.weight * (self.time_weight * time_loss + self.spectral_weight * spectral_loss)


def create_spectral_loss(
    variant: str = "multi_scale",
    **kwargs,
) -> nn.Module:
    """Factory function for spectral losses.

    Args:
        variant: "basic", "multi_scale", "stft", "multi_res_stft", "band_specific", "combined"
        **kwargs: Additional arguments

    Returns:
        Spectral loss module
    """
    if variant == "basic":
        return SpectralLoss(**kwargs)
    elif variant == "multi_scale":
        return MultiScaleSpectralLoss(**kwargs)
    elif variant == "stft":
        return STFTLoss(**kwargs)
    elif variant == "multi_res_stft":
        return MultiResolutionSTFTLoss(**kwargs)
    elif variant == "band_specific":
        return BandSpecificLoss(**kwargs)
    elif variant == "combined":
        return CombinedSpectralTimeLoss(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
