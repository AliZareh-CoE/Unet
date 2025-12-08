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
    5. cycle_consistent: Latent with cycle-consistency reconstruction constraint
    6. spectro_temporal: SpectroTemporalEncoder - FFT + multi-scale temporal conv (auto-conditioning)

This tier uses U-Net ONLY - no architecture comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal as scipy_signal
from tqdm import tqdm

# Import wavelet loss and SpectroTemporalEncoder from models.py
try:
    from models import build_wavelet_loss, SpectroTemporalEncoder
    HAS_WAVELET_LOSS = True
    HAS_SPECTRO_TEMPORAL = True
except ImportError:
    HAS_WAVELET_LOSS = False
    HAS_SPECTRO_TEMPORAL = False
    build_wavelet_loss = None
    SpectroTemporalEncoder = None

from experiments.common.cross_validation import create_data_splits
from experiments.common.config_registry import (
    get_registry,
    Tier1_5Result,
    ConditioningResult,
)


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier1_5_conditioning"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Conditioning SOURCES to test (what to condition on)
CONDITIONING_SOURCES = [
    "odor_onehot",        # One-hot odor identity (baseline)
    "cpc",                # Contrastive Predictive Coding embeddings
    "vqvae",              # Vector Quantized VAE codes
    "freq_disentangled",  # Frequency-band-specific latents
    "cycle_consistent",   # Cycle-consistent latent
    "spectro_temporal",   # SpectroTemporalEncoder (auto-conditioning from signal dynamics)
]

# Fast subset for dry-run
FAST_SOURCES = ["odor_onehot", "spectro_temporal"]

# Frequency bands for disentanglement
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 100),
}


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
# Conditioning Source Encoders
# =============================================================================

class OdorOneHotEncoder(nn.Module):
    """Simple one-hot encoding of odor identity (baseline)."""

    def __init__(self, n_odors: int, embed_dim: int = 64):
        super().__init__()
        self.n_odors = n_odors
        self.embedding = nn.Embedding(n_odors, embed_dim)

    def forward(self, odor_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            odor_ids: [B] tensor of odor indices

        Returns:
            [B, embed_dim] conditioning embeddings
        """
        return self.embedding(odor_ids)

    def encode_from_signal(self, x: torch.Tensor) -> torch.Tensor:
        """Not used for one-hot - requires explicit odor_ids."""
        raise NotImplementedError("OdorOneHot requires explicit odor_ids")


class CPCEncoder(nn.Module):
    """Contrastive Predictive Coding encoder.

    Learns predictive representations by maximizing mutual information
    between context and future timesteps.
    """

    def __init__(self, in_channels: int, embed_dim: int = 64, n_steps: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.n_steps = n_steps

        # Encoder: signal -> latent
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Context aggregator (GRU)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

        # Predictive heads for each future step
        self.predictors = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_steps)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, T] input signal

        Returns:
            context: [B, embed_dim] context embedding
            z_seq: [B, T', embed_dim] latent sequence for CPC loss
        """
        # Encode
        z = self.encoder(x)  # [B, embed_dim, T']
        z = z.transpose(1, 2)  # [B, T', embed_dim]

        # Context aggregation
        context, _ = self.gru(z)  # [B, T', embed_dim]

        # Use mean context as conditioning
        context_mean = context.mean(dim=1)  # [B, embed_dim]

        return context_mean, z

    def cpc_loss(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute CPC InfoNCE loss."""
        B, T, D = z.shape
        loss = 0.0

        for k, predictor in enumerate(self.predictors, 1):
            if T - k < 1:
                continue

            # Predict k steps ahead
            c_t = context[:, :-k]  # [B, T-k, D]
            z_tk = z[:, k:]        # [B, T-k, D] (positive samples)

            pred = predictor(c_t)  # [B, T-k, D]

            # InfoNCE: positive vs all negatives in batch
            pos = (pred * z_tk).sum(dim=-1)  # [B, T-k]

            # Negatives: all other z in batch
            neg = torch.bmm(pred, z.transpose(1, 2))  # [B, T-k, T]

            logits = torch.cat([pos.unsqueeze(-1), neg], dim=-1)
            labels = torch.zeros(B, T - k, dtype=torch.long, device=z.device)

            loss += F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss / len(self.predictors)


class VQVAEEncoder(nn.Module):
    """Vector Quantized VAE encoder.

    Learns discrete codebook representations of the signal.
    """

    def __init__(self, in_channels: int, embed_dim: int = 64, n_codes: int = 512):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.n_codes = n_codes

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, embed_dim, kernel_size=3, stride=2, padding=1),
        )

        # Codebook
        self.codebook = nn.Embedding(n_codes, embed_dim)
        self.codebook.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)

        # Decoder (for reconstruction loss)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize latents to nearest codebook entries (optimized with cdist)."""
        B, D, T = z.shape
        z_flat = z.transpose(1, 2).reshape(-1, D)  # [B*T, D]

        # Use cdist for efficient distance computation
        distances = torch.cdist(z_flat, self.codebook.weight, p=2).pow(2)  # [B*T, n_codes]

        # Nearest codes
        indices = distances.argmin(dim=1)
        z_q = self.codebook(indices)  # [B*T, D]

        # Reshape back
        z_q = z_q.reshape(B, T, D).transpose(1, 2)  # [B, D, T]
        indices = indices.reshape(B, T)

        return z_q, indices, distances

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, C, T] input signal

        Returns:
            conditioning: [B, embed_dim] mean quantized embedding
            losses: dict with vq_loss, commitment_loss, recon_loss
        """
        # Encode
        z_e = self.encoder(x)  # [B, D, T']

        # Quantize
        z_q, indices, _ = self.quantize(z_e)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Decode for reconstruction loss
        x_recon = self.decoder(z_q_st)

        # Losses
        vq_loss = F.mse_loss(z_q.detach(), z_e)  # Codebook learning
        commitment_loss = F.mse_loss(z_e.detach(), z_q)  # Encoder commitment

        # Pad/crop for reconstruction loss
        T_orig = x.shape[-1]
        T_recon = x_recon.shape[-1]
        if T_recon < T_orig:
            x_crop = x[..., :T_recon]
            recon_loss = F.mse_loss(x_recon, x_crop)
        else:
            x_recon_crop = x_recon[..., :T_orig]
            recon_loss = F.mse_loss(x_recon_crop, x)

        # Mean pooled embedding for conditioning
        conditioning = z_q_st.mean(dim=-1)  # [B, D]

        return conditioning, {
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
            "recon_loss": recon_loss,
        }


class FreqDisentangledEncoder(nn.Module):
    """Frequency-disentangled encoder.

    Learns separate latent representations for each frequency band.
    """

    def __init__(self, in_channels: int, embed_dim: int = 64, fs: float = 1000.0):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.fs = fs

        # Per-band encoders
        self.band_encoders = nn.ModuleDict()
        band_dim = embed_dim // len(FREQ_BANDS)

        for band_name in FREQ_BANDS:
            self.band_encoders[band_name] = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, band_dim),
            )

        # Final projection
        self.proj = nn.Linear(band_dim * len(FREQ_BANDS), embed_dim)

    def bandpass_filter(self, x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """Apply bandpass filter using FFT."""
        # FFT
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.fs).to(x.device)

        # Create bandpass mask
        mask = ((freqs >= low) & (freqs <= high)).float()
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, F]

        # Apply mask
        X_filtered = X * mask

        # Inverse FFT
        return torch.fft.irfft(X_filtered, n=x.shape[-1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input signal

        Returns:
            [B, embed_dim] frequency-disentangled embedding
        """
        band_embeddings = []

        for band_name, (low, high) in FREQ_BANDS.items():
            # Filter signal to band
            x_band = self.bandpass_filter(x, low, high)

            # Encode band
            z_band = self.band_encoders[band_name](x_band)
            band_embeddings.append(z_band)

        # Concatenate all band embeddings
        z_concat = torch.cat(band_embeddings, dim=-1)

        # Project to final embedding
        return self.proj(z_concat)


class CycleConsistentEncoder(nn.Module):
    """Cycle-consistent latent encoder.

    Learns latent that can reconstruct both input and output,
    enforcing cycle consistency.
    """

    def __init__(self, in_channels: int, out_channels: int, embed_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        # Encoder for input (OB)
        self.encoder_x = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Encoder for target (PCx) - for cycle consistency
        self.encoder_y = nn.Sequential(
            nn.Conv1d(out_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Decoder back to input (for cycle consistency)
        self.decoder_x = nn.Sequential(
            nn.Linear(embed_dim, 128 * 8),
            nn.Unflatten(1, (128, 8)),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose1d(32, in_channels, kernel_size=4, stride=4),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, C_in, T] input signal
            y: [B, C_out, T] target signal (for training)

        Returns:
            conditioning: [B, embed_dim] cycle-consistent embedding
            losses: dict with cycle_loss
        """
        # Encode input
        z_x = self.encoder_x(x)  # [B, embed_dim]

        losses = {}

        if y is not None:
            # Encode target
            z_y = self.encoder_y(y)  # [B, embed_dim]

            # Cycle consistency: z_x and z_y should be similar
            cycle_loss = F.mse_loss(z_x, z_y)

            # Reconstruction cycle: x -> z_x -> x_recon
            x_recon = self.decoder_x(z_x)
            T_orig = x.shape[-1]
            T_recon = x_recon.shape[-1]

            if T_recon >= T_orig:
                recon_loss = F.mse_loss(x_recon[..., :T_orig], x)
            else:
                recon_loss = F.mse_loss(x_recon, x[..., :T_recon])

            losses["cycle_loss"] = cycle_loss
            losses["recon_loss"] = recon_loss

        return z_x, losses


# =============================================================================
# Conditioned Translation Model - Uses CondUNet1D as backbone
# =============================================================================

class ConditionedTranslator(nn.Module):
    """U-Net based translator with various conditioning sources.

    Uses CondUNet1D from models.py as the backbone, testing different
    ways to derive the conditioning signal:
    - odor_onehot: Direct odor identity (baseline)
    - cpc: Contrastive Predictive Coding from input signal
    - vqvae: Vector Quantized VAE codes
    - freq_disentangled: Frequency-band specific latents
    - cycle_consistent: Cycle-consistent latent
    - spectro_temporal: SpectroTemporalEncoder (auto-conditioning from signal dynamics)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_source: str,
        embed_dim: int = 64,
        n_odors: int = 10,
    ):
        super().__init__()
        self.cond_source = cond_source
        self.embed_dim = embed_dim
        self.n_odors = n_odors

        # Import CondUNet1D from models.py - our main architecture
        from models import CondUNet1D

        # Create conditioning encoder based on source
        if cond_source == "odor_onehot":
            # Use CondUNet1D's built-in odor conditioning
            self.cond_encoder = None  # Not needed, UNet handles it
            self.unet = CondUNet1D(
                in_channels=in_channels,
                out_channels=out_channels,
                base=64,
                n_odors=n_odors,
                dropout=0.0,
                use_attention=True,
                attention_type="cross_freq_v2",
                norm_type="batch",
                cond_mode="cross_attn_gated",
                use_spectral_shift=False,
                n_downsample=4,
                conv_type="modern",
                use_se=True,
                conv_kernel_size=7,
                dilations=(1, 4, 16, 32),
            )
        else:
            # For other sources, we need to extract conditioning and inject it
            if cond_source == "cpc":
                self.cond_encoder = CPCEncoder(in_channels, embed_dim)
            elif cond_source == "vqvae":
                self.cond_encoder = VQVAEEncoder(in_channels, embed_dim)
            elif cond_source == "freq_disentangled":
                self.cond_encoder = FreqDisentangledEncoder(in_channels, embed_dim)
            elif cond_source == "cycle_consistent":
                self.cond_encoder = CycleConsistentEncoder(in_channels, out_channels, embed_dim)
            elif cond_source == "spectro_temporal":
                # SpectroTemporalEncoder from models.py - uses FFT + multi-scale temporal conv
                if not HAS_SPECTRO_TEMPORAL or SpectroTemporalEncoder is None:
                    raise ImportError("SpectroTemporalEncoder not available in models.py")
                # Use 128 for embed_dim to match UNet's default embedding dimension
                self.cond_encoder = SpectroTemporalEncoder(
                    in_channels=in_channels,
                    emb_dim=128,  # Match UNet's emb_dim
                    n_freq_bands=10,
                    temporal_hidden=64,
                    temporal_dilations=(1, 4, 16),
                    temporal_kernel=7,
                )
            else:
                raise ValueError(f"Unknown conditioning source: {cond_source}")

            # Create UNet with custom conditioning injection
            # We use the same architecture but replace odor embedding with our encoder
            self.unet = CondUNet1D(
                in_channels=in_channels,
                out_channels=out_channels,
                base=64,
                n_odors=1,  # Minimal, we'll override the conditioning
                dropout=0.0,
                use_attention=True,
                attention_type="cross_freq_v2",
                norm_type="batch",
                cond_mode="cross_attn_gated",
                use_spectral_shift=False,
                n_downsample=4,
                conv_type="modern",
                use_se=True,
                conv_kernel_size=7,
                dilations=(1, 4, 16, 32),
            )

            # Override the odor embedding to accept our conditioning
            # CondUNet1D uses emb_dim (default 128) for FiLM, we project our encoding to match
            unet_embed_dim = self.unet.embed.embedding_dim if hasattr(self.unet, 'embed') else 128
            self.cond_projector = nn.Linear(embed_dim, unet_embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        odor_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, C_in, T] input signal
            y: [B, C_out, T] target signal (for training some encoders)
            odor_ids: [B] odor indices (for odor_onehot)

        Returns:
            y_pred: [B, C_out, T] predicted output
            aux_losses: dict with auxiliary losses from conditioning encoder
        """
        aux_losses = {}

        if self.cond_source == "odor_onehot":
            # Use CondUNet1D's native odor conditioning
            if odor_ids is None:
                odor_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            y_pred = self.unet(x, odor_ids)
            return y_pred, aux_losses

        # For other conditioning sources, extract the conditioning embedding
        if self.cond_source == "cpc":
            cond, z_seq = self.cond_encoder(x)
            # CPC loss requires context from GRU
            _, context = self.cond_encoder.gru(z_seq)
            context = context.squeeze(0).unsqueeze(1).expand(-1, z_seq.size(1), -1)
            aux_losses["cpc_loss"] = self.cond_encoder.cpc_loss(z_seq, context)
        elif self.cond_source == "vqvae":
            cond, vq_losses = self.cond_encoder(x)
            aux_losses.update(vq_losses)
        elif self.cond_source == "freq_disentangled":
            cond = self.cond_encoder(x)
        elif self.cond_source == "cycle_consistent":
            cond, cycle_losses = self.cond_encoder(x, y)
            aux_losses.update(cycle_losses)
        elif self.cond_source == "spectro_temporal":
            # SpectroTemporalEncoder: extract style embedding from input signal dynamics
            # No auxiliary loss - purely unsupervised style extraction
            cond = self.cond_encoder(x, odor_emb=None)  # [B, 128] - already correct dim
        else:
            cond = torch.zeros(x.size(0), self.embed_dim, device=x.device)

        # Project conditioning to UNet's expected dimension
        # Note: spectro_temporal already outputs 128-dim, but we still project for consistency
        if self.cond_source == "spectro_temporal":
            # SpectroTemporalEncoder outputs 128-dim, which matches UNet's emb_dim
            cond_projected = cond
        else:
            cond_projected = self.cond_projector(cond)

        # Inject conditioning into UNet by temporarily replacing odor embedding output
        # We use a forward hook to inject our conditioning
        y_pred = self._forward_with_custom_cond(x, cond_projected)

        return y_pred, aux_losses

    def _forward_with_custom_cond(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward through UNet with custom conditioning injection."""
        # Store original embedding module
        original_embed = self.unet.embed if hasattr(self.unet, 'embed') else None

        if original_embed is not None:
            # Create a wrapper that returns our conditioning
            class ConditioningOverride(nn.Module):
                def __init__(self, cond_tensor):
                    super().__init__()
                    self.cond = cond_tensor

                def forward(self, x):
                    return self.cond

            # Temporarily replace embed
            self.unet.embed = ConditioningOverride(cond)

            try:
                # Forward with dummy odor_ids (will be ignored)
                dummy_odor = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                y_pred = self.unet(x, dummy_odor)
            finally:
                # Restore original
                self.unet.embed = original_embed
        else:
            # Fallback: just forward without conditioning
            y_pred = self.unet(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        return y_pred


# =============================================================================
# Training
# =============================================================================

def compute_full_metrics(y_pred: np.ndarray, y_true: np.ndarray, sample_rate: float = 1000.0) -> Dict[str, float]:
    """Compute full metrics matching tier1 for comparability.

    Returns:
        Dict with: r2, mae, pearson, psd_error_db, r2_delta, r2_theta, r2_alpha, r2_beta, r2_gamma
    """
    if y_pred.ndim == 2:
        y_pred = y_pred[:, np.newaxis, :]
        y_true = y_true[:, np.newaxis, :]

    N, C, T = y_pred.shape

    # R² - per-channel then averaged
    pred_flat = y_pred.transpose(1, 0, 2).reshape(C, -1)
    true_flat = y_true.transpose(1, 0, 2).reshape(C, -1)

    ss_res = np.sum((true_flat - pred_flat) ** 2, axis=1)
    true_means = np.mean(true_flat, axis=1, keepdims=True)
    ss_tot = np.sum((true_flat - true_means) ** 2, axis=1)

    r2_per_channel = 1 - ss_res / (ss_tot + 1e-8)
    r2 = float(np.mean(r2_per_channel))

    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Pearson correlation
    pred_centered = pred_flat - np.mean(pred_flat, axis=1, keepdims=True)
    true_centered = true_flat - np.mean(true_flat, axis=1, keepdims=True)
    numerator = np.sum(pred_centered * true_centered, axis=1)
    denominator = np.sqrt(np.sum(pred_centered ** 2, axis=1) * np.sum(true_centered ** 2, axis=1))
    correlations = numerator / (denominator + 1e-8)
    valid_mask = np.isfinite(correlations)
    pearson = float(np.mean(correlations[valid_mask])) if valid_mask.any() else 0.0

    # PSD Error in dB
    try:
        _, psd_pred = scipy_signal.welch(y_pred, fs=sample_rate, axis=-1, nperseg=min(256, T))
        _, psd_true = scipy_signal.welch(y_true, fs=sample_rate, axis=-1, nperseg=min(256, T))
        log_pred = np.log10(psd_pred + 1e-10)
        log_true = np.log10(psd_true + 1e-10)
        psd_error_db = float(10 * np.mean(np.abs(log_pred - log_true)))
    except Exception:
        psd_error_db = np.nan

    # Per-frequency-band R²
    band_r2 = {}
    try:
        nyq = sample_rate / 2
        for band_name, (f_low, f_high) in FREQ_BANDS.items():
            if f_low >= nyq:
                continue
            low = f_low / nyq
            high = min(f_high / nyq, 0.99)
            if low < high:
                try:
                    b, a = scipy_signal.butter(4, [low, high], btype='band')
                    pred_filt = scipy_signal.filtfilt(b, a, y_pred, axis=-1)
                    true_filt = scipy_signal.filtfilt(b, a, y_true, axis=-1)

                    pred_band_flat = pred_filt.transpose(1, 0, 2).reshape(C, -1)
                    true_band_flat = true_filt.transpose(1, 0, 2).reshape(C, -1)

                    ss_res_band = np.sum((true_band_flat - pred_band_flat) ** 2, axis=1)
                    true_band_means = np.mean(true_band_flat, axis=1, keepdims=True)
                    ss_tot_band = np.sum((true_band_flat - true_band_means) ** 2, axis=1)

                    r2_band = 1 - ss_res_band / (ss_tot_band + 1e-8)
                    band_r2[band_name] = float(np.mean(r2_band))
                except Exception:
                    pass
    except Exception:
        pass

    return {
        "r2": r2,
        "mae": mae,
        "pearson": pearson,
        "psd_error_db": psd_error_db,
        **{f"r2_{k}": v for k, v in band_r2.items()},
    }


def train_with_conditioning(
    model: ConditionedTranslator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    device: torch.device,
    lr: float = 0.0002,  # Same as train.py
    aux_loss_weight: float = 0.1,
    show_progress: bool = True,
    weight_l1: float = 1.0,
    weight_wavelet: float = 10.0,  # Same as train.py DEFAULT_CONFIG
) -> Dict[str, float]:
    """Train model and return validation metrics.

    Uses combined L1 + Wavelet loss (same as train.py):
        loss = weight_l1 * L1_loss + weight_wavelet * wavelet_loss

    Tracks best validation loss and returns metrics from best epoch.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Build wavelet loss (same config as train.py)
    wavelet_loss_fn = None
    if HAS_WAVELET_LOSS:
        wavelet_loss_fn = build_wavelet_loss(
            wavelet="morlet",
            omega0=3.0,
            use_complex_morlet=False,
        )
        if hasattr(wavelet_loss_fn, 'to'):
            wavelet_loss_fn = wavelet_loss_fn.to(device)

    # Track best validation loss for best model selection
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    epoch_iter = tqdm(range(n_epochs), desc="Training", leave=False, disable=not show_progress)

    for epoch in epoch_iter:
        # Training phase
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if len(batch) == 4:
                X, y, cond, odor_ids = batch
                X, y, odor_ids = X.to(device), y.to(device), odor_ids.to(device)
            elif len(batch) == 3:
                X, y, odor_ids = batch
                X, y, odor_ids = X.to(device), y.to(device), odor_ids.to(device)
            else:
                X, y = batch
                X, y = X.to(device), y.to(device)
                odor_ids = None

            optimizer.zero_grad()

            # Forward
            y_pred, aux_losses = model(X, y, odor_ids)

            # Main loss: L1 + Wavelet (same as train.py)
            l1_loss = F.l1_loss(y_pred, y)
            loss = weight_l1 * l1_loss

            if wavelet_loss_fn is not None:
                wav_loss = wavelet_loss_fn(y_pred, y)
                loss = loss + weight_wavelet * wav_loss

            # Add auxiliary losses from conditioning encoders
            for aux_name, aux_loss in aux_losses.items():
                if aux_loss is not None and not torch.isnan(aux_loss):
                    loss = loss + aux_loss_weight * aux_loss

            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

        scheduler.step()

        # Validation phase - track best model
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    X, y, cond, odor_ids = batch
                    X, y, odor_ids = X.to(device), y.to(device), odor_ids.to(device)
                elif len(batch) == 3:
                    X, y, odor_ids = batch
                    X, y, odor_ids = X.to(device), y.to(device), odor_ids.to(device)
                else:
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                    odor_ids = None

                y_pred, _ = model(X, y, odor_ids)

                # Compute validation loss (L1 + Wavelet)
                l1_loss = F.l1_loss(y_pred, y)
                batch_loss = weight_l1 * l1_loss

                if wavelet_loss_fn is not None:
                    wav_loss = wavelet_loss_fn(y_pred, y)
                    batch_loss = batch_loss + weight_wavelet * wav_loss

                if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
                    val_loss += batch_loss.item()
                    val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Update progress bar
        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_iter.set_postfix(loss=f"{avg_loss:.4f}", val_loss=f"{avg_val_loss:.4f}", best=best_epoch)

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if show_progress:
            print(f"    Loaded best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    # Final Validation with best model
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                X, y, cond, odor_ids = batch
                X, y, odor_ids = X.to(device), y.to(device), odor_ids.to(device)
            elif len(batch) == 3:
                X, y, odor_ids = batch
                X, y, odor_ids = X.to(device), y.to(device), odor_ids.to(device)
            else:
                X, y = batch
                X, y = X.to(device), y.to(device)
                odor_ids = None

            y_pred, _ = model(X, y, odor_ids)
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # Compute full metrics (matching tier1)
    return compute_full_metrics(preds, targets)


# =============================================================================
# Main Tier Function
# =============================================================================

def run_tier1_5(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    architecture: Optional[str] = None,
    conditionings: Optional[List[str]] = None,
    odor_labels: Optional[torch.Tensor] = None,
    n_epochs: int = 50,
    batch_size: int = 8,  # Same as train.py
    register: bool = True,
) -> Tier1_5Result:
    """Run Tier 1.5: Auto-Conditioning Signal Sources (Single Run).

    Args:
        X: Input data [N, C_in, T]
        y: Target data [N, C_out, T]
        device: Device to use
        architecture: Not used (kept for compatibility)
        conditionings: List of conditioning sources to test
        odor_labels: Odor ID labels [N] for odor_onehot conditioning
        n_epochs: Training epochs
        batch_size: Batch size
        register: Whether to register results

    Returns:
        Tier1_5Result with all conditioning comparisons
    """
    conditionings = conditionings or CONDITIONING_SOURCES

    print(f"\nTesting {len(conditionings)} conditioning SOURCES (single run each)")
    print(f"Sources: {conditionings}")
    print(f"Epochs: {n_epochs}")

    N = X.shape[0]
    in_channels = X.shape[1]
    out_channels = y.shape[1]

    # Create odor labels if not provided
    if odor_labels is None:
        n_odors = 10
        odor_labels = torch.randint(0, n_odors, (N,))
    else:
        n_odors = int(odor_labels.max().item()) + 1

    # Simple 80/20 train/val split (like tier1)
    np.random.seed(42)
    indices = np.random.permutation(N)
    n_train = int(N * 0.8)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    all_results: List[ConditioningRunResult] = []

    # Progress bar for conditioning sources
    cond_pbar = tqdm(conditionings, desc="Conditioning Sources", unit="source")

    for cond_source in cond_pbar:
        cond_pbar.set_postfix(source=cond_source)

        # Prepare data
        X_train = X[train_idx]
        y_train = y[train_idx]
        odor_train = odor_labels[train_idx]

        X_val = X[val_idx]
        y_val = y[val_idx]
        odor_val = odor_labels[val_idx]

        train_dataset = TensorDataset(X_train, y_train, odor_train)
        val_dataset = TensorDataset(X_val, y_val, odor_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Create model
        model = None
        try:
            model = ConditionedTranslator(
                in_channels=in_channels,
                out_channels=out_channels,
                cond_source=cond_source,
                embed_dim=64,
                n_odors=n_odors,
            ).to(device)

            start_time = time.time()
            metrics = train_with_conditioning(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=n_epochs,
                device=device,
                show_progress=True,
            )
            elapsed = time.time() - start_time

            result = ConditioningRunResult(
                conditioning=cond_source,
                r2=metrics["r2"],
                mae=metrics["mae"],
                pearson=metrics["pearson"],
                psd_error_db=metrics["psd_error_db"],
                r2_delta=metrics.get("r2_delta"),
                r2_theta=metrics.get("r2_theta"),
                r2_alpha=metrics.get("r2_alpha"),
                r2_beta=metrics.get("r2_beta"),
                r2_gamma=metrics.get("r2_gamma"),
                training_time=elapsed,
                success=True,
            )
            all_results.append(result)
            print(f"  {cond_source}: R²={metrics['r2']:.4f}, Pearson={metrics['pearson']:.4f}")

        except Exception as e:
            print(f"  {cond_source}: FAILED - {e}")
            all_results.append(ConditioningRunResult(
                conditioning=cond_source,
                r2=float("-inf"),
                mae=float("inf"),
                pearson=0.0,
                psd_error_db=float("inf"),
                training_time=0,
                success=False,
                error=str(e),
            ))

        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()

    # Build results dict (no aggregation needed for single run)
    cond_results = {}
    for result in all_results:
        if result.success:
            cond_results[result.conditioning] = ConditioningResult(
                conditioning=result.conditioning,
                r2_mean=result.r2,
                r2_std=0.0,  # Single run, no std
                mae_mean=result.mae,
                mae_std=0.0,
            )

    # Find best
    if cond_results:
        best_cond = max(cond_results.values(), key=lambda x: x.r2_mean)
    else:
        best_cond = ConditioningResult("none", 0.0, 0.0, 0.0, 0.0)

    # Architecture name (for compatibility)
    arch_name = "conditioned_translator"
    registry = get_registry()
    if registry.tier1 is not None and registry.tier1.top_architectures:
        arch_name = registry.tier1.top_architectures[0]

    result = Tier1_5Result(
        architecture=arch_name,
        results=cond_results,
        best_conditioning=best_cond.conditioning,
        best_r2_mean=best_cond.r2_mean,
        best_r2_std=best_cond.r2_std,
    )

    # Register
    if register:
        registry = get_registry(ARTIFACTS_DIR)
        registry.register_tier1_5(result)

    # Print summary
    print("\n" + "=" * 60)
    print("TIER 1.5 RESULTS: Auto-Conditioning Signal Sources")
    print("=" * 60)

    if cond_results:
        sorted_results = sorted(cond_results.values(), key=lambda x: x.r2_mean, reverse=True)
        for r in sorted_results:
            marker = " <-- BEST" if r.conditioning == best_cond.conditioning else ""
            print(f"  {r.conditioning:20s}: R² = {r.r2_mean:.4f} ± {r.r2_std:.4f}{marker}")
    else:
        print("  No valid results obtained")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tier 1.5: Auto-Conditioning Signal Sources (Single Run)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TIER 1.5: Auto-Conditioning Signal Sources (Single Run)")
    print("=" * 60)
    print()
    print("Testing conditioning SOURCES (what to condition on):")
    print("  1. odor_onehot:       One-hot odor identity (baseline)")
    print("  2. cpc:               Contrastive Predictive Coding")
    print("  3. vqvae:             Vector Quantized VAE codes")
    print("  4. freq_disentangled: Frequency-band-specific latents")
    print("  5. cycle_consistent:  Cycle-consistent latent")
    print("  6. spectro_temporal:  FFT + temporal conv (auto-conditioning)")
    print()

    # Load data
    if args.dry_run:
        N, C_in, C_out, T = 100, 32, 32, 500
        X = torch.randn(N, C_in, T)
        y = torch.randn(N, C_out, T)
        odor_labels = torch.randint(0, 5, (N,))
        conditionings = FAST_SOURCES
        n_epochs = 5
        print("[DRY-RUN] Using synthetic data\n")
    else:
        try:
            from data import prepare_data
            data = prepare_data()
            ob = data["ob"]
            pcx = data["pcx"]
            train_idx = data["train_idx"]
            val_idx = data["val_idx"]

            idx = np.concatenate([train_idx, val_idx])
            X = torch.from_numpy(ob[idx]).float()
            y = torch.from_numpy(pcx[idx]).float()

            # Try to get odor labels if available
            if "odor_labels" in data:
                odor_labels = torch.from_numpy(data["odor_labels"][idx]).long()
            else:
                odor_labels = torch.randint(0, 10, (len(idx),))

            conditionings = CONDITIONING_SOURCES
            n_epochs = args.epochs
            print(f"Loaded real data: {X.shape}\n")
        except Exception as e:
            print(f"Could not load real data ({e}), using synthetic\n")
            N, C_in, C_out, T = 160, 32, 32, 1000
            X = torch.randn(N, C_in, T)
            y = torch.randn(N, C_out, T)
            odor_labels = torch.randint(0, 10, (N,))
            conditionings = CONDITIONING_SOURCES
            n_epochs = args.epochs

    run_tier1_5(
        X=X,
        y=y,
        device=device,
        conditionings=conditionings,
        odor_labels=odor_labels,
        n_epochs=n_epochs,
    )


if __name__ == "__main__":
    main()
