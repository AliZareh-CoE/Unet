"""Neural signal translation model: CondUNet1D.

This module contains the main model architecture for OB->PCx neural signal translation:
- CondUNet1D: Conditional U-Net with FiLM modulation (our method)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Constants
# =============================================================================
BASE_CHANNELS = 128
NUM_ODORS = 7
SAMPLING_RATE_HZ = 1000
MAX_FREQ_HZ = 100  # LFP frequency limit (gamma band upper bound)

# Wavelet defaults
WAVELET_FAMILY = "morlet"
USE_COMPLEX_MORLET = True

# Neural frequency bands covering LFP range (1-100 Hz)
# Neuroscience-meaningful spacing:
# - Gamma (30-100 Hz): 3 bands for fast oscillations
# - Beta (12-30 Hz): 3 bands
# - Alpha (8-12 Hz): 1 band
# - Theta (4-8 Hz): 1 band
# - Delta (1-4 Hz): 2 bands
NEURAL_FREQ_BANDS = [
    100, 80, 60,         # Gamma
    40, 30, 20,          # Beta
    12, 8,               # Alpha
    4,                   # Theta
    2, 1,                # Delta
]


# =============================================================================
# Activation Function Helper
# =============================================================================

def get_activation(name: str) -> nn.Module:
    """Get activation function by name.

    Args:
        name: Activation name - one of: relu, leaky_relu, gelu, silu, mish

    Returns:
        PyTorch activation module
    """
    activations = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
        "mish": nn.Mish(inplace=True),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name]


# =============================================================================
# Core Building Blocks
# =============================================================================

class SelfAttention1D(nn.Module):
    """1D Self-Attention mechanism for capturing long-range dependencies."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.key = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.permute(0, 2, 1)
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        v = v.permute(0, 2, 1)
        out = torch.bmm(attention, v)
        out = out.permute(0, 2, 1)
        return x + self.gamma * out


class CrossFrequencyCouplingAttention(nn.Module):
    """Attention that models cross-frequency coupling (e.g., theta-gamma).

    Neural signals exhibit cross-frequency coupling where the phase of slow
    oscillations modulates the amplitude of fast oscillations. This attention
    mechanism explicitly models these interactions.

    Args:
        channels: Number of input/output channels
        n_heads: Number of attention heads
        sample_rate: Sampling rate in Hz
        bands: List of (low_freq, high_freq) tuples defining frequency bands
        chunk_size: Size of attention chunks to avoid OOM on long sequences
    """

    def __init__(
        self,
        channels: int,
        n_heads: int = 4,
        sample_rate: float = 1000.0,
        bands: Optional[List[Tuple[float, float]]] = None,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Default: neuroscience-relevant bands
        if bands is None:
            bands = [
                (1, 4),     # Delta
                (4, 8),     # Theta
                (8, 12),    # Alpha
                (12, 30),   # Beta
                (30, 80),   # Gamma
            ]
        self.bands = bands
        self.n_bands = len(bands)

        # Cross-band attention: each band attends to all bands
        self.band_q = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in bands])
        self.band_k = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in bands])
        self.band_v = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in bands])

        # Combine bands
        self.out_proj = nn.Conv1d(channels * self.n_bands, channels, 1)

        # Learnable cross-band coupling weights
        self.coupling_weights = nn.Parameter(torch.ones(self.n_bands, self.n_bands) / self.n_bands)
        self.gate = nn.Parameter(torch.zeros(1))
        self._filters_cache = None

    def _build_bandpass_filters(self, T: int, device: torch.device) -> List[torch.Tensor]:
        """Build FFT-based bandpass filter masks."""
        if self._filters_cache is not None and self._filters_cache[0].shape[-1] == T // 2 + 1:
            return [f.to(device) for f in self._filters_cache]

        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sample_rate)
        filters = []
        for (low, high) in self.bands:
            mask = torch.sigmoid((freqs - low) / 2) * torch.sigmoid((high - freqs) / 2)
            filters.append(mask)

        self._filters_cache = filters
        return [f.to(device) for f in filters]

    def _extract_band(self, x: torch.Tensor, band_mask: torch.Tensor) -> torch.Tensor:
        """Extract frequency band using FFT."""
        x_fft = torch.fft.rfft(x, dim=-1)
        x_filtered = x_fft * band_mask.unsqueeze(0).unsqueeze(0)
        return torch.fft.irfft(x_filtered, n=x.shape[-1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross-frequency coupling attention."""
        B, C, T = x.shape
        device = x.device

        original_dtype = x.dtype
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()

        filters = self._build_bandpass_filters(T, device)
        band_signals = [self._extract_band(x, f) for f in filters]

        band_outputs = []
        coupling = F.softmax(self.coupling_weights, dim=-1)
        chunk_size = self.chunk_size

        for i, x_band in enumerate(band_signals):
            x_band = x_band.to(original_dtype)
            q_full = self.band_q[i](x_band).reshape(B, self.n_heads, self.head_dim, T).transpose(2, 3)

            attended_full = torch.zeros_like(q_full)
            for j, x_other in enumerate(band_signals):
                x_other = x_other.to(original_dtype)
                k_full = self.band_k[j](x_other).reshape(B, self.n_heads, self.head_dim, T).transpose(2, 3)
                v_full = self.band_v[j](x_other).reshape(B, self.n_heads, self.head_dim, T).transpose(2, 3)

                out_chunks = []
                for chunk_start in range(0, T, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, T)
                    q_chunk = q_full[:, :, chunk_start:chunk_end, :]
                    k_chunk = k_full[:, :, chunk_start:chunk_end, :]
                    v_chunk = v_full[:, :, chunk_start:chunk_end, :]

                    attn = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    attn = F.softmax(attn, dim=-1)
                    out_chunk = torch.matmul(attn, v_chunk)
                    out_chunks.append(out_chunk)

                out = torch.cat(out_chunks, dim=2)
                attended_full = attended_full + coupling[i, j] * out

            band_outputs.append(attended_full.transpose(2, 3).reshape(B, C, T))

        combined = torch.cat(band_outputs, dim=1)
        out = self.out_proj(combined)
        return (x + self.gate * out).to(original_dtype)

    def extra_repr(self) -> str:
        band_str = ", ".join(f"{l}-{h}Hz" for l, h in self.bands)
        return f"channels={self.channels}, chunk_size={self.chunk_size}, bands=[{band_str}]"


class CrossFrequencyCouplingAttentionV2(nn.Module):
    """Optimized Cross-Frequency Coupling Attention for multi-GPU setups.

    Mathematically equivalent to CrossFrequencyCouplingAttention but with:
    - Flash Attention via scaled_dot_product_attention (2-4x faster on A100)
    - Batched FFT decomposition (single FFT, all bands in parallel)
    - Batched Q/K/V projections (eliminates O(n_bands²) loop overhead)
    - torch.compile compatible design

    Args:
        channels: Number of input/output channels
        n_heads: Number of attention heads
        sample_rate: Sampling rate in Hz
        bands: List of (low_freq, high_freq) tuples defining frequency bands
        enable_flash: Use Flash Attention when available (default: True)
    """

    def __init__(
        self,
        channels: int,
        n_heads: int = 4,
        sample_rate: float = 1000.0,
        bands: Optional[List[Tuple[float, float]]] = None,
        chunk_size: int = 256,  # Kept for API compatibility, not used
        enable_flash: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.sample_rate = sample_rate
        self.enable_flash = enable_flash
        self.chunk_size = chunk_size  # For API compat

        # Default: neuroscience-relevant bands
        if bands is None:
            bands = [
                (1, 4),     # Delta
                (4, 8),     # Theta
                (8, 12),    # Alpha
                (12, 30),   # Beta
                (30, 80),   # Gamma
            ]
        self.bands = bands
        self.n_bands = len(bands)

        # Batched projections: single Conv1d per Q/K/V across all bands
        # Shape: (n_bands * channels) -> allows parallel processing
        self.proj_q = nn.Conv1d(channels * self.n_bands, channels * self.n_bands, 1, groups=self.n_bands)
        self.proj_k = nn.Conv1d(channels * self.n_bands, channels * self.n_bands, 1, groups=self.n_bands)
        self.proj_v = nn.Conv1d(channels * self.n_bands, channels * self.n_bands, 1, groups=self.n_bands)

        # Output projection
        self.out_proj = nn.Conv1d(channels * self.n_bands, channels, 1)

        # Learnable cross-band coupling weights
        self.coupling_weights = nn.Parameter(torch.ones(self.n_bands, self.n_bands) / self.n_bands)
        self.gate = nn.Parameter(torch.zeros(1))

        # Cached filter masks (stacked for batched FFT)
        self._filters_cache: Optional[torch.Tensor] = None
        self._cache_len: int = 0

        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _build_bandpass_filters_batched(self, T: int, device: torch.device) -> torch.Tensor:
        """Build stacked FFT bandpass filter masks for all bands at once.

        Returns:
            Tensor of shape (n_bands, T//2+1) with all filter masks stacked
        """
        freq_len = T // 2 + 1
        if self._filters_cache is not None and self._cache_len == freq_len:
            return self._filters_cache.to(device)

        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sample_rate)
        masks = []
        for (low, high) in self.bands:
            mask = torch.sigmoid((freqs - low) / 2) * torch.sigmoid((high - freqs) / 2)
            masks.append(mask)

        # Stack: (n_bands, freq_len)
        self._filters_cache = torch.stack(masks, dim=0)
        self._cache_len = freq_len
        return self._filters_cache.to(device)

    def _extract_all_bands(self, x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
        """Extract all frequency bands in a single batched FFT operation.

        Args:
            x: Input tensor (B, C, T)
            filters: Stacked filter masks (n_bands, freq_len)

        Returns:
            Band signals stacked: (B, n_bands, C, T)
        """
        B, C, T = x.shape
        # Single FFT for input
        x_fft = torch.fft.rfft(x, dim=-1)  # (B, C, freq_len)

        # Expand for broadcasting: (B, 1, C, freq_len) * (1, n_bands, 1, freq_len)
        x_fft_expanded = x_fft.unsqueeze(1)  # (B, 1, C, freq_len)
        filters_expanded = filters.unsqueeze(0).unsqueeze(2)  # (1, n_bands, 1, freq_len)

        # Apply all filters at once
        x_filtered = x_fft_expanded * filters_expanded  # (B, n_bands, C, freq_len)

        # Single batched iFFT
        # Reshape for batch processing: (B * n_bands, C, freq_len)
        x_filtered_flat = x_filtered.reshape(B * self.n_bands, C, -1)
        band_signals = torch.fft.irfft(x_filtered_flat, n=T, dim=-1)  # (B * n_bands, C, T)

        return band_signals.reshape(B, self.n_bands, C, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply optimized cross-frequency coupling attention."""
        B, C, T = x.shape
        device = x.device
        original_dtype = x.dtype

        # FFT requires float32
        if x.dtype in (torch.bfloat16, torch.float16):
            x_float = x.float()
        else:
            x_float = x

        # === Batched FFT Band Extraction ===
        filters = self._build_bandpass_filters_batched(T, device)
        band_signals = self._extract_all_bands(x_float, filters)  # (B, n_bands, C, T)

        # Convert back to original dtype for attention
        band_signals = band_signals.to(original_dtype)

        # === Batched Q/K/V Projections ===
        # Reshape: (B, n_bands, C, T) -> (B, n_bands * C, T) for grouped conv
        band_flat = band_signals.reshape(B, self.n_bands * C, T)

        Q = self.proj_q(band_flat)  # (B, n_bands * C, T)
        K = self.proj_k(band_flat)
        V = self.proj_v(band_flat)

        # Reshape for multi-head attention: (B, n_bands, n_heads, head_dim, T)
        Q = Q.reshape(B, self.n_bands, self.n_heads, self.head_dim, T)
        K = K.reshape(B, self.n_bands, self.n_heads, self.head_dim, T)
        V = V.reshape(B, self.n_bands, self.n_heads, self.head_dim, T)

        # Transpose to (B, n_bands, n_heads, T, head_dim) for attention
        Q = Q.permute(0, 1, 2, 4, 3)  # (B, n_bands, n_heads, T, head_dim)
        K = K.permute(0, 1, 2, 4, 3)
        V = V.permute(0, 1, 2, 4, 3)

        # === Cross-Band Attention with Coupling ===
        coupling = F.softmax(self.coupling_weights, dim=-1)  # (n_bands, n_bands)

        # Compute all band-pair attention outputs
        # We need: for each query band i, attend to all key/value bands j with coupling[i,j]

        # Reshape for batch attention: merge B and n_heads
        # Q: (B, n_bands, n_heads, T, head_dim) -> (B * n_heads, n_bands, T, head_dim)
        Q_merged = Q.permute(0, 2, 1, 3, 4).reshape(B * self.n_heads, self.n_bands, T, self.head_dim)
        K_merged = K.permute(0, 2, 1, 3, 4).reshape(B * self.n_heads, self.n_bands, T, self.head_dim)
        V_merged = V.permute(0, 2, 1, 3, 4).reshape(B * self.n_heads, self.n_bands, T, self.head_dim)

        # Output accumulator
        out_bands = torch.zeros_like(Q_merged)  # (B * n_heads, n_bands, T, head_dim)

        # Cross-band attention: each band i attends to all bands j
        for i in range(self.n_bands):
            q_i = Q_merged[:, i:i+1, :, :]  # (B * n_heads, 1, T, head_dim)

            for j in range(self.n_bands):
                k_j = K_merged[:, j, :, :]  # (B * n_heads, T, head_dim)
                v_j = V_merged[:, j, :, :]  # (B * n_heads, T, head_dim)

                # Use Flash Attention if available (PyTorch 2.0+)
                if self.enable_flash and hasattr(F, 'scaled_dot_product_attention'):
                    # SDPA expects: (B, n_heads, T, head_dim) but we have merged B*n_heads
                    # Reshape for SDPA: add head dim of 1
                    attn_out = F.scaled_dot_product_attention(
                        q_i,  # (B * n_heads, 1, T, head_dim)
                        k_j.unsqueeze(1),  # (B * n_heads, 1, T, head_dim)
                        v_j.unsqueeze(1),  # (B * n_heads, 1, T, head_dim)
                        dropout_p=0.0,
                        is_causal=False,
                    ).squeeze(1)  # (B * n_heads, T, head_dim)
                else:
                    # Fallback to manual attention
                    attn_scores = torch.matmul(q_i.squeeze(1), k_j.transpose(-2, -1)) * self.scale
                    attn_probs = F.softmax(attn_scores, dim=-1)
                    attn_out = torch.matmul(attn_probs, v_j)  # (B * n_heads, T, head_dim)

                out_bands[:, i, :, :] = out_bands[:, i, :, :] + coupling[i, j] * attn_out

        # Reshape back: (B * n_heads, n_bands, T, head_dim) -> (B, n_bands, C, T)
        out_bands = out_bands.reshape(B, self.n_heads, self.n_bands, T, self.head_dim)
        out_bands = out_bands.permute(0, 2, 1, 4, 3)  # (B, n_bands, n_heads, head_dim, T)
        out_bands = out_bands.reshape(B, self.n_bands * C, T)

        # === Output Projection ===
        out = self.out_proj(out_bands)  # (B, C, T)

        return (x + self.gate * out).to(original_dtype)

    @classmethod
    def from_v1(cls, v1_module: 'CrossFrequencyCouplingAttention') -> 'CrossFrequencyCouplingAttentionV2':
        """Create V2 from existing V1 module, copying weights for seamless upgrade."""
        v2 = cls(
            channels=v1_module.channels,
            n_heads=v1_module.n_heads,
            sample_rate=v1_module.sample_rate,
            bands=v1_module.bands,
            chunk_size=v1_module.chunk_size,
        )

        # Copy coupling weights and gate directly
        v2.coupling_weights.data.copy_(v1_module.coupling_weights.data)
        v2.gate.data.copy_(v1_module.gate.data)

        # Copy Q/K/V weights from individual band projections to grouped conv
        with torch.no_grad():
            n_bands = v1_module.n_bands

            # Stack individual band weights into grouped conv format
            q_weights = torch.cat([v1_module.band_q[i].weight for i in range(n_bands)], dim=0)
            k_weights = torch.cat([v1_module.band_k[i].weight for i in range(n_bands)], dim=0)
            v_weights = torch.cat([v1_module.band_v[i].weight for i in range(n_bands)], dim=0)

            q_biases = torch.cat([v1_module.band_q[i].bias for i in range(n_bands)], dim=0)
            k_biases = torch.cat([v1_module.band_k[i].bias for i in range(n_bands)], dim=0)
            v_biases = torch.cat([v1_module.band_v[i].bias for i in range(n_bands)], dim=0)

            v2.proj_q.weight.data.copy_(q_weights)
            v2.proj_k.weight.data.copy_(k_weights)
            v2.proj_v.weight.data.copy_(v_weights)
            v2.proj_q.bias.data.copy_(q_biases)
            v2.proj_k.bias.data.copy_(k_biases)
            v2.proj_v.bias.data.copy_(v_biases)

            # Copy output projection
            v2.out_proj.weight.data.copy_(v1_module.out_proj.weight.data)
            v2.out_proj.bias.data.copy_(v1_module.out_proj.bias.data)

        return v2

    def extra_repr(self) -> str:
        band_str = ", ".join(f"{l}-{h}Hz" for l, h in self.bands)
        return f"channels={self.channels}, n_heads={self.n_heads}, flash={self.enable_flash}, bands=[{band_str}]"


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for conditional generation.

    Supported modes:
        - 'none': Bypass mode (unconditional)
        - 'cross_attn_gated': Cross-attention with output gating (default)
    """
    def __init__(self, channels: int, emb_dim: int, cond_mode: str = "cross_attn_gated"):
        super().__init__()
        self.cond_mode = cond_mode
        self.channels = channels
        self.emb_dim = emb_dim

        if cond_mode == "cross_attn_gated":
            # Cross-attention with output gating
            self.q_proj = nn.Conv1d(channels, channels, 1)
            self.k_proj = nn.Linear(emb_dim, channels)
            self.v_proj = nn.Linear(emb_dim, channels)
            self.out_proj = nn.Conv1d(channels, channels, 1)
            self.scale_factor = channels ** -0.5
            self.gate = nn.Sequential(
                nn.Linear(emb_dim, channels),
                nn.Sigmoid(),
            )
        elif cond_mode != "none":
            raise ValueError(f"Unknown cond_mode: {cond_mode}. Supported: 'none', 'cross_attn_gated'")

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cond_mode == "none" or emb is None:
            return x

        # Cross-attention with output gating (optimized)
        # q: [B, C, T], k/v: [B, C, 1]
        q = self.q_proj(x)
        k = self.k_proj(emb).unsqueeze(-1)
        v = self.v_proj(emb).unsqueeze(-1)
        # Fused: softmax(q * k * scale) * v - no expand needed, broadcasting handles it
        attn_weights = torch.softmax(q * k * self.scale_factor, dim=1)
        out = self.out_proj(attn_weights * v)
        # Fused gate multiply
        return x + self.gate(emb).unsqueeze(-1) * out


class ConvBlock(nn.Module):
    """Downsampling convolution block with normalization and FiLM conditioning."""
    def __init__(
        self,
        in_c: int,
        out_c: int,
        emb_dim: int,
        downsample: bool = False,
        dropout: float = 0.0,
        norm_type: str = "batch",
        cond_mode: str = "cross_attn_gated",
    ):
        super().__init__()
        stride = 2 if downsample else 1
        self.cond_mode = cond_mode

        # Build normalization layer
        if norm_type == "instance":
            norm = nn.InstanceNorm1d(out_c, affine=True)
        elif norm_type == "batch":
            norm = nn.BatchNorm1d(out_c)
        elif norm_type == "group":
            num_groups = min(8, out_c)
            while out_c % num_groups != 0:
                num_groups -= 1
            norm = nn.GroupNorm(num_groups, out_c)
        elif norm_type == "layer":
            # LayerNorm for 1D: normalize over channels and time
            norm = nn.GroupNorm(1, out_c)  # GroupNorm with 1 group = LayerNorm
        elif norm_type == "rms":
            # RMSNorm approximation using LayerNorm without centering
            norm = nn.GroupNorm(1, out_c, affine=True)  # Close approximation
        elif norm_type == "none":
            norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}. Available: batch, instance, group, layer, rms, none")

        layers = [
            nn.Conv1d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            norm,
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout1d(p=dropout))

        self.conv = nn.Sequential(*layers)
        self.film = FiLM(out_c, emb_dim, cond_mode=cond_mode)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.film(self.conv(x), emb)


class UpBlock(nn.Module):
    """Upsampling block with skip connections and FiLM conditioning.

    Supports configurable skip connection types:
    - "concat": Concatenate skip features with upsampled features (default, more capacity)
    - "add": Add skip features to upsampled features (simpler, fewer parameters)
    """
    def __init__(
        self,
        in_c: int,
        skip_c: int,
        out_c: int,
        emb_dim: int,
        dropout: float = 0.0,
        norm_type: str = "batch",
        cond_mode: str = "cross_attn_gated",
        skip_type: str = "concat",  # "concat" or "add"
    ):
        super().__init__()
        self.skip_type = skip_type
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1)

        if skip_type == "concat":
            # Concatenation: input to block is out_c + skip_c channels
            self.block = ConvBlock(out_c + skip_c, out_c, emb_dim, downsample=False,
                                   dropout=dropout, norm_type=norm_type, cond_mode=cond_mode)
            self.skip_proj = None
        elif skip_type == "add":
            # Addition: need to project skip to match out_c channels
            self.block = ConvBlock(out_c, out_c, emb_dim, downsample=False,
                                   dropout=dropout, norm_type=norm_type, cond_mode=cond_mode)
            # Project skip to out_c channels if dimensions don't match
            if skip_c != out_c:
                self.skip_proj = nn.Conv1d(skip_c, out_c, kernel_size=1)
            else:
                self.skip_proj = None
        else:
            raise ValueError(f"Unknown skip_type: {skip_type}. Available: concat, add")

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            if x.shape[-1] < skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                start = diff // 2
                skip = skip[..., start : start + x.shape[-1]]
            else:
                diff = x.shape[-1] - skip.shape[-1]
                start = diff // 2
                x = x[..., start : start + skip.shape[-1]]

        if self.skip_type == "concat":
            x = torch.cat([x, skip], dim=1)
        else:  # add
            if self.skip_proj is not None:
                skip = self.skip_proj(skip)
            x = x + skip

        return self.block(x, emb)


# =============================================================================
# Modern Convolution Blocks - Multi-scale Dilated Depthwise Separable + SE
# =============================================================================

class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-Excitation channel attention for 1D signals.

    Models inter-electrode correlations by learning channel-wise attention weights.
    Essential for neural signals where different electrodes capture correlated activity.

    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio for bottleneck (default: 4)
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        reduced = max(channels // reduction, 8)  # Ensure minimum 8 channels
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.GELU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class DilatedConv1d(nn.Module):
    """Standard dilated convolution for multi-scale temporal patterns.

    Uses STANDARD convolution (not depthwise separable) to preserve cross-channel
    learning. This is critical for neural LFP signals where cross-electrode
    coherence carries essential information about brain state.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size (default: 7, ConvNeXt-style)
        stride: Stride for downsampling
        dilation: Dilation rate for expanded receptive field
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        # Compute padding to preserve temporal dimension (accounting for dilation)
        padding = (kernel_size - 1) * dilation // 2

        # STANDARD conv - processes ALL channels together for cross-electrode learning
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultiScaleDilatedConv1d(nn.Module):
    """Multi-scale dilated convolution capturing all LFP frequency bands.

    Uses parallel STANDARD dilated convolutions (not depthwise separable) with
    rates tuned for neural frequency bands, then combines them via learnable weights.

    Cross-channel learning is preserved - critical for neural LFP signals where
    cross-electrode coherence carries essential information.

    Dilation rates for 1kHz sampling, kernel_size=7:
    - d=1:  receptive field = 7 samples = 7ms (gamma: 30-100Hz)
    - d=4:  receptive field = 25 samples = 25ms (beta: 12-30Hz)
    - d=16: receptive field = 97 samples = 97ms (alpha/theta: 4-12Hz)
    - d=32: receptive field = 193 samples = 193ms (delta: 1-4Hz)

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size (default: 7)
        stride: Stride for downsampling
        dilations: Tuple of dilation rates for each scale
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dilations: Tuple[int, ...] = (1, 4, 16, 32),
    ):
        super().__init__()
        self.n_scales = len(dilations)

        # Parallel standard dilated convolutions (preserves cross-channel learning)
        self.branches = nn.ModuleList([
            DilatedConv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=d,
            )
            for d in dilations
        ])

        # Learnable scale weights (softmax for stable combination)
        self.scale_weights = nn.Parameter(torch.ones(self.n_scales) / self.n_scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute weights once (fused softmax)
        weights = F.softmax(self.scale_weights, dim=0)
        # Stack branch outputs and weighted sum in single operation
        # This is more memory-efficient than accumulating
        branch_outputs = torch.stack([b(x) for b in self.branches], dim=0)  # [n_scales, B, C, T]
        # Einsum for weighted sum: [n] @ [n, B, C, T] -> [B, C, T]
        return torch.einsum('n,nbct->bct', weights, branch_outputs)


class ModernConvBlock(nn.Module):
    """Modern convolution block with multi-scale dilated convolutions and SE attention.

    This is a drop-in replacement for ConvBlock with:
    1. Multi-scale dilated convolutions for capturing all LFP frequency bands
    2. SE channel attention for modeling inter-electrode correlations
    3. FiLM conditioning preserved
    4. Configurable via conv_type parameter for backward compatibility

    Args:
        in_c: Input channels
        out_c: Output channels
        emb_dim: Embedding dimension for FiLM
        downsample: If True, use stride=2 for downsampling
        dropout: Dropout probability
        norm_type: Normalization type ("instance", "batch", "group")
        cond_mode: FiLM conditioning mode
        use_se: Whether to use SE attention (default: True)
        kernel_size: Kernel size for depthwise conv (default: 7)
        dilations: Dilation rates for multi-scale conv
    """
    def __init__(
        self,
        in_c: int,
        out_c: int,
        emb_dim: int,
        downsample: bool = False,
        dropout: float = 0.0,
        norm_type: str = "batch",
        cond_mode: str = "cross_attn_gated",
        use_se: bool = True,
        kernel_size: int = 7,
        dilations: Tuple[int, ...] = (1, 4, 16, 32),
    ):
        super().__init__()
        stride = 2 if downsample else 1
        self.use_se = use_se

        # Build normalization
        if norm_type == "instance":
            norm = nn.InstanceNorm1d(out_c, affine=True)
        elif norm_type == "batch":
            norm = nn.BatchNorm1d(out_c)
        elif norm_type == "group":
            num_groups = min(8, out_c)
            while out_c % num_groups != 0:
                num_groups -= 1
            norm = nn.GroupNorm(num_groups, out_c)
        elif norm_type == "layer":
            norm = nn.GroupNorm(1, out_c)  # GroupNorm with 1 group = LayerNorm
        elif norm_type == "rms":
            norm = nn.GroupNorm(1, out_c, affine=True)  # RMSNorm approximation
        elif norm_type == "none":
            norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}. Available: batch, instance, group, layer, rms, none")

        # Multi-scale dilated depthwise separable conv
        conv = MultiScaleDilatedConv1d(
            in_c, out_c,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations,
        )

        layers = [conv, norm, nn.GELU()]

        # Add SE attention for electrode correlation modeling
        if self.use_se:
            layers.append(SqueezeExcitation1D(out_c))

        if dropout > 0:
            layers.append(nn.Dropout1d(p=dropout))

        self.conv = nn.Sequential(*layers)
        self.film = FiLM(out_c, emb_dim, cond_mode=cond_mode)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.film(self.conv(x), emb)


class ModernUpBlock(nn.Module):
    """Upsampling block with modern convolutions.

    Uses ConvTranspose1d for upsampling, then modern conv block for refinement.

    Supports configurable skip connection types:
    - "concat": Concatenate skip features with upsampled features (default, more capacity)
    - "add": Add skip features to upsampled features (simpler, fewer parameters)

    Args:
        in_c: Input channels
        skip_c: Skip connection channels
        out_c: Output channels
        emb_dim: Embedding dimension for FiLM
        dropout: Dropout probability
        norm_type: Normalization type
        cond_mode: FiLM conditioning mode
        use_se: Whether to use SE attention
        kernel_size: Kernel size for depthwise conv
        dilations: Dilation rates for multi-scale conv
        skip_type: Skip connection type - "concat" or "add"
    """
    def __init__(
        self,
        in_c: int,
        skip_c: int,
        out_c: int,
        emb_dim: int,
        dropout: float = 0.0,
        norm_type: str = "batch",
        cond_mode: str = "cross_attn_gated",
        use_se: bool = True,
        kernel_size: int = 7,
        dilations: Tuple[int, ...] = (1, 4, 16, 32),
        skip_type: str = "concat",  # "concat" or "add"
    ):
        super().__init__()
        self.skip_type = skip_type
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1)

        if skip_type == "concat":
            # Concatenation: input to block is out_c + skip_c channels
            self.block = ModernConvBlock(
                out_c + skip_c, out_c, emb_dim,
                downsample=False,
                dropout=dropout,
                norm_type=norm_type,
                cond_mode=cond_mode,
                use_se=use_se,
                kernel_size=kernel_size,
                dilations=dilations,
            )
            self.skip_proj = None
        elif skip_type == "add":
            # Addition: need to project skip to match out_c channels
            self.block = ModernConvBlock(
                out_c, out_c, emb_dim,
                downsample=False,
                dropout=dropout,
                norm_type=norm_type,
                cond_mode=cond_mode,
                use_se=use_se,
                kernel_size=kernel_size,
                dilations=dilations,
            )
            # Project skip to out_c channels if dimensions don't match
            if skip_c != out_c:
                self.skip_proj = nn.Conv1d(skip_c, out_c, kernel_size=1)
            else:
                self.skip_proj = None
        else:
            raise ValueError(f"Unknown skip_type: {skip_type}. Available: concat, add")

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatches from convolutions
        if x.shape[-1] != skip.shape[-1]:
            if x.shape[-1] < skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                start = diff // 2
                skip = skip[..., start : start + x.shape[-1]]
            else:
                diff = x.shape[-1] - skip.shape[-1]
                start = diff // 2
                x = x[..., start : start + skip.shape[-1]]

        if self.skip_type == "concat":
            x = torch.cat([x, skip], dim=1)
        else:  # add
            if self.skip_proj is not None:
                skip = self.skip_proj(skip)
            x = x + skip

        return self.block(x, emb)


# =============================================================================
# Frequency Band Definitions and Utilities
# =============================================================================

# Neural frequency bands (Hz) - 10 bands for finer spectral resolution
# Finer bands allow more precise PSD correction across the LFP range
# Predefined neuroscience frequency bands (legacy, for backward compatibility)
FREQ_BANDS_NEURO = {
    "delta_lo": (1.0, 2.0),    # Slow delta
    "delta_hi": (2.0, 4.0),    # Fast delta
    "theta_lo": (4.0, 6.0),    # Slow theta
    "theta_hi": (6.0, 8.0),    # Fast theta
    "alpha": (8.0, 12.0),       # Alpha band
    "beta_lo": (12.0, 20.0),   # Low beta
    "beta_hi": (20.0, 30.0),   # High beta
    "gamma_lo": (30.0, 50.0),  # Low gamma
    "gamma_mid": (50.0, 70.0), # Mid gamma
    "gamma_hi": (70.0, 100.0), # High gamma
}
# Backward compatibility aliases
FREQ_BANDS = FREQ_BANDS_NEURO
FREQ_BAND_NAMES = list(FREQ_BANDS.keys())


def make_uniform_bands(band_width_hz: float, min_freq_hz: float = 1.0, max_freq_hz: float = 100.0) -> Dict[str, Tuple[float, float]]:
    """Generate uniformly-spaced frequency bands.

    Args:
        band_width_hz: Width of each band in Hz (e.g., 2.0 for 2Hz bands)
        min_freq_hz: Minimum frequency (default 1.0 Hz)
        max_freq_hz: Maximum frequency (default 100.0 Hz)

    Returns:
        Dict mapping band names to (low_freq, high_freq) tuples
    """
    bands = {}
    f = min_freq_hz
    i = 0
    while f < max_freq_hz:
        f_high = min(f + band_width_hz, max_freq_hz)
        bands[f"{int(f)}-{int(f_high)}Hz"] = (f, f_high)
        f = f_high
        i += 1
    return bands


# =============================================================================
# Envelope Histogram Matching (Closed-form amplitude dynamics correction)
# =============================================================================

class EnvelopeHistogramMatching(nn.Module):
    """Closed-form envelope scaling correction using mean/std matching.

    Applies linear scaling to match target amplitude statistics while preserving
    temporal structure. Call fit() with target data, then use forward() during inference.
    """

    def __init__(
        self,
        n_odors: int = NUM_ODORS,
    ):
        super().__init__()
        self.n_odors = n_odors

        # Store target envelope mean/std per odor [n_odors]
        # These are computed from target data via fit()
        self.register_buffer('target_mean', torch.zeros(n_odors))
        self.register_buffer('target_std', torch.ones(n_odors))
        self.register_buffer('fitted', torch.tensor(False))

    @torch.no_grad()
    def fit(
        self,
        targets: torch.Tensor,
        odor_ids: torch.Tensor,
    ) -> None:
        """Compute target envelope mean/std from data.

        Args:
            targets: Target signals [N, C, T]
            odor_ids: Odor indices [N]
        """
        device = targets.device
        N, C, T = targets.shape

        # Compute envelope for all targets
        targets_flat = targets.view(N * C, T).float()
        analytic = hilbert_torch(targets_flat)
        envelope = analytic.abs()  # [N*C, T]

        # Compute mean/std per odor
        for odor in range(self.n_odors):
            # Get all samples for this odor
            mask = (odor_ids == odor)
            if not mask.any():
                continue

            # Expand mask to match flattened shape
            mask_expanded = mask.unsqueeze(1).expand(-1, C).reshape(-1)
            odor_envelope = envelope[mask_expanded]  # [n_samples * C, T]

            # Compute mean and std of all envelope values for this odor
            self.target_mean[odor] = odor_envelope.mean()
            self.target_std[odor] = odor_envelope.std().clamp(min=1e-6)

        self.fitted.fill_(True)
        print(f"EnvelopeScaling fitted: mean=[{self.target_mean.min():.3f}, {self.target_mean.max():.3f}], "
              f"std=[{self.target_std.min():.3f}, {self.target_std.max():.3f}]")

    def forward(
        self,
        x: torch.Tensor,
        odor_ids: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply envelope mean/std scaling (preserves temporal structure).

        Args:
            x: Input signal [B, C, T]
            odor_ids: Odor indices [B]
            target: If provided, use target's envelope stats directly

        Returns:
            Signal with scaled envelope [B, C, T]
        """
        if not self.fitted and target is None:
            # Not fitted and no target provided - return input unchanged
            return x

        B, C, T = x.shape
        device = x.device
        original_dtype = x.dtype

        # Work in float32
        x_float = x.float()

        # Compute analytic signal
        x_flat = x_float.view(B * C, T)
        analytic = hilbert_torch(x_flat)
        envelope = analytic.abs()  # [B*C, T]
        phase = analytic.angle()  # [B*C, T]

        # Scale envelope for each sample
        corrected_envelope = torch.zeros_like(envelope)

        for b in range(B):
            odor = odor_ids[b].item()
            start_idx = b * C
            end_idx = (b + 1) * C

            pred_env = envelope[start_idx:end_idx]  # [C, T]

            # Compute prediction envelope stats
            pred_mean = pred_env.mean()
            pred_std = pred_env.std().clamp(min=1e-6)

            if target is not None:
                # Use target's envelope stats directly
                target_flat = target[b].float()  # [C, T]
                target_analytic = hilbert_torch(target_flat)
                target_env = target_analytic.abs()
                tgt_mean = target_env.mean()
                tgt_std = target_env.std().clamp(min=1e-6)
            else:
                # Use stored stats for this odor
                tgt_mean = self.target_mean[odor]
                tgt_std = self.target_std[odor]

            # Linear scaling: (x - mean_x) * (std_tgt / std_x) + mean_tgt
            # This preserves temporal structure while matching mean/std
            scale = tgt_std / pred_std
            corrected_envelope[start_idx:end_idx] = (pred_env - pred_mean) * scale + tgt_mean

        # Reconstruct signal with corrected envelope and original phase
        # The original signal is x = Re(analytic) = envelope * cos(phase)
        corrected_signal = corrected_envelope * torch.cos(phase)

        # Reshape back
        corrected_signal = corrected_signal.view(B, C, T)

        return corrected_signal.to(original_dtype)

    def extra_repr(self) -> str:
        if self.fitted:
            return (
                f"n_odors={self.n_odors}, fitted=True, "
                f"mean_range=[{self.target_mean.min():.3f}, {self.target_mean.max():.3f}], "
                f"std_range=[{self.target_std.min():.3f}, {self.target_std.max():.3f}]"
            )
        return f"n_odors={self.n_odors}, fitted=False"


# =============================================================================
# Spectro-Temporal Auto-Conditioning Encoder
# =============================================================================

class SpectroTemporalEncoder(nn.Module):
    """Lightweight encoder for unsupervised 'reaction type' identification.

    This encoder captures the input signal's spectral-temporal fingerprint
    to enable dynamic, signal-adaptive conditioning. Instead of relying
    solely on static odor labels, the model can identify specific 'reaction
    types' (e.g., beta events vs gamma bursts) directly from input dynamics.

    Architecture:
        Two parallel branches:
        1. Spectral Branch: FFT/PSD-based frequency content (shift-invariant)
        2. Temporal Branch: Multi-scale dilated convolutions for transient shapes

        The outputs are concatenated to produce a latent 'style' embedding
        that can be fused with the odor embedding for FiLM conditioning.

    Args:
        in_channels: Number of input channels (default: 32)
        emb_dim: Output embedding dimension (default: 128)
        n_freq_bands: Number of frequency bands in spectral branch (default: 10)
        temporal_hidden: Hidden dimension for temporal branch (default: 64)
        temporal_dilations: Dilation rates for multi-scale temporal conv
        temporal_kernel: Kernel size for temporal convolutions (default: 7)
        sample_rate: Sampling rate in Hz (default: 1000)
        max_freq: Maximum frequency to consider (default: 100 Hz)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        in_channels: int = 32,
        emb_dim: int = 128,
        n_freq_bands: int = 10,
        temporal_hidden: int = 64,
        temporal_dilations: Tuple[int, ...] = (1, 4, 16),
        temporal_kernel: int = 7,
        sample_rate: float = SAMPLING_RATE_HZ,
        max_freq: float = MAX_FREQ_HZ,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.n_freq_bands = n_freq_bands
        self.sample_rate = sample_rate
        self.max_freq = max_freq
        self.temporal_dilations = temporal_dilations

        # =====================================================================
        # Spectral Branch: FFT/PSD-based frequency content analysis
        # =====================================================================
        # Input: log-power in each frequency band → [B, in_channels, n_freq_bands]
        # Output: spectral embedding → [B, emb_dim // 2]
        spectral_input_dim = in_channels * n_freq_bands
        self.spectral_encoder = nn.Sequential(
            nn.Linear(spectral_input_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.LayerNorm(emb_dim // 2),
        )

        # =====================================================================
        # Temporal Branch: Multi-scale dilated convolutions for transients
        # =====================================================================
        # Each dilation captures different temporal scales:
        # - d=1:  fine-grained transients (gamma events ~10-30ms)
        # - d=4:  medium-scale patterns (beta bursts ~30-80ms)
        # - d=16: slow modulations (theta waves ~100-250ms)
        n_scales = len(temporal_dilations)

        # Parallel dilated convolution branches
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels, temporal_hidden,
                kernel_size=temporal_kernel,
                dilation=d,
                padding=(temporal_kernel - 1) * d // 2,
            )
            for d in temporal_dilations
        ])

        # Combine multi-scale features
        self.temporal_fusion = nn.Sequential(
            nn.Conv1d(temporal_hidden * n_scales, temporal_hidden, kernel_size=1),
            nn.BatchNorm1d(temporal_hidden),
            nn.GELU(),
        )

        # Temporal pooling + projection to embedding
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_hidden, emb_dim // 2),
            nn.LayerNorm(emb_dim // 2),
        )

        # =====================================================================
        # Output projection (optional gating for fusion strength)
        # =====================================================================
        self.output_gate = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Linear(emb_dim, emb_dim)

        # Cache for frequency band masks
        self._freq_masks: Optional[torch.Tensor] = None
        self._cached_length: int = 0

    def _build_freq_masks(self, T: int, device: torch.device) -> torch.Tensor:
        """Build frequency band masks for spectral analysis.

        Returns:
            Tensor of shape [n_freq_bands, n_fft // 2 + 1]
        """
        n_fft = T // 2 + 1
        if self._freq_masks is not None and self._cached_length == n_fft:
            return self._freq_masks.to(device)

        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sample_rate)

        # Create log-spaced frequency bands from 1 Hz to max_freq
        band_edges = torch.logspace(
            math.log10(1.0), math.log10(self.max_freq),
            self.n_freq_bands + 1,
        )

        masks = []
        for i in range(self.n_freq_bands):
            low = band_edges[i]
            high = band_edges[i + 1]
            # Soft band edges for smooth gradients
            mask = torch.sigmoid((freqs - low) * 2) * torch.sigmoid((high - freqs) * 2)
            # Normalize mask to sum to 1
            mask = mask / (mask.sum() + 1e-8)
            masks.append(mask)

        self._freq_masks = torch.stack(masks, dim=0)  # [n_bands, n_fft]
        self._cached_length = n_fft
        return self._freq_masks.to(device)

    def _spectral_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spectral branch embedding.

        Args:
            x: Input signal [B, C, T]

        Returns:
            Spectral embedding [B, emb_dim // 2]
        """
        B, C, T = x.shape
        device = x.device

        # Get frequency band masks
        masks = self._build_freq_masks(T, device)  # [n_bands, n_fft]

        # Compute FFT and power spectral density (requires float32)
        # FFT doesn't support BFloat16 (used by FSDP mixed precision)
        x_fft = torch.fft.rfft(x.float(), dim=-1)  # [B, C, n_fft]
        power = x_fft.abs().square()  # [B, C, n_fft]

        # Extract band powers: [B, C, n_bands]
        band_power = torch.einsum('bcf,kf->bck', power, masks)

        # Log-scale for better dynamic range
        band_power_log = torch.log(band_power + 1e-10)  # [B, C, n_bands]

        # Flatten and encode - convert back to original dtype for linear layers
        band_flat = band_power_log.reshape(B, -1)  # [B, C * n_bands]
        return self.spectral_encoder(band_flat.to(self._param_dtype))  # [B, emb_dim // 2]

    def _temporal_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute temporal branch embedding.

        Args:
            x: Input signal [B, C, T]

        Returns:
            Temporal embedding [B, emb_dim // 2]
        """
        # Multi-scale dilated convolutions
        temporal_features = []
        for conv in self.temporal_convs:
            feat = F.gelu(conv(x))  # [B, temporal_hidden, T]
            temporal_features.append(feat)

        # Concatenate scales: [B, temporal_hidden * n_scales, T]
        multi_scale = torch.cat(temporal_features, dim=1)

        # Fuse multi-scale features
        fused = self.temporal_fusion(multi_scale)  # [B, temporal_hidden, T]

        # Global average pooling
        pooled = self.temporal_pool(fused).squeeze(-1)  # [B, temporal_hidden]

        # Project to embedding space
        return self.temporal_proj(pooled)  # [B, emb_dim // 2]

    @property
    def _param_dtype(self) -> torch.dtype:
        """Get the dtype of model parameters (for mixed precision compatibility)."""
        return next(self.parameters()).dtype

    @torch.amp.autocast('cuda', enabled=False)
    def forward(
        self,
        x: torch.Tensor,
        odor_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute spectro-temporal style embedding.

        Args:
            x: Input signal [B, C, T]
            odor_emb: Optional odor embedding to fuse with [B, emb_dim]

        Returns:
            Style embedding [B, emb_dim], optionally fused with odor embedding
        """
        original_dtype = x.dtype
        param_dtype = self._param_dtype

        # IMPORTANT: Clone input to avoid autograd inplace errors
        # When x is already float32 and param_dtype is float32, .float() and .to()
        # return the same tensor (not copies). This can cause gradient computation
        # issues when the same input tensor is used in both the model and cond_encoder.
        x = x.clone()

        # FFT requires float32, but we need to match param dtype for linear layers
        x_float = x.float()

        # Spectral branch: FFT in float32, then convert for encoder
        spectral_emb = self._spectral_forward(x_float)  # [B, emb_dim // 2]

        # Temporal branch: run in param dtype for conv layers
        x_param = x.to(param_dtype)
        temporal_emb = self._temporal_forward(x_param)  # [B, emb_dim // 2]

        # Ensure both branches are in same dtype before concatenation
        spectral_emb = spectral_emb.to(param_dtype)
        temporal_emb = temporal_emb.to(param_dtype)

        # Concatenate branches
        style_emb = torch.cat([spectral_emb, temporal_emb], dim=-1)  # [B, emb_dim]

        # Apply gated output projection
        gate = self.output_gate(style_emb)
        style_emb = self.output_proj(style_emb) * gate

        # Fuse with odor embedding if provided
        if odor_emb is not None:
            # Additive fusion with learned style embedding
            style_emb = style_emb + odor_emb.to(param_dtype)

        return style_emb.to(original_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, emb_dim={self.emb_dim}, "
            f"n_freq_bands={self.n_freq_bands}, dilations={self.temporal_dilations}"
        )


# =============================================================================
# Additional Auto-Conditioning Encoders
# =============================================================================

# Frequency bands for FreqDisentangledEncoder
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 100),
}


class CPCEncoder(nn.Module):
    """Contrastive Predictive Coding encoder.

    Learns predictive representations by maximizing mutual information
    between context and future timesteps.
    """

    def __init__(self, in_channels: int, embed_dim: int = 128, n_steps: int = 4):
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, T] input signal

        Returns:
            context_mean: [B, embed_dim] mean context embedding (for conditioning)
            z_seq: [B, T', embed_dim] latent sequence (for CPC loss)
            context_seq: [B, T', embed_dim] full context sequence (for CPC loss)
        """
        # Encode
        z = self.encoder(x)  # [B, embed_dim, T']
        z = z.transpose(1, 2)  # [B, T', embed_dim]

        # Context aggregation
        context, _ = self.gru(z)  # [B, T', embed_dim]

        # Use mean context as conditioning
        context_mean = context.mean(dim=1)  # [B, embed_dim]

        return context_mean, z, context

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

    def __init__(self, in_channels: int, embed_dim: int = 128, n_codes: int = 512):
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

        # Codebook - better initialization for stable training
        self.codebook = nn.Embedding(n_codes, embed_dim)
        # Initialize with unit normal (std=1) scaled by 1/sqrt(embed_dim)
        nn.init.normal_(self.codebook.weight, mean=0, std=1.0 / math.sqrt(embed_dim))

        # Decoder (for reconstruction loss)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, in_channels, kernel_size=4, stride=2, padding=1),
        )

        # Output normalization for stable conditioning (like other encoders)
        self.output_norm = nn.LayerNorm(embed_dim)

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

        # Mean pooled embedding for conditioning with normalization
        conditioning = z_q_st.mean(dim=-1)  # [B, D]
        conditioning = self.output_norm(conditioning)  # Normalize for stable conditioning

        return conditioning, {
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
            "recon_loss": recon_loss,
        }


class FreqDisentangledEncoder(nn.Module):
    """Frequency-disentangled encoder.

    Learns separate latent representations for each frequency band.
    Uses smooth Gaussian bandpass filters and LayerNorm for stable embeddings.

    Includes auxiliary reconstruction loss to ensure meaningful embeddings.
    """

    def __init__(self, in_channels: int, embed_dim: int = 128, fs: float = 1000.0):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.fs = fs

        # Per-band encoders with LayerNorm for stable embeddings
        self.band_encoders = nn.ModuleDict()
        self.band_norms = nn.ModuleDict()
        self.band_decoders = nn.ModuleDict()  # For reconstruction loss
        band_dim = embed_dim // len(FREQ_BANDS)
        self.band_dim = band_dim

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
            # LayerNorm per band for stable scale
            self.band_norms[band_name] = nn.LayerNorm(band_dim)
            # Decoder to reconstruct band log-power (for auxiliary loss)
            self.band_decoders[band_name] = nn.Linear(band_dim, in_channels)

        # Final projection with normalization
        self.proj = nn.Linear(band_dim * len(FREQ_BANDS), embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)

    def bandpass_filter(self, x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """Apply smooth bandpass filter using FFT with Gaussian rolloff.

        Uses a smooth Gaussian rolloff instead of hard cutoff to avoid
        ringing artifacts (Gibbs phenomenon) and preserve signal amplitude.
        """
        # FFT - cast to float32 for BFloat16 compatibility (FSDP mixed precision)
        orig_dtype = x.dtype
        X = torch.fft.rfft(x.float(), dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.fs).to(x.device)

        # Compute band center and width for Gaussian rolloff
        center = (low + high) / 2.0
        bandwidth = high - low
        # Use 1/4 bandwidth as sigma for smooth rolloff (95% energy in band)
        sigma = bandwidth / 4.0 + 1e-6  # Avoid division by zero

        # Smooth Gaussian bandpass mask
        mask = torch.exp(-0.5 * ((freqs - center) / sigma) ** 2)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, F]

        # Apply mask
        X_filtered = X * mask

        # Inverse FFT and cast back to original dtype
        return torch.fft.irfft(X_filtered, n=x.shape[-1], dim=-1).to(orig_dtype)

    def _compute_band_power(self, x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """Compute log power in frequency band for each channel.

        Args:
            x: [B, C, T] input signal
            low: Low frequency bound
            high: High frequency bound

        Returns:
            [B, C] log power per channel in the band
        """
        # FFT doesn't support BFloat16 (used by FSDP mixed precision)
        X = torch.fft.rfft(x.float(), dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.fs).to(x.device)

        # Band mask
        mask = ((freqs >= low) & (freqs <= high)).float()

        # Compute power in band
        power = (X.abs().square() * mask).sum(dim=-1)  # [B, C]

        # Log scale for better dynamic range
        return torch.log(power + 1e-10)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, C, T] input signal

        Returns:
            embedding: [B, embed_dim] frequency-disentangled embedding
            band_info: dict with band embeddings and targets for auxiliary loss
        """
        band_embeddings = []
        band_info = {"embeddings": {}, "targets": {}}

        for band_name, (low, high) in FREQ_BANDS.items():
            # Filter signal to band
            x_band = self.bandpass_filter(x, low, high)

            # Encode band (before normalization, for reconstruction)
            z_band_raw = self.band_encoders[band_name](x_band)
            # Normalize for stable conditioning
            z_band = self.band_norms[band_name](z_band_raw)
            band_embeddings.append(z_band)

            # Store for auxiliary loss
            band_info["embeddings"][band_name] = z_band_raw  # Use pre-norm for recon
            band_info["targets"][band_name] = self._compute_band_power(x, low, high)

        # Concatenate all band embeddings
        z_concat = torch.cat(band_embeddings, dim=-1)

        # Project to final embedding with normalization
        z_out = self.proj(z_concat)
        return self.output_norm(z_out), band_info

    def recon_loss(self, band_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute auxiliary reconstruction loss.

        Each band embedding should be able to predict the log power
        in that frequency band. This ensures embeddings are meaningful.

        Args:
            band_info: dict from forward() with embeddings and targets

        Returns:
            Scalar reconstruction loss
        """
        total_loss = 0.0
        n_bands = 0

        for band_name in FREQ_BANDS.keys():
            z_band = band_info["embeddings"][band_name]  # [B, band_dim]
            target = band_info["targets"][band_name]  # [B, C]

            # Predict band power from embedding
            pred = self.band_decoders[band_name](z_band)  # [B, C]

            # MSE loss
            total_loss = total_loss + F.mse_loss(pred, target)
            n_bands += 1

        return total_loss / n_bands


class CycleConsistentEncoder(nn.Module):
    """Cycle-consistent latent encoder.

    Learns latent that can reconstruct both input and output,
    enforcing cycle consistency.
    """

    def __init__(self, in_channels: int, out_channels: int, embed_dim: int = 128):
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
# Conditional Distribution Correction Block
# =============================================================================

def hilbert_torch(x: torch.Tensor) -> torch.Tensor:
    """Differentiable Hilbert transform using FFT.

    Computes the analytic signal: x_analytic = x + j * H(x)
    where H(x) is the Hilbert transform of x.

    Args:
        x: Real-valued input signal [..., T]

    Returns:
        Complex-valued analytic signal [..., T]
    """
    N = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)

    # Build Hilbert mask
    h = torch.zeros(N, device=x.device, dtype=x.dtype)
    h[0] = 1
    if N % 2 == 0:
        h[1:N // 2] = 2
        h[N // 2] = 1
    else:
        h[1:(N + 1) // 2] = 2

    return torch.fft.ifft(X * h, dim=-1)


# =============================================================================
# Session Statistics Encoder (Literature-Based Approach)
# =============================================================================

class SessionStatisticsEncoder(nn.Module):
    """Encode session statistics into a conditioning vector for FiLM modulation.

    Computes per-channel statistics from raw input and encodes them into an embedding.
    Generalizes to unseen sessions by learning from signal characteristics.
    """

    def __init__(
        self,
        n_channels: int,
        emb_dim: int = 32,
        use_spectral_features: bool = False,
        hidden_dim: int = 64,
    ):
        """Initialize SessionStatisticsEncoder.

        Args:
            n_channels: Number of input channels (e.g., 32 for OB electrodes)
            emb_dim: Output embedding dimension
            use_spectral_features: If True, also include power in frequency bands
                                   (requires more computation but captures more info)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.n_channels = n_channels
        self.emb_dim = emb_dim
        self.use_spectral_features = use_spectral_features

        # Basic statistics: mean + std per channel = 2 * n_channels
        # Optional: add skewness, kurtosis for 4 * n_channels
        input_dim = n_channels * 2

        if use_spectral_features:
            # Add power in 5 frequency bands per channel
            # (delta, theta, alpha, beta, gamma)
            input_dim += n_channels * 5

        # MLP to encode statistics into embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

        # Initialize to produce near-zero embeddings initially
        # This ensures the model works even without session conditioning at start
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute session embedding from input statistics.

        Args:
            x: Raw input signal [B, C, T] BEFORE normalization
               This is important - we need the original statistics

        Returns:
            Session embedding [B, emb_dim]
        """
        # Compute per-channel statistics
        mean = x.mean(dim=-1)  # [B, C]
        std = x.std(dim=-1)    # [B, C]

        # Concatenate statistics
        stats = torch.cat([mean, std], dim=-1)  # [B, 2*C]

        if self.use_spectral_features:
            # Compute power in frequency bands (simplified version)
            # For full implementation, use scipy.signal.welch or torch equivalent
            # Here we use a simple FFT-based approach
            power_features = self._compute_band_power(x)  # [B, 5*C]
            stats = torch.cat([stats, power_features], dim=-1)

        # Encode to embedding
        return self.encoder(stats)

    def _compute_band_power(self, x: torch.Tensor) -> torch.Tensor:
        """Compute power in frequency bands.

        Simplified implementation using FFT.
        Assumes 1000 Hz sampling rate.

        Args:
            x: Input signal [B, C, T]

        Returns:
            Band power features [B, 5*C]
        """
        B, C, T = x.shape

        # FFT doesn't support BFloat16 (used by FSDP mixed precision)
        # Cast to float32 for FFT computation
        x_float = x.float()
        fft = torch.fft.rfft(x_float, dim=-1)
        power = (fft.real ** 2 + fft.imag ** 2)  # [B, C, T//2+1]

        # Frequency resolution
        freq_resolution = 1000.0 / T  # Assumes 1000 Hz sampling
        n_freqs = power.shape[-1]

        # Frequency bands (in Hz)
        bands = [
            (1, 4),    # Delta
            (4, 8),    # Theta
            (8, 13),   # Alpha
            (13, 30),  # Beta
            (30, 100), # Gamma
        ]

        band_powers = []
        for low, high in bands:
            low_idx = max(1, int(low / freq_resolution))
            high_idx = min(n_freqs - 1, int(high / freq_resolution))
            if high_idx > low_idx:
                band_power = power[:, :, low_idx:high_idx].mean(dim=-1)  # [B, C]
            else:
                band_power = torch.zeros(B, C, device=x.device, dtype=power.dtype)
            band_powers.append(band_power)

        # Cast back to original dtype (may be BFloat16 with FSDP)
        return torch.cat(band_powers, dim=-1).to(x.dtype)  # [B, 5*C]


class SessionAdaptiveScaling(nn.Module):
    """Session-aware output scaling using FiLM-style conditioning.

    Predicts scale (gamma) and bias (beta) from session statistics:
        output = input * gamma + beta
    """

    def __init__(
        self,
        n_channels: int,
        n_input_channels: int = 32,
        hidden_dim: int = 64,
    ):
        """Initialize SessionAdaptiveScaling.

        Args:
            n_channels: Number of output channels to scale
            n_input_channels: Number of input channels (for computing stats)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_input_channels = n_input_channels

        # Input: mean + std per input channel = 2 * n_input_channels
        input_dim = n_input_channels * 2

        # Predict scale (gamma) and bias (beta) per output channel
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_channels * 2),  # gamma and beta
        )

        # Initialize to predict identity transform (scale=1, bias=0)
        nn.init.zeros_(self.predictor[-1].weight)
        # Initialize bias to [1, 1, ..., 1, 0, 0, ..., 0] for identity
        with torch.no_grad():
            self.predictor[-1].bias[:n_channels] = 1.0  # scale = 1
            self.predictor[-1].bias[n_channels:] = 0.0  # bias = 0

    def forward(
        self,
        x: torch.Tensor,
        raw_input: torch.Tensor,
    ) -> torch.Tensor:
        """Apply session-adaptive scaling.

        Args:
            x: Output to scale [B, C, T]
            raw_input: Raw input signal [B, C_in, T] for computing statistics

        Returns:
            Scaled output [B, C, T]
        """
        # Compute session statistics from raw input
        mean = raw_input.mean(dim=-1)  # [B, C_in]
        std = raw_input.std(dim=-1)    # [B, C_in]
        stats = torch.cat([mean, std], dim=-1)  # [B, 2*C_in]

        # Predict scale and bias
        params = self.predictor(stats)  # [B, 2*C]
        gamma = params[:, :self.n_channels].unsqueeze(-1)  # [B, C, 1]
        beta = params[:, self.n_channels:].unsqueeze(-1)   # [B, C, 1]

        # Apply adaptive scaling
        return x * gamma + beta


class CovarianceExpansionAugmentation(nn.Module):
    """Generate synthetic sessions via covariance-based transformations.

    During training, this augmentation creates "virtual sessions" by:
    1. Computing per-channel statistics from a batch
    2. Applying random affine transformations that preserve covariance structure
    3. Adding controlled noise to simulate session variability

    This increases session diversity during training, helping the model
    generalize better to unseen sessions.

    Based on:
    - Mixup / CutMix style augmentations
    - Domain randomization techniques
    - Covariance-preserving transformations
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        shift_range: Tuple[float, float] = (-0.2, 0.2),
        noise_std: float = 0.05,
        prob: float = 0.5,
    ):
        """Initialize CovarianceExpansionAugmentation.

        Args:
            scale_range: Range for random per-channel scaling (min, max)
            shift_range: Range for random per-channel shift (min, max)
            noise_std: Standard deviation of additive noise
            prob: Probability of applying augmentation
        """
        super().__init__()
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.noise_std = noise_std
        self.prob = prob

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply covariance expansion augmentation.

        Args:
            x: Input signal [B, C, T]
            y: Target signal [B, C, T]

        Returns:
            Augmented (x, y) tuple
        """
        if not self.training or torch.rand(1).item() > self.prob:
            return x, y

        B, C, T = x.shape
        device = x.device

        # Generate random per-channel scale and shift
        scale = torch.empty(B, C, 1, device=device).uniform_(*self.scale_range)
        shift = torch.empty(B, C, 1, device=device).uniform_(*self.shift_range)

        # Apply transformation to both input and target (they're paired)
        x_aug = x * scale + shift * x.std(dim=-1, keepdim=True)
        y_aug = y * scale + shift * y.std(dim=-1, keepdim=True)

        # Add small noise to input only (target should remain clean)
        if self.noise_std > 0:
            noise = torch.randn_like(x_aug) * self.noise_std * x_aug.std(dim=-1, keepdim=True)
            x_aug = x_aug + noise

        return x_aug, y_aug

    def extra_repr(self) -> str:
        return f"scale_range={self.scale_range}, shift_range={self.shift_range}, noise_std={self.noise_std}, prob={self.prob}"


# =============================================================================
# Main Model: CondUNet1D
# =============================================================================

class CondUNet1D(nn.Module):
    """Conditional U-Net for neural signal translation with FiLM modulation.

    Supports FiLM conditioning, self-attention in bottleneck, multi-scale skip connections,
    modern convolutions (multi-scale dilated + SE), and configurable depth.
    """
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        base: int = BASE_CHANNELS,
        n_odors: int = NUM_ODORS,
        emb_dim: int = 128,
        dropout: float = 0.0,
        use_attention: bool = True,
        attention_type: str = "basic",
        norm_type: str = "batch",
        cond_mode: str = "cross_attn_gated",
        n_downsample: int = 2,
        conv_type: str = "standard",
        use_se: bool = True,
        conv_kernel_size: int = 7,
        dilations: Tuple[int, ...] = (1, 4, 16, 32),
        use_adaptive_scaling: bool = False,
        use_session_stats: bool = False,
        session_emb_dim: int = 32,
        session_use_spectral: bool = False,
        use_session_embedding: bool = False,
        n_sessions: int = 0,
        # NEW: Ablation-configurable parameters
        activation: str = "relu",  # Activation function: relu, leaky_relu, gelu, silu, mish
        n_heads: int = 4,  # Number of attention heads (for cross-frequency attention)
        skip_type: str = "add",  # Skip connection type: add, concat (future: attention, dense)
    ):
        super().__init__()
        self.cond_mode = cond_mode
        self.emb_dim = emb_dim
        self.attention_type = attention_type
        self.conv_type = conv_type
        self.n_downsample = n_downsample
        self.use_session_stats = use_session_stats
        self.use_session_embedding = use_session_embedding
        self.session_emb_dim = session_emb_dim
        self.n_sessions = n_sessions
        # NEW: Store ablation parameters
        self.activation_name = activation
        self.n_heads = n_heads
        self.skip_type = skip_type

        if cond_mode != "none":
            self.embed = nn.Embedding(n_odors, emb_dim)
        else:
            self.embed = None

        # Session conditioning using statistics-based encoder
        # This approach:
        # 1. Computes statistics (mean, std per channel) from raw input
        # 2. Encodes statistics into a conditioning vector
        # 3. Automatically generalizes to new/unseen sessions
        if use_session_stats:
            self.session_stats_encoder = SessionStatisticsEncoder(
                n_channels=in_channels,
                emb_dim=session_emb_dim,
                use_spectral_features=session_use_spectral,
            )
            # Project combined embedding (session stats + condition) to emb_dim
            self.session_proj = nn.Sequential(
                nn.Linear(session_emb_dim + emb_dim, emb_dim),
                get_activation(activation),
                nn.Linear(emb_dim, emb_dim),
            )
        else:
            self.session_stats_encoder = None
            self.session_proj = None

        # Learnable session embedding (lookup table approach)
        # This learns a unique embedding per session ID and combines it with FiLM conditioning.
        # NEW: Uses statistics-based matching for unseen sessions (soft attention over embeddings)
        if use_session_embedding:
            if n_sessions <= 0:
                raise ValueError(
                    f"use_session_embedding=True requires n_sessions > 0, got {n_sessions}"
                )
            self.session_embed = nn.Embedding(n_sessions, session_emb_dim)
            # Project combined embedding (session embed + condition) to emb_dim
            # Separate projection from session_proj to allow both to be enabled
            self.session_embed_proj = nn.Sequential(
                nn.Linear(session_emb_dim + emb_dim, emb_dim),
                get_activation(activation),
                nn.Linear(emb_dim, emb_dim),
            )
            # NEW: Statistics-based session matcher for generalization to unseen sessions
            # Computes input statistics and uses soft attention over all session embeddings
            # This allows the model to find the "nearest" session based on signal characteristics
            stat_dim = 64  # Dimension of statistics encoding
            self.session_stat_encoder = nn.Sequential(
                nn.Linear(in_channels * 4, stat_dim),  # 4 stats: mean, std, min, max per channel
                get_activation(activation),
                nn.Linear(stat_dim, stat_dim),
            )
            # Attention over session embeddings based on statistics
            self.session_stat_to_query = nn.Linear(stat_dim, session_emb_dim)
            self.session_embed_keys = nn.Linear(session_emb_dim, session_emb_dim)
            # Track which sessions are "known" (seen during training)
            self.register_buffer('known_sessions', torch.zeros(n_sessions, dtype=torch.bool))
            # Debug counter for verification
            self._session_embed_use_count = 0
        else:
            self.session_embed = None
            self.session_embed_proj = None

        # Channel progression: base -> base*2 -> base*4 -> ... up to base*8 max
        # For n_downsample=2: [base, base*2, base*4]
        # For n_downsample=4: [base, base*2, base*4, base*8, base*8]
        channels = [base]
        for i in range(n_downsample):
            channels.append(min(base * (2 ** (i + 1)), base * 8))
        
        # Build encoder and decoder as ModuleList for variable depth
        if conv_type == "modern":
            conv_kwargs = dict(
                use_se=use_se,
                kernel_size=conv_kernel_size,
                dilations=dilations,
            )
            self.inc = ModernConvBlock(in_channels, channels[0], emb_dim, dropout=dropout, norm_type=norm_type, cond_mode=cond_mode, **conv_kwargs)
            
            self.encoders = nn.ModuleList()
            for i in range(n_downsample):
                self.encoders.append(
                    ModernConvBlock(channels[i], channels[i+1], emb_dim, downsample=True, dropout=dropout, norm_type=norm_type, cond_mode=cond_mode, **conv_kwargs)
                )
            
            self.decoders = nn.ModuleList()
            for i in range(n_downsample):
                # Decoder goes in reverse: channels[n] -> channels[n-1]
                dec_in = channels[n_downsample - i]
                skip_in = channels[n_downsample - i - 1]
                dec_out = channels[n_downsample - i - 1]
                self.decoders.append(
                    ModernUpBlock(dec_in, skip_in, dec_out, emb_dim, dropout=dropout, norm_type=norm_type, cond_mode=cond_mode, skip_type=skip_type, **conv_kwargs)
                )
        else:
            # Standard: original Conv1d(kernel_size=3)
            self.inc = ConvBlock(in_channels, channels[0], emb_dim, dropout=dropout, norm_type=norm_type, cond_mode=cond_mode)

            self.encoders = nn.ModuleList()
            for i in range(n_downsample):
                self.encoders.append(
                    ConvBlock(channels[i], channels[i+1], emb_dim, downsample=True, dropout=dropout, norm_type=norm_type, cond_mode=cond_mode)
                )

            self.decoders = nn.ModuleList()
            for i in range(n_downsample):
                dec_in = channels[n_downsample - i]
                skip_in = channels[n_downsample - i - 1]
                dec_out = channels[n_downsample - i - 1]
                self.decoders.append(
                    UpBlock(dec_in, skip_in, dec_out, emb_dim, dropout=dropout, norm_type=norm_type, cond_mode=cond_mode, skip_type=skip_type)
                )

        # Build bottleneck with configurable attention
        bottleneck_ch = channels[n_downsample]
        mid_layers = [
            nn.Conv1d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1),
            get_activation(activation),
        ]
        if use_attention and attention_type != "none":
            mid_layers.append(self._build_attention(bottleneck_ch, attention_type))
        mid_layers.extend([
            nn.Conv1d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1),
            get_activation(activation),
        ])
        self.mid = nn.Sequential(*mid_layers)

        self.outc = nn.Conv1d(channels[0], out_channels, kernel_size=3, padding=1)

        # Residual path: learn PCx = OB + delta (preserves amplitude)
        # This 1x1 conv allows channel mapping if in_channels != out_channels
        self.use_residual = (in_channels == out_channels)
        if not self.use_residual:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # Initialize output layer to produce small values initially
        # so residual (input) dominates at start
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

        # Session-adaptive output scaling (FiLM-style)
        # Predicts scale/bias from input statistics - adapts to each session
        self.use_adaptive_scaling = use_adaptive_scaling
        if use_adaptive_scaling:
            self.adaptive_scaler = SessionAdaptiveScaling(
                n_channels=out_channels,
                n_input_channels=in_channels,
            )

        # Store in_channels for adaptive scaling
        self.in_channels = in_channels

        # Gradient checkpointing: trade compute for memory
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable/disable gradient checkpointing for memory efficiency.

        When enabled, encoder/decoder blocks are checkpointed to reduce memory
        at the cost of ~30% more compute (recomputes activations during backward).
        """
        self.gradient_checkpointing = enable

    def enable_adabn(self):
        """Enable Adaptive Batch Normalization (AdaBN) for domain adaptation.

        AdaBN is a simple, parameter-free domain adaptation technique from:
        "Revisiting Batch Normalization for Practical Domain Adaptation" (Li et al., 2016)

        When enabled, all BatchNorm layers compute statistics from the current batch
        instead of using running statistics. This allows the model to adapt to new
        sessions/domains without any additional training.

        Call this method before inference on a new session/domain.
        Call disable_adabn() to restore normal behavior.

        Literature:
        - AdaBN achieves comparable results to more complex domain adaptation methods
        - Works because BN statistics capture domain-specific characteristics
        - Particularly effective for neural signal translation where sessions have
          different baseline statistics due to electrode placement, impedance, etc.
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                # Set to training mode so it computes batch statistics
                module.training = True
                # Reset running stats so they don't interfere
                module.reset_running_stats()

    def disable_adabn(self):
        """Disable Adaptive Batch Normalization, restore normal inference behavior.

        This restores BatchNorm layers to use their learned running statistics.
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.training = False

    def calibrate_bn_stats(self, dataloader, device: torch.device, max_batches: int = 50):
        """Calibrate BatchNorm running statistics on a new session/domain.

        Alternative to AdaBN: instead of using batch statistics at inference,
        this method runs a forward pass on calibration data to update the
        running statistics to the new domain.

        Args:
            dataloader: DataLoader for the new session/domain (no labels needed)
            device: Device to run on
            max_batches: Maximum number of batches to use for calibration

        Note: Model should be in eval mode after this, not AdaBN mode.
        """
        # Set BN layers to training mode to update running stats
        self.train()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                # Forward pass to update BN running stats
                _ = self(x)
        # Set back to eval mode
        self.eval()

    def _build_attention(self, channels: int, attention_type: str) -> nn.Module:
        """Build attention module based on type.

        Args:
            channels: Number of channels
            attention_type: Type of attention ("basic" or "cross_freq")

        Returns:
            Attention module
        """
        if attention_type == "basic":
            return SelfAttention1D(channels)

        if attention_type == "cross_freq":
            return CrossFrequencyCouplingAttention(channels, n_heads=self.n_heads)

        if attention_type == "cross_freq_v2":
            return CrossFrequencyCouplingAttentionV2(channels, n_heads=self.n_heads)

        raise ValueError(f"Unknown attention_type: {attention_type}. Available: none, basic, cross_freq, cross_freq_v2")

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
        cond_emb: Optional[torch.Tensor] = None,
        session_ids: Optional[torch.Tensor] = None,  # Required if use_session_embedding=True
        return_bottleneck: bool = False,
        return_bottleneck_temporal: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            ob: Input signal [B, C, T]
            odor_ids: Odor class indices [B] for conditioning (uses internal embedding)
            cond_emb: External conditioning embedding [B, emb_dim] (bypasses internal embedding)
                      If provided, odor_ids is ignored.
            session_ids: Session indices [B] for learnable session embedding lookup.
                        Required if use_session_embedding=True.
                        Not needed for statistics-based conditioning (use_session_stats).
            return_bottleneck: If True, also return pooled bottleneck features [B, bottleneck_ch]
                              for label-based contrastive learning
            return_bottleneck_temporal: If True, return unpooled bottleneck features [B, bottleneck_ch, T']
                                        for temporal CEBRA (time-based contrastive learning)
                                        Takes precedence over return_bottleneck if both are True.

        Returns:
            If return_bottleneck=False and return_bottleneck_temporal=False: Predicted signal [B, C, T]
            If return_bottleneck=True: (predicted signal, pooled bottleneck features [B, bottleneck_ch])
            If return_bottleneck_temporal=True: (predicted signal, unpooled bottleneck features [B, bottleneck_ch, T'])
        """
        # =================================================================
        # SAVE RAW INPUT (for session-adaptive scaling)
        # =================================================================
        # We need the raw input statistics for adaptive output scaling
        raw_input = ob  # Save before normalization

        # =================================================================
        # SESSION STATISTICS ENCODING (must happen BEFORE normalization!)
        # =================================================================
        # Compute session-specific conditioning from RAW input statistics.
        # This captures session characteristics like:
        #   - Electrode impedance (affects amplitude)
        #   - Baseline activity levels
        #   - Recording condition variations
        # The model learns HOW to use these statistics for adaptation.
        # =================================================================
        session_emb = None
        if self.session_stats_encoder is not None:
            session_emb = self.session_stats_encoder(ob)  # [B, session_emb_dim]

        # =================================================================
        # INPUT NORMALIZATION - The key to session-agnostic generalization
        # =================================================================
        # Standard per-channel, per-sample z-score normalization
        ob = (ob - ob.mean(dim=-1, keepdim=True)) / (ob.std(dim=-1, keepdim=True) + 1e-8)

        # Use external embeddings if provided, otherwise use internal odor embedding
        if cond_emb is not None:
            emb = cond_emb
        elif self.embed is not None and odor_ids is not None:
            emb = self.embed(odor_ids)
        else:
            emb = None

        # Combine session statistics embedding with other conditioning if enabled
        if session_emb is not None:
            if emb is not None:
                # Concatenate session embedding with existing conditioning
                combined = torch.cat([session_emb, emb], dim=-1)  # [B, session_emb_dim + emb_dim]
                emb = self.session_proj(combined)  # [B, emb_dim]
            else:
                # No other conditioning - pad session embedding to emb_dim and project
                padding = torch.zeros(session_emb.size(0), self.emb_dim, device=session_emb.device)
                combined = torch.cat([session_emb, padding], dim=-1)
                emb = self.session_proj(combined)

        # =================================================================
        # LEARNABLE SESSION EMBEDDING with STATISTICS-BASED MATCHING
        # =================================================================
        # Training: Direct lookup (learns session-specific patterns)
        # Inference: Statistics-based soft attention (generalizes to unseen sessions)
        # =================================================================
        if self.session_embed is not None:
            # Compute input statistics for session matching (from raw_input, before normalization)
            # Stats: [mean, std, min, max] per channel -> [B, C*4]
            input_mean = raw_input.mean(dim=-1)  # [B, C]
            input_std = raw_input.std(dim=-1)    # [B, C]
            input_min = raw_input.min(dim=-1).values  # [B, C]
            input_max = raw_input.max(dim=-1).values  # [B, C]
            input_stats = torch.cat([input_mean, input_std, input_min, input_max], dim=-1)  # [B, C*4]

            # Encode statistics
            stat_encoding = self.session_stat_encoder(input_stats)  # [B, stat_dim]
            stat_query = self.session_stat_to_query(stat_encoding)  # [B, session_emb_dim]

            # Get all session embeddings and compute keys
            all_embeds = self.session_embed.weight  # [n_sessions, session_emb_dim]
            embed_keys = self.session_embed_keys(all_embeds)  # [n_sessions, session_emb_dim]

            # Compute attention: query @ keys^T -> softmax -> weighted sum
            # [B, session_emb_dim] @ [session_emb_dim, n_sessions] -> [B, n_sessions]
            attn_logits = torch.matmul(stat_query, embed_keys.T) / (self.session_emb_dim ** 0.5)
            attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, n_sessions]

            # Statistics-based session embedding (soft attention over all embeddings)
            stats_session_emb = torch.matmul(attn_weights, all_embeds)  # [B, session_emb_dim]

            if self.training and session_ids is not None:
                # TRAINING: Use direct lookup for known sessions (supervised learning)
                # Also mark these sessions as known
                self.known_sessions[session_ids] = True
                direct_session_emb = self.session_embed(session_ids)  # [B, session_emb_dim]

                # Blend: 70% direct (for supervised learning) + 30% stats-based (to train matcher)
                learned_session_emb = 0.7 * direct_session_emb + 0.3 * stats_session_emb
            else:
                # INFERENCE: Use statistics-based matching (generalizes to unseen sessions)
                learned_session_emb = stats_session_emb

            # Debug verification - print on first few uses
            if hasattr(self, '_session_embed_use_count'):
                self._session_embed_use_count += 1
                if self._session_embed_use_count <= 3:
                    mode = "TRAIN(blend)" if self.training else "EVAL(stats)"
                    top_sessions = attn_weights.argmax(dim=-1).tolist()[:8]
                    print(f"[SESSION EMBED VERIFY] Use #{self._session_embed_use_count} [{mode}]: "
                          f"matched_sessions={top_sessions}{'...' if len(attn_weights) > 8 else ''}, "
                          f"embed_norm={learned_session_emb.norm().item():.4f}")

            if emb is not None:
                # Concatenate learned session embedding with existing conditioning
                combined = torch.cat([learned_session_emb, emb], dim=-1)  # [B, session_emb_dim + emb_dim]
                emb = self.session_embed_proj(combined)  # [B, emb_dim]
            else:
                # No other conditioning - pad and project
                padding = torch.zeros(learned_session_emb.size(0), self.emb_dim, device=learned_session_emb.device)
                combined = torch.cat([learned_session_emb, padding], dim=-1)
                emb = self.session_embed_proj(combined)

        # Encoder path with skip connections
        # Use gradient checkpointing if enabled (saves memory, costs ~30% more compute)
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            skips = [checkpoint(self.inc, ob, emb, use_reentrant=False)]
            for encoder in self.encoders:
                skips.append(checkpoint(encoder, skips[-1], emb, use_reentrant=False))
        else:
            skips = [self.inc(ob, emb)]
            for encoder in self.encoders:
                skips.append(encoder(skips[-1], emb))

        # Bottleneck
        bottleneck = self.mid(skips[-1])

        # Save bottleneck features for contrastive learning (CEBRA-style)
        # For temporal CEBRA: keep time dimension (B, C, T') for time-based positive pairs
        # For label-based: pool to (B, C) for label-based positive pairs
        if return_bottleneck_temporal:
            # Unpooled: (B, C, T') - preserves temporal structure for true CEBRA
            bottleneck_features = bottleneck
        elif return_bottleneck:
            # Pooled: (B, C) - for label-supervised contrastive learning
            bottleneck_features = bottleneck.mean(dim=-1)
        else:
            bottleneck_features = None

        # Decoder path with skip connections (reverse order)
        x = bottleneck
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for i, decoder in enumerate(self.decoders):
                skip_idx = self.n_downsample - i - 1
                x = checkpoint(decoder, x, skips[skip_idx], emb, use_reentrant=False)
        else:
            for i, decoder in enumerate(self.decoders):
                skip_idx = self.n_downsample - i - 1
                x = decoder(x, skips[skip_idx], emb)

        delta = self.outc(x)

        # Residual connection: output = input + learned_delta
        # At initialization (delta≈0), output ≈ input, preserving amplitude
        if self.use_residual:
            out = ob + delta
        else:
            out = self.residual_conv(ob) + delta

        # Apply session-adaptive output scaling if enabled
        if self.use_adaptive_scaling:
            out = self.adaptive_scaler(out, raw_input)

        # Note: SpectralShift is now applied OUTSIDE the model in train.py
        if return_bottleneck or return_bottleneck_temporal:
            return out, bottleneck_features
        return out

    # =========================================================================
    # Hook Registration Methods for Recording/Analysis
    # =========================================================================

    def get_layer_names(self) -> List[str]:
        """Get names of all trackable layers for Grad-CAM and saliency.

        Returns:
            List of layer names in network order
        """
        names = ["inc"]
        names.extend([f"encoder_{i}" for i in range(self.n_downsample)])
        names.append("mid")
        names.extend([f"decoder_{i}" for i in range(self.n_downsample)])
        names.append("outc")
        return names

    def register_activation_hooks(
        self,
        callback: Callable[[str, torch.Tensor], None]
    ) -> List:
        """Register forward hooks for activation recording.

        Args:
            callback: Function(layer_name, activation) called on each forward

        Returns:
            List of hook handles (call .remove() to unregister)
        """
        hooks = []
        
        # inc
        hooks.append(self.inc.register_forward_hook(
            lambda m, i, o, n="inc": callback(n, o)
        ))
        
        # encoders
        for idx, enc in enumerate(self.encoders):
            name = f"encoder_{idx}"
            hooks.append(enc.register_forward_hook(
                lambda m, i, o, n=name: callback(n, o)
            ))
        
        # mid
        hooks.append(self.mid.register_forward_hook(
            lambda m, i, o, n="mid": callback(n, o)
        ))
        
        # decoders
        for idx, dec in enumerate(self.decoders):
            name = f"decoder_{idx}"
            hooks.append(dec.register_forward_hook(
                lambda m, i, o, n=name: callback(n, o)
            ))
        
        # outc
        hooks.append(self.outc.register_forward_hook(
            lambda m, i, o, n="outc": callback(n, o)
        ))
        
        return hooks

    def get_encoder_layers(self) -> List[nn.Module]:
        """Get encoder layer modules.

        Returns:
            List of encoder modules [inc, encoder_0, encoder_1, ...]
        """
        return [self.inc] + list(self.encoders)

    def get_decoder_layers(self) -> List[nn.Module]:
        """Get decoder layer modules.

        Returns:
            List of decoder modules [decoder_0, decoder_1, ..., outc]
        """
        return list(self.decoders) + [self.outc]

    def get_attention_module(self) -> Optional[nn.Module]:
        """Get cross-frequency coupling attention module if present.

        Returns:
            Attention module or None
        """
        for module in self.mid.modules():
            if hasattr(module, 'coupling_weights'):
                return module
        return None


# =============================================================================
# Session Matching for Inference
# =============================================================================

class SessionMatcher:
    """Automatic session matching for inference with session-conditioned models.

    During training, stores a "session signature" for each session - a statistical
    fingerprint computed from samples (mean/std per channel).

    At inference time with a new session:
    1. Take first N samples as warmup
    2. Compute signature from warmup samples
    3. Find nearest neighbor among known session signatures
    4. Use that session's embedding for all subsequent inference

    Example usage:
        # During training - register signatures
        matcher = SessionMatcher(n_channels=32)
        for session_id, data in training_data:
            matcher.register_session(session_id, data)
        matcher.save("session_signatures.pt")

        # At inference
        matcher = SessionMatcher.load("session_signatures.pt")
        matched_session_id = matcher.warmup(new_session_data[:100])
        # Use matched_session_id for all inference on this session
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_warmup_samples: int = 100,
        use_power_spectrum: bool = False,
        sample_rate: float = 1000.0,
    ):
        """Initialize SessionMatcher.

        Args:
            n_channels: Number of signal channels
            n_warmup_samples: Number of samples to use for warmup matching
            use_power_spectrum: If True, also include power spectrum features
            sample_rate: Sampling rate in Hz (for power spectrum bands)
        """
        self.n_channels = n_channels
        self.n_warmup_samples = n_warmup_samples
        self.use_power_spectrum = use_power_spectrum
        self.sample_rate = sample_rate

        # Store signatures: session_id -> signature vector
        self.signatures: Dict[int, np.ndarray] = {}
        self.session_names: Dict[int, str] = {}  # Optional: human-readable names

    def compute_signature(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Compute session signature from data samples.

        Args:
            data: Signal data [N, C, T] or [C, T] where N=samples, C=channels, T=time

        Returns:
            Signature vector (mean and std per channel, optionally power bands)
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        # Handle single sample vs batch
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # [1, C, T]

        # Compute per-channel statistics across all samples and time
        # data: [N, C, T]
        mean_per_channel = data.mean(axis=(0, 2))  # [C]
        std_per_channel = data.std(axis=(0, 2))    # [C]

        signature_parts = [mean_per_channel, std_per_channel]

        if self.use_power_spectrum:
            # Compute mean power in frequency bands (theta, beta, gamma)
            # This requires scipy, so keep it optional
            try:
                from scipy import signal as scipy_signal

                # Define frequency bands
                bands = {
                    'theta': (4, 8),
                    'alpha': (8, 13),
                    'beta': (13, 30),
                    'low_gamma': (30, 50),
                    'high_gamma': (50, 100),
                }

                band_powers = []
                for band_name, (low, high) in bands.items():
                    # Compute power spectral density
                    freqs, psd = scipy_signal.welch(
                        data.mean(axis=0),  # Average across samples: [C, T]
                        fs=self.sample_rate,
                        nperseg=min(256, data.shape[-1]),
                        axis=-1,
                    )
                    # Get power in band
                    band_mask = (freqs >= low) & (freqs <= high)
                    band_power = psd[:, band_mask].mean(axis=-1)  # [C]
                    band_powers.append(band_power)

                signature_parts.extend(band_powers)
            except ImportError:
                pass  # Skip power spectrum if scipy not available

        return np.concatenate(signature_parts)

    def register_session(
        self,
        session_id: int,
        data: Union[torch.Tensor, np.ndarray],
        session_name: Optional[str] = None,
    ):
        """Register a session's signature from training data.

        Args:
            session_id: Integer session ID (0-indexed)
            data: Sample data from this session [N, C, T]
            session_name: Optional human-readable session name
        """
        signature = self.compute_signature(data)
        self.signatures[session_id] = signature
        if session_name:
            self.session_names[session_id] = session_name

    def warmup(
        self,
        data: Union[torch.Tensor, np.ndarray],
        return_similarity: bool = False,
    ) -> Union[int, Tuple[int, float]]:
        """Match new session data to known sessions.

        Args:
            data: Warmup samples from new session [N, C, T] (uses first n_warmup_samples)
            return_similarity: If True, also return the similarity score

        Returns:
            Matched session ID (and optionally similarity score)
        """
        if len(self.signatures) == 0:
            raise ValueError("No sessions registered. Call register_session() first.")

        # Compute signature from warmup samples
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # Use only first n_warmup_samples
        data = data[:self.n_warmup_samples]
        query_sig = self.compute_signature(data)

        # Find nearest neighbor using cosine similarity
        best_session = -1
        best_similarity = -float('inf')

        for session_id, ref_sig in self.signatures.items():
            # Cosine similarity
            similarity = np.dot(query_sig, ref_sig) / (
                np.linalg.norm(query_sig) * np.linalg.norm(ref_sig) + 1e-8
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_session = session_id

        if return_similarity:
            return best_session, float(best_similarity)
        return best_session

    def save(self, path: Union[str, Path]):
        """Save session signatures to file."""
        import pickle
        state = {
            'signatures': self.signatures,
            'session_names': self.session_names,
            'n_channels': self.n_channels,
            'n_warmup_samples': self.n_warmup_samples,
            'use_power_spectrum': self.use_power_spectrum,
            'sample_rate': self.sample_rate,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SessionMatcher":
        """Load session signatures from file."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)

        matcher = cls(
            n_channels=state['n_channels'],
            n_warmup_samples=state['n_warmup_samples'],
            use_power_spectrum=state.get('use_power_spectrum', False),
            sample_rate=state.get('sample_rate', 1000.0),
        )
        matcher.signatures = state['signatures']
        matcher.session_names = state.get('session_names', {})
        return matcher

    @classmethod
    def from_checkpoint_state(cls, state: Dict[str, Any]) -> "SessionMatcher":
        """Load session matcher from checkpoint state dict.

        Args:
            state: Dictionary from checkpoint["session_matcher_state"]

        Returns:
            SessionMatcher with loaded signatures
        """
        matcher = cls(
            n_channels=state['n_channels'],
            n_warmup_samples=state['n_warmup_samples'],
            use_power_spectrum=state.get('use_power_spectrum', False),
            sample_rate=state.get('sample_rate', 1000.0),
        )
        matcher.signatures = state['signatures']
        matcher.session_names = state.get('session_names', {})
        return matcher

    def __len__(self) -> int:
        return len(self.signatures)

    def __repr__(self) -> str:
        return f"SessionMatcher(n_sessions={len(self)}, n_channels={self.n_channels})"


# =============================================================================
# Utility Functions
# =============================================================================

def pearson_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute batch-averaged Pearson correlation."""
    pred_centered = pred - pred.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    num = (pred_centered * target_centered).sum(dim=-1)
    denom = torch.sqrt((pred_centered**2).sum(dim=-1) * (target_centered**2).sum(dim=-1) + 1e-8)
    corr_channels = num / denom
    return corr_channels.mean()


def pearson_per_channel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute per-channel Pearson correlation, averaged over batch.

    Args:
        pred: Predicted tensor [B, C, T]
        target: Target tensor [B, C, T]

    Returns:
        Per-channel correlations [C] - averaged over batch dimension
    """
    pred_centered = pred - pred.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    num = (pred_centered * target_centered).sum(dim=-1)  # [B, C]
    denom = torch.sqrt((pred_centered**2).sum(dim=-1) * (target_centered**2).sum(dim=-1) + 1e-8)
    corr_channels = num / denom  # [B, C]
    return corr_channels.mean(dim=0)  # [C] - mean over batch


def cross_channel_correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute cross-channel correlation matrix between pred and target.

    Measures how much each predicted channel correlates with each target channel.
    Useful for understanding if model is mixing/confusing channels.

    Args:
        pred: Predicted tensor [B, C, T]
        target: Target tensor [B, C, T]

    Returns:
        Cross-correlation matrix [C, C] where entry [i,j] is correlation between
        pred channel i and target channel j, averaged over batch.
    """
    B, C, T = pred.shape

    # Center the data
    pred_centered = pred - pred.mean(dim=-1, keepdim=True)  # [B, C, T]
    target_centered = target - target.mean(dim=-1, keepdim=True)  # [B, C, T]

    # Compute cross-correlation: for each pred channel i and target channel j
    # corr[i,j] = sum(pred[i] * target[j]) / sqrt(sum(pred[i]^2) * sum(target[j]^2))
    cross_corr = torch.zeros(C, C, device=pred.device, dtype=pred.dtype)

    pred_norm = torch.sqrt((pred_centered**2).sum(dim=-1) + 1e-8)  # [B, C]
    target_norm = torch.sqrt((target_centered**2).sum(dim=-1) + 1e-8)  # [B, C]

    for i in range(C):
        for j in range(C):
            # Correlation between pred channel i and target channel j
            num = (pred_centered[:, i, :] * target_centered[:, j, :]).sum(dim=-1)  # [B]
            denom = pred_norm[:, i] * target_norm[:, j]  # [B]
            cross_corr[i, j] = (num / denom).mean()  # scalar, avg over batch

    return cross_corr


# =============================================================================
# GPU-Accelerated Metrics
# =============================================================================

def explained_variance_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute explained variance (R²) using PyTorch (GPU-accelerated).

    Measures the proportion of variance in the target that is predictable from the model.

    Formula: R² = 1 - (Var(error) / Var(target))
           = 1 - (MSE / Var(target))

    Args:
        pred: Predicted signal [B, C, T] or [C, T] or [T]
        target: Target signal (same shape as pred)

    Returns:
        R² score (scalar tensor). Range: (-∞, 1]
        1.0 = perfect reconstruction
        0.0 = model is as good as predicting the mean
        <0 = model is worse than predicting the mean
    """
    # Compute variance of target
    target_var = torch.var(target)

    # Compute MSE (mean squared error)
    mse = torch.mean((pred - target) ** 2)

    # Prevent division by zero
    eps = 1e-12
    target_var = torch.clamp(target_var, min=eps)

    # R² = 1 - (MSE / Var)
    r2 = 1.0 - (mse / target_var)

    return r2


def normalized_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute normalized root mean squared error using PyTorch (GPU-accelerated).

    Normalized RMSE expresses reconstruction error as a fraction of the target signal's
    standard deviation, making it interpretable across different signal amplitudes.

    Formula: NRMSE = RMSE / std(target)
           = sqrt(MSE) / std(target)

    Args:
        pred: Predicted signal [B, C, T] or [C, T] or [T]
        target: Target signal (same shape as pred)

    Returns:
        NRMSE (scalar tensor). Range: [0, ∞)
        0.0 = perfect reconstruction
        1.0 = error equals one standard deviation of target
    """
    # Compute RMSE
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse)

    # Compute target standard deviation
    target_std = torch.std(target)

    # Prevent division by zero
    eps = 1e-12
    target_std = torch.clamp(target_std, min=eps)

    # Normalize
    nrmse = rmse / target_std

    return nrmse


def hilbert_phase_torch(signal: torch.Tensor) -> torch.Tensor:
    """Compute instantaneous phase using Hilbert transform (GPU-accelerated).

    Args:
        signal: Input signal [B, C, T] or [C, T] or [T]

    Returns:
        Instantaneous phase in radians (same shape as input)
    """
    # Ensure float32 for FFT precision
    if signal.dtype in (torch.bfloat16, torch.float16):
        signal = signal.float()

    # Get original shape and flatten to 2D for batch processing
    orig_shape = signal.shape
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    elif signal.ndim == 3:
        B, C, T = signal.shape
        signal = signal.view(B * C, T)
    else:  # 2D
        pass

    N = signal.shape[-1]

    # FFT-based Hilbert transform
    fft_signal = torch.fft.fft(signal, dim=-1)

    # Create frequency mask for analytic signal
    h = torch.zeros(N, device=signal.device, dtype=signal.dtype)
    if N % 2 == 0:
        h[0] = 1
        h[1:N//2] = 2
        h[N//2] = 1
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    # Apply mask and inverse FFT
    analytic = torch.fft.ifft(fft_signal * h.unsqueeze(0), dim=-1)

    # Get phase
    phase = torch.atan2(analytic.imag, analytic.real)

    # Reshape back
    if len(orig_shape) == 1:
        phase = phase.squeeze(0)
    elif len(orig_shape) == 3:
        phase = phase.view(orig_shape)

    return phase


def plv_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Phase Locking Value (GPU-accelerated).

    PLV measures the consistency of phase difference between two signals.
    PLV = |mean(exp(i * phase_diff))|

    Args:
        pred: Predicted signal [B, C, T]
        target: Target signal [B, C, T]

    Returns:
        PLV value (scalar tensor). Range [0, 1].
        1.0 = perfectly phase-locked
        0.0 = random phase relationship
    """
    # Average across channels for global phase metric
    if pred.ndim == 3:
        pred = pred.mean(dim=1)  # [B, T]
        target = target.mean(dim=1)

    phase_pred = hilbert_phase_torch(pred)
    phase_target = hilbert_phase_torch(target)
    phase_diff = phase_pred - phase_target

    # PLV = |mean(exp(i * phase_diff))|
    plv = torch.abs(torch.mean(torch.exp(1j * phase_diff.to(torch.complex64))))
    return plv.real


def pli_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Phase Lag Index (GPU-accelerated).

    PLI is robust to volume conduction artifacts - only considers the
    asymmetry of the phase difference distribution.
    PLI = |mean(sign(sin(phase_diff)))|

    Args:
        pred: Predicted signal [B, C, T]
        target: Target signal [B, C, T]

    Returns:
        PLI value (scalar tensor). Range [0, 1].
        1.0 = consistent phase lead/lag
        0.0 = symmetric phase distribution (no consistent relationship)
    """
    # Average across channels
    if pred.ndim == 3:
        pred = pred.mean(dim=1)
        target = target.mean(dim=1)

    phase_pred = hilbert_phase_torch(pred)
    phase_target = hilbert_phase_torch(target)
    phase_diff = phase_pred - phase_target

    # PLI = |mean(sign(sin(phase_diff)))|
    pli = torch.abs(torch.mean(torch.sign(torch.sin(phase_diff))))
    return pli


def psd_error_db_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    fs: float = SAMPLING_RATE_HZ,
    nperseg: int = 512,
    freq_range: tuple = (0.0, MAX_FREQ_HZ),
) -> torch.Tensor:
    """Compute PSD error in dB (GPU-accelerated, fully vectorized).

    Computes mean absolute error between PSDs in decibels.
    Lower is better (0 = perfect match).

    Args:
        pred: Predicted signal [B, C, T]
        target: Target signal [B, C, T]
        fs: Sampling frequency in Hz
        nperseg: Segment length for Welch PSD
        freq_range: (min_freq, max_freq) to consider

    Returns:
        PSD error in dB (scalar tensor). Lower is better.
    """
    # Flatten batch and channel dimensions
    if pred.ndim == 3:
        B, C, T = pred.shape
        pred = pred.view(B * C, T)
        target = target.view(B * C, T)
    elif pred.ndim == 2:
        pass
    else:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # Ensure float32 for FFT
    if pred.dtype in (torch.bfloat16, torch.float16):
        pred = pred.float()
        target = target.float()

    n_samples = pred.shape[-1]
    hop = nperseg // 2

    # Vectorized Welch PSD using unfold (no Python loops)
    # Extract all segments at once: [N, T] -> [N, n_segments, nperseg]
    pred_segments = pred.unfold(dimension=-1, size=nperseg, step=hop)
    target_segments = target.unfold(dimension=-1, size=nperseg, step=hop)
    n_segments = pred_segments.shape[1]

    # Apply Hann window [nperseg] -> broadcasted
    window = torch.hann_window(nperseg, device=pred.device, dtype=pred.dtype)
    pred_windowed = pred_segments * window  # [N, n_seg, nperseg]
    target_windowed = target_segments * window

    # Batched FFT over all segments at once
    pred_fft = torch.fft.rfft(pred_windowed, dim=-1)  # [N, n_seg, n_freq]
    target_fft = torch.fft.rfft(target_windowed, dim=-1)

    # PSD: mean over samples and segments
    pred_psd = torch.mean(torch.abs(pred_fft) ** 2, dim=(0, 1))  # [n_freq]
    target_psd = torch.mean(torch.abs(target_fft) ** 2, dim=(0, 1))

    # Frequency bins
    freqs = torch.fft.rfftfreq(nperseg, d=1.0 / fs)

    # Filter to freq_range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

    pred_psd_filtered = pred_psd[freq_mask]
    target_psd_filtered = target_psd[freq_mask]

    # Convert to dB
    eps = 1e-12
    pred_db = 10 * torch.log10(pred_psd_filtered + eps)
    target_db = 10 * torch.log10(target_psd_filtered + eps)

    # Mean absolute error in dB
    psd_err = torch.mean(torch.abs(pred_db - target_db))

    return psd_err


def psd_diff_db_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    fs: float = SAMPLING_RATE_HZ,
    nperseg: int = 512,
    freq_range: tuple = (0.0, MAX_FREQ_HZ),
) -> torch.Tensor:
    """Compute PSD difference in dB (signed, GPU-accelerated, fully vectorized).

    Positive = prediction has more power than target.
    Negative = prediction has less power than target.
    Closer to 0 is better.

    Args:
        pred: Predicted signal [B, C, T]
        target: Target signal [B, C, T]
        fs: Sampling frequency in Hz
        nperseg: Segment length for Welch PSD
        freq_range: (min_freq, max_freq) to consider

    Returns:
        PSD difference in dB (scalar tensor). Closer to 0 is better.
    """
    # Flatten batch and channel dimensions
    if pred.ndim == 3:
        B, C, T = pred.shape
        pred = pred.view(B * C, T)
        target = target.view(B * C, T)
    elif pred.ndim == 2:
        pass
    else:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # Ensure float32 for FFT
    if pred.dtype in (torch.bfloat16, torch.float16):
        pred = pred.float()
        target = target.float()

    hop = nperseg // 2

    # Vectorized Welch PSD using unfold (no Python loops)
    pred_segments = pred.unfold(dimension=-1, size=nperseg, step=hop)
    target_segments = target.unfold(dimension=-1, size=nperseg, step=hop)

    # Apply Hann window
    window = torch.hann_window(nperseg, device=pred.device, dtype=pred.dtype)
    pred_windowed = pred_segments * window
    target_windowed = target_segments * window

    # Batched FFT
    pred_fft = torch.fft.rfft(pred_windowed, dim=-1)
    target_fft = torch.fft.rfft(target_windowed, dim=-1)

    # PSD: mean over samples and segments
    pred_psd = torch.mean(torch.abs(pred_fft) ** 2, dim=(0, 1))
    target_psd = torch.mean(torch.abs(target_fft) ** 2, dim=(0, 1))

    # Frequency bins
    freqs = torch.fft.rfftfreq(nperseg, d=1.0 / fs)
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

    pred_psd_filtered = pred_psd[freq_mask]
    target_psd_filtered = target_psd[freq_mask]

    # Convert to dB and compute mean difference (signed)
    eps = 1e-12
    pred_db = 10 * torch.log10(pred_psd_filtered + eps)
    target_db = 10 * torch.log10(target_psd_filtered + eps)

    psd_diff = torch.mean(pred_db - target_db)

    return psd_diff


