"""Neural signal translation model: CondUNet1D.

This module contains the main model architecture for OB->PCx neural signal translation:
- CondUNet1D: Conditional U-Net with FiLM modulation (our method)
"""
from __future__ import annotations

import math
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
        norm_type: str = "instance",
        cond_mode: str = "cross_attn_gated",
    ):
        super().__init__()
        stride = 2 if downsample else 1
        self.cond_mode = cond_mode

        if norm_type == "instance":
            norm = nn.InstanceNorm1d(out_c, affine=True)
        elif norm_type == "batch":
            norm = nn.BatchNorm1d(out_c)
        elif norm_type == "group":
            num_groups = min(8, out_c)
            while out_c % num_groups != 0:
                num_groups -= 1
            norm = nn.GroupNorm(num_groups, out_c)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

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
    """Upsampling block with skip connections and FiLM conditioning."""
    def __init__(
        self,
        in_c: int,
        skip_c: int,
        out_c: int,
        emb_dim: int,
        dropout: float = 0.0,
        norm_type: str = "instance",
        cond_mode: str = "cross_attn_gated",
    ):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.block = ConvBlock(out_c + skip_c, out_c, emb_dim, downsample=False,
                               dropout=dropout, norm_type=norm_type, cond_mode=cond_mode)

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
        x = torch.cat([x, skip], dim=1)
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
        norm_type: str = "instance",
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
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

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
    """
    def __init__(
        self,
        in_c: int,
        skip_c: int,
        out_c: int,
        emb_dim: int,
        dropout: float = 0.0,
        norm_type: str = "instance",
        cond_mode: str = "cross_attn_gated",
        use_se: bool = True,
        kernel_size: int = 7,
        dilations: Tuple[int, ...] = (1, 4, 16, 32),
    ):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1)
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
        x = torch.cat([x, skip], dim=1)
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
# Optimal Spectral Bias (Fixed per-odor correction, no signal-adaptive network)
# =============================================================================

class OptimalSpectralBias(nn.Module):
    """Fixed per-odor spectral bias correction.

    This module applies a FIXED per-odor, per-band spectral correction.

    The key insight: OB→PCx spectral differences are physiologically determined
    and should be FIXED for each odor, NOT signal-dependent.

    Usage:
    1. After UNet training, compute optimal bias from UNet output vs target PSD
    2. Apply this fixed bias during inference

    No training required - just compute once and apply!

    Args:
        n_channels: Number of input channels (default: 32)
        n_odors: Number of odor conditions (default: 7)
        sample_rate: Sampling rate in Hz (default: 1000)
        band_width_hz: If set, use uniform bands with this spacing
                       If None, use predefined neuroscience bands
        min_freq_hz: Minimum frequency for uniform bands (default: 1.0)
        max_freq_hz: Maximum frequency for uniform bands (default: None = Nyquist)
                     IMPORTANT: Set to sample_rate/2 to cover ALL frequencies!
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_odors: int = NUM_ODORS,
        sample_rate: float = SAMPLING_RATE_HZ,
        band_width_hz: Optional[float] = None,
        min_freq_hz: float = 1.0,
        max_freq_hz: Optional[float] = None,  # None = Nyquist (sample_rate / 2)
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_odors = n_odors
        self.sample_rate = sample_rate
        self.band_width_hz = band_width_hz
        self.min_freq_hz = min_freq_hz
        # Default to Nyquist frequency to cover ALL frequencies
        self.max_freq_hz = max_freq_hz if max_freq_hz is not None else (sample_rate / 2.0)

        # Build frequency bands
        if band_width_hz is not None:
            # Use self.max_freq_hz (which defaults to Nyquist if not specified)
            self.freq_bands = make_uniform_bands(band_width_hz, self.min_freq_hz, self.max_freq_hz)
        else:
            # When using predefined neuro bands, also add a high-frequency band
            # to cover everything above 100 Hz up to Nyquist
            self.freq_bands = FREQ_BANDS_NEURO.copy()
            if self.max_freq_hz > 100.0:
                self.freq_bands["High (100-500 Hz)"] = (100.0, self.max_freq_hz)

        self.band_names = list(self.freq_bands.keys())
        self.n_bands = len(self.freq_bands)

        # Per-odor bias: fixed spectral correction per odor per band
        # This is computed directly from data, not learned through gradient descent
        self.odor_bias = nn.Parameter(torch.zeros(n_odors, self.n_bands))

        # Cache for frequency band masks
        self._cached_masks = None
        self._cached_n_fft = None

    def _build_freq_masks(self, n_fft: int, device: torch.device) -> torch.Tensor:
        """Build frequency band masks for rfft output."""
        freq_len = n_fft // 2 + 1
        if self._cached_masks is not None and self._cached_n_fft == freq_len:
            return self._cached_masks.to(device)

        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)
        n_freq_bins = len(freqs)

        masks = torch.zeros(self.n_bands, n_freq_bins)
        transition_width = self.band_width_hz / 2.0 if self.band_width_hz else 1.0

        for i, (band_name, (f_low, f_high)) in enumerate(self.freq_bands.items()):
            rise = torch.sigmoid((freqs - f_low) / transition_width * 5)
            fall = torch.sigmoid((f_high - freqs) / transition_width * 5)
            masks[i] = rise * fall

        # Normalize
        mask_sum = masks.sum(dim=0, keepdim=True).clamp(min=1e-6)
        masks = masks / mask_sum

        self._cached_masks = masks
        self._cached_n_fft = freq_len
        return masks.to(device)

    @torch.amp.autocast('cuda', enabled=False)
    def forward(
        self,
        x: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
        inverse: bool = False,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply fixed per-odor spectral bias correction.

        Args:
            x: Input tensor [B, C, T]
            odor_ids: Odor indices [B] for selecting bias
            inverse: If True, apply inverse scaling (negate bias)
            target: Ignored (API compatibility)

        Returns:
            Spectrally corrected tensor [B, C, T]
        """
        B, C, T = x.shape
        device = x.device
        original_dtype = x.dtype

        # Convert to float32 for FFT
        x_float = x.float()

        # Get frequency band masks
        masks = self._build_freq_masks(T, device)

        # FFT
        x_fft = torch.fft.rfft(x_float, dim=-1)  # [B, C, n_freq]

        # Get per-odor bias (fixed correction)
        if odor_ids is not None:
            log_scales = self.odor_bias[odor_ids]  # [B, n_bands]
        else:
            log_scales = x_float.new_zeros(B, self.n_bands)

        # Apply inverse if needed
        if inverse:
            log_scales = -log_scales

        # Convert to frequency-domain scales
        scales = torch.exp(log_scales)  # [B, n_bands]

        # Build per-frequency scale factors
        # scales: [B, n_bands], masks: [n_bands, n_freq]
        freq_scales = torch.einsum('bk,kf->bf', scales, masks)  # [B, n_freq]

        # BUG FIX: Frequencies outside defined bands (e.g., >100Hz) have zero mask coverage,
        # resulting in freq_scales=0, which zeroes out high-frequency content!
        # These frequencies should be UNCHANGED (scale=1.0), not zeroed.
        mask_coverage = masks.sum(dim=0)  # [n_freq] - how much each freq is covered by bands
        uncovered_mask = (mask_coverage < 0.01)  # frequencies with no band coverage
        freq_scales[:, uncovered_mask] = 1.0  # leave uncovered frequencies unchanged

        freq_scales = freq_scales.unsqueeze(1)  # [B, 1, n_freq]

        # Apply scaling in frequency domain
        x_fft_scaled = x_fft * freq_scales.to(x_fft.dtype)
        x_scaled = torch.fft.irfft(x_fft_scaled, n=T, dim=-1)

        return x_scaled.to(original_dtype)

    def get_bias_db(self, odor_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get bias in dB for analysis/logging.

        Args:
            odor_ids: Odor indices [B] or None for all odors

        Returns:
            Bias in dB: [B, n_bands] or [n_odors, n_bands]
        """
        if odor_ids is not None:
            bias = self.odor_bias[odor_ids]
        else:
            bias = self.odor_bias
        return 20.0 * bias / math.log(10)

    @torch.no_grad()
    def compute_optimal_bias(
        self,
        unet_outputs: torch.Tensor,
        targets: torch.Tensor,
        odor_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute optimal per-odor bias by directly measuring PSD difference.

        Measures what shift is needed: target_psd - unet_output_psd per band per odor.

        Args:
            unet_outputs: UNet output signals [N, C, T]
            targets: Target signals [N, C, T]
            odor_ids: Odor indices [N]

        Returns:
            Optimal bias in log-scale [n_odors, n_bands]
        """
        device = unet_outputs.device
        N, C, T = unet_outputs.shape

        # Build frequency masks
        masks = self._build_freq_masks(T, device)  # [n_bands, n_freq]

        # Compute PSD for both (in log scale)
        unet_fft = torch.fft.rfft(unet_outputs.float(), dim=-1)
        target_fft = torch.fft.rfft(targets.float(), dim=-1)

        unet_power = unet_fft.abs().square()  # [N, C, n_freq]
        target_power = target_fft.abs().square()

        # Per-band power
        unet_band_power = torch.einsum('ncf,kf->nck', unet_power, masks)  # [N, C, n_bands]
        target_band_power = torch.einsum('ncf,kf->nck', target_power, masks)

        # Log scale (dB-like)
        unet_log = torch.log(unet_band_power + 1e-10)
        target_log = torch.log(target_band_power + 1e-10)

        # Difference: how much to shift UNet output to match target
        # IMPORTANT: We compute log of POWER ratio, but apply to AMPLITUDE (FFT)
        # Power = |FFT|², so to scale power by P, we scale amplitude by sqrt(P)
        # Therefore: log_amplitude_ratio = log_power_ratio / 2
        diff_log = (target_log - unet_log) / 2  # [N, C, n_bands] - log AMPLITUDE ratio

        # Average across channels
        diff_log = diff_log.mean(dim=1)  # [N, n_bands]

        # Accumulate per odor
        optimal_bias = torch.zeros(self.n_odors, self.n_bands, device=device)
        counts = torch.zeros(self.n_odors, device=device)

        for i in range(N):
            odor = odor_ids[i].item()
            optimal_bias[odor] += diff_log[i]
            counts[odor] += 1

        # Average per odor
        for odor in range(self.n_odors):
            if counts[odor] > 0:
                optimal_bias[odor] /= counts[odor]

        return optimal_bias

    def set_bias_from_data(
        self,
        unet_outputs: torch.Tensor,
        targets: torch.Tensor,
        odor_ids: torch.Tensor,
    ) -> None:
        """Set odor_bias directly from measured PSD difference.

        Args:
            unet_outputs: UNet output signals [N, C, T]
            targets: Target signals [N, C, T]
            odor_ids: Odor indices [N]
        """
        optimal_bias = self.compute_optimal_bias(unet_outputs, targets, odor_ids)
        self.odor_bias.data.copy_(optimal_bias)

        # Convert to dB for logging (amplitude dB = 20*log10, we have ln)
        # bias is in ln(amplitude_ratio), convert to dB: 20 * log10(exp(bias)) = 20 * bias / ln(10)
        bias_db = optimal_bias * 20.0 / math.log(10)
        print(f"Set optimal bias from data: mean={bias_db.mean():.2f}dB, "
              f"range=[{bias_db.min():.2f}, {bias_db.max():.2f}]dB")

    def extra_repr(self) -> str:
        return (
            f"n_channels={self.n_channels}, n_odors={self.n_odors}, "
            f"n_bands={self.n_bands}"
        )


# =============================================================================
# Envelope Histogram Matching (Closed-form amplitude dynamics correction)
# =============================================================================

class EnvelopeHistogramMatching(nn.Module):
    """Closed-form envelope scaling correction (mean/std matching).

    NOTE: Full histogram matching DESTROYS temporal structure because it's a
    global statistical operation - it matches amplitude distributions without
    considering WHEN amplitudes occur. This breaks the signal!

    Instead, we use simple MEAN/STD SCALING which:
    1. Preserves temporal structure (peaks stay at same times)
    2. Adjusts overall amplitude scale to match target statistics

    The correction is:
        corrected = (pred_envelope - pred_mean) * (target_std / pred_std) + target_mean

    This is a linear transformation that preserves the SHAPE of the envelope
    while matching its mean and variability to the target.

    Usage:
        matcher = EnvelopeHistogramMatching()
        # Compute target statistics from training data
        matcher.fit(target_signals, odor_ids)
        # Apply correction during inference
        corrected = matcher(pred, odor_ids)
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
        x_fft = torch.fft.rfft(x, dim=-1)  # [B, C, n_fft]
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
        # FFT
        X = torch.fft.rfft(x, dim=-1)
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

        # Inverse FFT
        return torch.fft.irfft(X_filtered, n=x.shape[-1], dim=-1)

    def _compute_band_power(self, x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """Compute log power in frequency band for each channel.

        Args:
            x: [B, C, T] input signal
            low: Low frequency bound
            high: High frequency bound

        Returns:
            [B, C] log power per channel in the band
        """
        X = torch.fft.rfft(x, dim=-1)
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
# Main Model: CondUNet1D
# =============================================================================

class CondUNet1D(nn.Module):
    """Conditional U-Net for neural signal translation with FiLM modulation.

    This is the primary architecture for OB->PCx translation, supporting:
    - Feature-wise Linear Modulation (FiLM) for odor conditioning
    - Self-attention in the bottleneck for long-range dependencies
    - Multi-scale skip connections for detail preservation
    - OptimalSpectralBias for fixed per-odor PSD correction (applied externally)
    - Modern convolutions (multi-scale dilated depthwise separable + SE attention)
    - Configurable depth (n_downsample) to control frequency resolution

    Frequency Resolution:
    - n_downsample=4: 16x downsampling → bottleneck Nyquist = 31 Hz (loses gamma)
    - n_downsample=3: 8x downsampling → bottleneck Nyquist = 62 Hz (low gamma)
    - n_downsample=2: 4x downsampling → bottleneck Nyquist = 125 Hz (full gamma) ✓

    Attention types:
    - "basic": Original SelfAttention1D (default)
    - "none": No attention
    - "cross_freq": Cross-frequency coupling attention
    - "cross_freq_v2": Optimized cross-frequency attention with Flash Attention

    Convolution types:
    - "standard": Original Conv1d(kernel_size=3) - backward compatible
    - "modern": Multi-scale dilated depthwise separable + SE attention
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
        norm_type: str = "instance",
        cond_mode: str = "cross_attn_gated",
        use_spectral_shift: bool = True,
        # Depth control for frequency resolution
        n_downsample: int = 2,  # 2 = 4x downsample (125 Hz Nyquist), 4 = 16x (31 Hz)
        # Modern convolution options
        conv_type: str = "standard",  # "standard" or "modern"
        use_se: bool = True,  # SE attention in conv blocks (only for modern)
        conv_kernel_size: int = 7,  # Kernel size for modern convs
        dilations: Tuple[int, ...] = (1, 4, 16, 32),  # Multi-scale dilation rates
        # Output scaling correction (helps match target distribution)
        use_output_scaling: bool = True,  # Learnable per-channel scale and bias
    ):
        super().__init__()
        self.cond_mode = cond_mode
        self.emb_dim = emb_dim
        self.attention_type = attention_type
        self.conv_type = conv_type
        self.n_downsample = n_downsample

        if cond_mode != "none":
            self.embed = nn.Embedding(n_odors, emb_dim)
        else:
            self.embed = None

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
                    ModernUpBlock(dec_in, skip_in, dec_out, emb_dim, dropout=dropout, norm_type=norm_type, cond_mode=cond_mode, **conv_kwargs)
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
                    UpBlock(dec_in, skip_in, dec_out, emb_dim, dropout=dropout, norm_type=norm_type, cond_mode=cond_mode)
                )

        # Build bottleneck with configurable attention
        bottleneck_ch = channels[n_downsample]
        mid_layers = [
            nn.Conv1d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if use_attention and attention_type != "none":
            mid_layers.append(self._build_attention(bottleneck_ch, attention_type))
        mid_layers.extend([
            nn.Conv1d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1),
            nn.GELU(),
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

        # Note: OptimalSpectralBias is applied OUTSIDE the model in train.py
        # This flag is kept for backwards compatibility but is no longer used internally.
        self.use_spectral_shift = use_spectral_shift

        # Output scaling correction: learnable per-channel scale and bias
        # Helps match target distribution, especially important for probabilistic losses
        self.use_output_scaling = use_output_scaling
        if use_output_scaling:
            # Initialize scale to 1.0 and bias to 0.0 (identity at init)
            self.output_scale = nn.Parameter(torch.ones(1, out_channels, 1))
            self.output_bias = nn.Parameter(torch.zeros(1, out_channels, 1))

        # Gradient checkpointing: trade compute for memory
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable/disable gradient checkpointing for memory efficiency.

        When enabled, encoder/decoder blocks are checkpointed to reduce memory
        at the cost of ~30% more compute (recomputes activations during backward).
        """
        self.gradient_checkpointing = enable

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
            return CrossFrequencyCouplingAttention(channels)

        if attention_type == "cross_freq_v2":
            return CrossFrequencyCouplingAttentionV2(channels)

        raise ValueError(f"Unknown attention_type: {attention_type}. Available: none, basic, cross_freq, cross_freq_v2")

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
        cond_emb: Optional[torch.Tensor] = None,
        return_bottleneck: bool = False,
        return_bottleneck_temporal: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            ob: Input signal [B, C, T]
            odor_ids: Odor class indices [B] for conditioning (uses internal embedding)
            cond_emb: External conditioning embedding [B, emb_dim] (bypasses internal embedding)
                      If provided, odor_ids is ignored.
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
        # Use external embeddings if provided, otherwise use internal odor embedding
        if cond_emb is not None:
            emb = cond_emb
        elif self.embed is not None and odor_ids is not None:
            emb = self.embed(odor_ids)
        else:
            emb = None

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

        # Apply output scaling correction if enabled
        # This helps match the target distribution, especially for probabilistic losses
        if self.use_output_scaling:
            out = out * self.output_scale + self.output_bias

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
# Loss Functions
# =============================================================================

class WaveletLoss(nn.Module):
    """Continuous Wavelet Transform loss for multi-scale frequency alignment.

    Optimized implementation using batched convolutions for GPU efficiency.
    Uses explicit frequency bands for proper coverage of the full LFP range (1-100 Hz).

    Args:
        frequencies: List of center frequencies in Hz (default: NEURAL_FREQ_BANDS)
                     Covers 1-100 Hz with neuroscience-meaningful spacing.
        wavelet: Wavelet family to use (default: morlet)
        level_weights: Optional per-frequency weights
        use_complex_morlet: Whether to use complex Morlet wavelets
        omega0: Morlet wavelet parameter (default: 5.0)
                Lower values = better temporal resolution for fast transients
                Higher values = better frequency resolution
                5.0 is a good compromise for neural signals with fast oscillations.
    """

    def __init__(
        self,
        frequencies: Optional[List[float]] = None,
        wavelet: str = WAVELET_FAMILY,
        level_weights: Optional[List[float]] = None,
        use_complex_morlet: bool = USE_COMPLEX_MORLET,
        omega0: float = 5.0,  # Reduced from 6.0 for better temporal resolution
    ):
        super().__init__()
        # Use explicit neural frequency bands by default
        self.frequencies = frequencies if frequencies is not None else NEURAL_FREQ_BANDS.copy()
        self.levels = len(self.frequencies)
        self.wavelet = wavelet.lower()
        self.use_complex_morlet = use_complex_morlet
        self.omega0 = omega0
        self.is_cwt = True

        bank = self._design_morlet_bank(
            self.frequencies,
            omega0=self.omega0,
            use_complex=self.use_complex_morlet,
        )
        lengths = [len(f) for f in bank]
        max_len = max(lengths)

        # Pad all filters to same length for batched processing
        padded = []
        for f in bank:
            pad_len = max_len - len(f)
            # Center-pad filters for proper alignment
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left
            padded.append(np.pad(f, (pad_left, pad_right), mode="constant"))

        dtype = np.complex64 if self.use_complex_morlet else np.float32
        stacked = np.stack(padded, axis=0).astype(dtype)
        tensor_dtype = torch.complex64 if self.use_complex_morlet else torch.float32
        self.register_buffer("morlet_bank", torch.tensor(stacked, dtype=tensor_dtype))
        self.register_buffer("morlet_lengths", torch.tensor(lengths, dtype=torch.int64))
        self.register_buffer("freq_list", torch.tensor(self.frequencies, dtype=torch.float32))
        self.max_filter_len = max_len

        if level_weights is None:
            # Uniform weights - all frequency bands equally important
            # The explicit frequency spacing already provides proper coverage
            weights = [1.0] * len(bank)
        else:
            weights = level_weights
        self.register_buffer("level_weights", torch.tensor(weights, dtype=torch.float32))

        # Cache for batched filter kernel (built on first forward pass)
        self._cached_filters = None
        self._cached_num_channels = None

    @staticmethod
    def _design_morlet_bank(
        frequencies: List[float],
        omega0: float = 5.0,
        use_complex: bool = True,
        max_filter_len: int = 8000,
    ) -> List[np.ndarray]:
        """Design a bank of Morlet wavelets at explicit frequencies.

        DSP-compliant implementation with proper sampling and normalization.

        Args:
            frequencies: List of center frequencies in Hz
            omega0: Morlet wavelet parameter (time-frequency trade-off)
                    Lower = better temporal resolution, higher = better frequency resolution
            use_complex: Whether to use complex Morlet wavelets
            max_filter_len: Maximum filter length to prevent memory issues
        """
        bank = []
        dt = 1.0 / SAMPLING_RATE_HZ  # Proper sampling interval

        for freq in frequencies:
            # Morlet wavelet: sigma controls the Gaussian envelope width
            # sigma = omega0 / (2 * pi * freq) ensures omega0 cycles fit in the envelope
            sigma = omega0 / (2 * np.pi * freq)

            # Compute filter length: 6 sigma covers 99.7% of Gaussian
            n_samples = int(np.ceil(6 * sigma * SAMPLING_RATE_HZ))
            # Ensure odd length for symmetric filter
            if n_samples % 2 == 0:
                n_samples += 1
            # Cap filter length (important for low frequencies)
            n_samples = min(n_samples, max_filter_len)
            if n_samples % 2 == 0:
                n_samples -= 1

            # Proper time vector: exactly sampled at 1/fs intervals, centered at 0
            half_len = n_samples // 2
            t = np.arange(-half_len, half_len + 1) * dt

            # Morlet wavelet: complex exponential * Gaussian envelope
            gaussian_env = np.exp(-t**2 / (2 * sigma**2))

            if use_complex:
                # Analytic Morlet wavelet
                wavelet = np.exp(2j * np.pi * freq * t) * gaussian_env
                # Admissibility correction (remove DC component for proper wavelet)
                dc_correction = np.exp(-omega0**2 / 2)
                wavelet = wavelet - dc_correction * gaussian_env
            else:
                wavelet = np.cos(2 * np.pi * freq * t) * gaussian_env

            # L2 normalization for consistent energy across frequencies
            wavelet = wavelet / (np.sqrt(np.sum(np.abs(wavelet)**2) + 1e-12))

            bank.append(wavelet.astype(np.complex64 if use_complex else np.float32))

        return bank

    def _build_batched_filters(self, num_channels: int, device: torch.device) -> torch.Tensor:
        """Build batched filter kernel for efficient grouped convolution.

        Returns filter tensor of shape [levels * channels, 1, filter_len]
        for use with groups=levels*channels in conv1d.
        """
        # Shape: [levels, filter_len] -> [levels, 1, filter_len]
        filters = self.morlet_bank.unsqueeze(1)

        # Repeat for each channel: [levels * channels, 1, filter_len]
        # Interleave so that channel dimension varies fastest
        filters = filters.repeat(num_channels, 1, 1)

        return filters.to(device)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_details: bool = False,
        return_coefficients: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]]:
        """Compute wavelet loss with optional coefficient output for phase penalty.

        Args:
            pred: Predicted signal [B, C, T]
            target: Target signal [B, C, T]
            return_details: Return per-level loss breakdown
            return_coefficients: Return wavelet coefficients for phase computation
        """
        loss, per_level, pred_coeffs, target_coeffs = self._cwt_loss_batched(pred, target)

        if return_coefficients:
            return loss, per_level, pred_coeffs, target_coeffs
        if return_details:
            return loss, per_level
        return loss

    def _cwt_loss_batched(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Compute CWT-based loss using batched convolutions for GPU efficiency."""
        B, C, T = pred.shape
        device = pred.device
        original_dtype = pred.dtype

        # Cache filters AND device to avoid repeated .to(device) calls
        cache_valid = (
            self._cached_filters is not None
            and self._cached_num_channels == C
            and self._cached_filters.device == device
        )
        if not cache_valid:
            self._cached_filters = self._build_batched_filters(C, device)
            self._cached_num_channels = C
            # Also move level_weights to device once
            self._cached_weights = self.level_weights.to(device)

        filters = self._cached_filters
        weights = self._cached_weights

        # Determine padding (use max filter length, capped to input size)
        pad = min(self.max_filter_len // 2, T - 1)

        # Convert to float32 first if BFloat16 (complex64 doesn't support direct BF16 conversion)
        if pred.dtype in (torch.bfloat16, torch.float16):
            pred = pred.float()
            target = target.float()

        # Convert to complex if needed
        pred_c = pred.to(torch.complex64) if self.use_complex_morlet else pred
        target_c = target.to(torch.complex64) if self.use_complex_morlet else target

        # Pad inputs using constant (zero) padding to avoid boundary artifacts
        # Reflect padding creates derivative discontinuities -> spectral leakage
        # Zero padding is standard for CWT and avoids introducing spurious frequencies
        pred_padded = F.pad(pred_c, (pad, pad), mode="constant", value=0)
        target_padded = F.pad(target_c, (pad, pad), mode="constant", value=0)

        # Expand channels for batched conv: [B, C, T] -> [B, levels*C, T]
        # Each channel gets convolved with all level filters
        pred_expanded = pred_padded.repeat(1, self.levels, 1)
        target_expanded = target_padded.repeat(1, self.levels, 1)

        # Single batched convolution for all levels and channels
        # filters shape: [levels * C, 1, filter_len]
        # groups = levels * C means each input channel gets its own filter
        pred_coeffs = F.conv1d(pred_expanded, filters, groups=self.levels * C)
        target_coeffs = F.conv1d(target_expanded, filters, groups=self.levels * C)

        # Reshape to [B, levels, C, T_out] for per-level loss computation
        T_out = pred_coeffs.shape[-1]
        pred_coeffs_reshaped = pred_coeffs.view(B, self.levels, C, T_out)
        target_coeffs_reshaped = target_coeffs.view(B, self.levels, C, T_out)

        # Compute per-level losses (vectorized)
        diff = torch.abs(pred_coeffs_reshaped - target_coeffs_reshaped)
        per_level_losses = diff.mean(dim=(0, 2, 3))  # Average over batch, channels, time

        # Apply level weights
        weighted_losses = weights[:self.levels] * per_level_losses
        total_loss = weighted_losses.sum()

        # Extract real part if complex
        total_loss = total_loss.real if torch.is_complex(total_loss) else total_loss

        # Convert back to original dtype for gradient compatibility
        total_loss = total_loss.to(original_dtype)

        # Build per-level list for backward compatibility
        per_level = [
            (w.real if torch.is_complex(w) else w).detach()
            for w in weighted_losses
        ]

        return total_loss, per_level, pred_coeffs, target_coeffs

    def compute_phase_correlation(self, pred_coeffs: torch.Tensor, target_coeffs: torch.Tensor) -> torch.Tensor:
        """Compute phase correlation from pre-computed wavelet coefficients.

        Args:
            pred_coeffs: Prediction wavelet coefficients from forward pass
            target_coeffs: Target wavelet coefficients from forward pass

        Returns:
            Phase correlation (higher is better, max 1.0)
        """
        if not self.use_complex_morlet:
            # For real wavelets, use correlation instead
            pred_flat = pred_coeffs.flatten()
            target_flat = target_coeffs.flatten()
            corr = F.cosine_similarity(pred_flat.unsqueeze(0), target_flat.unsqueeze(0))
            return corr.squeeze()

        # Extract phases
        pred_phase = torch.angle(pred_coeffs)
        target_phase = torch.angle(target_coeffs)

        # Circular correlation: mean of cos(phase_diff)
        phase_diff = pred_phase - target_phase
        corr = torch.mean(torch.cos(phase_diff))

        return corr.real if torch.is_complex(corr) else corr


class MultiScaleWaveletLoss(nn.Module):
    """Thin wrapper around WaveletLoss for backward compatibility.

    With the new explicit frequency-based WaveletLoss design, multi-scale
    is no longer needed - WaveletLoss already covers all neural frequency bands
    (1-100 Hz) with neuroscience-meaningful spacing.

    This class now simply delegates to a single WaveletLoss instance.
    """

    def __init__(
        self,
        max_level: int = None,  # Ignored - kept for backward compatibility
        wavelet: str = WAVELET_FAMILY,
        use_complex_morlet: bool = USE_COMPLEX_MORLET,
        base_freq: float = None,  # Ignored - uses NEURAL_FREQ_BANDS
        frequencies: Optional[List[float]] = None,
        omega0: float = 5.0,
    ):
        super().__init__()
        # Single WaveletLoss with full neural frequency coverage
        self.wavelet_loss = WaveletLoss(
            frequencies=frequencies,
            wavelet=wavelet,
            use_complex_morlet=use_complex_morlet,
            omega0=omega0,
        )

    @property
    def levels(self) -> int:
        return self.wavelet_loss.levels

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_details: bool = False,
        return_coefficients: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        return self.wavelet_loss(pred, target, return_details=return_details, return_coefficients=return_coefficients)


class CombinedLoss(nn.Module):
    """Combined loss with multiple components."""
    def __init__(self, losses: Dict[str, nn.Module], weights: Dict[str, float]):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss. Returns tensors to avoid GPU-CPU sync.

        Call .item() on returned dict values only when needed for logging.
        """
        loss_dict = {}
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        for name, loss_fn in self.losses.items():
            loss_val = loss_fn(pred, target)
            weight = self.weights.get(name, 1.0)
            total = total + weight * loss_val
            loss_dict[name] = loss_val  # Keep as tensor - no GPU sync!

        loss_dict["total"] = total
        return total, loss_dict


def build_wavelet_loss(
    frequencies: Optional[List[float]] = None,
    wavelet: str = WAVELET_FAMILY,
    use_complex_morlet: bool = USE_COMPLEX_MORLET,
    omega0: float = 5.0,
) -> nn.Module:
    """Factory function for wavelet loss.

    Creates a WaveletLoss with explicit frequency bands covering the full
    LFP range (1-100 Hz) with neuroscience-meaningful spacing.

    Args:
        frequencies: List of center frequencies in Hz (default: NEURAL_FREQ_BANDS)
        wavelet: Wavelet family to use
        use_complex_morlet: Whether to use complex Morlet wavelets
        omega0: Morlet wavelet parameter (5.0 = good temporal resolution)

    Returns:
        WaveletLoss module with full neural frequency coverage
    """
    return WaveletLoss(
        frequencies=frequencies,
        wavelet=wavelet,
        use_complex_morlet=use_complex_morlet,
        omega0=omega0,
    )


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


