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
# Spectral Shift Block - Per-Channel Amplitude Scaling
# =============================================================================

class SpectralShiftBlock(nn.Module):
    """Learnable per-channel amplitude scaling for PSD correction.

    This is the simplest possible spectral correction: multiply each channel
    by a learnable scale factor. In log-PSD domain, this corresponds to a
    constant dB offset per channel.

    Math:
        signal × k  →  PSD × k²  →  log_PSD + 20·log₁₀(k) dB

    If OB channel i has systematically -3dB less power than PCx channel i,
    the model learns scale[i] ≈ 1.41 (= 10^(3/20)) to compensate.

    This is placed BEFORE the U-Net residual connection so the U-Net learns
    the fine details on top of the corrected amplitude.

    Args:
        n_channels: Number of input channels (default: 32)
        init_shift_db: Initial PSD shift in dB (default: 0.0 = no shift)
    """

    def __init__(self, n_channels: int = 32, init_shift_db: float = 0.0):
        super().__init__()
        self.n_channels = n_channels

        # Per-channel log-scale: scale = exp(log_scale)
        # Convert init_shift_db to log_scale: shift_db = 20 * log_scale / ln(10)
        # So: log_scale = shift_db * ln(10) / 20
        init_log_scale = init_shift_db * math.log(10) / 20.0
        self.log_scale = nn.Parameter(torch.full((n_channels,), init_log_scale))

    def get_scale(self) -> torch.Tensor:
        """Get the actual scale factors (always positive)."""
        return torch.exp(self.log_scale)

    def get_shift_db(self) -> torch.Tensor:
        """Get equivalent PSD shift in dB per channel."""
        # 20 * log10(scale) = 20 * log10(exp(log_scale)) = 20 * log_scale / ln(10)
        return 20.0 * self.log_scale / math.log(10)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply per-channel amplitude scaling.

        Args:
            x: Input tensor [B, C, T]
            target: Ignored (kept for API compatibility)

        Returns:
            Scaled tensor [B, C, T]
        """
        # scale: [C] → [1, C, 1] for broadcasting
        # CRITICAL: Ensure scale is on same device/dtype as input (FSDP + mixed precision)
        scale = self.get_scale().to(device=x.device, dtype=x.dtype).view(1, -1, 1)

        return x * scale

    def extra_repr(self) -> str:
        shift_db = self.get_shift_db()
        mean_shift = shift_db.mean().item()
        std_shift = shift_db.std().item()
        return f"n_channels={self.n_channels}, mean_shift={mean_shift:.2f}dB, std={std_shift:.2f}dB"


# =============================================================================
# Frequency-Band Spectral Shift - Per-Band Amplitude Scaling
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


class FrequencyBandSpectralShift(nn.Module):
    """Learnable per-frequency-band amplitude scaling for PSD correction.

    Unlike SpectralShiftBlock which applies a flat dB shift per channel,
    this module applies DIFFERENT dB shifts to different frequency bands.
    This allows correcting frequency-dependent PSD mismatches.

    Supports two band modes:
    1. Uniform spacing: Set band_width_hz (e.g., 2.0 for 2Hz bands)
    2. Neuroscience bands: Leave band_width_hz=None to use predefined bands

    Supports odor conditioning: different odors can have different spectral
    corrections (e.g., some odors may evoke stronger gamma, others more theta).

    Uses FFT to decompose signal, applies band-specific scaling in frequency
    domain, then inverse FFT to reconstruct.

    Args:
        sample_rate: Sampling rate in Hz (default: 1000)
        init_shift_db: Initial PSD shift in dB for all bands (default: 0.0)
        per_channel: If True, learn separate shifts per channel
        n_channels: Number of channels (only used if per_channel=True)
        n_odors: Number of odor conditions (default: 7)
        conditional: If True, learn odor-specific adjustments on top of baseline
        init_global_gain_db: Initial global gain in dB (default: 0.0)
        band_width_hz: If set, use uniform bands with this spacing (e.g., 2.0 for 2Hz)
                       If None, use predefined neuroscience bands (default: None)
        min_freq_hz: Minimum frequency for uniform bands (default: 1.0)
        max_freq_hz: Maximum frequency for uniform bands (default: 100.0)
    """

    def __init__(
        self,
        sample_rate: float = SAMPLING_RATE_HZ,
        init_shift_db: float = 0.0,
        per_channel: bool = False,
        n_channels: int = 32,
        n_odors: int = NUM_ODORS,
        conditional: bool = False,
        init_global_gain_db: float = 0.0,
        band_width_hz: Optional[float] = None,
        min_freq_hz: float = 1.0,
        max_freq_hz: float = 100.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.per_channel = per_channel
        self.n_channels = n_channels
        self.n_odors = n_odors
        self.conditional = conditional
        self.band_width_hz = band_width_hz
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz

        # Build frequency bands (uniform or predefined)
        if band_width_hz is not None:
            # Uniform spacing mode
            self.freq_bands = make_uniform_bands(band_width_hz, min_freq_hz, max_freq_hz)
        else:
            # Predefined neuroscience bands
            self.freq_bands = FREQ_BANDS_NEURO

        self.band_names = list(self.freq_bands.keys())
        self.n_bands = len(self.freq_bands)

        # Convert init dB to log_scale
        init_log_scale = init_shift_db * math.log(10) / 20.0

        # GLOBAL GAIN: Single scalar that applies uniform dB shift to ALL frequencies
        # This fixes overall power level, while per-band params fix spectral shape
        # Final scale = global_gain * per_band_scale
        init_global_log = init_global_gain_db * math.log(10) / 20.0
        self.log_global_gain = nn.Parameter(torch.tensor(init_global_log))

        if per_channel:
            # [n_channels, n_bands] - different shift per channel per band
            self.log_scale = nn.Parameter(
                torch.full((n_channels, self.n_bands), init_log_scale)
            )
        else:
            # [n_bands] - same shift for all channels, different per band
            self.log_scale = nn.Parameter(
                torch.full((self.n_bands,), init_log_scale)
            )

        # Odor conditioning: per-odor adjustments (additive in log-space)
        # Final scale = base_scale * odor_scale (multiplicative in linear space)
        if conditional:
            # [n_odors, n_bands] - per-odor delta in log-space
            # Initialize to zero so all odors start with same baseline
            self.odor_log_scale = nn.Parameter(
                torch.zeros(n_odors, self.n_bands)
            )
        else:
            self.odor_log_scale = None

        # Cache for frequency band masks (built on first forward)
        self._cached_masks = None
        self._cached_n_fft = None

    def _build_freq_masks(self, n_fft: int, device: torch.device) -> torch.Tensor:
        """Build frequency band masks for rfft output.

        Returns:
            Tensor of shape [n_bands, n_freq_bins] with soft masks for each band
        """
        if self._cached_masks is not None and self._cached_n_fft == n_fft:
            return self._cached_masks.to(device)

        # Frequency bins for rfft
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)
        n_freq_bins = len(freqs)

        # Build masks for each band with soft edges (smooth transitions)
        masks = torch.zeros(self.n_bands, n_freq_bins)
        # Transition width scales with band width for smooth blending
        transition_width = self.band_width_hz / 2.0 if self.band_width_hz else 1.0

        for i, (band_name, (f_low, f_high)) in enumerate(self.freq_bands.items()):
            # Soft mask using sigmoid transitions
            # Rising edge at f_low, falling edge at f_high
            rise = torch.sigmoid((freqs - f_low) / transition_width * 5)
            fall = torch.sigmoid((f_high - freqs) / transition_width * 5)
            masks[i] = rise * fall

        # Normalize so masks sum to ~1 at each frequency (avoid gaps/overlaps)
        mask_sum = masks.sum(dim=0, keepdim=True).clamp(min=1e-6)
        masks = masks / mask_sum

        self._cached_masks = masks
        self._cached_n_fft = n_fft
        return masks.to(device)

    def get_global_gain(self) -> torch.Tensor:
        """Get global gain factor (scalar)."""
        return torch.exp(self.log_global_gain)

    def get_global_gain_db(self) -> float:
        """Get global gain in dB."""
        return (20.0 * self.log_global_gain / math.log(10)).item()

    def get_scale(self, odor_ids: Optional[torch.Tensor] = None, inverse: bool = False) -> torch.Tensor:
        """Get actual scale factors (always positive) per band.

        INCLUDES global gain: final_scale = global_gain * per_band_scale

        Args:
            odor_ids: Optional odor indices [B] for conditional scaling
            inverse: If True, return 1/scale (for reverse direction).
                     This ensures forward and reverse are consistent inverses.

        Returns:
            Scale factors. Shape depends on conditioning:
            - Unconditional: [n_bands] or [n_channels, n_bands]
            - Conditional: [B, n_bands] or [B, n_channels, n_bands]
        """
        # Start with global gain (applies to all frequencies uniformly)
        log_total = self.log_global_gain + self.log_scale

        # Add odor-specific adjustments if conditional
        if self.conditional and odor_ids is not None and self.odor_log_scale is not None:
            # odor_log_scale: [n_odors, n_bands]
            # odor_ids: [B] -> select [B, n_bands]
            odor_delta = self.odor_log_scale[odor_ids]  # [B, n_bands]

            if self.per_channel:
                # log_total: [C, n_bands], odor_delta: [B, n_bands]
                # Result: [B, C, n_bands]
                log_total = log_total.unsqueeze(0) + odor_delta.unsqueeze(1)
            else:
                # log_total: [n_bands], odor_delta: [B, n_bands]
                # Result: [B, n_bands]
                log_total = log_total.unsqueeze(0) + odor_delta

        # Inverse mode: negate log_scale so scale = 1/original_scale
        # If forward is +3dB, inverse is -3dB
        if inverse:
            log_total = -log_total

        return torch.exp(log_total)

    def get_shift_db(self, odor_ids: Optional[torch.Tensor] = None, inverse: bool = False) -> torch.Tensor:
        """Get equivalent PSD shift in dB per band (INCLUDES global gain)."""
        log_total = self.log_global_gain + self.log_scale

        if self.conditional and odor_ids is not None and self.odor_log_scale is not None:
            odor_delta = self.odor_log_scale[odor_ids]
            if self.per_channel:
                log_total = log_total.unsqueeze(0) + odor_delta.unsqueeze(1)
            else:
                log_total = log_total.unsqueeze(0) + odor_delta

        if inverse:
            log_total = -log_total

        return 20.0 * log_total / math.log(10)

    def get_shift_db_dict(self, odor_id: Optional[int] = None) -> Dict[str, float]:
        """Get dB shifts as a dict with band names.

        Args:
            odor_id: Optional single odor index for conditional mode

        Returns:
            Dict mapping band names to dB shifts
        """
        if odor_id is not None and self.conditional:
            odor_tensor = torch.tensor([odor_id], device=self.log_scale.device)
            shifts = self.get_shift_db(odor_tensor)[0]  # [n_bands] or [C, n_bands]
        else:
            shifts = self.get_shift_db()

        if self.per_channel and shifts.ndim > 1:
            # Return mean across channels for each band
            shifts = shifts.mean(dim=0)

        return {name: shifts[i].item() for i, name in enumerate(self.band_names)}

    def get_odor_shift_db_dict(self) -> Dict[str, Dict[str, float]]:
        """Get per-odor dB shifts (only for conditional mode).

        Returns:
            Dict mapping odor index to band shifts dict
        """
        if not self.conditional or self.odor_log_scale is None:
            return {}

        result = {}
        for odor_id in range(self.n_odors):
            result[f"odor_{odor_id}"] = self.get_shift_db_dict(odor_id=odor_id)
        return result

    @torch.amp.autocast('cuda', enabled=False)
    def forward(
        self,
        x: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
        inverse: bool = False,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply frequency-band-specific amplitude scaling.

        Args:
            x: Input tensor [B, C, T]
            odor_ids: Optional odor indices [B] for conditional scaling
            inverse: If True, apply inverse scaling (for reverse direction).
                     Forward uses scale, reverse uses 1/scale.
                     This ensures bidirectional consistency: if forward is +3dB,
                     reverse is automatically -3dB.
            target: Ignored (kept for API compatibility)

        Returns:
            Scaled tensor [B, C, T]
        """
        B, C, T = x.shape
        device = x.device
        original_dtype = x.dtype

        # Convert to float32 once (FFT precision)
        x_float = x.float()

        # FFT
        x_fft = torch.fft.rfft(x_float, dim=-1)  # [B, C, n_freq]

        # Get frequency band masks [n_bands, n_freq]
        masks = self._build_freq_masks(T, device)

        # Get scales - shape depends on conditioning
        # inverse=True for reverse direction (uses 1/scale)
        scales = self.get_scale(odor_ids, inverse=inverse).to(device=device, dtype=x_float.dtype)

        # Build per-frequency scale factors
        if self.conditional and odor_ids is not None:
            # Conditional mode: scales have batch dimension
            if self.per_channel:
                # scales: [B, C, n_bands], masks: [n_bands, n_freq]
                # Result: [B, C, n_freq]
                freq_scales = torch.einsum('bck,kf->bcf', scales, masks)
            else:
                # scales: [B, n_bands], masks: [n_bands, n_freq]
                # Result: [B, n_freq]
                freq_scales = torch.einsum('bk,kf->bf', scales, masks)
                # Add channel dim: [B, 1, n_freq]
                freq_scales = freq_scales.unsqueeze(1)
        else:
            # Unconditional mode: no batch dimension in scales
            if self.per_channel:
                # scales: [C, n_bands], masks: [n_bands, n_freq]
                # Result: [C, n_freq]
                freq_scales = torch.einsum('ck,kf->cf', scales, masks)
                # Add batch dim: [1, C, n_freq]
                freq_scales = freq_scales.unsqueeze(0)
            else:
                # scales: [n_bands], masks: [n_bands, n_freq]
                # Result: [n_freq]
                freq_scales = torch.einsum('k,kf->f', scales, masks)
                # Add batch and channel dims: [1, 1, n_freq]
                freq_scales = freq_scales.unsqueeze(0).unsqueeze(0)

        # Apply scaling in frequency domain
        x_fft_scaled = x_fft * freq_scales

        # Inverse FFT
        x_scaled = torch.fft.irfft(x_fft_scaled, n=T, dim=-1)

        return x_scaled.to(original_dtype)

    def extra_repr(self) -> str:
        global_gain_db = self.get_global_gain_db()
        shifts = self.get_shift_db_dict()
        shift_str = ", ".join(f"{k}={v:+.1f}dB" for k, v in shifts.items())
        cond_str = f", conditional={self.conditional}" if self.conditional else ""
        n_params = 1 + self.n_bands * (self.n_channels if self.per_channel else 1)  # +1 for global gain
        if self.conditional:
            n_params += self.n_odors * self.n_bands
        return f"global={global_gain_db:+.1f}dB, per_channel={self.per_channel}{cond_str}, n_params={n_params}, bands=[{shift_str}]"


# =============================================================================
# Adaptive Spectral Shift - Signal-Adaptive Dynamic Shifting
# =============================================================================

class AdaptiveSpectralShift(nn.Module):
    """Dynamic signal-adaptive spectral shift with odor conditioning.

    Unlike FrequencyBandSpectralShift which learns STATIC per-band shifts,
    this module DYNAMICALLY computes shifts based on:
    1. The input signal's current spectral content (analyzes what's there)
    2. The odor condition (what transformation is needed)

    This allows the model to apply different spectral corrections to different
    signals even with the same odor - adapting to the signal's actual state.

    Architecture:
    1. Extract per-band power from input signal via FFT
    2. Encode band powers + odor embedding through a small network
    3. Predict per-band log-scale factors (dB shifts)
    4. Apply in frequency domain

    Args:
        n_channels: Number of input channels (default: 32)
        n_odors: Number of odor conditions (default: 7)
        emb_dim: Odor embedding dimension (default: 128)
        hidden_dim: Hidden dimension for shift predictor (default: 64)
        sample_rate: Sampling rate in Hz (default: 1000)
        band_width_hz: If set, use uniform bands with this spacing
                       If None, use predefined neuroscience bands
        min_freq_hz: Minimum frequency for uniform bands (default: 1.0)
        max_freq_hz: Maximum frequency for uniform bands (default: 100.0)
        max_shift_db: Maximum allowed shift in dB (default: 12.0)
        per_channel: If True, predict separate shifts per channel (default: False)
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_odors: int = NUM_ODORS,
        emb_dim: int = 128,
        hidden_dim: int = 64,
        sample_rate: float = SAMPLING_RATE_HZ,
        band_width_hz: Optional[float] = None,
        min_freq_hz: float = 1.0,
        max_freq_hz: float = 100.0,
        max_shift_db: float = 12.0,
        per_channel: bool = False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_odors = n_odors
        self.emb_dim = emb_dim
        self.sample_rate = sample_rate
        self.band_width_hz = band_width_hz
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.per_channel = per_channel

        # Maximum shift in log-scale (for tanh clamping)
        self.max_log_scale = max_shift_db * math.log(10) / 20.0

        # Build frequency bands
        if band_width_hz is not None:
            self.freq_bands = make_uniform_bands(band_width_hz, min_freq_hz, max_freq_hz)
        else:
            self.freq_bands = FREQ_BANDS_NEURO

        self.band_names = list(self.freq_bands.keys())
        self.n_bands = len(self.freq_bands)

        # Odor embedding
        self.embed = nn.Embedding(n_odors, emb_dim)

        # Feature extraction: band powers -> features
        # Input: n_bands (mean power per band, log-scaled)
        # We also add statistics: mean, std, max across bands
        band_feature_dim = self.n_bands + 3  # band powers + stats

        if per_channel:
            # Per-channel mode: process each channel's band powers
            self.band_encoder = nn.Sequential(
                nn.Linear(band_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # Combine with odor embedding
            self.shift_predictor = nn.Sequential(
                nn.Linear(hidden_dim + emb_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.n_bands),
                nn.Tanh(),  # Output in [-1, 1], scaled by max_shift
            )
        else:
            # Global mode: average band powers across channels
            self.band_encoder = nn.Sequential(
                nn.Linear(band_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # Combine with odor embedding
            self.shift_predictor = nn.Sequential(
                nn.Linear(hidden_dim + emb_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.n_bands),
                nn.Tanh(),  # Output in [-1, 1], scaled by max_shift
            )

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

    def _extract_band_powers(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Extract per-band power from signal.

        Args:
            x: Input signal [B, C, T]
            masks: Frequency band masks [n_bands, n_freq]

        Returns:
            Band powers in log scale [B, C, n_bands] or [B, n_bands]
        """
        B, C, T = x.shape

        # FFT
        x_fft = torch.fft.rfft(x, dim=-1)  # [B, C, n_freq]
        power = x_fft.abs().square()  # [B, C, n_freq] - faster than abs()**2

        # Compute per-band power: sum power within each band
        # masks: [n_bands, n_freq], power: [B, C, n_freq]
        # Result: [B, C, n_bands]
        band_power = torch.einsum('bcf,kf->bck', power, masks)

        # Log-scale for better numerical behavior (add eps for stability)
        band_power_log = torch.log(band_power + 1e-10)

        if not self.per_channel:
            # Average across channels: [B, n_bands]
            band_power_log = band_power_log.mean(dim=1)

        return band_power_log

    def _compute_band_features(self, band_powers: torch.Tensor) -> torch.Tensor:
        """Compute features from band powers.

        Args:
            band_powers: [B, n_bands] or [B, C, n_bands]

        Returns:
            Features: [B, n_bands + 3] or [B, C, n_bands + 3]
        """
        if self.per_channel:
            # [B, C, n_bands]
            mean_power = band_powers.mean(dim=-1, keepdim=True)
            std_power = band_powers.std(dim=-1, keepdim=True)
            max_power = band_powers.max(dim=-1, keepdim=True)[0]
            features = torch.cat([band_powers, mean_power, std_power, max_power], dim=-1)
        else:
            # [B, n_bands]
            mean_power = band_powers.mean(dim=-1, keepdim=True)
            std_power = band_powers.std(dim=-1, keepdim=True)
            max_power = band_powers.max(dim=-1, keepdim=True)[0]
            features = torch.cat([band_powers, mean_power, std_power, max_power], dim=-1)

        return features

    @torch.amp.autocast('cuda', enabled=False)
    def forward(
        self,
        x: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
        inverse: bool = False,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply dynamic signal-adaptive spectral shift.

        Args:
            x: Input tensor [B, C, T]
            odor_ids: Odor indices [B] for conditioning
            inverse: If True, apply inverse scaling (negate predicted shifts)
            target: Ignored (API compatibility)

        Returns:
            Spectrally shifted tensor [B, C, T]
        """
        B, C, T = x.shape
        device = x.device
        original_dtype = x.dtype

        # Convert to float32 for FFT (do once)
        x_float = x.float()

        # Get frequency band masks
        masks = self._build_freq_masks(T, device)

        # === Compute FFT once, reuse for both analysis and transformation ===
        x_fft = torch.fft.rfft(x_float, dim=-1)  # [B, C, n_freq]

        # === Step 1: Analyze input signal (reuse x_fft) ===
        power = x_fft.abs().square()  # [B, C, n_freq]
        band_power = torch.einsum('bcf,kf->bck', power, masks)
        band_power_log = torch.log(band_power + 1e-10)
        if not self.per_channel:
            band_power_log = band_power_log.mean(dim=1)  # [B, n_bands]

        # Compute features inline (avoid function call overhead)
        if self.per_channel:
            mean_power = band_power_log.mean(dim=-1, keepdim=True)
            std_power = band_power_log.std(dim=-1, keepdim=True)
            max_power = band_power_log.amax(dim=-1, keepdim=True)
            band_features = torch.cat([band_power_log, mean_power, std_power, max_power], dim=-1)
        else:
            mean_power = band_power_log.mean(dim=-1, keepdim=True)
            std_power = band_power_log.std(dim=-1, keepdim=True)
            max_power = band_power_log.amax(dim=-1, keepdim=True)
            band_features = torch.cat([band_power_log, mean_power, std_power, max_power], dim=-1)

        # === Step 2: Get odor embedding ===
        if odor_ids is not None:
            odor_emb = self.embed(odor_ids)  # [B, emb_dim]
        else:
            odor_emb = x_float.new_zeros(B, self.emb_dim)  # Stays on same device/dtype

        # === Step 3: Predict per-band shifts ===
        if self.per_channel:
            # Encode each channel's band features
            # band_features: [B, C, n_bands+3]
            band_encoded = self.band_encoder(band_features)  # [B, C, hidden_dim]

            # Expand odor embedding to match: [B, C, emb_dim]
            odor_emb_expanded = odor_emb.unsqueeze(1).expand(-1, C, -1)

            # Concatenate and predict shifts
            combined = torch.cat([band_encoded, odor_emb_expanded], dim=-1)  # [B, C, hidden_dim + emb_dim]
            log_scales = self.shift_predictor(combined)  # [B, C, n_bands] in [-1, 1]
            log_scales = log_scales * self.max_log_scale  # Scale to max dB range
        else:
            # Encode global band features
            band_encoded = self.band_encoder(band_features)  # [B, hidden_dim]

            # Concatenate with odor embedding
            combined = torch.cat([band_encoded, odor_emb], dim=-1)  # [B, hidden_dim + emb_dim]
            log_scales = self.shift_predictor(combined)  # [B, n_bands] in [-1, 1]
            log_scales = log_scales * self.max_log_scale  # Scale to max dB range

        # === Step 4: Apply inverse if needed ===
        if inverse:
            log_scales = -log_scales

        # === Step 5: Convert to frequency-domain scales and apply ===
        scales = torch.exp(log_scales)  # [B, n_bands] or [B, C, n_bands]

        # Build per-frequency scale factors
        if self.per_channel:
            # scales: [B, C, n_bands], masks: [n_bands, n_freq]
            freq_scales = torch.einsum('bck,kf->bcf', scales, masks)  # [B, C, n_freq]
        else:
            # scales: [B, n_bands], masks: [n_bands, n_freq]
            freq_scales = torch.einsum('bk,kf->bf', scales, masks)  # [B, n_freq]
            freq_scales = freq_scales.unsqueeze(1)  # [B, 1, n_freq]

        # === Step 5: Apply scaling in frequency domain (reuse x_fft) ===
        x_fft_scaled = x_fft * freq_scales.to(x_fft.dtype)
        x_scaled = torch.fft.irfft(x_fft_scaled, n=T, dim=-1)

        return x_scaled.to(original_dtype)

    def get_predicted_shifts_db(
        self,
        x: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get predicted dB shifts for analysis/logging (without applying them).

        Args:
            x: Input tensor [B, C, T]
            odor_ids: Odor indices [B]

        Returns:
            Predicted shifts in dB: [B, n_bands] or [B, C, n_bands]
        """
        B, C, T = x.shape
        device = x.device

        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()

        masks = self._build_freq_masks(T, device)
        band_powers = self._extract_band_powers(x, masks)
        band_features = self._compute_band_features(band_powers)

        if odor_ids is not None:
            odor_emb = self.embed(odor_ids)
        else:
            odor_emb = torch.zeros(B, self.emb_dim, device=device, dtype=x.dtype)

        if self.per_channel:
            band_encoded = self.band_encoder(band_features)
            odor_emb_expanded = odor_emb.unsqueeze(1).expand(-1, C, -1)
            combined = torch.cat([band_encoded, odor_emb_expanded], dim=-1)
            log_scales = self.shift_predictor(combined) * self.max_log_scale
        else:
            band_encoded = self.band_encoder(band_features)
            combined = torch.cat([band_encoded, odor_emb], dim=-1)
            log_scales = self.shift_predictor(combined) * self.max_log_scale

        # Convert to dB
        return 20.0 * log_scales / math.log(10)

    def extra_repr(self) -> str:
        max_db = self.max_log_scale * 20.0 / math.log(10)
        return (
            f"n_channels={self.n_channels}, n_bands={self.n_bands}, "
            f"per_channel={self.per_channel}, max_shift={max_db:.1f}dB"
        )


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

    def __init__(self, in_channels: int, embed_dim: int = 128, fs: float = 1000.0):
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
# Main Model: CondUNet1D
# =============================================================================

class CondUNet1D(nn.Module):
    """Conditional U-Net for neural signal translation with FiLM modulation.

    This is the primary architecture for OB->PCx translation, supporting:
    - Feature-wise Linear Modulation (FiLM) for odor conditioning
    - Self-attention in the bottleneck for long-range dependencies
    - Multi-scale skip connections for detail preservation
    - SpectralShiftBlock for learnable per-channel PSD correction
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

        # Note: SpectralShiftBlock is now applied OUTSIDE the model in train.py
        # to avoid FSDP parameter duplication issues. This flag is kept for
        # backwards compatibility but is no longer used internally.
        self.use_spectral_shift = use_spectral_shift

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
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            ob: Input signal [B, C, T]
            odor_ids: Odor class indices [B] for conditioning (uses internal embedding)
            cond_emb: External conditioning embedding [B, emb_dim] (bypasses internal embedding)
                      If provided, odor_ids is ignored.

        Returns:
            Predicted signal [B, C, T]
        """
        # Use external embeddings if provided, otherwise use internal odor embedding
        if cond_emb is not None:
            emb = cond_emb
        elif self.embed is not None and odor_ids is not None:
            emb = self.embed(odor_ids)
        else:
            emb = None

        # Encoder path with skip connections
        skips = [self.inc(ob, emb)]
        for encoder in self.encoders:
            skips.append(encoder(skips[-1], emb))
        
        # Bottleneck
        x = self.mid(skips[-1])
        
        # Decoder path with skip connections (reverse order)
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

        # Note: SpectralShift is now applied OUTSIDE the model in train.py
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

class TemporalSmoothness(nn.Module):
    """Temporal smoothness loss (second derivative penalty)."""
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        d2 = signal[..., 2:] - 2 * signal[..., 1:-1] + signal[..., :-2]
        return torch.mean(d2**2)


class HighFrequencySpectralLoss(nn.Module):
    """Spectral loss that computes FFT-based PSD matching.

    This loss computes the Power Spectral Density and applies frequency-dependent
    weights. With low_freq_cutoff=0, it provides uniform weighting across the
    full 0-100 Hz LFP range.

    Args:
        sample_rate: Sampling rate in Hz (default: 1000)
        low_freq_cutoff: Frequencies below this get weight=1, above get high_freq_boost
                         Set to 0 for uniform weighting (default: 0)
        high_freq_boost: Multiplicative boost for frequencies above low_freq_cutoff (default: 1.0)
        max_freq: Maximum frequency to consider (default: 100 Hz, LFP range)
        use_log_psd: Whether to compare log-magnitude PSDs (default: True)
    """

    def __init__(
        self,
        sample_rate: float = SAMPLING_RATE_HZ,
        low_freq_cutoff: float = 0.0,  # No cutoff - full range
        high_freq_boost: float = 1.0,  # Uniform weighting by default
        max_freq: float = MAX_FREQ_HZ,
        use_log_psd: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.low_freq_cutoff = low_freq_cutoff
        self.high_freq_boost = high_freq_boost
        self.max_freq = max_freq
        self.use_log_psd = use_log_psd
        self._freq_weights = None
        self._cached_length = None

    def _get_freq_weights(self, n_fft: int, device: torch.device) -> torch.Tensor:
        """Build frequency-dependent weights (cached for efficiency)."""
        if self._freq_weights is not None and self._cached_length == n_fft:
            return self._freq_weights.to(device)

        # Frequency bins for rfft output
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)

        # Weight: 1.0 for low frequencies, high_freq_boost for high frequencies
        weights = torch.ones_like(freqs)
        if self.low_freq_cutoff > 0:
            weights[freqs >= self.low_freq_cutoff] = self.high_freq_boost

            # Smooth transition around cutoff (helps with gradient stability)
            transition_width = 10.0  # Hz
            transition_mask = (freqs >= self.low_freq_cutoff - transition_width) & (freqs < self.low_freq_cutoff)
            if transition_mask.any():
                t = (freqs[transition_mask] - (self.low_freq_cutoff - transition_width)) / transition_width
                weights[transition_mask] = 1.0 + (self.high_freq_boost - 1.0) * t

        # Zero out frequencies above max_freq (100 Hz for LFP)
        weights[freqs > self.max_freq] = 0.0

        self._freq_weights = weights
        self._cached_length = n_fft
        return weights.to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute frequency-weighted spectral loss.

        Args:
            pred: Predicted signal [B, C, T]
            target: Target signal [B, C, T]

        Returns:
            Scalar loss value
        """
        B, C, T = pred.shape

        # Convert to float32 if needed (BFloat16 not supported by FFT)
        original_dtype = pred.dtype
        if pred.dtype in (torch.bfloat16, torch.float16):
            pred = pred.float()
            target = target.float()

        # Compute FFT
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)

        # Compute magnitude (PSD)
        pred_psd = torch.abs(pred_fft) ** 2
        target_psd = torch.abs(target_fft) ** 2

        if self.use_log_psd:
            # Log-magnitude for better dynamic range
            eps = 1e-10
            pred_psd = torch.log(pred_psd + eps)
            target_psd = torch.log(target_psd + eps)

        # Get frequency weights
        weights = self._get_freq_weights(T, pred.device)  # [F]

        # Compute weighted MSE
        diff = (pred_psd - target_psd) ** 2  # [B, C, F]
        weighted_diff = diff * weights.unsqueeze(0).unsqueeze(0)  # Broadcast weights
        loss = weighted_diff.mean()

        return loss.to(original_dtype)


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
    # Legacy parameters (ignored, kept for backward compatibility)
    levels: int = None,
    multiscale: bool = None,
    base_freq: float = None,
) -> nn.Module:
    """Factory function for wavelet loss.

    Creates a WaveletLoss with explicit frequency bands covering the full
    LFP range (1-100 Hz) with neuroscience-meaningful spacing.

    Args:
        frequencies: List of center frequencies in Hz (default: NEURAL_FREQ_BANDS)
        wavelet: Wavelet family to use
        use_complex_morlet: Whether to use complex Morlet wavelets
        omega0: Morlet wavelet parameter (5.0 = good temporal resolution)
        levels: DEPRECATED - ignored, kept for backward compatibility
        multiscale: DEPRECATED - ignored, kept for backward compatibility
        base_freq: DEPRECATED - ignored, kept for backward compatibility

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

def cwt_phase_corr(
    pred: torch.Tensor,
    target: torch.Tensor,
    wavelet_loss: Optional[nn.Module],
    pred_coeffs: Optional[torch.Tensor] = None,
    target_coeffs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute phase correlation between pred and target using CWT.

    Optimized version that can reuse pre-computed wavelet coefficients from
    the wavelet loss forward pass, avoiding redundant computation.

    Args:
        pred: Predicted signal [B, C, T]
        target: Target signal [B, C, T]
        wavelet_loss: WaveletLoss module (used if coefficients not provided)
        pred_coeffs: Pre-computed prediction coefficients (optional, for efficiency)
        target_coeffs: Pre-computed target coefficients (optional, for efficiency)

    Returns:
        Phase correlation scalar (higher is better, max 1.0)
    """
    # Fast path: use pre-computed coefficients from wavelet loss
    if pred_coeffs is not None and target_coeffs is not None:
        if hasattr(wavelet_loss, 'compute_phase_correlation'):
            return wavelet_loss.compute_phase_correlation(pred_coeffs, target_coeffs)
        # Fallback: compute phase correlation directly
        pred_phase = torch.angle(pred_coeffs)
        target_phase = torch.angle(target_coeffs)
        phase_diff = pred_phase - target_phase
        corr = torch.mean(torch.cos(phase_diff))
        return corr.real if torch.is_complex(corr) else corr

    # Legacy path: compute coefficients from scratch
    if wavelet_loss is None or not hasattr(wavelet_loss, 'morlet_bank'):
        return pred.new_tensor(0.0)

    # Use first level wavelet for phase computation
    filt = wavelet_loss.morlet_bank[0] if hasattr(wavelet_loss, 'morlet_bank') else None
    if filt is None:
        return pred.new_tensor(0.0)

    # Convert to float32 first if BFloat16 (complex64 doesn't support direct BF16 conversion)
    if pred.dtype in (torch.bfloat16, torch.float16):
        pred = pred.float()
        target = target.float()

    pred_c = pred.to(torch.complex64)
    target_c = target.to(torch.complex64)

    filt_t = filt.view(1, 1, -1).to(dtype=torch.complex64, device=pred.device)
    pad = filt_t.shape[-1] // 2
    # Cap padding to input size - 1 to prevent reflect padding error
    max_pad = pred.shape[-1] - 1
    pad = min(pad, max_pad)
    filt_rep = filt_t.repeat(pred.shape[1], 1, 1)

    # Use constant (zero) padding to avoid boundary artifacts from reflect padding
    p_coeff = F.conv1d(F.pad(pred_c, (pad, pad), mode="constant", value=0), filt_rep, groups=pred_c.shape[1])
    t_coeff = F.conv1d(F.pad(target_c, (pad, pad), mode="constant", value=0), filt_rep, groups=target_c.shape[1])

    # Phase correlation
    p_phase = torch.angle(p_coeff)
    t_phase = torch.angle(t_coeff)
    phase_diff = p_phase - t_phase

    # Circular correlation
    corr = torch.mean(torch.cos(phase_diff))
    return corr


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


# =============================================================================
# Loss Functions
# =============================================================================

class TemporalGradientLoss(nn.Module):
    """Loss that matches temporal derivatives of signals.

    Matching signal derivatives captures transient events, sharp waves,
    and other rapid changes that may be missed by direct signal comparison.

    Args:
        orders: List of derivative orders to compute (1=first derivative, etc.)
        weights: Weights for each derivative order
    """

    def __init__(
        self,
        orders: List[int] = [1, 2],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.orders = orders

        if weights is None:
            weights = [1.0] * len(orders)
        self.weights = weights

        total = sum(weights)
        self.normalized_weights = [w / total for w in weights]

    def _gradient(self, x: torch.Tensor, order: int) -> torch.Tensor:
        """Compute temporal gradient of given order using finite differences."""
        for _ in range(order):
            x = x[..., 1:] - x[..., :-1]
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute temporal gradient loss."""
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        for order, weight in zip(self.orders, self.normalized_weights):
            pred_grad = self._gradient(pred, order)
            target_grad = self._gradient(target, order)
            grad_loss = F.l1_loss(pred_grad, target_grad)
            total_loss = total_loss + weight * grad_loss

        return total_loss

    def extra_repr(self) -> str:
        orders_str = ", ".join(f"d{o}={w:.2f}" for o, w in zip(self.orders, self.weights))
        return f"orders=[{orders_str}]"


def build_loss(loss_type: str, **kwargs) -> nn.Module:
    """Build a loss function by name.

    Args:
        loss_type: Type of loss ("temporal_gradient")
        **kwargs: Additional arguments for the loss

    Returns:
        Loss module
    """
    if loss_type == "temporal_gradient":
        return TemporalGradientLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: temporal_gradient")
