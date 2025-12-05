"""
Frequency-Aware Disentangled Conditioning
=========================================

Dual-stream conditioning that separately encodes low-frequency state
and high-frequency dynamics from the input OB signal.

Key insight: Neural signals have distinct information in different
frequency bands:
- Low frequencies (<10Hz): Slow state changes, respiratory rhythm
- High frequencies (>30Hz): Fast oscillatory dynamics, gamma bursts

By processing these separately, we can capture both slow state and
fast dynamics in the conditioning representation.

Search parameters:
- low_freq_cutoff: Cutoff between low/mid frequencies
- high_freq_cutoff: Cutoff between mid/high frequencies
- stream_balance: Relative weight of low vs high stream
- use_cross_stream_attention: Whether streams attend to each other
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandpassFilter(nn.Module):
    """Differentiable bandpass filter using FFT.

    Args:
        sample_rate: Sampling rate in Hz
        low_freq: Low cutoff frequency
        high_freq: High cutoff frequency
        transition_width: Width of transition bands
    """

    def __init__(
        self,
        sample_rate: float = 1000.0,
        low_freq: float = 0.0,
        high_freq: float = 100.0,
        transition_width: float = 2.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.transition_width = transition_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bandpass filter.

        Args:
            x: Input signal [B, C, T]

        Returns:
            Filtered signal [B, C, T]
        """
        B, C, T = x.shape

        # FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(T, d=1/self.sample_rate).to(x.device)

        # Build soft filter mask
        if self.low_freq > 0:
            low_mask = torch.sigmoid(
                (freqs - self.low_freq) / self.transition_width
            )
        else:
            low_mask = torch.ones_like(freqs)

        high_mask = torch.sigmoid(
            (self.high_freq - freqs) / self.transition_width
        )

        filter_mask = low_mask * high_mask
        filter_mask = filter_mask.view(1, 1, -1)

        # Apply filter
        x_fft_filtered = x_fft * filter_mask

        # Inverse FFT
        return torch.fft.irfft(x_fft_filtered, n=T, dim=-1)


class FrequencyStreamEncoder(nn.Module):
    """Encoder for a specific frequency stream.

    Args:
        in_channels: Number of input channels
        out_dim: Output dimension
        kernel_size: Convolution kernel size
        n_layers: Number of conv layers
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_dim: int = 64,
        kernel_size: int = 15,
        n_layers: int = 2,
    ):
        super().__init__()

        layers = []
        current_channels = in_channels

        for i in range(n_layers):
            out_channels = out_dim if i == n_layers - 1 else in_channels * 2
            stride = 4 if i == 0 else 2
            layers.extend([
                nn.Conv1d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                ),
                nn.GELU(),
                nn.BatchNorm1d(out_channels),
            ])
            current_channels = out_channels

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to fixed-size representation.

        Args:
            x: Input signal [B, C, T]

        Returns:
            Encoding [B, out_dim]
        """
        h = self.encoder(x)
        return self.pool(h).squeeze(-1)


class CrossStreamAttention(nn.Module):
    """Cross-attention between frequency streams.

    Args:
        dim: Feature dimension
        n_heads: Number of attention heads
    """

    def __init__(self, dim: int = 64, n_heads: int = 4):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-attention.

        Args:
            query: Query features [B, dim]
            key_value: Key/value features [B, dim]

        Returns:
            Attended features [B, dim]
        """
        B = query.shape[0]

        # Project
        q = self.q_proj(query).view(B, self.n_heads, self.head_dim)
        k = self.k_proj(key_value).view(B, self.n_heads, self.head_dim)
        v = self.v_proj(key_value).view(B, self.n_heads, self.head_dim)

        # Attention (single position, so simplified)
        attn = (q * k).sum(dim=-1, keepdim=True) * self.scale
        attn = F.softmax(attn, dim=1)  # Over heads

        # Apply attention
        out = (attn * v).sum(dim=1)  # [B, head_dim]

        # Expand and project
        out = out.repeat(1, self.n_heads)  # [B, dim]
        return self.out_proj(out)


class FreqDisentangledConditioner(nn.Module):
    """Frequency-disentangled conditioning module.

    Processes low and high frequency content separately, then combines
    for final conditioning embedding.

    Args:
        in_channels: Number of input channels
        emb_dim: Final embedding dimension
        low_cutoff: Cutoff frequency for low-pass
        high_cutoff: Cutoff frequency for high-pass
        stream_balance: Weight for low stream (1 - this for high)
        use_cross_attention: Whether streams attend to each other
        n_odors: Number of odor classes
        use_odor: Whether to include odor information
    """

    def __init__(
        self,
        in_channels: int = 32,
        emb_dim: int = 128,
        low_cutoff: float = 8.0,
        high_cutoff: float = 30.0,
        stream_balance: float = 0.5,
        use_cross_attention: bool = True,
        n_odors: int = 7,
        use_odor: bool = True,
        sample_rate: float = 1000.0,
    ):
        super().__init__()

        self.stream_balance = stream_balance
        self.use_cross_attention = use_cross_attention
        self.use_odor = use_odor

        stream_dim = emb_dim // 2

        # Bandpass filters
        self.low_filter = BandpassFilter(
            sample_rate=sample_rate,
            low_freq=0.0,
            high_freq=low_cutoff,
        )
        self.high_filter = BandpassFilter(
            sample_rate=sample_rate,
            low_freq=high_cutoff,
            high_freq=sample_rate / 2 - 1,
        )

        # Stream encoders
        # Low freq: larger kernel (slow patterns)
        self.low_encoder = FrequencyStreamEncoder(
            in_channels=in_channels,
            out_dim=stream_dim,
            kernel_size=21,  # Larger for slow patterns
            n_layers=2,
        )
        # High freq: smaller kernel (fast patterns)
        self.high_encoder = FrequencyStreamEncoder(
            in_channels=in_channels,
            out_dim=stream_dim,
            kernel_size=9,  # Smaller for fast patterns
            n_layers=3,
        )

        # Cross-stream attention (optional)
        if use_cross_attention:
            self.low_to_high_attn = CrossStreamAttention(stream_dim)
            self.high_to_low_attn = CrossStreamAttention(stream_dim)

        # Combination network
        combine_in_dim = stream_dim * 2
        if use_odor:
            self.odor_embed = nn.Embedding(n_odors, stream_dim)
            combine_in_dim += stream_dim

        self.combine = nn.Sequential(
            nn.Linear(combine_in_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Learnable stream balance
        self.balance_logit = nn.Parameter(
            torch.tensor(math.log(stream_balance / (1 - stream_balance + 1e-8)))
        )

    @property
    def learned_balance(self) -> torch.Tensor:
        """Current stream balance."""
        return torch.sigmoid(self.balance_logit)

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute frequency-disentangled conditioning.

        Args:
            ob: OB signal [B, C, T]
            odor_ids: Odor class indices [B]

        Returns:
            (embedding, aux_loss): [B, emb_dim], scalar (0 for this module)
        """
        # Filter into frequency bands
        low_signal = self.low_filter(ob)
        high_signal = self.high_filter(ob)

        # Encode each stream
        low_emb = self.low_encoder(low_signal)  # [B, stream_dim]
        high_emb = self.high_encoder(high_signal)  # [B, stream_dim]

        # Cross-stream attention (optional)
        if self.use_cross_attention:
            low_enhanced = low_emb + self.high_to_low_attn(low_emb, high_emb)
            high_enhanced = high_emb + self.low_to_high_attn(high_emb, low_emb)
            low_emb = low_enhanced
            high_emb = high_enhanced

        # Weighted combination
        balance = self.learned_balance
        combined_streams = torch.cat([
            balance * low_emb,
            (1 - balance) * high_emb,
        ], dim=-1)

        # Add odor if enabled
        if self.use_odor and odor_ids is not None:
            odor_emb = self.odor_embed(odor_ids)
            combined_streams = torch.cat([combined_streams, odor_emb], dim=-1)

        # Final embedding
        emb = self.combine(combined_streams)

        # No auxiliary loss for this module
        aux_loss = torch.tensor(0.0, device=ob.device)

        return emb, aux_loss


def create_freq_disentangled_conditioner(
    config: Dict,
    in_channels: int = 32,
    emb_dim: int = 128,
    n_odors: int = 7,
    sample_rate: float = 1000.0,
) -> FreqDisentangledConditioner:
    """Factory function to create frequency-disentangled conditioner.

    Args:
        config: Configuration dictionary
        in_channels: Number of input channels
        emb_dim: Embedding dimension
        n_odors: Number of odor classes
        sample_rate: Sampling rate

    Returns:
        Configured FreqDisentangledConditioner
    """
    return FreqDisentangledConditioner(
        in_channels=in_channels,
        emb_dim=emb_dim,
        low_cutoff=config.get("low_freq_cutoff", 8.0),
        high_cutoff=config.get("high_freq_cutoff", 30.0),
        stream_balance=config.get("stream_balance", 0.5),
        use_cross_attention=config.get("use_cross_stream_attention", True),
        n_odors=n_odors,
        use_odor=config.get("use_odor", True),
        sample_rate=sample_rate,
    )
