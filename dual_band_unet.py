"""Dual-Band U-Net for Neural Signal Translation.

This module implements a parallel two-branch U-Net architecture that decomposes
signals into low and high frequency bands, translates each with a specialized
U-Net, then reconstructs.

Architecture:
    OB_input -> [Decompose] -> OB_low  -> [UNet_low (4 downsample)]  -> PCx_low_pred  -> [Reconstruct] -> PCx_pred
                            -> OB_high -> [UNet_high (2 downsample)] -> PCx_high_pred ->

Key design decisions:
- Low band (1-30 Hz): Uses 4 downsample levels (Nyquist 31 Hz) for large receptive field
- High band (30-100 Hz): Uses 2 downsample levels (Nyquist 125 Hz) to preserve fast oscillations
- Butterworth filters for clean band separation (differentiable via scipy.signal.sosfilt)
- Both branches use odor conditioning similar to CondUNet1D
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt

# Import building blocks from models.py
from models import (
    ConvBlock,
    UpBlock,
    ModernConvBlock,
    ModernUpBlock,
    SelfAttention1D,
    CrossFrequencyCouplingAttention,
    CrossFrequencyCouplingAttentionV2,
    FiLM,
    BASE_CHANNELS,
    NUM_ODORS,
)


# =============================================================================
# Frequency Decomposition Module
# =============================================================================

class ButterworthDecomposition(nn.Module):
    """Differentiable frequency decomposition using Butterworth filters.

    Decomposes input signal into low and high frequency bands using
    zero-phase Butterworth filtering. The decomposition is applied
    as preprocessing (not learned).

    Args:
        cutoff_freq: Cutoff frequency in Hz (default: 30 Hz)
        sampling_rate: Sampling rate in Hz (default: 1000 Hz)
        order: Filter order (default: 4)
        low_band: Tuple of (low, high) for low band (default: (1, 30))
        high_band: Tuple of (low, high) for high band (default: (30, 100))

    Note:
        The decomposition is NOT differentiable (uses scipy sosfiltfilt).
        For end-to-end training, gradients flow through the U-Nets but
        not through the decomposition itself.
    """

    def __init__(
        self,
        cutoff_freq: float = 30.0,
        sampling_rate: float = 1000.0,
        order: int = 4,
        low_band: Tuple[float, float] = (1.0, 30.0),
        high_band: Tuple[float, float] = (30.0, 100.0),
    ):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order
        self.low_band = low_band
        self.high_band = high_band
        self.nyquist = sampling_rate / 2.0

        # Pre-compute filter coefficients (second-order sections for stability)
        # Low-pass filter for low band
        self.sos_low = butter(
            order,
            [low_band[0] / self.nyquist, low_band[1] / self.nyquist],
            btype='bandpass',
            output='sos'
        )

        # High-pass filter for high band
        self.sos_high = butter(
            order,
            [high_band[0] / self.nyquist, high_band[1] / self.nyquist],
            btype='bandpass',
            output='sos'
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose signal into low and high frequency bands.

        Args:
            x: Input signal [B, C, T]

        Returns:
            Tuple of (x_low, x_high) each with shape [B, C, T]
        """
        device = x.device
        dtype = x.dtype
        B, C, T = x.shape

        # Move to CPU and convert to numpy for scipy filtering
        x_np = x.detach().float().cpu().numpy()

        # Apply zero-phase filtering
        x_low_np = np.zeros_like(x_np)
        x_high_np = np.zeros_like(x_np)

        for b in range(B):
            for c in range(C):
                x_low_np[b, c] = sosfiltfilt(self.sos_low, x_np[b, c])
                x_high_np[b, c] = sosfiltfilt(self.sos_high, x_np[b, c])

        # Convert back to torch tensors
        x_low = torch.from_numpy(x_low_np).to(dtype=dtype, device=device)
        x_high = torch.from_numpy(x_high_np).to(dtype=dtype, device=device)

        return x_low, x_high

    def extra_repr(self) -> str:
        return (
            f"cutoff={self.cutoff_freq}Hz, fs={self.sampling_rate}Hz, "
            f"order={self.order}, low={self.low_band}, high={self.high_band}"
        )


class DifferentiableButterworthDecomposition(nn.Module):
    """Differentiable frequency decomposition using FFT-based filtering.

    This is a differentiable alternative to ButterworthDecomposition that
    allows gradients to flow through the decomposition. Uses FFT-based
    frequency-domain filtering with smooth transitions.

    Args:
        cutoff_freq: Cutoff frequency in Hz (default: 30 Hz)
        sampling_rate: Sampling rate in Hz (default: 1000 Hz)
        transition_width: Width of transition band in Hz (default: 5 Hz)
        low_band: Tuple of (low, high) for low band (default: (1, 30))
        high_band: Tuple of (low, high) for high band (default: (30, 100))
    """

    def __init__(
        self,
        cutoff_freq: float = 30.0,
        sampling_rate: float = 1000.0,
        transition_width: float = 5.0,
        low_band: Tuple[float, float] = (1.0, 30.0),
        high_band: Tuple[float, float] = (30.0, 100.0),
    ):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.transition_width = transition_width
        self.low_band = low_band
        self.high_band = high_band

        # Filter masks will be created on first forward pass based on signal length
        self._filter_masks_cached = False
        self._cached_T = None
        self.register_buffer('low_mask', None)
        self.register_buffer('high_mask', None)

    def _create_masks(self, T: int, device: torch.device):
        """Create frequency domain masks for filtering."""
        # Frequency bins
        freqs = torch.fft.rfftfreq(T, d=1.0/self.sampling_rate).to(device)

        # Create smooth bandpass masks using sigmoid transitions
        def smooth_bandpass(f, low, high, transition):
            """Create smooth bandpass filter using sigmoid transitions."""
            # Low edge: sigmoid transition from 0 to 1
            low_edge = torch.sigmoid((f - low) / (transition / 4))
            # High edge: sigmoid transition from 1 to 0
            high_edge = torch.sigmoid((high - f) / (transition / 4))
            return low_edge * high_edge

        self.low_mask = smooth_bandpass(
            freqs, self.low_band[0], self.low_band[1], self.transition_width
        )
        self.high_mask = smooth_bandpass(
            freqs, self.high_band[0], self.high_band[1], self.transition_width
        )

        self._filter_masks_cached = True
        self._cached_T = T

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose signal into low and high frequency bands.

        Args:
            x: Input signal [B, C, T]

        Returns:
            Tuple of (x_low, x_high) each with shape [B, C, T]
        """
        B, C, T = x.shape

        # Create or update masks if needed
        if not self._filter_masks_cached or self._cached_T != T:
            self._create_masks(T, x.device)

        # FFT
        x_fft = torch.fft.rfft(x.float(), dim=-1)

        # Apply masks
        x_low_fft = x_fft * self.low_mask.unsqueeze(0).unsqueeze(0)
        x_high_fft = x_fft * self.high_mask.unsqueeze(0).unsqueeze(0)

        # Inverse FFT
        x_low = torch.fft.irfft(x_low_fft, n=T, dim=-1).to(x.dtype)
        x_high = torch.fft.irfft(x_high_fft, n=T, dim=-1).to(x.dtype)

        return x_low, x_high

    def extra_repr(self) -> str:
        return (
            f"cutoff={self.cutoff_freq}Hz, fs={self.sampling_rate}Hz, "
            f"transition={self.transition_width}Hz, low={self.low_band}, high={self.high_band}"
        )


# =============================================================================
# Single-Band U-Net Branch
# =============================================================================

class UNetBranch(nn.Module):
    """Single U-Net branch for one frequency band.

    This is a configurable U-Net that can be used for either low or high
    frequency bands with different depths.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base channel count (doubled at each level)
        n_downsample: Number of downsample levels (determines Nyquist frequency)
        n_odors: Number of odor classes for conditioning
        emb_dim: Embedding dimension for conditioning
        dropout: Dropout rate
        use_attention: Whether to use attention in bottleneck
        attention_type: Type of attention ("basic", "cross_freq", "cross_freq_v2")
        norm_type: Normalization type ("batch", "instance", "group")
        cond_mode: Conditioning mode ("cross_attn_gated", "none", etc.)
        conv_type: Convolution type ("standard", "modern")
        use_se: Use squeeze-excitation in modern convs
        conv_kernel_size: Kernel size for modern convs
        dilations: Dilation rates for modern convs
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        base_channels: int = 64,
        n_downsample: int = 2,
        n_odors: int = NUM_ODORS,
        emb_dim: int = 128,
        dropout: float = 0.0,
        use_attention: bool = True,
        attention_type: str = "cross_freq_v2",
        norm_type: str = "batch",
        cond_mode: str = "cross_attn_gated",
        conv_type: str = "modern",
        use_se: bool = True,
        conv_kernel_size: int = 7,
        dilations: Tuple[int, ...] = (1, 4, 16, 32),
    ):
        super().__init__()
        self.n_downsample = n_downsample
        self.cond_mode = cond_mode
        self.emb_dim = emb_dim
        self.attention_type = attention_type
        self.conv_type = conv_type

        # Odor embedding (shared across branches in DualBandUNet)
        if cond_mode != "none":
            self.embed = nn.Embedding(n_odors, emb_dim)
        else:
            self.embed = None

        # Channel progression: base -> base*2 -> base*4 -> ... up to base*8 max
        channels = [base_channels]
        for i in range(n_downsample):
            channels.append(min(base_channels * (2 ** (i + 1)), base_channels * 8))

        # Build encoder and decoder
        if conv_type == "modern":
            conv_kwargs = dict(
                use_se=use_se,
                kernel_size=conv_kernel_size,
                dilations=dilations,
            )
            self.inc = ModernConvBlock(
                in_channels, channels[0], emb_dim,
                dropout=dropout, norm_type=norm_type, cond_mode=cond_mode,
                **conv_kwargs
            )

            self.encoders = nn.ModuleList()
            for i in range(n_downsample):
                self.encoders.append(
                    ModernConvBlock(
                        channels[i], channels[i+1], emb_dim,
                        downsample=True, dropout=dropout,
                        norm_type=norm_type, cond_mode=cond_mode,
                        **conv_kwargs
                    )
                )

            self.decoders = nn.ModuleList()
            for i in range(n_downsample):
                dec_in = channels[n_downsample - i]
                skip_in = channels[n_downsample - i - 1]
                dec_out = channels[n_downsample - i - 1]
                self.decoders.append(
                    ModernUpBlock(
                        dec_in, skip_in, dec_out, emb_dim,
                        dropout=dropout, norm_type=norm_type, cond_mode=cond_mode,
                        **conv_kwargs
                    )
                )
        else:
            # Standard convolutions
            self.inc = ConvBlock(
                in_channels, channels[0], emb_dim,
                dropout=dropout, norm_type=norm_type, cond_mode=cond_mode
            )

            self.encoders = nn.ModuleList()
            for i in range(n_downsample):
                self.encoders.append(
                    ConvBlock(
                        channels[i], channels[i+1], emb_dim,
                        downsample=True, dropout=dropout,
                        norm_type=norm_type, cond_mode=cond_mode
                    )
                )

            self.decoders = nn.ModuleList()
            for i in range(n_downsample):
                dec_in = channels[n_downsample - i]
                skip_in = channels[n_downsample - i - 1]
                dec_out = channels[n_downsample - i - 1]
                self.decoders.append(
                    UpBlock(
                        dec_in, skip_in, dec_out, emb_dim,
                        dropout=dropout, norm_type=norm_type, cond_mode=cond_mode
                    )
                )

        # Bottleneck with attention
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

        # Output convolution
        self.outc = nn.Conv1d(channels[0], out_channels, kernel_size=3, padding=1)

        # Residual connection (if in/out channels match)
        self.use_residual = (in_channels == out_channels)
        if not self.use_residual:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # Initialize output to produce small values (residual dominates initially)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def _build_attention(self, channels: int, attention_type: str) -> nn.Module:
        """Build attention module based on type."""
        if attention_type == "basic":
            return SelfAttention1D(channels)
        elif attention_type == "cross_freq":
            return CrossFrequencyCouplingAttention(channels)
        elif attention_type == "cross_freq_v2":
            return CrossFrequencyCouplingAttentionV2(channels)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

    def forward(
        self,
        x: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input signal [B, C, T]
            odor_ids: Odor class indices [B] for conditioning
            cond_emb: External conditioning embedding [B, emb_dim] (optional)

        Returns:
            Output signal [B, C, T]
        """
        # Get conditioning embedding
        if cond_emb is not None:
            emb = cond_emb
        elif self.embed is not None and odor_ids is not None:
            emb = self.embed(odor_ids)
        else:
            emb = None

        # Encoder path
        skips = [self.inc(x, emb)]
        for encoder in self.encoders:
            skips.append(encoder(skips[-1], emb))

        # Bottleneck
        h = self.mid(skips[-1])

        # Decoder path
        for i, decoder in enumerate(self.decoders):
            skip_idx = self.n_downsample - i - 1
            h = decoder(h, skips[skip_idx], emb)

        # Output
        delta = self.outc(h)

        # Residual connection
        if self.use_residual:
            out = x + delta
        else:
            out = self.residual_conv(x) + delta

        return out


# =============================================================================
# Dual-Band U-Net (Main Model)
# =============================================================================

class DualBandUNet(nn.Module):
    """Dual-Band U-Net for neural signal translation.

    Decomposes signals into low (1-30 Hz) and high (30-100 Hz) frequency bands,
    processes each with a specialized U-Net branch, then reconstructs.

    Architecture:
        OB -> [Decompose] -> OB_low  -> [UNet_low (4 downsample)]  -> PCx_low  -> [Sum] -> PCx_pred
                          -> OB_high -> [UNet_high (2 downsample)] -> PCx_high ->

    Args:
        in_channels: Number of input channels (default: 32)
        out_channels: Number of output channels (default: 32)
        base_channels: Base channel count for U-Nets (default: 64)
        n_odors: Number of odor classes for conditioning (default: 7)
        emb_dim: Embedding dimension (default: 128)
        cutoff_freq: Frequency cutoff between bands in Hz (default: 30)
        sampling_rate: Sampling rate in Hz (default: 1000)
        low_band: Low frequency band (low_hz, high_hz) (default: (1, 30))
        high_band: High frequency band (low_hz, high_hz) (default: (30, 100))
        n_downsample_low: Downsample levels for low-freq branch (default: 4)
        n_downsample_high: Downsample levels for high-freq branch (default: 2)
        use_differentiable_decomp: Use FFT-based decomposition (default: True)
        dropout: Dropout rate (default: 0.0)
        use_attention: Use attention in bottleneck (default: True)
        attention_type: Attention type (default: "cross_freq_v2")
        norm_type: Normalization type (default: "batch")
        cond_mode: Conditioning mode (default: "cross_attn_gated")
        conv_type: Convolution type (default: "modern")
        share_embedding: Share odor embedding between branches (default: True)
        reconstruction_mode: How to combine bands ("sum", "weighted", "learned")
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        base_channels: int = 64,
        n_odors: int = NUM_ODORS,
        emb_dim: int = 128,
        cutoff_freq: float = 30.0,
        sampling_rate: float = 1000.0,
        low_band: Tuple[float, float] = (1.0, 30.0),
        high_band: Tuple[float, float] = (30.0, 100.0),
        n_downsample_low: int = 4,
        n_downsample_high: int = 2,
        use_differentiable_decomp: bool = True,
        dropout: float = 0.0,
        use_attention: bool = True,
        attention_type: str = "cross_freq_v2",
        norm_type: str = "batch",
        cond_mode: str = "cross_attn_gated",
        conv_type: str = "modern",
        share_embedding: bool = True,
        reconstruction_mode: str = "sum",  # "sum", "weighted", "learned"
        use_se: bool = True,
        conv_kernel_size: int = 7,
        dilations: Tuple[int, ...] = (1, 4, 16, 32),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.low_band = low_band
        self.high_band = high_band
        self.n_downsample_low = n_downsample_low
        self.n_downsample_high = n_downsample_high
        self.reconstruction_mode = reconstruction_mode
        self.share_embedding = share_embedding

        # Frequency decomposition
        if use_differentiable_decomp:
            self.decompose = DifferentiableButterworthDecomposition(
                cutoff_freq=cutoff_freq,
                sampling_rate=sampling_rate,
                low_band=low_band,
                high_band=high_band,
            )
        else:
            self.decompose = ButterworthDecomposition(
                cutoff_freq=cutoff_freq,
                sampling_rate=sampling_rate,
                low_band=low_band,
                high_band=high_band,
            )

        # Shared embedding (if enabled)
        if share_embedding and cond_mode != "none":
            self.shared_embed = nn.Embedding(n_odors, emb_dim)
        else:
            self.shared_embed = None

        # Common kwargs for both branches
        branch_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            n_odors=n_odors,
            emb_dim=emb_dim,
            dropout=dropout,
            use_attention=use_attention,
            attention_type=attention_type,
            norm_type=norm_type,
            cond_mode=cond_mode if not share_embedding else "none",  # Disable internal embedding if shared
            conv_type=conv_type,
            use_se=use_se,
            conv_kernel_size=conv_kernel_size,
            dilations=dilations,
        )

        # Low-frequency branch (4 downsample -> Nyquist 31 Hz)
        self.unet_low = UNetBranch(
            n_downsample=n_downsample_low,
            **branch_kwargs
        )

        # High-frequency branch (2 downsample -> Nyquist 125 Hz)
        self.unet_high = UNetBranch(
            n_downsample=n_downsample_high,
            **branch_kwargs
        )

        # Reconstruction weights (for weighted/learned modes)
        if reconstruction_mode == "weighted":
            # Fixed weights (can be tuned as hyperparameters)
            self.register_buffer('weight_low', torch.tensor(1.0))
            self.register_buffer('weight_high', torch.tensor(1.0))
        elif reconstruction_mode == "learned":
            # Learnable per-channel weights
            self.weight_low = nn.Parameter(torch.ones(1, out_channels, 1))
            self.weight_high = nn.Parameter(torch.ones(1, out_channels, 1))

        # Print architecture info
        nyquist_low = sampling_rate / (2 ** (n_downsample_low + 1))
        nyquist_high = sampling_rate / (2 ** (n_downsample_high + 1))
        print(f"DualBandUNet: low_band={low_band}, high_band={high_band}")
        print(f"  Low branch: {n_downsample_low} downsample -> bottleneck Nyquist = {nyquist_low:.1f} Hz")
        print(f"  High branch: {n_downsample_high} downsample -> bottleneck Nyquist = {nyquist_high:.1f} Hz")

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
        cond_emb: Optional[torch.Tensor] = None,
        return_bands: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            ob: Input OB signal [B, C, T]
            odor_ids: Odor class indices [B] for conditioning
            cond_emb: External conditioning embedding [B, emb_dim] (optional)
            return_bands: If True, also return individual band predictions

        Returns:
            If return_bands=False:
                pcx_pred: Reconstructed prediction [B, C, T]
            If return_bands=True:
                Tuple of (pcx_pred, pcx_low_pred, pcx_high_pred)
        """
        # Get conditioning embedding
        if self.shared_embed is not None and odor_ids is not None:
            emb = self.shared_embed(odor_ids)
        elif cond_emb is not None:
            emb = cond_emb
        else:
            emb = None

        # Decompose input into frequency bands
        ob_low, ob_high = self.decompose(ob)

        # Process each band with its specialized U-Net
        pcx_low_pred = self.unet_low(ob_low, cond_emb=emb)
        pcx_high_pred = self.unet_high(ob_high, cond_emb=emb)

        # Reconstruct output
        if self.reconstruction_mode == "sum":
            pcx_pred = pcx_low_pred + pcx_high_pred
        elif self.reconstruction_mode == "weighted":
            pcx_pred = self.weight_low * pcx_low_pred + self.weight_high * pcx_high_pred
        elif self.reconstruction_mode == "learned":
            pcx_pred = self.weight_low * pcx_low_pred + self.weight_high * pcx_high_pred
        else:
            raise ValueError(f"Unknown reconstruction_mode: {self.reconstruction_mode}")

        if return_bands:
            return pcx_pred, pcx_low_pred, pcx_high_pred
        return pcx_pred

    def decompose_target(self, pcx_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose target signal for band-wise loss computation.

        Args:
            pcx_target: Target PCx signal [B, C, T]

        Returns:
            Tuple of (pcx_target_low, pcx_target_high) each [B, C, T]
        """
        return self.decompose(pcx_target)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"low_band={self.low_band}, high_band={self.high_band}, "
            f"n_downsample_low={self.n_downsample_low}, n_downsample_high={self.n_downsample_high}, "
            f"reconstruction={self.reconstruction_mode}"
        )


# =============================================================================
# Dual-Band Loss Function
# =============================================================================

class DualBandLoss(nn.Module):
    """Combined loss function for dual-band U-Net training.

    Computes:
        total_loss = loss(pred, target)
                   + lambda_low * loss(pred_low, target_low)
                   + lambda_high * loss(pred_high, target_high)

    Args:
        base_loss: Base loss function (e.g., nn.MSELoss, nn.L1Loss)
        lambda_low: Weight for low-band loss (default: 0.5)
        lambda_high: Weight for high-band loss (default: 0.5)
        use_spectral_loss: Include spectral loss component (default: True)
        spectral_weight: Weight for spectral loss (default: 1.0)
    """

    def __init__(
        self,
        base_loss: nn.Module = None,
        lambda_low: float = 0.5,
        lambda_high: float = 0.5,
        use_spectral_loss: bool = True,
        spectral_weight: float = 1.0,
    ):
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else nn.L1Loss()
        self.lambda_low = lambda_low
        self.lambda_high = lambda_high
        self.use_spectral_loss = use_spectral_loss
        self.spectral_weight = spectral_weight

    def _spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss (log-magnitude MSE in frequency domain)."""
        pred_fft = torch.fft.rfft(pred.float(), dim=-1)
        target_fft = torch.fft.rfft(target.float(), dim=-1)

        # Log magnitude (add epsilon for stability)
        pred_mag = torch.log(torch.abs(pred_fft) + 1e-8)
        target_mag = torch.log(torch.abs(target_fft) + 1e-8)

        return F.mse_loss(pred_mag, target_mag)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_low: torch.Tensor,
        target_low: torch.Tensor,
        pred_high: torch.Tensor,
        target_high: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined dual-band loss.

        Args:
            pred: Full reconstructed prediction [B, C, T]
            target: Full target [B, C, T]
            pred_low: Low-band prediction [B, C, T]
            target_low: Low-band target [B, C, T]
            pred_high: High-band prediction [B, C, T]
            target_high: High-band target [B, C, T]

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual components
        """
        # Base temporal losses
        loss_full = self.base_loss(pred, target)
        loss_low = self.base_loss(pred_low, target_low)
        loss_high = self.base_loss(pred_high, target_high)

        # Combined temporal loss
        total_loss = loss_full + self.lambda_low * loss_low + self.lambda_high * loss_high

        # Spectral losses (optional)
        if self.use_spectral_loss:
            spec_loss_full = self._spectral_loss(pred, target)
            spec_loss_low = self._spectral_loss(pred_low, target_low)
            spec_loss_high = self._spectral_loss(pred_high, target_high)

            spec_total = spec_loss_full + self.lambda_low * spec_loss_low + self.lambda_high * spec_loss_high
            total_loss = total_loss + self.spectral_weight * spec_total
        else:
            spec_loss_full = torch.tensor(0.0)
            spec_loss_low = torch.tensor(0.0)
            spec_loss_high = torch.tensor(0.0)

        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_full': loss_full.item(),
            'loss_low': loss_low.item(),
            'loss_high': loss_high.item(),
            'spec_loss_full': spec_loss_full.item() if self.use_spectral_loss else 0.0,
            'spec_loss_low': spec_loss_low.item() if self.use_spectral_loss else 0.0,
            'spec_loss_high': spec_loss_high.item() if self.use_spectral_loss else 0.0,
        }

        return total_loss, loss_dict

    def extra_repr(self) -> str:
        return f"lambda_low={self.lambda_low}, lambda_high={self.lambda_high}, spectral={self.use_spectral_loss}"


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_dual_band_unet():
    """Test the DualBandUNet implementation."""
    print("Testing DualBandUNet...")

    # Create model
    model = DualBandUNet(
        in_channels=32,
        out_channels=32,
        base_channels=64,
        n_odors=7,
        n_downsample_low=4,
        n_downsample_high=2,
        use_differentiable_decomp=True,
        conv_type="modern",
    )

    # Count parameters
    n_params = count_parameters(model)
    n_params_low = count_parameters(model.unet_low)
    n_params_high = count_parameters(model.unet_high)
    print(f"Total parameters: {n_params:,}")
    print(f"  Low branch: {n_params_low:,}")
    print(f"  High branch: {n_params_high:,}")

    # Test forward pass
    B, C, T = 4, 32, 5000
    ob = torch.randn(B, C, T)
    odor_ids = torch.randint(0, 7, (B,))

    # Test without return_bands
    pcx_pred = model(ob, odor_ids)
    print(f"Input shape: {ob.shape}")
    print(f"Output shape: {pcx_pred.shape}")

    # Test with return_bands
    pcx_pred, pcx_low, pcx_high = model(ob, odor_ids, return_bands=True)
    print(f"Low band output shape: {pcx_low.shape}")
    print(f"High band output shape: {pcx_high.shape}")

    # Test decompose_target
    target = torch.randn(B, C, T)
    target_low, target_high = model.decompose_target(target)
    print(f"Target low shape: {target_low.shape}")
    print(f"Target high shape: {target_high.shape}")

    # Test loss
    loss_fn = DualBandLoss(lambda_low=0.5, lambda_high=0.5, use_spectral_loss=True)
    total_loss, loss_dict = loss_fn(pcx_pred, target, pcx_low, target_low, pcx_high, target_high)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss components: {loss_dict}")

    print("\nDualBandUNet test passed!")


if __name__ == "__main__":
    test_dual_band_unet()
