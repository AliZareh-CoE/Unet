"""Modular neural signal translation models.

This module provides a configurable U-Net architecture for translating
neural signals between brain regions. The model adapts to any input/output
channel configuration.

Key components:
- UNet1D: Core U-Net architecture for signal translation
- ConvBlock/UpBlock: Encoder/decoder building blocks
- Attention modules: Self-attention and cross-frequency coupling
- Loss functions: L1, MSE, Wavelet-based losses

Usage:
    from models import UNet1D
    from config import ModelConfig

    cfg = ModelConfig(in_channels=32, out_channels=64, base_channels=64)
    model = UNet1D.from_config(cfg)

    output = model(input_signal)  # (batch, 32, samples) -> (batch, 64, samples)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# =============================================================================
# Attention Modules
# =============================================================================

class SelfAttention1D(nn.Module):
    """1D Self-Attention for capturing long-range temporal dependencies."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        reduced = max(channels // reduction, 8)
        self.query = nn.Conv1d(channels, reduced, 1)
        self.key = nn.Conv1d(channels, reduced, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        q = self.query(x).permute(0, 2, 1)  # (B, T, C')
        k = self.key(x)  # (B, C', T)
        v = self.value(x).permute(0, 2, 1)  # (B, T, C)

        attn = torch.bmm(q, k)  # (B, T, T)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1)  # (B, C, T)

        return x + self.gamma * out


class CrossFrequencyAttention(nn.Module):
    """Cross-frequency coupling attention for neural signals.

    Models interactions between different frequency bands (e.g., theta-gamma
    coupling) which are important in neural signal processing.
    """

    def __init__(
        self,
        channels: int,
        n_heads: int = 4,
        sample_rate: float = 1000.0,
        bands: Optional[List[Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.sample_rate = sample_rate
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Default frequency bands
        if bands is None:
            bands = [(1, 4), (4, 8), (8, 12), (12, 30), (30, 80)]
        self.bands = bands
        self.n_bands = len(bands)

        # Projections (grouped conv for efficiency)
        self.proj_q = nn.Conv1d(channels * self.n_bands, channels * self.n_bands, 1, groups=self.n_bands)
        self.proj_k = nn.Conv1d(channels * self.n_bands, channels * self.n_bands, 1, groups=self.n_bands)
        self.proj_v = nn.Conv1d(channels * self.n_bands, channels * self.n_bands, 1, groups=self.n_bands)
        self.out_proj = nn.Conv1d(channels * self.n_bands, channels, 1)

        # Learnable coupling weights between bands
        self.coupling = nn.Parameter(torch.ones(self.n_bands, self.n_bands) / self.n_bands)
        self.gate = nn.Parameter(torch.zeros(1))

        self._filter_cache = None

    def _build_filters(self, T: int, device: torch.device) -> torch.Tensor:
        """Build bandpass filter masks."""
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sample_rate)
        masks = []
        for low, high in self.bands:
            mask = torch.sigmoid((freqs - low) / 2) * torch.sigmoid((high - freqs) / 2)
            masks.append(mask)
        return torch.stack(masks, dim=0).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        device = x.device

        # Extract frequency bands via FFT
        x_fft = torch.fft.rfft(x.float(), dim=-1)
        filters = self._build_filters(T, device)

        # Apply filters to get band signals
        x_fft_exp = x_fft.unsqueeze(1)  # (B, 1, C, freq)
        filters_exp = filters.unsqueeze(0).unsqueeze(2)  # (1, bands, 1, freq)
        band_fft = x_fft_exp * filters_exp  # (B, bands, C, freq)
        band_fft_flat = band_fft.reshape(B * self.n_bands, C, -1)
        band_signals = torch.fft.irfft(band_fft_flat, n=T, dim=-1)  # (B*bands, C, T)
        band_signals = band_signals.reshape(B, self.n_bands * C, T).to(x.dtype)

        # Q, K, V projections
        Q = self.proj_q(band_signals).reshape(B, self.n_bands, C, T)
        K = self.proj_k(band_signals).reshape(B, self.n_bands, C, T)
        V = self.proj_v(band_signals).reshape(B, self.n_bands, C, T)

        # Cross-band attention with coupling weights
        coupling = F.softmax(self.coupling, dim=-1)
        out_bands = []
        for i in range(self.n_bands):
            q_i = Q[:, i]  # (B, C, T)
            attended = torch.zeros_like(q_i)
            for j in range(self.n_bands):
                k_j = K[:, j]
                v_j = V[:, j]
                # Simplified attention
                attn = torch.bmm(q_i.transpose(1, 2), k_j) * self.scale  # (B, T, T)
                attn = F.softmax(attn, dim=-1)
                out = torch.bmm(attn, v_j.transpose(1, 2)).transpose(1, 2)  # (B, C, T)
                attended = attended + coupling[i, j] * out
            out_bands.append(attended)

        # Combine bands
        combined = torch.cat(out_bands, dim=1)  # (B, bands*C, T)
        out = self.out_proj(combined)

        return x + self.gate * out


# =============================================================================
# Convolution Blocks
# =============================================================================

class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.GELU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _ = x.shape
        y = self.pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1)
        return x * y


class MultiScaleConv(nn.Module):
    """Multi-scale dilated convolution for capturing different frequency bands."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dilations: Tuple[int, ...] = (1, 4, 16, 32),
    ):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=(kernel_size - 1) * d // 2, dilation=d,
            )
            for d in dilations
        ])
        self.weights = nn.Parameter(torch.ones(len(dilations)) / len(dilations))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.weights, dim=0)
        outputs = torch.stack([b(x) for b in self.branches], dim=0)
        return torch.einsum('n,nbct->bct', w, outputs)


class ConvBlock(nn.Module):
    """Encoder convolution block with optional downsampling."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        downsample: bool = False,
        dropout: float = 0.0,
        norm_type: str = "batch",
        use_se: bool = True,
        use_multiscale: bool = True,
        kernel_size: int = 7,
        dilations: Tuple[int, ...] = (1, 4, 16, 32),
    ):
        super().__init__()
        stride = 2 if downsample else 1

        # Normalization
        if norm_type == "batch":
            norm = nn.BatchNorm1d(out_c)
        elif norm_type == "instance":
            norm = nn.InstanceNorm1d(out_c, affine=True)
        elif norm_type == "group":
            num_groups = min(8, out_c)
            while out_c % num_groups != 0:
                num_groups -= 1
            norm = nn.GroupNorm(num_groups, out_c)
        else:
            norm = nn.Identity()

        # Convolution
        if use_multiscale:
            conv = MultiScaleConv(in_c, out_c, kernel_size, stride, dilations)
        else:
            conv = nn.Conv1d(in_c, out_c, 3, stride=stride, padding=1)

        layers = [conv, norm, nn.GELU()]
        if use_se:
            layers.append(SqueezeExcite(out_c))
        if dropout > 0:
            layers.append(nn.Dropout1d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Decoder block with upsampling and skip connection."""

    def __init__(
        self,
        in_c: int,
        skip_c: int,
        out_c: int,
        dropout: float = 0.0,
        norm_type: str = "batch",
        use_se: bool = True,
        use_multiscale: bool = True,
        kernel_size: int = 7,
        dilations: Tuple[int, ...] = (1, 4, 16, 32),
    ):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(
            out_c + skip_c, out_c,
            downsample=False, dropout=dropout, norm_type=norm_type,
            use_se=use_se, use_multiscale=use_multiscale,
            kernel_size=kernel_size, dilations=dilations,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                skip = skip[..., diff // 2: diff // 2 + x.shape[-1]]
            else:
                x = x[..., -diff // 2: -diff // 2 + skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# =============================================================================
# U-Net Model
# =============================================================================

class UNet1D(nn.Module):
    """1D U-Net for neural signal translation.

    Translates signals from one brain region to another. Architecture
    adapts to any input/output channel configuration.

    Args:
        in_channels: Number of input channels (source region)
        out_channels: Number of output channels (target region)
        base_channels: Base feature channels (doubled at each level)
        n_downsample: Number of downsampling levels
        dropout: Dropout probability
        use_attention: Whether to use attention in bottleneck
        attention_type: Type of attention ("self", "cross_freq")
        norm_type: Normalization type ("batch", "instance", "group")
        use_se: Use squeeze-excitation attention
        use_multiscale: Use multi-scale dilated convolutions
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        base_channels: int = 64,
        n_downsample: int = 4,
        dropout: float = 0.0,
        use_attention: bool = True,
        attention_type: str = "cross_freq",
        norm_type: str = "batch",
        use_se: bool = True,
        use_multiscale: bool = True,
        kernel_size: int = 7,
        dilations: Tuple[int, ...] = (1, 4, 16, 32),
        sample_rate: float = 1000.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_downsample = n_downsample

        # Build channel progression
        channels = [base_channels * (2 ** i) for i in range(n_downsample + 1)]

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, channels[0], kernel_size=7, padding=3)

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(n_downsample):
            self.encoders.append(ConvBlock(
                channels[i], channels[i + 1],
                downsample=True, dropout=dropout, norm_type=norm_type,
                use_se=use_se, use_multiscale=use_multiscale,
                kernel_size=kernel_size, dilations=dilations,
            ))

        # Bottleneck
        bottleneck_c = channels[-1]
        self.bottleneck = ConvBlock(
            bottleneck_c, bottleneck_c,
            downsample=False, dropout=dropout, norm_type=norm_type,
            use_se=use_se, use_multiscale=use_multiscale,
            kernel_size=kernel_size, dilations=dilations,
        )

        # Attention
        self.use_attention = use_attention
        if use_attention:
            if attention_type == "cross_freq":
                self.attention = CrossFrequencyAttention(
                    bottleneck_c, n_heads=4, sample_rate=sample_rate,
                )
            else:
                self.attention = SelfAttention1D(bottleneck_c)

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(n_downsample - 1, -1, -1):
            self.decoders.append(UpBlock(
                channels[i + 1], channels[i], channels[i],
                dropout=dropout, norm_type=norm_type,
                use_se=use_se, use_multiscale=use_multiscale,
                kernel_size=kernel_size, dilations=dilations,
            ))

        # Output projection
        self.output_proj = nn.Conv1d(channels[0], out_channels, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, in_channels, samples)

        Returns:
            Output tensor (batch, out_channels, samples)
        """
        # Input projection
        x = self.input_proj(x)

        # Encoder with skip connections
        skips = [x]
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)
        if self.use_attention:
            x = self.attention(x)

        # Decoder with skip connections
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 2)]
            x = decoder(x, skip)

        # Output projection
        return self.output_proj(x)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "UNet1D":
        """Create model from configuration."""
        use_multiscale = cfg.conv_type == "modern"
        attention_type = "cross_freq" if "cross_freq" in cfg.attention_type else "self"

        return cls(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            base_channels=cfg.base_channels,
            n_downsample=cfg.n_downsample,
            dropout=cfg.dropout,
            use_attention=cfg.use_attention,
            attention_type=attention_type,
            norm_type=cfg.norm_type,
            use_se=cfg.use_se,
            use_multiscale=use_multiscale,
            kernel_size=cfg.conv_kernel_size,
            dilations=cfg.conv_dilations,
        )

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Loss Functions
# =============================================================================

class WaveletLoss(nn.Module):
    """Wavelet-based loss for time-frequency matching.

    Uses complex Morlet wavelets to compare signals in the time-frequency
    domain, which is important for neural signals with oscillatory content.
    """

    def __init__(
        self,
        sample_rate: float = 1000.0,
        freqs: Optional[List[float]] = None,
        omega0: float = 6.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.omega0 = omega0

        # Default: log-spaced frequencies covering LFP range
        if freqs is None:
            freqs = [2, 4, 8, 12, 20, 30, 50, 80]
        self.freqs = freqs
        self.n_freqs = len(freqs)

        # Pre-compute wavelets (will be created on first forward)
        self._wavelets = None
        self._wavelet_len = None

    def _build_wavelets(self, T: int, device: torch.device) -> torch.Tensor:
        """Build Morlet wavelets for each frequency."""
        if self._wavelets is not None and self._wavelet_len == T:
            return self._wavelets.to(device)

        wavelets = []
        t = torch.arange(T, dtype=torch.float32, device=device) / self.sample_rate
        t = t - t.mean()

        for f in self.freqs:
            sigma = self.omega0 / (2 * math.pi * f)
            gauss = torch.exp(-t ** 2 / (2 * sigma ** 2))
            wave = gauss * torch.exp(2j * math.pi * f * t)
            wave = wave / wave.abs().sum()  # Normalize
            wavelets.append(wave)

        self._wavelets = torch.stack(wavelets, dim=0)  # (n_freqs, T)
        self._wavelet_len = T
        return self._wavelets.to(device)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Compute wavelet transform.

        Args:
            x: Input (batch, channels, samples)

        Returns:
            Wavelet coefficients (batch, channels, n_freqs, samples)
        """
        B, C, T = x.shape
        wavelets = self._build_wavelets(T, x.device)

        # FFT-based convolution for efficiency
        x_fft = torch.fft.fft(x.float(), dim=-1)
        w_fft = torch.fft.fft(wavelets, dim=-1)

        # Broadcast: (B, C, 1, T) * (1, 1, n_freqs, T)
        coeffs_fft = x_fft.unsqueeze(2) * w_fft.unsqueeze(0).unsqueeze(0).conj()
        coeffs = torch.fft.ifft(coeffs_fft, dim=-1)

        return coeffs.abs().to(x.dtype)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute wavelet loss.

        Args:
            pred: Predicted signal (batch, channels, samples)
            target: Target signal (batch, channels, samples)

        Returns:
            Scalar loss value
        """
        pred_wt = self.transform(pred)
        target_wt = self.transform(target)
        return F.l1_loss(pred_wt, target_wt)


class CombinedLoss(nn.Module):
    """Combined loss with L1/MSE and optional wavelet components."""

    def __init__(
        self,
        loss_type: str = "l1",
        weight_l1: float = 1.0,
        weight_wavelet: float = 0.0,
        sample_rate: float = 1000.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.weight_l1 = weight_l1
        self.weight_wavelet = weight_wavelet

        # Primary loss
        if "huber" in loss_type:
            self.primary_loss = nn.HuberLoss()
        elif "mse" in loss_type:
            self.primary_loss = nn.MSELoss()
        else:
            self.primary_loss = nn.L1Loss()

        # Wavelet loss
        if "wavelet" in loss_type or weight_wavelet > 0:
            self.wavelet_loss = WaveletLoss(sample_rate=sample_rate)
        else:
            self.wavelet_loss = None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Returns:
            total_loss: Combined scalar loss
            components: Dictionary of individual loss components
        """
        # Primary loss
        primary = self.primary_loss(pred, target)
        total = self.weight_l1 * primary
        components = {"primary": primary.item()}

        # Wavelet loss
        if self.wavelet_loss is not None and self.weight_wavelet > 0:
            wavelet = self.wavelet_loss(pred, target)
            total = total + self.weight_wavelet * wavelet
            components["wavelet"] = wavelet.item()

        components["total"] = total.item()
        return total, components


# =============================================================================
# Metrics
# =============================================================================

def pearson_corr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Pearson correlation between predictions and targets.

    Args:
        pred: Predicted signals (batch, channels, samples)
        target: Target signals (batch, channels, samples)

    Returns:
        Mean correlation across batch and channels
    """
    # Flatten to (batch * channels, samples)
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])

    # Center
    pred_centered = pred_flat - pred_flat.mean(dim=-1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=-1, keepdim=True)

    # Correlation
    num = (pred_centered * target_centered).sum(dim=-1)
    denom = pred_centered.norm(dim=-1) * target_centered.norm(dim=-1)
    corr = num / denom.clamp(min=1e-8)

    return corr.mean()


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute explained variance score.

    EV = 1 - Var(target - pred) / Var(target)
    """
    residual = target - pred
    var_residual = residual.var()
    var_target = target.var()
    return 1 - var_residual / var_target.clamp(min=1e-8)


def normalized_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute RMSE normalized by target std."""
    mse = F.mse_loss(pred, target)
    rmse = torch.sqrt(mse)
    return rmse / target.std().clamp(min=1e-8)
