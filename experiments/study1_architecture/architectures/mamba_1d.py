"""
Mamba/S4 1D Architecture
========================

State-space model for neural signal translation.
Efficient for long sequences with linear complexity.

Key characteristics:
- State-space model (SSM) based on S4/Mamba
- O(T) complexity for sequence processing
- Excellent for long-range dependencies
- Hardware-efficient selective scan
- ~30-60M parameters

References:
    Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces" (2022)
    Gu & Dao "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class S4DKernel(nn.Module):
    """Diagonal S4 (S4D) kernel.

    Simplified S4 using diagonal state matrix for efficiency.
    Implements the continuous-time SSM: x' = Ax + Bu, y = Cx + Du

    Args:
        d_model: Model dimension
        d_state: State dimension
        dt_min: Minimum discretization step
        dt_max: Maximum discretization step
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state

        # Initialize A as diagonal matrix with HiPPO initialization
        # A = -1/2 + i * omega where omega are learnable
        A_real = torch.ones(d_model, d_state) * -0.5
        A_imag = torch.arange(d_state).float().unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(torch.complex(A_real.abs(), A_imag.abs()).abs()))
        self.A_angle = nn.Parameter(torch.atan2(A_imag, A_real))

        # B is learned
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

        # C is learned
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(d_model))

        # Discretization step
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Apply S4D kernel.

        Args:
            u: Input [B, D, T]

        Returns:
            Output [B, D, T]
        """
        B, D, T = u.shape

        # Get discretized kernel
        dt = self.log_dt.exp()  # [D]
        A = -torch.exp(self.A_log) * torch.exp(1j * self.A_angle)  # [D, N] complex

        # Discretize (ZOH): A_bar = exp(A * dt)
        dt_expanded = dt.unsqueeze(-1)  # [D, 1]
        A_bar = torch.exp(A * dt_expanded)  # [D, N] complex

        # B_bar = (A_bar - I) / A * B
        # For diagonal A: B_bar = (exp(A*dt) - 1) / A * B ≈ dt * B for small dt
        B_bar = self.B * dt_expanded  # Simplified

        # Compute convolution kernel
        # K[i] = C @ A_bar^i @ B_bar
        powers = torch.arange(T, device=u.device).float()  # [T]
        A_bar_powers = A_bar.unsqueeze(-1) ** powers.unsqueeze(0).unsqueeze(0)  # [D, N, T]

        # K = sum_n C[n] * A^t * B[n] for each t
        K = torch.einsum('dn,dnt,dn->dt', self.C.to(A_bar_powers.dtype), A_bar_powers, B_bar.to(A_bar_powers.dtype))
        K = K.real  # [D, T]

        # Convolve with input using FFT
        K_fft = torch.fft.rfft(K, n=2*T)  # [D, T+1]
        u_fft = torch.fft.rfft(u, n=2*T)  # [B, D, T+1]

        y = torch.fft.irfft(K_fft.unsqueeze(0) * u_fft, n=2*T)[..., :T]  # [B, D, T]

        # Add skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * u

        return y


class SelectiveSSM(nn.Module):
    """Selective State-Space Model (Mamba-style).

    Input-dependent discretization for selective information flow.

    Args:
        d_model: Model dimension
        d_state: State dimension
        d_conv: Local convolution width
        expand: Expansion factor
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Local convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Fixed A (diagonal)
        A = torch.arange(1, d_state + 1).float()
        self.A_log = nn.Parameter(torch.log(A))

        # D (skip)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [B, T, D]

        Returns:
            Output [B, T, D]
        """
        B, T, D = x.shape

        # Input projection
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # Local convolution
        x = x.transpose(1, 2)  # [B, d_inner, T]
        x = self.conv1d(x)[:, :, :T]  # [B, d_inner, T]
        x = x.transpose(1, 2)  # [B, T, d_inner]

        x = F.silu(x)

        # SSM computation
        y = self._ssm(x)  # [B, T, d_inner]

        # Gate
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Run SSM computation.

        Args:
            x: Input [B, T, d_inner]

        Returns:
            Output [B, T, d_inner]
        """
        B, T, D = x.shape
        N = self.d_state

        # Get input-dependent parameters
        x_dbl = self.x_proj(x)  # [B, T, 2*N + 1]
        delta, B_proj, C_proj = x_dbl.split([1, N, N], dim=-1)

        # Delta (discretization step)
        delta = F.softplus(delta)  # [B, T, 1]

        # A
        A = -torch.exp(self.A_log)  # [N]

        # Discretize
        # A_bar = exp(A * delta)
        A_bar = torch.exp(A * delta)  # [B, T, N]
        B_bar = delta * B_proj  # [B, T, N]

        # Sequential scan (simple implementation)
        # For efficiency, this should use parallel scan or hardware-specific kernels
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(T):
            h = A_bar[:, t:t+1, :] * h + B_bar[:, t:t+1, :] * x[:, t:t+1, :].unsqueeze(-1)
            y_t = (h * C_proj[:, t:t+1, :].unsqueeze(2)).sum(-1)  # [B, 1, D]
            ys.append(y_t)

        y = torch.cat(ys, dim=1)  # [B, T, D]

        # Skip connection
        y = y + self.D * x

        return y


class MambaBlock(nn.Module):
    """Mamba block with selective SSM.

    Args:
        d_model: Model dimension
        d_state: State dimension
        d_conv: Conv kernel width
        expand: Expansion factor
        dropout: Dropout probability
        condition_dim: Conditioning dimension
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        condition_dim: int = 0,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

        # Conditioning
        if condition_dim > 0:
            self.gamma = nn.Linear(condition_dim, d_model)
            self.beta = nn.Linear(condition_dim, d_model)
            nn.init.ones_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)
        else:
            self.gamma = None
            self.beta = None

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        x = x + residual

        # Conditioning
        if self.gamma is not None and condition is not None:
            gamma = self.gamma(condition).unsqueeze(1)
            beta = self.beta(condition).unsqueeze(1)
            x = gamma * x + beta

        return x


class Mamba1D(nn.Module):
    """Mamba for neural signal translation.

    State-space model with selective mechanism for efficient
    long-range sequence modeling.

    Args:
        in_channels: Input channels (default: 32)
        out_channels: Output channels (default: 32)
        d_model: Model dimension (default: 256)
        d_state: State dimension (default: 16)
        d_conv: Conv kernel width (default: 4)
        expand: Expansion factor (default: 2)
        num_layers: Number of Mamba blocks (default: 6)
        num_classes: Odor classes (default: 7)
        condition_dim: Conditioning dimension (default: 64)
        use_conditioning: Use conditioning (default: True)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 6,
        num_classes: int = 7,
        condition_dim: int = 64,
        use_conditioning: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_residual = (in_channels == out_channels)

        # Conditioning
        if use_conditioning:
            self.embed = nn.Embedding(num_classes, condition_dim)
            nn.init.normal_(self.embed.weight, std=0.02)
            cond_dim = condition_dim
        else:
            self.embed = None
            cond_dim = 0

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)

        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout, cond_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Conv1d(d_model, out_channels, kernel_size=1)

        # Residual
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        if ob.dim() != 3:
            raise ValueError(f"Expected 3D input [B, C, T], got shape {ob.shape}")

        B, C, T = ob.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")

        # Conditioning
        if self.embed is not None and odor_ids is not None:
            condition = self.embed(odor_ids)
        else:
            condition = None

        # Input projection
        x = self.input_proj(ob)  # [B, D, T]
        x = x.transpose(1, 2)  # [B, T, D]

        # Mamba blocks
        for block in self.blocks:
            x = block(x, condition)

        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, D, T]

        # Output
        delta = self.output_proj(x)

        # Residual
        if self.use_residual:
            out = ob + delta
        elif self.residual_proj is not None:
            out = self.residual_proj(ob) + delta
        else:
            out = delta

        return out

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'input_proj': sum(p.numel() for p in self.input_proj.parameters()),
            'blocks': sum(p.numel() for block in self.blocks for p in block.parameters()),
            'output_proj': sum(p.numel() for p in self.output_proj.parameters()),
        }

    def get_layer_names(self) -> List[str]:
        """Get layer names."""
        names = ['input_proj']
        names.extend([f'block_{i}' for i in range(self.num_layers)])
        names.extend(['norm', 'output_proj'])
        return names


def create_mamba(
    variant: str = "standard",
    **kwargs,
) -> nn.Module:
    """Factory function for Mamba variants.

    Args:
        variant: "standard", "small", or "large"
        **kwargs: Model arguments

    Returns:
        Mamba model
    """
    if variant == "standard":
        return Mamba1D(**kwargs)
    elif variant == "small":
        kwargs.setdefault('d_model', 128)
        kwargs.setdefault('d_state', 8)
        kwargs.setdefault('num_layers', 4)
        return Mamba1D(**kwargs)
    elif variant == "large":
        kwargs.setdefault('d_model', 512)
        kwargs.setdefault('d_state', 32)
        kwargs.setdefault('num_layers', 8)
        return Mamba1D(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Available: standard, small, large")
