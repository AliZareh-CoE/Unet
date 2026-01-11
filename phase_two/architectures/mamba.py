"""
Mamba1D for Neural Signal Translation
=====================================

Selective State Space Model based on Mamba (Gu & Dao, 2023).
Provides O(N) complexity with selective state updates.

Simplified implementation focusing on core SSM concepts.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress harmless tensor deallocation warnings under FSDP
warnings.filterwarnings("ignore", message=".*Deallocating Tensor.*")


class SelectiveSSM(nn.Module):
    """Selective State Space Model (simplified Mamba core).

    Implements the selective scan mechanism:
        x[t] = A * x[t-1] + B * u[t]
        y[t] = C * x[t]

    With input-dependent (selective) A, B, C matrices.
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
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt

        # Discretization parameters
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A is fixed (learned but not input-dependent)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.register_buffer('A', -A)

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with selective scan.

        Args:
            x: Input [B, T, D]

        Returns:
            Output [B, T, D]
        """
        B, T, D = x.shape

        # Input projection -> z (gate) and x (main path)
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_main, z = xz.chunk(2, dim=-1)  # [B, T, d_inner] each

        # Convolution for local context
        x_main = x_main.transpose(1, 2)  # [B, d_inner, T]
        x_main = self.conv1d(x_main)[:, :, :T]  # Keep same length
        x_main = x_main.transpose(1, 2)  # [B, T, d_inner]
        x_main = F.silu(x_main)

        # Compute selective SSM parameters
        x_ssm = self.x_proj(x_main)  # [B, T, d_state*2 + 1]
        B_param = x_ssm[:, :, :self.d_state]  # [B, T, d_state]
        C_param = x_ssm[:, :, self.d_state:2*self.d_state]  # [B, T, d_state]
        dt = F.softplus(x_ssm[:, :, -1:])  # [B, T, 1]

        # Discretize A using dt
        dt_proj = self.dt_proj(dt)  # [B, T, d_inner]
        A_discrete = torch.exp(self.A.unsqueeze(0).unsqueeze(0) * dt_proj.unsqueeze(-1))  # [B, T, d_inner, d_state]

        # Selective scan (simplified - parallel implementation)
        # For full efficiency, use CUDA kernel or chunked scan
        y = self._selective_scan(x_main, A_discrete, B_param, C_param, self.D)

        # Gate and output
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y

    def _selective_scan(
        self,
        u: torch.Tensor,      # [B, T, d_inner]
        A: torch.Tensor,      # [B, T, d_inner, d_state]
        B: torch.Tensor,      # [B, T, d_state]
        C: torch.Tensor,      # [B, T, d_state]
        D: torch.Tensor,      # [d_inner]
    ) -> torch.Tensor:
        """Simplified selective scan.

        Note: For production, use optimized CUDA implementation.
        This is a sequential fallback for correctness.
        """
        B_size, T, d_inner = u.shape
        d_state = B.shape[-1]

        # Initialize state
        h = torch.zeros(B_size, d_inner, d_state, device=u.device, dtype=u.dtype)

        outputs = []
        for t in range(T):
            # State update: h = A * h + B * u
            h = A[:, t] * h + B[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)

            # Output: y = C * h + D * u
            y = (C[:, t].unsqueeze(1) * h).sum(-1) + D * u[:, t]
            outputs.append(y)

        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    """Mamba block with residual connection."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class Mamba1D(nn.Module):
    """Mamba for 1D signal translation.

    State-space model with selective updates for
    efficient sequence modeling.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        d_model: Model dimension
        d_state: State dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        n_layers: Number of Mamba blocks

    Input: [B, C_in, T]
    Output: [B, C_out, T]
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input projection
        self.input_proj = nn.Linear(in_channels, d_model)

        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, out_channels)

        # Residual
        self.use_residual = in_channels == out_channels

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C_in, T]

        Returns:
            Output tensor [B, C_out, T]
        """
        identity = x

        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        # Project
        x = self.input_proj(x)

        # Mamba blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_proj(x)

        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        # Residual
        if self.use_residual:
            x = x + identity

        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
