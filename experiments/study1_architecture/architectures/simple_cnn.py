"""
Simple CNN Architecture
=======================

Shallow CNN baseline for neural signal translation.
Demonstrates minimal deep learning performance without complex architecture.

Key characteristics:
- 4-6 convolutional layers
- No downsampling/upsampling (maintains temporal resolution)
- Skip connections for residual learning
- Optional FiLM conditioning
- ~10M parameters
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.

    Applies affine transformation conditioned on external embedding:
    output = gamma * input + beta

    Args:
        num_features: Number of features to modulate
        condition_dim: Dimension of conditioning vector
    """

    def __init__(self, num_features: int, condition_dim: int):
        super().__init__()
        self.num_features = num_features
        self.gamma_proj = nn.Linear(condition_dim, num_features)
        self.beta_proj = nn.Linear(condition_dim, num_features)

        # Initialize gamma to 1 (identity), beta to 0
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            x: Input tensor [B, C, T]
            condition: Conditioning vector [B, condition_dim]

        Returns:
            Modulated tensor [B, C, T]
        """
        gamma = self.gamma_proj(condition).unsqueeze(-1)  # [B, C, 1]
        beta = self.beta_proj(condition).unsqueeze(-1)  # [B, C, 1]
        return gamma * x + beta


class ConvBlock(nn.Module):
    """Basic convolutional block with optional FiLM conditioning.

    Structure: Conv -> Norm -> Activation -> [FiLM]

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        condition_dim: Conditioning dimension (0 to disable)
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        condition_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        if condition_dim > 0:
            self.film = FiLMLayer(out_channels, condition_dim)
        else:
            self.film = None

        # Skip connection if channels match
        self.skip = nn.Identity() if in_channels == out_channels else \
            nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [B, C, T]
            condition: Conditioning vector [B, condition_dim]

        Returns:
            Output [B, C_out, T]
        """
        residual = self.skip(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)

        if self.film is not None and condition is not None:
            out = self.film(out, condition)

        out = self.dropout(out)

        return out + residual


class SimpleCNN(nn.Module):
    """Simple CNN for neural signal translation.

    A shallow but effective CNN baseline that demonstrates what can be achieved
    with basic deep learning techniques. Useful for establishing that complex
    architectures provide meaningful improvements.

    Architecture:
        - Input projection
        - N convolutional blocks with skip connections
        - Output projection
        - Global residual connection

    Args:
        in_channels: Number of input channels (default: 32)
        out_channels: Number of output channels (default: 32)
        hidden_channels: Hidden channel dimension (default: 64)
        num_layers: Number of conv blocks (default: 4)
        kernel_size: Convolution kernel size (default: 7)
        num_classes: Number of odor classes (default: 7)
        condition_dim: Conditioning embedding dimension (default: 64)
        use_conditioning: Whether to use odor conditioning (default: True)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        hidden_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 7,
        num_classes: int = 7,
        condition_dim: int = 64,
        use_conditioning: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_conditioning = use_conditioning
        self.use_residual = (in_channels == out_channels)

        # Conditioning embedding
        if use_conditioning:
            self.embed = nn.Embedding(num_classes, condition_dim)
            nn.init.normal_(self.embed.weight, std=0.02)
            cond_dim = condition_dim
        else:
            self.embed = None
            cond_dim = 0

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # Stacked convolutional blocks
        self.blocks = nn.ModuleList([
            ConvBlock(
                hidden_channels, hidden_channels,
                kernel_size=kernel_size,
                condition_dim=cond_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

        # Residual projection if channels differ
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

        # Initialize output to produce small delta initially
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            ob: Input signal [B, C, T]
            odor_ids: Odor class indices [B] for conditioning

        Returns:
            Predicted signal [B, C_out, T]
        """
        # Validate input
        if ob.dim() != 3:
            raise ValueError(f"Expected 3D input [B, C, T], got shape {ob.shape}")

        B, C, T = ob.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")

        # Get conditioning
        if self.embed is not None and odor_ids is not None:
            condition = self.embed(odor_ids)  # [B, condition_dim]
        else:
            condition = None

        # Forward through network
        x = self.input_proj(ob)

        for block in self.blocks:
            x = block(x, condition)

        delta = self.output_proj(x)

        # Residual connection
        if self.use_residual:
            out = ob + delta
        elif self.residual_proj is not None:
            out = self.residual_proj(ob) + delta
        else:
            out = delta

        return out

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters.

        Returns:
            Dictionary with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        breakdown = {
            'total': total,
            'trainable': trainable,
            'input_proj': sum(p.numel() for p in self.input_proj.parameters()),
            'blocks': sum(p.numel() for block in self.blocks for p in block.parameters()),
            'output_proj': sum(p.numel() for p in self.output_proj.parameters()),
        }

        if self.embed is not None:
            breakdown['embed'] = self.embed.weight.numel()

        if self.residual_proj is not None:
            breakdown['residual_proj'] = sum(p.numel() for p in self.residual_proj.parameters())

        return breakdown

    def get_layer_names(self) -> list:
        """Get names of all layers."""
        names = ['input_proj']
        names.extend([f'block_{i}' for i in range(self.num_layers)])
        names.append('output_proj')
        return names


class SimpleCNNWithPooling(nn.Module):
    """Simple CNN with downsampling and upsampling.

    Adds simple pooling/upsampling to capture multi-scale features.
    This is between SimpleCNN and full U-Net complexity.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        hidden_channels: Hidden dimension
        num_layers: Layers per scale
        num_scales: Number of downsampling levels
        kernel_size: Conv kernel size
        num_classes: Number of odor classes
        use_conditioning: Whether to use conditioning
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_scales: int = 2,
        kernel_size: int = 7,
        num_classes: int = 7,
        use_conditioning: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales
        self.use_residual = (in_channels == out_channels)

        condition_dim = 64 if use_conditioning else 0

        # Conditioning
        if use_conditioning:
            self.embed = nn.Embedding(num_classes, condition_dim)
            nn.init.normal_(self.embed.weight, std=0.02)
        else:
            self.embed = None

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # Encoder (downsampling)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = hidden_channels

        for scale in range(num_scales):
            # Conv blocks at this scale
            blocks = nn.ModuleList([
                ConvBlock(ch, ch, kernel_size, condition_dim, dropout)
                for _ in range(num_layers)
            ])
            self.encoders.append(blocks)

            # Downsample
            self.downsamples.append(
                nn.Conv1d(ch, ch * 2, kernel_size=4, stride=2, padding=1)
            )
            ch *= 2

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ConvBlock(ch, ch, kernel_size, condition_dim, dropout)
            for _ in range(num_layers)
        ])

        # Decoder (upsampling)
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for scale in range(num_scales):
            # Upsample
            self.upsamples.append(
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=4, stride=2, padding=1)
            )
            ch //= 2

            # Conv blocks (with skip from encoder)
            blocks = nn.ModuleList([
                ConvBlock(ch * 2 if i == 0 else ch, ch, kernel_size, condition_dim, dropout)
                for i in range(num_layers)
            ])
            self.decoders.append(blocks)

        # Output projection
        self.output_proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

        # Residual projection
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

        # Initialize output small
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

        # Get conditioning
        if self.embed is not None and odor_ids is not None:
            condition = self.embed(odor_ids)
        else:
            condition = None

        # Input projection
        x = self.input_proj(ob)

        # Encoder with skip connections
        skips = []
        for blocks, down in zip(self.encoders, self.downsamples):
            for block in blocks:
                x = block(x, condition)
            skips.append(x)
            x = down(x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x, condition)

        # Decoder
        for blocks, up, skip in zip(self.decoders, self.upsamples, reversed(skips)):
            x = up(x)
            # Handle size mismatch from pooling
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, condition)

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
        return {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def create_simple_cnn(
    variant: str = "basic",
    **kwargs,
) -> nn.Module:
    """Factory function to create SimpleCNN variants.

    Args:
        variant: "basic" (no pooling) or "pooling" (with downsampling)
        **kwargs: Arguments passed to model

    Returns:
        CNN model
    """
    if variant == "basic":
        return SimpleCNN(**kwargs)
    elif variant == "pooling":
        return SimpleCNNWithPooling(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Available: basic, pooling")
