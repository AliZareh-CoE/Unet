"""
WaveNet 1D Architecture
=======================

Dilated causal convolution network for neural signal translation.
Based on DeepMind's WaveNet architecture adapted for signal-to-signal translation.

Key characteristics:
- Dilated causal convolutions (exponentially increasing receptive field)
- Gated activation units (tanh * sigmoid)
- Residual + skip connections
- Optional FiLM conditioning
- ~30M parameters

Reference:
    van den Oord et al. "WaveNet: A Generative Model for Raw Audio" (2016)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal 1D convolution with dilation.

    Pads input so output depends only on past and current timesteps.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size
        dilation: Dilation factor
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # We handle padding manually for causality
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal padding.

        Args:
            x: Input [B, C, T]

        Returns:
            Output [B, C_out, T]
        """
        # Pad only on the left (past) side for causality
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class GatedResidualBlock(nn.Module):
    """Gated residual block with dilated convolution.

    Structure:
        input -> dilated_conv -> [tanh(filter) * sigmoid(gate)] -> 1x1 -> + -> output
                                                                    |
        input ---------------------------------------------------------^

    Also produces a skip connection output for the skip aggregation path.

    Args:
        channels: Number of channels
        kernel_size: Kernel size for dilated conv
        dilation: Dilation factor
        condition_dim: Conditioning dimension (0 to disable)
        skip_channels: Skip connection channels (default: same as channels)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
        condition_dim: int = 0,
        skip_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.skip_channels = skip_channels or channels

        # Dilated causal conv producing filter and gate
        self.dilated_conv = CausalConv1d(
            channels, channels * 2,  # 2x for filter and gate
            kernel_size=kernel_size,
            dilation=dilation,
        )

        # Conditioning projection
        if condition_dim > 0:
            self.cond_proj = nn.Linear(condition_dim, channels * 2)
        else:
            self.cond_proj = None

        # Output projection (1x1 conv)
        self.output_proj = nn.Conv1d(channels, channels, kernel_size=1)

        # Skip projection (1x1 conv)
        self.skip_proj = nn.Conv1d(channels, self.skip_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input [B, C, T]
            condition: Conditioning vector [B, condition_dim]

        Returns:
            (residual_output, skip_output): Both [B, C, T]
        """
        # Dilated convolution
        h = self.dilated_conv(x)  # [B, 2*C, T]

        # Add conditioning
        if self.cond_proj is not None and condition is not None:
            cond = self.cond_proj(condition).unsqueeze(-1)  # [B, 2*C, 1]
            h = h + cond

        # Gated activation
        filter_gate = h.chunk(2, dim=1)  # Split into filter and gate
        h = torch.tanh(filter_gate[0]) * torch.sigmoid(filter_gate[1])

        # Skip connection output
        skip = self.skip_proj(h)

        # Residual connection
        residual = self.output_proj(h)
        out = x + residual

        return out, skip


class WaveNetStack(nn.Module):
    """Stack of gated residual blocks with exponentially increasing dilation.

    Args:
        channels: Channel dimension
        kernel_size: Kernel size for dilated convs
        num_layers: Number of layers per stack
        num_stacks: Number of dilation cycle repeats
        condition_dim: Conditioning dimension
        skip_channels: Skip connection channels
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 2,
        num_layers: int = 10,
        num_stacks: int = 2,
        condition_dim: int = 0,
        skip_channels: Optional[int] = None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_stacks = num_stacks
        self.skip_channels = skip_channels or channels

        # Build blocks with exponentially increasing dilation
        # Dilation pattern: 1, 2, 4, 8, ..., 2^(num_layers-1), repeat num_stacks times
        self.blocks = nn.ModuleList()

        for stack in range(num_stacks):
            for layer in range(num_layers):
                dilation = 2 ** layer
                self.blocks.append(
                    GatedResidualBlock(
                        channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        condition_dim=condition_dim,
                        skip_channels=self.skip_channels,
                    )
                )

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input [B, C, T]
            condition: Conditioning vector [B, condition_dim]

        Returns:
            (output, skip_sum): Output and aggregated skip connections
        """
        skip_sum = None

        for block in self.blocks:
            x, skip = block(x, condition)

            if skip_sum is None:
                skip_sum = skip
            else:
                skip_sum = skip_sum + skip

        return x, skip_sum

    def receptive_field(self) -> int:
        """Calculate total receptive field size."""
        # For dilation pattern 1, 2, 4, ..., 2^(L-1) repeated S times
        # RF = S * (2^L - 1) * (K - 1) + 1
        # where L = num_layers, S = num_stacks, K = kernel_size
        per_stack = sum(2 ** i for i in range(self.num_layers))
        return self.num_stacks * per_stack + 1


class WaveNet1D(nn.Module):
    """WaveNet for neural signal translation.

    Adapted from the original audio generation architecture for
    signal-to-signal translation with bidirectional processing.

    Architecture:
        - Input projection
        - Stacked gated residual blocks with dilated convolutions
        - Skip connection aggregation
        - Output projection with residual

    Args:
        in_channels: Input channels (default: 32)
        out_channels: Output channels (default: 32)
        residual_channels: Hidden channel dimension (default: 64)
        skip_channels: Skip connection channels (default: 256)
        num_layers: Layers per dilation cycle (default: 10)
        num_stacks: Number of dilation cycles (default: 2)
        kernel_size: Dilated conv kernel size (default: 2)
        num_classes: Number of odor classes (default: 7)
        condition_dim: Conditioning embedding dim (default: 64)
        use_conditioning: Whether to use odor conditioning (default: True)
        causal: If True, use causal convolutions (default: False for translation)
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        residual_channels: int = 64,
        skip_channels: int = 256,
        num_layers: int = 10,
        num_stacks: int = 2,
        kernel_size: int = 2,
        num_classes: int = 7,
        condition_dim: int = 64,
        use_conditioning: bool = True,
        causal: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.causal = causal
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
        self.input_proj = nn.Conv1d(in_channels, residual_channels, kernel_size=1)

        # WaveNet stack
        self.wavenet_stack = WaveNetStack(
            channels=residual_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            num_stacks=num_stacks,
            condition_dim=cond_dim,
            skip_channels=skip_channels,
        )

        # Output layers
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(skip_channels, out_channels, kernel_size=1)

        # Residual projection
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

        # Initialize output small for residual learning
        nn.init.zeros_(self.output_conv2.weight)
        nn.init.zeros_(self.output_conv2.bias)

    def forward(
        self,
        ob: torch.Tensor,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            ob: Input signal [B, C, T]
            odor_ids: Odor class indices [B]

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
            condition = self.embed(odor_ids)
        else:
            condition = None

        # Input projection
        x = self.input_proj(ob)

        # WaveNet stack
        _, skip_sum = self.wavenet_stack(x, condition)

        # Output layers with ReLU
        out = F.relu(skip_sum)
        out = F.relu(self.output_conv1(out))
        delta = self.output_conv2(out)

        # Residual connection
        if self.use_residual:
            out = ob + delta
        elif self.residual_proj is not None:
            out = self.residual_proj(ob) + delta
        else:
            out = delta

        return out

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'input_proj': sum(p.numel() for p in self.input_proj.parameters()),
            'wavenet_stack': sum(p.numel() for p in self.wavenet_stack.parameters()),
            'output': sum(p.numel() for p in self.output_conv1.parameters()) + \
                      sum(p.numel() for p in self.output_conv2.parameters()),
            'embed': self.embed.weight.numel() if self.embed else 0,
        }

    def receptive_field(self) -> int:
        """Get receptive field size."""
        return self.wavenet_stack.receptive_field()

    def get_layer_names(self) -> List[str]:
        """Get layer names."""
        names = ['input_proj', 'wavenet_stack', 'output_conv1', 'output_conv2']
        return names


class WaveNetBidirectional(nn.Module):
    """Bidirectional WaveNet for non-causal translation.

    Processes signal in both forward and backward directions,
    then combines the results.

    Args:
        Same as WaveNet1D
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        residual_channels: int = 64,
        skip_channels: int = 128,  # Half since we have two directions
        num_layers: int = 8,
        num_stacks: int = 2,
        kernel_size: int = 2,
        num_classes: int = 7,
        condition_dim: int = 64,
        use_conditioning: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = (in_channels == out_channels)

        # Conditioning
        if use_conditioning:
            self.embed = nn.Embedding(num_classes, condition_dim)
            nn.init.normal_(self.embed.weight, std=0.02)
            cond_dim = condition_dim
        else:
            self.embed = None
            cond_dim = 0

        # Forward WaveNet
        self.forward_proj = nn.Conv1d(in_channels, residual_channels, kernel_size=1)
        self.forward_stack = WaveNetStack(
            residual_channels, kernel_size, num_layers, num_stacks,
            cond_dim, skip_channels,
        )

        # Backward WaveNet
        self.backward_proj = nn.Conv1d(in_channels, residual_channels, kernel_size=1)
        self.backward_stack = WaveNetStack(
            residual_channels, kernel_size, num_layers, num_stacks,
            cond_dim, skip_channels,
        )

        # Combine forward and backward
        self.combine = nn.Conv1d(skip_channels * 2, skip_channels, kernel_size=1)

        # Output
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(skip_channels, out_channels, kernel_size=1)

        # Residual
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

        nn.init.zeros_(self.output_conv2.weight)
        nn.init.zeros_(self.output_conv2.bias)

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

        # Forward direction
        fwd = self.forward_proj(ob)
        _, fwd_skip = self.forward_stack(fwd, condition)

        # Backward direction (flip, process, flip back)
        bwd = self.backward_proj(ob.flip(-1))
        _, bwd_skip = self.backward_stack(bwd, condition)
        bwd_skip = bwd_skip.flip(-1)

        # Combine
        combined = torch.cat([fwd_skip, bwd_skip], dim=1)
        combined = self.combine(combined)

        # Output
        out = F.relu(combined)
        out = F.relu(self.output_conv1(out))
        delta = self.output_conv2(out)

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


def create_wavenet(
    variant: str = "standard",
    **kwargs,
) -> nn.Module:
    """Factory function for WaveNet variants.

    Args:
        variant: "standard" or "bidirectional"
        **kwargs: Arguments for model

    Returns:
        WaveNet model
    """
    if variant == "standard":
        return WaveNet1D(**kwargs)
    elif variant == "bidirectional":
        return WaveNetBidirectional(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Available: standard, bidirectional")
