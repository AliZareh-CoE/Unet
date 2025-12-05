"""
Tests for WaveNet 1D Architecture
=================================

Comprehensive tests for Nature Methods quality assurance.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from experiments.study1_architecture.architectures.wavenet_1d import (
    WaveNet1D,
    WaveNetBidirectional,
    WaveNetStack,
    GatedResidualBlock,
    CausalConv1d,
    create_wavenet,
)


# Test configuration
BATCH_SIZE = 4
IN_CHANNELS = 32
OUT_CHANNELS = 32
TIME_STEPS = 2048
NUM_CLASSES = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestCausalConv1d:
    """Tests for causal convolution."""

    def test_output_shape(self):
        """Test output maintains input shape."""
        conv = CausalConv1d(32, 64, kernel_size=2, dilation=1).to(DEVICE)
        x = torch.randn(BATCH_SIZE, 32, TIME_STEPS, device=DEVICE)

        out = conv(x)
        assert out.shape == (BATCH_SIZE, 64, TIME_STEPS)

    def test_causality(self):
        """Test that output at time t depends only on inputs t and earlier."""
        conv = CausalConv1d(1, 1, kernel_size=2, dilation=1).to(DEVICE)
        conv.eval()

        # Create input with known pattern
        x = torch.zeros(1, 1, 100, device=DEVICE)
        x[0, 0, 50] = 1.0  # Impulse at t=50

        with torch.no_grad():
            out = conv(x)

        # Output should be non-zero only at t >= 50
        assert (out[0, 0, :50].abs() < 1e-6).all(), "Causal violation: output before impulse"

    def test_increasing_dilation(self):
        """Test with various dilation factors."""
        for dilation in [1, 2, 4, 8, 16]:
            conv = CausalConv1d(32, 32, kernel_size=2, dilation=dilation).to(DEVICE)
            x = torch.randn(2, 32, 512, device=DEVICE)
            out = conv(x)
            assert out.shape == x.shape


class TestGatedResidualBlock:
    """Tests for gated residual block."""

    def test_output_shapes(self):
        """Test output shapes."""
        block = GatedResidualBlock(64, dilation=4, condition_dim=32).to(DEVICE)
        x = torch.randn(BATCH_SIZE, 64, TIME_STEPS, device=DEVICE)
        cond = torch.randn(BATCH_SIZE, 32, device=DEVICE)

        residual, skip = block(x, cond)

        assert residual.shape == x.shape
        assert skip.shape == x.shape

    def test_without_conditioning(self):
        """Test block without conditioning."""
        block = GatedResidualBlock(64, dilation=4, condition_dim=0).to(DEVICE)
        x = torch.randn(BATCH_SIZE, 64, TIME_STEPS, device=DEVICE)

        residual, skip = block(x, None)

        assert residual.shape == x.shape
        assert skip.shape == x.shape


class TestWaveNetStack:
    """Tests for WaveNet stack."""

    def test_forward(self):
        """Test stack forward pass."""
        stack = WaveNetStack(
            channels=64,
            num_layers=4,
            num_stacks=2,
            condition_dim=32,
        ).to(DEVICE)

        x = torch.randn(BATCH_SIZE, 64, TIME_STEPS, device=DEVICE)
        cond = torch.randn(BATCH_SIZE, 32, device=DEVICE)

        out, skip = stack(x, cond)

        assert out.shape == x.shape
        assert skip.shape == x.shape

    def test_receptive_field(self):
        """Test receptive field calculation."""
        stack = WaveNetStack(
            channels=64,
            num_layers=10,
            num_stacks=2,
        )

        rf = stack.receptive_field()
        # RF = 2 * (2^10 - 1) + 1 = 2 * 1023 + 1 = 2047
        assert rf > 2000, f"Receptive field {rf} too small"


class TestWaveNet1D:
    """Tests for WaveNet1D model."""

    @pytest.fixture
    def model(self):
        """Create default WaveNet model."""
        return WaveNet1D(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            residual_channels=64,
            skip_channels=128,
            num_layers=6,
            num_stacks=2,
            num_classes=NUM_CLASSES,
        ).to(DEVICE)

    @pytest.fixture
    def input_data(self):
        """Create test input data."""
        ob = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        odor_ids = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)
        return ob, odor_ids

    def test_forward_shape(self, model, input_data):
        """Test output shape."""
        ob, odor_ids = input_data
        output = model(ob, odor_ids)

        assert output.shape == (BATCH_SIZE, OUT_CHANNELS, TIME_STEPS)

    def test_forward_without_conditioning(self, input_data):
        """Test without conditioning."""
        model = WaveNet1D(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            use_conditioning=False,
        ).to(DEVICE)

        ob, _ = input_data
        output = model(ob, None)

        assert output.shape == (BATCH_SIZE, OUT_CHANNELS, TIME_STEPS)

    def test_gradient_flow(self, model, input_data):
        """Test gradient flow."""
        ob, odor_ids = input_data
        ob.requires_grad_(True)

        output = model(ob, odor_ids)
        loss = output.mean()
        loss.backward()

        assert ob.grad is not None
        assert torch.isfinite(ob.grad).all()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient for {name}"

    def test_no_nan_inf(self, model, input_data):
        """Test output is finite."""
        ob, odor_ids = input_data
        output = model(ob, odor_ids)

        assert torch.isfinite(output).all()

    def test_residual_at_init(self, input_data):
        """Test residual behavior at initialization."""
        model = WaveNet1D(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            use_conditioning=False,
        ).to(DEVICE)

        ob, _ = input_data
        output = model(ob, None)

        diff = (output - ob).abs().mean()
        assert diff < 1.0, f"Initial output differs by {diff:.4f}"

    def test_conditioning_effect(self, model, input_data):
        """Test conditioning affects output."""
        ob, _ = input_data

        odor_0 = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
        odor_1 = torch.ones(BATCH_SIZE, dtype=torch.long, device=DEVICE)

        out_0 = model(ob, odor_0)
        out_1 = model(ob, odor_1)

        diff = (out_0 - out_1).abs().mean()
        assert diff > 1e-6, "Conditioning has no effect"

    def test_input_validation(self, model):
        """Test input validation."""
        wrong_dims = torch.randn(BATCH_SIZE, IN_CHANNELS, device=DEVICE)
        with pytest.raises(ValueError, match="Expected 3D"):
            model(wrong_dims, None)

        wrong_channels = torch.randn(BATCH_SIZE, IN_CHANNELS + 1, TIME_STEPS, device=DEVICE)
        with pytest.raises(ValueError, match="Expected .* input channels"):
            model(wrong_channels, None)

    def test_parameter_count(self, model):
        """Test parameter counting."""
        counts = model.count_parameters()

        assert counts['total'] > 0
        assert counts['trainable'] == counts['total']

    def test_receptive_field(self, model):
        """Test receptive field reporting."""
        rf = model.receptive_field()
        assert rf > 0

    def test_variable_time_lengths(self, model):
        """Test with different time lengths."""
        for T in [512, 1024, 2048, 4096]:
            x = torch.randn(2, IN_CHANNELS, T, device=DEVICE)
            out = model(x, None)
            assert out.shape == x.shape, f"Failed for T={T}"


class TestWaveNetBidirectional:
    """Tests for bidirectional WaveNet."""

    @pytest.fixture
    def model(self):
        """Create bidirectional model."""
        return WaveNetBidirectional(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            residual_channels=64,
            skip_channels=64,
            num_layers=4,
            num_stacks=2,
        ).to(DEVICE)

    @pytest.fixture
    def input_data(self):
        """Create test data."""
        ob = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        odor_ids = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)
        return ob, odor_ids

    def test_forward_shape(self, model, input_data):
        """Test output shape."""
        ob, odor_ids = input_data
        output = model(ob, odor_ids)

        assert output.shape == ob.shape

    def test_gradient_flow(self, model, input_data):
        """Test gradients."""
        ob, odor_ids = input_data
        ob.requires_grad_(True)

        output = model(ob, odor_ids)
        loss = output.mean()
        loss.backward()

        assert ob.grad is not None
        assert torch.isfinite(ob.grad).all()


class TestFactoryFunction:
    """Tests for create_wavenet factory."""

    def test_create_standard(self):
        """Test creating standard variant."""
        model = create_wavenet("standard")
        assert isinstance(model, WaveNet1D)

    def test_create_bidirectional(self):
        """Test creating bidirectional variant."""
        model = create_wavenet("bidirectional")
        assert isinstance(model, WaveNetBidirectional)

    def test_invalid_variant(self):
        """Test invalid variant raises error."""
        with pytest.raises(ValueError):
            create_wavenet("invalid")

    def test_kwargs(self):
        """Test kwargs passthrough."""
        model = create_wavenet("standard", residual_channels=128)
        assert model.residual_channels == 128


class TestMemoryAndPerformance:
    """Memory and performance tests."""

    def test_parameter_budget(self):
        """Test WaveNet is within parameter budget."""
        model = WaveNet1D(
            in_channels=32,
            out_channels=32,
            residual_channels=64,
            skip_channels=256,
            num_layers=10,
            num_stacks=2,
        )

        counts = model.count_parameters()

        # WaveNet should have < 50M parameters
        assert counts['total'] < 50_000_000, f"Too many params: {counts['total']}"

    def test_deterministic(self):
        """Test deterministic output."""
        model = WaveNet1D(num_layers=4, num_stacks=1).to(DEVICE).eval()

        x = torch.randn(2, 32, 512, device=DEVICE)
        odor = torch.tensor([0, 1], device=DEVICE)

        with torch.no_grad():
            out1 = model(x, odor)
            out2 = model(x, odor)

        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
