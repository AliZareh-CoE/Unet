"""
Tests for FNet 1D Architecture
==============================

Comprehensive tests for Nature Methods quality assurance.
"""

from __future__ import annotations

import pytest
import torch

from experiments.study1_architecture.architectures.fnet_1d import (
    FNet1D,
    FNetWithConv,
    FNetBlock,
    FourierMixing1D,
    FourierMixing2D,
    create_fnet,
)


BATCH_SIZE = 4
IN_CHANNELS = 32
OUT_CHANNELS = 32
TIME_STEPS = 2048
NUM_CLASSES = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestFourierMixing:
    """Tests for Fourier mixing layers."""

    def test_1d_mixing_shape(self):
        """Test 1D mixing preserves shape."""
        mixing = FourierMixing1D()
        x = torch.randn(BATCH_SIZE, 64, TIME_STEPS, device=DEVICE)
        out = mixing(x)
        assert out.shape == x.shape

    def test_2d_mixing_shape(self):
        """Test 2D mixing preserves shape."""
        mixing = FourierMixing2D()
        x = torch.randn(BATCH_SIZE, 64, TIME_STEPS, device=DEVICE)
        out = mixing(x)
        assert out.shape == x.shape

    def test_mixing_is_deterministic(self):
        """Test mixing is deterministic (no learned params)."""
        mixing = FourierMixing1D()
        x = torch.randn(2, 32, 128, device=DEVICE)

        out1 = mixing(x)
        out2 = mixing(x)

        assert torch.allclose(out1, out2)


class TestFNetBlock:
    """Tests for FNet block."""

    def test_forward_shape(self):
        """Test block preserves shape."""
        block = FNetBlock(channels=64, ffn_ratio=4).to(DEVICE)
        x = torch.randn(BATCH_SIZE, 64, TIME_STEPS, device=DEVICE)
        out = block(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradients flow through block."""
        block = FNetBlock(channels=64).to(DEVICE)
        x = torch.randn(2, 64, 256, device=DEVICE, requires_grad=True)

        out = block(x)
        loss = out.mean()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestFNet1D:
    """Tests for FNet1D model."""

    @pytest.fixture
    def model(self):
        """Create default FNet model."""
        return FNet1D(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            hidden_dim=128,
            num_layers=4,
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
        model = FNet1D(
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
        """Test residual behavior at init."""
        model = FNet1D(
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

    def test_parameter_count(self, model):
        """Test parameter counting."""
        counts = model.count_parameters()
        assert counts['total'] > 0
        assert counts['trainable'] == counts['total']

    def test_with_patching(self, input_data):
        """Test with patch embedding."""
        model = FNet1D(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            patch_size=4,
        ).to(DEVICE)

        ob, odor_ids = input_data
        output = model(ob, odor_ids)
        assert output.shape == ob.shape

    def test_variable_time(self, model):
        """Test with various time lengths."""
        for T in [512, 1024, 2048]:
            x = torch.randn(2, IN_CHANNELS, T, device=DEVICE)
            out = model(x, None)
            assert out.shape == x.shape


class TestFNetWithConv:
    """Tests for FNet with convolutional layers."""

    @pytest.fixture
    def model(self):
        """Create FNetWithConv model."""
        return FNetWithConv(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            hidden_dim=128,
            num_layers=2,
            conv_layers=2,
        ).to(DEVICE)

    def test_forward_shape(self, model):
        """Test output shape."""
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        out = model(x, None)
        assert out.shape == x.shape

    def test_gradient_flow(self, model):
        """Test gradients."""
        x = torch.randn(2, IN_CHANNELS, 512, device=DEVICE, requires_grad=True)
        odor = torch.tensor([0, 1], device=DEVICE)

        out = model(x, odor)
        loss = out.mean()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestFactoryFunction:
    """Tests for create_fnet factory."""

    def test_create_standard(self):
        """Test standard variant."""
        model = create_fnet("standard")
        assert isinstance(model, FNet1D)

    def test_create_conv(self):
        """Test conv variant."""
        model = create_fnet("conv")
        assert isinstance(model, FNetWithConv)

    def test_create_2d(self):
        """Test 2d mixing variant."""
        model = create_fnet("2d")
        assert isinstance(model, FNet1D)

    def test_invalid_variant(self):
        """Test invalid variant."""
        with pytest.raises(ValueError):
            create_fnet("invalid")

    def test_kwargs(self):
        """Test kwargs passthrough."""
        model = create_fnet("standard", hidden_dim=512)
        assert model.hidden_dim == 512


class TestMemoryAndPerformance:
    """Memory and performance tests."""

    def test_parameter_budget(self):
        """Test FNet is within budget."""
        model = FNet1D(
            in_channels=32,
            out_channels=32,
            hidden_dim=256,
            num_layers=6,
        )
        counts = model.count_parameters()
        assert counts['total'] < 100_000_000

    def test_deterministic(self):
        """Test deterministic output."""
        model = FNet1D(num_layers=2).to(DEVICE).eval()
        x = torch.randn(2, 32, 512, device=DEVICE)
        odor = torch.tensor([0, 1], device=DEVICE)

        with torch.no_grad():
            out1 = model(x, odor)
            out2 = model(x, odor)

        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
