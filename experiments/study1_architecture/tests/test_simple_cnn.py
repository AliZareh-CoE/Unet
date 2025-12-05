"""
Tests for Simple CNN Architecture
=================================

Comprehensive tests for Nature Methods quality assurance.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from experiments.study1_architecture.architectures.simple_cnn import (
    SimpleCNN,
    SimpleCNNWithPooling,
    ConvBlock,
    FiLMLayer,
    create_simple_cnn,
)


# Test configuration
BATCH_SIZE = 4
IN_CHANNELS = 32
OUT_CHANNELS = 32
TIME_STEPS = 2048
NUM_CLASSES = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestFiLMLayer:
    """Tests for FiLM modulation layer."""

    def test_film_shape(self):
        """Test FiLM layer output shape."""
        film = FiLMLayer(64, 32).to(DEVICE)
        x = torch.randn(BATCH_SIZE, 64, TIME_STEPS, device=DEVICE)
        condition = torch.randn(BATCH_SIZE, 32, device=DEVICE)

        out = film(x, condition)
        assert out.shape == x.shape

    def test_film_identity_init(self):
        """Test that FiLM initializes to identity transform."""
        film = FiLMLayer(64, 32).to(DEVICE)
        x = torch.randn(BATCH_SIZE, 64, TIME_STEPS, device=DEVICE)
        condition = torch.zeros(BATCH_SIZE, 32, device=DEVICE)

        out = film(x, condition)
        # With zero condition, gamma should be ~1, beta ~0
        assert torch.allclose(out, x, atol=0.1)


class TestConvBlock:
    """Tests for ConvBlock."""

    def test_forward_shape(self):
        """Test ConvBlock output shape."""
        block = ConvBlock(64, 64, kernel_size=7, condition_dim=32).to(DEVICE)
        x = torch.randn(BATCH_SIZE, 64, TIME_STEPS, device=DEVICE)
        condition = torch.randn(BATCH_SIZE, 32, device=DEVICE)

        out = block(x, condition)
        assert out.shape == x.shape

    def test_channel_change(self):
        """Test ConvBlock with channel change."""
        block = ConvBlock(32, 64, kernel_size=7).to(DEVICE)
        x = torch.randn(BATCH_SIZE, 32, TIME_STEPS, device=DEVICE)

        out = block(x, None)
        assert out.shape == (BATCH_SIZE, 64, TIME_STEPS)


class TestSimpleCNN:
    """Tests for SimpleCNN model."""

    @pytest.fixture
    def model(self):
        """Create a default SimpleCNN model."""
        return SimpleCNN(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            hidden_channels=64,
            num_layers=4,
            num_classes=NUM_CLASSES,
            use_conditioning=True,
        ).to(DEVICE)

    @pytest.fixture
    def input_data(self):
        """Create test input data."""
        ob = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        odor_ids = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)
        return ob, odor_ids

    def test_forward_pass_shape(self, model, input_data):
        """Test output shape."""
        ob, odor_ids = input_data
        output = model(ob, odor_ids)

        assert output.shape == (BATCH_SIZE, OUT_CHANNELS, TIME_STEPS)

    def test_forward_without_conditioning(self, input_data):
        """Test forward pass without conditioning."""
        model = SimpleCNN(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            use_conditioning=False,
        ).to(DEVICE)

        ob, _ = input_data
        output = model(ob, None)

        assert output.shape == (BATCH_SIZE, OUT_CHANNELS, TIME_STEPS)

    def test_gradient_flow(self, model, input_data):
        """Test gradient flow through model."""
        ob, odor_ids = input_data
        ob.requires_grad_(True)

        output = model(ob, odor_ids)
        loss = output.mean()
        loss.backward()

        assert ob.grad is not None, "No gradient for input"
        assert torch.isfinite(ob.grad).all(), "Gradient contains NaN/Inf"

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient for {name}"

    def test_no_nan_inf_output(self, model, input_data):
        """Test output is finite."""
        ob, odor_ids = input_data
        output = model(ob, odor_ids)

        assert torch.isfinite(output).all()

    def test_residual_at_init(self, model, input_data):
        """Test that output ≈ input at initialization."""
        ob, _ = input_data

        model = SimpleCNN(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            use_conditioning=False,
        ).to(DEVICE)

        output = model(ob, None)
        diff = (output - ob).abs().mean()

        # Should be reasonably close due to residual connection
        assert diff < 1.0, f"Initial output differs by {diff:.4f}"

    def test_conditioning_effect(self, model, input_data):
        """Test that conditioning affects output."""
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
        assert 'input_proj' in counts
        assert 'blocks' in counts
        assert 'output_proj' in counts

    def test_different_configs(self):
        """Test different configuration combinations."""
        configs = [
            {'hidden_channels': 32, 'num_layers': 2},
            {'hidden_channels': 128, 'num_layers': 6},
            {'kernel_size': 3},
            {'kernel_size': 15},
        ]

        for config in configs:
            model = SimpleCNN(**config).to(DEVICE)
            x = torch.randn(2, 32, 1024, device=DEVICE)
            out = model(x, None)
            assert out.shape == x.shape, f"Failed with config {config}"


class TestSimpleCNNWithPooling:
    """Tests for SimpleCNNWithPooling model."""

    @pytest.fixture
    def model(self):
        """Create model with pooling."""
        return SimpleCNNWithPooling(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            hidden_channels=64,
            num_scales=2,
            num_classes=NUM_CLASSES,
        ).to(DEVICE)

    @pytest.fixture
    def input_data(self):
        """Create test data."""
        ob = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        odor_ids = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)
        return ob, odor_ids

    def test_forward_shape(self, model, input_data):
        """Test output shape matches input."""
        ob, odor_ids = input_data
        output = model(ob, odor_ids)

        assert output.shape == ob.shape

    def test_gradient_flow(self, model, input_data):
        """Test gradients flow properly."""
        ob, odor_ids = input_data
        ob.requires_grad_(True)

        output = model(ob, odor_ids)
        loss = output.mean()
        loss.backward()

        assert ob.grad is not None
        assert torch.isfinite(ob.grad).all()

    def test_various_time_lengths(self, model):
        """Test with different time lengths."""
        for T in [512, 1024, 2048, 4096]:
            x = torch.randn(2, IN_CHANNELS, T, device=DEVICE)
            out = model(x, None)
            assert out.shape == x.shape, f"Failed for T={T}"


class TestFactoryFunction:
    """Tests for create_simple_cnn factory."""

    def test_create_basic(self):
        """Test creating basic variant."""
        model = create_simple_cnn("basic")
        assert isinstance(model, SimpleCNN)

    def test_create_pooling(self):
        """Test creating pooling variant."""
        model = create_simple_cnn("pooling")
        assert isinstance(model, SimpleCNNWithPooling)

    def test_invalid_variant(self):
        """Test invalid variant raises error."""
        with pytest.raises(ValueError):
            create_simple_cnn("invalid")

    def test_kwargs_passthrough(self):
        """Test kwargs are passed through."""
        model = create_simple_cnn("basic", hidden_channels=128)
        assert model.hidden_channels == 128


class TestMemoryAndPerformance:
    """Tests for memory and performance."""

    def test_parameter_budget(self):
        """Test parameter count is within budget."""
        model = SimpleCNN(
            in_channels=32,
            out_channels=32,
            hidden_channels=64,
            num_layers=4,
        )

        counts = model.count_parameters()

        # SimpleCNN should have < 10M parameters
        assert counts['total'] < 10_000_000, \
            f"SimpleCNN has too many params: {counts['total']}"

    def test_deterministic(self):
        """Test deterministic output."""
        model = SimpleCNN().to(DEVICE).eval()

        x = torch.randn(2, 32, 1024, device=DEVICE)
        odor = torch.tensor([0, 1], device=DEVICE)

        with torch.no_grad():
            out1 = model(x, odor)
            out2 = model(x, odor)

        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
