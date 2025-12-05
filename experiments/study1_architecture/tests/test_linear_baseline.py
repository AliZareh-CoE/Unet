"""
Tests for Linear Baseline Architecture
======================================

Comprehensive tests for Nature Methods quality assurance.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from experiments.study1_architecture.architectures.linear_baseline import (
    LinearBaseline,
    LinearChannelMixer,
    create_linear_baseline,
)


# Test configuration
BATCH_SIZE = 4
IN_CHANNELS = 32
OUT_CHANNELS = 32
TIME_STEPS = 2048
NUM_CLASSES = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestLinearBaseline:
    """Tests for LinearBaseline model."""

    @pytest.fixture
    def model(self):
        """Create a default LinearBaseline model."""
        return LinearBaseline(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
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
        """Test that output shape matches expected dimensions."""
        ob, odor_ids = input_data
        output = model(ob, odor_ids)

        assert output.shape == (BATCH_SIZE, OUT_CHANNELS, TIME_STEPS), \
            f"Expected shape {(BATCH_SIZE, OUT_CHANNELS, TIME_STEPS)}, got {output.shape}"

    def test_forward_pass_without_conditioning(self, input_data):
        """Test forward pass without odor conditioning."""
        model = LinearBaseline(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            use_conditioning=False,
        ).to(DEVICE)

        ob, _ = input_data
        output = model(ob, None)

        assert output.shape == (BATCH_SIZE, OUT_CHANNELS, TIME_STEPS)

    def test_forward_pass_with_none_odor_ids(self, model, input_data):
        """Test forward pass with conditioning enabled but no odor_ids provided."""
        ob, _ = input_data
        output = model(ob, None)

        assert output.shape == (BATCH_SIZE, OUT_CHANNELS, TIME_STEPS)

    def test_gradient_flow(self, model, input_data):
        """Test that gradients flow through the model."""
        ob, odor_ids = input_data
        ob.requires_grad_(True)

        output = model(ob, odor_ids)
        loss = output.mean()
        loss.backward()

        # Check gradient exists and is finite
        assert ob.grad is not None, "No gradient computed for input"
        assert torch.isfinite(ob.grad).all(), "Gradient contains NaN or Inf"

        # Check model gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN/Inf"

    def test_no_nan_inf_output(self, model, input_data):
        """Test that output contains no NaN or Inf values."""
        ob, odor_ids = input_data
        output = model(ob, odor_ids)

        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_residual_connection_initial(self, model, input_data):
        """Test that at initialization, output ≈ input (residual behavior)."""
        ob, _ = input_data

        # Reinitialize with zeros
        model = LinearBaseline(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            use_conditioning=False,
            use_residual=True,
        ).to(DEVICE)

        output = model(ob, None)

        # With proper initialization, output should be close to input
        # (delta ≈ 0, so out ≈ ob + delta ≈ ob)
        diff = (output - ob).abs().mean()
        assert diff < 0.5, f"Initial output differs from input by {diff:.4f}"

    def test_conditioning_effect(self, model, input_data):
        """Test that different odor_ids produce different outputs."""
        ob, _ = input_data

        # Same input, different odor IDs
        odor_0 = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
        odor_1 = torch.ones(BATCH_SIZE, dtype=torch.long, device=DEVICE)

        out_0 = model(ob, odor_0)
        out_1 = model(ob, odor_1)

        # Outputs should differ
        diff = (out_0 - out_1).abs().mean()
        assert diff > 1e-6, "Conditioning has no effect on output"

    def test_input_validation_wrong_dims(self, model):
        """Test that wrong input dimensions raise an error."""
        wrong_input = torch.randn(BATCH_SIZE, IN_CHANNELS, device=DEVICE)  # Missing T dimension

        with pytest.raises(ValueError, match="Expected 3D input"):
            model(wrong_input, None)

    def test_input_validation_wrong_channels(self, model):
        """Test that wrong number of channels raises an error."""
        wrong_input = torch.randn(BATCH_SIZE, IN_CHANNELS + 1, TIME_STEPS, device=DEVICE)

        with pytest.raises(ValueError, match="Expected .* input channels"):
            model(wrong_input, None)

    def test_parameter_count(self, model):
        """Test parameter counting."""
        counts = model.count_parameters()

        assert 'total' in counts
        assert 'trainable' in counts
        assert counts['total'] > 0
        assert counts['trainable'] == counts['total']  # All params should be trainable

    def test_different_channel_sizes(self):
        """Test with different input/output channel sizes."""
        model = LinearBaseline(
            in_channels=16,
            out_channels=32,
            use_conditioning=True,
        ).to(DEVICE)

        ob = torch.randn(BATCH_SIZE, 16, TIME_STEPS, device=DEVICE)
        odor_ids = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)

        output = model(ob, odor_ids)
        assert output.shape == (BATCH_SIZE, 32, TIME_STEPS)


class TestLinearChannelMixer:
    """Tests for LinearChannelMixer model."""

    @pytest.fixture
    def model(self):
        """Create a default LinearChannelMixer model."""
        return LinearChannelMixer(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            num_classes=NUM_CLASSES,
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

    def test_gradient_flow(self, model, input_data):
        """Test gradient flow."""
        ob, odor_ids = input_data
        ob.requires_grad_(True)

        output = model(ob, odor_ids)
        loss = output.mean()
        loss.backward()

        assert ob.grad is not None
        assert torch.isfinite(ob.grad).all()

    def test_no_nan_inf(self, model, input_data):
        """Test no NaN/Inf in output."""
        ob, odor_ids = input_data
        output = model(ob, odor_ids)

        assert torch.isfinite(output).all()


class TestFactoryFunction:
    """Tests for the create_linear_baseline factory function."""

    def test_create_simple(self):
        """Test creating simple variant."""
        model = create_linear_baseline("simple")
        assert isinstance(model, LinearBaseline)
        assert not model.shared_weights

    def test_create_shared(self):
        """Test creating shared weights variant."""
        model = create_linear_baseline("shared")
        assert isinstance(model, LinearBaseline)
        assert model.shared_weights

    def test_create_mixer(self):
        """Test creating mixer variant."""
        model = create_linear_baseline("mixer")
        assert isinstance(model, LinearChannelMixer)

    def test_create_invalid(self):
        """Test that invalid variant raises error."""
        with pytest.raises(ValueError, match="Unknown variant"):
            create_linear_baseline("invalid_variant")

    def test_kwargs_passthrough(self):
        """Test that kwargs are passed to model."""
        model = create_linear_baseline("simple", in_channels=16, out_channels=64)
        assert model.in_channels == 16
        assert model.out_channels == 64


class TestMemoryAndPerformance:
    """Tests for memory usage and performance characteristics."""

    def test_memory_footprint(self):
        """Test that model has reasonable memory footprint."""
        model = LinearBaseline(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
        ).to(DEVICE)

        counts = model.count_parameters()

        # Linear baseline should have << 10M parameters
        assert counts['total'] < 10_000_000, \
            f"Linear baseline has too many parameters: {counts['total']}"

        # Should have very few parameters (mostly the 1x1 conv)
        expected_max = IN_CHANNELS * OUT_CHANNELS + OUT_CHANNELS + NUM_CLASSES * OUT_CHANNELS
        assert counts['total'] <= expected_max * 2, \
            f"Parameter count {counts['total']} exceeds expected max {expected_max * 2}"

    def test_deterministic_output(self):
        """Test that model produces deterministic output."""
        model = LinearBaseline().to(DEVICE).eval()

        torch.manual_seed(42)
        ob = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        odor_ids = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)

        with torch.no_grad():
            out1 = model(ob, odor_ids)
            out2 = model(ob, odor_ids)

        assert torch.allclose(out1, out2), "Model output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
