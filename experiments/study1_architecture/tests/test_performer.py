"""
Tests for Performer 1D Architecture
===================================

Comprehensive tests for Nature Methods quality assurance.
"""

from __future__ import annotations

import pytest
import torch

from experiments.study1_architecture.architectures.performer_1d import (
    Performer1D,
    PerformerBlock,
    FAVORPlusAttention,
    create_performer,
)


BATCH_SIZE = 4
IN_CHANNELS = 32
OUT_CHANNELS = 32
TIME_STEPS = 2048
NUM_CLASSES = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestFAVORPlusAttention:
    """Tests for FAVOR+ attention."""

    def test_output_shape(self):
        """Test attention output shape."""
        attn = FAVORPlusAttention(256, num_heads=8).to(DEVICE)
        x = torch.randn(BATCH_SIZE, 64, 256, device=DEVICE)

        out = attn(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradient flow."""
        attn = FAVORPlusAttention(128, num_heads=4).to(DEVICE)
        x = torch.randn(2, 32, 128, device=DEVICE, requires_grad=True)

        out = attn(x)
        loss = out.mean()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_long_sequence(self):
        """Test with long sequences (advantage of linear attention)."""
        attn = FAVORPlusAttention(128, num_heads=4).to(DEVICE)
        x = torch.randn(2, 1024, 128, device=DEVICE)  # Long sequence

        out = attn(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


class TestPerformerBlock:
    """Tests for Performer block."""

    def test_output_shape(self):
        """Test block output shape."""
        block = PerformerBlock(256, num_heads=8).to(DEVICE)
        x = torch.randn(BATCH_SIZE, 64, 256, device=DEVICE)

        out = block(x)
        assert out.shape == x.shape

    def test_with_conditioning(self):
        """Test with conditioning."""
        block = PerformerBlock(256, condition_dim=64).to(DEVICE)
        x = torch.randn(BATCH_SIZE, 64, 256, device=DEVICE)
        cond = torch.randn(BATCH_SIZE, 64, device=DEVICE)

        out = block(x, cond)
        assert out.shape == x.shape


class TestPerformer1D:
    """Tests for Performer1D model."""

    @pytest.fixture
    def model(self):
        """Create default model."""
        return Performer1D(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            patch_size=16,
            embed_dim=128,
            num_layers=4,
            num_heads=4,
            num_classes=NUM_CLASSES,
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
        assert output.shape == (BATCH_SIZE, OUT_CHANNELS, TIME_STEPS)

    def test_forward_without_conditioning(self, input_data):
        """Test without conditioning."""
        model = Performer1D(
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
        """Test residual at initialization."""
        model = Performer1D(
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

    def test_variable_time_lengths(self, model):
        """Test various time lengths."""
        for T in [512, 1024, 2048]:
            x = torch.randn(2, IN_CHANNELS, T, device=DEVICE)
            out = model(x, None)
            assert out.shape == x.shape


class TestFactoryFunction:
    """Tests for create_performer factory."""

    def test_create_standard(self):
        """Test standard variant."""
        model = create_performer("standard")
        assert isinstance(model, Performer1D)

    def test_create_fast(self):
        """Test fast variant."""
        model = create_performer("fast")
        assert isinstance(model, Performer1D)

    def test_invalid_variant(self):
        """Test invalid variant."""
        with pytest.raises(ValueError):
            create_performer("invalid")

    def test_kwargs(self):
        """Test kwargs passthrough."""
        model = create_performer("standard", embed_dim=512)
        assert model.embed_dim == 512


class TestMemoryAndPerformance:
    """Memory and performance tests."""

    def test_parameter_budget(self):
        """Test Performer is within budget."""
        model = Performer1D(
            in_channels=32,
            out_channels=32,
            embed_dim=256,
            num_layers=6,
        )
        counts = model.count_parameters()
        assert counts['total'] < 200_000_000

    def test_deterministic(self):
        """Test deterministic output."""
        model = Performer1D(num_layers=2).to(DEVICE).eval()
        x = torch.randn(2, 32, 512, device=DEVICE)
        odor = torch.tensor([0, 1], device=DEVICE)

        with torch.no_grad():
            out1 = model(x, odor)
            out2 = model(x, odor)

        # Performer should be deterministic in eval mode
        assert torch.allclose(out1, out2, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
