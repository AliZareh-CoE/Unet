"""
Tests for ConditionalDistributionBlock
======================================

Comprehensive tests for the distribution correction block including:
- Identity behavior with α=0
- Rayleigh envelope correction with α=1
- Gradient flow verification
- Gradient isolation (no flow back to UNet)
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import (
    ConditionalDistributionBlock,
    hilbert_torch,
    soft_rank,
    rayleigh_quantile,
    CondUNet1D,
    SpectroTemporalEncoder,
)


# Test configuration
BATCH_SIZE = 4
IN_CHANNELS = 32
OUT_CHANNELS = 32
TIME_STEPS = 512  # Shorter for faster tests
CONDITION_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestHilbertTransform:
    """Tests for differentiable Hilbert transform."""

    def test_hilbert_output_shape(self):
        """Test Hilbert transform output shape."""
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        analytic = hilbert_torch(x)
        assert analytic.shape == x.shape
        assert analytic.is_complex()

    def test_hilbert_envelope_positive(self):
        """Test that envelope is always positive."""
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        analytic = hilbert_torch(x)
        envelope = torch.abs(analytic)
        assert (envelope >= 0).all()

    def test_hilbert_real_part_preserved(self):
        """Test that real part of analytic signal equals original."""
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        analytic = hilbert_torch(x)
        real_part = analytic.real
        # Should be approximately equal (small numerical errors)
        assert torch.allclose(real_part, x, atol=1e-5)

    def test_hilbert_gradients_flow(self):
        """Test that gradients flow through Hilbert transform."""
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE, requires_grad=True)
        analytic = hilbert_torch(x)
        envelope = torch.abs(analytic)
        loss = envelope.mean()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestSoftRank:
    """Tests for differentiable soft ranking."""

    def test_soft_rank_shape(self):
        """Test soft rank output shape."""
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        ranks = soft_rank(x, temperature=1.0)
        assert ranks.shape == x.shape

    def test_soft_rank_range(self):
        """Test soft rank values are in expected range."""
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        ranks = soft_rank(x, temperature=1.0)
        # Ranks should be roughly in [0, T-1]
        assert ranks.min() >= -1.0
        assert ranks.max() <= TIME_STEPS

    def test_soft_rank_ordering(self):
        """Test that soft ranking preserves order for distinct values."""
        # Simple case: sorted input
        x = torch.arange(10, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        ranks = soft_rank(x, temperature=0.01)  # Low temperature for hard ranking
        # Ranks should be increasing
        rank_diffs = ranks[..., 1:] - ranks[..., :-1]
        assert (rank_diffs > 0).all()

    def test_soft_rank_gradients(self):
        """Test gradients flow through soft rank."""
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, 64, device=DEVICE, requires_grad=True)
        ranks = soft_rank(x, temperature=1.0)
        loss = ranks.mean()
        loss.backward()
        assert x.grad is not None


class TestRayleighQuantile:
    """Tests for Rayleigh inverse CDF."""

    def test_rayleigh_quantile_shape(self):
        """Test Rayleigh quantile output shape."""
        p = torch.rand(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        q = rayleigh_quantile(p, sigma=1.0)
        assert q.shape == p.shape

    def test_rayleigh_quantile_positive(self):
        """Test Rayleigh quantiles are positive."""
        p = torch.rand(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        q = rayleigh_quantile(p, sigma=1.0)
        assert (q >= 0).all()

    def test_rayleigh_quantile_monotonic(self):
        """Test Rayleigh quantile function is monotonic."""
        p = torch.linspace(0.01, 0.99, 100, device=DEVICE)
        q = rayleigh_quantile(p, sigma=1.0)
        q_diffs = q[1:] - q[:-1]
        assert (q_diffs > 0).all()

    def test_rayleigh_quantile_known_values(self):
        """Test known values of Rayleigh quantile."""
        # F^{-1}(0.5) = σ * sqrt(-2 * ln(0.5)) = σ * sqrt(2 * ln(2)) ≈ 1.177 * σ
        p = torch.tensor([0.5], device=DEVICE)
        q = rayleigh_quantile(p, sigma=1.0)
        expected = math.sqrt(2 * math.log(2))
        assert torch.allclose(q, torch.tensor([expected], device=DEVICE), atol=0.01)


class TestConditionalDistributionBlock:
    """Tests for ConditionalDistributionBlock."""

    @pytest.fixture
    def distribution_block(self):
        """Create a default distribution block."""
        return ConditionalDistributionBlock(
            condition_dim=CONDITION_DIM,
            alpha_init=0.5,
            learnable_alpha=True,
            soft_rank_temperature=0.1,
            hidden_dim=64,
        ).to(DEVICE)

    @pytest.fixture
    def input_data(self):
        """Create test input data."""
        y_hat = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        condition = torch.randn(BATCH_SIZE, CONDITION_DIM, device=DEVICE)
        return y_hat, condition

    def test_output_shape(self, distribution_block, input_data):
        """Test output shape matches input."""
        y_hat, condition = input_data
        y_corrected = distribution_block(y_hat, condition)
        assert y_corrected.shape == y_hat.shape

    def test_alpha_zero_is_identity(self):
        """Test that α=0 produces identity transform (output ≈ input)."""
        # Create block with fixed α=0
        block = ConditionalDistributionBlock(
            condition_dim=CONDITION_DIM,
            alpha_init=0.0,
            learnable_alpha=False,
        ).to(DEVICE)

        y_hat = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)
        condition = torch.randn(BATCH_SIZE, CONDITION_DIM, device=DEVICE)

        y_corrected = block(y_hat, condition)

        # With α=0, the corrected signal should be approximately reconstructed
        # from original envelope and phase: y ≈ |y| * cos(angle(analytic(y)))
        # Due to Hilbert transform reconstruction, there may be small differences
        # but the envelope should be preserved
        analytic = hilbert_torch(y_hat.float())
        orig_envelope = torch.abs(analytic)
        corrected_analytic = hilbert_torch(y_corrected.float())
        corrected_envelope = torch.abs(corrected_analytic)

        # Envelopes should match closely
        assert torch.allclose(orig_envelope, corrected_envelope, atol=0.1)

    def test_alpha_one_increases_rayleigh(self):
        """Test that α=1 makes envelope more Rayleigh-like."""
        # Create block with fixed α=1
        block = ConditionalDistributionBlock(
            condition_dim=CONDITION_DIM,
            alpha_init=1.0,
            learnable_alpha=False,
        ).to(DEVICE)

        # Create input with non-Rayleigh envelope (e.g., uniform)
        y_hat = torch.rand(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE) * 2 - 1
        condition = torch.randn(BATCH_SIZE, CONDITION_DIM, device=DEVICE)

        y_corrected = block(y_hat, condition)

        # Compute Rayleigh statistic: CV² = Var(A) / Mean(A)²
        # For Rayleigh: CV² = (4 - π) / π ≈ 0.273
        target_cv_squared = (4.0 - math.pi) / math.pi

        corrected_analytic = hilbert_torch(y_corrected.float())
        corrected_envelope = torch.abs(corrected_analytic)

        env_mean = corrected_envelope.mean(dim=-1)
        env_var = corrected_envelope.var(dim=-1)
        cv_squared = env_var / (env_mean.pow(2) + 1e-8)

        # Should be closer to Rayleigh target
        mean_cv_squared = cv_squared.mean().item()
        # Allow some tolerance since soft ranking is approximate
        assert abs(mean_cv_squared - target_cv_squared) < 0.5

    def test_learnable_alpha(self, input_data):
        """Test that learnable alpha changes with training."""
        block = ConditionalDistributionBlock(
            condition_dim=CONDITION_DIM,
            alpha_init=0.5,
            learnable_alpha=True,
        ).to(DEVICE)

        y_hat, condition = input_data

        # Get initial alpha
        initial_alpha = block.get_alpha(condition).mean().item()

        # Forward pass and compute loss
        y_corrected = block(y_hat, condition)
        losses = block.compute_losses(y_corrected)
        loss = losses["envelope_loss"] + losses["phase_loss"]

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert block.alpha_net[-1].weight.grad is not None

        # Update weights
        with torch.no_grad():
            for param in block.parameters():
                if param.grad is not None:
                    param -= 0.1 * param.grad

        # Get updated alpha
        updated_alpha = block.get_alpha(condition).mean().item()

        # Alpha should have changed
        assert initial_alpha != updated_alpha

    def test_gradients_flow(self, distribution_block, input_data):
        """Test that gradients flow through the block."""
        y_hat, condition = input_data
        y_hat.requires_grad_(True)
        condition.requires_grad_(True)

        y_corrected = distribution_block(y_hat, condition)
        loss = y_corrected.mean()
        loss.backward()

        assert y_hat.grad is not None
        assert condition.grad is not None

    def test_compute_losses_shape(self, distribution_block, input_data):
        """Test compute_losses returns correct structure."""
        y_hat, condition = input_data
        y_corrected = distribution_block(y_hat, condition)
        losses = distribution_block.compute_losses(y_corrected)

        assert "envelope_loss" in losses
        assert "phase_loss" in losses
        assert losses["envelope_loss"].shape == ()  # Scalar
        assert losses["phase_loss"].shape == ()  # Scalar


class TestGradientIsolation:
    """Tests for gradient isolation between UNet and distribution block."""

    @pytest.fixture
    def unet(self):
        """Create a simple UNet for testing."""
        return CondUNet1D(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            base=64,
            n_odors=7,
            emb_dim=CONDITION_DIM,
            n_downsample=2,
            conv_type="standard",
        ).to(DEVICE)

    @pytest.fixture
    def cond_encoder(self):
        """Create conditioning encoder."""
        return SpectroTemporalEncoder(
            in_channels=IN_CHANNELS,
            emb_dim=CONDITION_DIM,
        ).to(DEVICE)

    @pytest.fixture
    def distribution_block(self):
        """Create distribution block."""
        return ConditionalDistributionBlock(
            condition_dim=CONDITION_DIM,
            alpha_init=0.5,
            learnable_alpha=True,
        ).to(DEVICE)

    def test_unet_gradients_not_affected_by_distribution_loss(
        self, unet, cond_encoder, distribution_block
    ):
        """Test that distribution loss does NOT update UNet gradients."""
        # Input data
        ob = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)

        # Forward pass through conditioning encoder
        cond_emb = cond_encoder(ob)

        # Forward pass through UNet
        y_hat = unet(ob, cond_emb=cond_emb)

        # CRITICAL: Detach before distribution block
        y_hat_detached = y_hat.detach()

        # Forward through distribution block with detached input
        y_corrected = distribution_block(y_hat_detached, cond_emb.detach())

        # Compute distribution loss
        losses = distribution_block.compute_losses(y_corrected)
        dist_loss = losses["envelope_loss"] + losses["phase_loss"]

        # Backward on distribution loss
        dist_loss.backward()

        # UNet should have NO gradients (because we detached)
        for name, param in unet.named_parameters():
            assert param.grad is None, f"UNet param {name} has gradient but shouldn't!"

        # Distribution block SHOULD have gradients
        has_grad = False
        for param in distribution_block.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "Distribution block should have gradients"

    def test_unet_weights_unchanged_after_distribution_update(
        self, unet, cond_encoder, distribution_block
    ):
        """Test that UNet weights remain unchanged when only distribution loss is backpropagated."""
        # Store initial UNet weights
        initial_weights = {}
        for name, param in unet.named_parameters():
            initial_weights[name] = param.clone().detach()

        # Input data
        ob = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)

        # Forward pass through conditioning encoder
        cond_emb = cond_encoder(ob)

        # Forward pass through UNet
        y_hat = unet(ob, cond_emb=cond_emb)

        # CRITICAL: Detach before distribution block
        y_hat_detached = y_hat.detach()

        # Forward through distribution block
        y_corrected = distribution_block(y_hat_detached, cond_emb.detach())

        # Compute and backward distribution loss
        losses = distribution_block.compute_losses(y_corrected)
        dist_loss = losses["envelope_loss"] + losses["phase_loss"]
        dist_loss.backward()

        # Create optimizer for distribution block only
        optimizer = torch.optim.Adam(distribution_block.parameters(), lr=0.01)
        optimizer.step()

        # Verify UNet weights unchanged
        for name, param in unet.named_parameters():
            assert torch.equal(
                param, initial_weights[name]
            ), f"UNet param {name} changed but shouldn't!"

    def test_distribution_block_weights_do_change(
        self, unet, cond_encoder, distribution_block
    ):
        """Test that distribution block weights DO change with optimizer step."""
        # Store initial distribution block weights
        initial_weights = {}
        for name, param in distribution_block.named_parameters():
            initial_weights[name] = param.clone().detach()

        # Input data
        ob = torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS, device=DEVICE)

        # Forward pass
        cond_emb = cond_encoder(ob)
        y_hat = unet(ob, cond_emb=cond_emb)
        y_corrected = distribution_block(y_hat.detach(), cond_emb.detach())

        # Backward
        losses = distribution_block.compute_losses(y_corrected)
        dist_loss = losses["envelope_loss"] + losses["phase_loss"]
        dist_loss.backward()

        # Optimizer step
        optimizer = torch.optim.Adam(distribution_block.parameters(), lr=0.1)
        optimizer.step()

        # Verify at least some weights changed
        weights_changed = False
        for name, param in distribution_block.named_parameters():
            if not torch.equal(param, initial_weights[name]):
                weights_changed = True
                break

        assert weights_changed, "Distribution block weights should change after optimizer step"


class TestMemoryEfficiency:
    """Tests for memory efficiency of soft ranking."""

    def test_soft_rank_memory_scales_reasonably(self):
        """Test that soft ranking doesn't explode memory for moderate sequences."""
        # Use smaller sequence for memory test
        T = 256  # Reasonable sequence length

        x = torch.randn(2, 4, T, device=DEVICE)

        # Should not raise out-of-memory error
        try:
            ranks = soft_rank(x, temperature=0.1)
            assert ranks.shape == x.shape
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("Not enough GPU memory for soft ranking test")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
