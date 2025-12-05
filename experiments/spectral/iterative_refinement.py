"""
Iterative Refinement with R² Feedback
=====================================

Gradient-free iterative spectral correction that uses R² as feedback
to prevent over-correction. Each step is constrained to not degrade R².

Algorithm:
1. Start with predicted signal
2. Propose small spectral correction
3. Compute R² of proposed signal
4. If R² improved (or within tolerance), accept; else reduce step
5. Repeat for N iterations

This is essentially gradient-free optimization of spectral correction
with R² as the objective constraint.

Search parameters:
- n_iterations: Number of refinement iterations
- step_size: Initial step size for corrections
- r2_tolerance: Maximum allowed R² degradation
- use_momentum: Whether to use momentum in iterations
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IterativeR2ConstrainedRefinement(nn.Module):
    """Gradient-free iterative PSD correction with R² constraint.

    Each iteration proposes a spectral correction and only accepts it
    if R² doesn't degrade beyond a tolerance threshold.

    Args:
        n_channels: Number of input channels
        n_iterations: Number of refinement iterations
        step_size: Base step size for corrections
        r2_tolerance: Maximum allowed R² degradation per step
        use_momentum: Use momentum in iteration
        momentum: Momentum coefficient (if use_momentum=True)
        n_bands: Number of frequency bands for correction
        sample_rate: Sampling rate in Hz
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_iterations: int = 5,
        step_size: float = 0.1,
        r2_tolerance: float = 0.01,
        use_momentum: bool = False,
        momentum: float = 0.9,
        n_bands: int = 8,
        sample_rate: float = 1000.0,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.r2_tolerance = r2_tolerance
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.n_bands = n_bands
        self.sample_rate = sample_rate

        # Learnable per-iteration correction directions
        self.correction_directions = nn.ParameterList([
            nn.Parameter(torch.randn(n_bands) * 0.01)
            for _ in range(n_iterations)
        ])

        # Learnable per-iteration step sizes
        self.log_step_sizes = nn.Parameter(
            torch.full((n_iterations,), float(torch.tensor(step_size).log()))
        )

        # Create frequency band edges
        self.band_edges = self._create_band_edges(n_bands, 1.0, 100.0)

    def _create_band_edges(
        self,
        n_bands: int,
        f_min: float,
        f_max: float,
    ) -> List[float]:
        """Create log-spaced frequency band edges."""
        import math
        log_min = math.log10(f_min)
        log_max = math.log10(f_max)
        edges = torch.linspace(log_min, log_max, n_bands + 1)
        return (10 ** edges).tolist()

    def _compute_r2(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute R² (explained variance)."""
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        return 1 - ss_res / (ss_tot + 1e-8)

    def _build_band_masks(
        self,
        n_freqs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build frequency band masks."""
        freqs = torch.fft.rfftfreq(
            (n_freqs - 1) * 2,
            d=1/self.sample_rate,
        ).to(device)

        masks = []
        for i in range(len(self.band_edges) - 1):
            f_low = self.band_edges[i]
            f_high = self.band_edges[i + 1]
            mask = ((freqs >= f_low) & (freqs < f_high)).to(dtype)
            masks.append(mask)

        return torch.stack(masks, dim=0)  # [n_bands, n_freqs]

    def _apply_spectral_correction(
        self,
        x: torch.Tensor,
        correction_factors: torch.Tensor,
    ) -> torch.Tensor:
        """Apply spectral correction to signal.

        Args:
            x: Input signal [B, C, T]
            correction_factors: Per-band corrections [n_bands]

        Returns:
            Corrected signal [B, C, T]
        """
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        # FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        magnitude = x_fft.abs()
        phase = x_fft.angle()

        # Build masks
        n_freqs = x_fft.shape[-1]
        masks = self._build_band_masks(n_freqs, device, dtype)

        # Apply corrections
        corrected_magnitude = magnitude.clone()
        for band_idx in range(self.n_bands):
            mask = masks[band_idx].view(1, 1, -1)
            corr = torch.exp(correction_factors[band_idx])
            corrected_magnitude = corrected_magnitude * (1 + (corr - 1) * mask)

        # Reconstruct
        corrected_fft = corrected_magnitude * torch.exp(1j * phase)
        return torch.fft.irfft(corrected_fft, n=T, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply iterative R²-constrained refinement.

        Args:
            x: Input signal [B, C, T]
            target: Target signal (required for R² computation)
            odor_ids: Not used

        Returns:
            Refined signal [B, C, T]
        """
        if target is None:
            # Without target, just apply learned corrections
            return self._apply_without_constraint(x)

        current = x.clone()
        device = x.device
        dtype = x.dtype

        # Track for momentum
        prev_correction = torch.zeros(self.n_bands, device=device, dtype=dtype)

        # Initial R²
        best_r2 = self._compute_r2(current, target)
        best_signal = current.clone()

        for i in range(self.n_iterations):
            # Get correction direction for this iteration
            direction = self.correction_directions[i]
            step = torch.exp(self.log_step_sizes[i])

            # Apply momentum if enabled
            if self.use_momentum and i > 0:
                correction = step * direction + self.momentum * prev_correction
            else:
                correction = step * direction

            # Propose corrected signal
            proposed = self._apply_spectral_correction(current, correction)

            # Check R² constraint
            new_r2 = self._compute_r2(proposed, target)

            if new_r2 >= best_r2 - self.r2_tolerance:
                # Accept the correction
                current = proposed
                prev_correction = correction

                if new_r2 > best_r2:
                    best_r2 = new_r2
                    best_signal = current.clone()
            else:
                # Reject - R² degraded too much
                # Try smaller step
                for scale in [0.5, 0.25, 0.1]:
                    smaller_correction = scale * correction
                    proposed_small = self._apply_spectral_correction(
                        current, smaller_correction
                    )
                    small_r2 = self._compute_r2(proposed_small, target)

                    if small_r2 >= best_r2 - self.r2_tolerance:
                        current = proposed_small
                        prev_correction = smaller_correction
                        if small_r2 > best_r2:
                            best_r2 = small_r2
                            best_signal = current.clone()
                        break

        return best_signal

    def _apply_without_constraint(self, x: torch.Tensor) -> torch.Tensor:
        """Apply corrections without R² constraint (no target available)."""
        current = x

        for i in range(self.n_iterations):
            direction = self.correction_directions[i]
            step = torch.exp(self.log_step_sizes[i])
            correction = step * direction
            current = self._apply_spectral_correction(current, correction)

        return current


class AdaptiveIterativeRefinement(nn.Module):
    """Iterative refinement with learned stopping criterion.

    Instead of fixed iterations, learns when to stop refining
    based on signal characteristics.

    Args:
        n_channels: Number of channels
        max_iterations: Maximum refinement iterations
        hidden_dim: Hidden dimension for stopping network
        n_bands: Number of frequency bands
        sample_rate: Sampling rate in Hz
    """

    def __init__(
        self,
        n_channels: int = 32,
        max_iterations: int = 10,
        hidden_dim: int = 64,
        n_bands: int = 8,
        sample_rate: float = 1000.0,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.max_iterations = max_iterations
        self.n_bands = n_bands
        self.sample_rate = sample_rate

        # Analyzer to determine correction needed
        self.analyzer = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, 15, stride=5, padding=7),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Per-iteration correction predictor
        self.correction_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_bands),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Stopping criterion predictor
        self.stop_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Max correction magnitude
        self.max_correction = nn.Parameter(torch.tensor(0.2))

        # Band edges
        self.band_edges = self._create_band_edges(n_bands, 1.0, 100.0)

    def _create_band_edges(self, n_bands, f_min, f_max):
        import math
        log_min = math.log10(f_min)
        log_max = math.log10(f_max)
        edges = torch.linspace(log_min, log_max, n_bands + 1)
        return (10 ** edges).tolist()

    def _apply_correction(self, x, corrections):
        """Apply frequency-domain correction."""
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        x_fft = torch.fft.rfft(x, dim=-1)
        magnitude = x_fft.abs()
        phase = x_fft.angle()

        n_freqs = x_fft.shape[-1]
        freqs = torch.fft.rfftfreq((n_freqs-1)*2, d=1/self.sample_rate).to(device)

        corrected_mag = magnitude.clone()
        for i in range(self.n_bands):
            f_low = self.band_edges[i]
            f_high = self.band_edges[i + 1]
            mask = ((freqs >= f_low) & (freqs < f_high)).to(dtype).view(1, 1, -1)
            corr = 1 + corrections[:, i].view(-1, 1, 1) * self.max_correction
            corrected_mag = corrected_mag * (1 + (corr - 1) * mask)

        corrected_fft = corrected_mag * torch.exp(1j * phase)
        return torch.fft.irfft(corrected_fft, n=T, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        odor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply adaptive iterative refinement."""
        current = x

        for _ in range(self.max_iterations):
            # Analyze current signal
            features = self.analyzer(current)

            # Check stopping criterion
            stop_prob = self.stop_predictor(features)
            if not self.training and stop_prob.mean() > 0.8:
                break

            # Predict corrections
            corrections = self.correction_predictor(features)

            # Apply
            current = self._apply_correction(current, corrections)

        return current


def create_iterative_refinement_module(
    config: Dict,
    n_channels: int = 32,
    sample_rate: float = 1000.0,
) -> nn.Module:
    """Factory function to create iterative refinement module from config.

    Args:
        config: Configuration dictionary
        n_channels: Number of channels
        sample_rate: Sampling rate

    Returns:
        Configured module
    """
    use_adaptive = config.get("use_adaptive", False)

    if use_adaptive:
        return AdaptiveIterativeRefinement(
            n_channels=n_channels,
            max_iterations=config.get("n_iterations", 5),
            hidden_dim=config.get("hidden_dim", 64),
            n_bands=config.get("n_bands", 8),
            sample_rate=sample_rate,
        )
    else:
        return IterativeR2ConstrainedRefinement(
            n_channels=n_channels,
            n_iterations=config.get("n_iterations", 5),
            step_size=config.get("step_size", 0.1),
            r2_tolerance=config.get("r2_tolerance", 0.01),
            use_momentum=config.get("use_momentum", False),
            momentum=config.get("momentum", 0.9),
            n_bands=config.get("n_bands", 8),
            sample_rate=sample_rate,
        )
