#!/usr/bin/env python3
"""
Tier 2.5: Probabilistic Loss Ablation
======================================

Tests probabilistic losses that model signal distributions:
    - Rayleigh: For signal envelope (amplitude) distribution
    - von Mises: For phase distribution (circular statistics)
    - Chi-squared: For power distribution
    - Gumbel: For extreme values (peaks/troughs)
    - Gamma: For positive-valued signals
    - Log-Normal: For multiplicative noise

These are combined with base losses to see if probabilistic modeling helps.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.common.cross_validation import create_data_splits
from experiments.common.config_registry import (
    get_registry,
    Tier2_5Result,
    ProbabilisticLossResult,
)


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tier2_5_probabilistic"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Probabilistic losses to test
PROBABILISTIC_LOSSES = [
    "none",           # Base loss only (baseline)
    "rayleigh",       # Envelope distribution
    "von_mises",      # Phase distribution
    "chi_squared",    # Power distribution
    "gumbel",         # Peak distribution
    "gamma",          # Positive-valued
    "log_normal",     # Multiplicative noise
    "mixture",        # Gaussian mixture
]

# Base losses to combine with
BASE_LOSSES = ["huber", "spectral"]

FAST_PROB_LOSSES = ["none", "rayleigh", "von_mises"]


# =============================================================================
# Probabilistic Loss Functions
# =============================================================================

class RayleighLoss(nn.Module):
    """Rayleigh distribution loss for signal envelope.

    The envelope of neural signals often follows a Rayleigh distribution.
    This loss encourages the predicted envelope to match the target envelope distribution.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute analytic signal envelope via Hilbert transform approximation
        # Using simple smoothed absolute value as proxy
        pred_env = torch.abs(pred)
        target_env = torch.abs(target)

        # Rayleigh NLL: -log(x/sigma^2) + x^2/(2*sigma^2)
        # Estimate sigma from target
        sigma_sq = torch.mean(target_env ** 2, dim=-1, keepdim=True) / 2 + 1e-8

        # NLL for predicted envelope
        nll = -torch.log(pred_env / sigma_sq + 1e-8) + pred_env ** 2 / (2 * sigma_sq)

        return self.weight * torch.mean(nll)


class VonMisesLoss(nn.Module):
    """von Mises distribution loss for phase.

    Neural oscillations have phase relationships that follow circular statistics.
    von Mises is the circular analog of the Gaussian distribution.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute instantaneous phase via Hilbert transform approximation
        # Using gradient as proxy for phase derivative
        pred_phase = torch.atan2(
            F.pad(pred[..., 1:] - pred[..., :-1], (0, 1)),
            pred + 1e-8
        )
        target_phase = torch.atan2(
            F.pad(target[..., 1:] - target[..., :-1], (0, 1)),
            target + 1e-8
        )

        # Phase difference
        phase_diff = pred_phase - target_phase

        # von Mises loss: -kappa * cos(phase_diff)
        # Using kappa=1 for simplicity
        loss = 1.0 - torch.cos(phase_diff)

        return self.weight * torch.mean(loss)


class ChiSquaredLoss(nn.Module):
    """Chi-squared distribution loss for power.

    The power (squared amplitude) of neural signals follows chi-squared distribution.
    """

    def __init__(self, weight: float = 0.1, df: int = 2):
        super().__init__()
        self.weight = weight
        self.df = df  # degrees of freedom

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Power
        pred_power = pred ** 2 + 1e-8
        target_power = target ** 2 + 1e-8

        # Chi-squared NLL (using Gamma with shape=df/2, scale=2)
        k = self.df / 2
        # Simplified: match power distributions via KL-like divergence
        loss = pred_power / target_power - torch.log(pred_power / target_power + 1e-8) - 1

        return self.weight * torch.mean(loss)


class GumbelLoss(nn.Module):
    """Gumbel distribution loss for peaks/extreme values.

    Peak values in neural signals follow extreme value distributions.
    """

    def __init__(self, weight: float = 0.1, percentile: float = 95):
        super().__init__()
        self.weight = weight
        self.percentile = percentile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Focus on peak values
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        # Get top percentile values
        k = int((1 - self.percentile / 100) * pred_flat.shape[-1])
        k = max(k, 1)

        pred_peaks, _ = torch.topk(pred_flat, k, dim=-1)
        target_peaks, _ = torch.topk(target_flat, k, dim=-1)

        # Match peak distributions
        # Gumbel location and scale estimation
        target_mu = torch.mean(target_peaks, dim=-1, keepdim=True)
        target_beta = torch.std(target_peaks, dim=-1, keepdim=True) * np.sqrt(6) / np.pi + 1e-8

        # Gumbel NLL
        z = (pred_peaks - target_mu) / target_beta
        nll = z + torch.exp(-z) + torch.log(target_beta)

        return self.weight * torch.mean(nll)


class GammaLoss(nn.Module):
    """Gamma distribution loss for positive-valued signals."""

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure positive
        pred_pos = F.softplus(pred) + 1e-8
        target_pos = torch.abs(target) + 1e-8

        # Estimate gamma parameters from target
        target_mean = torch.mean(target_pos, dim=-1, keepdim=True)
        target_var = torch.var(target_pos, dim=-1, keepdim=True) + 1e-8

        # Method of moments: shape = mean^2/var, rate = mean/var
        shape = target_mean ** 2 / target_var
        rate = target_mean / target_var

        # Gamma NLL
        nll = -shape * torch.log(rate) - (shape - 1) * torch.log(pred_pos) + rate * pred_pos

        return self.weight * torch.mean(nll)


class LogNormalLoss(nn.Module):
    """Log-normal distribution loss for multiplicative processes."""

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Transform to log space (handle negatives)
        pred_log = torch.log(torch.abs(pred) + 1e-8)
        target_log = torch.log(torch.abs(target) + 1e-8)

        # Estimate parameters
        mu = torch.mean(target_log, dim=-1, keepdim=True)
        sigma = torch.std(target_log, dim=-1, keepdim=True) + 1e-8

        # Log-normal NLL
        nll = pred_log + ((pred_log - mu) ** 2) / (2 * sigma ** 2)

        return self.weight * torch.mean(nll)


class GaussianMixtureLoss(nn.Module):
    """Gaussian Mixture Model loss for multi-modal distributions."""

    def __init__(self, weight: float = 0.1, n_components: int = 3):
        super().__init__()
        self.weight = weight
        self.n_components = n_components

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, T = target.shape

        # Flatten
        target_flat = target.reshape(B, -1)

        # Simple k-means style estimation of mixture components
        # Use quantiles as component centers
        quantiles = torch.linspace(0.1, 0.9, self.n_components).to(target.device)
        centers = torch.quantile(target_flat, quantiles, dim=-1).T  # [B, K]
        sigma = torch.std(target_flat, dim=-1, keepdim=True) / self.n_components + 1e-8

        # Compute responsibilities for predicted values
        pred_flat = pred.reshape(B, -1)

        # Distance to each component
        dists = (pred_flat.unsqueeze(-1) - centers.unsqueeze(1)) ** 2  # [B, N, K]

        # Soft assignment
        log_probs = -dists / (2 * sigma.unsqueeze(-1) ** 2)
        log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)

        # NLL (negative log likelihood of mixture)
        nll = -torch.logsumexp(log_probs, dim=-1)

        return self.weight * torch.mean(nll)


PROB_LOSS_REGISTRY = {
    "rayleigh": RayleighLoss,
    "von_mises": VonMisesLoss,
    "chi_squared": ChiSquaredLoss,
    "gumbel": GumbelLoss,
    "gamma": GammaLoss,
    "log_normal": LogNormalLoss,
    "mixture": GaussianMixtureLoss,
}


# =============================================================================
# Combined Loss
# =============================================================================

class CombinedProbabilisticLoss(nn.Module):
    """Combines base loss with probabilistic loss."""

    def __init__(
        self,
        base_loss: str = "huber",
        prob_loss: str = "none",
        prob_weight: float = 0.1,
    ):
        super().__init__()

        # Base loss
        if base_loss == "huber":
            self.base = nn.HuberLoss()
        elif base_loss == "mse":
            self.base = nn.MSELoss()
        elif base_loss == "l1":
            self.base = nn.L1Loss()
        elif base_loss == "spectral":
            self.base = SpectralLoss()
        else:
            self.base = nn.HuberLoss()

        # Probabilistic loss
        if prob_loss == "none" or prob_loss not in PROB_LOSS_REGISTRY:
            self.prob = None
        else:
            self.prob = PROB_LOSS_REGISTRY[prob_loss](weight=prob_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base(pred, target)

        if self.prob is not None:
            loss = loss + self.prob(pred, target)

        return loss


class SpectralLoss(nn.Module):
    """Multi-scale spectral loss."""

    def __init__(self):
        super().__init__()
        self.fft_sizes = [256, 512, 1024]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0

        for fft_size in self.fft_sizes:
            if pred.shape[-1] < fft_size:
                continue

            pred_fft = torch.fft.rfft(pred, n=fft_size, dim=-1)
            target_fft = torch.fft.rfft(target, n=fft_size, dim=-1)

            # Magnitude loss
            pred_mag = torch.abs(pred_fft)
            target_mag = torch.abs(target_fft)

            loss = loss + F.l1_loss(pred_mag, target_mag)

            # Log magnitude loss
            pred_log = torch.log(pred_mag + 1e-8)
            target_log = torch.log(target_mag + 1e-8)

            loss = loss + F.l1_loss(pred_log, target_log)

        return loss / len(self.fft_sizes)


# =============================================================================
# Training
# =============================================================================

def train_with_prob_loss(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    base_loss: str,
    prob_loss: str,
    n_epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> Tuple[float, float]:
    """Train with probabilistic loss combination."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = CombinedProbabilisticLoss(base_loss=base_loss, prob_loss=prob_loss)

    model.train()
    for epoch in range(n_epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

    # Validation
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    preds_flat = preds.reshape(preds.shape[0], -1)
    targets_flat = targets.reshape(targets.shape[0], -1)

    ss_res = np.sum((targets_flat - preds_flat) ** 2)
    ss_tot = np.sum((targets_flat - np.mean(targets_flat, axis=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    mae = np.mean(np.abs(targets_flat - preds_flat))

    return float(r2), float(mae)


# =============================================================================
# Main Tier Function
# =============================================================================

def run_tier2_5(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    architecture: Optional[str] = None,
    base_losses: Optional[List[str]] = None,
    prob_losses: Optional[List[str]] = None,
    n_seeds: int = 3,
    n_folds: int = 3,
    n_epochs: int = 50,
    batch_size: int = 32,
    register: bool = True,
) -> Tier2_5Result:
    """Run Tier 2.5: Probabilistic Loss Ablation.

    Tests combinations of base losses with probabilistic losses.
    """
    if architecture is None:
        registry = get_registry()
        if registry.tier2 is not None:
            architecture = registry.tier2.best_architecture
        elif registry.tier1 is not None:
            architecture = registry.tier1.top_architectures[0]
        else:
            architecture = "unet"

    base_losses = base_losses or BASE_LOSSES
    prob_losses = prob_losses or PROBABILISTIC_LOSSES

    print(f"\nTesting {len(base_losses)} base × {len(prob_losses)} probabilistic losses")
    print(f"Architecture: {architecture}")

    N = X.shape[0]
    in_channels = X.shape[1]
    out_channels = y.shape[1]

    splits = create_data_splits(n_samples=N, n_folds=n_folds, holdout_fraction=0.0, seed=42)

    all_results = {}

    for base_loss in base_losses:
        for prob_loss in prob_losses:
            combo_name = f"{base_loss}+{prob_loss}" if prob_loss != "none" else base_loss
            print(f"\n  {combo_name}...", end=" ", flush=True)

            r2s = []

            for seed in range(n_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                for split in splits.cv_splits:
                    X_train = X[split.train_indices]
                    y_train = y[split.train_indices]
                    X_val = X[split.val_indices]
                    y_val = y[split.val_indices]

                    train_loader = DataLoader(
                        TensorDataset(X_train, y_train),
                        batch_size=batch_size,
                        shuffle=True,
                    )
                    val_loader = DataLoader(
                        TensorDataset(X_val, y_val),
                        batch_size=batch_size,
                    )

                    model = None
                    try:
                        from models import create_architecture, UNet1DConditioned

                        VARIANT_MAP = {
                            "linear": "simple", "cnn": "basic", "wavenet": "standard",
                            "fnet": "standard", "vit": "standard", "performer": "standard",
                            "mamba": "standard",
                        }

                        if architecture == "unet":
                            model = UNet1DConditioned(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                base_channels=64,
                            ).to(device)
                        else:
                            model = create_architecture(
                                architecture,
                                variant=VARIANT_MAP.get(architecture, "standard"),
                                in_channels=in_channels,
                                out_channels=out_channels,
                            ).to(device)

                        r2, mae = train_with_prob_loss(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            base_loss=base_loss,
                            prob_loss=prob_loss,
                            n_epochs=n_epochs,
                            device=device,
                        )
                        r2s.append(r2)

                    except Exception as e:
                        print(f"Error: {e}")

                    finally:
                        if model is not None:
                            del model
                        gc.collect()
                        torch.cuda.empty_cache()

            if r2s:
                all_results[combo_name] = ProbabilisticLossResult(
                    base_loss=base_loss,
                    prob_loss=prob_loss,
                    r2_mean=float(np.mean(r2s)),
                    r2_std=float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
                )
                print(f"R² = {all_results[combo_name].r2_mean:.4f}")

    # Find best
    best = max(all_results.values(), key=lambda x: x.r2_mean)

    result = Tier2_5Result(
        architecture=architecture,
        results=all_results,
        best_combination=f"{best.base_loss}+{best.prob_loss}" if best.prob_loss != "none" else best.base_loss,
        best_r2_mean=best.r2_mean,
        best_r2_std=best.r2_std,
    )

    if register:
        registry = get_registry(ARTIFACTS_DIR)
        registry.register_tier2_5(result)

    # Print summary
    print("\n" + "=" * 60)
    print("TIER 2.5 RESULTS: Probabilistic Loss Ablation")
    print("=" * 60)

    sorted_results = sorted(all_results.values(), key=lambda x: x.r2_mean, reverse=True)
    for r in sorted_results[:10]:
        name = f"{r.base_loss}+{r.prob_loss}" if r.prob_loss != "none" else r.base_loss
        marker = " <-- BEST" if r == best else ""
        print(f"  {name:25s}: R² = {r.r2_mean:.4f} ± {r.r2_std:.4f}{marker}")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tier 2.5: Probabilistic Loss Ablation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TIER 2.5: Probabilistic Loss Ablation")
    print("=" * 60)

    if args.dry_run:
        N, C, T = 100, 32, 500
        X = torch.randn(N, C, T)
        y = torch.randn(N, C, T)
        prob_losses = FAST_PROB_LOSSES
        n_epochs, n_seeds, n_folds = 5, 2, 2
    else:
        from data import prepare_data
        data = prepare_data()
        idx = np.concatenate([data["train_idx"], data["val_idx"]])
        X = torch.from_numpy(data["ob"][idx]).float()
        y = torch.from_numpy(data["pcx"][idx]).float()
        prob_losses = PROBABILISTIC_LOSSES
        n_epochs, n_seeds, n_folds = 50, 3, 3

    run_tier2_5(
        X=X, y=y, device=device,
        prob_losses=prob_losses,
        n_seeds=n_seeds, n_folds=n_folds, n_epochs=n_epochs,
    )


if __name__ == "__main__":
    main()
