"""
Literature-Standard Loss Functions for Neural Signal Translation
================================================================

Loss functions commonly used in the neural signal processing literature,
including losses from audio synthesis, domain translation, and time series.

References:
- Perceptual Loss: Johnson et al., "Perceptual Losses for Real-Time Style Transfer" (ECCV 2016)
- Adversarial Loss: Goodfellow et al., "Generative Adversarial Networks" (NeurIPS 2014)
- Contrastive Loss: Chen et al., "A Simple Framework for Contrastive Learning" (ICML 2020)
- Log-Cosh: Adaptive robust loss for regression
- Correlation Loss: Direct optimization of Pearson correlation
- SSIM: Wang et al., "Image Quality Assessment" (IEEE TIP 2004)
- Earth Mover: Arjovsky et al., "Wasserstein GAN" (ICML 2017)
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Log-Cosh Loss - Smoother than L1, more robust than MSE
# =============================================================================

class LogCoshLoss(nn.Module):
    """Log-Cosh loss function.

    Behaves like L1 for large errors and like L2 for small errors.
    Smooth everywhere (differentiable), unlike Huber loss.

    L(x) = log(cosh(x)) ≈ |x| - log(2) for large |x|
    L(x) = log(cosh(x)) ≈ x²/2 for small |x|

    Args:
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        # log(cosh(x)) = x + softplus(-2x) - log(2)
        # This formulation is more numerically stable
        loss = diff + F.softplus(-2.0 * diff) - math.log(2.0)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =============================================================================
# Correlation Loss - Directly optimize Pearson/Spearman correlation
# =============================================================================

class PearsonCorrelationLoss(nn.Module):
    """Loss based on Pearson correlation coefficient.

    Maximizes correlation between prediction and target.
    Loss = 1 - r (or -r to maximize correlation)

    Args:
        per_channel: Compute loss per channel then average
        eps: Small value for numerical stability
    """

    def __init__(self, per_channel: bool = True, eps: float = 1e-8):
        super().__init__()
        self.per_channel = per_channel
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation loss.

        Args:
            pred: [B, C, T] predictions
            target: [B, C, T] targets

        Returns:
            Loss value (1 - mean correlation)
        """
        if self.per_channel:
            # Flatten time dimension: [B, C, T] -> [B*C, T]
            b, c, t = pred.shape
            pred_flat = pred.view(b * c, t)
            target_flat = target.view(b * c, t)

            # Compute correlation per (batch, channel) pair
            pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
            target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)

            pred_std = pred_centered.std(dim=1) + self.eps
            target_std = target_centered.std(dim=1) + self.eps

            corr = (pred_centered * target_centered).mean(dim=1) / (pred_std * target_std)

            return 1.0 - corr.mean()
        else:
            # Global correlation
            pred_flat = pred.flatten()
            target_flat = target.flatten()

            pred_centered = pred_flat - pred_flat.mean()
            target_centered = target_flat - target_flat.mean()

            corr = (pred_centered * target_centered).sum() / (
                pred_centered.norm() * target_centered.norm() + self.eps
            )

            return 1.0 - corr


class ConcordanceCorrelationLoss(nn.Module):
    """Lin's Concordance Correlation Coefficient (CCC) Loss.

    Measures both precision AND accuracy, unlike Pearson which only measures
    linear relationship. Important for neural signal translation where we
    want predictions to match ground truth exactly, not just correlate.

    CCC = 2 * cov(x,y) / (var(x) + var(y) + (mean(x) - mean(y))^2)

    Args:
        per_channel: Compute per channel then average
        eps: Numerical stability
    """

    def __init__(self, per_channel: bool = True, eps: float = 1e-8):
        super().__init__()
        self.per_channel = per_channel
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            b, c, t = pred.shape
            pred_flat = pred.view(b * c, t)
            target_flat = target.view(b * c, t)
        else:
            pred_flat = pred.flatten().unsqueeze(0)
            target_flat = target.flatten().unsqueeze(0)

        pred_mean = pred_flat.mean(dim=1)
        target_mean = target_flat.mean(dim=1)
        pred_var = pred_flat.var(dim=1)
        target_var = target_flat.var(dim=1)

        covariance = ((pred_flat - pred_mean.unsqueeze(1)) *
                      (target_flat - target_mean.unsqueeze(1))).mean(dim=1)

        ccc = (2 * covariance) / (
            pred_var + target_var + (pred_mean - target_mean) ** 2 + self.eps
        )

        return 1.0 - ccc.mean()


# =============================================================================
# SSIM-Like Loss for Time Series
# =============================================================================

class TemporalSSIMLoss(nn.Module):
    """Structural Similarity (SSIM) adapted for 1D time series.

    Computes local structure similarity using sliding windows,
    similar to image SSIM but for neural signals.

    Args:
        window_size: Size of the sliding window
        k1, k2: SSIM stability constants
    """

    def __init__(
        self,
        window_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        data_range: float = 2.0,  # Assuming normalized [-1, 1]
    ):
        super().__init__()
        self.window_size = window_size
        self.k1 = k1
        self.k2 = k2
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2

        # Create Gaussian window
        sigma = window_size / 6.0
        coords = torch.arange(window_size).float() - window_size // 2
        gaussian = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.sum()
        self.register_buffer("window", gaussian.view(1, 1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss.

        Args:
            pred: [B, C, T] predictions
            target: [B, C, T] targets

        Returns:
            1 - SSIM (loss to minimize)
        """
        b, c, t = pred.shape

        # Reshape for grouped conv: [B*C, 1, T]
        pred = pred.view(b * c, 1, t)
        target = target.view(b * c, 1, t)

        # Pad for same output size
        pad = self.window_size // 2
        pred_padded = F.pad(pred, (pad, pad), mode="reflect")
        target_padded = F.pad(target, (pad, pad), mode="reflect")

        # Local means
        mu_x = F.conv1d(pred_padded, self.window)
        mu_y = F.conv1d(target_padded, self.window)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        # Local variances
        sigma_x_sq = F.conv1d(pred_padded ** 2, self.window) - mu_x_sq
        sigma_y_sq = F.conv1d(target_padded ** 2, self.window) - mu_y_sq
        sigma_xy = F.conv1d(pred_padded * target_padded, self.window) - mu_xy

        # SSIM formula
        ssim = ((2 * mu_xy + self.c1) * (2 * sigma_xy + self.c2)) / (
            (mu_x_sq + mu_y_sq + self.c1) * (sigma_x_sq + sigma_y_sq + self.c2)
        )

        return 1.0 - ssim.mean()


# =============================================================================
# Contrastive Loss for Neural Signals
# =============================================================================

class ContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss for neural signal translation.

    Encourages predictions to be similar to their corresponding targets
    and dissimilar to other targets in the batch.

    Args:
        temperature: Softmax temperature
        projection_dim: Dimension of projection head (0 = no projection)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        projection_dim: int = 128,
        input_dim: int = 32,
    ):
        super().__init__()
        self.temperature = temperature

        if projection_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
            )
        else:
            self.projection = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            pred: [B, C, T] predictions
            target: [B, C, T] targets

        Returns:
            InfoNCE loss
        """
        b = pred.shape[0]

        # Pool over time: [B, C, T] -> [B, C]
        pred_pooled = pred.mean(dim=-1)
        target_pooled = target.mean(dim=-1)

        # Project if using projection head
        if self.projection is not None:
            pred_proj = self.projection(pred_pooled)
            target_proj = self.projection(target_pooled)
        else:
            pred_proj = pred_pooled
            target_proj = target_pooled

        # Normalize
        pred_norm = F.normalize(pred_proj, dim=-1)
        target_norm = F.normalize(target_proj, dim=-1)

        # Compute similarity matrix
        logits = torch.mm(pred_norm, target_norm.t()) / self.temperature

        # Labels: diagonal elements are positives
        labels = torch.arange(b, device=pred.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


# =============================================================================
# Adversarial / GAN Loss
# =============================================================================

class AdversarialLoss(nn.Module):
    """Adversarial loss for neural signal translation (GAN-style).

    Uses a discriminator to encourage realistic signal generation.
    This module contains both generator and discriminator losses.

    Args:
        discriminator: Optional pre-built discriminator (creates default if None)
        loss_type: 'vanilla', 'lsgan', 'wgan', or 'hinge'
        n_channels: Number of input channels
    """

    def __init__(
        self,
        discriminator: Optional[nn.Module] = None,
        loss_type: str = "lsgan",
        n_channels: int = 32,
    ):
        super().__init__()
        self.loss_type = loss_type

        if discriminator is None:
            self.discriminator = self._build_discriminator(n_channels)
        else:
            self.discriminator = discriminator

    def _build_discriminator(self, n_channels: int) -> nn.Module:
        """Build a simple 1D discriminator."""
        return nn.Sequential(
            # [B, C, T] -> [B, 64, T/2]
            nn.Conv1d(n_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # [B, 64, T/2] -> [B, 128, T/4]
            nn.Conv1d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2),
            # [B, 128, T/4] -> [B, 256, T/8]
            nn.Conv1d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(0.2),
            # [B, 256, T/8] -> [B, 1, T/16]
            nn.Conv1d(256, 1, 4, stride=2, padding=1),
        )

    def discriminator_loss(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute discriminator loss.

        Args:
            real: Real (target) signals [B, C, T]
            fake: Generated (predicted) signals [B, C, T]

        Returns:
            Discriminator loss
        """
        real_pred = self.discriminator(real)
        fake_pred = self.discriminator(fake.detach())

        if self.loss_type == "vanilla":
            real_loss = F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )
            return (real_loss + fake_loss) / 2

        elif self.loss_type == "lsgan":
            real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
            fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
            return (real_loss + fake_loss) / 2

        elif self.loss_type == "wgan":
            return fake_pred.mean() - real_pred.mean()

        elif self.loss_type == "hinge":
            real_loss = F.relu(1.0 - real_pred).mean()
            fake_loss = F.relu(1.0 + fake_pred).mean()
            return (real_loss + fake_loss) / 2

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def generator_loss(self, fake: torch.Tensor) -> torch.Tensor:
        """Compute generator loss (for the translator model).

        Args:
            fake: Generated signals [B, C, T]

        Returns:
            Generator adversarial loss
        """
        fake_pred = self.discriminator(fake)

        if self.loss_type == "vanilla":
            return F.binary_cross_entropy_with_logits(
                fake_pred, torch.ones_like(fake_pred)
            )
        elif self.loss_type == "lsgan":
            return F.mse_loss(fake_pred, torch.ones_like(fake_pred))
        elif self.loss_type == "wgan":
            return -fake_pred.mean()
        elif self.loss_type == "hinge":
            return -fake_pred.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute generator adversarial loss (default forward).

        For training, use discriminator_loss and generator_loss separately.
        """
        return self.generator_loss(pred)


# =============================================================================
# Focal Loss - Handle imbalanced errors
# =============================================================================

class FocalMSELoss(nn.Module):
    """Focal MSE Loss - focuses on hard examples.

    Applies focal weighting to MSE loss, down-weighting easy examples
    and focusing on hard-to-predict regions.

    L_focal = (1 - exp(-MSE))^gamma * MSE

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = (pred - target) ** 2

        # Focal weight: down-weight easy examples
        focal_weight = (1 - torch.exp(-mse)) ** self.gamma
        focal_loss = focal_weight * mse

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# =============================================================================
# Earth Mover's Distance (Wasserstein) Loss
# =============================================================================

class SlicedWassersteinLoss(nn.Module):
    """Sliced Wasserstein Distance for neural signal distributions.

    Approximates Earth Mover's Distance using random 1D projections.
    Useful for matching overall signal distributions.

    Args:
        n_projections: Number of random projections
    """

    def __init__(self, n_projections: int = 50):
        super().__init__()
        self.n_projections = n_projections

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Sliced Wasserstein Distance.

        Args:
            pred: [B, C, T] predictions
            target: [B, C, T] targets

        Returns:
            Sliced Wasserstein distance
        """
        b, c, t = pred.shape
        device = pred.device

        # Flatten to [B, C*T]
        pred_flat = pred.view(b, -1)
        target_flat = target.view(b, -1)

        # Random projections
        dim = c * t
        projections = torch.randn(self.n_projections, dim, device=device)
        projections = F.normalize(projections, dim=1)

        # Project and sort
        pred_proj = torch.mm(pred_flat, projections.t())  # [B, n_proj]
        target_proj = torch.mm(target_flat, projections.t())

        pred_sorted, _ = pred_proj.sort(dim=0)
        target_sorted, _ = target_proj.sort(dim=0)

        # Wasserstein distance on sorted projections
        swd = (pred_sorted - target_sorted).abs().mean()

        return swd


# =============================================================================
# Perceptual Loss using Learned Neural Features
# =============================================================================

class NeuralPerceptualLoss(nn.Module):
    """Perceptual loss using learned neural signal features.

    Instead of using pretrained image features, this uses a small
    feature extractor trained on neural signals or uses intermediate
    features from a pretrained neural translation model.

    Args:
        feature_extractor: Optional pre-trained feature extractor
        layers: Which layers to extract features from
        weights: Weights for each layer's contribution
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        n_channels: int = 32,
        feature_dims: List[int] = [64, 128, 256],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()

        if feature_extractor is None:
            # Build simple feature extractor
            self.feature_extractor = self._build_extractor(n_channels, feature_dims)
        else:
            self.feature_extractor = feature_extractor

        self.weights = weights or [1.0 / len(feature_dims)] * len(feature_dims)
        self.feature_dims = feature_dims

    def _build_extractor(
        self, n_channels: int, feature_dims: List[int]
    ) -> nn.ModuleList:
        """Build multi-scale feature extractor."""
        layers = nn.ModuleList()
        in_dim = n_channels

        for out_dim in feature_dims:
            layers.append(nn.Sequential(
                nn.Conv1d(in_dim, out_dim, 4, stride=2, padding=1),
                nn.InstanceNorm1d(out_dim),
                nn.LeakyReLU(0.2),
            ))
            in_dim = out_dim

        return layers

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        features = []
        for layer in self.feature_extractor:
            x = layer(x)
            features.append(x)
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss.

        Args:
            pred: [B, C, T] predictions
            target: [B, C, T] targets

        Returns:
            Weighted perceptual loss
        """
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        loss = 0.0
        for w, pf, tf in zip(self.weights, pred_features, target_features):
            loss = loss + w * F.l1_loss(pf, tf)

        return loss


# =============================================================================
# Combined Multi-Objective Loss
# =============================================================================

class MultiObjectiveLoss(nn.Module):
    """Combines multiple loss functions with learnable or fixed weights.

    Common in neural signal translation to balance reconstruction,
    spectral, and adversarial objectives.

    Args:
        losses: Dict of {name: loss_module}
        weights: Dict of {name: weight} or None for learnable
        learnable_weights: If True, weights are learned parameters
    """

    def __init__(
        self,
        losses: dict,
        weights: Optional[dict] = None,
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.learnable_weights = learnable_weights

        if learnable_weights:
            # Log-space weights for numerical stability
            self.log_weights = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1))
                for name in losses.keys()
            })
        else:
            self.fixed_weights = weights or {name: 1.0 for name in losses.keys()}

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss.

        Returns:
            (total_loss, loss_dict) - total loss and individual components
        """
        loss_dict = {}
        total_loss = 0.0

        for name, loss_fn in self.losses.items():
            loss_val = loss_fn(pred, target)
            loss_dict[name] = loss_val.item()

            if self.learnable_weights:
                weight = torch.exp(-self.log_weights[name])
                # Uncertainty weighting: L_total = sum(L_i / 2σ_i² + log(σ_i))
                total_loss = total_loss + weight * loss_val + self.log_weights[name]
            else:
                total_loss = total_loss + self.fixed_weights[name] * loss_val

        return total_loss, loss_dict


# =============================================================================
# Factory Functions
# =============================================================================

def create_literature_loss(name: str, **kwargs) -> nn.Module:
    """Create a literature-standard loss by name.

    Args:
        name: Loss name
        **kwargs: Loss-specific parameters

    Returns:
        Loss module
    """
    registry = {
        "log_cosh": LogCoshLoss,
        "pearson": PearsonCorrelationLoss,
        "concordance": ConcordanceCorrelationLoss,
        "ccc": ConcordanceCorrelationLoss,  # Alias
        "ssim": TemporalSSIMLoss,
        "contrastive": ContrastiveLoss,
        "adversarial": AdversarialLoss,
        "gan": AdversarialLoss,  # Alias
        "focal_mse": FocalMSELoss,
        "focal": FocalMSELoss,  # Alias
        "wasserstein": SlicedWassersteinLoss,
        "swd": SlicedWassersteinLoss,  # Alias
        "perceptual": NeuralPerceptualLoss,
    }

    if name not in registry:
        raise ValueError(f"Unknown loss: {name}. Available: {list(registry.keys())}")

    return registry[name](**kwargs)
