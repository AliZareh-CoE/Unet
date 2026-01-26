"""
Neural Signal Validation Framework
===================================

Comprehensive deep learning model analysis framework for scientific publications.
Designed for cross-regional neural signal translation models (Region A -> Region B).

This framework provides systematic model interrogation covering:
- Part 1: Training Dynamics (learning curves, gradient flow, weight evolution)
- Part 2: Input Attribution (channel importance, temporal saliency)
- Part 3: Spectral Analysis (band-wise accuracy, PSD comparison, phase/coherence)
- Part 4: Latent Space Analysis (bottleneck visualization, RSA, dimensionality)
- Part 5: Conditioning Mechanism Analysis (embedding visualization, ablation)
- Part 6: Loss Landscape & Optimization (3D landscape, sharpness)
- Part 7: Generalization Analysis (per-session breakdown, cross-subject)
- Part 8: Error Analysis (distribution, worst cases, residuals)
- Part 9: Architectural Insights (ablation summary, skip connections)
- Part 10: Baseline Comparisons (linear methods, complexity curves)

Usage:
    from utils.neural_validation import NeuralSignalValidator

    validator = NeuralSignalValidator(model, config)
    results = validator.run_full_analysis(train_loader, val_loader, test_loader)
    validator.generate_report(results, output_dir="validation_results")

Author: Neural Signal Translation Project
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy.stats import pearsonr, spearmanr, gaussian_kde
from scipy.signal import hilbert, welch, coherence as scipy_coherence
from torch.utils.data import DataLoader

# Local imports
try:
    from .dsp_metrics import (
        FREQUENCY_BANDS,
        compute_psd,
        compare_psd,
        compute_plv,
        compute_pli,
        compute_coherence,
        compute_envelope_correlation,
    )
    from .statistics import (
        confidence_interval,
        bootstrap_ci,
        compute_fold_statistics,
        cohens_d,
    )
except ImportError:
    # Fallback for direct execution
    from dsp_metrics import (
        FREQUENCY_BANDS,
        compute_psd,
        compare_psd,
        compute_plv,
        compute_pli,
        compute_coherence,
        compute_envelope_correlation,
    )
    from statistics import (
        confidence_interval,
        bootstrap_ci,
        compute_fold_statistics,
        cohens_d,
    )


# =============================================================================
# Configuration and Result Classes
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for validation analysis."""

    # General
    device: str = "cuda"
    seed: int = 42
    fs: float = 1000.0  # Sampling frequency

    # Training dynamics
    track_gradients: bool = True
    track_weights: bool = True
    gradient_clip_percentile: float = 99.0

    # Input attribution
    n_occlusion_samples: int = 100
    temporal_perturbation_width: int = 10

    # Spectral analysis
    nperseg: int = 256
    frequency_bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(FREQUENCY_BANDS)
    )

    # Latent space
    n_umap_samples: int = 1000
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    pca_variance_threshold: float = 0.90

    # Loss landscape
    landscape_resolution: int = 25
    landscape_range: float = 1.0

    # Error analysis
    n_worst_cases: int = 20
    error_histogram_bins: int = 50

    # Output
    save_figures: bool = True
    figure_format: str = "png"
    figure_dpi: int = 300


@dataclass
class TrainingDynamicsResult:
    """Results from training dynamics analysis."""

    # Learning curves
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_r2: List[float] = field(default_factory=list)
    val_r2: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)

    # Per-fold learning curves (for LOSO)
    fold_train_losses: Dict[str, List[float]] = field(default_factory=dict)
    fold_val_losses: Dict[str, List[float]] = field(default_factory=dict)
    fold_val_r2: Dict[str, List[float]] = field(default_factory=dict)

    # Gradient flow
    gradient_norms: Dict[str, List[float]] = field(default_factory=dict)  # layer -> norms
    gradient_flow_heatmap: Optional[np.ndarray] = None  # [layers, epochs]

    # Weight evolution
    weight_norms: Dict[str, List[float]] = field(default_factory=dict)  # layer -> norms
    weight_changes: Dict[str, List[float]] = field(default_factory=dict)  # layer -> delta norms

    # Loss components
    loss_components: Dict[str, List[float]] = field(default_factory=dict)

    # Convergence metrics
    convergence_epoch: Optional[int] = None
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    overfitting_gap: Optional[float] = None


@dataclass
class InputAttributionResult:
    """Results from input attribution analysis."""

    # Channel importance
    channel_importance_occlusion: Optional[np.ndarray] = None  # [n_channels]
    channel_importance_gradient: Optional[np.ndarray] = None  # [n_channels]
    channel_ranking: Optional[List[int]] = None

    # Temporal saliency
    temporal_saliency: Optional[np.ndarray] = None  # [time_steps]
    temporal_receptive_field: Optional[np.ndarray] = None
    critical_time_window: Optional[Tuple[int, int]] = None

    # Input ablation
    ablation_results: Dict[str, float] = field(default_factory=dict)


@dataclass
class SpectralAnalysisResult:
    """Results from spectral analysis."""

    # Band-wise accuracy
    band_r2: Dict[str, float] = field(default_factory=dict)
    band_correlation: Dict[str, float] = field(default_factory=dict)

    # PSD comparison
    psd_correlation: Optional[float] = None
    psd_rmse: Optional[float] = None
    mean_target_psd: Optional[np.ndarray] = None
    mean_pred_psd: Optional[np.ndarray] = None
    psd_frequencies: Optional[np.ndarray] = None

    # Spectral error
    spectral_error_by_freq: Optional[np.ndarray] = None
    worst_frequency_bands: List[str] = field(default_factory=list)

    # Coherence
    coherence_real: Dict[str, float] = field(default_factory=dict)
    coherence_predicted: Dict[str, float] = field(default_factory=dict)
    coherence_preservation: Dict[str, float] = field(default_factory=dict)

    # Phase
    plv_real: Dict[str, float] = field(default_factory=dict)
    plv_predicted: Dict[str, float] = field(default_factory=dict)
    phase_preservation: Dict[str, float] = field(default_factory=dict)


@dataclass
class LatentSpaceResult:
    """Results from latent space analysis."""

    # Bottleneck features
    bottleneck_features: Optional[np.ndarray] = None  # [n_samples, feature_dim]

    # Visualization coordinates
    umap_coords: Optional[np.ndarray] = None  # [n_samples, 2]
    tsne_coords: Optional[np.ndarray] = None  # [n_samples, 2]
    pca_coords: Optional[np.ndarray] = None  # [n_samples, 2]

    # Metadata for coloring
    sample_sessions: Optional[np.ndarray] = None
    sample_labels: Optional[np.ndarray] = None

    # Dimensionality
    pca_explained_variance: Optional[np.ndarray] = None
    effective_dimensionality: Optional[int] = None
    intrinsic_dimensionality: Optional[float] = None

    # RSA
    rsa_input_correlation: Optional[float] = None
    rsa_output_correlation: Optional[float] = None

    # Layer-wise
    layer_representations: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class LossLandscapeResult:
    """Results from loss landscape analysis."""

    # 3D surface
    landscape_grid: Optional[np.ndarray] = None  # [resolution, resolution]
    alpha_range: Optional[np.ndarray] = None
    beta_range: Optional[np.ndarray] = None
    direction1: Optional[np.ndarray] = None
    direction2: Optional[np.ndarray] = None

    # Sharpness
    sharpness_measure: Optional[float] = None
    hessian_eigenvalues: Optional[np.ndarray] = None
    flatness_ratio: Optional[float] = None

    # Minimum info
    minimum_loss: Optional[float] = None
    minimum_location: Optional[Tuple[float, float]] = None


@dataclass
class GeneralizationResult:
    """Results from generalization analysis."""

    # Per-session breakdown
    session_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    session_r2: Dict[str, float] = field(default_factory=dict)
    session_correlation: Dict[str, float] = field(default_factory=dict)

    # Aggregate statistics
    mean_r2: Optional[float] = None
    std_r2: Optional[float] = None
    ci_r2: Optional[Tuple[float, float]] = None

    # Cross-subject (if applicable)
    within_subject_r2: Optional[float] = None
    cross_subject_r2: Optional[float] = None

    # Temporal generalization
    early_session_r2: Optional[float] = None
    late_session_r2: Optional[float] = None
    temporal_drift: Optional[float] = None

    # Signal quality correlation
    snr_performance_correlation: Optional[float] = None
    outlier_sessions: List[str] = field(default_factory=list)


@dataclass
class ErrorAnalysisResult:
    """Results from error analysis."""

    # Distribution
    error_mean: Optional[float] = None
    error_std: Optional[float] = None
    error_skewness: Optional[float] = None
    error_kurtosis: Optional[float] = None
    error_histogram: Optional[np.ndarray] = None
    error_bins: Optional[np.ndarray] = None

    # Error vs characteristics
    error_vs_amplitude: Optional[Tuple[np.ndarray, np.ndarray]] = None
    error_vs_variance: Optional[Tuple[np.ndarray, np.ndarray]] = None
    error_vs_frequency: Dict[str, float] = field(default_factory=dict)

    # Worst cases
    worst_case_indices: Optional[np.ndarray] = None
    worst_case_errors: Optional[np.ndarray] = None
    worst_case_inputs: Optional[np.ndarray] = None
    worst_case_targets: Optional[np.ndarray] = None
    worst_case_predictions: Optional[np.ndarray] = None

    # Residuals
    residual_autocorrelation: Optional[np.ndarray] = None
    residual_has_structure: Optional[bool] = None
    residual_psd: Optional[np.ndarray] = None


@dataclass
class ArchitecturalInsightsResult:
    """Results from architectural insights analysis."""

    # Component ablation
    ablation_performance: Dict[str, float] = field(default_factory=dict)
    ablation_delta: Dict[str, float] = field(default_factory=dict)
    component_ranking: List[str] = field(default_factory=list)

    # Skip connections
    skip_activation_norms: Dict[str, float] = field(default_factory=dict)
    skip_information_ratio: Dict[str, float] = field(default_factory=dict)

    # Depth analysis
    depth_performance: Dict[int, float] = field(default_factory=dict)
    optimal_depth: Optional[int] = None


@dataclass
class BaselineComparisonResult:
    """Results from baseline comparisons."""

    # Linear baselines
    ridge_r2: Optional[float] = None
    wiener_r2: Optional[float] = None
    linear_per_channel_r2: Optional[np.ndarray] = None

    # Band-specific comparison
    band_linear_r2: Dict[str, float] = field(default_factory=dict)
    band_model_r2: Dict[str, float] = field(default_factory=dict)
    band_improvement: Dict[str, float] = field(default_factory=dict)

    # Complexity analysis
    model_sizes: List[int] = field(default_factory=list)
    model_r2_scores: List[float] = field(default_factory=list)
    optimal_complexity: Optional[int] = None


@dataclass
class FullValidationResult:
    """Complete validation results."""

    training_dynamics: Optional[TrainingDynamicsResult] = None
    input_attribution: Optional[InputAttributionResult] = None
    spectral_analysis: Optional[SpectralAnalysisResult] = None
    latent_space: Optional[LatentSpaceResult] = None
    loss_landscape: Optional[LossLandscapeResult] = None
    generalization: Optional[GeneralizationResult] = None
    error_analysis: Optional[ErrorAnalysisResult] = None
    architectural_insights: Optional[ArchitecturalInsightsResult] = None
    baseline_comparison: Optional[BaselineComparisonResult] = None

    # Summary
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: Optional[str] = None
    config_used: Optional[ValidationConfig] = None


# =============================================================================
# PART 1: TRAINING DYNAMICS
# =============================================================================

class TrainingDynamicsAnalyzer:
    """Analyze training dynamics: learning curves, gradient flow, weight evolution."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self._gradient_history: Dict[str, List[float]] = {}
        self._weight_history: Dict[str, List[float]] = {}
        self._loss_history: Dict[str, List[float]] = {}

    def setup_gradient_hooks(self, model: nn.Module) -> List:
        """Register hooks to track gradient norms during training.

        Args:
            model: The neural network model

        Returns:
            List of hook handles (call .remove() to cleanup)
        """
        hooks = []

        def make_hook(name):
            def hook(grad):
                if grad is not None:
                    norm = grad.norm().item()
                    if name not in self._gradient_history:
                        self._gradient_history[name] = []
                    self._gradient_history[name].append(norm)
            return hook

        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(make_hook(name))
                hooks.append(hook)

        return hooks

    def record_weight_norms(self, model: nn.Module, epoch: int):
        """Record weight norms for all layers at current epoch.

        Args:
            model: The neural network model
            epoch: Current epoch number
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                norm = param.data.norm().item()
                if name not in self._weight_history:
                    self._weight_history[name] = []
                self._weight_history[name].append(norm)

    def record_loss_components(self, loss_dict: Dict[str, float], epoch: int):
        """Record individual loss components.

        Args:
            loss_dict: Dictionary of loss component names to values
            epoch: Current epoch number
        """
        for name, value in loss_dict.items():
            if name not in self._loss_history:
                self._loss_history[name] = []
            self._loss_history[name].append(value)

    def analyze_learning_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: Optional[Dict[str, List[float]]] = None,
        val_metrics: Optional[Dict[str, List[float]]] = None,
    ) -> TrainingDynamicsResult:
        """Analyze learning curves for convergence and overfitting.

        Args:
            train_losses: Training loss per epoch
            val_losses: Validation loss per epoch
            train_metrics: Optional dict of training metrics (e.g., r2, correlation)
            val_metrics: Optional dict of validation metrics

        Returns:
            TrainingDynamicsResult with learning curve analysis
        """
        result = TrainingDynamicsResult()
        result.train_losses = train_losses
        result.val_losses = val_losses
        result.epochs = list(range(len(train_losses)))

        if train_metrics and 'r2' in train_metrics:
            result.train_r2 = train_metrics['r2']
        if val_metrics and 'r2' in val_metrics:
            result.val_r2 = val_metrics['r2']

        # Find convergence point (when val loss stops improving significantly)
        if len(val_losses) > 10:
            smoothed = np.convolve(val_losses, np.ones(5)/5, mode='valid')
            diffs = np.diff(smoothed)
            # Convergence when improvement < 1% of current loss
            threshold = 0.01 * np.mean(smoothed)
            converged_mask = np.abs(diffs) < threshold
            if np.any(converged_mask):
                result.convergence_epoch = int(np.argmax(converged_mask)) + 5

        result.final_train_loss = train_losses[-1] if train_losses else None
        result.final_val_loss = val_losses[-1] if val_losses else None

        # Overfitting gap (train-val difference normalized)
        if result.final_train_loss and result.final_val_loss:
            result.overfitting_gap = (result.final_val_loss - result.final_train_loss) / result.final_val_loss

        return result

    def compute_gradient_flow_heatmap(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Create gradient flow heatmap (layers x epochs).

        Args:
            model: The model (for layer names if not provided)
            layer_names: Optional list of layer names to include

        Returns:
            Heatmap array [n_layers, n_epochs]
        """
        if not self._gradient_history:
            return np.array([])

        if layer_names is None:
            layer_names = list(self._gradient_history.keys())

        n_epochs = max(len(v) for v in self._gradient_history.values())
        n_layers = len(layer_names)

        heatmap = np.zeros((n_layers, n_epochs))
        for i, name in enumerate(layer_names):
            if name in self._gradient_history:
                grads = self._gradient_history[name]
                heatmap[i, :len(grads)] = grads

        # Clip outliers for visualization
        if self.config.gradient_clip_percentile < 100:
            clip_val = np.percentile(heatmap[heatmap > 0], self.config.gradient_clip_percentile)
            heatmap = np.clip(heatmap, 0, clip_val)

        return heatmap

    def analyze_weight_evolution(self) -> Dict[str, np.ndarray]:
        """Analyze how weights changed over training.

        Returns:
            Dict with 'norms' and 'changes' per layer
        """
        results = {}

        for name, norms in self._weight_history.items():
            norms_arr = np.array(norms)
            changes = np.diff(norms_arr)

            results[name] = {
                'norms': norms_arr,
                'changes': changes,
                'total_change': norms_arr[-1] - norms_arr[0] if len(norms_arr) > 1 else 0,
                'learning_rate': np.mean(np.abs(changes)) if len(changes) > 0 else 0,
            }

        return results

    def get_result(self) -> TrainingDynamicsResult:
        """Get complete training dynamics result."""
        result = TrainingDynamicsResult()
        result.gradient_norms = dict(self._gradient_history)
        result.weight_norms = dict(self._weight_history)
        result.loss_components = dict(self._loss_history)

        if self._gradient_history:
            layer_names = list(self._gradient_history.keys())
            result.gradient_flow_heatmap = self.compute_gradient_flow_heatmap(None, layer_names)

        weight_analysis = self.analyze_weight_evolution()
        result.weight_changes = {k: v['changes'].tolist() for k, v in weight_analysis.items()}

        return result


# =============================================================================
# PART 2: INPUT ATTRIBUTION
# =============================================================================

class InputAttributionAnalyzer:
    """Analyze what input features drive model predictions."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def channel_importance_occlusion(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Compute channel importance by zeroing out each channel.

        Args:
            model: The trained model
            dataloader: Data loader for evaluation
            device: Computation device
            n_samples: Number of samples to use (None = all)

        Returns:
            Array of importance scores [n_channels]
        """
        model.eval()
        n_samples = n_samples or self.config.n_occlusion_samples

        # Get baseline performance
        baseline_loss = self._compute_loss(model, dataloader, device, n_samples)

        # Get number of channels from first batch
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            n_channels = x.shape[1]
            break

        importance = np.zeros(n_channels)

        for ch in range(n_channels):
            # Create masked forward function
            def masked_forward(model, x, *args, **kwargs):
                x_masked = x.clone()
                x_masked[:, ch, :] = 0
                return model(x_masked, *args, **kwargs)

            # Compute loss with channel zeroed
            loss_masked = self._compute_loss_with_forward(
                model, dataloader, device, n_samples, masked_forward
            )

            # Importance = performance drop
            importance[ch] = loss_masked - baseline_loss

        return importance

    def channel_importance_gradient(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Compute channel importance using gradient magnitude.

        Args:
            model: The trained model
            dataloader: Data loader
            device: Computation device
            n_samples: Number of samples to use

        Returns:
            Array of importance scores [n_channels]
        """
        model.eval()
        n_samples = n_samples or self.config.n_occlusion_samples

        importance_sum = None
        count = 0

        for batch in dataloader:
            if count >= n_samples:
                break

            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch, batch

            x = x.to(device).requires_grad_(True)
            y = y.to(device)

            # Forward pass
            pred = model(x)

            # Compute loss
            loss = F.mse_loss(pred, y)

            # Backward to get gradients w.r.t. input
            loss.backward()

            # Gradient magnitude per channel (average over time)
            grad_importance = x.grad.abs().mean(dim=(0, 2)).cpu().numpy()

            if importance_sum is None:
                importance_sum = grad_importance
            else:
                importance_sum += grad_importance

            count += x.shape[0]
            x.grad = None

        return importance_sum / max(count, 1)

    def temporal_saliency(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        method: str = "integrated_gradients",
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Compute temporal saliency - which time points influence output most.

        Args:
            model: The trained model
            dataloader: Data loader
            device: Computation device
            method: "input_x_gradient" or "integrated_gradients"
            n_samples: Number of samples to use

        Returns:
            Saliency scores [time_steps]
        """
        model.eval()
        n_samples = n_samples or self.config.n_occlusion_samples

        saliency_sum = None
        count = 0

        for batch in dataloader:
            if count >= n_samples:
                break

            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch, batch

            x = x.to(device)
            y = y.to(device)

            if method == "integrated_gradients":
                saliency = self._integrated_gradients(model, x, y, steps=20)
            else:  # input_x_gradient
                saliency = self._input_x_gradient(model, x, y)

            # Average over batch and channels
            saliency_temporal = saliency.abs().mean(dim=(0, 1)).cpu().numpy()

            if saliency_sum is None:
                saliency_sum = saliency_temporal
            else:
                saliency_sum += saliency_temporal

            count += x.shape[0]

        return saliency_sum / max(count, 1)

    def temporal_receptive_field(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        perturbation_width: Optional[int] = None,
    ) -> np.ndarray:
        """Measure actual temporal receptive field via perturbation.

        Args:
            model: The trained model
            dataloader: Data loader
            device: Computation device
            perturbation_width: Width of perturbation window

        Returns:
            Impact scores per time point [time_steps]
        """
        model.eval()
        perturbation_width = perturbation_width or self.config.temporal_perturbation_width

        # Get sample dimensions
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            time_steps = x.shape[2]
            break

        impact = np.zeros(time_steps)

        # Get baseline predictions
        baseline_outputs = []
        inputs = []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            with torch.no_grad():
                pred = model(x)
            baseline_outputs.append(pred.cpu())
            inputs.append(x.cpu())

            if len(inputs) * x.shape[0] >= self.config.n_occlusion_samples:
                break

        baseline_outputs = torch.cat(baseline_outputs, dim=0)
        inputs = torch.cat(inputs, dim=0)

        # Perturb each time window and measure impact
        for t in range(0, time_steps, perturbation_width):
            t_end = min(t + perturbation_width, time_steps)

            x_perturbed = inputs.clone()
            # Add noise to time window
            noise = torch.randn_like(x_perturbed[:, :, t:t_end]) * x_perturbed.std()
            x_perturbed[:, :, t:t_end] = noise

            with torch.no_grad():
                pred_perturbed = model(x_perturbed.to(device)).cpu()

            # Measure change in output
            change = (pred_perturbed - baseline_outputs).abs().mean().item()
            impact[t:t_end] = change

        return impact

    def input_ablation_study(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Ablate early/middle/late portions of input.

        Args:
            model: The trained model
            dataloader: Data loader
            device: Computation device

        Returns:
            Dict with ablation results (portion -> performance)
        """
        model.eval()

        # Get time dimension
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            time_steps = x.shape[2]
            break

        third = time_steps // 3
        segments = {
            'full': (0, time_steps),
            'early_ablated': (third, time_steps),  # Remove first third
            'middle_ablated': (0, third, 2*third, time_steps),  # Remove middle
            'late_ablated': (0, 2*third),  # Remove last third
            'early_only': (0, third),
            'middle_only': (third, 2*third),
            'late_only': (2*third, time_steps),
        }

        results = {}

        for name, bounds in segments.items():
            if name == 'middle_ablated':
                # Special handling for middle ablation
                r2 = self._compute_r2_with_mask(model, dataloader, device,
                                                keep_ranges=[(0, bounds[1]), (bounds[2], bounds[3])])
            else:
                r2 = self._compute_r2_with_mask(model, dataloader, device,
                                                keep_ranges=[bounds])
            results[name] = r2

        return results

    def _integrated_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        steps: int = 50,
    ) -> torch.Tensor:
        """Compute integrated gradients."""
        x.requires_grad_(True)

        # Baseline is zeros
        baseline = torch.zeros_like(x)

        # Accumulate gradients along path
        integrated_grads = torch.zeros_like(x)

        for i in range(steps):
            alpha = i / steps
            interpolated = baseline + alpha * (x - baseline)
            interpolated.requires_grad_(True)

            pred = model(interpolated)
            loss = F.mse_loss(pred, y)
            loss.backward()

            integrated_grads += interpolated.grad
            interpolated.grad = None

        # Scale by input
        integrated_grads = (x - baseline) * integrated_grads / steps

        return integrated_grads.detach()

    def _input_x_gradient(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute input x gradient saliency."""
        x.requires_grad_(True)

        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()

        saliency = x * x.grad
        x.grad = None

        return saliency.detach()

    def _compute_loss(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        n_samples: int,
    ) -> float:
        """Compute average loss over samples."""
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                if count >= n_samples:
                    break

                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch, batch

                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = F.mse_loss(pred, y, reduction='sum')

                total_loss += loss.item()
                count += x.shape[0]

        return total_loss / max(count, 1)

    def _compute_loss_with_forward(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        n_samples: int,
        forward_fn: Callable,
    ) -> float:
        """Compute loss using custom forward function."""
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                if count >= n_samples:
                    break

                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch, batch

                x, y = x.to(device), y.to(device)
                pred = forward_fn(model, x)
                loss = F.mse_loss(pred, y, reduction='sum')

                total_loss += loss.item()
                count += x.shape[0]

        return total_loss / max(count, 1)

    def _compute_r2_with_mask(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        keep_ranges: List[Tuple[int, int]],
    ) -> float:
        """Compute R² keeping only specified time ranges."""
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch, batch

                x, y = x.to(device), y.to(device)

                # Mask input to keep only specified ranges
                x_masked = torch.zeros_like(x)
                for start, end in keep_ranges:
                    x_masked[:, :, start:end] = x[:, :, start:end]

                pred = model(x_masked)

                all_targets.append(y.cpu().numpy())
                all_preds.append(pred.cpu().numpy())

                if len(all_targets) * x.shape[0] >= self.config.n_occlusion_samples:
                    break

        targets = np.concatenate(all_targets, axis=0).flatten()
        preds = np.concatenate(all_preds, axis=0).flatten()

        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)

        return 1 - ss_res / (ss_tot + 1e-10)


# =============================================================================
# PART 3: SPECTRAL ANALYSIS
# =============================================================================

class SpectralAnalyzer:
    """Analyze spectral properties of predictions vs targets."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def band_wise_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        fs: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute R² and correlation for each frequency band.

        Args:
            predictions: Predicted signals [n_samples, n_channels, time] or [n_samples, time]
            targets: Target signals (same shape)
            fs: Sampling frequency

        Returns:
            Dict with band names -> {r2, correlation, mse}
        """
        fs = fs or self.config.fs

        # Flatten to 2D if needed
        if predictions.ndim == 3:
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])

        results = {}

        for band_name, (low, high) in self.config.frequency_bands.items():
            # Skip bands outside Nyquist
            if high >= fs / 2:
                continue

            # Bandpass filter
            nyq = fs / 2
            b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')

            pred_filtered = signal.filtfilt(b, a, predictions, axis=-1)
            target_filtered = signal.filtfilt(b, a, targets, axis=-1)

            # Flatten for metrics
            pred_flat = pred_filtered.flatten()
            target_flat = target_filtered.flatten()

            # R²
            ss_res = np.sum((target_flat - pred_flat) ** 2)
            ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)

            # Correlation
            corr, _ = pearsonr(pred_flat, target_flat)

            # MSE
            mse = np.mean((target_flat - pred_flat) ** 2)

            results[band_name] = {
                'r2': float(r2),
                'correlation': float(corr),
                'mse': float(mse),
            }

        return results

    def psd_comparison(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        fs: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compare power spectral densities of predictions vs targets.

        Args:
            predictions: Predicted signals
            targets: Target signals
            fs: Sampling frequency

        Returns:
            Dict with PSD comparison metrics and data
        """
        fs = fs or self.config.fs
        nperseg = self.config.nperseg

        # Flatten to 1D signals for averaging
        if predictions.ndim > 1:
            pred_flat = predictions.reshape(-1, predictions.shape[-1])
            target_flat = targets.reshape(-1, targets.shape[-1])
        else:
            pred_flat = predictions.reshape(1, -1)
            target_flat = targets.reshape(1, -1)

        # Compute mean PSD across all samples
        all_pred_psd = []
        all_target_psd = []

        for i in range(pred_flat.shape[0]):
            freqs, psd_pred = welch(pred_flat[i], fs=fs, nperseg=nperseg)
            _, psd_target = welch(target_flat[i], fs=fs, nperseg=nperseg)
            all_pred_psd.append(psd_pred)
            all_target_psd.append(psd_target)

        mean_pred_psd = np.mean(all_pred_psd, axis=0)
        mean_target_psd = np.mean(all_target_psd, axis=0)

        # Normalize for comparison
        pred_norm = mean_pred_psd / (np.sum(mean_pred_psd) + 1e-10)
        target_norm = mean_target_psd / (np.sum(mean_target_psd) + 1e-10)

        # Correlation between PSDs
        psd_corr, _ = pearsonr(pred_norm, target_norm)

        # RMSE between normalized PSDs
        psd_rmse = np.sqrt(np.mean((pred_norm - target_norm) ** 2))

        # Error per frequency
        spectral_error = np.abs(pred_norm - target_norm)

        return {
            'psd_correlation': float(psd_corr),
            'psd_rmse': float(psd_rmse),
            'frequencies': freqs,
            'mean_pred_psd': mean_pred_psd,
            'mean_target_psd': mean_target_psd,
            'spectral_error': spectral_error,
        }

    def coherence_analysis(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray,
        fs: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compare coherence preservation: input-target vs input-prediction.

        Args:
            inputs: Input signals (Region A)
            targets: Target signals (Region B real)
            predictions: Predicted signals (Region B predicted)
            fs: Sampling frequency

        Returns:
            Dict with coherence metrics per band
        """
        fs = fs or self.config.fs
        nperseg = self.config.nperseg

        # Average coherence across samples
        def mean_band_coherence(sig1, sig2):
            if sig1.ndim > 1:
                sig1 = sig1.reshape(-1, sig1.shape[-1])
                sig2 = sig2.reshape(-1, sig2.shape[-1])

            band_coh = {band: [] for band in self.config.frequency_bands}

            for i in range(min(sig1.shape[0], 100)):  # Limit samples
                freqs, coh = scipy_coherence(sig1[i], sig2[i], fs=fs, nperseg=nperseg)

                for band_name, (low, high) in self.config.frequency_bands.items():
                    mask = (freqs >= low) & (freqs <= high)
                    if np.any(mask):
                        band_coh[band_name].append(np.mean(coh[mask]))

            return {band: np.mean(vals) if vals else 0 for band, vals in band_coh.items()}

        coh_real = mean_band_coherence(inputs, targets)
        coh_pred = mean_band_coherence(inputs, predictions)

        # Preservation ratio
        preservation = {}
        for band in self.config.frequency_bands:
            if coh_real[band] > 0:
                preservation[band] = coh_pred[band] / coh_real[band]
            else:
                preservation[band] = 1.0

        return {
            'coherence_real': coh_real,
            'coherence_predicted': coh_pred,
            'coherence_preservation': preservation,
        }

    def phase_analysis(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray,
        fs: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Analyze phase relationship preservation (PLV).

        Args:
            inputs: Input signals
            targets: Target signals
            predictions: Predicted signals
            fs: Sampling frequency

        Returns:
            Dict with PLV metrics per band
        """
        fs = fs or self.config.fs

        def mean_band_plv(sig1, sig2):
            if sig1.ndim > 1:
                sig1 = sig1.flatten()
                sig2 = sig2.flatten()

            band_plv = {}
            nyq = fs / 2

            for band_name, (low, high) in self.config.frequency_bands.items():
                if high >= nyq:
                    band_plv[band_name] = np.nan
                    continue

                try:
                    plv = compute_plv(sig1, sig2, fs=fs, band=(low, high))
                    band_plv[band_name] = plv
                except Exception:
                    band_plv[band_name] = np.nan

            return band_plv

        # Use subset for efficiency
        n_samples = min(inputs.shape[0] if inputs.ndim > 1 else 1, 50)

        if inputs.ndim > 1:
            inputs_sub = inputs[:n_samples].reshape(-1, inputs.shape[-1])
            targets_sub = targets[:n_samples].reshape(-1, targets.shape[-1])
            preds_sub = predictions[:n_samples].reshape(-1, predictions.shape[-1])
        else:
            inputs_sub = inputs
            targets_sub = targets
            preds_sub = predictions

        plv_real = mean_band_plv(inputs_sub.flatten(), targets_sub.flatten())
        plv_pred = mean_band_plv(inputs_sub.flatten(), preds_sub.flatten())

        preservation = {}
        for band in self.config.frequency_bands:
            if plv_real[band] > 0 and not np.isnan(plv_real[band]):
                preservation[band] = plv_pred[band] / plv_real[band]
            else:
                preservation[band] = 1.0

        return {
            'plv_real': plv_real,
            'plv_predicted': plv_pred,
            'phase_preservation': preservation,
        }


# =============================================================================
# PART 4: LATENT SPACE ANALYSIS
# =============================================================================

class LatentSpaceAnalyzer:
    """Analyze learned representations in the model's latent space."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def extract_bottleneck_features(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        n_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract bottleneck features from the model.

        Args:
            model: The trained model (must support return_bottleneck=True)
            dataloader: Data loader
            device: Computation device
            n_samples: Max samples to extract

        Returns:
            Tuple of (features, session_ids, labels) - session_ids and labels may be None
        """
        model.eval()
        n_samples = n_samples or self.config.n_umap_samples

        features_list = []
        sessions_list = []
        labels_list = []
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                if count >= n_samples:
                    break

                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                    # Try to get labels/sessions if available
                    if len(batch) > 2:
                        labels_list.append(batch[2].cpu().numpy())
                    if len(batch) > 3:
                        sessions_list.append(batch[3].cpu().numpy())
                else:
                    x = batch.to(device)

                # Extract bottleneck features
                try:
                    _, bottleneck = model(x, return_bottleneck=True)
                    features_list.append(bottleneck.cpu().numpy())
                except TypeError:
                    # Model doesn't support return_bottleneck
                    # Use activation hooks instead
                    features = self._extract_via_hooks(model, x)
                    features_list.append(features)

                count += x.shape[0]

        features = np.concatenate(features_list, axis=0)[:n_samples]

        sessions = np.concatenate(sessions_list, axis=0)[:n_samples] if sessions_list else None
        labels = np.concatenate(labels_list, axis=0)[:n_samples] if labels_list else None

        return features, sessions, labels

    def _extract_via_hooks(self, model: nn.Module, x: torch.Tensor) -> np.ndarray:
        """Extract features using activation hooks."""
        features = []

        def hook_fn(module, input, output):
            # Global average pool if needed
            if output.dim() == 3:
                features.append(output.mean(dim=-1).cpu().numpy())
            else:
                features.append(output.cpu().numpy())

        # Register hook on bottleneck (mid layer)
        if hasattr(model, 'mid'):
            handle = model.mid.register_forward_hook(hook_fn)
        else:
            # Try to find a bottleneck-like layer
            for name, module in model.named_modules():
                if 'mid' in name.lower() or 'bottleneck' in name.lower():
                    handle = module.register_forward_hook(hook_fn)
                    break

        # Forward pass
        _ = model(x)
        handle.remove()

        return features[0] if features else np.zeros((x.shape[0], 1))

    def visualize_latent_space(
        self,
        features: np.ndarray,
        method: str = "umap",
        **kwargs
    ) -> np.ndarray:
        """Project features to 2D for visualization.

        Args:
            features: Feature matrix [n_samples, n_features]
            method: "umap", "tsne", or "pca"
            **kwargs: Additional arguments for the method

        Returns:
            2D coordinates [n_samples, 2]
        """
        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(
                    n_neighbors=kwargs.get('n_neighbors', self.config.umap_n_neighbors),
                    min_dist=kwargs.get('min_dist', self.config.umap_min_dist),
                    n_components=2,
                    random_state=self.config.seed,
                )
                coords = reducer.fit_transform(features)
            except ImportError:
                warnings.warn("UMAP not installed, falling back to PCA")
                method = "pca"

        if method == "tsne":
            try:
                from sklearn.manifold import TSNE
                reducer = TSNE(
                    n_components=2,
                    perplexity=kwargs.get('perplexity', 30),
                    random_state=self.config.seed,
                )
                coords = reducer.fit_transform(features)
            except ImportError:
                warnings.warn("sklearn not installed, falling back to PCA")
                method = "pca"

        if method == "pca":
            # Simple PCA implementation without sklearn
            features_centered = features - np.mean(features, axis=0)
            cov = np.cov(features_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort by eigenvalue descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            coords = features_centered @ eigenvectors[:, :2]

        return coords

    def compute_dimensionality(
        self,
        features: np.ndarray,
        variance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Analyze effective dimensionality of latent space.

        Args:
            features: Feature matrix [n_samples, n_features]
            variance_threshold: Threshold for counting components (e.g., 0.90)

        Returns:
            Dict with dimensionality metrics
        """
        variance_threshold = variance_threshold or self.config.pca_variance_threshold

        # Center features
        features_centered = features - np.mean(features, axis=0)

        # Compute covariance and eigenvalues
        cov = np.cov(features_centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
        eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability

        # Explained variance ratio
        total_var = np.sum(eigenvalues)
        explained_variance = eigenvalues / (total_var + 1e-10)
        cumulative_variance = np.cumsum(explained_variance)

        # Effective dimensionality (components for threshold)
        effective_dim = np.searchsorted(cumulative_variance, variance_threshold) + 1

        # Intrinsic dimensionality (participation ratio)
        intrinsic_dim = (np.sum(eigenvalues) ** 2) / (np.sum(eigenvalues ** 2) + 1e-10)

        return {
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'effective_dimensionality': int(effective_dim),
            'intrinsic_dimensionality': float(intrinsic_dim),
            'total_features': features.shape[1],
        }

    def representational_similarity_analysis(
        self,
        features: np.ndarray,
        inputs: np.ndarray,
        outputs: np.ndarray,
    ) -> Dict[str, float]:
        """Compare latent representation structure to input/output structure.

        Args:
            features: Latent features [n_samples, n_features]
            inputs: Input signals [n_samples, ...]
            outputs: Output signals [n_samples, ...]

        Returns:
            Dict with RSA correlations
        """
        n_samples = min(features.shape[0], 500)  # Limit for efficiency

        # Compute similarity matrices
        def compute_similarity_matrix(data):
            data_flat = data.reshape(data.shape[0], -1)[:n_samples]
            # Correlation-based similarity
            sim = np.corrcoef(data_flat)
            return sim

        sim_features = compute_similarity_matrix(features)
        sim_inputs = compute_similarity_matrix(inputs)
        sim_outputs = compute_similarity_matrix(outputs)

        # Extract upper triangular (excluding diagonal)
        triu_idx = np.triu_indices(n_samples, k=1)

        feat_vec = sim_features[triu_idx]
        input_vec = sim_inputs[triu_idx]
        output_vec = sim_outputs[triu_idx]

        # Correlate similarity structures
        rsa_input, _ = pearsonr(feat_vec, input_vec)
        rsa_output, _ = pearsonr(feat_vec, output_vec)

        return {
            'rsa_input_correlation': float(rsa_input),
            'rsa_output_correlation': float(rsa_output),
        }

    def layer_wise_analysis(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        n_samples: int = 200,
    ) -> Dict[str, np.ndarray]:
        """Extract representations at each layer.

        Args:
            model: The model
            dataloader: Data loader
            device: Device
            n_samples: Number of samples

        Returns:
            Dict of layer_name -> features [n_samples, n_features]
        """
        model.eval()

        layer_features = {}

        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                # Global average pool temporal dimension
                if output.dim() == 3:
                    feat = output.mean(dim=-1)
                else:
                    feat = output
                if name not in layer_features:
                    layer_features[name] = []
                layer_features[name].append(feat.detach().cpu().numpy())
            return hook_fn

        # Register hooks
        hooks = []
        if hasattr(model, 'get_layer_names'):
            layer_names = model.get_layer_names()
            if hasattr(model, 'register_activation_hooks'):
                def callback(name, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    if output.dim() == 3:
                        feat = output.mean(dim=-1)
                    else:
                        feat = output
                    if name not in layer_features:
                        layer_features[name] = []
                    layer_features[name].append(feat.detach().cpu().numpy())
                hooks = model.register_activation_hooks(callback)

        # Collect features
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                if count >= n_samples:
                    break

                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)

                _ = model(x)
                count += x.shape[0]

        # Remove hooks
        for h in hooks:
            h.remove()

        # Concatenate features
        result = {}
        for name, feat_list in layer_features.items():
            result[name] = np.concatenate(feat_list, axis=0)[:n_samples]

        return result


# =============================================================================
# PART 5: CONDITIONING MECHANISM ANALYSIS
# =============================================================================

class ConditioningAnalyzer:
    """Analyze conditioning/modulation mechanisms."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def extract_conditioning_embeddings(
        self,
        model: nn.Module,
        n_conditions: int,
        device: torch.device,
    ) -> np.ndarray:
        """Extract learned conditioning embeddings.

        Args:
            model: The model with embedding layer
            n_conditions: Number of condition classes
            device: Device

        Returns:
            Embedding matrix [n_conditions, emb_dim]
        """
        model.eval()

        if hasattr(model, 'embed') and model.embed is not None:
            with torch.no_grad():
                indices = torch.arange(n_conditions, device=device)
                embeddings = model.embed(indices).cpu().numpy()
            return embeddings

        return np.array([])

    def conditioning_ablation(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Measure performance with and without conditioning.

        Args:
            model: The model
            dataloader: Data loader
            device: Device

        Returns:
            Dict with performance metrics
        """
        model.eval()

        results = {}

        # With conditioning (normal)
        r2_with = self._compute_r2(model, dataloader, device, use_conditioning=True)
        results['r2_with_conditioning'] = r2_with

        # Without conditioning (zero embeddings or skip)
        if hasattr(model, 'cond_mode') and model.cond_mode != 'none':
            r2_without = self._compute_r2(model, dataloader, device, use_conditioning=False)
            results['r2_without_conditioning'] = r2_without
            results['conditioning_contribution'] = r2_with - r2_without

        return results

    def _compute_r2(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        use_conditioning: bool = True,
    ) -> float:
        """Compute R² with optional conditioning ablation."""
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(device), batch[1].to(device)
                    odor_ids = batch[2].to(device) if len(batch) > 2 else None
                else:
                    x, y = batch.to(device), batch.to(device)
                    odor_ids = None

                if use_conditioning:
                    pred = model(x, odor_ids=odor_ids)
                else:
                    # Pass zero embedding or None
                    if hasattr(model, 'emb_dim'):
                        zero_emb = torch.zeros(x.shape[0], model.emb_dim, device=device)
                        pred = model(x, cond_emb=zero_emb)
                    else:
                        pred = model(x)

                all_targets.append(y.cpu().numpy())
                all_preds.append(pred.cpu().numpy())

                if len(all_targets) * x.shape[0] >= 500:
                    break

        targets = np.concatenate(all_targets, axis=0).flatten()
        preds = np.concatenate(all_preds, axis=0).flatten()

        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)

        return float(1 - ss_res / (ss_tot + 1e-10))

    def analyze_gating_patterns(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """Analyze gating weights if model uses gated attention.

        Args:
            model: The model
            dataloader: Data loader
            device: Device

        Returns:
            Dict with gating pattern statistics
        """
        model.eval()

        gating_weights = []

        # Hook to capture gating weights
        def hook_fn(module, input, output):
            if hasattr(module, 'gate') or 'gate' in str(type(module)).lower():
                # Try to extract gating values
                if isinstance(output, tuple) and len(output) > 1:
                    gating_weights.append(output[1].detach().cpu().numpy())

        # Register hooks on modules that might have gating
        hooks = []
        for name, module in model.named_modules():
            if 'gate' in name.lower() or 'attention' in name.lower():
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)
                _ = model(x)
                break  # Just need one batch

        # Cleanup
        for h in hooks:
            h.remove()

        if gating_weights:
            return {
                'mean_gating': np.mean(gating_weights),
                'std_gating': np.std(gating_weights),
                'gating_distribution': np.concatenate(gating_weights, axis=0) if gating_weights else np.array([]),
            }

        return {}


# =============================================================================
# PART 6: LOSS LANDSCAPE ANALYSIS
# =============================================================================

class LossLandscapeAnalyzer:
    """Analyze the loss landscape around the trained model."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def compute_loss_landscape(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        resolution: Optional[int] = None,
        range_scale: Optional[float] = None,
    ) -> LossLandscapeResult:
        """Compute 2D loss landscape around current weights.

        Args:
            model: The trained model
            dataloader: Data loader for loss computation
            device: Computation device
            resolution: Grid resolution
            range_scale: Scale of perturbation range

        Returns:
            LossLandscapeResult with landscape data
        """
        resolution = resolution or self.config.landscape_resolution
        range_scale = range_scale or self.config.landscape_range

        model.eval()

        # Get current weights as flat vector
        weights_original = self._get_weights_flat(model)

        # Generate random directions
        np.random.seed(self.config.seed)
        direction1 = np.random.randn(len(weights_original))
        direction2 = np.random.randn(len(weights_original))

        # Normalize directions
        direction1 = direction1 / np.linalg.norm(direction1)
        direction2 = direction2 / np.linalg.norm(direction2)

        # Filter orthogonalize direction2
        direction2 = direction2 - np.dot(direction1, direction2) * direction1
        direction2 = direction2 / np.linalg.norm(direction2)

        # Scale by weight norm
        weight_norm = np.linalg.norm(weights_original)
        direction1 = direction1 * weight_norm * range_scale
        direction2 = direction2 * weight_norm * range_scale

        # Create grid
        alphas = np.linspace(-1, 1, resolution)
        betas = np.linspace(-1, 1, resolution)

        landscape = np.zeros((resolution, resolution))

        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # Perturb weights
                perturbed = weights_original + alpha * direction1 + beta * direction2
                self._set_weights_flat(model, perturbed)

                # Compute loss
                loss = self._compute_loss(model, dataloader, device, max_batches=5)
                landscape[i, j] = loss

        # Restore original weights
        self._set_weights_flat(model, weights_original)

        # Find minimum
        min_idx = np.unravel_index(np.argmin(landscape), landscape.shape)
        min_location = (alphas[min_idx[0]], betas[min_idx[1]])

        result = LossLandscapeResult(
            landscape_grid=landscape,
            alpha_range=alphas,
            beta_range=betas,
            direction1=direction1,
            direction2=direction2,
            minimum_loss=float(np.min(landscape)),
            minimum_location=min_location,
        )

        # Compute sharpness
        center_loss = landscape[resolution//2, resolution//2]
        edge_losses = [landscape[0, resolution//2], landscape[-1, resolution//2],
                       landscape[resolution//2, 0], landscape[resolution//2, -1]]
        result.sharpness_measure = float(np.mean(edge_losses) - center_loss)

        return result

    def _get_weights_flat(self, model: nn.Module) -> np.ndarray:
        """Get all model weights as flat numpy array."""
        weights = []
        for param in model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def _set_weights_flat(self, model: nn.Module, weights: np.ndarray):
        """Set model weights from flat numpy array."""
        idx = 0
        for param in model.parameters():
            size = param.numel()
            param.data = torch.tensor(
                weights[idx:idx+size].reshape(param.shape),
                dtype=param.dtype,
                device=param.device
            )
            idx += size

    def _compute_loss(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        max_batches: int = 10,
    ) -> float:
        """Compute average loss over batches."""
        total_loss = 0
        count = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(device), batch[1].to(device)
                else:
                    x, y = batch.to(device), batch.to(device)

                pred = model(x)
                loss = F.mse_loss(pred, y)

                total_loss += loss.item()
                count += 1

        return total_loss / max(count, 1)


# =============================================================================
# PART 7: GENERALIZATION ANALYSIS
# =============================================================================

class GeneralizationAnalyzer:
    """Analyze model generalization across sessions/subjects."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def per_session_analysis(
        self,
        model: nn.Module,
        session_dataloaders: Dict[str, DataLoader],
        device: torch.device,
    ) -> GeneralizationResult:
        """Compute metrics for each session separately.

        Args:
            model: The trained model
            session_dataloaders: Dict of session_name -> dataloader
            device: Computation device

        Returns:
            GeneralizationResult with per-session breakdown
        """
        model.eval()
        result = GeneralizationResult()

        r2_values = []

        for session_name, loader in session_dataloaders.items():
            metrics = self._compute_session_metrics(model, loader, device)
            result.session_metrics[session_name] = metrics
            result.session_r2[session_name] = metrics['r2']
            result.session_correlation[session_name] = metrics['correlation']
            r2_values.append(metrics['r2'])

        # Aggregate statistics
        if r2_values:
            result.mean_r2 = float(np.mean(r2_values))
            result.std_r2 = float(np.std(r2_values))
            result.ci_r2 = confidence_interval(np.array(r2_values))

            # Identify outliers (> 2 std from mean)
            threshold = result.mean_r2 - 2 * result.std_r2
            result.outlier_sessions = [
                name for name, r2 in result.session_r2.items()
                if r2 < threshold
            ]

        return result

    def _compute_session_metrics(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Compute metrics for a single session."""
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(device), batch[1].to(device)
                else:
                    x, y = batch.to(device), batch.to(device)

                pred = model(x)

                all_targets.append(y.cpu().numpy())
                all_preds.append(pred.cpu().numpy())

        targets = np.concatenate(all_targets, axis=0).flatten()
        preds = np.concatenate(all_preds, axis=0).flatten()

        # R²
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        # Correlation
        corr, _ = pearsonr(targets, preds)

        # MAE
        mae = np.mean(np.abs(targets - preds))

        return {
            'r2': float(r2),
            'correlation': float(corr),
            'mae': float(mae),
            'n_samples': len(targets),
        }

    def signal_quality_correlation(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> float:
        """Compute correlation between input SNR and prediction quality.

        Args:
            model: The model
            dataloader: Data loader
            device: Device

        Returns:
            Correlation coefficient
        """
        model.eval()

        snr_values = []
        error_values = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(device), batch[1].to(device)
                else:
                    x, y = batch.to(device), batch.to(device)

                pred = model(x)

                # Compute per-sample SNR (signal variance / noise estimate)
                for i in range(x.shape[0]):
                    signal_var = x[i].var().item()
                    noise_est = (x[i] - x[i].mean()).abs().median().item() * 1.4826
                    snr = signal_var / (noise_est ** 2 + 1e-10)
                    snr_values.append(snr)

                    # Per-sample error
                    error = F.mse_loss(pred[i], y[i]).item()
                    error_values.append(error)

                if len(snr_values) >= 500:
                    break

        corr, _ = pearsonr(snr_values, error_values)
        return float(corr)


# =============================================================================
# PART 8: ERROR ANALYSIS
# =============================================================================

class ErrorAnalyzer:
    """Analyze prediction errors and failure modes."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def analyze_error_distribution(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> ErrorAnalysisResult:
        """Analyze the distribution of prediction errors.

        Args:
            predictions: Predicted signals
            targets: Target signals

        Returns:
            ErrorAnalysisResult with distribution statistics
        """
        errors = (predictions - targets).flatten()

        result = ErrorAnalysisResult()
        result.error_mean = float(np.mean(errors))
        result.error_std = float(np.std(errors))

        # Higher moments
        from scipy.stats import skew, kurtosis
        result.error_skewness = float(skew(errors))
        result.error_kurtosis = float(kurtosis(errors))

        # Histogram
        result.error_histogram, result.error_bins = np.histogram(
            errors, bins=self.config.error_histogram_bins
        )

        return result

    def error_vs_characteristics(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Analyze how error correlates with input characteristics.

        Args:
            model: The model
            dataloader: Data loader
            device: Device

        Returns:
            Dict with error correlations
        """
        model.eval()

        amplitudes = []
        variances = []
        errors = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(device), batch[1].to(device)
                else:
                    x, y = batch.to(device), batch.to(device)

                pred = model(x)

                for i in range(x.shape[0]):
                    amp = x[i].abs().mean().item()
                    var = x[i].var().item()
                    err = F.mse_loss(pred[i], y[i]).item()

                    amplitudes.append(amp)
                    variances.append(var)
                    errors.append(err)

                if len(errors) >= 1000:
                    break

        corr_amp, _ = pearsonr(amplitudes, errors)
        corr_var, _ = pearsonr(variances, errors)

        return {
            'error_amplitude_correlation': float(corr_amp),
            'error_variance_correlation': float(corr_var),
            'amplitudes': np.array(amplitudes),
            'variances': np.array(variances),
            'errors': np.array(errors),
        }

    def worst_case_analysis(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        n_worst: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Identify and return worst prediction cases.

        Args:
            model: The model
            dataloader: Data loader
            device: Device
            n_worst: Number of worst cases to return

        Returns:
            Dict with worst case data
        """
        model.eval()
        n_worst = n_worst or self.config.n_worst_cases

        all_inputs = []
        all_targets = []
        all_preds = []
        all_errors = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(device), batch[1].to(device)
                else:
                    x, y = batch.to(device), batch.to(device)

                pred = model(x)

                for i in range(x.shape[0]):
                    err = F.mse_loss(pred[i], y[i]).item()
                    all_inputs.append(x[i].cpu().numpy())
                    all_targets.append(y[i].cpu().numpy())
                    all_preds.append(pred[i].cpu().numpy())
                    all_errors.append(err)

        # Sort by error
        indices = np.argsort(all_errors)[::-1][:n_worst]

        return {
            'worst_indices': indices,
            'worst_errors': np.array([all_errors[i] for i in indices]),
            'worst_inputs': np.array([all_inputs[i] for i in indices]),
            'worst_targets': np.array([all_targets[i] for i in indices]),
            'worst_predictions': np.array([all_preds[i] for i in indices]),
        }

    def residual_analysis(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        fs: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Analyze residuals for structure.

        Args:
            predictions: Predicted signals
            targets: Target signals
            fs: Sampling frequency

        Returns:
            Dict with residual analysis
        """
        fs = fs or self.config.fs

        residuals = targets - predictions

        # Flatten for analysis
        if residuals.ndim > 1:
            residuals_flat = residuals.reshape(-1, residuals.shape[-1])
        else:
            residuals_flat = residuals.reshape(1, -1)

        # Autocorrelation of residuals
        mean_autocorr = []
        for i in range(min(residuals_flat.shape[0], 100)):
            res = residuals_flat[i]
            autocorr = np.correlate(res, res, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            mean_autocorr.append(autocorr[:min(100, len(autocorr))])

        mean_autocorr = np.mean(mean_autocorr, axis=0)

        # Check for structure (significant autocorrelation beyond lag 1)
        has_structure = np.any(np.abs(mean_autocorr[2:20]) > 0.1)

        # PSD of residuals
        freqs, psd = welch(residuals_flat.mean(axis=0), fs=fs, nperseg=self.config.nperseg)

        return {
            'autocorrelation': mean_autocorr,
            'has_structure': bool(has_structure),
            'residual_psd': psd,
            'psd_frequencies': freqs,
        }


# =============================================================================
# PART 9: ARCHITECTURAL INSIGHTS
# =============================================================================

class ArchitecturalAnalyzer:
    """Analyze architectural components and their contributions."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def skip_connection_analysis(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Analyze information flow through skip connections.

        Args:
            model: The model
            dataloader: Data loader
            device: Device

        Returns:
            Dict with skip connection metrics
        """
        model.eval()

        skip_norms = {}
        decoder_norms = {}

        # Hook to capture activations
        def make_hook(storage, name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                norm = output.norm(dim=(1, 2)).mean().item()
                if name not in storage:
                    storage[name] = []
                storage[name].append(norm)
            return hook_fn

        hooks = []

        # Register hooks on encoders (skip sources)
        if hasattr(model, 'encoders'):
            for i, enc in enumerate(model.encoders):
                hooks.append(enc.register_forward_hook(make_hook(skip_norms, f'skip_{i}')))

        # Register hooks on decoders
        if hasattr(model, 'decoders'):
            for i, dec in enumerate(model.decoders):
                hooks.append(dec.register_forward_hook(make_hook(decoder_norms, f'decoder_{i}')))

        # Forward pass
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)
                _ = model(x)
                break

        # Cleanup
        for h in hooks:
            h.remove()

        # Compute ratios
        results = {}
        for name, norms in skip_norms.items():
            results[f'{name}_norm'] = float(np.mean(norms))

        # Skip to decoder ratio
        for i in range(min(len(skip_norms), len(decoder_norms))):
            skip_key = f'skip_{i}'
            dec_key = f'decoder_{i}'
            if skip_key in skip_norms and dec_key in decoder_norms:
                ratio = np.mean(skip_norms[skip_key]) / (np.mean(decoder_norms[dec_key]) + 1e-10)
                results[f'skip_decoder_ratio_{i}'] = float(ratio)

        return results

    def depth_analysis(
        self,
        model_class,
        model_kwargs: Dict,
        dataloader: DataLoader,
        device: torch.device,
        depths: List[int] = [1, 2, 3, 4],
    ) -> Dict[int, float]:
        """Analyze performance at different network depths.

        Note: This requires training models at different depths.
        Returns placeholder if models not provided.

        Args:
            model_class: Model class to instantiate
            model_kwargs: Base kwargs for model
            dataloader: Data loader
            device: Device
            depths: Depths to test

        Returns:
            Dict of depth -> R² (placeholder values if not trained)
        """
        # This would require training models at different depths
        # Return placeholder indicating this needs separate training runs
        return {d: None for d in depths}


# =============================================================================
# PART 10: BASELINE COMPARISONS
# =============================================================================

class BaselineAnalyzer:
    """Compare model against baseline methods."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def linear_baselines(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        test_inputs: np.ndarray,
        test_targets: np.ndarray,
    ) -> Dict[str, float]:
        """Compare against linear baseline methods.

        Args:
            inputs: Training inputs [n_train, channels, time]
            targets: Training targets
            test_inputs: Test inputs
            test_targets: Test targets

        Returns:
            Dict with baseline R² scores
        """
        results = {}

        # Flatten for linear regression
        X_train = inputs.reshape(inputs.shape[0], -1)
        y_train = targets.reshape(targets.shape[0], -1)
        X_test = test_inputs.reshape(test_inputs.shape[0], -1)
        y_test = test_targets.reshape(test_targets.shape[0], -1)

        # Ridge regression
        try:
            from sklearn.linear_model import Ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            y_pred = ridge.predict(X_test)

            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            results['ridge_r2'] = float(1 - ss_res / (ss_tot + 1e-10))
        except ImportError:
            # Manual ridge regression
            results['ridge_r2'] = self._manual_ridge(X_train, y_train, X_test, y_test)

        # Per-channel linear regression
        n_channels = inputs.shape[1]
        channel_r2 = []

        for ch in range(n_channels):
            X_ch_train = inputs[:, ch, :].reshape(inputs.shape[0], -1)
            y_ch_train = targets[:, ch, :].reshape(targets.shape[0], -1)
            X_ch_test = test_inputs[:, ch, :].reshape(test_inputs.shape[0], -1)
            y_ch_test = test_targets[:, ch, :].reshape(test_targets.shape[0], -1)

            r2 = self._manual_ridge(X_ch_train, y_ch_train, X_ch_test, y_ch_test, alpha=1.0)
            channel_r2.append(r2)

        results['per_channel_r2'] = channel_r2
        results['mean_per_channel_r2'] = float(np.mean(channel_r2))

        return results

    def _manual_ridge(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        alpha: float = 1.0,
    ) -> float:
        """Simple ridge regression without sklearn."""
        # Add bias term
        X_train_b = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_test_b = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

        # Ridge solution: (X'X + alpha*I)^-1 X'y
        n_features = X_train_b.shape[1]
        XtX = X_train_b.T @ X_train_b + alpha * np.eye(n_features)
        Xty = X_train_b.T @ y_train

        try:
            weights = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

        y_pred = X_test_b @ weights

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

        return float(1 - ss_res / (ss_tot + 1e-10))

    def band_specific_comparison(
        self,
        model: nn.Module,
        inputs: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray,
        test_inputs: np.ndarray,
        test_targets: np.ndarray,
        fs: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compare model vs linear baseline per frequency band.

        Args:
            model: The trained model
            inputs: Training inputs
            targets: Training targets
            predictions: Model predictions on test set
            test_inputs: Test inputs
            test_targets: Test targets
            fs: Sampling frequency

        Returns:
            Dict with per-band comparison
        """
        fs = fs or self.config.fs

        results = {}

        for band_name, (low, high) in self.config.frequency_bands.items():
            if high >= fs / 2:
                continue

            # Filter signals
            nyq = fs / 2
            b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')

            # Filter test targets and predictions
            targets_filt = signal.filtfilt(b, a, test_targets, axis=-1)
            preds_filt = signal.filtfilt(b, a, predictions, axis=-1)

            # Model R² in this band
            ss_res = np.sum((targets_filt - preds_filt) ** 2)
            ss_tot = np.sum((targets_filt - np.mean(targets_filt)) ** 2)
            model_r2 = 1 - ss_res / (ss_tot + 1e-10)

            # Linear baseline in this band
            inputs_filt_train = signal.filtfilt(b, a, inputs, axis=-1)
            targets_filt_train = signal.filtfilt(b, a, targets, axis=-1)
            inputs_filt_test = signal.filtfilt(b, a, test_inputs, axis=-1)

            linear_r2 = self._manual_ridge(
                inputs_filt_train.reshape(inputs.shape[0], -1),
                targets_filt_train.reshape(targets.shape[0], -1),
                inputs_filt_test.reshape(test_inputs.shape[0], -1),
                targets_filt.reshape(test_targets.shape[0], -1),
            )

            results[band_name] = {
                'model_r2': float(model_r2),
                'linear_r2': float(linear_r2),
                'improvement': float(model_r2 - linear_r2),
            }

        return results


# =============================================================================
# MAIN VALIDATOR CLASS
# =============================================================================

class NeuralSignalValidator:
    """Main validation orchestrator for neural signal translation models."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ValidationConfig] = None,
    ):
        """Initialize validator.

        Args:
            model: The trained model to analyze
            config: Validation configuration
        """
        self.model = model
        self.config = config or ValidationConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        # Initialize analyzers
        self.training_analyzer = TrainingDynamicsAnalyzer(self.config)
        self.attribution_analyzer = InputAttributionAnalyzer(self.config)
        self.spectral_analyzer = SpectralAnalyzer(self.config)
        self.latent_analyzer = LatentSpaceAnalyzer(self.config)
        self.conditioning_analyzer = ConditioningAnalyzer(self.config)
        self.landscape_analyzer = LossLandscapeAnalyzer(self.config)
        self.generalization_analyzer = GeneralizationAnalyzer(self.config)
        self.error_analyzer = ErrorAnalyzer(self.config)
        self.architectural_analyzer = ArchitecturalAnalyzer(self.config)
        self.baseline_analyzer = BaselineAnalyzer(self.config)

    def run_full_analysis(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        session_loaders: Optional[Dict[str, DataLoader]] = None,
        training_history: Optional[Dict[str, List]] = None,
        skip_expensive: bool = False,
    ) -> FullValidationResult:
        """Run complete validation analysis.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            session_loaders: Dict of session-specific loaders for generalization analysis
            training_history: Dict with training metrics history
            skip_expensive: Skip computationally expensive analyses

        Returns:
            FullValidationResult with all analysis results
        """
        import datetime

        result = FullValidationResult()
        result.config_used = self.config
        result.analysis_timestamp = datetime.datetime.now().isoformat()

        self.model.to(self.device)
        self.model.eval()

        # Collect predictions for analyses that need them
        loader = test_loader or val_loader or train_loader
        if loader:
            inputs, targets, predictions = self._collect_predictions(loader)
        else:
            inputs, targets, predictions = None, None, None

        # Part 1: Training Dynamics
        if training_history:
            print("Analyzing training dynamics...")
            result.training_dynamics = self.training_analyzer.analyze_learning_curves(
                train_losses=training_history.get('train_loss', []),
                val_losses=training_history.get('val_loss', []),
                train_metrics={'r2': training_history.get('train_r2', [])},
                val_metrics={'r2': training_history.get('val_r2', [])},
            )

        # Part 2: Input Attribution
        if loader and not skip_expensive:
            print("Analyzing input attribution...")
            result.input_attribution = InputAttributionResult()
            result.input_attribution.channel_importance_gradient = \
                self.attribution_analyzer.channel_importance_gradient(
                    self.model, loader, self.device
                )
            result.input_attribution.temporal_saliency = \
                self.attribution_analyzer.temporal_saliency(
                    self.model, loader, self.device
                )

        # Part 3: Spectral Analysis
        if predictions is not None:
            print("Analyzing spectral properties...")
            result.spectral_analysis = SpectralAnalysisResult()

            band_accuracy = self.spectral_analyzer.band_wise_accuracy(predictions, targets)
            result.spectral_analysis.band_r2 = {k: v['r2'] for k, v in band_accuracy.items()}
            result.spectral_analysis.band_correlation = {k: v['correlation'] for k, v in band_accuracy.items()}

            psd_result = self.spectral_analyzer.psd_comparison(predictions, targets)
            result.spectral_analysis.psd_correlation = psd_result['psd_correlation']
            result.spectral_analysis.psd_rmse = psd_result['psd_rmse']
            result.spectral_analysis.mean_target_psd = psd_result['mean_target_psd']
            result.spectral_analysis.mean_pred_psd = psd_result['mean_pred_psd']
            result.spectral_analysis.psd_frequencies = psd_result['frequencies']

            # Coherence and phase
            coherence_result = self.spectral_analyzer.coherence_analysis(inputs, targets, predictions)
            result.spectral_analysis.coherence_real = coherence_result['coherence_real']
            result.spectral_analysis.coherence_predicted = coherence_result['coherence_predicted']
            result.spectral_analysis.coherence_preservation = coherence_result['coherence_preservation']

            phase_result = self.spectral_analyzer.phase_analysis(inputs, targets, predictions)
            result.spectral_analysis.plv_real = phase_result['plv_real']
            result.spectral_analysis.plv_predicted = phase_result['plv_predicted']
            result.spectral_analysis.phase_preservation = phase_result['phase_preservation']

        # Part 4: Latent Space
        if loader:
            print("Analyzing latent space...")
            result.latent_space = LatentSpaceResult()

            features, sessions, labels = self.latent_analyzer.extract_bottleneck_features(
                self.model, loader, self.device
            )
            result.latent_space.bottleneck_features = features
            result.latent_space.sample_sessions = sessions
            result.latent_space.sample_labels = labels

            if len(features) > 0:
                result.latent_space.pca_coords = self.latent_analyzer.visualize_latent_space(
                    features, method="pca"
                )

                dim_result = self.latent_analyzer.compute_dimensionality(features)
                result.latent_space.pca_explained_variance = dim_result['explained_variance']
                result.latent_space.effective_dimensionality = dim_result['effective_dimensionality']
                result.latent_space.intrinsic_dimensionality = dim_result['intrinsic_dimensionality']

                if inputs is not None:
                    rsa_result = self.latent_analyzer.representational_similarity_analysis(
                        features[:len(inputs)], inputs, targets
                    )
                    result.latent_space.rsa_input_correlation = rsa_result['rsa_input_correlation']
                    result.latent_space.rsa_output_correlation = rsa_result['rsa_output_correlation']

        # Part 5: Conditioning Analysis
        if hasattr(self.model, 'cond_mode') and self.model.cond_mode != 'none':
            print("Analyzing conditioning mechanism...")
            if loader:
                cond_ablation = self.conditioning_analyzer.conditioning_ablation(
                    self.model, loader, self.device
                )
                result.summary_metrics['conditioning'] = cond_ablation

        # Part 6: Loss Landscape (expensive)
        if loader and not skip_expensive:
            print("Computing loss landscape...")
            result.loss_landscape = self.landscape_analyzer.compute_loss_landscape(
                self.model, loader, self.device
            )

        # Part 7: Generalization
        if session_loaders:
            print("Analyzing generalization...")
            result.generalization = self.generalization_analyzer.per_session_analysis(
                self.model, session_loaders, self.device
            )

        # Part 8: Error Analysis
        if predictions is not None:
            print("Analyzing errors...")
            result.error_analysis = self.error_analyzer.analyze_error_distribution(
                predictions, targets
            )

            if loader:
                error_chars = self.error_analyzer.error_vs_characteristics(
                    self.model, loader, self.device
                )
                result.error_analysis.error_vs_amplitude = (
                    error_chars['amplitudes'], error_chars['errors']
                )
                result.error_analysis.error_vs_variance = (
                    error_chars['variances'], error_chars['errors']
                )

                worst_cases = self.error_analyzer.worst_case_analysis(
                    self.model, loader, self.device
                )
                result.error_analysis.worst_case_indices = worst_cases['worst_indices']
                result.error_analysis.worst_case_errors = worst_cases['worst_errors']
                result.error_analysis.worst_case_inputs = worst_cases['worst_inputs']
                result.error_analysis.worst_case_targets = worst_cases['worst_targets']
                result.error_analysis.worst_case_predictions = worst_cases['worst_predictions']

            residual_result = self.error_analyzer.residual_analysis(predictions, targets)
            result.error_analysis.residual_autocorrelation = residual_result['autocorrelation']
            result.error_analysis.residual_has_structure = residual_result['has_structure']
            result.error_analysis.residual_psd = residual_result['residual_psd']

        # Part 9: Architectural Insights
        if loader:
            print("Analyzing architecture...")
            result.architectural_insights = ArchitecturalInsightsResult()
            skip_result = self.architectural_analyzer.skip_connection_analysis(
                self.model, loader, self.device
            )
            result.architectural_insights.skip_activation_norms = {
                k: v for k, v in skip_result.items() if 'norm' in k
            }
            result.architectural_insights.skip_information_ratio = {
                k: v for k, v in skip_result.items() if 'ratio' in k
            }

        # Part 10: Baseline Comparison
        if inputs is not None and train_loader:
            print("Computing baseline comparisons...")
            train_inputs, train_targets, _ = self._collect_predictions(train_loader, include_preds=False)

            result.baseline_comparison = BaselineComparisonResult()
            linear_result = self.baseline_analyzer.linear_baselines(
                train_inputs, train_targets, inputs, targets
            )
            result.baseline_comparison.ridge_r2 = linear_result.get('ridge_r2')
            result.baseline_comparison.linear_per_channel_r2 = np.array(
                linear_result.get('per_channel_r2', [])
            )

            # Band-specific comparison
            band_comp = self.baseline_analyzer.band_specific_comparison(
                self.model, train_inputs, train_targets, predictions,
                inputs, targets
            )
            result.baseline_comparison.band_linear_r2 = {k: v['linear_r2'] for k, v in band_comp.items()}
            result.baseline_comparison.band_model_r2 = {k: v['model_r2'] for k, v in band_comp.items()}
            result.baseline_comparison.band_improvement = {k: v['improvement'] for k, v in band_comp.items()}

        # Compute summary metrics
        result.summary_metrics = self._compute_summary(result)

        print("Validation complete!")
        return result

    def _collect_predictions(
        self,
        dataloader: DataLoader,
        max_samples: int = 1000,
        include_preds: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Collect inputs, targets, and predictions from dataloader."""
        inputs_list = []
        targets_list = []
        preds_list = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    x, y = batch.to(self.device), batch.to(self.device)

                inputs_list.append(x.cpu().numpy())
                targets_list.append(y.cpu().numpy())

                if include_preds:
                    pred = self.model(x)
                    preds_list.append(pred.cpu().numpy())

                if sum(arr.shape[0] for arr in inputs_list) >= max_samples:
                    break

        inputs = np.concatenate(inputs_list, axis=0)[:max_samples]
        targets = np.concatenate(targets_list, axis=0)[:max_samples]

        if include_preds:
            predictions = np.concatenate(preds_list, axis=0)[:max_samples]
        else:
            predictions = None

        return inputs, targets, predictions

    def _compute_summary(self, result: FullValidationResult) -> Dict[str, Any]:
        """Compute summary metrics from full results."""
        summary = {}

        # Overall performance
        if result.spectral_analysis:
            summary['overall_r2'] = np.mean(list(result.spectral_analysis.band_r2.values()))
            summary['band_r2'] = result.spectral_analysis.band_r2
            summary['psd_correlation'] = result.spectral_analysis.psd_correlation

        # Generalization
        if result.generalization:
            summary['mean_session_r2'] = result.generalization.mean_r2
            summary['std_session_r2'] = result.generalization.std_r2
            summary['n_outlier_sessions'] = len(result.generalization.outlier_sessions)

        # Error characteristics
        if result.error_analysis:
            summary['error_mean'] = result.error_analysis.error_mean
            summary['error_std'] = result.error_analysis.error_std
            summary['residual_has_structure'] = result.error_analysis.residual_has_structure

        # Latent space
        if result.latent_space:
            summary['effective_dimensionality'] = result.latent_space.effective_dimensionality
            summary['rsa_output_correlation'] = result.latent_space.rsa_output_correlation

        # Baseline comparison
        if result.baseline_comparison:
            summary['ridge_baseline_r2'] = result.baseline_comparison.ridge_r2
            if result.spectral_analysis and result.baseline_comparison.ridge_r2:
                overall_r2 = summary.get('overall_r2', 0)
                summary['improvement_over_linear'] = overall_r2 - result.baseline_comparison.ridge_r2

        # Loss landscape
        if result.loss_landscape:
            summary['loss_landscape_sharpness'] = result.loss_landscape.sharpness_measure

        return summary

    def save_results(
        self,
        result: FullValidationResult,
        output_dir: Union[str, Path],
    ):
        """Save validation results to disk.

        Args:
            result: The validation results
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary as JSON
        summary_path = output_dir / "validation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(result.summary_metrics, f, indent=2, default=str)

        # Save numerical results as NPZ
        arrays_to_save = {}

        if result.spectral_analysis:
            if result.spectral_analysis.mean_pred_psd is not None:
                arrays_to_save['pred_psd'] = result.spectral_analysis.mean_pred_psd
                arrays_to_save['target_psd'] = result.spectral_analysis.mean_target_psd
                arrays_to_save['psd_frequencies'] = result.spectral_analysis.psd_frequencies

        if result.latent_space:
            if result.latent_space.bottleneck_features is not None:
                arrays_to_save['bottleneck_features'] = result.latent_space.bottleneck_features
            if result.latent_space.pca_coords is not None:
                arrays_to_save['latent_pca'] = result.latent_space.pca_coords

        if result.loss_landscape:
            if result.loss_landscape.landscape_grid is not None:
                arrays_to_save['loss_landscape'] = result.loss_landscape.landscape_grid

        if result.error_analysis:
            if result.error_analysis.worst_case_inputs is not None:
                arrays_to_save['worst_inputs'] = result.error_analysis.worst_case_inputs
                arrays_to_save['worst_targets'] = result.error_analysis.worst_case_targets
                arrays_to_save['worst_predictions'] = result.error_analysis.worst_case_predictions

        if arrays_to_save:
            np.savez(output_dir / "validation_arrays.npz", **arrays_to_save)

        print(f"Results saved to {output_dir}")

    def generate_report(
        self,
        result: FullValidationResult,
        output_dir: Union[str, Path],
    ) -> str:
        """Generate a text report summarizing validation results.

        Args:
            result: The validation results
            output_dir: Output directory

        Returns:
            Path to the report file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_lines = [
            "=" * 80,
            "NEURAL SIGNAL VALIDATION REPORT",
            "=" * 80,
            f"Generated: {result.analysis_timestamp}",
            "",
        ]

        # Summary section
        report_lines.extend([
            "SUMMARY METRICS",
            "-" * 40,
        ])
        for key, value in result.summary_metrics.items():
            if isinstance(value, float):
                report_lines.append(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                report_lines.append(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        report_lines.append(f"    {k}: {v:.4f}")
            else:
                report_lines.append(f"  {key}: {value}")
        report_lines.append("")

        # Spectral Analysis
        if result.spectral_analysis:
            report_lines.extend([
                "SPECTRAL ANALYSIS",
                "-" * 40,
                f"  PSD Correlation: {result.spectral_analysis.psd_correlation:.4f}",
                "",
                "  Band-wise R²:",
            ])
            for band, r2 in result.spectral_analysis.band_r2.items():
                report_lines.append(f"    {band}: {r2:.4f}")
            report_lines.append("")

        # Generalization
        if result.generalization:
            report_lines.extend([
                "GENERALIZATION",
                "-" * 40,
                f"  Mean R² across sessions: {result.generalization.mean_r2:.4f} +/- {result.generalization.std_r2:.4f}",
            ])
            if result.generalization.outlier_sessions:
                report_lines.append(f"  Outlier sessions: {result.generalization.outlier_sessions}")
            report_lines.append("")

        # Error Analysis
        if result.error_analysis:
            report_lines.extend([
                "ERROR ANALYSIS",
                "-" * 40,
                f"  Mean Error: {result.error_analysis.error_mean:.6f}",
                f"  Error Std: {result.error_analysis.error_std:.6f}",
                f"  Skewness: {result.error_analysis.error_skewness:.4f}",
                f"  Kurtosis: {result.error_analysis.error_kurtosis:.4f}",
                f"  Residuals have structure: {result.error_analysis.residual_has_structure}",
                "",
            ])

        # Latent Space
        if result.latent_space:
            report_lines.extend([
                "LATENT SPACE",
                "-" * 40,
                f"  Effective dimensionality (90% var): {result.latent_space.effective_dimensionality}",
                f"  Intrinsic dimensionality: {result.latent_space.intrinsic_dimensionality:.2f}",
            ])
            if result.latent_space.rsa_output_correlation:
                report_lines.append(f"  RSA (output): {result.latent_space.rsa_output_correlation:.4f}")
            report_lines.append("")

        # Baseline Comparison
        if result.baseline_comparison:
            report_lines.extend([
                "BASELINE COMPARISON",
                "-" * 40,
            ])
            if result.baseline_comparison.ridge_r2:
                report_lines.append(f"  Ridge Regression R²: {result.baseline_comparison.ridge_r2:.4f}")
            if result.baseline_comparison.band_improvement:
                report_lines.append("  Model improvement over linear by band:")
                for band, imp in result.baseline_comparison.band_improvement.items():
                    report_lines.append(f"    {band}: +{imp:.4f}")
            report_lines.append("")

        # Loss Landscape
        if result.loss_landscape:
            report_lines.extend([
                "LOSS LANDSCAPE",
                "-" * 40,
                f"  Minimum loss: {result.loss_landscape.minimum_loss:.6f}",
                f"  Sharpness: {result.loss_landscape.sharpness_measure:.6f}",
                "",
            ])

        report_lines.extend([
            "=" * 80,
            "END OF REPORT",
            "=" * 80,
        ])

        report_text = "\n".join(report_lines)

        report_path = output_dir / "validation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(report_text)
        return str(report_path)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_validation(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    """Run quick validation returning essential metrics.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device string

    Returns:
        Dict with essential metrics
    """
    config = ValidationConfig(device=device)
    validator = NeuralSignalValidator(model, config)

    result = validator.run_full_analysis(
        test_loader=test_loader,
        skip_expensive=True,
    )

    return result.summary_metrics


def validate_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model_class,
    model_kwargs: Dict,
    test_loader: DataLoader,
    device: str = "cuda",
) -> FullValidationResult:
    """Load model from checkpoint and run validation.

    Args:
        checkpoint_path: Path to model checkpoint
        model_class: Model class to instantiate
        model_kwargs: Model constructor arguments
        test_loader: Test data loader
        device: Device string

    Returns:
        FullValidationResult
    """
    # Load model
    model = model_class(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Run validation
    config = ValidationConfig(device=device)
    validator = NeuralSignalValidator(model, config)

    return validator.run_full_analysis(test_loader=test_loader)


__all__ = [
    # Config and results
    'ValidationConfig',
    'TrainingDynamicsResult',
    'InputAttributionResult',
    'SpectralAnalysisResult',
    'LatentSpaceResult',
    'LossLandscapeResult',
    'GeneralizationResult',
    'ErrorAnalysisResult',
    'ArchitecturalInsightsResult',
    'BaselineComparisonResult',
    'FullValidationResult',
    # Analyzers
    'TrainingDynamicsAnalyzer',
    'InputAttributionAnalyzer',
    'SpectralAnalyzer',
    'LatentSpaceAnalyzer',
    'ConditioningAnalyzer',
    'LossLandscapeAnalyzer',
    'GeneralizationAnalyzer',
    'ErrorAnalyzer',
    'ArchitecturalAnalyzer',
    'BaselineAnalyzer',
    # Main class
    'NeuralSignalValidator',
    # Convenience functions
    'quick_validation',
    'validate_from_checkpoint',
]
