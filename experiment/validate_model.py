"""Model validation and evaluation utilities.

This module provides tools for evaluating trained neural signal translation
models, including metrics computation and visualization.

Usage:
    # Evaluate a trained model
    python validate_model.py --checkpoint artifacts/checkpoints/best_model.pt

    # Generate detailed report
    python validate_model.py --checkpoint best_model.pt --output-dir results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from config import DataConfig, ModelConfig, ExperimentConfig
from data import load_data, create_dataloaders
from models import UNet1D, pearson_corr, explained_variance, normalized_rmse


# =============================================================================
# Metrics
# =============================================================================

@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: float = 1000.0,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.

    Args:
        pred: Predicted signals (batch, channels, samples)
        target: Target signals (batch, channels, samples)
        sample_rate: Sampling rate in Hz

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    # Correlation
    metrics["pearson_r"] = pearson_corr(pred, target).item()

    # Explained variance
    metrics["explained_variance"] = explained_variance(pred, target).item()

    # Normalized RMSE
    metrics["nrmse"] = normalized_rmse(pred, target).item()

    # MSE and MAE
    metrics["mse"] = F.mse_loss(pred, target).item()
    metrics["mae"] = F.l1_loss(pred, target).item()

    # R-squared
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    metrics["r_squared"] = (1 - ss_res / ss_tot.clamp(min=1e-8)).item()

    # Power Spectral Density error
    try:
        pred_psd = compute_psd(pred, sample_rate)
        target_psd = compute_psd(target, sample_rate)
        psd_error = torch.abs(pred_psd - target_psd).mean()
        metrics["psd_error"] = psd_error.item()

        # Log PSD error (dB)
        pred_psd_db = 10 * torch.log10(pred_psd.clamp(min=1e-10))
        target_psd_db = 10 * torch.log10(target_psd.clamp(min=1e-10))
        metrics["psd_error_db"] = torch.abs(pred_psd_db - target_psd_db).mean().item()
    except Exception:
        metrics["psd_error"] = float("nan")
        metrics["psd_error_db"] = float("nan")

    return metrics


def compute_psd(
    x: torch.Tensor,
    sample_rate: float = 1000.0,
    nperseg: int = 256,
) -> torch.Tensor:
    """Compute power spectral density using Welch's method.

    Args:
        x: Input signal (batch, channels, samples)
        sample_rate: Sampling rate in Hz
        nperseg: Segment length for Welch's method

    Returns:
        PSD estimate (batch, channels, n_freqs)
    """
    B, C, T = x.shape

    # Use FFT-based PSD estimation
    # Window the signal
    n_segments = max(1, T // nperseg)
    hop = nperseg // 2

    psd_sum = None
    count = 0

    for start in range(0, T - nperseg + 1, hop):
        segment = x[..., start:start + nperseg]
        # Apply Hann window
        window = torch.hann_window(nperseg, device=x.device)
        segment = segment * window

        # FFT
        fft = torch.fft.rfft(segment.float(), dim=-1)
        psd = (fft.abs() ** 2) / (nperseg * sample_rate)

        if psd_sum is None:
            psd_sum = psd
        else:
            psd_sum = psd_sum + psd
        count += 1

    return psd_sum / max(count, 1)


def compute_frequency_band_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: float = 1000.0,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics per frequency band.

    Args:
        pred: Predicted signals
        target: Target signals
        sample_rate: Sampling rate in Hz

    Returns:
        Dictionary of band names to metrics
    """
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 80),
    }

    results = {}
    T = pred.shape[-1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / sample_rate)

    for band_name, (f_low, f_high) in bands.items():
        # Bandpass filter
        mask = (freqs >= f_low) & (freqs < f_high)

        pred_fft = torch.fft.rfft(pred.float(), dim=-1)
        target_fft = torch.fft.rfft(target.float(), dim=-1)

        pred_fft_filtered = pred_fft * mask.float().unsqueeze(0).unsqueeze(0)
        target_fft_filtered = target_fft * mask.float().unsqueeze(0).unsqueeze(0)

        pred_filtered = torch.fft.irfft(pred_fft_filtered, n=T, dim=-1)
        target_filtered = torch.fft.irfft(target_fft_filtered, n=T, dim=-1)

        results[band_name] = {
            "correlation": pearson_corr(pred_filtered, target_filtered).item(),
            "mse": F.mse_loss(pred_filtered, target_filtered).item(),
        }

    return results


# =============================================================================
# Evaluation
# =============================================================================

class Evaluator:
    """Model evaluation manager."""

    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        sample_rate: float = 1000.0,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.data_loader = data_loader
        self.device = device
        self.sample_rate = sample_rate

    @torch.no_grad()
    def evaluate(self) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Run full evaluation.

        Returns:
            overall_metrics: Dictionary of overall metrics
            band_metrics: Dictionary of per-band metrics
        """
        all_pred = []
        all_target = []

        for batch in self.data_loader:
            x = batch["input"].to(self.device)
            y = batch["target"].to(self.device)

            pred = self.model(x)

            all_pred.append(pred.cpu())
            all_target.append(y.cpu())

        all_pred = torch.cat(all_pred, dim=0)
        all_target = torch.cat(all_target, dim=0)

        # Compute metrics
        overall = compute_metrics(all_pred, all_target, self.sample_rate)
        bands = compute_frequency_band_metrics(all_pred, all_target, self.sample_rate)

        return overall, bands

    @torch.no_grad()
    def get_predictions(self, n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get sample predictions for visualization.

        Args:
            n_samples: Number of samples to return

        Returns:
            inputs, predictions, targets as numpy arrays
        """
        inputs = []
        preds = []
        targets = []

        count = 0
        for batch in self.data_loader:
            x = batch["input"].to(self.device)
            y = batch["target"].to(self.device)

            pred = self.model(x)

            for i in range(min(x.shape[0], n_samples - count)):
                inputs.append(x[i].cpu().numpy())
                preds.append(pred[i].cpu().numpy())
                targets.append(y[i].cpu().numpy())
                count += 1

            if count >= n_samples:
                break

        return np.array(inputs), np.array(preds), np.array(targets)


# =============================================================================
# Visualization
# =============================================================================

def plot_sample_predictions(
    inputs: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    channel: int = 0,
    n_samples: int = 3,
    sample_rate: float = 1000.0,
    save_path: Optional[Path] = None,
) -> None:
    """Plot sample predictions vs targets.

    Args:
        inputs: Input signals (n, channels, samples)
        predictions: Predicted signals
        targets: Target signals
        channel: Which channel to plot
        n_samples: Number of samples to plot
        sample_rate: Sampling rate in Hz
        save_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting")
        return

    n_samples = min(n_samples, len(inputs))
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        t = np.arange(targets.shape[-1]) / sample_rate

        # Time domain
        ax = axes[i, 0]
        ax.plot(t, targets[i, channel], 'b-', alpha=0.7, label='Target', linewidth=0.8)
        ax.plot(t, predictions[i, channel], 'r-', alpha=0.7, label='Prediction', linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Sample {i + 1} - Time Domain')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Frequency domain (PSD)
        ax = axes[i, 1]
        freqs = np.fft.rfftfreq(targets.shape[-1], d=1.0 / sample_rate)
        target_psd = np.abs(np.fft.rfft(targets[i, channel])) ** 2
        pred_psd = np.abs(np.fft.rfft(predictions[i, channel])) ** 2

        ax.semilogy(freqs[:100], target_psd[:100], 'b-', alpha=0.7, label='Target')
        ax.semilogy(freqs[:100], pred_psd[:100], 'r-', alpha=0.7, label='Prediction')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title(f'Sample {i + 1} - PSD')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()


def plot_metrics_summary(
    metrics: Dict[str, float],
    band_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
) -> None:
    """Plot metrics summary.

    Args:
        metrics: Overall metrics dictionary
        band_metrics: Per-band metrics dictionary
        save_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overall metrics bar chart
    ax = axes[0]
    metric_names = ['pearson_r', 'explained_variance', 'r_squared']
    values = [metrics.get(m, 0) for m in metric_names]
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    bars = ax.bar(metric_names, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Overall Metrics')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)

    # Per-band correlation
    ax = axes[1]
    bands = list(band_metrics.keys())
    correlations = [band_metrics[b]['correlation'] for b in bands]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bands)))
    bars = ax.bar(bands, correlations, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation by Frequency Band')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, correlations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str,
                        help="Path to experiment config (optional)")
    parser.add_argument("--data-path", type=str,
                        help="Path to data file (overrides config)")
    parser.add_argument("--output-dir", type=str, default="evaluation",
                        help="Output directory for results")
    parser.add_argument("--device", type=str,
                        help="Device to use")
    parser.add_argument("--n-plot-samples", type=int, default=5,
                        help="Number of samples to plot")

    return parser.parse_args()


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Setup device
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Load config
    config_path = Path(args.config) if args.config else checkpoint_path.parent.parent / "config.json"
    if config_path.exists():
        config = ExperimentConfig.load(config_path)
        data_cfg = config.data
        model_cfg = config.model
    else:
        # Try to infer from checkpoint
        print("Warning: No config found, using checkpoint metadata")
        data_cfg_dict = checkpoint.get("data_cfg", {})
        # Handle tuple conversion for channel ranges
        if "region1_channels" in data_cfg_dict and isinstance(data_cfg_dict["region1_channels"], list):
            data_cfg_dict["region1_channels"] = tuple(data_cfg_dict["region1_channels"])
        if "region2_channels" in data_cfg_dict and isinstance(data_cfg_dict["region2_channels"], list):
            data_cfg_dict["region2_channels"] = tuple(data_cfg_dict["region2_channels"])
        data_cfg = DataConfig(**data_cfg_dict) if data_cfg_dict else DataConfig(data_path="")
        model_cfg = ModelConfig(
            in_channels=checkpoint.get("in_channels", 32),
            out_channels=checkpoint.get("out_channels", 32),
        )

    # Override data path if specified
    if args.data_path:
        data_cfg.data_path = args.data_path

    # Check data path
    if not data_cfg.data_path or not Path(data_cfg.data_path).exists():
        print(f"Error: Data file not found: {data_cfg.data_path}")
        print("Please specify --data-path")
        return

    # Load data
    print(f"Loading data from {data_cfg.data_path}")
    region1, region2, labels = load_data(data_cfg)
    _, _, test_loader = create_dataloaders(region1, region2, labels, data_cfg)

    # Create model
    model = UNet1D.from_config(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded with {model.count_parameters():,} parameters")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    print("\nRunning evaluation...")
    evaluator = Evaluator(model, test_loader, device, data_cfg.sampling_rate)
    overall_metrics, band_metrics = evaluator.evaluate()

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print("\nOverall Metrics:")
    for name, value in overall_metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\nPer-Band Metrics:")
    for band, metrics in band_metrics.items():
        print(f"  {band}:")
        for name, value in metrics.items():
            print(f"    {name}: {value:.4f}")

    # Save results
    results = {
        "overall": overall_metrics,
        "per_band": band_metrics,
        "checkpoint": str(checkpoint_path),
    }
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    if HAS_MATPLOTLIB:
        print("\nGenerating plots...")

        # Sample predictions
        inputs, preds, targets = evaluator.get_predictions(args.n_plot_samples)
        plot_sample_predictions(
            inputs, preds, targets,
            save_path=output_dir / "sample_predictions.png",
            sample_rate=data_cfg.sampling_rate,
        )

        # Metrics summary
        plot_metrics_summary(
            overall_metrics, band_metrics,
            save_path=output_dir / "metrics_summary.png",
        )

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
