#!/usr/bin/env python3
"""
Analyze Instantaneous Frequency Distribution for Loss Ablation Experiments

This script:
1. Loads checkpoints from each loss ablation experiment
2. Generates predictions on validation data
3. Computes instantaneous frequency using Hilbert transform
4. Plots distribution comparison: Generated vs Real
5. Saves plots to each experiment's folder

Usage:
    python analyze_instantaneous_freq.py [--exp-dir PATH] [--n-samples N]

Example:
    python analyze_instantaneous_freq.py --exp-dir artifacts/loss_ablation --n-samples 100
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.signal import hilbert
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import CondUNet1D
from data import prepare_data, create_dataloaders, SAMPLING_RATE_HZ


def compute_instantaneous_frequency(
    signal: np.ndarray,
    fs: float = 1000.0,
    min_freq: float = 1.0,
    max_freq: float = 100.0,
) -> np.ndarray:
    """
    Compute instantaneous frequency using Hilbert transform.

    Args:
        signal: Input signal [C, T] or [T]
        fs: Sampling frequency in Hz
        min_freq: Minimum valid frequency (filter out artifacts)
        max_freq: Maximum valid frequency (filter out artifacts)

    Returns:
        Instantaneous frequency values (flattened, filtered)
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    all_inst_freq = []

    for ch in range(signal.shape[0]):
        # Get analytic signal via Hilbert transform
        analytic = hilbert(signal[ch])

        # Compute instantaneous phase
        inst_phase = np.unwrap(np.angle(analytic))

        # Compute instantaneous frequency: d(phase)/dt / (2*pi)
        inst_freq = np.diff(inst_phase) * fs / (2 * np.pi)

        # Filter to valid range (remove artifacts from edges, noise)
        valid_mask = (inst_freq >= min_freq) & (inst_freq <= max_freq)
        all_inst_freq.extend(inst_freq[valid_mask])

    return np.array(all_inst_freq)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[CondUNet1D, Dict]:
    """Load model from checkpoint, inferring architecture from state dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get state dict and clean keys (remove FSDP/DDP prefixes)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', {}))
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '').replace('_fsdp_wrapped_module.', '')
        clean_state_dict[new_key] = v

    # Infer n_downsample from number of encoders
    n_downsample = 0
    for key in clean_state_dict:
        if key.startswith('encoders.'):
            idx = int(key.split('.')[1])
            n_downsample = max(n_downsample, idx + 1)

    # Infer base channels from inc block output
    base_channels = 64  # default
    for key in clean_state_dict:
        if 'inc.conv.0.branches.0.bias' in key:  # inception conv
            base_channels = clean_state_dict[key].shape[0]
            break
        elif 'inc.conv.2.bias' in key:  # standard conv
            base_channels = clean_state_dict[key].shape[0]
            break

    # Infer conv_type from state dict keys
    conv_type = 'inception' if any('branches' in k for k in clean_state_dict) else 'standard'

    # Get cond_mode from checkpoint (this IS saved)
    cond_mode = checkpoint.get('cond_mode', 'cross_attn_gated')

    # Build model with inferred parameters
    model = CondUNet1D(
        in_channels=32,
        out_channels=32,
        base=base_channels,
        n_odors=7,
        emb_dim=128,
        dropout=0.0,
        use_attention=True,
        attention_type='cross_freq_v2',
        norm_type='batch',
        cond_mode=cond_mode,
        use_spectral_shift=True,
        n_downsample=n_downsample,
        conv_type=conv_type,
        use_se=True,
        conv_kernel_size=7,
        dilations=(1, 4, 16, 32),
        kernel_sizes=(3, 5, 7),
        use_dilation_grid=False,
        use_output_scaling=True,
    ).to(device)

    # Create inferred config dict for return
    config = {
        'n_downsample': n_downsample,
        'base_channels': base_channels,
        'conv_type': conv_type,
        'cond_mode': cond_mode,
    }

    # Load weights
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    return model, config


def generate_predictions(
    model: CondUNet1D,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: Optional[int] = None,
    config: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and collect real targets.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run on
        n_samples: If None, use all samples in dataloader
        config: Model config (unused, for compatibility)

    Returns:
        pred_signals: [N, C, T] array of predictions
        real_signals: [N, C, T] array of real targets
    """
    pred_list = []
    real_list = []

    model.eval()
    with torch.no_grad():
        for ob, pcx, odor in tqdm(dataloader, desc="Generating predictions"):
            ob = ob.to(device)
            odor = odor.to(device)

            # Generate prediction
            pred = model(ob, odor)

            # Crop to match target size if needed
            if pred.shape[-1] > pcx.shape[-1]:
                diff = pred.shape[-1] - pcx.shape[-1]
                start = diff // 2
                pred = pred[..., start:start + pcx.shape[-1]]

            pred_list.append(pred.cpu().numpy())
            real_list.append(pcx.numpy())

    pred_signals = np.concatenate(pred_list, axis=0)
    real_signals = np.concatenate(real_list, axis=0)

    return pred_signals, real_signals


def plot_instantaneous_freq_comparison(
    pred_signals: np.ndarray,
    real_signals: np.ndarray,
    exp_name: str,
    save_path: Path,
    fs: float = 1000.0,
    n_bins: int = 100,
):
    """
    Plot instantaneous frequency distribution comparison.

    Args:
        pred_signals: [N, C, T] predicted signals
        real_signals: [N, C, T] real target signals
        exp_name: Experiment name for title
        save_path: Path to save the figure
        fs: Sampling frequency
        n_bins: Number of histogram bins
    """
    print(f"Computing instantaneous frequencies for {exp_name}...")

    # Compute instantaneous frequency for all samples
    pred_inst_freq = []
    real_inst_freq = []

    for i in tqdm(range(len(pred_signals)), desc="Computing inst. freq"):
        pred_if = compute_instantaneous_frequency(pred_signals[i], fs=fs)
        real_if = compute_instantaneous_frequency(real_signals[i], fs=fs)
        pred_inst_freq.extend(pred_if)
        real_inst_freq.extend(real_if)

    pred_inst_freq = np.array(pred_inst_freq)
    real_inst_freq = np.array(real_inst_freq)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Instantaneous Frequency Analysis: {exp_name}', fontsize=14, fontweight='bold')

    # 1. Overlaid histograms
    ax1 = axes[0, 0]
    bins = np.linspace(1, 100, n_bins)
    ax1.hist(real_inst_freq, bins=bins, alpha=0.6, label='Real (Target)', density=True, color='blue')
    ax1.hist(pred_inst_freq, bins=bins, alpha=0.6, label='Generated (Pred)', density=True, color='red')
    ax1.set_xlabel('Instantaneous Frequency (Hz)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison')
    ax1.legend()
    ax1.set_xlim(1, 100)
    ax1.grid(True, alpha=0.3)

    # 2. Log-scale histogram (to see tails better)
    ax2 = axes[0, 1]
    ax2.hist(real_inst_freq, bins=bins, alpha=0.6, label='Real', density=True, color='blue')
    ax2.hist(pred_inst_freq, bins=bins, alpha=0.6, label='Generated', density=True, color='red')
    ax2.set_xlabel('Instantaneous Frequency (Hz)')
    ax2.set_ylabel('Density (log scale)')
    ax2.set_title('Distribution Comparison (Log Scale)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.set_xlim(1, 100)
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    ax3.hist(real_inst_freq, bins=bins, alpha=0.6, label='Real', density=True, cumulative=True, color='blue')
    ax3.hist(pred_inst_freq, bins=bins, alpha=0.6, label='Generated', density=True, cumulative=True, color='red')
    ax3.set_xlabel('Instantaneous Frequency (Hz)')
    ax3.set_ylabel('Cumulative Density')
    ax3.set_title('Cumulative Distribution')
    ax3.legend()
    ax3.set_xlim(1, 100)
    ax3.grid(True, alpha=0.3)

    # 4. Statistics comparison (box plot style)
    ax4 = axes[1, 1]

    # Compute statistics
    stats_text = []
    stats_text.append("Statistics Comparison:")
    stats_text.append("-" * 40)
    stats_text.append(f"{'Metric':<20} {'Real':>10} {'Generated':>10}")
    stats_text.append("-" * 40)

    real_mean = np.mean(real_inst_freq)
    pred_mean = np.mean(pred_inst_freq)
    stats_text.append(f"{'Mean (Hz)':<20} {real_mean:>10.2f} {pred_mean:>10.2f}")

    real_std = np.std(real_inst_freq)
    pred_std = np.std(pred_inst_freq)
    stats_text.append(f"{'Std Dev (Hz)':<20} {real_std:>10.2f} {pred_std:>10.2f}")

    real_median = np.median(real_inst_freq)
    pred_median = np.median(pred_inst_freq)
    stats_text.append(f"{'Median (Hz)':<20} {real_median:>10.2f} {pred_median:>10.2f}")

    # Percentiles
    for p in [25, 75, 90, 95]:
        real_p = np.percentile(real_inst_freq, p)
        pred_p = np.percentile(pred_inst_freq, p)
        stats_text.append(f"{f'{p}th percentile':<20} {real_p:>10.2f} {pred_p:>10.2f}")

    # Frequency band analysis
    stats_text.append("-" * 40)
    stats_text.append("Frequency Band Coverage (%):")
    bands = [
        ("Delta (1-4 Hz)", 1, 4),
        ("Theta (4-8 Hz)", 4, 8),
        ("Alpha (8-13 Hz)", 8, 13),
        ("Beta (13-30 Hz)", 13, 30),
        ("Low Gamma (30-50 Hz)", 30, 50),
        ("High Gamma (50-100 Hz)", 50, 100),
    ]

    for band_name, low, high in bands:
        real_pct = 100 * np.sum((real_inst_freq >= low) & (real_inst_freq < high)) / len(real_inst_freq)
        pred_pct = 100 * np.sum((pred_inst_freq >= low) & (pred_inst_freq < high)) / len(pred_inst_freq)
        stats_text.append(f"{band_name:<20} {real_pct:>10.1f} {pred_pct:>10.1f}")

    # Display as text
    ax4.axis('off')
    ax4.text(0.05, 0.95, '\n'.join(stats_text), transform=ax4.transAxes,
             fontfamily='monospace', fontsize=9, verticalalignment='top')

    plt.tight_layout()

    # Save figure
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")

    # Return key metrics for summary
    return {
        'exp_name': exp_name,
        'real_mean': real_mean,
        'pred_mean': pred_mean,
        'real_std': real_std,
        'pred_std': pred_std,
        'mean_diff': abs(pred_mean - real_mean),
        'std_diff': abs(pred_std - real_std),
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze instantaneous frequency distributions')
    parser.add_argument('--exp-dir', type=str, default='artifacts/loss_ablation',
                        help='Directory containing experiment folders')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for inference')
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    device = torch.device(args.device)

    print(f"Experiment directory: {exp_dir}")
    print(f"Device: {device}")
    print()

    # Find all experiment folders with checkpoints
    experiments = []
    for d in sorted(exp_dir.iterdir()):
        if d.is_dir():
            checkpoint_path = d / "checkpoints" / "best_model.pt"
            if checkpoint_path.exists():
                experiments.append((d.name, checkpoint_path))
            else:
                # Try alternative names
                for alt_name in ["best_model_stage2.pt", "checkpoint.pt"]:
                    alt_path = d / "checkpoints" / alt_name
                    if alt_path.exists():
                        experiments.append((d.name, alt_path))
                        break

    if not experiments:
        print(f"No experiments found in {exp_dir}")
        print("Make sure experiments have been run and checkpoints exist.")
        return

    print(f"Found {len(experiments)} experiments:")
    for name, path in experiments:
        print(f"  - {name}")
    print()

    # Load test data once
    print("Loading test data...")
    data = prepare_data(seed=42)
    loaders = create_dataloaders(data, batch_size=args.batch_size)
    test_loader = loaders['test']
    print(f"Test samples: {len(test_loader.dataset)}")
    print()

    # Analyze each experiment
    all_metrics = []

    for exp_name, checkpoint_path in experiments:
        print(f"\n{'='*60}")
        print(f"Analyzing: {exp_name}")
        print(f"{'='*60}")

        try:
            # Load model
            print(f"Loading checkpoint: {checkpoint_path}")
            model, config = load_checkpoint(checkpoint_path, device)

            # Generate predictions on ALL test samples
            pred_signals, real_signals = generate_predictions(
                model, test_loader, device,
            )

            # Plot and save
            save_path = exp_dir / exp_name / "instantaneous_freq_analysis.png"
            metrics = plot_instantaneous_freq_comparison(
                pred_signals, real_signals,
                exp_name=exp_name,
                save_path=save_path,
                fs=config.get('sampling_rate', SAMPLING_RATE_HZ),
            )
            all_metrics.append(metrics)

        except Exception as e:
            print(f"ERROR analyzing {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Instantaneous Frequency Analysis")
    print("="*80)
    print(f"{'Experiment':<30} {'Real Mean':>10} {'Pred Mean':>10} {'Mean Diff':>10} {'Std Diff':>10}")
    print("-"*80)

    for m in sorted(all_metrics, key=lambda x: x['mean_diff']):
        print(f"{m['exp_name']:<30} {m['real_mean']:>10.2f} {m['pred_mean']:>10.2f} "
              f"{m['mean_diff']:>10.2f} {m['std_diff']:>10.2f}")

    print("-"*80)
    print("Lower 'Mean Diff' and 'Std Diff' = better frequency distribution match")
    print()

    # Save summary to CSV
    summary_path = exp_dir / "instantaneous_freq_summary.csv"
    with open(summary_path, 'w') as f:
        f.write("experiment,real_mean,pred_mean,real_std,pred_std,mean_diff,std_diff\n")
        for m in all_metrics:
            f.write(f"{m['exp_name']},{m['real_mean']:.4f},{m['pred_mean']:.4f},"
                    f"{m['real_std']:.4f},{m['pred_std']:.4f},"
                    f"{m['mean_diff']:.4f},{m['std_diff']:.4f}\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
