#!/usr/bin/env python3
"""Test Session Normalization - Verify cross-session distribution alignment.

This script tests the proposed normalization strategies:
1. Per-session Z-score normalization
2. Spectral (PSD) normalization
3. Combined normalization

Generates before/after comparison plots to verify effectiveness.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# Limit thread counts BEFORE importing numpy/scipy
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_MAX_THREADS'] = '4'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch, hilbert
from scipy.interpolate import interp1d
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data import prepare_data, SAMPLING_RATE_HZ


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_per_session_zscore(
    data: np.ndarray,
    session_ids: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Z-score normalize each session independently.

    For each session, each channel:
    - Subtract session+channel mean
    - Divide by session+channel std

    This removes session-specific DC offsets and gain differences.

    Args:
        data: Signal array [trials, channels, time]
        session_ids: Session ID per trial [trials]
        verbose: Print progress

    Returns:
        normalized: Normalized data [trials, channels, time]
        stats: Dict of per-session statistics for inverse transform
    """
    normalized = data.copy()
    session_stats = {}

    unique_sessions = np.unique(session_ids)

    if verbose:
        print(f"\n{'='*60}")
        print("Applying Per-Session Z-Score Normalization")
        print(f"{'='*60}")

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = data[mask]  # [n_trials, channels, time]
        n_trials = sess_data.shape[0]

        # Compute statistics per channel (across all trials and time in session)
        # Shape: [channels]
        mean_per_channel = sess_data.mean(axis=(0, 2))  # Mean over trials and time
        std_per_channel = sess_data.std(axis=(0, 2))    # Std over trials and time

        # Avoid division by zero
        std_per_channel = np.maximum(std_per_channel, 1e-8)

        # Normalize: (x - mean) / std
        # Broadcast: [n_trials, channels, time] - [channels] -> need to reshape
        normalized[mask] = (sess_data - mean_per_channel[None, :, None]) / std_per_channel[None, :, None]

        session_stats[sess_id] = {
            'mean': mean_per_channel,
            'std': std_per_channel,
            'n_trials': n_trials,
        }

        if verbose:
            print(f"  Session {sess_id}: {n_trials} trials, "
                  f"mean range [{mean_per_channel.min():.3f}, {mean_per_channel.max():.3f}], "
                  f"std range [{std_per_channel.min():.3f}, {std_per_channel.max():.3f}]")

    if verbose:
        print(f"{'='*60}\n")

    return normalized, session_stats


def compute_session_psd(
    data: np.ndarray,
    session_ids: np.ndarray,
    fs: int = 1000,
    nperseg: int = 256,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Compute average PSD for each session.

    Args:
        data: Signal array [trials, channels, time]
        session_ids: Session ID per trial
        fs: Sampling frequency
        nperseg: Segment length for Welch method

    Returns:
        freqs: Frequency bins
        session_psds: Dict mapping session_id -> average PSD [channels, freqs]
    """
    unique_sessions = np.unique(session_ids)
    session_psds = {}

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = data[mask]  # [n_trials, channels, time]

        # Compute PSD for each trial, then average
        psds = []
        for trial in sess_data:
            freqs, psd = welch(trial, fs=fs, nperseg=nperseg, axis=1)
            psds.append(psd)

        session_psds[sess_id] = np.mean(psds, axis=0)  # [channels, freqs]

    return freqs, session_psds


def normalize_spectral(
    data: np.ndarray,
    session_ids: np.ndarray,
    reference_psd: Optional[np.ndarray] = None,
    train_session_ids: Optional[np.ndarray] = None,
    fs: int = 1000,
    nperseg: int = 256,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize each session's PSD to match a reference.

    This corrects for different amplifier gains and frequency responses
    across recording sessions.

    Args:
        data: Signal array [trials, channels, time]
        session_ids: Session ID per trial
        reference_psd: Target PSD to match [channels, freqs]. If None, use median of train sessions.
        train_session_ids: Session IDs to use for computing reference (if reference_psd is None)
        fs: Sampling frequency
        nperseg: Segment length for Welch method
        verbose: Print progress

    Returns:
        normalized: Spectrally normalized data
        info: Dict with reference PSD and correction filters
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Applying Spectral (PSD) Normalization")
        print(f"{'='*60}")

    # Compute session PSDs
    freqs, session_psds = compute_session_psd(data, session_ids, fs, nperseg)

    # Compute reference PSD if not provided
    if reference_psd is None:
        if train_session_ids is not None:
            # Use median of training sessions
            train_psds = [session_psds[sid] for sid in np.unique(train_session_ids) if sid in session_psds]
        else:
            # Use median of all sessions
            train_psds = list(session_psds.values())

        reference_psd = np.median(train_psds, axis=0)  # [channels, freqs]
        if verbose:
            print(f"  Reference PSD: median of {len(train_psds)} sessions")

    # Normalize each session
    normalized = data.copy()
    correction_filters = {}

    unique_sessions = np.unique(session_ids)
    n_time = data.shape[2]

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_psd = session_psds[sess_id]  # [channels, freqs]

        # Compute correction filter: sqrt(reference / session)
        # This scales the amplitude spectrum to match reference
        with np.errstate(divide='ignore', invalid='ignore'):
            correction = np.sqrt(reference_psd / (sess_psd + 1e-10))
        correction = np.nan_to_num(correction, nan=1.0, posinf=1.0, neginf=1.0)

        # Clip extreme corrections to avoid amplifying noise
        correction = np.clip(correction, 0.1, 10.0)

        # Interpolate correction filter to match FFT bins
        n_fft = n_time // 2 + 1
        fft_freqs = np.fft.rfftfreq(n_time, 1/fs)

        correction_interp = np.zeros((data.shape[1], n_fft))
        for ch in range(data.shape[1]):
            interp_func = interp1d(freqs, correction[ch], kind='linear',
                                   fill_value='extrapolate', bounds_error=False)
            correction_interp[ch] = interp_func(fft_freqs)

        correction_filters[sess_id] = correction_interp

        # Apply correction in frequency domain
        sess_data = data[mask]
        for i, trial in enumerate(sess_data):
            fft = np.fft.rfft(trial, axis=1)
            fft_corrected = fft * correction_interp
            normalized[mask][i] = np.fft.irfft(fft_corrected, n=n_time, axis=1)

        if verbose:
            avg_correction = np.mean(correction)
            print(f"  Session {sess_id}: avg correction factor {avg_correction:.2f}")

    info = {
        'reference_psd': reference_psd,
        'freqs': freqs,
        'correction_filters': correction_filters,
        'session_psds': session_psds,
    }

    if verbose:
        print(f"{'='*60}\n")

    return normalized, info


def normalize_combined(
    data: np.ndarray,
    session_ids: np.ndarray,
    train_session_ids: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply both z-score and spectral normalization.

    Order: Z-score first (removes DC/gain), then spectral (fixes frequency response)

    Args:
        data: Signal array [trials, channels, time]
        session_ids: Session ID per trial
        train_session_ids: Session IDs for reference computation
        verbose: Print progress

    Returns:
        normalized: Fully normalized data
        info: Dict with all normalization parameters
    """
    if verbose:
        print("\n" + "="*70)
        print("COMBINED NORMALIZATION PIPELINE")
        print("="*70)

    # Step 1: Z-score normalization
    zscore_normalized, zscore_stats = normalize_per_session_zscore(
        data, session_ids, verbose=verbose
    )

    # Step 2: Spectral normalization
    spectral_normalized, spectral_info = normalize_spectral(
        zscore_normalized, session_ids,
        train_session_ids=train_session_ids,
        verbose=verbose
    )

    info = {
        'zscore_stats': zscore_stats,
        'spectral_info': spectral_info,
    }

    return spectral_normalized, info


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_session_statistics(
    data: np.ndarray,
    session_ids: np.ndarray,
) -> Dict:
    """Compute comprehensive statistics per session."""
    stats_dict = {}
    unique_sessions = np.unique(session_ids)

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = data[mask]

        # Per-channel mean and std
        channel_means = sess_data.mean(axis=(0, 2))  # [channels]
        channel_stds = sess_data.std(axis=(0, 2))

        # Overall statistics
        overall_mean = sess_data.mean()
        overall_std = sess_data.std()

        # Envelope (via Hilbert)
        envelopes = []
        for trial in sess_data[:10]:  # Sample for speed
            for ch in range(trial.shape[0]):
                env = np.abs(hilbert(trial[ch]))
                envelopes.extend(env)

        # PSD
        freqs, psd = welch(sess_data[0], fs=SAMPLING_RATE_HZ, nperseg=256, axis=1)
        avg_psd = np.mean([welch(t, fs=SAMPLING_RATE_HZ, nperseg=256, axis=1)[1]
                          for t in sess_data[:10]], axis=0)

        stats_dict[sess_id] = {
            'channel_means': channel_means,
            'channel_stds': channel_stds,
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'envelope_mean': np.mean(envelopes),
            'envelope_std': np.std(envelopes),
            'psd': avg_psd,
            'psd_freqs': freqs,
        }

    return stats_dict


def compute_cross_session_metrics(stats_before: Dict, stats_after: Dict) -> Dict:
    """Compute metrics showing how well sessions are aligned."""
    sessions = list(stats_before.keys())

    # Channel mean variance across sessions
    means_before = np.array([stats_before[s]['channel_means'] for s in sessions])
    means_after = np.array([stats_after[s]['channel_means'] for s in sessions])

    mean_var_before = np.var(means_before, axis=0).mean()  # Avg variance per channel
    mean_var_after = np.var(means_after, axis=0).mean()

    # Channel std variance across sessions
    stds_before = np.array([stats_before[s]['channel_stds'] for s in sessions])
    stds_after = np.array([stats_after[s]['channel_stds'] for s in sessions])

    std_var_before = np.var(stds_before, axis=0).mean()
    std_var_after = np.var(stds_after, axis=0).mean()

    # PSD similarity (using correlation)
    psds_before = np.array([stats_before[s]['psd'].mean(axis=0) for s in sessions])  # Avg over channels
    psds_after = np.array([stats_after[s]['psd'].mean(axis=0) for s in sessions])

    psd_corr_before = np.corrcoef(psds_before).mean()
    psd_corr_after = np.corrcoef(psds_after).mean()

    return {
        'mean_variance': {'before': mean_var_before, 'after': mean_var_after},
        'std_variance': {'before': std_var_before, 'after': std_var_after},
        'psd_correlation': {'before': psd_corr_before, 'after': psd_corr_after},
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_comparison(
    data_before: np.ndarray,
    data_after: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict,
    train_idx: np.ndarray,
    output_dir: Path,
    title_suffix: str = "",
):
    """Generate before/after comparison plots."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])
    n_sessions = len(unique_sessions)

    colors = plt.cm.tab20(np.linspace(0, 1, n_sessions))

    # ==========================================================================
    # Plot 1: Per-Channel Mean Comparison
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, sess_id in enumerate(unique_sessions):
        mask = session_ids == sess_id
        linestyle = '-' if sess_id in train_sessions else '--'

        # Before
        channel_means_before = data_before[mask].mean(axis=(0, 2))
        axes[0].plot(channel_means_before, color=colors[i], linestyle=linestyle,
                    label=idx_to_session[sess_id], alpha=0.8)

        # After
        channel_means_after = data_after[mask].mean(axis=(0, 2))
        axes[1].plot(channel_means_after, color=colors[i], linestyle=linestyle,
                    label=idx_to_session[sess_id], alpha=0.8)

    axes[0].set_title('Per-Channel Mean by Session (BEFORE)')
    axes[0].set_xlabel('Channel')
    axes[0].set_ylabel('Mean')
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    axes[0].axhline(y=0, color='k', linestyle=':', alpha=0.5)

    axes[1].set_title('Per-Channel Mean by Session (AFTER)')
    axes[1].set_xlabel('Channel')
    axes[1].set_ylabel('Mean')
    axes[1].axhline(y=0, color='k', linestyle=':', alpha=0.5)

    # Compute variance reduction
    means_before = np.array([data_before[session_ids == s].mean(axis=(0, 2)) for s in unique_sessions])
    means_after = np.array([data_after[session_ids == s].mean(axis=(0, 2)) for s in unique_sessions])
    var_before = np.var(means_before, axis=0).mean()
    var_after = np.var(means_after, axis=0).mean()
    reduction = (1 - var_after / var_before) * 100 if var_before > 0 else 0

    fig.suptitle(f'Per-Channel Mean Comparison{title_suffix}\n'
                 f'Cross-session variance: {var_before:.6f} → {var_after:.6f} ({reduction:.1f}% reduction)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_channel_means.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ==========================================================================
    # Plot 2: PSD Comparison
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, sess_id in enumerate(unique_sessions):
        mask = session_ids == sess_id
        linestyle = '-' if sess_id in train_sessions else '--'

        # Sample trials for PSD
        sess_data_before = data_before[mask][:10]
        sess_data_after = data_after[mask][:10]

        # Compute average PSD
        psds_before = []
        psds_after = []
        for trial_b, trial_a in zip(sess_data_before, sess_data_after):
            freqs, psd_b = welch(trial_b, fs=SAMPLING_RATE_HZ, nperseg=256, axis=1)
            _, psd_a = welch(trial_a, fs=SAMPLING_RATE_HZ, nperseg=256, axis=1)
            psds_before.append(psd_b.mean(axis=0))  # Avg over channels
            psds_after.append(psd_a.mean(axis=0))

        avg_psd_before = np.mean(psds_before, axis=0)
        avg_psd_after = np.mean(psds_after, axis=0)

        axes[0].semilogy(freqs, avg_psd_before, color=colors[i], linestyle=linestyle,
                        label=idx_to_session[sess_id], alpha=0.8)
        axes[1].semilogy(freqs, avg_psd_after, color=colors[i], linestyle=linestyle,
                        label=idx_to_session[sess_id], alpha=0.8)

    axes[0].set_title('PSD by Session (BEFORE)')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('PSD')
    axes[0].set_xlim([0, 100])
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    axes[1].set_title('PSD by Session (AFTER)')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('PSD')
    axes[1].set_xlim([0, 100])

    fig.suptitle(f'PSD Comparison{title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_psd.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ==========================================================================
    # Plot 3: Envelope Distribution Comparison
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, sess_id in enumerate(unique_sessions):
        mask = session_ids == sess_id
        linestyle = '-' if sess_id in train_sessions else '--'

        # Compute envelope (sample for speed)
        sess_data_before = data_before[mask][:20]
        sess_data_after = data_after[mask][:20]

        env_before = []
        env_after = []
        for trial_b, trial_a in zip(sess_data_before, sess_data_after):
            for ch in range(trial_b.shape[0]):
                env_before.extend(np.abs(hilbert(trial_b[ch])))
                env_after.extend(np.abs(hilbert(trial_a[ch])))

        # KDE
        from scipy.stats import gaussian_kde
        try:
            kde_before = gaussian_kde(np.array(env_before)[:10000])
            kde_after = gaussian_kde(np.array(env_after)[:10000])
            x = np.linspace(0, 5, 200)

            axes[0].plot(x, kde_before(x), color=colors[i], linestyle=linestyle,
                        label=idx_to_session[sess_id], alpha=0.8)
            axes[1].plot(x, kde_after(x), color=colors[i], linestyle=linestyle,
                        label=idx_to_session[sess_id], alpha=0.8)
        except:
            pass

    axes[0].set_title('Envelope Distribution by Session (BEFORE)')
    axes[0].set_xlabel('Envelope Magnitude')
    axes[0].set_ylabel('Density')
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    axes[1].set_title('Envelope Distribution by Session (AFTER)')
    axes[1].set_xlabel('Envelope Magnitude')
    axes[1].set_ylabel('Density')

    fig.suptitle(f'Envelope Distribution Comparison{title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_envelope.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ==========================================================================
    # Plot 4: Train vs Held-out KS Test
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    train_mask = np.isin(session_ids, list(train_sessions))
    holdout_mask = ~train_mask

    # Flatten data for comparison
    train_before = data_before[train_mask].flatten()
    holdout_before = data_before[holdout_mask].flatten()
    train_after = data_after[train_mask].flatten()
    holdout_after = data_after[holdout_mask].flatten()

    # Sample for histogram
    n_sample = 100000

    # Before - histogram
    axes[0, 0].hist(np.random.choice(train_before, min(n_sample, len(train_before)), replace=False),
                    bins=100, alpha=0.5, label='Train', density=True)
    axes[0, 0].hist(np.random.choice(holdout_before, min(n_sample, len(holdout_before)), replace=False),
                    bins=100, alpha=0.5, label='Held-out', density=True)
    ks_before = stats.ks_2samp(train_before[:n_sample], holdout_before[:n_sample])
    axes[0, 0].set_title(f'Amplitude Distribution (BEFORE)\nKS p={ks_before.pvalue:.2e}')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Amplitude')
    axes[0, 0].set_ylabel('Density')

    # After - histogram
    axes[0, 1].hist(np.random.choice(train_after, min(n_sample, len(train_after)), replace=False),
                    bins=100, alpha=0.5, label='Train', density=True)
    axes[0, 1].hist(np.random.choice(holdout_after, min(n_sample, len(holdout_after)), replace=False),
                    bins=100, alpha=0.5, label='Held-out', density=True)
    ks_after = stats.ks_2samp(train_after[:n_sample], holdout_after[:n_sample])
    axes[0, 1].set_title(f'Amplitude Distribution (AFTER)\nKS p={ks_after.pvalue:.2e}')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Amplitude')
    axes[0, 1].set_ylabel('Density')

    # Gradient comparison
    train_grad_before = np.diff(data_before[train_mask], axis=2).flatten()
    holdout_grad_before = np.diff(data_before[holdout_mask], axis=2).flatten()
    train_grad_after = np.diff(data_after[train_mask], axis=2).flatten()
    holdout_grad_after = np.diff(data_after[holdout_mask], axis=2).flatten()

    axes[1, 0].hist(np.random.choice(train_grad_before, min(n_sample, len(train_grad_before)), replace=False),
                    bins=100, alpha=0.5, label='Train', density=True)
    axes[1, 0].hist(np.random.choice(holdout_grad_before, min(n_sample, len(holdout_grad_before)), replace=False),
                    bins=100, alpha=0.5, label='Held-out', density=True)
    ks_grad_before = stats.ks_2samp(train_grad_before[:n_sample], holdout_grad_before[:n_sample])
    axes[1, 0].set_title(f'Gradient Distribution (BEFORE)\nKS p={ks_grad_before.pvalue:.2e}')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Gradient')
    axes[1, 0].set_ylabel('Density')

    axes[1, 1].hist(np.random.choice(train_grad_after, min(n_sample, len(train_grad_after)), replace=False),
                    bins=100, alpha=0.5, label='Train', density=True)
    axes[1, 1].hist(np.random.choice(holdout_grad_after, min(n_sample, len(holdout_grad_after)), replace=False),
                    bins=100, alpha=0.5, label='Held-out', density=True)
    ks_grad_after = stats.ks_2samp(train_grad_after[:n_sample], holdout_grad_after[:n_sample])
    axes[1, 1].set_title(f'Gradient Distribution (AFTER)\nKS p={ks_grad_after.pvalue:.2e}')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Gradient')
    axes[1, 1].set_ylabel('Density')

    fig.suptitle(f'Train vs Held-out Distribution Comparison{title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_train_vs_holdout.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ==========================================================================
    # Print Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("NORMALIZATION RESULTS SUMMARY")
    print("="*70)
    print(f"\nPer-Channel Mean Variance Across Sessions:")
    print(f"  BEFORE: {var_before:.6f}")
    print(f"  AFTER:  {var_after:.6f}")
    print(f"  Reduction: {reduction:.1f}%")

    print(f"\nKS Test (Train vs Held-out) - Amplitude:")
    print(f"  BEFORE: p = {ks_before.pvalue:.2e}")
    print(f"  AFTER:  p = {ks_after.pvalue:.2e}")
    print(f"  Improvement: {ks_after.pvalue / ks_before.pvalue:.1f}x" if ks_before.pvalue > 0 else "")

    print(f"\nKS Test (Train vs Held-out) - Gradient:")
    print(f"  BEFORE: p = {ks_grad_before.pvalue:.2e}")
    print(f"  AFTER:  p = {ks_grad_after.pvalue:.2e}")

    print(f"\nPlots saved to: {output_dir}")
    print("="*70)

    return {
        'mean_variance_before': var_before,
        'mean_variance_after': var_after,
        'ks_amplitude_before': ks_before.pvalue,
        'ks_amplitude_after': ks_after.pvalue,
        'ks_gradient_before': ks_grad_before.pvalue,
        'ks_gradient_after': ks_grad_after.pvalue,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test session normalization strategies')
    parser.add_argument('--output-dir', type=str, default='artifacts/normalization_test',
                        help='Output directory for plots')
    parser.add_argument('--method', type=str, default='combined',
                        choices=['zscore', 'spectral', 'combined'],
                        help='Normalization method to test')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SESSION NORMALIZATION TEST")
    print("="*70)

    # Load data with session splits
    print("\nLoading data...")
    try:
        data = prepare_data(
            split_by_session=True,
            n_val_sessions=4,
            no_test_set=True,
            separate_val_sessions=True,
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nThis script requires the data files to be available.")
        print("To test per-session normalization, you can:")
        print("  1. Mount/copy the data files to /data/")
        print("  2. Or run training directly with --per-session-normalize flag:")
        print("     python train.py --split-by-session --per-session-normalize")
        print("\nPer-session normalization is enabled by default in the config.")
        print("Check DEFAULT_CONFIG['per_session_normalize'] in train.py")
        return None

    ob = data['ob']
    session_ids = data['session_ids']
    idx_to_session = data['idx_to_session']
    train_idx = data['train_idx']

    print(f"\nData shape: {ob.shape}")
    print(f"Sessions: {len(np.unique(session_ids))}")
    print(f"Train samples: {len(train_idx)}")

    # Get train session IDs for reference computation
    train_session_ids = session_ids[train_idx]

    # Apply normalization based on method
    print(f"\nApplying {args.method} normalization...")

    if args.method == 'zscore':
        ob_normalized, info = normalize_per_session_zscore(ob, session_ids)
        title_suffix = " (Z-Score Only)"
    elif args.method == 'spectral':
        ob_normalized, info = normalize_spectral(ob, session_ids, train_session_ids=train_session_ids)
        title_suffix = " (Spectral Only)"
    else:  # combined
        ob_normalized, info = normalize_combined(ob, session_ids, train_session_ids=train_session_ids)
        title_suffix = " (Combined: Z-Score + Spectral)"

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    results = plot_comparison(
        ob, ob_normalized, session_ids, idx_to_session, train_idx,
        output_dir, title_suffix=title_suffix
    )

    print("\n✅ Test complete! Check the plots in:", output_dir)

    return results


if __name__ == '__main__':
    main()
