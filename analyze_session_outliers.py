#!/usr/bin/env python3
"""
Session Outlier Analysis for Phase 3
=====================================
Analyzes WHY certain sessions are outliers in cross-session generalization.

Produces:
1. PCA plot showing session clustering (outliers clearly separated)
2. Per-session performance breakdown
3. Analysis of session differences (spectral content, SNR, recording conditions)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def load_session_data():
    """Load the olfactory dataset and extract per-session statistics."""
    import pandas as pd
    from data import load_signals, load_odor_labels, extract_window, DATA_PATH, ODOR_CSV_PATH

    print("Loading olfactory dataset...")

    # Load raw signals and metadata
    signals = load_signals(DATA_PATH)  # [N, 2, C, T]
    num_trials = signals.shape[0]

    # Load odor labels and metadata
    odors, vocab = load_odor_labels(ODOR_CSV_PATH, num_trials)

    # Load session metadata from CSV
    df = pd.read_csv(ODOR_CSV_PATH)
    if len(df) > num_trials:
        df = df.iloc[:num_trials]

    # Extract window (standard preprocessing)
    windowed = extract_window(signals)  # [N, 2, C, T_window]
    ob_all = windowed[:, 0]  # [N, C, T]
    pcx_all = windowed[:, 1]  # [N, C, T]

    # Get unique sessions
    sessions = df['recording_id'].unique()
    print(f"Found {len(sessions)} sessions: {list(sessions)}")

    session_stats = {}
    session_data = {}

    for session in sessions:
        mask = df['recording_id'] == session
        indices = np.where(mask)[0]

        # Get signals for this session
        ob_stack = ob_all[indices]  # [N_session, C, T]
        pcx_stack = pcx_all[indices]  # [N_session, C, T]

        session_odors = df.loc[mask, 'odor_name'].unique().tolist() if 'odor_name' in df.columns else []

        session_data[session] = {
            'ob': ob_stack,
            'pcx': pcx_stack,
            'n_trials': len(indices),
            'odors': session_odors,
        }

        # Compute statistics
        session_stats[session] = compute_session_statistics(ob_stack, pcx_stack, session)

    return session_data, session_stats


def compute_session_statistics(ob: np.ndarray, pcx: np.ndarray, session_name: str) -> Dict:
    """Compute comprehensive statistics for a session.

    Args:
        ob: OB signals [N, C, T]
        pcx: PCx signals [N, C, T]
        session_name: Session identifier

    Returns:
        Dict of statistics
    """
    stats_dict = {
        'session': session_name,
        'n_trials': ob.shape[0],
        'n_channels': ob.shape[1],
        'n_timepoints': ob.shape[2],
    }

    # Amplitude statistics
    stats_dict['ob_mean'] = float(np.mean(ob))
    stats_dict['ob_std'] = float(np.std(ob))
    stats_dict['ob_min'] = float(np.min(ob))
    stats_dict['ob_max'] = float(np.max(ob))
    stats_dict['ob_range'] = stats_dict['ob_max'] - stats_dict['ob_min']

    stats_dict['pcx_mean'] = float(np.mean(pcx))
    stats_dict['pcx_std'] = float(np.std(pcx))
    stats_dict['pcx_min'] = float(np.min(pcx))
    stats_dict['pcx_max'] = float(np.max(pcx))
    stats_dict['pcx_range'] = stats_dict['pcx_max'] - stats_dict['pcx_min']

    # SNR estimation (signal variance / noise variance)
    # Estimate noise as high-frequency content
    ob_flat = ob.reshape(-1, ob.shape[-1])
    pcx_flat = pcx.reshape(-1, pcx.shape[-1])

    # Simple SNR: variance of signal / variance of diff (proxy for noise)
    ob_noise_est = np.var(np.diff(ob_flat, axis=-1))
    pcx_noise_est = np.var(np.diff(pcx_flat, axis=-1))

    stats_dict['ob_snr'] = float(np.var(ob_flat) / (ob_noise_est + 1e-10))
    stats_dict['pcx_snr'] = float(np.var(pcx_flat) / (pcx_noise_est + 1e-10))

    # Spectral statistics (power in frequency bands)
    fs = 1000  # Sampling rate

    # Average power spectrum across trials and channels
    ob_avg = ob.mean(axis=(0, 1))  # [T]
    pcx_avg = pcx.mean(axis=(0, 1))  # [T]

    # Compute FFT
    n = len(ob_avg)
    freqs = np.fft.rfftfreq(n, 1/fs)
    ob_fft = np.abs(np.fft.rfft(ob_avg))
    pcx_fft = np.abs(np.fft.rfft(pcx_avg))

    # Power in frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 100),
    }

    for band_name, (f_low, f_high) in bands.items():
        mask = (freqs >= f_low) & (freqs < f_high)
        stats_dict[f'ob_{band_name}_power'] = float(np.mean(ob_fft[mask]**2))
        stats_dict[f'pcx_{band_name}_power'] = float(np.mean(pcx_fft[mask]**2))

    # Total power
    stats_dict['ob_total_power'] = float(np.mean(ob_fft**2))
    stats_dict['pcx_total_power'] = float(np.mean(pcx_fft**2))

    # Relative band powers (as fraction of total)
    for band_name in bands:
        stats_dict[f'ob_{band_name}_rel'] = stats_dict[f'ob_{band_name}_power'] / (stats_dict['ob_total_power'] + 1e-10)
        stats_dict[f'pcx_{band_name}_rel'] = stats_dict[f'pcx_{band_name}_power'] / (stats_dict['pcx_total_power'] + 1e-10)

    # Cross-correlation between OB and PCx (proxy for coupling strength)
    ob_norm = (ob_avg - ob_avg.mean()) / (ob_avg.std() + 1e-10)
    pcx_norm = (pcx_avg - pcx_avg.mean()) / (pcx_avg.std() + 1e-10)
    xcorr = np.correlate(ob_norm, pcx_norm, mode='full')
    stats_dict['ob_pcx_max_xcorr'] = float(np.max(np.abs(xcorr)) / len(ob_norm))

    # Channel variance (electrode quality proxy)
    stats_dict['ob_channel_var'] = float(np.var(ob.mean(axis=(0, 2))))  # Variance across channels
    stats_dict['pcx_channel_var'] = float(np.var(pcx.mean(axis=(0, 2))))

    return stats_dict


def create_pca_plot(session_stats: Dict, performance: Dict, output_path: Path):
    """Create PCA plot showing session clustering with outliers highlighted.

    Args:
        session_stats: Dict of session -> statistics
        performance: Dict of session -> R² performance
        output_path: Path to save figure
    """
    # Extract feature matrix
    sessions = list(session_stats.keys())

    # Select features for PCA
    feature_names = [
        'ob_std', 'pcx_std', 'ob_snr', 'pcx_snr',
        'ob_delta_rel', 'ob_theta_rel', 'ob_beta_rel', 'ob_gamma_rel',
        'pcx_delta_rel', 'pcx_theta_rel', 'pcx_beta_rel', 'pcx_gamma_rel',
        'ob_pcx_max_xcorr', 'ob_channel_var', 'pcx_channel_var',
    ]

    X = []
    for session in sessions:
        row = [session_stats[session].get(f, 0) for f in feature_names]
        X.append(row)
    X = np.array(X)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Get performance values
    r2_values = [performance.get(s, 0) for s in sessions]
    r2_array = np.array(r2_values)

    # Identify outliers (sessions with R² significantly below mean)
    r2_mean = np.mean(r2_array)
    r2_std = np.std(r2_array)
    outlier_threshold = r2_mean - 1.0 * r2_std  # 1 std below mean
    is_outlier = r2_array < outlier_threshold

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: PCA colored by R²
    ax1 = axes[0]
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=r2_array,
                          cmap='RdYlGn', s=200, edgecolors='black', linewidths=2)

    # Highlight outliers
    ax1.scatter(X_pca[is_outlier, 0], X_pca[is_outlier, 1],
                facecolors='none', edgecolors='red', s=350, linewidths=3,
                label='Outliers (low R²)')

    # Add session labels
    for i, session in enumerate(sessions):
        ax1.annotate(session, (X_pca[i, 0], X_pca[i, 1]),
                     fontsize=8, ha='center', va='bottom',
                     xytext=(0, 10), textcoords='offset points')

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    ax1.set_title('Session Clustering by Signal Characteristics\n(Color = R² Performance)')
    ax1.legend(loc='upper right')

    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('R² Score')

    # Plot 2: Feature importance (PCA loadings)
    ax2 = axes[1]
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Plot arrows for top features
    for i, fname in enumerate(feature_names):
        ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                  head_width=0.05, head_length=0.02, fc='blue', ec='blue', alpha=0.7)
        ax2.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, fname,
                 fontsize=7, ha='center', va='center')

    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('PC1 Loading')
    ax2.set_ylabel('PC2 Loading')
    ax2.set_title('Feature Contributions to PCA')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA plot to {output_path}")

    return sessions, is_outlier, X_pca


def create_performance_breakdown(session_stats: Dict, performance: Dict, output_path: Path):
    """Create per-session performance breakdown with statistics.

    Args:
        session_stats: Dict of session -> statistics
        performance: Dict of session -> R² performance
        output_path: Path to save figure
    """
    sessions = sorted(performance.keys(), key=lambda s: performance[s], reverse=True)
    r2_values = [performance[s] for s in sessions]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: R² by session (bar chart)
    ax1 = axes[0, 0]
    colors = ['green' if r > 0.2 else 'orange' if r > 0.1 else 'red' for r in r2_values]
    bars = ax1.bar(range(len(sessions)), r2_values, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(sessions)))
    ax1.set_xticklabels(sessions, rotation=45, ha='right')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Per-Session R² Performance')
    ax1.axhline(y=np.mean(r2_values), color='blue', linestyle='--', label=f'Mean: {np.mean(r2_values):.3f}')
    ax1.legend()

    # Plot 2: SNR comparison
    ax2 = axes[0, 1]
    ob_snr = [session_stats[s]['ob_snr'] for s in sessions]
    pcx_snr = [session_stats[s]['pcx_snr'] for s in sessions]
    x = np.arange(len(sessions))
    width = 0.35
    ax2.bar(x - width/2, ob_snr, width, label='OB SNR', color='steelblue')
    ax2.bar(x + width/2, pcx_snr, width, label='PCx SNR', color='coral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sessions, rotation=45, ha='right')
    ax2.set_ylabel('SNR')
    ax2.set_title('Signal-to-Noise Ratio by Session')
    ax2.legend()

    # Plot 3: Spectral content (gamma power - important for neural signals)
    ax3 = axes[1, 0]
    ob_gamma = [session_stats[s]['ob_gamma_rel'] for s in sessions]
    pcx_gamma = [session_stats[s]['pcx_gamma_rel'] for s in sessions]
    ax3.bar(x - width/2, ob_gamma, width, label='OB Gamma', color='purple')
    ax3.bar(x + width/2, pcx_gamma, width, label='PCx Gamma', color='gold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sessions, rotation=45, ha='right')
    ax3.set_ylabel('Relative Gamma Power')
    ax3.set_title('Gamma Band Power (30-100 Hz) by Session')
    ax3.legend()

    # Plot 4: OB-PCx correlation vs performance
    ax4 = axes[1, 1]
    xcorr = [session_stats[s]['ob_pcx_max_xcorr'] for s in sessions]
    colors = [performance[s] for s in sessions]
    scatter = ax4.scatter(xcorr, r2_values, c=colors, cmap='RdYlGn', s=150, edgecolors='black')
    for i, s in enumerate(sessions):
        ax4.annotate(s, (xcorr[i], r2_values[i]), fontsize=8, ha='left')
    ax4.set_xlabel('OB-PCx Cross-Correlation')
    ax4.set_ylabel('R² Score')
    ax4.set_title('Performance vs OB-PCx Coupling')

    # Add correlation line
    slope, intercept, r, p, se = stats.linregress(xcorr, r2_values)
    ax4.plot([min(xcorr), max(xcorr)],
             [slope*min(xcorr)+intercept, slope*max(xcorr)+intercept],
             'r--', label=f'r={r:.2f}, p={p:.3f}')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved performance breakdown to {output_path}")


def analyze_outlier_differences(session_stats: Dict, performance: Dict, outlier_sessions: List[str]):
    """Analyze what makes outlier sessions different.

    Args:
        session_stats: Dict of session -> statistics
        performance: Dict of session -> R² performance
        outlier_sessions: List of outlier session names
    """
    print("\n" + "="*70)
    print("OUTLIER SESSION ANALYSIS")
    print("="*70)

    good_sessions = [s for s in session_stats if s not in outlier_sessions]

    # Compare key metrics
    metrics_to_compare = [
        ('ob_snr', 'OB Signal-to-Noise Ratio'),
        ('pcx_snr', 'PCx Signal-to-Noise Ratio'),
        ('ob_std', 'OB Amplitude Std'),
        ('pcx_std', 'PCx Amplitude Std'),
        ('ob_gamma_rel', 'OB Relative Gamma Power'),
        ('pcx_gamma_rel', 'PCx Relative Gamma Power'),
        ('ob_pcx_max_xcorr', 'OB-PCx Cross-Correlation'),
        ('ob_channel_var', 'OB Channel Variance'),
        ('pcx_channel_var', 'PCx Channel Variance'),
        ('n_trials', 'Number of Trials'),
    ]

    print(f"\nOutlier sessions: {outlier_sessions}")
    print(f"Good sessions: {good_sessions}")
    print(f"\n{'Metric':<35} {'Outliers':<15} {'Good':<15} {'Diff %':<10}")
    print("-"*75)

    significant_diffs = []

    for metric, name in metrics_to_compare:
        outlier_vals = [session_stats[s][metric] for s in outlier_sessions]
        good_vals = [session_stats[s][metric] for s in good_sessions]

        outlier_mean = np.mean(outlier_vals)
        good_mean = np.mean(good_vals)

        if good_mean != 0:
            diff_pct = ((outlier_mean - good_mean) / abs(good_mean)) * 100
        else:
            diff_pct = 0

        # Statistical test
        if len(outlier_vals) >= 2 and len(good_vals) >= 2:
            t_stat, p_val = stats.ttest_ind(outlier_vals, good_vals)
            sig = "*" if p_val < 0.1 else ""
        else:
            p_val = 1.0
            sig = ""

        print(f"{name:<35} {outlier_mean:<15.4f} {good_mean:<15.4f} {diff_pct:>+8.1f}% {sig}")

        if abs(diff_pct) > 20:
            significant_diffs.append((name, diff_pct, outlier_mean, good_mean))

    print("\n" + "="*70)
    print("KEY DIFFERENCES (>20% difference):")
    print("="*70)

    for name, diff_pct, outlier_val, good_val in sorted(significant_diffs, key=lambda x: abs(x[1]), reverse=True):
        direction = "HIGHER" if diff_pct > 0 else "LOWER"
        print(f"  - {name}: Outliers are {abs(diff_pct):.1f}% {direction}")
        print(f"    Outliers: {outlier_val:.4f}, Good sessions: {good_val:.4f}")

    # Per-session detailed breakdown
    print("\n" + "="*70)
    print("PER-SESSION DETAILED BREAKDOWN:")
    print("="*70)

    for session in sorted(session_stats.keys(), key=lambda s: performance.get(s, 0)):
        s = session_stats[session]
        r2 = performance.get(session, 0)
        is_outlier = "** OUTLIER **" if session in outlier_sessions else ""

        print(f"\n{session} (R²={r2:.3f}) {is_outlier}")
        print(f"  Trials: {s['n_trials']}, Channels: {s['n_channels']}")
        print(f"  OB:  SNR={s['ob_snr']:.2f}, std={s['ob_std']:.4f}, gamma={s['ob_gamma_rel']:.4f}")
        print(f"  PCx: SNR={s['pcx_snr']:.2f}, std={s['pcx_std']:.4f}, gamma={s['pcx_gamma_rel']:.4f}")
        print(f"  OB-PCx correlation: {s['ob_pcx_max_xcorr']:.4f}")


def create_power_spectrum_comparison(session_data: Dict, performance: Dict, output_path: Path):
    """Create power spectrum comparison across sessions.

    Args:
        session_data: Dict of session -> raw signal data
        performance: Dict of session -> R² performance
        output_path: Path to save figure
    """
    fs = 1000  # Sampling rate

    # Sort sessions by performance
    sessions = sorted(performance.keys(), key=lambda s: performance[s], reverse=True)
    n_sessions = len(sessions)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colors based on performance
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=min(performance.values()), vmax=max(performance.values()))

    # Plot 1: OB Power Spectra
    ax1 = axes[0, 0]
    for session in sessions:
        if session not in session_data:
            continue
        ob = session_data[session]['ob']  # [N, C, T]
        # Average across trials and channels
        ob_avg = ob.mean(axis=(0, 1))
        n = len(ob_avg)
        freqs = np.fft.rfftfreq(n, 1/fs)
        psd = np.abs(np.fft.rfft(ob_avg))**2

        # Smooth with moving average
        window = 5
        psd_smooth = np.convolve(psd, np.ones(window)/window, mode='same')

        color = cmap(norm(performance[session]))
        ax1.semilogy(freqs[freqs < 100], psd_smooth[freqs < 100],
                     label=f"{session} (R²={performance[session]:.2f})",
                     color=color, alpha=0.8)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power (log)')
    ax1.set_title('OB Power Spectra by Session')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_xlim(0, 100)

    # Plot 2: PCx Power Spectra
    ax2 = axes[0, 1]
    for session in sessions:
        if session not in session_data:
            continue
        pcx = session_data[session]['pcx']
        pcx_avg = pcx.mean(axis=(0, 1))
        n = len(pcx_avg)
        freqs = np.fft.rfftfreq(n, 1/fs)
        psd = np.abs(np.fft.rfft(pcx_avg))**2
        psd_smooth = np.convolve(psd, np.ones(5)/5, mode='same')

        color = cmap(norm(performance[session]))
        ax2.semilogy(freqs[freqs < 100], psd_smooth[freqs < 100],
                     label=f"{session}", color=color, alpha=0.8)

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (log)')
    ax2.set_title('PCx Power Spectra by Session')
    ax2.set_xlim(0, 100)

    # Plot 3: Band power comparison (stacked bar)
    ax3 = axes[1, 0]
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    x = np.arange(len(sessions))
    width = 0.15
    colors_bands = ['navy', 'blue', 'green', 'orange', 'red']

    for i, band in enumerate(bands):
        ob_powers = []
        for session in sessions:
            if session in session_data:
                ob = session_data[session]['ob'].mean(axis=(0, 1))
                freqs = np.fft.rfftfreq(len(ob), 1/fs)
                psd = np.abs(np.fft.rfft(ob))**2

                band_ranges = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12),
                               'beta': (12, 30), 'gamma': (30, 100)}
                f_low, f_high = band_ranges[band]
                mask = (freqs >= f_low) & (freqs < f_high)
                ob_powers.append(np.mean(psd[mask]))
            else:
                ob_powers.append(0)

        ax3.bar(x + i*width, ob_powers, width, label=band, color=colors_bands[i])

    ax3.set_xticks(x + 2*width)
    ax3.set_xticklabels(sessions, rotation=45, ha='right')
    ax3.set_ylabel('Power')
    ax3.set_title('OB Band Power by Session')
    ax3.legend()

    # Plot 4: Gamma/Theta ratio (important for neural coding)
    ax4 = axes[1, 1]
    gamma_theta_ratios = []
    for session in sessions:
        if session in session_data:
            ob = session_data[session]['ob'].mean(axis=(0, 1))
            freqs = np.fft.rfftfreq(len(ob), 1/fs)
            psd = np.abs(np.fft.rfft(ob))**2

            gamma_mask = (freqs >= 30) & (freqs < 100)
            theta_mask = (freqs >= 4) & (freqs < 8)
            ratio = np.mean(psd[gamma_mask]) / (np.mean(psd[theta_mask]) + 1e-10)
            gamma_theta_ratios.append(ratio)
        else:
            gamma_theta_ratios.append(0)

    colors = [cmap(norm(performance[s])) for s in sessions]
    ax4.bar(range(len(sessions)), gamma_theta_ratios, color=colors, edgecolor='black')
    ax4.set_xticks(range(len(sessions)))
    ax4.set_xticklabels(sessions, rotation=45, ha='right')
    ax4.set_ylabel('Gamma/Theta Ratio')
    ax4.set_title('Gamma/Theta Power Ratio (color=R²)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved power spectrum comparison to {output_path}")


def create_amplitude_distributions(session_data: Dict, performance: Dict, output_path: Path):
    """Create signal amplitude distribution comparison.

    Args:
        session_data: Dict of session -> raw signal data
        performance: Dict of session -> R² performance
        output_path: Path to save figure
    """
    sessions = sorted(performance.keys(), key=lambda s: performance[s], reverse=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=min(performance.values()), vmax=max(performance.values()))

    # Plot 1: OB amplitude histograms
    ax1 = axes[0, 0]
    for session in sessions:
        if session not in session_data:
            continue
        ob = session_data[session]['ob'].flatten()
        color = cmap(norm(performance[session]))
        ax1.hist(ob, bins=100, alpha=0.5, label=f"{session}", color=color, density=True)

    ax1.set_xlabel('Amplitude')
    ax1.set_ylabel('Density')
    ax1.set_title('OB Signal Amplitude Distributions')
    ax1.legend(fontsize=7)
    ax1.set_xlim(-5, 5)

    # Plot 2: PCx amplitude histograms
    ax2 = axes[0, 1]
    for session in sessions:
        if session not in session_data:
            continue
        pcx = session_data[session]['pcx'].flatten()
        color = cmap(norm(performance[session]))
        ax2.hist(pcx, bins=100, alpha=0.5, label=f"{session}", color=color, density=True)

    ax2.set_xlabel('Amplitude')
    ax2.set_ylabel('Density')
    ax2.set_title('PCx Signal Amplitude Distributions')
    ax2.set_xlim(-5, 5)

    # Plot 3: Per-channel variance comparison
    ax3 = axes[1, 0]
    x = np.arange(len(sessions))
    width = 0.35

    ob_vars = []
    pcx_vars = []
    for session in sessions:
        if session in session_data:
            ob = session_data[session]['ob']  # [N, C, T]
            pcx = session_data[session]['pcx']
            # Variance per channel, then mean across channels
            ob_vars.append(np.mean(np.var(ob, axis=(0, 2))))
            pcx_vars.append(np.mean(np.var(pcx, axis=(0, 2))))
        else:
            ob_vars.append(0)
            pcx_vars.append(0)

    ax3.bar(x - width/2, ob_vars, width, label='OB', color='steelblue')
    ax3.bar(x + width/2, pcx_vars, width, label='PCx', color='coral')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sessions, rotation=45, ha='right')
    ax3.set_ylabel('Mean Channel Variance')
    ax3.set_title('Signal Variance by Session')
    ax3.legend()

    # Plot 4: Dynamic range (max - min)
    ax4 = axes[1, 1]
    ob_ranges = []
    pcx_ranges = []
    for session in sessions:
        if session in session_data:
            ob = session_data[session]['ob']
            pcx = session_data[session]['pcx']
            ob_ranges.append(np.max(ob) - np.min(ob))
            pcx_ranges.append(np.max(pcx) - np.min(pcx))
        else:
            ob_ranges.append(0)
            pcx_ranges.append(0)

    ax4.bar(x - width/2, ob_ranges, width, label='OB', color='steelblue')
    ax4.bar(x + width/2, pcx_ranges, width, label='PCx', color='coral')
    ax4.set_xticks(x)
    ax4.set_xticklabels(sessions, rotation=45, ha='right')
    ax4.set_ylabel('Dynamic Range')
    ax4.set_title('Signal Dynamic Range by Session')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved amplitude distributions to {output_path}")


def create_snr_analysis(session_data: Dict, performance: Dict, output_path: Path):
    """Create detailed SNR analysis.

    Args:
        session_data: Dict of session -> raw signal data
        performance: Dict of session -> R² performance
        output_path: Path to save figure
    """
    sessions = sorted(performance.keys(), key=lambda s: performance[s], reverse=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute multiple SNR estimates
    snr_diff = {'ob': [], 'pcx': []}  # Based on diff (high-freq noise proxy)
    snr_var = {'ob': [], 'pcx': []}   # Based on variance ratio
    snr_freq = {'ob': [], 'pcx': []}  # Based on frequency content

    for session in sessions:
        if session not in session_data:
            for key in ['ob', 'pcx']:
                snr_diff[key].append(0)
                snr_var[key].append(0)
                snr_freq[key].append(0)
            continue

        ob = session_data[session]['ob']
        pcx = session_data[session]['pcx']

        for key, data in [('ob', ob), ('pcx', pcx)]:
            flat = data.reshape(-1, data.shape[-1])

            # Method 1: Diff-based (high-freq noise)
            noise_var = np.var(np.diff(flat, axis=-1))
            signal_var = np.var(flat)
            snr_diff[key].append(signal_var / (noise_var + 1e-10))

            # Method 2: Low-freq / high-freq ratio
            avg = data.mean(axis=(0, 1))
            freqs = np.fft.rfftfreq(len(avg), 1/1000)
            psd = np.abs(np.fft.rfft(avg))**2
            low_power = np.mean(psd[(freqs >= 1) & (freqs < 50)])
            high_power = np.mean(psd[(freqs >= 50) & (freqs < 200)])
            snr_freq[key].append(low_power / (high_power + 1e-10))

            # Method 3: Inter-trial consistency (signal should be consistent)
            trial_means = data.mean(axis=-1)  # [N, C]
            snr_var[key].append(np.mean(trial_means) / (np.std(trial_means) + 1e-10))

    # Plot 1: Diff-based SNR
    ax1 = axes[0, 0]
    x = np.arange(len(sessions))
    width = 0.35
    ax1.bar(x - width/2, snr_diff['ob'], width, label='OB', color='steelblue')
    ax1.bar(x + width/2, snr_diff['pcx'], width, label='PCx', color='coral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sessions, rotation=45, ha='right')
    ax1.set_ylabel('SNR (diff-based)')
    ax1.set_title('SNR: Signal Var / High-Freq Noise')
    ax1.legend()

    # Plot 2: Frequency-based SNR
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, snr_freq['ob'], width, label='OB', color='steelblue')
    ax2.bar(x + width/2, snr_freq['pcx'], width, label='PCx', color='coral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sessions, rotation=45, ha='right')
    ax2.set_ylabel('SNR (freq-based)')
    ax2.set_title('SNR: Low-Freq / High-Freq Power')
    ax2.legend()

    # Plot 3: SNR vs R² correlation
    ax3 = axes[1, 0]
    r2_vals = [performance[s] for s in sessions]

    ax3.scatter(snr_diff['ob'], r2_vals, c='steelblue', s=100, label='OB SNR', alpha=0.7)
    ax3.scatter(snr_diff['pcx'], r2_vals, c='coral', s=100, label='PCx SNR', alpha=0.7)

    for i, s in enumerate(sessions):
        ax3.annotate(s, (snr_diff['ob'][i], r2_vals[i]), fontsize=7)

    # Fit line for OB
    slope, intercept, r, p, se = stats.linregress(snr_diff['ob'], r2_vals)
    ax3.set_xlabel('SNR (diff-based)')
    ax3.set_ylabel('R² Score')
    ax3.set_title(f'SNR vs Performance (OB r={r:.2f}, p={p:.3f})')
    ax3.legend()

    # Plot 4: Combined SNR score
    ax4 = axes[1, 1]
    # Normalize each SNR metric and combine
    snr_combined = []
    for i in range(len(sessions)):
        combined = (snr_diff['ob'][i] / (max(snr_diff['ob']) + 1e-10) +
                    snr_diff['pcx'][i] / (max(snr_diff['pcx']) + 1e-10) +
                    snr_freq['ob'][i] / (max(snr_freq['ob']) + 1e-10) +
                    snr_freq['pcx'][i] / (max(snr_freq['pcx']) + 1e-10)) / 4
        snr_combined.append(combined)

    colors = plt.cm.RdYlGn([performance[s] / max(performance.values()) for s in sessions])
    ax4.bar(range(len(sessions)), snr_combined, color=colors, edgecolor='black')
    ax4.set_xticks(range(len(sessions)))
    ax4.set_xticklabels(sessions, rotation=45, ha='right')
    ax4.set_ylabel('Combined SNR Score')
    ax4.set_title('Combined SNR Score (color=R²)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved SNR analysis to {output_path}")


def main():
    """Main analysis pipeline."""
    output_dir = Path("results/session_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-session R² from your dry run results
    # UPDATE THESE with actual values from your experiments
    performance = {
        '141208-1': 0.144,  # From epoch 5 output
        '141208-2': None,   # Training session
        '141209': None,     # Training session
        '160819': None,     # Training session
        '160820': None,     # Training session
        '170608': None,     # Training session
        '170609': 0.226,    # From epoch 5 output
        '170614': 0.432,    # From epoch 5 output
        '170618': None,     # Training session
        '170619': 0.437,    # From epoch 5 output
        '170621': None,     # Training session
        '170622': None,     # Training session
    }

    # Filter to only validation sessions
    performance = {k: v for k, v in performance.items() if v is not None}

    print("="*70)
    print("SESSION OUTLIER ANALYSIS FOR PHASE 3")
    print("="*70)

    # Load data and compute statistics
    session_data, session_stats = load_session_data()

    # Filter to sessions we have performance for
    session_stats = {k: v for k, v in session_stats.items() if k in performance}

    # Identify outliers
    r2_values = list(performance.values())
    r2_mean = np.mean(r2_values)
    r2_std = np.std(r2_values)
    outlier_threshold = r2_mean - 1.0 * r2_std

    outlier_sessions = [s for s, r2 in performance.items() if r2 < outlier_threshold]
    print(f"\nOutlier threshold: R² < {outlier_threshold:.3f}")
    print(f"Outlier sessions: {outlier_sessions}")

    # Create visualizations
    print("\nGenerating visualizations...")

    create_pca_plot(session_stats, performance, output_dir / "session_pca.png")
    create_performance_breakdown(session_stats, performance, output_dir / "session_performance.png")

    # NEW: Additional visualizations
    create_power_spectrum_comparison(session_data, performance, output_dir / "power_spectra.png")
    create_amplitude_distributions(session_data, performance, output_dir / "amplitude_distributions.png")
    create_snr_analysis(session_data, performance, output_dir / "snr_analysis.png")

    # Detailed analysis
    analyze_outlier_differences(session_stats, performance, outlier_sessions)

    # Save statistics to JSON
    stats_output = output_dir / "session_statistics.json"
    with open(stats_output, 'w') as f:
        json.dump(session_stats, f, indent=2)
    print(f"\nSaved session statistics to {stats_output}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print("Files generated:")
    print("  - session_pca.png: PCA clustering of sessions")
    print("  - session_performance.png: Performance breakdown")
    print("  - power_spectra.png: Power spectrum comparison")
    print("  - amplitude_distributions.png: Signal amplitude distributions")
    print("  - snr_analysis.png: SNR estimates and correlation")
    print("  - session_statistics.json: Raw statistics")


if __name__ == "__main__":
    main()
