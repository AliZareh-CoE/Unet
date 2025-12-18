#!/usr/bin/env python3
"""Session Analysis Script - Visualize cross-session variability for augmentation planning.

This script generates 20 figures analyzing session differences:

=== Structural Analysis (Figures 1-10) ===
1. PCA by Session (train vs held-out)
2. PCA by Odor (within sessions)
3. t-SNE by Session
4. Session Covariance Matrices
5. Per-Session Power Spectral Density (PSD)
6. Session Similarity Matrix (correlation)
7. Within vs Between Session Distances
8. Odor Separability per Session
9. Channel Statistics by Session
10. Session-Odor Interaction (combined view)

=== Probability Distribution Analysis (Figures 11-20) ===
11. Envelope Magnitude Distribution (Hilbert transform)
12. Instantaneous Frequency Distribution (Hilbert transform)
13. Phase Distribution (Hilbert transform)
14. Raw Signal Amplitude Distribution
15. Peak Amplitude Distribution
16. Zero-Crossing Rate Distribution
17. Signal Gradient Distribution
18. RMS Energy Distribution
19. Kurtosis Distribution by Session
20. Skewness Distribution by Session

IMPORTANT: No data leakage - PCA/scalers fit ONLY on training data!

Usage:
    python analyze_sessions.py --output-dir figures/session_analysis
    python analyze_sessions.py --n-val-sessions 4 --output-dir figures/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import welch, hilbert, find_peaks
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed

# Use all available cores
N_JOBS = mp.cpu_count()
print(f"Using {N_JOBS} CPU cores for parallel processing")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# ==============================================================================
# PARALLEL PROCESSING HELPERS
# ==============================================================================

def _compute_envelope_for_trial(trial):
    """Compute envelope for all channels in a trial."""
    envelopes = []
    for ch in range(trial.shape[0]):
        analytic = hilbert(trial[ch])
        envelopes.append(np.abs(analytic))
    return np.concatenate(envelopes)


def _compute_inst_freq_for_trial(trial, fs):
    """Compute instantaneous frequency for all channels in a trial."""
    inst_freqs = []
    for ch in range(trial.shape[0]):
        analytic = hilbert(trial[ch])
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) * fs / (2 * np.pi)
        inst_freq = inst_freq[(inst_freq > 0) & (inst_freq < fs / 2)]
        inst_freqs.append(inst_freq)
    return np.concatenate(inst_freqs) if inst_freqs else np.array([])


def _compute_phase_for_trial(trial):
    """Compute phase for all channels in a trial."""
    phases = []
    for ch in range(trial.shape[0]):
        analytic = hilbert(trial[ch])
        phases.append(np.angle(analytic))
    return np.concatenate(phases)


def _compute_peaks_for_trial(trial):
    """Compute peaks for all channels in a trial."""
    peaks = []
    for ch in range(trial.shape[0]):
        peak_idx, _ = find_peaks(trial[ch], distance=5)
        if len(peak_idx) > 0:
            peaks.append(trial[ch][peak_idx])
    return np.concatenate(peaks) if peaks else np.array([])


def _compute_zcr_for_trial(trial):
    """Compute zero-crossing rate for all channels in a trial."""
    zcrs = []
    for ch in range(trial.shape[0]):
        zcr = np.sum(np.abs(np.diff(np.sign(trial[ch]))) > 0) / len(trial[ch])
        zcrs.append(zcr)
    return np.array(zcrs)


def _compute_gradient_for_trial(trial, fs):
    """Compute gradient for all channels in a trial."""
    grads = []
    for ch in range(trial.shape[0]):
        grads.append(np.diff(trial[ch]) * fs)
    return np.concatenate(grads)


def _compute_kurtosis_for_trial(trial):
    """Compute kurtosis for all channels in a trial."""
    kurts = []
    for ch in range(trial.shape[0]):
        k = stats.kurtosis(trial[ch])
        if not np.isnan(k) and not np.isinf(k):
            kurts.append(k)
    return np.array(kurts)


def _compute_skewness_for_trial(trial):
    """Compute skewness for all channels in a trial."""
    skews = []
    for ch in range(trial.shape[0]):
        s = stats.skew(trial[ch])
        if not np.isnan(s) and not np.isinf(s):
            skews.append(s)
    return np.array(skews)

from data import (
    prepare_data,
    load_session_ids,
    SAMPLING_RATE_HZ,
    DATA_PATH,
    ODOR_CSV_PATH,
)


def set_style():
    """Set publication-quality plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


def flatten_trials(data: np.ndarray) -> np.ndarray:
    """Flatten trials from (trials, channels, time) to (trials, features)."""
    n_trials = data.shape[0]
    return data.reshape(n_trials, -1)


def compute_trial_features(data: np.ndarray) -> np.ndarray:
    """Extract meaningful features from each trial for dimensionality reduction.

    Instead of using raw flattened data, extract:
    - Mean per channel
    - Std per channel
    - Power in frequency bands
    """
    n_trials, n_channels, n_time = data.shape

    features = []
    for i in range(n_trials):
        trial = data[i]

        # Basic statistics per channel
        mean = trial.mean(axis=1)  # (channels,)
        std = trial.std(axis=1)    # (channels,)

        # Power in frequency bands (delta, theta, alpha, beta, gamma)
        freqs, psd = welch(trial, fs=SAMPLING_RATE_HZ, nperseg=min(256, n_time), axis=1)

        # Band powers
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
        }

        band_powers = []
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if band_mask.any():
                band_power = psd[:, band_mask].mean(axis=1)  # (channels,)
                band_powers.append(band_power)

        # Concatenate all features
        trial_features = np.concatenate([mean, std] + band_powers)
        features.append(trial_features)

    return np.array(features)


def figure_1_pca_by_session(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 1: PCA colored by session, showing train vs held-out split."""
    print("Creating Figure 1: PCA by Session...")

    # Extract features
    features = compute_trial_features(ob)

    # Fit PCA ONLY on training data (no leakage!)
    scaler = StandardScaler()
    pca = PCA(n_components=2)

    train_features = features[train_idx]
    scaler.fit(train_features)
    pca.fit(scaler.transform(train_features))

    # Transform ALL data using training-fitted PCA
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    # Get unique sessions
    unique_sessions = np.unique(session_ids)
    n_sessions = len(unique_sessions)

    # Determine which sessions are train vs held-out
    train_sessions = set(session_ids[train_idx])
    val_sessions = set(session_ids[val_idx])
    test_sessions = set(session_ids[test_idx]) if len(test_idx) > 0 else set()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, n_sessions))
    session_colors = {sess: colors[i] for i, sess in enumerate(unique_sessions)}

    # Left plot: All sessions
    ax = axes[0]
    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_name = idx_to_session[sess_id]

        # Determine marker based on split
        if sess_id in train_sessions:
            marker = 'o'
            label = f"{sess_name} (train)"
        elif sess_id in val_sessions:
            marker = 's'
            label = f"{sess_name} (val)"
        else:
            marker = '^'
            label = f"{sess_name} (test)"

        ax.scatter(
            features_pca[mask, 0],
            features_pca[mask, 1],
            c=[session_colors[sess_id]],
            marker=marker,
            alpha=0.6,
            s=30,
            label=label,
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA of OB Signals by Session\n(○=train, □=val, △=test)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    # Right plot: Session centroids with confidence ellipses
    ax = axes[1]
    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_name = idx_to_session[sess_id]

        sess_pca = features_pca[mask]
        centroid = sess_pca.mean(axis=0)

        # Determine style
        if sess_id in train_sessions:
            marker = 'o'
            edgecolor = 'black'
        else:
            marker = 's'
            edgecolor = 'red'

        ax.scatter(
            centroid[0], centroid[1],
            c=[session_colors[sess_id]],
            marker=marker,
            s=200,
            edgecolors=edgecolor,
            linewidths=2,
            label=sess_name,
        )

        # Add 95% confidence ellipse
        if len(sess_pca) > 2:
            cov = np.cov(sess_pca.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
            width, height = 2 * np.sqrt(eigenvalues * 5.991)  # 95% confidence

            ellipse = matplotlib.patches.Ellipse(
                centroid, width, height, angle=angle,
                fill=False, edgecolor=session_colors[sess_id],
                linestyle='--' if sess_id not in train_sessions else '-',
                linewidth=1.5,
            )
            ax.add_patch(ellipse)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Session Centroids with 95% Confidence Ellipses\n(black edge=train, red edge=held-out)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_pca_by_session.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig1_pca_by_session.png'}")


def figure_2_pca_by_odor(
    ob: np.ndarray,
    odors: np.ndarray,
    vocab: Dict[str, int],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 2: PCA colored by odor class."""
    print("Creating Figure 2: PCA by Odor...")

    # Extract features
    features = compute_trial_features(ob)

    # Fit on training only
    scaler = StandardScaler()
    pca = PCA(n_components=2)

    train_features = features[train_idx]
    scaler.fit(train_features)
    pca.fit(scaler.transform(train_features))

    # Transform all
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    # Reverse vocab for labels
    id_to_odor = {v: k for k, v in vocab.items()}
    unique_odors = np.unique(odors)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_odors)))

    for i, odor_id in enumerate(unique_odors):
        mask = odors == odor_id
        odor_name = id_to_odor.get(odor_id, f"Odor {odor_id}")

        ax.scatter(
            features_pca[mask, 0],
            features_pca[mask, 1],
            c=[colors[i]],
            alpha=0.6,
            s=30,
            label=odor_name,
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA of OB Signals by Odor Class')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_pca_by_odor.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig2_pca_by_odor.png'}")


def figure_3_tsne_by_session(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
    n_samples: int = 500,
):
    """Figure 3: t-SNE visualization by session."""
    print("Creating Figure 3: t-SNE by Session...")

    # Subsample for t-SNE (it's slow)
    n_total = len(ob)
    if n_total > n_samples:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_total, n_samples, replace=False)
    else:
        sample_idx = np.arange(n_total)

    # Extract features
    features = compute_trial_features(ob[sample_idx])
    sampled_sessions = session_ids[sample_idx]

    # Determine which are training samples
    train_set = set(train_idx)
    is_train = np.array([idx in train_set for idx in sample_idx])

    # Fit t-SNE
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    features_tsne = tsne.fit_transform(features_scaled)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_sessions = np.unique(sampled_sessions)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sessions)))
    session_colors = {sess: colors[i] for i, sess in enumerate(unique_sessions)}

    for sess_id in unique_sessions:
        mask = sampled_sessions == sess_id
        sess_name = idx_to_session[sess_id]

        # Split into train and held-out
        train_mask = mask & is_train
        holdout_mask = mask & ~is_train

        if train_mask.any():
            ax.scatter(
                features_tsne[train_mask, 0],
                features_tsne[train_mask, 1],
                c=[session_colors[sess_id]],
                marker='o',
                alpha=0.6,
                s=30,
                label=f"{sess_name} (train)",
            )

        if holdout_mask.any():
            ax.scatter(
                features_tsne[holdout_mask, 0],
                features_tsne[holdout_mask, 1],
                c=[session_colors[sess_id]],
                marker='s',
                alpha=0.6,
                s=30,
                label=f"{sess_name} (held-out)",
            )

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f't-SNE of OB Signals by Session (n={len(sample_idx)})')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_tsne_by_session.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig3_tsne_by_session.png'}")


def figure_4_session_covariances(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 4: Covariance matrices per session."""
    print("Creating Figure 4: Session Covariance Matrices...")

    unique_sessions = np.unique(session_ids)
    n_sessions = len(unique_sessions)

    # Compute covariance for each session
    session_covs = {}
    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = ob[mask]  # (trials, channels, time)

        # Concatenate all trials
        n_trials, n_channels, n_time = sess_data.shape
        data_concat = sess_data.transpose(1, 0, 2).reshape(n_channels, -1)

        # Compute covariance
        cov = np.cov(data_concat)
        session_covs[sess_id] = cov

    # Determine grid size
    n_cols = min(4, n_sessions)
    n_rows = (n_sessions + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes).flatten()

    # Determine which sessions are training
    train_sessions = set(session_ids[train_idx])

    for i, sess_id in enumerate(unique_sessions):
        ax = axes[i]
        cov = session_covs[sess_id]
        sess_name = idx_to_session[sess_id]

        # Normalize for visualization
        vmax = np.percentile(np.abs(cov), 99)

        im = ax.imshow(cov, cmap='coolwarm', vmin=-vmax, vmax=vmax)

        title_suffix = " (train)" if sess_id in train_sessions else " (held-out)"
        ax.set_title(f'{sess_name}{title_suffix}', fontsize=10)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Channel')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide unused subplots
    for i in range(n_sessions, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Channel Covariance Matrices by Session', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_session_covariances.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig4_session_covariances.png'}")


def figure_5_session_psd(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 5: Power Spectral Density per session."""
    print("Creating Figure 5: Per-Session PSD...")

    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sessions)))

    # Left: Individual session PSDs
    ax = axes[0]
    for i, sess_id in enumerate(unique_sessions):
        mask = session_ids == sess_id
        sess_data = ob[mask]  # (trials, channels, time)
        sess_name = idx_to_session[sess_id]

        # Average across trials and channels
        mean_signal = sess_data.mean(axis=(0, 1))  # (time,)

        # Compute PSD
        freqs, psd = welch(sess_data.reshape(-1, sess_data.shape[-1]).mean(axis=0),
                          fs=SAMPLING_RATE_HZ, nperseg=256)

        linestyle = '-' if sess_id in train_sessions else '--'
        ax.semilogy(freqs, psd, color=colors[i], linestyle=linestyle,
                   label=sess_name, alpha=0.8)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('PSD by Session\n(solid=train, dashed=held-out)')
    ax.set_xlim(0, 100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: PSD variance across sessions
    ax = axes[1]

    # Collect PSDs
    all_psds = []
    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = ob[mask]

        # Per-channel PSD, then average
        psds = []
        for trial in sess_data:
            for ch in range(trial.shape[0]):
                freqs, psd = welch(trial[ch], fs=SAMPLING_RATE_HZ, nperseg=256)
                psds.append(psd)
        all_psds.append(np.mean(psds, axis=0))

    all_psds = np.array(all_psds)
    mean_psd = all_psds.mean(axis=0)
    std_psd = all_psds.std(axis=0)

    ax.semilogy(freqs, mean_psd, 'b-', linewidth=2, label='Mean')
    ax.fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd,
                   alpha=0.3, color='blue', label='±1 std')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Cross-Session PSD Variability')
    ax.set_xlim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_session_psd.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig5_session_psd.png'}")


def figure_6_session_similarity(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 6: Session similarity matrix based on distribution distances."""
    print("Creating Figure 6: Session Similarity Matrix...")

    unique_sessions = np.unique(session_ids)
    n_sessions = len(unique_sessions)
    train_sessions = set(session_ids[train_idx])

    # Compute session statistics
    session_stats = {}
    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = ob[mask]

        # Mean and std per channel
        mean = sess_data.mean(axis=(0, 2))  # (channels,)
        std = sess_data.std(axis=(0, 2))    # (channels,)

        session_stats[sess_id] = np.concatenate([mean, std])

    # Compute pairwise distances
    stat_matrix = np.array([session_stats[s] for s in unique_sessions])
    distances = squareform(pdist(stat_matrix, metric='euclidean'))

    # Convert to similarity
    similarity = 1 / (1 + distances)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Session names
    session_names = [idx_to_session[s] for s in unique_sessions]

    # Add markers for train vs held-out
    session_labels = []
    for s in unique_sessions:
        name = idx_to_session[s]
        if s in train_sessions:
            session_labels.append(f"{name}")
        else:
            session_labels.append(f"{name}*")

    im = ax.imshow(similarity, cmap='viridis')

    ax.set_xticks(range(n_sessions))
    ax.set_yticks(range(n_sessions))
    ax.set_xticklabels(session_labels, rotation=45, ha='right')
    ax.set_yticklabels(session_labels)

    # Add values
    for i in range(n_sessions):
        for j in range(n_sessions):
            ax.text(j, i, f'{similarity[i, j]:.2f}',
                   ha='center', va='center', fontsize=8,
                   color='white' if similarity[i, j] < 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='Similarity')
    ax.set_title('Session Similarity Matrix\n(* = held-out session)')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_session_similarity.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig6_session_similarity.png'}")


def figure_7_within_between_distances(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    output_dir: Path,
    n_samples: int = 100,
):
    """Figure 7: Within-session vs between-session distances."""
    print("Creating Figure 7: Within vs Between Session Distances...")

    # Extract features
    features = compute_trial_features(ob)

    unique_sessions = np.unique(session_ids)

    within_distances = []
    between_distances = []

    rng = np.random.RandomState(42)

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_features = features[mask]

        # Sample within-session distances
        n_sess = len(sess_features)
        if n_sess > 1:
            for _ in range(min(n_samples, n_sess * (n_sess - 1) // 2)):
                i, j = rng.choice(n_sess, 2, replace=False)
                dist = np.linalg.norm(sess_features[i] - sess_features[j])
                within_distances.append(dist)

        # Sample between-session distances
        other_mask = session_ids != sess_id
        other_features = features[other_mask]

        if len(other_features) > 0:
            for _ in range(n_samples):
                i = rng.randint(len(sess_features))
                j = rng.randint(len(other_features))
                dist = np.linalg.norm(sess_features[i] - other_features[j])
                between_distances.append(dist)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Histograms
    ax = axes[0]
    ax.hist(within_distances, bins=50, alpha=0.6, label='Within-session', density=True)
    ax.hist(between_distances, bins=50, alpha=0.6, label='Between-session', density=True)
    ax.set_xlabel('Euclidean Distance')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Pairwise Distances')
    ax.legend()
    ax.axvline(np.mean(within_distances), color='C0', linestyle='--',
               label=f'Within mean: {np.mean(within_distances):.2f}')
    ax.axvline(np.mean(between_distances), color='C1', linestyle='--',
               label=f'Between mean: {np.mean(between_distances):.2f}')

    # Right: Box plot
    ax = axes[1]
    data = [within_distances, between_distances]
    bp = ax.boxplot(data, labels=['Within-session', 'Between-session'], patch_artist=True)
    bp['boxes'][0].set_facecolor('C0')
    bp['boxes'][1].set_facecolor('C1')
    ax.set_ylabel('Euclidean Distance')
    ax.set_title('Within vs Between Session Distances')

    # Statistical test
    stat, pval = stats.mannwhitneyu(within_distances, between_distances)
    ax.text(0.5, 0.95, f'Mann-Whitney U p-value: {pval:.2e}',
           transform=ax.transAxes, ha='center', fontsize=10)

    # Ratio
    ratio = np.mean(between_distances) / np.mean(within_distances)
    ax.text(0.5, 0.88, f'Between/Within ratio: {ratio:.2f}x',
           transform=ax.transAxes, ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_within_between_distances.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig7_within_between_distances.png'}")


def figure_8_odor_separability_per_session(
    ob: np.ndarray,
    odors: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    vocab: Dict[str, int],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 8: Odor class separability within each session."""
    print("Creating Figure 8: Odor Separability per Session...")

    # Extract features
    features = compute_trial_features(ob)

    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])

    # Compute silhouette-like score per session
    from sklearn.metrics import silhouette_score

    session_scores = {}
    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_features = features[mask]
        sess_odors = odors[mask]

        # Need at least 2 classes with 2+ samples
        unique_odors = np.unique(sess_odors)
        if len(unique_odors) >= 2 and len(sess_features) >= 4:
            try:
                score = silhouette_score(sess_features, sess_odors)
                session_scores[sess_id] = score
            except:
                session_scores[sess_id] = np.nan
        else:
            session_scores[sess_id] = np.nan

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    session_names = [idx_to_session[s] for s in unique_sessions]
    scores = [session_scores[s] for s in unique_sessions]
    colors = ['C0' if s in train_sessions else 'C1' for s in unique_sessions]

    bars = ax.bar(range(len(unique_sessions)), scores, color=colors)

    ax.set_xticks(range(len(unique_sessions)))
    ax.set_xticklabels(session_names, rotation=45, ha='right')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Odor Class Separability per Session\n(higher = better separability)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='C0', label='Train sessions'),
        Patch(facecolor='C1', label='Held-out sessions'),
    ]
    ax.legend(handles=legend_elements)

    # Add mean line
    valid_scores = [s for s in scores if not np.isnan(s)]
    if valid_scores:
        ax.axhline(np.mean(valid_scores), color='red', linestyle='-',
                  label=f'Mean: {np.mean(valid_scores):.3f}')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig8_odor_separability_per_session.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig8_odor_separability_per_session.png'}")


def figure_9_channel_statistics(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 9: Channel-wise statistics per session."""
    print("Creating Figure 9: Channel Statistics by Session...")

    unique_sessions = np.unique(session_ids)
    n_sessions = len(unique_sessions)
    n_channels = ob.shape[1]
    train_sessions = set(session_ids[train_idx])

    # Compute per-session, per-channel mean and std
    session_means = np.zeros((n_sessions, n_channels))
    session_stds = np.zeros((n_sessions, n_channels))

    for i, sess_id in enumerate(unique_sessions):
        mask = session_ids == sess_id
        sess_data = ob[mask]
        session_means[i] = sess_data.mean(axis=(0, 2))
        session_stds[i] = sess_data.std(axis=(0, 2))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Mean per channel (line plot)
    ax = axes[0, 0]
    for i, sess_id in enumerate(unique_sessions):
        sess_name = idx_to_session[sess_id]
        linestyle = '-' if sess_id in train_sessions else '--'
        ax.plot(session_means[i], linestyle=linestyle, alpha=0.7, label=sess_name)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Mean Amplitude')
    ax.set_title('Per-Channel Mean by Session\n(solid=train, dashed=held-out)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)

    # Top-right: Std per channel (line plot)
    ax = axes[0, 1]
    for i, sess_id in enumerate(unique_sessions):
        sess_name = idx_to_session[sess_id]
        linestyle = '-' if sess_id in train_sessions else '--'
        ax.plot(session_stds[i], linestyle=linestyle, alpha=0.7, label=sess_name)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Per-Channel Std by Session')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)

    # Bottom-left: Heatmap of means
    ax = axes[1, 0]
    session_labels = [f"{idx_to_session[s]}{'*' if s not in train_sessions else ''}"
                     for s in unique_sessions]
    im = ax.imshow(session_means, aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Session')
    ax.set_yticks(range(n_sessions))
    ax.set_yticklabels(session_labels)
    ax.set_title('Channel Means Heatmap (* = held-out)')
    plt.colorbar(im, ax=ax)

    # Bottom-right: Channel-wise variance across sessions
    ax = axes[1, 1]
    channel_variance = session_means.var(axis=0)
    ax.bar(range(n_channels), channel_variance)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Variance Across Sessions')
    ax.set_title('Channel-wise Cross-Session Variance\n(high = inconsistent across sessions)')

    # Highlight high-variance channels
    threshold = np.percentile(channel_variance, 90)
    high_var_channels = np.where(channel_variance > threshold)[0]
    for ch in high_var_channels:
        ax.axvline(ch, color='red', alpha=0.3, linewidth=2)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig9_channel_statistics.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig9_channel_statistics.png'}")


def figure_10_session_odor_interaction(
    ob: np.ndarray,
    odors: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    vocab: Dict[str, int],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 10: Session-Odor interaction visualization."""
    print("Creating Figure 10: Session-Odor Interaction...")

    # Extract features
    features = compute_trial_features(ob)

    # Fit PCA on training data only
    scaler = StandardScaler()
    pca = PCA(n_components=2)

    train_features = features[train_idx]
    scaler.fit(train_features)
    pca.fit(scaler.transform(train_features))

    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    unique_sessions = np.unique(session_ids)
    unique_odors = np.unique(odors)
    train_sessions = set(session_ids[train_idx])

    id_to_odor = {v: k for k, v in vocab.items()}

    # Create figure with subplots per session
    n_cols = min(4, len(unique_sessions))
    n_rows = (len(unique_sessions) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = np.atleast_2d(axes).flatten()

    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_odors)))

    for idx, sess_id in enumerate(unique_sessions):
        ax = axes[idx]
        sess_mask = session_ids == sess_id
        sess_name = idx_to_session[sess_id]

        for i, odor_id in enumerate(unique_odors):
            mask = sess_mask & (odors == odor_id)
            if mask.any():
                odor_name = id_to_odor.get(odor_id, f"Odor {odor_id}")
                ax.scatter(
                    features_pca[mask, 0],
                    features_pca[mask, 1],
                    c=[colors[i]],
                    alpha=0.6,
                    s=30,
                    label=odor_name,
                )

        title_suffix = " (train)" if sess_id in train_sessions else " (held-out)"
        ax.set_title(f'{sess_name}{title_suffix}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        if idx == 0:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    # Hide unused subplots
    for idx in range(len(unique_sessions), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('PCA by Odor within Each Session', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig10_session_odor_interaction.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig10_session_odor_interaction.png'}")


# ==============================================================================
# PROBABILITY DISTRIBUTION FIGURES (11-20)
# ==============================================================================

def figure_11_envelope_magnitude_distribution(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 11: Envelope magnitude distribution using Hilbert transform."""
    print("Creating Figure 11: Envelope Magnitude Distribution...")

    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect envelope magnitudes per session - PARALLEL
    session_envelopes = {}
    all_envelopes_train = []
    all_envelopes_holdout = []

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = ob[mask]

        # Parallel computation across all trials - NO SUBSAMPLING
        envelopes_list = Parallel(n_jobs=N_JOBS, backend='threading')(
            delayed(_compute_envelope_for_trial)(trial) for trial in sess_data
        )
        envelopes = np.concatenate(envelopes_list) if envelopes_list else np.array([])

        session_envelopes[sess_id] = envelopes

        if sess_id in train_sessions:
            all_envelopes_train.append(envelopes)
        else:
            all_envelopes_holdout.append(envelopes)

    all_envelopes_train = np.concatenate(all_envelopes_train) if all_envelopes_train else np.array([])
    all_envelopes_holdout = np.concatenate(all_envelopes_holdout) if all_envelopes_holdout else np.array([])

    # Left: Per-session envelope distribution (KDE)
    ax = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sessions)))

    for i, sess_id in enumerate(unique_sessions):
        sess_name = idx_to_session[sess_id]
        linestyle = '-' if sess_id in train_sessions else '--'
        env = session_envelopes[sess_id]

        # Use KDE for smooth distribution
        try:
            kde = stats.gaussian_kde(env)
            x_range = np.linspace(0, np.percentile(env, 99), 200)
            ax.plot(x_range, kde(x_range), color=colors[i], linestyle=linestyle,
                   label=sess_name, alpha=0.8)
        except:
            ax.hist(env, bins=50, density=True, alpha=0.3, color=colors[i], label=sess_name)

    ax.set_xlabel('Envelope Magnitude')
    ax.set_ylabel('Density')
    ax.set_title('Envelope Magnitude Distribution by Session\n(solid=train, dashed=held-out)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.set_xlim(left=0)

    # Right: Train vs Held-out comparison
    ax = axes[1]
    if all_envelopes_train and all_envelopes_holdout:
        ax.hist(all_envelopes_train, bins=100, density=True, alpha=0.5,
               label='Train sessions', color='C0')
        ax.hist(all_envelopes_holdout, bins=100, density=True, alpha=0.5,
               label='Held-out sessions', color='C1')

        # KS test
        stat, pval = stats.ks_2samp(
            np.array(all_envelopes_train)[:10000],
            np.array(all_envelopes_holdout)[:10000]
        )
        ax.text(0.95, 0.95, f'KS test p-value: {pval:.2e}',
               transform=ax.transAxes, ha='right', va='top', fontsize=10)

    ax.set_xlabel('Envelope Magnitude')
    ax.set_ylabel('Density')
    ax.set_title('Envelope Distribution: Train vs Held-out')
    ax.legend()
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig11_envelope_magnitude_distribution.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig11_envelope_magnitude_distribution.png'}")


def figure_12_instantaneous_frequency_distribution(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 12: Instantaneous frequency distribution using Hilbert transform."""
    print("Creating Figure 12: Instantaneous Frequency Distribution...")

    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    session_inst_freqs = {}
    all_freqs_train = []
    all_freqs_holdout = []

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = ob[mask]

        # Parallel computation - NO SUBSAMPLING
        inst_freqs_list = Parallel(n_jobs=N_JOBS, backend='threading')(
            delayed(_compute_inst_freq_for_trial)(trial, SAMPLING_RATE_HZ) for trial in sess_data
        )
        inst_freqs = np.concatenate([f for f in inst_freqs_list if len(f) > 0]) if inst_freqs_list else np.array([])

        session_inst_freqs[sess_id] = inst_freqs

        if sess_id in train_sessions:
            all_freqs_train.append(inst_freqs)
        else:
            all_freqs_holdout.append(inst_freqs)

    all_freqs_train = np.concatenate(all_freqs_train) if all_freqs_train else np.array([])
    all_freqs_holdout = np.concatenate(all_freqs_holdout) if all_freqs_holdout else np.array([])

    # Left: Per-session distribution
    ax = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sessions)))

    for i, sess_id in enumerate(unique_sessions):
        sess_name = idx_to_session[sess_id]
        linestyle = '-' if sess_id in train_sessions else '--'
        freqs = session_inst_freqs[sess_id]

        if len(freqs) > 0:
            try:
                kde = stats.gaussian_kde(freqs)
                x_range = np.linspace(0, min(100, np.percentile(freqs, 99)), 200)
                ax.plot(x_range, kde(x_range), color=colors[i], linestyle=linestyle,
                       label=sess_name, alpha=0.8)
            except:
                ax.hist(freqs, bins=50, density=True, alpha=0.3, color=colors[i])

    ax.set_xlabel('Instantaneous Frequency (Hz)')
    ax.set_ylabel('Density')
    ax.set_title('Instantaneous Frequency Distribution by Session\n(solid=train, dashed=held-out)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.set_xlim(0, 100)

    # Right: Train vs Held-out
    ax = axes[1]
    if all_freqs_train and all_freqs_holdout:
        bins = np.linspace(0, 100, 100)
        ax.hist(all_freqs_train, bins=bins, density=True, alpha=0.5,
               label='Train sessions', color='C0')
        ax.hist(all_freqs_holdout, bins=bins, density=True, alpha=0.5,
               label='Held-out sessions', color='C1')

        stat, pval = stats.ks_2samp(
            np.array(all_freqs_train)[:10000],
            np.array(all_freqs_holdout)[:10000]
        )
        ax.text(0.95, 0.95, f'KS test p-value: {pval:.2e}',
               transform=ax.transAxes, ha='right', va='top', fontsize=10)

    ax.set_xlabel('Instantaneous Frequency (Hz)')
    ax.set_ylabel('Density')
    ax.set_title('Instantaneous Frequency: Train vs Held-out')
    ax.legend()
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig12_instantaneous_frequency_distribution.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig12_instantaneous_frequency_distribution.png'}")


def figure_13_phase_distribution(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 13: Phase distribution using Hilbert transform."""
    print("Creating Figure 13: Phase Distribution...")

    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    session_phases = {}
    all_phases_train = []
    all_phases_holdout = []

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = ob[mask]

        # Parallel computation - NO SUBSAMPLING
        phases_list = Parallel(n_jobs=N_JOBS, backend='threading')(
            delayed(_compute_phase_for_trial)(trial) for trial in sess_data
        )
        phases = np.concatenate(phases_list) if phases_list else np.array([])

        session_phases[sess_id] = phases

        if sess_id in train_sessions:
            all_phases_train.append(phases)
        else:
            all_phases_holdout.append(phases)

    all_phases_train = np.concatenate(all_phases_train) if all_phases_train else np.array([])
    all_phases_holdout = np.concatenate(all_phases_holdout) if all_phases_holdout else np.array([])

    # Left: Polar histogram per session
    ax = axes[0]
    ax.remove()
    ax = fig.add_subplot(1, 2, 1, projection='polar')

    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sessions)))

    for i, sess_id in enumerate(unique_sessions):
        sess_name = idx_to_session[sess_id]
        linestyle = '-' if sess_id in train_sessions else '--'
        phases = session_phases[sess_id]

        if len(phases) > 0:
            # Histogram in polar coordinates
            bins = np.linspace(-np.pi, np.pi, 37)  # 36 bins
            hist, bin_edges = np.histogram(phases, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax.plot(bin_centers, hist, color=colors[i], linestyle=linestyle,
                   label=sess_name, alpha=0.7)

    ax.set_title('Phase Distribution by Session\n(polar histogram)')
    ax.legend(bbox_to_anchor=(1.3, 1), loc='upper left', fontsize=7)

    # Right: Train vs Held-out (Cartesian)
    ax = axes[1]
    if all_phases_train and all_phases_holdout:
        bins = np.linspace(-np.pi, np.pi, 100)
        ax.hist(all_phases_train, bins=bins, density=True, alpha=0.5,
               label='Train sessions', color='C0')
        ax.hist(all_phases_holdout, bins=bins, density=True, alpha=0.5,
               label='Held-out sessions', color='C1')

        # Circular mean test (Rayleigh test for uniformity)
        from scipy.stats import circmean, circstd
        train_mean = circmean(all_phases_train[:10000])
        holdout_mean = circmean(all_phases_holdout[:10000])
        ax.axvline(train_mean, color='C0', linestyle='--', label=f'Train mean: {train_mean:.2f}')
        ax.axvline(holdout_mean, color='C1', linestyle='--', label=f'Held-out mean: {holdout_mean:.2f}')

    ax.set_xlabel('Phase (radians)')
    ax.set_ylabel('Density')
    ax.set_title('Phase Distribution: Train vs Held-out')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fig13_phase_distribution.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig13_phase_distribution.png'}")


def figure_14_amplitude_distribution(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 14: Raw signal amplitude distribution."""
    print("Creating Figure 14: Raw Amplitude Distribution...")

    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    session_amplitudes = {}
    all_amps_train = []
    all_amps_holdout = []

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = ob[mask]

        # NO SUBSAMPLING - use all data
        amps = sess_data.flatten()
        session_amplitudes[sess_id] = amps

        if sess_id in train_sessions:
            all_amps_train.append(amps)
        else:
            all_amps_holdout.append(amps)

    all_amps_train = np.concatenate(all_amps_train) if all_amps_train else np.array([])
    all_amps_holdout = np.concatenate(all_amps_holdout) if all_amps_holdout else np.array([])

    # Left: Per-session distribution
    ax = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sessions)))

    # Determine common range
    all_amps = np.concatenate(list(session_amplitudes.values()))
    p1, p99 = np.percentile(all_amps, [1, 99])

    for i, sess_id in enumerate(unique_sessions):
        sess_name = idx_to_session[sess_id]
        linestyle = '-' if sess_id in train_sessions else '--'
        amps = session_amplitudes[sess_id]

        try:
            kde = stats.gaussian_kde(amps)
            x_range = np.linspace(p1, p99, 200)
            ax.plot(x_range, kde(x_range), color=colors[i], linestyle=linestyle,
                   label=sess_name, alpha=0.8)
        except:
            ax.hist(amps, bins=50, density=True, alpha=0.3, color=colors[i])

    ax.set_xlabel('Signal Amplitude')
    ax.set_ylabel('Density')
    ax.set_title('Raw Amplitude Distribution by Session\n(solid=train, dashed=held-out)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    # Right: Train vs Held-out with statistics
    ax = axes[1]
    all_amps_train = np.array(all_amps_train)
    all_amps_holdout = np.array(all_amps_holdout)

    if len(all_amps_train) > 0 and len(all_amps_holdout) > 0:
        bins = np.linspace(p1, p99, 100)
        ax.hist(all_amps_train, bins=bins, density=True, alpha=0.5,
               label='Train sessions', color='C0')
        ax.hist(all_amps_holdout, bins=bins, density=True, alpha=0.5,
               label='Held-out sessions', color='C1')

        # Statistics
        train_mean = np.mean(all_amps_train)
        holdout_mean = np.mean(all_amps_holdout)
        train_std = np.std(all_amps_train)
        holdout_std = np.std(all_amps_holdout)

        stats_text = f'Train: μ={train_mean:.3f}, σ={train_std:.3f}\n'
        stats_text += f'Held-out: μ={holdout_mean:.3f}, σ={holdout_std:.3f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               ha='right', va='top', fontsize=10, family='monospace')

    ax.set_xlabel('Signal Amplitude')
    ax.set_ylabel('Density')
    ax.set_title('Amplitude Distribution: Train vs Held-out')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fig14_amplitude_distribution.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig14_amplitude_distribution.png'}")


def figure_15_peak_amplitude_distribution(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 15: Peak amplitude distribution (local maxima)."""
    print("Creating Figure 15: Peak Amplitude Distribution...")

    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    session_peaks = {}
    all_peaks_train = []
    all_peaks_holdout = []

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = ob[mask]

        # Parallel computation - NO SUBSAMPLING
        peaks_list = Parallel(n_jobs=N_JOBS, backend='threading')(
            delayed(_compute_peaks_for_trial)(trial) for trial in sess_data
        )
        peaks = np.concatenate([p for p in peaks_list if len(p) > 0]) if peaks_list else np.array([0])

        session_peaks[sess_id] = peaks

        if sess_id in train_sessions:
            all_peaks_train.append(peaks)
        else:
            all_peaks_holdout.append(peaks)

    all_peaks_train = np.concatenate(all_peaks_train) if all_peaks_train else np.array([])
    all_peaks_holdout = np.concatenate(all_peaks_holdout) if all_peaks_holdout else np.array([])

    # Left: Per-session peak distribution
    ax = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sessions)))

    # Determine common range
    all_peaks = np.concatenate(list(session_peaks.values()))
    p1, p99 = np.percentile(all_peaks, [5, 95])

    for i, sess_id in enumerate(unique_sessions):
        sess_name = idx_to_session[sess_id]
        linestyle = '-' if sess_id in train_sessions else '--'
        peaks = session_peaks[sess_id]

        if len(peaks) > 10:
            try:
                kde = stats.gaussian_kde(peaks)
                x_range = np.linspace(p1, p99, 200)
                ax.plot(x_range, kde(x_range), color=colors[i], linestyle=linestyle,
                       label=sess_name, alpha=0.8)
            except:
                pass

    ax.set_xlabel('Peak Amplitude')
    ax.set_ylabel('Density')
    ax.set_title('Peak Amplitude Distribution by Session\n(solid=train, dashed=held-out)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    # Right: Train vs Held-out
    ax = axes[1]
    if all_peaks_train and all_peaks_holdout:
        bins = np.linspace(p1, p99, 100)
        ax.hist(all_peaks_train, bins=bins, density=True, alpha=0.5,
               label='Train sessions', color='C0')
        ax.hist(all_peaks_holdout, bins=bins, density=True, alpha=0.5,
               label='Held-out sessions', color='C1')

        stat, pval = stats.ks_2samp(
            np.array(all_peaks_train)[:10000],
            np.array(all_peaks_holdout)[:10000]
        )
        ax.text(0.95, 0.95, f'KS test p-value: {pval:.2e}',
               transform=ax.transAxes, ha='right', va='top', fontsize=10)

    ax.set_xlabel('Peak Amplitude')
    ax.set_ylabel('Density')
    ax.set_title('Peak Amplitude: Train vs Held-out')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fig15_peak_amplitude_distribution.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig15_peak_amplitude_distribution.png'}")


def figure_16_zero_crossing_rate_distribution(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 16: Zero-crossing rate distribution per trial."""
    print("Creating Figure 16: Zero-Crossing Rate Distribution...")

    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    session_zcrs = {}
    all_zcrs_train = []
    all_zcrs_holdout = []

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = ob[mask]

        # Parallel computation
        zcrs_list = Parallel(n_jobs=N_JOBS, backend='threading')(
            delayed(_compute_zcr_for_trial)(trial) for trial in sess_data
        )
        zcrs = np.concatenate(zcrs_list) if zcrs_list else np.array([])

        session_zcrs[sess_id] = zcrs

        if sess_id in train_sessions:
            all_zcrs_train.append(zcrs)
        else:
            all_zcrs_holdout.append(zcrs)

    all_zcrs_train = np.concatenate(all_zcrs_train) if all_zcrs_train else np.array([])
    all_zcrs_holdout = np.concatenate(all_zcrs_holdout) if all_zcrs_holdout else np.array([])

    # Left: Box plot per session
    ax = axes[0]
    data = [session_zcrs[s] for s in unique_sessions]
    labels = [idx_to_session[s] for s in unique_sessions]
    colors_list = ['C0' if s in train_sessions else 'C1' for s in unique_sessions]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)

    ax.set_xlabel('Session')
    ax.set_ylabel('Zero-Crossing Rate')
    ax.set_title('Zero-Crossing Rate by Session\n(blue=train, orange=held-out)')
    ax.tick_params(axis='x', rotation=45)

    # Right: Distribution comparison
    ax = axes[1]
    if len(all_zcrs_train) > 0 and len(all_zcrs_holdout) > 0:
        ax.hist(all_zcrs_train, bins=50, density=True, alpha=0.5,
               label='Train sessions', color='C0')
        ax.hist(all_zcrs_holdout, bins=50, density=True, alpha=0.5,
               label='Held-out sessions', color='C1')

        train_mean = np.mean(all_zcrs_train)
        holdout_mean = np.mean(all_zcrs_holdout)
        ax.axvline(train_mean, color='C0', linestyle='--')
        ax.axvline(holdout_mean, color='C1', linestyle='--')

        stat, pval = stats.ks_2samp(all_zcrs_train, all_zcrs_holdout)
        ax.text(0.95, 0.95, f'KS test p-value: {pval:.2e}',
               transform=ax.transAxes, ha='right', va='top', fontsize=10)

    ax.set_xlabel('Zero-Crossing Rate')
    ax.set_ylabel('Density')
    ax.set_title('Zero-Crossing Rate: Train vs Held-out')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fig16_zero_crossing_rate_distribution.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig16_zero_crossing_rate_distribution.png'}")


def figure_17_gradient_distribution(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 17: Signal gradient (first derivative) distribution."""
    print("Creating Figure 17: Signal Gradient Distribution...")

    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    session_grads = {}
    all_grads_train = []
    all_grads_holdout = []

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = ob[mask]

        # Parallel computation - NO SUBSAMPLING
        grads_list = Parallel(n_jobs=N_JOBS, backend='threading')(
            delayed(_compute_gradient_for_trial)(trial, SAMPLING_RATE_HZ) for trial in sess_data
        )
        grads = np.concatenate(grads_list) if grads_list else np.array([])

        session_grads[sess_id] = grads

        if sess_id in train_sessions:
            all_grads_train.append(grads)
        else:
            all_grads_holdout.append(grads)

    all_grads_train = np.concatenate(all_grads_train) if all_grads_train else np.array([])
    all_grads_holdout = np.concatenate(all_grads_holdout) if all_grads_holdout else np.array([])

    # Left: Per-session gradient distribution
    ax = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sessions)))

    all_grads = np.concatenate(list(session_grads.values()))
    p1, p99 = np.percentile(all_grads, [1, 99])

    for i, sess_id in enumerate(unique_sessions):
        sess_name = idx_to_session[sess_id]
        linestyle = '-' if sess_id in train_sessions else '--'
        grads = session_grads[sess_id]

        try:
            kde = stats.gaussian_kde(grads)
            x_range = np.linspace(p1, p99, 200)
            ax.plot(x_range, kde(x_range), color=colors[i], linestyle=linestyle,
                   label=sess_name, alpha=0.8)
        except:
            pass

    ax.set_xlabel('Signal Gradient (dV/dt)')
    ax.set_ylabel('Density')
    ax.set_title('Signal Gradient Distribution by Session\n(solid=train, dashed=held-out)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    # Right: Train vs Held-out
    ax = axes[1]
    if all_grads_train and all_grads_holdout:
        bins = np.linspace(p1, p99, 100)
        ax.hist(all_grads_train, bins=bins, density=True, alpha=0.5,
               label='Train sessions', color='C0')
        ax.hist(all_grads_holdout, bins=bins, density=True, alpha=0.5,
               label='Held-out sessions', color='C1')

        # Statistics
        train_std = np.std(all_grads_train)
        holdout_std = np.std(all_grads_holdout)
        ax.text(0.95, 0.95, f'Train σ: {train_std:.2f}\nHeld-out σ: {holdout_std:.2f}',
               transform=ax.transAxes, ha='right', va='top', fontsize=10)

    ax.set_xlabel('Signal Gradient (dV/dt)')
    ax.set_ylabel('Density')
    ax.set_title('Gradient Distribution: Train vs Held-out')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fig17_gradient_distribution.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig17_gradient_distribution.png'}")


def figure_18_rms_energy_distribution(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 18: RMS energy distribution per trial and channel."""
    print("Creating Figure 18: RMS Energy Distribution...")

    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Compute RMS per trial (average across channels)
    session_rms_trial = {}
    # Compute RMS per channel per trial
    session_rms_channel = {}

    all_rms_train = []
    all_rms_holdout = []

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = ob[mask]

        # Per-trial RMS (averaged across channels)
        rms_trials = np.sqrt(np.mean(sess_data ** 2, axis=(1, 2)))
        session_rms_trial[sess_id] = rms_trials

        # Per-channel RMS for each trial
        rms_channels = np.sqrt(np.mean(sess_data ** 2, axis=2))  # (trials, channels)
        session_rms_channel[sess_id] = rms_channels.flatten()

        if sess_id in train_sessions:
            all_rms_train.extend(rms_trials)
        else:
            all_rms_holdout.extend(rms_trials)

    # Top-left: Trial RMS box plot per session
    ax = axes[0, 0]
    data = [session_rms_trial[s] for s in unique_sessions]
    labels = [idx_to_session[s] for s in unique_sessions]
    colors_list = ['C0' if s in train_sessions else 'C1' for s in unique_sessions]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)

    ax.set_xlabel('Session')
    ax.set_ylabel('RMS Energy')
    ax.set_title('Per-Trial RMS Energy by Session\n(blue=train, orange=held-out)')
    ax.tick_params(axis='x', rotation=45)

    # Top-right: Trial RMS distribution
    ax = axes[0, 1]
    if all_rms_train and all_rms_holdout:
        ax.hist(all_rms_train, bins=50, density=True, alpha=0.5,
               label='Train sessions', color='C0')
        ax.hist(all_rms_holdout, bins=50, density=True, alpha=0.5,
               label='Held-out sessions', color='C1')

        stat, pval = stats.ks_2samp(all_rms_train, all_rms_holdout)
        ax.text(0.95, 0.95, f'KS test p-value: {pval:.2e}',
               transform=ax.transAxes, ha='right', va='top', fontsize=10)

    ax.set_xlabel('RMS Energy')
    ax.set_ylabel('Density')
    ax.set_title('Trial RMS Distribution: Train vs Held-out')
    ax.legend()

    # Bottom-left: Per-channel RMS distribution per session (KDE)
    ax = axes[1, 0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sessions)))

    for i, sess_id in enumerate(unique_sessions):
        sess_name = idx_to_session[sess_id]
        linestyle = '-' if sess_id in train_sessions else '--'
        rms = session_rms_channel[sess_id]

        try:
            kde = stats.gaussian_kde(rms)
            x_range = np.linspace(0, np.percentile(rms, 99), 200)
            ax.plot(x_range, kde(x_range), color=colors[i], linestyle=linestyle,
                   label=sess_name, alpha=0.8)
        except:
            pass

    ax.set_xlabel('Channel RMS Energy')
    ax.set_ylabel('Density')
    ax.set_title('Per-Channel RMS Distribution by Session')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.set_xlim(left=0)

    # Bottom-right: RMS statistics summary
    ax = axes[1, 1]

    sess_names = [idx_to_session[s] for s in unique_sessions]
    means = [np.mean(session_rms_trial[s]) for s in unique_sessions]
    stds = [np.std(session_rms_trial[s]) for s in unique_sessions]
    colors_bars = ['C0' if s in train_sessions else 'C1' for s in unique_sessions]

    x = np.arange(len(unique_sessions))
    ax.bar(x, means, yerr=stds, color=colors_bars, alpha=0.7, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(sess_names, rotation=45, ha='right')
    ax.set_xlabel('Session')
    ax.set_ylabel('Mean RMS Energy (±std)')
    ax.set_title('Session RMS Statistics')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig18_rms_energy_distribution.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig18_rms_energy_distribution.png'}")


def figure_19_kurtosis_distribution(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 19: Kurtosis distribution by session (per-channel)."""
    print("Creating Figure 19: Kurtosis Distribution...")

    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    session_kurtosis = {}
    all_kurt_train = []
    all_kurt_holdout = []

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = ob[mask]

        # Parallel computation
        kurts_list = Parallel(n_jobs=N_JOBS, backend='threading')(
            delayed(_compute_kurtosis_for_trial)(trial) for trial in sess_data
        )
        kurts = np.concatenate([k for k in kurts_list if len(k) > 0]) if kurts_list else np.array([])

        session_kurtosis[sess_id] = kurts

        if sess_id in train_sessions:
            all_kurt_train.append(kurts)
        else:
            all_kurt_holdout.append(kurts)

    all_kurt_train = np.concatenate(all_kurt_train) if all_kurt_train else np.array([])
    all_kurt_holdout = np.concatenate(all_kurt_holdout) if all_kurt_holdout else np.array([])

    # Left: Box plot per session
    ax = axes[0]
    data = [session_kurtosis[s] for s in unique_sessions]
    labels = [idx_to_session[s] for s in unique_sessions]
    colors_list = ['C0' if s in train_sessions else 'C1' for s in unique_sessions]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)

    ax.set_xlabel('Session')
    ax.set_ylabel('Kurtosis')
    ax.set_title('Per-Channel Kurtosis by Session\n(blue=train, orange=held-out)')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Gaussian (k=0)')

    # Right: Distribution comparison
    ax = axes[1]
    if all_kurt_train and all_kurt_holdout:
        # Clip extreme values for visualization
        p1, p99 = np.percentile(all_kurt_train + all_kurt_holdout, [1, 99])
        bins = np.linspace(p1, p99, 80)

        ax.hist(all_kurt_train, bins=bins, density=True, alpha=0.5,
               label='Train sessions', color='C0')
        ax.hist(all_kurt_holdout, bins=bins, density=True, alpha=0.5,
               label='Held-out sessions', color='C1')

        train_mean = np.mean(all_kurt_train)
        holdout_mean = np.mean(all_kurt_holdout)
        ax.axvline(train_mean, color='C0', linestyle='--', label=f'Train mean: {train_mean:.2f}')
        ax.axvline(holdout_mean, color='C1', linestyle='--', label=f'Held-out mean: {holdout_mean:.2f}')
        ax.axvline(0, color='gray', linestyle='-', alpha=0.5, label='Gaussian')

        stat, pval = stats.ks_2samp(all_kurt_train, all_kurt_holdout)
        ax.text(0.95, 0.95, f'KS test p-value: {pval:.2e}',
               transform=ax.transAxes, ha='right', va='top', fontsize=10)

    ax.set_xlabel('Kurtosis')
    ax.set_ylabel('Density')
    ax.set_title('Kurtosis Distribution: Train vs Held-out\n(>0: heavy-tailed, <0: light-tailed)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig19_kurtosis_distribution.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig19_kurtosis_distribution.png'}")


def figure_20_skewness_distribution(
    ob: np.ndarray,
    session_ids: np.ndarray,
    idx_to_session: Dict[int, str],
    train_idx: np.ndarray,
    output_dir: Path,
):
    """Figure 20: Skewness distribution by session (per-channel)."""
    print("Creating Figure 20: Skewness Distribution...")

    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    session_skewness = {}
    all_skew_train = []
    all_skew_holdout = []

    for sess_id in unique_sessions:
        mask = session_ids == sess_id
        sess_data = ob[mask]

        # Parallel computation
        skews_list = Parallel(n_jobs=N_JOBS, backend='threading')(
            delayed(_compute_skewness_for_trial)(trial) for trial in sess_data
        )
        skews = np.concatenate([s for s in skews_list if len(s) > 0]) if skews_list else np.array([])

        session_skewness[sess_id] = skews

        if sess_id in train_sessions:
            all_skew_train.append(skews)
        else:
            all_skew_holdout.append(skews)

    all_skew_train = np.concatenate(all_skew_train) if all_skew_train else np.array([])
    all_skew_holdout = np.concatenate(all_skew_holdout) if all_skew_holdout else np.array([])

    # Left: Box plot per session
    ax = axes[0]
    data = [session_skewness[s] for s in unique_sessions]
    labels = [idx_to_session[s] for s in unique_sessions]
    colors_list = ['C0' if s in train_sessions else 'C1' for s in unique_sessions]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)

    ax.set_xlabel('Session')
    ax.set_ylabel('Skewness')
    ax.set_title('Per-Channel Skewness by Session\n(blue=train, orange=held-out)')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Symmetric (s=0)')

    # Right: Distribution comparison
    ax = axes[1]
    if all_skew_train and all_skew_holdout:
        p1, p99 = np.percentile(all_skew_train + all_skew_holdout, [1, 99])
        bins = np.linspace(p1, p99, 80)

        ax.hist(all_skew_train, bins=bins, density=True, alpha=0.5,
               label='Train sessions', color='C0')
        ax.hist(all_skew_holdout, bins=bins, density=True, alpha=0.5,
               label='Held-out sessions', color='C1')

        train_mean = np.mean(all_skew_train)
        holdout_mean = np.mean(all_skew_holdout)
        ax.axvline(train_mean, color='C0', linestyle='--', label=f'Train mean: {train_mean:.2f}')
        ax.axvline(holdout_mean, color='C1', linestyle='--', label=f'Held-out mean: {holdout_mean:.2f}')
        ax.axvline(0, color='gray', linestyle='-', alpha=0.5, label='Symmetric')

        stat, pval = stats.ks_2samp(all_skew_train, all_skew_holdout)
        ax.text(0.95, 0.95, f'KS test p-value: {pval:.2e}',
               transform=ax.transAxes, ha='right', va='top', fontsize=10)

    ax.set_xlabel('Skewness')
    ax.set_ylabel('Density')
    ax.set_title('Skewness Distribution: Train vs Held-out\n(>0: right-skewed, <0: left-skewed)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig20_skewness_distribution.png')
    plt.close()
    print(f"  Saved: {output_dir / 'fig20_skewness_distribution.png'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze session differences for augmentation planning')
    parser.add_argument('--output-dir', type=str, default='figures/session_analysis',
                        help='Output directory for figures')
    parser.add_argument('--n-val-sessions', type=int, default=4,
                        help='Number of validation sessions')
    parser.add_argument('--n-test-sessions', type=int, default=0,
                        help='Number of test sessions (0 for no test set)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--force-recreate-splits', action='store_true',
                        help='Force recreate splits (ignore cached splits)')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Session Analysis for Augmentation Planning")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Validation sessions: {args.n_val_sessions}")
    print(f"Test sessions: {args.n_test_sessions}")
    print()

    # Set plot style
    set_style()

    # Load data with session splits
    print("Loading data...")
    if args.force_recreate_splits:
        print("  (forcing recreation of splits)")
    data = prepare_data(
        split_by_session=True,
        n_val_sessions=args.n_val_sessions,
        n_test_sessions=args.n_test_sessions,
        no_test_set=(args.n_test_sessions == 0),
        seed=args.seed,
        force_recreate_splits=args.force_recreate_splits,
    )

    ob = data['ob']
    pcx = data['pcx']
    odors = data['odors']
    vocab = data['vocab']
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data.get('test_idx', np.array([]))
    session_ids = data['session_ids']
    idx_to_session = data['idx_to_session']

    print(f"Data shape: {ob.shape}")
    print(f"Sessions: {len(np.unique(session_ids))}")
    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print()

    # Generate all figures IN PARALLEL
    print("Generating ALL 20 figures in PARALLEL...")
    print(f"Using {N_JOBS} CPU cores")
    print()

    # Define all figure functions with their arguments
    figure_tasks = [
        # Structural Analysis (1-10)
        (figure_1_pca_by_session, (ob, session_ids, idx_to_session, train_idx, val_idx, test_idx, output_dir)),
        (figure_2_pca_by_odor, (ob, odors, vocab, train_idx, output_dir)),
        (figure_3_tsne_by_session, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_4_session_covariances, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_5_session_psd, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_6_session_similarity, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_7_within_between_distances, (ob, session_ids, idx_to_session, output_dir)),
        (figure_8_odor_separability_per_session, (ob, odors, session_ids, idx_to_session, vocab, train_idx, output_dir)),
        (figure_9_channel_statistics, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_10_session_odor_interaction, (ob, odors, session_ids, idx_to_session, vocab, train_idx, output_dir)),
        # Probability Distribution Figures (11-20)
        (figure_11_envelope_magnitude_distribution, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_12_instantaneous_frequency_distribution, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_13_phase_distribution, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_14_amplitude_distribution, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_15_peak_amplitude_distribution, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_16_zero_crossing_rate_distribution, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_17_gradient_distribution, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_18_rms_energy_distribution, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_19_kurtosis_distribution, (ob, session_ids, idx_to_session, train_idx, output_dir)),
        (figure_20_skewness_distribution, (ob, session_ids, idx_to_session, train_idx, output_dir)),
    ]

    # Run all figures in parallel using ThreadPoolExecutor
    # (matplotlib is not process-safe, but thread-safe with Agg backend)
    with ThreadPoolExecutor(max_workers=min(20, N_JOBS)) as executor:
        futures = []
        for func, args in figure_tasks:
            futures.append(executor.submit(func, *args))

        # Wait for all to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error generating figure: {e}")

    print()
    print("=" * 60)
    print("Analysis complete!")
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)

    # Print summary statistics
    print()
    print("SUMMARY STATISTICS FOR AUGMENTATION PLANNING:")
    print("-" * 60)

    # Compute some key metrics
    features = compute_trial_features(ob)

    within_dists = []
    between_dists = []
    rng = np.random.RandomState(42)

    for sess_id in np.unique(session_ids):
        mask = session_ids == sess_id
        sess_features = features[mask]

        if len(sess_features) > 1:
            for _ in range(100):
                i, j = rng.choice(len(sess_features), 2, replace=False)
                within_dists.append(np.linalg.norm(sess_features[i] - sess_features[j]))

        other_features = features[~mask]
        if len(other_features) > 0:
            for _ in range(100):
                i = rng.randint(len(sess_features))
                j = rng.randint(len(other_features))
                between_dists.append(np.linalg.norm(sess_features[i] - other_features[j]))

    ratio = np.mean(between_dists) / np.mean(within_dists)

    print(f"Within-session distance (mean): {np.mean(within_dists):.2f}")
    print(f"Between-session distance (mean): {np.mean(between_dists):.2f}")
    print(f"Between/Within ratio: {ratio:.2f}x")
    print()

    if ratio > 1.5:
        print("⚠️  HIGH session variability detected!")
        print("   Recommended augmentations:")
        print("   - Covariance augmentation (cov_aug_strength: 0.3-0.5)")
        print("   - Session-aware mixup (aug_session_mixup: True)")
        print("   - Channel scaling (aug_channel_scale: True)")
        print("   - DC offset augmentation (aug_dc_offset: True)")
    else:
        print("✓ Moderate session variability")
        print("   Standard augmentations should suffice")

    print()


if __name__ == '__main__':
    main()
