#!/usr/bin/env python3
"""Session Analysis Script - Visualize cross-session variability for augmentation planning.

This script generates 10 figures analyzing session differences:
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
from scipy.signal import welch
from scipy.spatial.distance import pdist, squareform

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

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

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
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
    data = prepare_data(
        split_by_session=True,
        n_val_sessions=args.n_val_sessions,
        n_test_sessions=args.n_test_sessions,
        no_test_set=(args.n_test_sessions == 0),
        seed=args.seed,
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

    # Generate all figures
    print("Generating figures...")
    print()

    figure_1_pca_by_session(ob, session_ids, idx_to_session, train_idx, val_idx, test_idx, output_dir)
    figure_2_pca_by_odor(ob, odors, vocab, train_idx, output_dir)
    figure_3_tsne_by_session(ob, session_ids, idx_to_session, train_idx, output_dir)
    figure_4_session_covariances(ob, session_ids, idx_to_session, train_idx, output_dir)
    figure_5_session_psd(ob, session_ids, idx_to_session, train_idx, output_dir)
    figure_6_session_similarity(ob, session_ids, idx_to_session, train_idx, output_dir)
    figure_7_within_between_distances(ob, session_ids, idx_to_session, output_dir)
    figure_8_odor_separability_per_session(ob, odors, session_ids, idx_to_session, vocab, train_idx, output_dir)
    figure_9_channel_statistics(ob, session_ids, idx_to_session, train_idx, output_dir)
    figure_10_session_odor_interaction(ob, odors, session_ids, idx_to_session, vocab, train_idx, output_dir)

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
