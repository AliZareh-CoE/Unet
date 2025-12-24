#!/usr/bin/env python3
"""Session Analysis Script - Visualize cross-session variability for augmentation planning.

This script generates 20 figures analyzing session differences:
- Figures 1-10: Structural analysis (PCA, t-SNE, covariance, etc.)
- Figures 11-20: Probability distributions (envelope, phase, kurtosis, etc.)

FAST VERSION: Precomputes all features once, then generates figures quickly.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Limit thread counts BEFORE importing numpy/scipy
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.signal import welch, hilbert, find_peaks
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data import (
    prepare_data, SAMPLING_RATE_HZ,
    normalize_per_session, normalize_spectral_per_session, normalize_combined_per_session
)


def set_style():
    """Set publication-quality plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


# ==============================================================================
# FAST PRECOMPUTATION
# ==============================================================================

def precompute_features(ob: np.ndarray, session_ids: np.ndarray) -> Dict:
    """Precompute ALL signal features in one pass using vectorized operations."""
    print("\nPrecomputing all signal features (vectorized)...")
    n_trials, n_channels, n_time = ob.shape
    unique_sessions = np.unique(session_ids)
    fs = SAMPLING_RATE_HZ

    features = {k: {} for k in [
        'envelope', 'inst_freq', 'phase', 'amplitude', 'peaks',
        'zcr', 'gradient', 'kurtosis', 'skewness', 'rms_trial', 'rms_channel'
    ]}

    for sess_id in tqdm(unique_sessions, desc="Sessions"):
        mask = session_ids == sess_id
        sess_data = ob[mask]
        n_sess = sess_data.shape[0]

        # Amplitude (raw)
        features['amplitude'][sess_id] = sess_data.flatten()

        # RMS - vectorized
        features['rms_trial'][sess_id] = np.sqrt(np.mean(sess_data ** 2, axis=(1, 2)))
        features['rms_channel'][sess_id] = np.sqrt(np.mean(sess_data ** 2, axis=2)).flatten()

        # Zero-crossing rate - vectorized
        signs = np.sign(sess_data)
        zcr = np.sum(np.abs(np.diff(signs, axis=2)) > 0, axis=2) / n_time
        features['zcr'][sess_id] = zcr.flatten()

        # Gradient - vectorized
        features['gradient'][sess_id] = (np.diff(sess_data, axis=2) * fs).flatten()

        # Kurtosis/Skewness - vectorized
        flat = sess_data.reshape(-1, n_time)
        k = stats.kurtosis(flat, axis=1)
        s = stats.skew(flat, axis=1)
        features['kurtosis'][sess_id] = k[np.isfinite(k)]
        features['skewness'][sess_id] = s[np.isfinite(s)]

        # Hilbert features - per channel (vectorized across trials)
        envs, phases, freqs, peaks = [], [], [], []
        for ch in range(n_channels):
            ch_data = sess_data[:, ch, :]
            analytic = hilbert(ch_data, axis=1)
            env = np.abs(analytic)
            ph = np.angle(analytic)

            envs.append(env.flatten())
            phases.append(ph.flatten())

            # Inst freq
            unwrapped = np.unwrap(ph, axis=1)
            inst_freq = np.diff(unwrapped, axis=1) * fs / (2 * np.pi)
            valid = inst_freq[(inst_freq > 0) & (inst_freq < fs / 2)]
            freqs.append(valid)

            # Peaks
            for t in range(n_sess):
                peak_idx, _ = find_peaks(ch_data[t], distance=5)
                if len(peak_idx) > 0:
                    peaks.append(ch_data[t, peak_idx])

        features['envelope'][sess_id] = np.concatenate(envs)
        features['phase'][sess_id] = np.concatenate(phases)
        features['inst_freq'][sess_id] = np.concatenate(freqs) if freqs else np.array([])
        features['peaks'][sess_id] = np.concatenate(peaks) if peaks else np.array([0])

    print("Precomputation complete!\n")
    return features


def compute_trial_features(data: np.ndarray) -> np.ndarray:
    """Extract features for dimensionality reduction."""
    n_trials, n_channels, n_time = data.shape
    features = []

    for i in range(n_trials):
        trial = data[i]
        mean = trial.mean(axis=1)
        std = trial.std(axis=1)

        freqs, psd = welch(trial, fs=SAMPLING_RATE_HZ, nperseg=min(256, n_time), axis=1)
        bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 100)]
        band_powers = []
        for low, high in bands:
            mask = (freqs >= low) & (freqs <= high)
            if mask.any():
                band_powers.append(psd[:, mask].mean(axis=1))

        features.append(np.concatenate([mean, std] + band_powers))

    return np.array(features)


# ==============================================================================
# FIGURE FUNCTIONS (1-10: Structural)
# ==============================================================================

def figure_1_pca_by_session(ob, session_ids, idx_to_session, train_idx, val_idx, test_idx, output_dir):
    print("Creating Figure 1: PCA by Session...")
    features = compute_trial_features(ob)

    scaler = StandardScaler()
    pca = PCA(n_components=2)
    train_feat = features[train_idx]
    scaler.fit(train_feat)
    pca.fit(scaler.transform(train_feat))

    feat_pca = pca.transform(scaler.transform(features))

    unique_sess = np.unique(session_ids)
    train_sess = set(session_ids[train_idx])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sess)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for i, sid in enumerate(unique_sess):
        mask = session_ids == sid
        marker = 'o' if sid in train_sess else 's'
        label = f"{idx_to_session[sid]} ({'train' if sid in train_sess else 'held-out'})"
        ax.scatter(feat_pca[mask, 0], feat_pca[mask, 1], c=[colors[i]], marker=marker,
                  alpha=0.6, s=30, label=label)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA by Session')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    ax = axes[1]
    for i, sid in enumerate(unique_sess):
        mask = session_ids == sid
        centroid = feat_pca[mask].mean(axis=0)
        marker = 'o' if sid in train_sess else 's'
        ax.scatter(centroid[0], centroid[1], c=[colors[i]], marker=marker, s=200,
                  edgecolors='black' if sid in train_sess else 'red', linewidths=2,
                  label=idx_to_session[sid])

    ax.set_xlabel(f'PC1')
    ax.set_ylabel(f'PC2')
    ax.set_title('Session Centroids')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_pca_by_session.png')
    plt.close()
    print(f"  Saved: fig1_pca_by_session.png")


def figure_2_pca_by_odor(ob, odors, vocab, train_idx, output_dir):
    print("Creating Figure 2: PCA by Odor...")
    features = compute_trial_features(ob)

    scaler = StandardScaler()
    pca = PCA(n_components=2)
    scaler.fit(features[train_idx])
    pca.fit(scaler.transform(features[train_idx]))
    feat_pca = pca.transform(scaler.transform(features))

    id_to_odor = {v: k for k, v in vocab.items()}
    unique_odors = np.unique(odors)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_odors)))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, oid in enumerate(unique_odors):
        mask = odors == oid
        ax.scatter(feat_pca[mask, 0], feat_pca[mask, 1], c=[colors[i]], alpha=0.6, s=30,
                  label=id_to_odor.get(oid, f"Odor {oid}"))

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA by Odor')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_pca_by_odor.png')
    plt.close()
    print(f"  Saved: fig2_pca_by_odor.png")


def figure_3_tsne_by_session(ob, session_ids, idx_to_session, train_idx, output_dir, n_samples=500):
    print("Creating Figure 3: t-SNE by Session...")

    n_total = len(ob)
    if n_total > n_samples:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_total, n_samples, replace=False)
    else:
        sample_idx = np.arange(n_total)

    features = compute_trial_features(ob[sample_idx])
    sampled_sess = session_ids[sample_idx]

    scaler = StandardScaler()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    feat_tsne = tsne.fit_transform(scaler.fit_transform(features))

    unique_sess = np.unique(sampled_sess)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sess)))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, sid in enumerate(unique_sess):
        mask = sampled_sess == sid
        ax.scatter(feat_tsne[mask, 0], feat_tsne[mask, 1], c=[colors[i]], alpha=0.6, s=30,
                  label=idx_to_session[sid])

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f't-SNE by Session (n={len(sample_idx)})')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_tsne_by_session.png')
    plt.close()
    print(f"  Saved: fig3_tsne_by_session.png")


def figure_4_session_covariances(ob, session_ids, idx_to_session, train_idx, output_dir):
    print("Creating Figure 4: Session Covariance Matrices...")

    unique_sess = np.unique(session_ids)
    n_sess = len(unique_sess)
    train_sess = set(session_ids[train_idx])

    n_cols = min(4, n_sess)
    n_rows = (n_sess + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes).flatten()

    for i, sid in enumerate(unique_sess):
        mask = session_ids == sid
        sess_data = ob[mask]
        n_ch = sess_data.shape[1]
        data_concat = sess_data.transpose(1, 0, 2).reshape(n_ch, -1)
        cov = np.cov(data_concat)

        ax = axes[i]
        vmax = np.percentile(np.abs(cov), 99)
        ax.imshow(cov, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        suffix = " (train)" if sid in train_sess else " (held-out)"
        ax.set_title(f'{idx_to_session[sid]}{suffix}', fontsize=10)

    for i in range(n_sess, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Channel Covariance by Session', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_session_covariances.png')
    plt.close()
    print(f"  Saved: fig4_session_covariances.png")


def figure_5_session_psd(ob, session_ids, idx_to_session, train_idx, output_dir):
    print("Creating Figure 5: Per-Session PSD...")

    unique_sess = np.unique(session_ids)
    train_sess = set(session_ids[train_idx])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sess)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, sid in enumerate(unique_sess):
        mask = session_ids == sid
        sess_data = ob[mask]
        mean_sig = sess_data.reshape(-1, sess_data.shape[-1]).mean(axis=0)
        freqs, psd = welch(mean_sig, fs=SAMPLING_RATE_HZ, nperseg=256)

        ls = '-' if sid in train_sess else '--'
        ax.semilogy(freqs, psd, color=colors[i], linestyle=ls, label=idx_to_session[sid], alpha=0.8)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('PSD by Session (solid=train, dashed=held-out)')
    ax.set_xlim(0, 100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_session_psd.png')
    plt.close()
    print(f"  Saved: fig5_session_psd.png")


def figure_6_session_similarity(ob, session_ids, idx_to_session, train_idx, output_dir):
    print("Creating Figure 6: Session Similarity Matrix...")

    unique_sess = np.unique(session_ids)
    train_sess = set(session_ids[train_idx])

    session_stats = {}
    for sid in unique_sess:
        mask = session_ids == sid
        sess_data = ob[mask]
        mean = sess_data.mean(axis=(0, 2))
        std = sess_data.std(axis=(0, 2))
        session_stats[sid] = np.concatenate([mean, std])

    stat_matrix = np.array([session_stats[s] for s in unique_sess])
    distances = squareform(pdist(stat_matrix, metric='euclidean'))
    similarity = 1 / (1 + distances)

    fig, ax = plt.subplots(figsize=(10, 8))
    labels = [f"{idx_to_session[s]}{'*' if s not in train_sess else ''}" for s in unique_sess]

    im = ax.imshow(similarity, cmap='viridis')
    ax.set_xticks(range(len(unique_sess)))
    ax.set_yticks(range(len(unique_sess)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, label='Similarity')
    ax.set_title('Session Similarity (* = held-out)')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_session_similarity.png')
    plt.close()
    print(f"  Saved: fig6_session_similarity.png")


def figure_7_within_between_distances(ob, session_ids, idx_to_session, output_dir, n_samples=100):
    print("Creating Figure 7: Within vs Between Session Distances...")

    features = compute_trial_features(ob)
    unique_sess = np.unique(session_ids)
    rng = np.random.RandomState(42)

    within, between = [], []
    for sid in unique_sess:
        mask = session_ids == sid
        sess_feat = features[mask]
        n_s = len(sess_feat)

        if n_s > 1:
            for _ in range(min(n_samples, n_s*(n_s-1)//2)):
                i, j = rng.choice(n_s, 2, replace=False)
                within.append(np.linalg.norm(sess_feat[i] - sess_feat[j]))

        other_feat = features[~mask]
        if len(other_feat) > 0:
            for _ in range(n_samples):
                i = rng.randint(len(sess_feat))
                j = rng.randint(len(other_feat))
                between.append(np.linalg.norm(sess_feat[i] - other_feat[j]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(within, bins=50, alpha=0.6, label='Within-session', density=True)
    ax.hist(between, bins=50, alpha=0.6, label='Between-session', density=True)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Density')
    ax.set_title('Pairwise Distances')
    ax.legend()

    ax = axes[1]
    bp = ax.boxplot([within, between], tick_labels=['Within', 'Between'], patch_artist=True)
    bp['boxes'][0].set_facecolor('C0')
    bp['boxes'][1].set_facecolor('C1')
    ratio = np.mean(between) / np.mean(within)
    ax.set_title(f'Between/Within ratio: {ratio:.2f}x')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_within_between_distances.png')
    plt.close()
    print(f"  Saved: fig7_within_between_distances.png")


def figure_8_odor_separability(ob, odors, session_ids, idx_to_session, vocab, train_idx, output_dir):
    print("Creating Figure 8: Odor Separability per Session...")

    features = compute_trial_features(ob)
    unique_sess = np.unique(session_ids)
    train_sess = set(session_ids[train_idx])

    scores = {}
    for sid in unique_sess:
        mask = session_ids == sid
        sess_feat = features[mask]
        sess_odors = odors[mask]
        if len(np.unique(sess_odors)) >= 2 and len(sess_feat) >= 4:
            try:
                scores[sid] = silhouette_score(sess_feat, sess_odors)
            except:
                scores[sid] = np.nan
        else:
            scores[sid] = np.nan

    fig, ax = plt.subplots(figsize=(12, 6))
    names = [idx_to_session[s] for s in unique_sess]
    vals = [scores[s] for s in unique_sess]
    colors = ['C0' if s in train_sess else 'C1' for s in unique_sess]

    ax.bar(range(len(unique_sess)), vals, color=colors)
    ax.set_xticks(range(len(unique_sess)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Odor Separability per Session')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig8_odor_separability_per_session.png')
    plt.close()
    print(f"  Saved: fig8_odor_separability_per_session.png")


def figure_9_channel_statistics(ob, session_ids, idx_to_session, train_idx, output_dir):
    print("Creating Figure 9: Channel Statistics by Session...")

    unique_sess = np.unique(session_ids)
    train_sess = set(session_ids[train_idx])
    n_ch = ob.shape[1]

    sess_means = np.zeros((len(unique_sess), n_ch))
    for i, sid in enumerate(unique_sess):
        mask = session_ids == sid
        sess_means[i] = ob[mask].mean(axis=(0, 2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for i, sid in enumerate(unique_sess):
        ls = '-' if sid in train_sess else '--'
        ax.plot(sess_means[i], linestyle=ls, alpha=0.7, label=idx_to_session[sid])
    ax.set_xlabel('Channel')
    ax.set_ylabel('Mean')
    ax.set_title('Per-Channel Mean by Session')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    ax = axes[1]
    ch_var = sess_means.var(axis=0)
    ax.bar(range(n_ch), ch_var)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Variance across sessions')
    ax.set_title('Cross-Session Channel Variance')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig9_channel_statistics.png')
    plt.close()
    print(f"  Saved: fig9_channel_statistics.png")


def figure_10_session_odor_interaction(ob, odors, session_ids, idx_to_session, vocab, train_idx, output_dir):
    print("Creating Figure 10: Session-Odor Interaction...")

    features = compute_trial_features(ob)
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    scaler.fit(features[train_idx])
    pca.fit(scaler.transform(features[train_idx]))
    feat_pca = pca.transform(scaler.transform(features))

    unique_sess = np.unique(session_ids)
    unique_odors = np.unique(odors)
    train_sess = set(session_ids[train_idx])
    id_to_odor = {v: k for k, v in vocab.items()}
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_odors)))

    n_cols = min(4, len(unique_sess))
    n_rows = (len(unique_sess) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = np.atleast_2d(axes).flatten()

    for idx, sid in enumerate(unique_sess):
        ax = axes[idx]
        sess_mask = session_ids == sid

        for i, oid in enumerate(unique_odors):
            mask = sess_mask & (odors == oid)
            if mask.any():
                ax.scatter(feat_pca[mask, 0], feat_pca[mask, 1], c=[colors[i]], alpha=0.6, s=30,
                          label=id_to_odor.get(oid, f"Odor {oid}") if idx == 0 else None)

        suffix = " (train)" if sid in train_sess else " (held-out)"
        ax.set_title(f'{idx_to_session[sid]}{suffix}')

    for idx in range(len(unique_sess), len(axes)):
        axes[idx].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.suptitle('PCA by Odor within Each Session', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig10_session_odor_interaction.png')
    plt.close()
    print(f"  Saved: fig10_session_odor_interaction.png")


# ==============================================================================
# FIGURE FUNCTIONS (11-20: Probability Distributions)
# ==============================================================================

def _plot_dist(data, idx_to_session, train_sess, output_dir, fname, xlabel, title, xlim=None, boxplot=False):
    """Generic distribution plot helper."""
    unique_sess = sorted(data.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sess)))

    train_data = [data[s] for s in unique_sess if s in train_sess]
    holdout_data = [data[s] for s in unique_sess if s not in train_sess]
    all_train = np.concatenate(train_data) if train_data else np.array([])
    all_holdout = np.concatenate(holdout_data) if holdout_data else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if boxplot:
        ax = axes[0]
        bp = ax.boxplot([data[s] for s in unique_sess],
                       tick_labels=[idx_to_session[s] for s in unique_sess], patch_artist=True)
        for i, (patch, sid) in enumerate(zip(bp['boxes'], unique_sess)):
            patch.set_facecolor('C0' if sid in train_sess else 'C1')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel(xlabel)
        ax.set_title(f'{title} by Session')
    else:
        ax = axes[0]
        all_vals = np.concatenate(list(data.values()))
        p1, p99 = np.percentile(all_vals, [1, 99])
        x_range = np.linspace(p1, p99, 200)

        for i, sid in enumerate(unique_sess):
            ls = '-' if sid in train_sess else '--'
            d = data[sid]
            if len(d) > 50000:
                d = np.random.choice(d, 50000, replace=False)
            try:
                kde = stats.gaussian_kde(d)
                ax.plot(x_range, kde(x_range), color=colors[i], linestyle=ls,
                       label=idx_to_session[sid], alpha=0.8)
            except:
                pass

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(f'{title} by Session')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        if xlim:
            ax.set_xlim(xlim)

    ax = axes[1]
    if len(all_train) > 0 and len(all_holdout) > 0:
        if len(all_train) > 200000:
            all_train = np.random.choice(all_train, 200000, replace=False)
        if len(all_holdout) > 200000:
            all_holdout = np.random.choice(all_holdout, 200000, replace=False)

        combined = np.concatenate([all_train, all_holdout])
        p1, p99 = np.percentile(combined, [1, 99])
        bins = np.linspace(p1, p99, 100)

        ax.hist(all_train, bins=bins, density=True, alpha=0.5, label='Train', color='C0')
        ax.hist(all_holdout, bins=bins, density=True, alpha=0.5, label='Held-out', color='C1')

        stat, pval = stats.ks_2samp(all_train[:10000], all_holdout[:10000])
        ax.text(0.95, 0.95, f'KS p={pval:.2e}', transform=ax.transAxes, ha='right', va='top')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(f'{title}: Train vs Held-out')
    ax.legend()
    if xlim:
        ax.set_xlim(xlim)

    plt.tight_layout()
    plt.savefig(output_dir / fname)
    plt.close()


def figures_11_20_distributions(features, idx_to_session, train_sess, output_dir):
    """Generate all 10 distribution figures using precomputed features."""

    configs = [
        ('envelope', 'fig11_envelope_magnitude_distribution.png', 'Envelope Magnitude', 'Envelope', (0, None), False),
        ('inst_freq', 'fig12_instantaneous_frequency_distribution.png', 'Inst. Frequency (Hz)', 'Instantaneous Frequency', (0, 100), False),
        ('phase', 'fig13_phase_distribution.png', 'Phase (rad)', 'Phase', (-np.pi, np.pi), False),
        ('amplitude', 'fig14_amplitude_distribution.png', 'Amplitude', 'Raw Amplitude', None, False),
        ('peaks', 'fig15_peak_amplitude_distribution.png', 'Peak Amplitude', 'Peak Amplitude', None, False),
        ('zcr', 'fig16_zero_crossing_rate_distribution.png', 'ZCR', 'Zero-Crossing Rate', None, True),
        ('gradient', 'fig17_gradient_distribution.png', 'Gradient', 'Signal Gradient', None, False),
        ('rms_trial', 'fig18_rms_energy_distribution.png', 'RMS Energy', 'RMS Energy', None, True),
        ('kurtosis', 'fig19_kurtosis_distribution.png', 'Kurtosis', 'Kurtosis', None, True),
        ('skewness', 'fig20_skewness_distribution.png', 'Skewness', 'Skewness', None, True),
    ]

    for i, (key, fname, xlabel, title, xlim, boxplot) in enumerate(configs, 11):
        print(f"Creating Figure {i}: {title}...")
        _plot_dist(features[key], idx_to_session, train_sess, output_dir, fname, xlabel, title, xlim, boxplot)
        print(f"  Saved: {fname}")


# ==============================================================================
# NORMALIZATION COMPARISON FIGURES
# ==============================================================================

def figure_norm_channel_means(ob_before: np.ndarray, ob_after: np.ndarray,
                               session_ids: np.ndarray, idx_to_session: Dict,
                               train_idx: np.ndarray, output_dir: Path, suffix: str = ""):
    """Compare per-channel means before/after normalization."""
    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sessions)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, sess_id in enumerate(unique_sessions):
        mask = session_ids == sess_id
        linestyle = '-' if sess_id in train_sessions else '--'
        linewidth = 1.5 if sess_id in train_sessions else 2.5

        # Before
        means_before = ob_before[mask].mean(axis=(0, 2))
        axes[0].plot(means_before, color=colors[i], linestyle=linestyle,
                    linewidth=linewidth, label=idx_to_session[sess_id], alpha=0.8)

        # After
        means_after = ob_after[mask].mean(axis=(0, 2))
        axes[1].plot(means_after, color=colors[i], linestyle=linestyle,
                    linewidth=linewidth, label=idx_to_session[sess_id], alpha=0.8)

    axes[0].set_title('Per-Channel Mean by Session (BEFORE)', fontsize=14)
    axes[0].set_xlabel('Channel')
    axes[0].set_ylabel('Mean')
    axes[0].axhline(y=0, color='k', linestyle=':', alpha=0.5)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    axes[1].set_title('Per-Channel Mean by Session (AFTER)', fontsize=14)
    axes[1].set_xlabel('Channel')
    axes[1].set_ylabel('Mean')
    axes[1].axhline(y=0, color='k', linestyle=':', alpha=0.5)

    # Compute variance reduction
    all_means_before = np.array([ob_before[session_ids == s].mean(axis=(0, 2)) for s in unique_sessions])
    all_means_after = np.array([ob_after[session_ids == s].mean(axis=(0, 2)) for s in unique_sessions])
    var_before = np.var(all_means_before, axis=0).mean()
    var_after = np.var(all_means_after, axis=0).mean()
    reduction = (1 - var_after / var_before) * 100 if var_before > 0 else 0

    fig.suptitle(f'Per-Channel Mean Comparison{suffix}\n'
                 f'Cross-session variance: {var_before:.6f} → {var_after:.6f} ({reduction:.1f}% reduction)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'norm_comparison_channel_means{suffix.replace(" ", "_").lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Channel means variance reduction: {reduction:.1f}%")


def figure_norm_psd(ob_before: np.ndarray, ob_after: np.ndarray,
                    session_ids: np.ndarray, idx_to_session: Dict,
                    train_idx: np.ndarray, output_dir: Path, suffix: str = ""):
    """Compare PSD before/after normalization."""
    unique_sessions = np.unique(session_ids)
    train_sessions = set(session_ids[train_idx])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sessions)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    psd_spread_before = []
    psd_spread_after = []

    for i, sess_id in enumerate(unique_sessions):
        mask = session_ids == sess_id
        linestyle = '-' if sess_id in train_sessions else '--'
        linewidth = 1.5 if sess_id in train_sessions else 2.5

        # Sample trials for speed
        sess_before = ob_before[mask][:20]
        sess_after = ob_after[mask][:20]

        # Compute average PSD
        psds_before = []
        psds_after = []
        for trial_b, trial_a in zip(sess_before, sess_after):
            freqs, psd_b = welch(trial_b, fs=SAMPLING_RATE_HZ, nperseg=256, axis=1)
            _, psd_a = welch(trial_a, fs=SAMPLING_RATE_HZ, nperseg=256, axis=1)
            psds_before.append(psd_b.mean(axis=0))
            psds_after.append(psd_a.mean(axis=0))

        avg_psd_before = np.mean(psds_before, axis=0)
        avg_psd_after = np.mean(psds_after, axis=0)

        psd_spread_before.append(avg_psd_before)
        psd_spread_after.append(avg_psd_after)

        axes[0].semilogy(freqs, avg_psd_before, color=colors[i], linestyle=linestyle,
                        linewidth=linewidth, label=idx_to_session[sess_id], alpha=0.8)
        axes[1].semilogy(freqs, avg_psd_after, color=colors[i], linestyle=linestyle,
                        linewidth=linewidth, label=idx_to_session[sess_id], alpha=0.8)

    axes[0].set_title('PSD by Session (BEFORE)', fontsize=14)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('PSD')
    axes[0].set_xlim([0, 100])
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    axes[1].set_title('PSD by Session (AFTER)', fontsize=14)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('PSD')
    axes[1].set_xlim([0, 100])

    # Compute PSD alignment improvement
    psd_spread_before = np.array(psd_spread_before)
    psd_spread_after = np.array(psd_spread_after)
    spread_before = np.std(psd_spread_before, axis=0).mean()
    spread_after = np.std(psd_spread_after, axis=0).mean()
    reduction = (1 - spread_after / spread_before) * 100 if spread_before > 0 else 0

    fig.suptitle(f'PSD Comparison{suffix}\n'
                 f'Cross-session PSD std: {spread_before:.4f} → {spread_after:.4f} ({reduction:.1f}% reduction)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'norm_comparison_psd{suffix.replace(" ", "_").lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PSD spread reduction: {reduction:.1f}%")


def figure_norm_ks_test(ob_before: np.ndarray, ob_after: np.ndarray,
                        session_ids: np.ndarray, train_idx: np.ndarray,
                        output_dir: Path, suffix: str = ""):
    """Compare train vs held-out distributions with KS test."""
    train_sessions = set(session_ids[train_idx])
    train_mask = np.isin(session_ids, list(train_sessions))
    holdout_mask = ~train_mask

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    n_sample = 50000

    # Amplitude comparison
    train_before = ob_before[train_mask].flatten()
    holdout_before = ob_before[holdout_mask].flatten()
    train_after = ob_after[train_mask].flatten()
    holdout_after = ob_after[holdout_mask].flatten()

    axes[0, 0].hist(np.random.choice(train_before, min(n_sample, len(train_before)), replace=False),
                    bins=100, alpha=0.5, label='Train', density=True)
    axes[0, 0].hist(np.random.choice(holdout_before, min(n_sample, len(holdout_before)), replace=False),
                    bins=100, alpha=0.5, label='Held-out', density=True)
    ks_before = stats.ks_2samp(train_before[:n_sample], holdout_before[:n_sample])
    axes[0, 0].set_title(f'Amplitude Distribution (BEFORE)\nKS p={ks_before.pvalue:.2e}', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Amplitude')
    axes[0, 0].set_ylabel('Density')

    axes[0, 1].hist(np.random.choice(train_after, min(n_sample, len(train_after)), replace=False),
                    bins=100, alpha=0.5, label='Train', density=True)
    axes[0, 1].hist(np.random.choice(holdout_after, min(n_sample, len(holdout_after)), replace=False),
                    bins=100, alpha=0.5, label='Held-out', density=True)
    ks_after = stats.ks_2samp(train_after[:n_sample], holdout_after[:n_sample])
    axes[0, 1].set_title(f'Amplitude Distribution (AFTER)\nKS p={ks_after.pvalue:.2e}', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Amplitude')
    axes[0, 1].set_ylabel('Density')

    # Gradient comparison
    train_grad_before = np.diff(ob_before[train_mask], axis=2).flatten()
    holdout_grad_before = np.diff(ob_before[holdout_mask], axis=2).flatten()
    train_grad_after = np.diff(ob_after[train_mask], axis=2).flatten()
    holdout_grad_after = np.diff(ob_after[holdout_mask], axis=2).flatten()

    axes[1, 0].hist(np.random.choice(train_grad_before, min(n_sample, len(train_grad_before)), replace=False),
                    bins=100, alpha=0.5, label='Train', density=True)
    axes[1, 0].hist(np.random.choice(holdout_grad_before, min(n_sample, len(holdout_grad_before)), replace=False),
                    bins=100, alpha=0.5, label='Held-out', density=True)
    ks_grad_before = stats.ks_2samp(train_grad_before[:n_sample], holdout_grad_before[:n_sample])
    axes[1, 0].set_title(f'Gradient Distribution (BEFORE)\nKS p={ks_grad_before.pvalue:.2e}', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Gradient')
    axes[1, 0].set_ylabel('Density')

    axes[1, 1].hist(np.random.choice(train_grad_after, min(n_sample, len(train_grad_after)), replace=False),
                    bins=100, alpha=0.5, label='Train', density=True)
    axes[1, 1].hist(np.random.choice(holdout_grad_after, min(n_sample, len(holdout_grad_after)), replace=False),
                    bins=100, alpha=0.5, label='Held-out', density=True)
    ks_grad_after = stats.ks_2samp(train_grad_after[:n_sample], holdout_grad_after[:n_sample])
    axes[1, 1].set_title(f'Gradient Distribution (AFTER)\nKS p={ks_grad_after.pvalue:.2e}', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Gradient')
    axes[1, 1].set_ylabel('Density')

    fig.suptitle(f'Train vs Held-out Distribution Comparison{suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'norm_comparison_ks_test{suffix.replace(" ", "_").lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  KS test amplitude: p={ks_before.pvalue:.2e} → p={ks_after.pvalue:.2e}")
    print(f"  KS test gradient: p={ks_grad_before.pvalue:.2e} → p={ks_grad_after.pvalue:.2e}")

    return {
        'ks_amplitude_before': ks_before.pvalue,
        'ks_amplitude_after': ks_after.pvalue,
        'ks_gradient_before': ks_grad_before.pvalue,
        'ks_gradient_after': ks_grad_after.pvalue,
    }


def figure_norm_summary(results: Dict, output_dir: Path, suffix: str = ""):
    """Create summary figure with all normalization metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Bar chart for variance reduction
    if 'mean_var_before' in results and 'mean_var_after' in results:
        x = ['Before', 'After']
        y = [results['mean_var_before'], results['mean_var_after']]
        bars = axes[0].bar(x, y, color=['#ff6b6b', '#4ecdc4'])
        axes[0].set_title('Channel Mean Variance')
        axes[0].set_ylabel('Variance')
        for bar, val in zip(bars, y):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Bar chart for PSD spread reduction
    if 'psd_spread_before' in results and 'psd_spread_after' in results:
        x = ['Before', 'After']
        y = [results['psd_spread_before'], results['psd_spread_after']]
        bars = axes[1].bar(x, y, color=['#ff6b6b', '#4ecdc4'])
        axes[1].set_title('PSD Cross-Session Spread')
        axes[1].set_ylabel('Std')
        for bar, val in zip(bars, y):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # KS test p-values (log scale)
    if 'ks_amplitude_before' in results:
        x = ['Amp Before', 'Amp After', 'Grad Before', 'Grad After']
        y = [results['ks_amplitude_before'], results['ks_amplitude_after'],
             results['ks_gradient_before'], results['ks_gradient_after']]
        colors = ['#ff6b6b', '#4ecdc4', '#ff6b6b', '#4ecdc4']
        bars = axes[2].bar(x, y, color=colors)
        axes[2].set_yscale('log')
        axes[2].set_title('KS Test p-values (Train vs Held-out)')
        axes[2].set_ylabel('p-value (log scale)')
        axes[2].axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
        axes[2].legend()
        axes[2].tick_params(axis='x', rotation=45)

    fig.suptitle(f'Normalization Impact Summary{suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'norm_summary{suffix.replace(" ", "_").lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def run_normalization_comparison(ob: np.ndarray, session_ids: np.ndarray,
                                  idx_to_session: Dict, train_idx: np.ndarray,
                                  output_dir: Path, method: str = 'combined'):
    """Run full normalization comparison analysis."""
    print(f"\n{'='*60}")
    print(f"NORMALIZATION COMPARISON: {method.upper()}")
    print(f"{'='*60}")

    train_session_ids = session_ids[train_idx]
    ob_before = ob.copy()

    # Apply normalization based on method
    if method == 'zscore':
        ob_after, _ = normalize_per_session(ob_before, session_ids, verbose=True)
        suffix = " (Z-Score)"
    elif method == 'spectral':
        ob_after, _ = normalize_spectral_per_session(
            ob_before, session_ids, train_session_ids=train_session_ids, verbose=True
        )
        suffix = " (Spectral)"
    else:  # combined
        ob_after, _ = normalize_combined_per_session(
            ob_before, session_ids, train_session_ids=train_session_ids, verbose=True
        )
        suffix = " (Combined)"

    # Generate comparison figures
    print("\nGenerating comparison figures...")

    print("  Figure: Channel means comparison")
    figure_norm_channel_means(ob_before, ob_after, session_ids, idx_to_session, train_idx, output_dir, suffix)

    print("  Figure: PSD comparison")
    figure_norm_psd(ob_before, ob_after, session_ids, idx_to_session, train_idx, output_dir, suffix)

    print("  Figure: KS test comparison")
    ks_results = figure_norm_ks_test(ob_before, ob_after, session_ids, train_idx, output_dir, suffix)

    # Compute additional metrics for summary
    unique_sessions = np.unique(session_ids)
    all_means_before = np.array([ob_before[session_ids == s].mean(axis=(0, 2)) for s in unique_sessions])
    all_means_after = np.array([ob_after[session_ids == s].mean(axis=(0, 2)) for s in unique_sessions])

    results = {
        'mean_var_before': np.var(all_means_before, axis=0).mean(),
        'mean_var_after': np.var(all_means_after, axis=0).mean(),
        **ks_results
    }

    print("  Figure: Summary")
    figure_norm_summary(results, output_dir, suffix)

    print(f"\n{'='*60}")
    print("NORMALIZATION COMPARISON COMPLETE")
    print(f"{'='*60}")

    return ob_after, results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Session analysis for augmentation planning')
    parser.add_argument('--output-dir', type=str, default='figures/session_analysis')
    parser.add_argument('--n-val-sessions', type=int, default=4)
    parser.add_argument('--n-test-sessions', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force-recreate-splits', action='store_true')
    # Normalization comparison options
    parser.add_argument('--normalize', type=str, choices=['zscore', 'spectral', 'combined', 'all'],
                        help='Test normalization method and generate before/after comparison')
    parser.add_argument('--skip-standard-analysis', action='store_true',
                        help='Skip standard 20 figures, only run normalization comparison')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Session Analysis for Augmentation Planning")
    print("=" * 60)

    set_style()

    print("\nLoading data...")
    data = prepare_data(
        split_by_session=True,
        n_val_sessions=args.n_val_sessions,
        n_test_sessions=args.n_test_sessions,
        no_test_set=(args.n_test_sessions == 0),
        seed=args.seed,
        force_recreate_splits=args.force_recreate_splits,
    )

    ob = data['ob']
    odors = data['odors']
    vocab = data['vocab']
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data.get('test_idx', np.array([]))
    session_ids = data['session_ids']
    idx_to_session = data['idx_to_session']

    print(f"\nData: {ob.shape}, {len(np.unique(session_ids))} sessions")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Run normalization comparison if requested
    if args.normalize:
        norm_output_dir = output_dir / 'normalization'
        norm_output_dir.mkdir(parents=True, exist_ok=True)

        if args.normalize == 'all':
            # Run all normalization methods
            for method in ['zscore', 'spectral', 'combined']:
                run_normalization_comparison(ob, session_ids, idx_to_session, train_idx,
                                            norm_output_dir, method=method)
        else:
            run_normalization_comparison(ob, session_ids, idx_to_session, train_idx,
                                        norm_output_dir, method=args.normalize)

        print(f"\nNormalization comparison figures saved to: {norm_output_dir}")

    # Skip standard analysis if requested
    if args.skip_standard_analysis:
        print("\nSkipping standard 20 figures (--skip-standard-analysis)")
        print("=" * 60)
        return

    # Precompute all features ONCE
    features = precompute_features(ob, session_ids)
    train_sess = set(session_ids[train_idx])

    # Generate structural figures (1-10)
    print("\n--- Generating Structural Figures (1-10) ---")
    figure_1_pca_by_session(ob, session_ids, idx_to_session, train_idx, val_idx, test_idx, output_dir)
    figure_2_pca_by_odor(ob, odors, vocab, train_idx, output_dir)
    figure_3_tsne_by_session(ob, session_ids, idx_to_session, train_idx, output_dir)
    figure_4_session_covariances(ob, session_ids, idx_to_session, train_idx, output_dir)
    figure_5_session_psd(ob, session_ids, idx_to_session, train_idx, output_dir)
    figure_6_session_similarity(ob, session_ids, idx_to_session, train_idx, output_dir)
    figure_7_within_between_distances(ob, session_ids, idx_to_session, output_dir)
    figure_8_odor_separability(ob, odors, session_ids, idx_to_session, vocab, train_idx, output_dir)
    figure_9_channel_statistics(ob, session_ids, idx_to_session, train_idx, output_dir)
    figure_10_session_odor_interaction(ob, odors, session_ids, idx_to_session, vocab, train_idx, output_dir)

    # Generate distribution figures (11-20) using precomputed features
    print("\n--- Generating Distribution Figures (11-20) ---")
    figures_11_20_distributions(features, idx_to_session, train_sess, output_dir)

    print("\n" + "=" * 60)
    print(f"All 20 figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
