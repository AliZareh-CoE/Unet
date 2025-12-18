#!/usr/bin/env python3
"""GPU-Accelerated Augmentation Explorer for Cross-Session Generalization.

Uses PyTorch GPU acceleration for:
- Feature extraction (FFT, Hilbert transform, covariance)
- Distance computation (MMD, coverage)
- Parallel augmentation evaluation

Usage:
    python experiment/augmentation_explorer_gpu.py --n-val-sessions 4 --output-dir figures/augmentation_exploration

Author: Claude (Anthropic)
"""
from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ['NUMEXPR_MAX_THREADS'] = '16'

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as sp_signal
from scipy.linalg import sqrtm, eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Check GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================================
# Data Loading
# =============================================================================

def load_data(n_val_sessions: int = 4, force_recreate_splits: bool = False, seed: int = 42) -> Dict[str, Any]:
    """Load data with session-based splits."""
    from experiment.data import (
        load_signals, load_odor_labels, load_session_ids,
        load_or_create_session_splits, DATA_PATH, ODOR_CSV_PATH
    )

    print("Loading neural signals...")
    signals = load_signals(DATA_PATH)
    n_trials = signals.shape[0]
    print(f"  Loaded {n_trials} trials, shape: {signals.shape}")

    ob = signals[:, 0, :, :]
    pcx = signals[:, 1, :, :]

    odors, odor_vocab = load_odor_labels(ODOR_CSV_PATH, n_trials)
    session_ids, session_to_idx, idx_to_session = load_session_ids(ODOR_CSV_PATH, num_trials=n_trials)

    train_idx, val_idx, test_idx, split_info = load_or_create_session_splits(
        session_ids=session_ids, odors=odors, n_test_sessions=0, n_val_sessions=n_val_sessions,
        seed=seed, force_recreate=force_recreate_splits, idx_to_session=idx_to_session, no_test_set=True,
    )

    print(f"  Train: {len(train_idx)} trials, Val: {len(val_idx)} trials")

    return {
        'ob': ob, 'pcx': pcx, 'odors': odors, 'session_ids': session_ids,
        'idx_to_session': idx_to_session, 'train_idx': train_idx, 'val_idx': val_idx,
        'train_sessions': split_info.get('train_sessions', []),
        'val_sessions': split_info.get('val_sessions', []),
    }


# =============================================================================
# GPU Feature Extraction
# =============================================================================

def extract_features_gpu(ob: np.ndarray, pcx: np.ndarray, device: torch.device = DEVICE) -> torch.Tensor:
    """Extract features using GPU acceleration."""
    n_trials, n_channels, n_time = ob.shape

    # Move to GPU
    ob_t = torch.from_numpy(ob).float().to(device)
    pcx_t = torch.from_numpy(pcx).float().to(device)
    combined = torch.cat([ob_t, pcx_t], dim=1)  # (n_trials, 64, n_time)

    features_list = []

    # 1. Band power features using FFT
    fft = torch.fft.rfft(combined, dim=-1)
    psd = (fft.abs() ** 2) / n_time
    freqs = torch.fft.rfftfreq(n_time, d=1/1000).to(device)

    bands = [(1, 4), (4, 12), (12, 30), (30, 60), (60, 100)]
    for f_low, f_high in bands:
        mask = (freqs >= f_low) & (freqs <= f_high)
        band_power = psd[:, :, mask].mean(dim=-1)  # (n_trials, 64)
        features_list.append(band_power)

    # 2. Temporal statistics
    signal_mean = combined.mean(dim=-1)
    signal_std = combined.std(dim=-1)
    gradient = torch.diff(combined, dim=-1)
    gradient_mean = gradient.abs().mean(dim=-1)
    gradient_std = gradient.std(dim=-1)

    features_list.extend([signal_mean, signal_std, gradient_mean, gradient_std])

    # 3. Simple envelope approximation (absolute value smoothed)
    # Full Hilbert is expensive on GPU, use rolling window RMS instead
    window_size = 50
    rms = F.avg_pool1d(combined.abs(), kernel_size=window_size, stride=1, padding=window_size//2)
    rms = rms[:, :, :n_time]  # Trim to original size
    rms_mean = rms.mean(dim=-1)
    rms_std = rms.std(dim=-1)

    features_list.extend([rms_mean, rms_std])

    # 4. Covariance features (flattened upper triangle)
    # Compute in batches to save memory
    batch_size = 100
    cov_features = []
    triu_indices = torch.triu_indices(64, 64, offset=1)

    for start in range(0, n_trials, batch_size):
        end = min(start + batch_size, n_trials)
        batch = combined[start:end]
        batch_centered = batch - batch.mean(dim=-1, keepdim=True)

        # Covariance: (B, C, C) = (B, C, T) @ (B, T, C) / T
        cov = torch.bmm(batch_centered, batch_centered.transpose(1, 2)) / n_time
        cov_flat = cov[:, triu_indices[0], triu_indices[1]]
        cov_features.append(cov_flat)

    cov_flat = torch.cat(cov_features, dim=0)
    features_list.append(cov_flat)

    # Concatenate all features
    all_features = torch.cat(features_list, dim=1)

    # Normalize each feature
    mean = all_features.mean(dim=0, keepdim=True)
    std = all_features.std(dim=0, keepdim=True) + 1e-8
    all_features = (all_features - mean) / std

    return all_features


def extract_covariance_gpu(ob: np.ndarray, pcx: np.ndarray, device: torch.device = DEVICE) -> torch.Tensor:
    """Extract covariance matrices on GPU."""
    n_trials, n_channels, n_time = ob.shape

    ob_t = torch.from_numpy(ob).float().to(device)
    pcx_t = torch.from_numpy(pcx).float().to(device)
    combined = torch.cat([ob_t, pcx_t], dim=1)

    combined_centered = combined - combined.mean(dim=-1, keepdim=True)
    cov = torch.bmm(combined_centered, combined_centered.transpose(1, 2)) / n_time

    return cov


# =============================================================================
# GPU Distance Metrics
# =============================================================================

def compute_mmd_gpu(X: torch.Tensor, Y: torch.Tensor, gamma: Optional[float] = None) -> float:
    """Compute MMD on GPU using RBF kernel."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # Subsample if too large
    max_samples = 1000
    if len(X) > max_samples:
        idx = torch.randperm(len(X))[:max_samples]
        X = X[idx]
    if len(Y) > max_samples:
        idx = torch.randperm(len(Y))[:max_samples]
        Y = Y[idx]

    # RBF kernel using efficient pairwise distance
    def rbf_kernel(A, B):
        A_sq = (A ** 2).sum(dim=1, keepdim=True)
        B_sq = (B ** 2).sum(dim=1, keepdim=True)
        dists = A_sq + B_sq.T - 2 * torch.mm(A, B.T)
        return torch.exp(-gamma * dists)

    K_XX = rbf_kernel(X, X)
    K_YY = rbf_kernel(Y, Y)
    K_XY = rbf_kernel(X, Y)

    n, m = len(X), len(Y)

    mmd2 = (K_XX.sum() - K_XX.trace()) / (n * (n - 1))
    mmd2 += (K_YY.sum() - K_YY.trace()) / (m * (m - 1))
    mmd2 -= 2 * K_XY.mean()

    return float(max(0, mmd2.cpu().item()))


def compute_coverage_gpu(X_train: torch.Tensor, X_val: torch.Tensor, k: int = 5) -> float:
    """Compute coverage using GPU-accelerated distance computation."""
    max_samples = 1000
    if len(X_train) > max_samples:
        idx = torch.randperm(len(X_train))[:max_samples]
        X_train = X_train[idx]
    if len(X_val) > max_samples:
        idx = torch.randperm(len(X_val))[:max_samples]
        X_val = X_val[idx]

    # Pairwise distances
    train_sq = (X_train ** 2).sum(dim=1, keepdim=True)
    val_sq = (X_val ** 2).sum(dim=1, keepdim=True)

    # Train-train distances for threshold
    train_dists = train_sq + train_sq.T - 2 * torch.mm(X_train, X_train.T)
    train_dists = torch.sqrt(torch.clamp(train_dists, min=0))
    train_dists.fill_diagonal_(float('inf'))

    knn_dists, _ = train_dists.topk(k, dim=1, largest=False)
    threshold = knn_dists[:, -1].quantile(0.95)

    # Val-train distances
    val_train_dists = val_sq + train_sq.T - 2 * torch.mm(X_val, X_train.T)
    val_train_dists = torch.sqrt(torch.clamp(val_train_dists, min=0))

    min_dists, _ = val_train_dists.min(dim=1)
    covered = (min_dists <= threshold).float().mean()

    return float(covered.cpu().item())


def compute_cov_distance_gpu(cov1: torch.Tensor, cov2: torch.Tensor) -> float:
    """Compute Frobenius distance between mean covariance matrices."""
    mean_cov1 = cov1.mean(dim=0)
    mean_cov2 = cov2.mean(dim=0)
    diff = mean_cov1 - mean_cov2
    fro_dist = torch.sqrt((diff ** 2).sum())
    return float(fro_dist.cpu().item() / np.sqrt(mean_cov1.shape[0] * mean_cov1.shape[1]))


# =============================================================================
# Augmentation Functions (NumPy for compatibility)
# =============================================================================

def aug_covariance_expansion(ob: np.ndarray, pcx: np.ndarray, strength: float = 0.3, mode: str = 'random'):
    """Covariance augmentation."""
    n_trials, n_channels, n_time = ob.shape
    ob_aug, pcx_aug = np.zeros_like(ob), np.zeros_like(pcx)

    for i in range(n_trials):
        try:
            ob_cov = np.cov(ob[i])
            pcx_cov = np.cov(pcx[i])
            ob_eigvals, ob_eigvecs = eigh(ob_cov)
            pcx_eigvals, pcx_eigvecs = eigh(pcx_cov)

            if mode == 'expand':
                scale = 1 + strength * np.random.rand(n_channels)
            elif mode == 'shrink':
                scale = 1 - strength * np.random.rand(n_channels)
            else:
                scale = 1 + strength * (2 * np.random.rand(n_channels) - 1)
            scale = np.clip(scale, 0.1, 10.0)

            ob_eigvals_new = np.maximum(ob_eigvals * scale, 1e-6)
            pcx_eigvals_new = np.maximum(pcx_eigvals * scale, 1e-6)

            ob_transform = ob_eigvecs @ np.diag(np.sqrt(ob_eigvals_new / (ob_eigvals + 1e-10))) @ ob_eigvecs.T
            pcx_transform = pcx_eigvecs @ np.diag(np.sqrt(pcx_eigvals_new / (pcx_eigvals + 1e-10))) @ pcx_eigvecs.T

            ob_centered = ob[i] - ob[i].mean(axis=-1, keepdims=True)
            pcx_centered = pcx[i] - pcx[i].mean(axis=-1, keepdims=True)

            ob_aug[i] = ob_transform @ ob_centered + ob[i].mean(axis=-1, keepdims=True)
            pcx_aug[i] = pcx_transform @ pcx_centered + pcx[i].mean(axis=-1, keepdims=True)
        except Exception:
            ob_aug[i], pcx_aug[i] = ob[i], pcx[i]

    return ob_aug, pcx_aug


def aug_cross_session_mixing(ob: np.ndarray, pcx: np.ndarray, session_ids: np.ndarray, alpha: float = 0.3):
    """Cross-session mixing."""
    n_trials = ob.shape[0]
    unique_sessions = np.unique(session_ids)

    if len(unique_sessions) < 2:
        return ob, pcx

    ob_aug, pcx_aug = np.zeros_like(ob), np.zeros_like(pcx)

    for i in range(n_trials):
        my_session = session_ids[i]
        diff_indices = np.where(session_ids != my_session)[0]

        if len(diff_indices) == 0:
            ob_aug[i], pcx_aug[i] = ob[i], pcx[i]
            continue

        lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5
        partner = np.random.choice(diff_indices)
        ob_aug[i] = lam * ob[i] + (1 - lam) * ob[partner]
        pcx_aug[i] = lam * pcx[i] + (1 - lam) * pcx[partner]

    return ob_aug, pcx_aug


def aug_euclidean_alignment(ob: np.ndarray, pcx: np.ndarray):
    """Euclidean Alignment."""
    n_trials, n_channels, n_time = ob.shape

    ob_covs = np.array([np.cov(ob[i]) for i in range(n_trials)])
    pcx_covs = np.array([np.cov(pcx[i]) for i in range(n_trials)])
    ob_mean_cov = ob_covs.mean(axis=0)
    pcx_mean_cov = pcx_covs.mean(axis=0)

    try:
        ob_sqrt_inv = np.linalg.inv(sqrtm(ob_mean_cov + 1e-6 * np.eye(n_channels)))
        pcx_sqrt_inv = np.linalg.inv(sqrtm(pcx_mean_cov + 1e-6 * np.eye(n_channels)))
        ref_sqrt = np.eye(n_channels)
    except Exception:
        return ob, pcx

    ob_align = np.real(ref_sqrt @ ob_sqrt_inv)
    pcx_align = np.real(ref_sqrt @ pcx_sqrt_inv)

    ob_aligned, pcx_aligned = np.zeros_like(ob), np.zeros_like(pcx)
    for i in range(n_trials):
        ob_centered = ob[i] - ob[i].mean(axis=-1, keepdims=True)
        pcx_centered = pcx[i] - pcx[i].mean(axis=-1, keepdims=True)
        ob_aligned[i] = ob_align @ ob_centered + ob[i].mean(axis=-1, keepdims=True)
        pcx_aligned[i] = pcx_align @ pcx_centered + pcx[i].mean(axis=-1, keepdims=True)

    return ob_aligned, pcx_aligned


def aug_amplitude_jitter(ob: np.ndarray, pcx: np.ndarray, scale_range: Tuple[float, float] = (0.7, 1.3)):
    """Per-channel amplitude jittering."""
    n_trials, n_channels, _ = ob.shape
    scales_ob = np.random.uniform(scale_range[0], scale_range[1], (n_trials, n_channels, 1))
    scales_pcx = np.random.uniform(scale_range[0], scale_range[1], (n_trials, n_channels, 1))
    return ob * scales_ob, pcx * scales_pcx


def aug_noise_injection(ob: np.ndarray, pcx: np.ndarray, noise_std: float = 0.1):
    """Add Gaussian noise."""
    ob_std = ob.std(axis=-1, keepdims=True)
    pcx_std = pcx.std(axis=-1, keepdims=True)
    return ob + np.random.randn(*ob.shape) * noise_std * ob_std, pcx + np.random.randn(*pcx.shape) * noise_std * pcx_std


# =============================================================================
# Augmentation Configuration & Evaluation
# =============================================================================

@dataclass
class AugConfig:
    """Augmentation configuration."""
    name: str
    use_ea: bool = False
    use_cov: bool = False
    cov_strength: float = 0.3
    cov_mode: str = 'random'
    use_csm: bool = False
    csm_alpha: float = 0.3
    use_amp: bool = False
    amp_range: Tuple[float, float] = (0.7, 1.3)
    use_noise: bool = False
    noise_std: float = 0.1
    n_passes: int = 3

    def to_dict(self):
        return self.__dict__.copy()


def apply_augmentation(ob, pcx, session_ids, config: AugConfig):
    """Apply augmentation configuration."""
    ob_aug, pcx_aug = ob.copy(), pcx.copy()

    if config.use_ea:
        ob_aug, pcx_aug = aug_euclidean_alignment(ob_aug, pcx_aug)
    if config.use_cov:
        ob_aug, pcx_aug = aug_covariance_expansion(ob_aug, pcx_aug, config.cov_strength, config.cov_mode)
    if config.use_csm:
        ob_aug, pcx_aug = aug_cross_session_mixing(ob_aug, pcx_aug, session_ids, config.csm_alpha)
    if config.use_amp:
        ob_aug, pcx_aug = aug_amplitude_jitter(ob_aug, pcx_aug, config.amp_range)
    if config.use_noise:
        ob_aug, pcx_aug = aug_noise_injection(ob_aug, pcx_aug, config.noise_std)

    return ob_aug, pcx_aug


def evaluate_config(ob_train, pcx_train, ob_val, pcx_val, session_ids_train, val_features, val_cov, config: AugConfig, device):
    """Evaluate a single configuration."""
    all_features = []
    all_covs = []

    for _ in range(config.n_passes):
        ob_aug, pcx_aug = apply_augmentation(ob_train, pcx_train, session_ids_train, config)
        aug_features = extract_features_gpu(ob_aug, pcx_aug, device)
        aug_cov = extract_covariance_gpu(ob_aug, pcx_aug, device)
        all_features.append(aug_features)
        all_covs.append(aug_cov)

    train_features = torch.cat(all_features, dim=0)
    train_cov = torch.cat(all_covs, dim=0)

    mmd = compute_mmd_gpu(train_features, val_features)
    coverage = compute_coverage_gpu(train_features, val_features)
    cov_dist = compute_cov_distance_gpu(train_cov, val_cov)

    score = coverage - 0.3 * mmd - 0.1 * cov_dist

    return {
        'config': config.to_dict(),
        'mmd': mmd,
        'coverage': coverage,
        'cov_distance': cov_dist,
        'score': score,
    }


def generate_search_space() -> List[AugConfig]:
    """Generate search space."""
    configs = []

    # Baseline
    configs.append(AugConfig(name='baseline'))
    configs.append(AugConfig(name='ea_only', use_ea=True))

    # Covariance variations
    for strength in [0.1, 0.2, 0.3, 0.5, 0.7]:
        for mode in ['expand', 'shrink', 'random']:
            configs.append(AugConfig(name=f'cov_{mode}_{strength}', use_cov=True, cov_strength=strength, cov_mode=mode))

    # Cross-session mixing
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
        configs.append(AugConfig(name=f'csm_{alpha}', use_csm=True, csm_alpha=alpha))

    # Combinations
    for strength in [0.2, 0.3, 0.5]:
        configs.append(AugConfig(name=f'ea_cov_{strength}', use_ea=True, use_cov=True, cov_strength=strength))

    for alpha in [0.2, 0.3, 0.4]:
        configs.append(AugConfig(name=f'ea_csm_{alpha}', use_ea=True, use_csm=True, csm_alpha=alpha))

    for strength in [0.2, 0.3]:
        for alpha in [0.2, 0.3]:
            configs.append(AugConfig(name=f'cov_{strength}_csm_{alpha}', use_cov=True, cov_strength=strength, use_csm=True, csm_alpha=alpha))

    # Full combinations
    for strength in [0.2, 0.3, 0.5]:
        for alpha in [0.2, 0.3, 0.4]:
            configs.append(AugConfig(
                name=f'ea_cov_{strength}_csm_{alpha}',
                use_ea=True, use_cov=True, cov_strength=strength, use_csm=True, csm_alpha=alpha
            ))

    # Kitchen sink
    for strength in [0.2, 0.3]:
        for alpha in [0.2, 0.3]:
            for noise in [0.05, 0.1]:
                configs.append(AugConfig(
                    name=f'full_{strength}_{alpha}_{noise}',
                    use_ea=True, use_cov=True, cov_strength=strength, cov_mode='random',
                    use_csm=True, csm_alpha=alpha, use_noise=True, noise_std=noise,
                    use_amp=True, amp_range=(0.8, 1.2)
                ))

    return configs


# =============================================================================
# Visualization
# =============================================================================

def plot_results(results: List[Dict], output_dir: Path):
    """Generate result visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_sorted = sorted(results, key=lambda r: r['score'], reverse=True)
    top_n = min(20, len(results_sorted))
    top_results = results_sorted[:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    names = [r['config']['name'] for r in top_results]
    scores = [r['score'] for r in top_results]
    mmds = [r['mmd'] for r in top_results]
    coverages = [r['coverage'] for r in top_results]

    ax = axes[0, 0]
    bars = ax.barh(range(top_n), scores)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Combined Score (higher is better)')
    ax.set_title(f'Top {top_n} Augmentation Configurations')
    ax.invert_yaxis()
    for bar, score in zip(bars, scores):
        bar.set_color(plt.cm.RdYlGn((score - min(scores)) / (max(scores) - min(scores) + 1e-10)))

    ax = axes[0, 1]
    bars = ax.barh(range(top_n), mmds)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('MMD (lower is better)')
    ax.set_title('Maximum Mean Discrepancy')
    ax.invert_yaxis()
    for bar, mmd in zip(bars, mmds):
        bar.set_color(plt.cm.RdYlGn_r((mmd - min(mmds)) / (max(mmds) - min(mmds) + 1e-10)))

    ax = axes[1, 0]
    bars = ax.barh(range(top_n), coverages)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Coverage (higher is better)')
    ax.set_title('Validation Coverage')
    ax.invert_yaxis()
    for bar, cov in zip(bars, coverages):
        bar.set_color(plt.cm.RdYlGn((cov - min(coverages)) / (max(coverages) - min(coverages) + 1e-10)))

    # Scatter plot
    ax = axes[1, 1]
    all_scores = [r['score'] for r in results]
    all_mmds = [r['mmd'] for r in results]
    all_coverages = [r['coverage'] for r in results]
    scatter = ax.scatter(all_mmds, all_coverages, c=all_scores, cmap='RdYlGn', alpha=0.7, s=50)
    plt.colorbar(scatter, ax=ax, label='Score')
    ax.set_xlabel('MMD (lower is better)')
    ax.set_ylabel('Coverage (higher is better)')
    ax.set_title('All Configurations: MMD vs Coverage')

    plt.tight_layout()
    plt.savefig(output_dir / 'augmentation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'augmentation_comparison.png'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GPU-Accelerated Augmentation Explorer')
    parser.add_argument('--n-val-sessions', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='figures/augmentation_exploration')
    parser.add_argument('--force-recreate-splits', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GPU-ACCELERATED AUGMENTATION EXPLORER")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    data = load_data(args.n_val_sessions, args.force_recreate_splits, args.seed)

    ob, pcx = data['ob'], data['pcx']
    train_idx, val_idx = data['train_idx'], data['val_idx']
    session_ids = data['session_ids']

    ob_train, pcx_train = ob[train_idx], pcx[train_idx]
    ob_val, pcx_val = ob[val_idx], pcx[val_idx]
    session_ids_train = session_ids[train_idx]

    # Extract validation features (once)
    print("\n2. Extracting validation features on GPU...")
    val_features = extract_features_gpu(ob_val, pcx_val, DEVICE)
    val_cov = extract_covariance_gpu(ob_val, pcx_val, DEVICE)
    print(f"   Validation features shape: {val_features.shape}")

    # Generate search space
    print("\n3. Generating search space...")
    configs = generate_search_space()
    print(f"   {len(configs)} configurations to evaluate")

    # Evaluate
    print("\n4. Evaluating configurations...")
    results = []
    for config in tqdm(configs, desc="Evaluating"):
        result = evaluate_config(
            ob_train, pcx_train, ob_val, pcx_val,
            session_ids_train, val_features, val_cov, config, DEVICE
        )
        results.append(result)

        # Clear GPU memory periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Sort and report
    results_sorted = sorted(results, key=lambda r: r['score'], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 10 AUGMENTATION CONFIGURATIONS")
    print("=" * 70)

    for i, result in enumerate(results_sorted[:10]):
        print(f"\n{i+1}. {result['config']['name']}")
        print(f"   Score: {result['score']:.4f} | Coverage: {result['coverage']:.4f} | MMD: {result['mmd']:.4f}")

    # Save results
    print("\n5. Saving results...")
    with open(output_dir / 'augmentation_results.json', 'w') as f:
        json.dump(results_sorted, f, indent=2)

    # Plot
    print("\n6. Generating visualizations...")
    plot_results(results, output_dir)

    # Summary
    best = results_sorted[0]
    baseline = next(r for r in results if r['config']['name'] == 'baseline')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBaseline: Score={baseline['score']:.4f}, Coverage={baseline['coverage']:.4f}, MMD={baseline['mmd']:.4f}")
    print(f"\nBest: {best['config']['name']}")
    print(f"  Score: {best['score']:.4f} (+{best['score'] - baseline['score']:.4f})")
    print(f"  Coverage: {best['coverage']:.4f} (+{best['coverage'] - baseline['coverage']:.4f})")
    print(f"  MMD: {best['mmd']:.4f} ({best['mmd'] - baseline['mmd']:+.4f})")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
