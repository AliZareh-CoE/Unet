#!/usr/bin/env python3
"""GUARANTEED COVERAGE: Find augmentations that cover ALL validation sessions.

The goal is NOT average coverage - it's WORST-CASE coverage.
We need augmented training data to explain EVERY validation session.

Key metrics:
- MIN_SESSION_COVERAGE: Coverage of the WORST validation session (must be high!)
- PER_SESSION_COVERAGE: Coverage for each validation session individually
- COVERAGE_VARIANCE: Low variance = consistent coverage across sessions

Usage:
    python experiment/augmentation_explorer_gpu.py --n-val-sessions 4 --output-dir figures/augmentation_exploration
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm, eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Multi-GPU
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"{'='*70}")
print(f"GUARANTEED COVERAGE MODE: {N_GPUS} GPUs")
print(f"{'='*70}")
if N_GPUS > 0:
    for i in range(N_GPUS):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory/1e9:.1f} GB)")


# =============================================================================
# Data Loading
# =============================================================================

def load_data(n_val_sessions: int = 4, seed: int = 42) -> Dict[str, Any]:
    """Load data with session-based splits."""
    try:
        from experiment.data import load_signals, load_odor_labels, load_session_ids, DATA_PATH, ODOR_CSV_PATH
    except ModuleNotFoundError:
        from data import load_signals, load_odor_labels, load_session_ids, DATA_PATH, ODOR_CSV_PATH

    print("\nLoading neural signals...")
    signals = load_signals(DATA_PATH)
    n_trials = signals.shape[0]
    print(f"  Loaded {n_trials} trials, shape: {signals.shape}")

    ob = signals[:, 0, :, :]
    pcx = signals[:, 1, :, :]

    odors, _ = load_odor_labels(ODOR_CSV_PATH, n_trials)
    session_ids, session_to_idx, idx_to_session = load_session_ids(ODOR_CSV_PATH, num_trials=n_trials)

    unique_sessions = np.unique(session_ids)
    n_sessions = len(unique_sessions)
    print(f"  {n_sessions} unique sessions")

    np.random.seed(seed)
    perm = np.random.permutation(unique_sessions)
    val_session_set = set(perm[:n_val_sessions])
    train_session_set = set(perm[n_val_sessions:])

    train_idx = np.where([s not in val_session_set for s in session_ids])[0]
    val_idx = np.where([s in val_session_set for s in session_ids])[0]

    # Get per-session indices for validation
    val_session_indices = {}
    for sess_id in val_session_set:
        sess_idx = np.where(session_ids == sess_id)[0]
        val_session_indices[idx_to_session[sess_id]] = sess_idx

    print(f"  Train: {len(train_idx)} trials ({len(train_session_set)} sessions)")
    print(f"  Val: {len(val_idx)} trials ({len(val_session_set)} sessions)")
    for sess_name, sess_idx in val_session_indices.items():
        print(f"    - {sess_name}: {len(sess_idx)} trials")

    return {
        'ob': ob, 'pcx': pcx, 'session_ids': session_ids,
        'idx_to_session': idx_to_session, 'train_idx': train_idx, 'val_idx': val_idx,
        'val_session_indices': val_session_indices,
        'train_session_set': train_session_set,
        'val_session_set': val_session_set,
    }


# =============================================================================
# Feature Extraction (GPU)
# =============================================================================

def extract_features_gpu(ob: np.ndarray, pcx: np.ndarray, device: torch.device) -> torch.Tensor:
    """Extract multi-scale features on GPU."""
    n_trials, n_channels, n_time = ob.shape

    ob_t = torch.from_numpy(ob.astype(np.float32)).to(device)
    pcx_t = torch.from_numpy(pcx.astype(np.float32)).to(device)
    combined = torch.cat([ob_t, pcx_t], dim=1)  # (n_trials, 64, n_time)

    features = []

    # 1. Band power (5 bands × 64 channels = 320 features)
    fft = torch.fft.rfft(combined, dim=-1)
    psd = (fft.abs() ** 2) / n_time
    freqs = torch.fft.rfftfreq(n_time, d=1/1000).to(device)

    for f_low, f_high in [(1, 4), (4, 12), (12, 30), (30, 60), (60, 100)]:
        mask = (freqs >= f_low) & (freqs <= f_high)
        features.append(psd[:, :, mask].mean(dim=-1))

    # 2. Temporal stats (4 × 64 = 256 features)
    features.append(combined.mean(dim=-1))
    features.append(combined.std(dim=-1))
    features.append(torch.diff(combined, dim=-1).abs().mean(dim=-1))
    features.append(torch.diff(combined, dim=-1).std(dim=-1))

    # 3. RMS envelope stats (2 × 64 = 128 features)
    rms = F.avg_pool1d(combined.abs(), kernel_size=50, stride=1, padding=25)[:, :, :n_time]
    features.append(rms.mean(dim=-1))
    features.append(rms.std(dim=-1))

    # 4. Covariance upper triangle (2016 features)
    triu_idx = torch.triu_indices(64, 64, offset=1, device=device)
    cov_feats = []
    for start in range(0, n_trials, 200):
        end = min(start + 200, n_trials)
        batch = combined[start:end]
        batch_c = batch - batch.mean(dim=-1, keepdim=True)
        cov = torch.bmm(batch_c, batch_c.transpose(1, 2)) / n_time
        cov_feats.append(cov[:, triu_idx[0], triu_idx[1]])
    features.append(torch.cat(cov_feats, dim=0))

    # Concatenate and normalize
    all_feats = torch.cat(features, dim=1)
    mean = all_feats.mean(dim=0, keepdim=True)
    std = all_feats.std(dim=0, keepdim=True) + 1e-8
    return (all_feats - mean) / std


# =============================================================================
# GUARANTEED COVERAGE METRICS
# =============================================================================

def compute_per_session_coverage(
    train_features: torch.Tensor,
    val_features_dict: Dict[str, torch.Tensor],
    k: int = 5,
    percentile: float = 95.0,
) -> Dict[str, float]:
    """Compute coverage for EACH validation session separately.

    This is the key metric - we need ALL sessions to have high coverage.
    """
    # Compute threshold from training data k-NN distances
    max_train = min(1000, len(train_features))
    train_sub = train_features[torch.randperm(len(train_features))[:max_train]]

    train_sq = (train_sub ** 2).sum(dim=1, keepdim=True)
    train_dists = torch.sqrt(torch.clamp(train_sq + train_sq.T - 2 * torch.mm(train_sub, train_sub.T), min=0))
    train_dists.fill_diagonal_(float('inf'))
    knn_dists = train_dists.topk(k, dim=1, largest=False)[0][:, -1]
    threshold = knn_dists.quantile(percentile / 100.0)

    # Compute coverage for each validation session
    coverages = {}
    for sess_name, val_feats in val_features_dict.items():
        val_sq = (val_feats ** 2).sum(dim=1, keepdim=True)
        train_sq_full = (train_features ** 2).sum(dim=1, keepdim=True)

        # Subsample train if needed for memory
        max_compare = min(2000, len(train_features))
        train_sub_idx = torch.randperm(len(train_features))[:max_compare]
        train_sub_feats = train_features[train_sub_idx]
        train_sq_sub = (train_sub_feats ** 2).sum(dim=1, keepdim=True)

        val_train_dists = torch.sqrt(torch.clamp(
            val_sq + train_sq_sub.T - 2 * torch.mm(val_feats, train_sub_feats.T), min=0
        ))
        min_dists = val_train_dists.min(dim=1)[0]
        coverage = (min_dists <= threshold).float().mean().item()
        coverages[sess_name] = coverage

    return coverages


def compute_mmd_multi_scale(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Multi-scale MMD with multiple kernel bandwidths."""
    max_samples = 500
    if len(X) > max_samples:
        X = X[torch.randperm(len(X))[:max_samples]]
    if len(Y) > max_samples:
        Y = Y[torch.randperm(len(Y))[:max_samples]]

    # Multiple bandwidths for multi-scale comparison
    base_gamma = 1.0 / X.shape[1]
    gammas = [base_gamma * 0.1, base_gamma, base_gamma * 10]

    total_mmd = 0.0
    for gamma in gammas:
        def rbf(A, B):
            A_sq = (A ** 2).sum(dim=1, keepdim=True)
            B_sq = (B ** 2).sum(dim=1, keepdim=True)
            return torch.exp(-gamma * (A_sq + B_sq.T - 2 * torch.mm(A, B.T)))

        K_XX, K_YY, K_XY = rbf(X, X), rbf(Y, Y), rbf(X, Y)
        n, m = len(X), len(Y)
        mmd2 = (K_XX.sum() - K_XX.trace()) / (n * (n - 1))
        mmd2 += (K_YY.sum() - K_YY.trace()) / (m * (m - 1))
        mmd2 -= 2 * K_XY.mean()
        total_mmd += max(0, mmd2.item())

    return total_mmd / len(gammas)


# =============================================================================
# Augmentation Functions
# =============================================================================

def aug_covariance(ob, pcx, strength=0.3, mode='random'):
    """Covariance eigenvalue perturbation."""
    n_trials, n_ch, _ = ob.shape
    ob_aug, pcx_aug = np.zeros_like(ob), np.zeros_like(pcx)

    for i in range(n_trials):
        try:
            for arr, aug_arr in [(ob, ob_aug), (pcx, pcx_aug)]:
                eigvals, eigvecs = eigh(np.cov(arr[i]))
                if mode == 'expand':
                    scale = 1 + strength * np.random.rand(n_ch)
                elif mode == 'shrink':
                    scale = 1 - strength * np.random.rand(n_ch)
                else:
                    scale = 1 + strength * (2 * np.random.rand(n_ch) - 1)
                scale = np.clip(scale, 0.1, 10.0)
                eigvals_new = np.maximum(eigvals * scale, 1e-6)
                T = eigvecs @ np.diag(np.sqrt(eigvals_new / (eigvals + 1e-10))) @ eigvecs.T
                centered = arr[i] - arr[i].mean(axis=-1, keepdims=True)
                aug_arr[i] = T @ centered + arr[i].mean(axis=-1, keepdims=True)
        except:
            ob_aug[i], pcx_aug[i] = ob[i], pcx[i]
    return ob_aug, pcx_aug


def aug_cross_session_mix(ob, pcx, session_ids, alpha=0.3):
    """Mixup between sessions."""
    n_trials = ob.shape[0]
    unique_sess = np.unique(session_ids)
    if len(unique_sess) < 2:
        return ob, pcx

    ob_aug, pcx_aug = np.zeros_like(ob), np.zeros_like(pcx)
    for i in range(n_trials):
        diff_idx = np.where(session_ids != session_ids[i])[0]
        if len(diff_idx) == 0:
            ob_aug[i], pcx_aug[i] = ob[i], pcx[i]
        else:
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5
            partner = np.random.choice(diff_idx)
            ob_aug[i] = lam * ob[i] + (1 - lam) * ob[partner]
            pcx_aug[i] = lam * pcx[i] + (1 - lam) * pcx[partner]
    return ob_aug, pcx_aug


def aug_euclidean_align(ob, pcx):
    """Whiten to identity covariance."""
    n_trials, n_ch, _ = ob.shape
    try:
        ob_cov = np.array([np.cov(ob[i]) for i in range(n_trials)]).mean(axis=0)
        pcx_cov = np.array([np.cov(pcx[i]) for i in range(n_trials)]).mean(axis=0)
        ob_T = np.real(np.linalg.inv(sqrtm(ob_cov + 1e-6 * np.eye(n_ch))))
        pcx_T = np.real(np.linalg.inv(sqrtm(pcx_cov + 1e-6 * np.eye(n_ch))))
    except:
        return ob, pcx

    ob_out, pcx_out = np.zeros_like(ob), np.zeros_like(pcx)
    for i in range(n_trials):
        ob_c = ob[i] - ob[i].mean(axis=-1, keepdims=True)
        pcx_c = pcx[i] - pcx[i].mean(axis=-1, keepdims=True)
        ob_out[i] = ob_T @ ob_c + ob[i].mean(axis=-1, keepdims=True)
        pcx_out[i] = pcx_T @ pcx_c + pcx[i].mean(axis=-1, keepdims=True)
    return ob_out, pcx_out


def aug_amplitude(ob, pcx, scale_range=(0.7, 1.3)):
    """Per-channel amplitude jitter."""
    n_trials, n_ch, _ = ob.shape
    s_ob = np.random.uniform(scale_range[0], scale_range[1], (n_trials, n_ch, 1))
    s_pcx = np.random.uniform(scale_range[0], scale_range[1], (n_trials, n_ch, 1))
    return ob * s_ob, pcx * s_pcx


def aug_noise(ob, pcx, std=0.1):
    """Gaussian noise."""
    return (ob + np.random.randn(*ob.shape) * std * ob.std(axis=-1, keepdims=True),
            pcx + np.random.randn(*pcx.shape) * std * pcx.std(axis=-1, keepdims=True))


def aug_time_shift(ob, pcx, max_shift=0.1):
    """Circular time shift."""
    n_trials, _, n_time = ob.shape
    shifts = np.random.randint(-int(n_time * max_shift), int(n_time * max_shift) + 1, n_trials)
    ob_aug = np.array([np.roll(ob[i], shifts[i], axis=-1) for i in range(n_trials)])
    pcx_aug = np.array([np.roll(pcx[i], shifts[i], axis=-1) for i in range(n_trials)])
    return ob_aug, pcx_aug


def aug_channel_dropout(ob, pcx, p=0.1):
    """Random channel zeroing."""
    mask = (np.random.rand(ob.shape[0], ob.shape[1], 1) > p).astype(np.float32)
    return ob * mask, pcx * mask


def aug_freq_mask(ob, pcx, n_masks=2, max_width=20):
    """Zero out random frequency bands."""
    n_trials, _, n_time = ob.shape
    ob_aug, pcx_aug = ob.copy(), pcx.copy()
    freqs = np.fft.rfftfreq(n_time, d=1/1000)

    for i in range(n_trials):
        ob_fft = np.fft.rfft(ob_aug[i], axis=-1)
        pcx_fft = np.fft.rfft(pcx_aug[i], axis=-1)
        for _ in range(n_masks):
            f_start = np.random.randint(1, 80)
            f_width = np.random.randint(5, max_width)
            mask = (freqs >= f_start) & (freqs <= f_start + f_width)
            ob_fft[:, mask] = 0
            pcx_fft[:, mask] = 0
        ob_aug[i] = np.fft.irfft(ob_fft, n=n_time, axis=-1)
        pcx_aug[i] = np.fft.irfft(pcx_fft, n=n_time, axis=-1)
    return ob_aug, pcx_aug


def aug_dc_shift(ob, pcx, max_shift=0.3):
    """Random per-channel DC offset."""
    n_trials, n_ch, _ = ob.shape
    ob_std = ob.std()
    pcx_std = pcx.std()
    ob_shift = np.random.uniform(-max_shift, max_shift, (n_trials, n_ch, 1)) * ob_std
    pcx_shift = np.random.uniform(-max_shift, max_shift, (n_trials, n_ch, 1)) * pcx_std
    return ob + ob_shift, pcx + pcx_shift


# =============================================================================
# Augmentation Config
# =============================================================================

@dataclass
class AugConfig:
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
    use_time_shift: bool = False
    shift_max: float = 0.1
    use_dropout: bool = False
    dropout_p: float = 0.1
    use_freq_mask: bool = False
    use_dc_shift: bool = False
    dc_max: float = 0.3
    n_passes: int = 3  # More passes for stability

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def apply_augmentation(ob, pcx, session_ids, config: AugConfig):
    """Apply augmentation pipeline."""
    ob_aug, pcx_aug = ob.copy(), pcx.copy()

    if config.use_ea:
        ob_aug, pcx_aug = aug_euclidean_align(ob_aug, pcx_aug)
    if config.use_cov:
        ob_aug, pcx_aug = aug_covariance(ob_aug, pcx_aug, config.cov_strength, config.cov_mode)
    if config.use_csm:
        ob_aug, pcx_aug = aug_cross_session_mix(ob_aug, pcx_aug, session_ids, config.csm_alpha)
    if config.use_amp:
        ob_aug, pcx_aug = aug_amplitude(ob_aug, pcx_aug, config.amp_range)
    if config.use_noise:
        ob_aug, pcx_aug = aug_noise(ob_aug, pcx_aug, config.noise_std)
    if config.use_time_shift:
        ob_aug, pcx_aug = aug_time_shift(ob_aug, pcx_aug, config.shift_max)
    if config.use_dropout:
        ob_aug, pcx_aug = aug_channel_dropout(ob_aug, pcx_aug, config.dropout_p)
    if config.use_freq_mask:
        ob_aug, pcx_aug = aug_freq_mask(ob_aug, pcx_aug)
    if config.use_dc_shift:
        ob_aug, pcx_aug = aug_dc_shift(ob_aug, pcx_aug, config.dc_max)

    return ob_aug, pcx_aug


# =============================================================================
# Search Space - AGGRESSIVE
# =============================================================================

def generate_search_space() -> List[AugConfig]:
    """INSANE 1024+ CONFIGURATION SPACE - RAPE THIS DATA LIKE A HORNY HORSE."""
    configs = []

    # ==========================================================================
    # BASELINE (2)
    # ==========================================================================
    configs.append(AugConfig(name='baseline'))
    configs.append(AugConfig(name='ea_only', use_ea=True))

    # ==========================================================================
    # SINGLE AUGMENTATIONS - FINE GRAINED (100+)
    # ==========================================================================

    # Covariance: 12 strengths × 3 modes = 36
    for s in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for m in ['expand', 'shrink', 'random']:
            configs.append(AugConfig(name=f'cov_{m}_{s}', use_cov=True, cov_strength=s, cov_mode=m))

    # Cross-session mixing: 12 alphas
    for a in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        configs.append(AugConfig(name=f'csm_{a}', use_csm=True, csm_alpha=a))

    # Noise: 10 levels
    for n in [0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3]:
        configs.append(AugConfig(name=f'noise_{n}', use_noise=True, noise_std=n))

    # Amplitude: 8 ranges
    for lo in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        hi = 2.0 - lo
        configs.append(AugConfig(name=f'amp_{lo}', use_amp=True, amp_range=(lo, hi)))

    # Time shift: 6 levels
    for ts in [0.02, 0.05, 0.08, 0.1, 0.15, 0.2]:
        configs.append(AugConfig(name=f'tshift_{ts}', use_time_shift=True, shift_max=ts))

    # Dropout: 6 levels
    for dp in [0.02, 0.05, 0.08, 0.1, 0.15, 0.2]:
        configs.append(AugConfig(name=f'dropout_{dp}', use_dropout=True, dropout_p=dp))

    # DC shift: 6 levels
    for dc in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        configs.append(AugConfig(name=f'dc_{dc}', use_dc_shift=True, dc_max=dc))

    # Freq mask
    configs.append(AugConfig(name='freqmask', use_freq_mask=True))

    # ==========================================================================
    # DOUBLE COMBINATIONS (200+)
    # ==========================================================================

    # EA + Cov: 8 strengths × 3 modes = 24
    for s in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for m in ['expand', 'shrink', 'random']:
            configs.append(AugConfig(name=f'ea_cov_{m}_{s}', use_ea=True, use_cov=True, cov_strength=s, cov_mode=m))

    # EA + CSM: 10 alphas
    for a in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]:
        configs.append(AugConfig(name=f'ea_csm_{a}', use_ea=True, use_csm=True, csm_alpha=a))

    # Cov + CSM: 6 × 6 = 36
    for s in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        for a in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            configs.append(AugConfig(name=f'cov_{s}_csm_{a}', use_cov=True, cov_strength=s, use_csm=True, csm_alpha=a))

    # Cov + Noise: 6 × 5 = 30
    for s in [0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
        for n in [0.05, 0.08, 0.1, 0.15, 0.2]:
            configs.append(AugConfig(name=f'cov_{s}_n_{n}', use_cov=True, cov_strength=s, use_noise=True, noise_std=n))

    # CSM + Noise: 6 × 5 = 30
    for a in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        for n in [0.05, 0.08, 0.1, 0.15, 0.2]:
            configs.append(AugConfig(name=f'csm_{a}_n_{n}', use_csm=True, csm_alpha=a, use_noise=True, noise_std=n))

    # CSM + Amp: 6 × 4 = 24
    for a in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        for lo in [0.6, 0.7, 0.8, 0.9]:
            configs.append(AugConfig(name=f'csm_{a}_amp_{lo}', use_csm=True, csm_alpha=a, use_amp=True, amp_range=(lo, 2-lo)))

    # ==========================================================================
    # TRIPLE COMBINATIONS - EA + COV + CSM (100+)
    # ==========================================================================

    # EA + Cov + CSM: 6 × 6 = 36
    for s in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        for a in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            configs.append(AugConfig(
                name=f'ea_cov_{s}_csm_{a}',
                use_ea=True, use_cov=True, cov_strength=s, use_csm=True, csm_alpha=a
            ))

    # EA + Cov + Noise: 5 × 5 = 25
    for s in [0.2, 0.25, 0.3, 0.4, 0.5]:
        for n in [0.05, 0.08, 0.1, 0.15, 0.2]:
            configs.append(AugConfig(
                name=f'ea_cov_{s}_n_{n}',
                use_ea=True, use_cov=True, cov_strength=s, use_noise=True, noise_std=n
            ))

    # EA + CSM + Noise: 5 × 5 = 25
    for a in [0.2, 0.25, 0.3, 0.4, 0.5]:
        for n in [0.05, 0.08, 0.1, 0.15, 0.2]:
            configs.append(AugConfig(
                name=f'ea_csm_{a}_n_{n}',
                use_ea=True, use_csm=True, csm_alpha=a, use_noise=True, noise_std=n
            ))

    # Cov + CSM + Noise: 4 × 4 × 4 = 64
    for s in [0.2, 0.3, 0.4, 0.5]:
        for a in [0.2, 0.3, 0.4, 0.5]:
            for n in [0.05, 0.1, 0.15, 0.2]:
                configs.append(AugConfig(
                    name=f'cov_{s}_csm_{a}_n_{n}',
                    use_cov=True, cov_strength=s, use_csm=True, csm_alpha=a, use_noise=True, noise_std=n
                ))

    # ==========================================================================
    # QUAD COMBINATIONS - EA + COV + CSM + NOISE (150+)
    # ==========================================================================

    # EA + Cov + CSM + Noise: 5 × 5 × 4 = 100
    for s in [0.15, 0.2, 0.3, 0.4, 0.5]:
        for a in [0.15, 0.2, 0.3, 0.4, 0.5]:
            for n in [0.05, 0.1, 0.15, 0.2]:
                configs.append(AugConfig(
                    name=f'ea_cov_{s}_csm_{a}_n_{n}',
                    use_ea=True, use_cov=True, cov_strength=s,
                    use_csm=True, csm_alpha=a, use_noise=True, noise_std=n
                ))

    # EA + Cov + CSM + Amp: 4 × 4 × 4 = 64
    for s in [0.2, 0.3, 0.4, 0.5]:
        for a in [0.2, 0.3, 0.4, 0.5]:
            for lo in [0.6, 0.7, 0.8, 0.9]:
                configs.append(AugConfig(
                    name=f'ea_cov_{s}_csm_{a}_amp_{lo}',
                    use_ea=True, use_cov=True, cov_strength=s,
                    use_csm=True, csm_alpha=a, use_amp=True, amp_range=(lo, 2-lo)
                ))

    # ==========================================================================
    # QUINT COMBINATIONS - 5 AUGMENTATIONS (150+)
    # ==========================================================================

    # EA + Cov + CSM + Noise + Amp: 4 × 4 × 3 × 3 = 144
    for s in [0.2, 0.3, 0.4, 0.5]:
        for a in [0.2, 0.3, 0.4, 0.5]:
            for n in [0.05, 0.1, 0.15]:
                for lo in [0.7, 0.8, 0.9]:
                    configs.append(AugConfig(
                        name=f'q5_{s}_{a}_{n}_{lo}',
                        use_ea=True, use_cov=True, cov_strength=s,
                        use_csm=True, csm_alpha=a,
                        use_noise=True, noise_std=n,
                        use_amp=True, amp_range=(lo, 2-lo)
                    ))

    # ==========================================================================
    # FULL COMBOS WITH TIME SHIFT (100+)
    # ==========================================================================

    # EA + Cov + CSM + Noise + Amp + TimeShift: 3 × 3 × 3 × 3 = 81
    for s in [0.2, 0.3, 0.4]:
        for a in [0.2, 0.3, 0.4]:
            for n in [0.05, 0.1, 0.15]:
                for ts in [0.05, 0.1, 0.15]:
                    configs.append(AugConfig(
                        name=f'full_{s}_{a}_{n}_{ts}',
                        use_ea=True, use_cov=True, cov_strength=s,
                        use_csm=True, csm_alpha=a,
                        use_noise=True, noise_std=n,
                        use_amp=True, amp_range=(0.7, 1.3),
                        use_time_shift=True, shift_max=ts
                    ))

    # ==========================================================================
    # ULTRA COMBOS WITH DC SHIFT (50+)
    # ==========================================================================

    # EA + Cov + CSM + Noise + Amp + DC: 3 × 3 × 3 × 3 = 81
    for s in [0.2, 0.3, 0.4]:
        for a in [0.2, 0.3, 0.4]:
            for n in [0.05, 0.1, 0.15]:
                for dc in [0.1, 0.2, 0.3]:
                    configs.append(AugConfig(
                        name=f'ultra_{s}_{a}_{n}_{dc}',
                        use_ea=True, use_cov=True, cov_strength=s,
                        use_csm=True, csm_alpha=a,
                        use_noise=True, noise_std=n,
                        use_amp=True, amp_range=(0.7, 1.3),
                        use_dc_shift=True, dc_max=dc
                    ))

    # ==========================================================================
    # MEGA COMBOS WITH DROPOUT (50+)
    # ==========================================================================

    # EA + Cov + CSM + Noise + Dropout: 3 × 3 × 3 × 3 = 81
    for s in [0.2, 0.3, 0.4]:
        for a in [0.2, 0.3, 0.4]:
            for n in [0.05, 0.1, 0.15]:
                for dp in [0.05, 0.1, 0.15]:
                    configs.append(AugConfig(
                        name=f'mega_{s}_{a}_{n}_{dp}',
                        use_ea=True, use_cov=True, cov_strength=s,
                        use_csm=True, csm_alpha=a,
                        use_noise=True, noise_std=n,
                        use_dropout=True, dropout_p=dp
                    ))

    # ==========================================================================
    # HYPER COMBOS WITH FREQ MASK (50+)
    # ==========================================================================

    # EA + Cov + CSM + Noise + FreqMask: 4 × 4 × 3 = 48
    for s in [0.2, 0.3, 0.4, 0.5]:
        for a in [0.2, 0.3, 0.4, 0.5]:
            for n in [0.05, 0.1, 0.15]:
                configs.append(AugConfig(
                    name=f'hyper_{s}_{a}_{n}',
                    use_ea=True, use_cov=True, cov_strength=s,
                    use_csm=True, csm_alpha=a,
                    use_noise=True, noise_std=n,
                    use_amp=True, amp_range=(0.7, 1.3),
                    use_freq_mask=True
                ))

    # ==========================================================================
    # INSANE - ALL THE FUCKING THINGS (50+)
    # ==========================================================================

    # Everything combined: 4 × 4 × 3 = 48
    for s in [0.2, 0.3, 0.4, 0.5]:
        for a in [0.2, 0.3, 0.4, 0.5]:
            for n in [0.05, 0.1, 0.15]:
                configs.append(AugConfig(
                    name=f'insane_{s}_{a}_{n}',
                    use_ea=True, use_cov=True, cov_strength=s, cov_mode='random',
                    use_csm=True, csm_alpha=a,
                    use_noise=True, noise_std=n,
                    use_amp=True, amp_range=(0.6, 1.4),
                    use_time_shift=True, shift_max=0.1,
                    use_dropout=True, dropout_p=0.1,
                    use_freq_mask=True,
                    use_dc_shift=True, dc_max=0.2
                ))

    # ==========================================================================
    # GODMODE - EXTREME VARIATIONS (30+)
    # ==========================================================================

    # High covariance + high CSM
    for s in [0.6, 0.7, 0.8]:
        for a in [0.5, 0.55, 0.6]:
            configs.append(AugConfig(
                name=f'god_high_{s}_{a}',
                use_ea=True, use_cov=True, cov_strength=s, cov_mode='random',
                use_csm=True, csm_alpha=a,
                use_noise=True, noise_std=0.15,
                use_amp=True, amp_range=(0.5, 1.5)
            ))

    # Low covariance + high everything else
    for s in [0.1, 0.15]:
        for a in [0.4, 0.5, 0.6]:
            for n in [0.15, 0.2, 0.25]:
                configs.append(AugConfig(
                    name=f'god_low_{s}_{a}_{n}',
                    use_ea=True, use_cov=True, cov_strength=s, cov_mode='expand',
                    use_csm=True, csm_alpha=a,
                    use_noise=True, noise_std=n,
                    use_amp=True, amp_range=(0.5, 1.5),
                    use_time_shift=True, shift_max=0.15
                ))

    # Shrink-only covariance variations
    for s in [0.3, 0.4, 0.5, 0.6]:
        for a in [0.3, 0.4, 0.5]:
            configs.append(AugConfig(
                name=f'shrink_{s}_{a}',
                use_ea=True, use_cov=True, cov_strength=s, cov_mode='shrink',
                use_csm=True, csm_alpha=a,
                use_noise=True, noise_std=0.1
            ))

    # Expand-only covariance variations
    for s in [0.3, 0.4, 0.5, 0.6]:
        for a in [0.3, 0.4, 0.5]:
            configs.append(AugConfig(
                name=f'expand_{s}_{a}',
                use_ea=True, use_cov=True, cov_strength=s, cov_mode='expand',
                use_csm=True, csm_alpha=a,
                use_noise=True, noise_std=0.1
            ))

    # No EA variations (test if EA helps or hurts)
    for s in [0.2, 0.3, 0.4, 0.5]:
        for a in [0.2, 0.3, 0.4, 0.5]:
            for n in [0.05, 0.1, 0.15]:
                configs.append(AugConfig(
                    name=f'noea_{s}_{a}_{n}',
                    use_ea=False, use_cov=True, cov_strength=s,
                    use_csm=True, csm_alpha=a,
                    use_noise=True, noise_std=n,
                    use_amp=True, amp_range=(0.7, 1.3)
                ))

    print(f"\n  🔥🔥🔥 GENERATED {len(configs)} CONFIGURATIONS - ABSOLUTELY INSANE! 🔥🔥🔥")
    return configs


# =============================================================================
# Evaluation with GUARANTEED metrics
# =============================================================================

def evaluate_config(
    config: AugConfig,
    ob_train: np.ndarray,
    pcx_train: np.ndarray,
    ob_val: np.ndarray,
    pcx_val: np.ndarray,
    session_ids_train: np.ndarray,
    val_session_indices: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate with per-session coverage guarantee."""

    try:
        # Multiple augmentation passes
        all_features = []
        for _ in range(config.n_passes):
            ob_aug, pcx_aug = apply_augmentation(ob_train, pcx_train, session_ids_train, config)
            feats = extract_features_gpu(ob_aug, pcx_aug, device)
            all_features.append(feats)

        train_features = torch.cat(all_features, dim=0)

        # Extract features for each validation session
        val_features_dict = {}
        for sess_name, sess_idx in val_session_indices.items():
            ob_sess = ob_val[np.isin(np.arange(len(ob_val)), sess_idx - val_session_indices[list(val_session_indices.keys())[0]][0] + np.arange(len(sess_idx)))]
            # Actually we need to index correctly
            pass

        # Simpler: just get val features per session from original indices
        val_features_all = extract_features_gpu(ob_val, pcx_val, device)

        # Split val features by session
        val_features_dict = {}
        start_idx = 0
        for sess_name, sess_idx in val_session_indices.items():
            n_sess = len(sess_idx)
            val_features_dict[sess_name] = val_features_all[start_idx:start_idx + n_sess]
            start_idx += n_sess

        # Compute per-session coverage
        per_session_cov = compute_per_session_coverage(train_features, val_features_dict)

        # Key metrics
        min_coverage = min(per_session_cov.values())
        mean_coverage = np.mean(list(per_session_cov.values()))
        coverage_std = np.std(list(per_session_cov.values()))

        # Overall MMD
        mmd = compute_mmd_multi_scale(train_features, val_features_all)

        # Score: prioritize WORST-CASE coverage
        # High min_coverage + low variance + low MMD
        score = min_coverage * 100 - coverage_std * 50 - mmd * 10

        # Cleanup
        del train_features, val_features_all, all_features
        torch.cuda.empty_cache()

        return {
            'config': config.to_dict(),
            'min_coverage': min_coverage,
            'mean_coverage': mean_coverage,
            'coverage_std': coverage_std,
            'per_session_coverage': per_session_cov,
            'mmd': mmd,
            'score': score,
        }

    except Exception as e:
        return {
            'config': config.to_dict(),
            'min_coverage': 0.0,
            'mean_coverage': 0.0,
            'coverage_std': 1.0,
            'per_session_coverage': {},
            'mmd': 999.0,
            'score': -999.0,
            'error': str(e),
        }


# =============================================================================
# Visualization
# =============================================================================

def plot_results(results: List[Dict], output_dir: Path):
    """Visualize with focus on per-session coverage."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_sorted = sorted(results, key=lambda r: r['score'], reverse=True)
    top_n = min(25, len(results_sorted))
    top = results_sorted[:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    names = [r['config']['name'][:25] for r in top]
    min_covs = [r['min_coverage'] for r in top]
    mean_covs = [r['mean_coverage'] for r in top]
    scores = [r['score'] for r in top]

    # Min coverage (most important!)
    ax = axes[0, 0]
    colors = [plt.cm.RdYlGn(c) for c in min_covs]
    ax.barh(range(top_n), min_covs, color=colors)
    ax.axvline(x=0.95, color='red', linestyle='--', label='95% threshold')
    ax.axvline(x=0.99, color='green', linestyle='--', label='99% threshold')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('MIN Session Coverage (MUST BE HIGH!)')
    ax.set_title('Worst-Case Coverage per Config')
    ax.legend()
    ax.invert_yaxis()

    # Mean coverage
    ax = axes[0, 1]
    colors = [plt.cm.RdYlGn(c) for c in mean_covs]
    ax.barh(range(top_n), mean_covs, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Mean Coverage')
    ax.set_title('Average Coverage')
    ax.invert_yaxis()

    # Score
    ax = axes[1, 0]
    score_norm = [(s - min(scores)) / (max(scores) - min(scores) + 1e-10) for s in scores]
    colors = [plt.cm.RdYlGn(s) for s in score_norm]
    ax.barh(range(top_n), scores, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Score (higher = better)')
    ax.set_title('Combined Score')
    ax.invert_yaxis()

    # Scatter: min vs mean coverage
    ax = axes[1, 1]
    all_min = [r['min_coverage'] for r in results]
    all_mean = [r['mean_coverage'] for r in results]
    all_scores = [r['score'] for r in results]
    scatter = ax.scatter(all_mean, all_min, c=all_scores, cmap='RdYlGn', alpha=0.6, s=40)
    plt.colorbar(scatter, ax=ax, label='Score')
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean Coverage')
    ax.set_ylabel('MIN Coverage (guarantee)')
    ax.set_title('Coverage: Mean vs Worst-Case')

    # Mark top 5
    for i, r in enumerate(results_sorted[:5]):
        ax.annotate(str(i+1), (r['mean_coverage'], r['min_coverage']), fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'guaranteed_coverage_results.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Per-session coverage heatmap for top configs
    fig, ax = plt.subplots(figsize=(14, 10))

    top_10 = results_sorted[:10]
    sessions = list(top_10[0]['per_session_coverage'].keys()) if top_10[0]['per_session_coverage'] else []

    if sessions:
        matrix = []
        for r in top_10:
            row = [r['per_session_coverage'].get(s, 0) for s in sessions]
            matrix.append(row)
        matrix = np.array(matrix)

        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(sessions)))
        ax.set_xticklabels(sessions, rotation=45, ha='right')
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels([r['config']['name'][:20] for r in top_10])
        ax.set_xlabel('Validation Session')
        ax.set_ylabel('Augmentation Config')
        ax.set_title('Per-Session Coverage Heatmap (Top 10 Configs)')
        plt.colorbar(im, ax=ax, label='Coverage')

        # Add text annotations
        for i in range(len(top_10)):
            for j in range(len(sessions)):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=8,
                       color='white' if matrix[i,j] < 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(output_dir / 'per_session_coverage_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {output_dir}/guaranteed_coverage_results.png")
    print(f"Saved: {output_dir}/per_session_coverage_heatmap.png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Guaranteed Coverage Augmentation Explorer')
    parser.add_argument('--n-val-sessions', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='figures/augmentation_exploration')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load data
    print("\n1. Loading data...")
    data = load_data(args.n_val_sessions, args.seed)

    ob, pcx = data['ob'], data['pcx']
    train_idx, val_idx = data['train_idx'], data['val_idx']
    session_ids = data['session_ids']
    val_session_indices = data['val_session_indices']

    ob_train, pcx_train = ob[train_idx], pcx[train_idx]
    ob_val, pcx_val = ob[val_idx], pcx[val_idx]
    session_ids_train = session_ids[train_idx]

    # Generate configs
    print("\n2. Generating search space...")
    configs = generate_search_space()

    # Evaluate
    print(f"\n3. Evaluating {len(configs)} configurations...")
    results = []

    for config in tqdm(configs, desc="Evaluating"):
        result = evaluate_config(
            config, ob_train, pcx_train, ob_val, pcx_val,
            session_ids_train, val_session_indices, device
        )
        results.append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Sort by score
    results_sorted = sorted(results, key=lambda r: r['score'], reverse=True)

    # Print results
    print("\n" + "="*70)
    print("TOP 15 CONFIGURATIONS (by guaranteed coverage)")
    print("="*70)

    for i, r in enumerate(results_sorted[:15]):
        print(f"\n{i+1}. {r['config']['name']}")
        print(f"   MIN Coverage: {r['min_coverage']:.4f} | Mean: {r['mean_coverage']:.4f} | Std: {r['coverage_std']:.4f}")
        print(f"   MMD: {r['mmd']:.4f} | Score: {r['score']:.2f}")
        if r['per_session_coverage']:
            cov_str = ', '.join([f"{k}:{v:.2f}" for k,v in r['per_session_coverage'].items()])
            print(f"   Per-session: {cov_str}")

    # Save
    print("\n4. Saving results...")
    with open(output_dir / 'guaranteed_coverage_results.json', 'w') as f:
        json.dump(results_sorted, f, indent=2, default=str)

    # Plot
    print("\n5. Generating visualizations...")
    plot_results(results, output_dir)

    # Summary
    best = results_sorted[0]
    baseline = next((r for r in results if r['config']['name'] == 'baseline'), results_sorted[-1])

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBaseline: MIN={baseline['min_coverage']:.4f}, Mean={baseline['mean_coverage']:.4f}")
    print(f"\n🏆 BEST: {best['config']['name']}")
    print(f"   MIN Coverage: {best['min_coverage']:.4f} (+{best['min_coverage'] - baseline['min_coverage']:.4f})")
    print(f"   Mean Coverage: {best['mean_coverage']:.4f}")
    print(f"   Coverage Variance: {best['coverage_std']:.4f}")

    if best['min_coverage'] >= 0.95:
        print("\n✅ GUARANTEE ACHIEVED: Worst-case session coverage >= 95%")
    elif best['min_coverage'] >= 0.90:
        print("\n⚠️  CLOSE: Worst-case coverage >= 90% but < 95%")
    else:
        print("\n❌ NEEDS WORK: Worst-case coverage < 90%")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
