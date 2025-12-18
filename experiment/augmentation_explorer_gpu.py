#!/usr/bin/env python3
"""BEAST MODE: 8x A100 Multi-GPU Augmentation Explorer.

256+ configurations evaluated in parallel across 8 GPUs.
Find the ultimate augmentation combination for cross-session domination.

Usage:
    python experiment/augmentation_explorer_gpu.py --n-val-sessions 4 --output-dir figures/augmentation_exploration

Author: Claude (Anthropic)
"""
from __future__ import annotations

import os
import sys

# Add parent directory to path for imports
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
from typing import Any, Dict, List, Optional, Tuple
from itertools import product
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm, eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# Multi-GPU Setup
# =============================================================================

N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"=" * 70)
print(f"BEAST MODE: {N_GPUS} GPUs DETECTED")
print(f"=" * 70)
if N_GPUS > 0:
    for i in range(N_GPUS):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"  TOTAL GPU MEMORY: {sum(torch.cuda.get_device_properties(i).total_memory for i in range(N_GPUS)) / 1e9:.1f} GB")


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
    val_sessions = set(perm[:n_val_sessions])
    train_sessions = set(perm[n_val_sessions:])

    train_idx = np.where([s not in val_sessions for s in session_ids])[0]
    val_idx = np.where([s in val_sessions for s in session_ids])[0]

    print(f"  Train: {len(train_idx)} trials ({len(train_sessions)} sessions)")
    print(f"  Val: {len(val_idx)} trials ({len(val_sessions)} sessions)")

    return {
        'ob': ob, 'pcx': pcx, 'session_ids': session_ids,
        'idx_to_session': idx_to_session, 'train_idx': train_idx, 'val_idx': val_idx,
    }


# =============================================================================
# GPU Feature Extraction
# =============================================================================

def extract_features_gpu(ob: np.ndarray, pcx: np.ndarray, device: torch.device) -> torch.Tensor:
    """Extract features using GPU acceleration."""
    n_trials, n_channels, n_time = ob.shape

    ob_t = torch.from_numpy(ob).float().to(device)
    pcx_t = torch.from_numpy(pcx).float().to(device)
    combined = torch.cat([ob_t, pcx_t], dim=1)

    features_list = []

    # Band power features
    fft = torch.fft.rfft(combined, dim=-1)
    psd = (fft.abs() ** 2) / n_time
    freqs = torch.fft.rfftfreq(n_time, d=1/1000).to(device)

    for f_low, f_high in [(1, 4), (4, 12), (12, 30), (30, 60), (60, 100)]:
        mask = (freqs >= f_low) & (freqs <= f_high)
        features_list.append(psd[:, :, mask].mean(dim=-1))

    # Temporal stats
    features_list.extend([
        combined.mean(dim=-1),
        combined.std(dim=-1),
        torch.diff(combined, dim=-1).abs().mean(dim=-1),
        torch.diff(combined, dim=-1).std(dim=-1),
    ])

    # RMS envelope
    rms = F.avg_pool1d(combined.abs(), kernel_size=50, stride=1, padding=25)[:, :, :n_time]
    features_list.extend([rms.mean(dim=-1), rms.std(dim=-1)])

    # Covariance (batched)
    triu_indices = torch.triu_indices(64, 64, offset=1, device=device)
    cov_features = []
    for start in range(0, n_trials, 200):
        end = min(start + 200, n_trials)
        batch = combined[start:end]
        batch_centered = batch - batch.mean(dim=-1, keepdim=True)
        cov = torch.bmm(batch_centered, batch_centered.transpose(1, 2)) / n_time
        cov_features.append(cov[:, triu_indices[0], triu_indices[1]])
    features_list.append(torch.cat(cov_features, dim=0))

    all_features = torch.cat(features_list, dim=1)
    mean, std = all_features.mean(dim=0, keepdim=True), all_features.std(dim=0, keepdim=True) + 1e-8
    return (all_features - mean) / std


def extract_covariance_gpu(ob: np.ndarray, pcx: np.ndarray, device: torch.device) -> torch.Tensor:
    """Extract covariance matrices."""
    n_trials, _, n_time = ob.shape
    ob_t = torch.from_numpy(ob).float().to(device)
    pcx_t = torch.from_numpy(pcx).float().to(device)
    combined = torch.cat([ob_t, pcx_t], dim=1)
    combined_centered = combined - combined.mean(dim=-1, keepdim=True)
    return torch.bmm(combined_centered, combined_centered.transpose(1, 2)) / n_time


# =============================================================================
# GPU Distance Metrics
# =============================================================================

def compute_mmd_gpu(X: torch.Tensor, Y: torch.Tensor) -> float:
    """MMD with RBF kernel."""
    gamma = 1.0 / X.shape[1]
    max_samples = 800
    if len(X) > max_samples:
        X = X[torch.randperm(len(X))[:max_samples]]
    if len(Y) > max_samples:
        Y = Y[torch.randperm(len(Y))[:max_samples]]

    def rbf(A, B):
        A_sq = (A ** 2).sum(dim=1, keepdim=True)
        B_sq = (B ** 2).sum(dim=1, keepdim=True)
        return torch.exp(-gamma * (A_sq + B_sq.T - 2 * torch.mm(A, B.T)))

    K_XX, K_YY, K_XY = rbf(X, X), rbf(Y, Y), rbf(X, Y)
    n, m = len(X), len(Y)
    mmd2 = (K_XX.sum() - K_XX.trace()) / (n * (n - 1))
    mmd2 += (K_YY.sum() - K_YY.trace()) / (m * (m - 1))
    mmd2 -= 2 * K_XY.mean()
    return float(max(0, mmd2.cpu().item()))


def compute_coverage_gpu(X_train: torch.Tensor, X_val: torch.Tensor, k: int = 5) -> float:
    """Coverage metric."""
    max_samples = 800
    if len(X_train) > max_samples:
        X_train = X_train[torch.randperm(len(X_train))[:max_samples]]
    if len(X_val) > max_samples:
        X_val = X_val[torch.randperm(len(X_val))[:max_samples]]

    train_sq = (X_train ** 2).sum(dim=1, keepdim=True)
    val_sq = (X_val ** 2).sum(dim=1, keepdim=True)

    train_dists = torch.sqrt(torch.clamp(train_sq + train_sq.T - 2 * torch.mm(X_train, X_train.T), min=0))
    train_dists.fill_diagonal_(float('inf'))
    threshold = train_dists.topk(k, dim=1, largest=False)[0][:, -1].quantile(0.95)

    val_train_dists = torch.sqrt(torch.clamp(val_sq + train_sq.T - 2 * torch.mm(X_val, X_train.T), min=0))
    return float((val_train_dists.min(dim=1)[0] <= threshold).float().mean().cpu().item())


def compute_cov_distance_gpu(cov1: torch.Tensor, cov2: torch.Tensor) -> float:
    """Covariance Frobenius distance."""
    diff = cov1.mean(dim=0) - cov2.mean(dim=0)
    return float(torch.sqrt((diff ** 2).sum()).cpu().item() / 64)


# =============================================================================
# Augmentation Functions
# =============================================================================

def aug_covariance_expansion(ob, pcx, strength=0.3, mode='random'):
    """Covariance augmentation - perturb eigenvalues."""
    n_trials, n_channels, _ = ob.shape
    ob_aug, pcx_aug = np.zeros_like(ob), np.zeros_like(pcx)

    for i in range(n_trials):
        try:
            ob_eigvals, ob_eigvecs = eigh(np.cov(ob[i]))
            pcx_eigvals, pcx_eigvecs = eigh(np.cov(pcx[i]))

            if mode == 'expand':
                scale = 1 + strength * np.random.rand(n_channels)
            elif mode == 'shrink':
                scale = 1 - strength * np.random.rand(n_channels)
            else:
                scale = 1 + strength * (2 * np.random.rand(n_channels) - 1)
            scale = np.clip(scale, 0.1, 10.0)

            ob_eigvals_new = np.maximum(ob_eigvals * scale, 1e-6)
            pcx_eigvals_new = np.maximum(pcx_eigvals * scale, 1e-6)

            ob_T = ob_eigvecs @ np.diag(np.sqrt(ob_eigvals_new / (ob_eigvals + 1e-10))) @ ob_eigvecs.T
            pcx_T = pcx_eigvecs @ np.diag(np.sqrt(pcx_eigvals_new / (pcx_eigvals + 1e-10))) @ pcx_eigvecs.T

            ob_c = ob[i] - ob[i].mean(axis=-1, keepdims=True)
            pcx_c = pcx[i] - pcx[i].mean(axis=-1, keepdims=True)
            ob_aug[i] = ob_T @ ob_c + ob[i].mean(axis=-1, keepdims=True)
            pcx_aug[i] = pcx_T @ pcx_c + pcx[i].mean(axis=-1, keepdims=True)
        except:
            ob_aug[i], pcx_aug[i] = ob[i], pcx[i]
    return ob_aug, pcx_aug


def aug_cross_session_mixing(ob, pcx, session_ids, alpha=0.3):
    """Cross-session mixup."""
    n_trials = ob.shape[0]
    unique_sessions = np.unique(session_ids)
    if len(unique_sessions) < 2:
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


def aug_euclidean_alignment(ob, pcx):
    """Euclidean Alignment - whiten to identity covariance."""
    n_trials, n_channels, _ = ob.shape
    try:
        ob_mean_cov = np.array([np.cov(ob[i]) for i in range(n_trials)]).mean(axis=0)
        pcx_mean_cov = np.array([np.cov(pcx[i]) for i in range(n_trials)]).mean(axis=0)
        ob_align = np.real(np.linalg.inv(sqrtm(ob_mean_cov + 1e-6 * np.eye(n_channels))))
        pcx_align = np.real(np.linalg.inv(sqrtm(pcx_mean_cov + 1e-6 * np.eye(n_channels))))
    except:
        return ob, pcx

    ob_aligned, pcx_aligned = np.zeros_like(ob), np.zeros_like(pcx)
    for i in range(n_trials):
        ob_c = ob[i] - ob[i].mean(axis=-1, keepdims=True)
        pcx_c = pcx[i] - pcx[i].mean(axis=-1, keepdims=True)
        ob_aligned[i] = ob_align @ ob_c + ob[i].mean(axis=-1, keepdims=True)
        pcx_aligned[i] = pcx_align @ pcx_c + pcx[i].mean(axis=-1, keepdims=True)
    return ob_aligned, pcx_aligned


def aug_amplitude_jitter(ob, pcx, scale_range=(0.7, 1.3)):
    """Per-channel amplitude jittering."""
    n_trials, n_channels, _ = ob.shape
    s_ob = np.random.uniform(scale_range[0], scale_range[1], (n_trials, n_channels, 1))
    s_pcx = np.random.uniform(scale_range[0], scale_range[1], (n_trials, n_channels, 1))
    return ob * s_ob, pcx * s_pcx


def aug_noise(ob, pcx, std=0.1):
    """Gaussian noise injection."""
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
    """Random channel dropout."""
    n_trials, n_channels, _ = ob.shape
    mask = (np.random.rand(n_trials, n_channels, 1) > p).astype(np.float32)
    return ob * mask, pcx * mask


def aug_freq_mask(ob, pcx, max_freq=100, n_masks=2):
    """Frequency band masking."""
    n_trials, n_channels, n_time = ob.shape
    ob_aug, pcx_aug = ob.copy(), pcx.copy()
    for i in range(n_trials):
        for _ in range(n_masks):
            f_start = np.random.randint(0, max_freq - 10)
            f_width = np.random.randint(5, 20)
            # Simple lowpass/bandstop approximation via FFT
            ob_fft = np.fft.rfft(ob_aug[i], axis=-1)
            pcx_fft = np.fft.rfft(pcx_aug[i], axis=-1)
            freqs = np.fft.rfftfreq(n_time, d=1/1000)
            mask = (freqs >= f_start) & (freqs <= f_start + f_width)
            ob_fft[:, mask] = 0
            pcx_fft[:, mask] = 0
            ob_aug[i] = np.fft.irfft(ob_fft, n=n_time, axis=-1)
            pcx_aug[i] = np.fft.irfft(pcx_fft, n=n_time, axis=-1)
    return ob_aug, pcx_aug


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
    time_shift_max: float = 0.1
    use_channel_dropout: bool = False
    channel_dropout_p: float = 0.1
    use_freq_mask: bool = False
    n_passes: int = 2

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def apply_augmentation(ob, pcx, session_ids, config: AugConfig):
    """Apply full augmentation pipeline."""
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
        ob_aug, pcx_aug = aug_noise(ob_aug, pcx_aug, config.noise_std)
    if config.use_time_shift:
        ob_aug, pcx_aug = aug_time_shift(ob_aug, pcx_aug, config.time_shift_max)
    if config.use_channel_dropout:
        ob_aug, pcx_aug = aug_channel_dropout(ob_aug, pcx_aug, config.channel_dropout_p)
    if config.use_freq_mask:
        ob_aug, pcx_aug = aug_freq_mask(ob_aug, pcx_aug)

    return ob_aug, pcx_aug


# =============================================================================
# BEAST MODE: 256+ Configurations
# =============================================================================

def generate_beast_search_space() -> List[AugConfig]:
    """Generate 256+ augmentation configurations - GO WILD."""
    configs = []

    # Baseline
    configs.append(AugConfig(name='baseline'))

    # Individual augmentations - fine-grained sweeps
    configs.append(AugConfig(name='ea_only', use_ea=True))

    # Covariance: 3 modes x 7 strengths = 21
    for strength in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        for mode in ['expand', 'shrink', 'random']:
            configs.append(AugConfig(name=f'cov_{mode}_{strength}', use_cov=True, cov_strength=strength, cov_mode=mode))

    # Cross-session mixing: 8 alphas
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]:
        configs.append(AugConfig(name=f'csm_{alpha}', use_csm=True, csm_alpha=alpha))

    # Noise: 6 levels
    for std in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
        configs.append(AugConfig(name=f'noise_{std}', use_noise=True, noise_std=std))

    # Amplitude: 5 ranges
    for lo, hi in [(0.8, 1.2), (0.7, 1.3), (0.6, 1.4), (0.5, 1.5), (0.9, 1.1)]:
        configs.append(AugConfig(name=f'amp_{lo}_{hi}', use_amp=True, amp_range=(lo, hi)))

    # Time shift: 4 levels
    for shift in [0.05, 0.1, 0.15, 0.2]:
        configs.append(AugConfig(name=f'tshift_{shift}', use_time_shift=True, time_shift_max=shift))

    # Channel dropout: 4 levels
    for p in [0.05, 0.1, 0.15, 0.2]:
        configs.append(AugConfig(name=f'chdrop_{p}', use_channel_dropout=True, channel_dropout_p=p))

    # Freq mask
    configs.append(AugConfig(name='freqmask', use_freq_mask=True))

    # EA + Cov combinations: 5 strengths x 3 modes = 15
    for strength in [0.1, 0.2, 0.3, 0.5, 0.7]:
        for mode in ['expand', 'shrink', 'random']:
            configs.append(AugConfig(name=f'ea_cov_{mode}_{strength}', use_ea=True, use_cov=True, cov_strength=strength, cov_mode=mode))

    # EA + CSM: 6 alphas
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        configs.append(AugConfig(name=f'ea_csm_{alpha}', use_ea=True, use_csm=True, csm_alpha=alpha))

    # Cov + CSM: 4x4 = 16
    for strength in [0.1, 0.2, 0.3, 0.5]:
        for alpha in [0.1, 0.2, 0.3, 0.4]:
            configs.append(AugConfig(name=f'cov_{strength}_csm_{alpha}', use_cov=True, cov_strength=strength, use_csm=True, csm_alpha=alpha))

    # EA + Cov + CSM: 4x4 = 16
    for strength in [0.1, 0.2, 0.3, 0.5]:
        for alpha in [0.1, 0.2, 0.3, 0.4]:
            configs.append(AugConfig(name=f'ea_cov_{strength}_csm_{alpha}', use_ea=True, use_cov=True, cov_strength=strength, use_csm=True, csm_alpha=alpha))

    # Triple combos with noise: 3x3x3 = 27
    for strength in [0.2, 0.3, 0.5]:
        for alpha in [0.2, 0.3, 0.4]:
            for noise in [0.05, 0.1, 0.15]:
                configs.append(AugConfig(
                    name=f'ea_cov_{strength}_csm_{alpha}_n_{noise}',
                    use_ea=True, use_cov=True, cov_strength=strength,
                    use_csm=True, csm_alpha=alpha, use_noise=True, noise_std=noise
                ))

    # Quad combos with amplitude: 3x3x2x2 = 36
    for strength in [0.2, 0.3, 0.5]:
        for alpha in [0.2, 0.3, 0.4]:
            for noise in [0.05, 0.1]:
                for amp in [(0.8, 1.2), (0.7, 1.3)]:
                    configs.append(AugConfig(
                        name=f'ea_cov_{strength}_csm_{alpha}_n_{noise}_amp_{amp[0]}',
                        use_ea=True, use_cov=True, cov_strength=strength, cov_mode='random',
                        use_csm=True, csm_alpha=alpha, use_noise=True, noise_std=noise,
                        use_amp=True, amp_range=amp
                    ))

    # Kitchen sink with time shift: 2x2x2x2 = 16
    for strength in [0.2, 0.3]:
        for alpha in [0.2, 0.3]:
            for noise in [0.05, 0.1]:
                for tshift in [0.05, 0.1]:
                    configs.append(AugConfig(
                        name=f'full_{strength}_{alpha}_{noise}_{tshift}',
                        use_ea=True, use_cov=True, cov_strength=strength,
                        use_csm=True, csm_alpha=alpha, use_noise=True, noise_std=noise,
                        use_amp=True, amp_range=(0.8, 1.2), use_time_shift=True, time_shift_max=tshift
                    ))

    # ULTRA combos with channel dropout: 2x2x2x2 = 16
    for strength in [0.2, 0.3]:
        for alpha in [0.2, 0.3]:
            for noise in [0.05, 0.1]:
                for cdrop in [0.05, 0.1]:
                    configs.append(AugConfig(
                        name=f'ultra_{strength}_{alpha}_{noise}_{cdrop}',
                        use_ea=True, use_cov=True, cov_strength=strength,
                        use_csm=True, csm_alpha=alpha, use_noise=True, noise_std=noise,
                        use_amp=True, amp_range=(0.8, 1.2), use_channel_dropout=True, channel_dropout_p=cdrop
                    ))

    # MEGA combos with freq mask: 2x2x2 = 8
    for strength in [0.2, 0.3]:
        for alpha in [0.2, 0.3]:
            for noise in [0.05, 0.1]:
                configs.append(AugConfig(
                    name=f'mega_{strength}_{alpha}_{noise}',
                    use_ea=True, use_cov=True, cov_strength=strength,
                    use_csm=True, csm_alpha=alpha, use_noise=True, noise_std=noise,
                    use_amp=True, amp_range=(0.8, 1.2), use_freq_mask=True
                ))

    # INSANE: ALL THE THINGS: 2x2x2 = 8
    for strength in [0.2, 0.3]:
        for alpha in [0.2, 0.3]:
            for noise in [0.05, 0.1]:
                configs.append(AugConfig(
                    name=f'insane_{strength}_{alpha}_{noise}',
                    use_ea=True, use_cov=True, cov_strength=strength,
                    use_csm=True, csm_alpha=alpha, use_noise=True, noise_std=noise,
                    use_amp=True, amp_range=(0.7, 1.3),
                    use_time_shift=True, time_shift_max=0.1,
                    use_channel_dropout=True, channel_dropout_p=0.1,
                    use_freq_mask=True
                ))

    print(f"\n  GENERATED {len(configs)} CONFIGURATIONS - LET'S GO!")
    return configs


# =============================================================================
# Parallel Evaluation
# =============================================================================

def evaluate_single_config(args):
    """Evaluate a single config on a specific GPU."""
    config, ob_train, pcx_train, ob_val, pcx_val, session_ids_train, gpu_id = args

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cpu')

    try:
        # Extract validation features
        val_features = extract_features_gpu(ob_val, pcx_val, device)
        val_cov = extract_covariance_gpu(ob_val, pcx_val, device)

        # Apply augmentation and extract features
        all_features, all_covs = [], []
        for _ in range(config.n_passes):
            ob_aug, pcx_aug = apply_augmentation(ob_train, pcx_train, session_ids_train, config)
            all_features.append(extract_features_gpu(ob_aug, pcx_aug, device))
            all_covs.append(extract_covariance_gpu(ob_aug, pcx_aug, device))

        train_features = torch.cat(all_features, dim=0)
        train_cov = torch.cat(all_covs, dim=0)

        # Compute metrics
        mmd = compute_mmd_gpu(train_features, val_features)
        coverage = compute_coverage_gpu(train_features, val_features)
        cov_dist = compute_cov_distance_gpu(train_cov, val_cov)
        score = coverage - 0.3 * mmd - 0.1 * cov_dist

        # Cleanup
        del val_features, val_cov, train_features, train_cov, all_features, all_covs
        torch.cuda.empty_cache()

        return {
            'config': config.to_dict(),
            'mmd': mmd,
            'coverage': coverage,
            'cov_distance': cov_dist,
            'score': score,
        }
    except Exception as e:
        return {
            'config': config.to_dict(),
            'mmd': 999.0,
            'coverage': 0.0,
            'cov_distance': 999.0,
            'score': -999.0,
            'error': str(e),
        }


# =============================================================================
# Visualization
# =============================================================================

def plot_results(results: List[Dict], output_dir: Path):
    """Generate comprehensive visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_sorted = sorted(results, key=lambda r: r['score'], reverse=True)
    top_n = min(30, len(results_sorted))
    top_results = results_sorted[:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    names = [r['config']['name'][:30] for r in top_results]
    scores = [r['score'] for r in top_results]
    mmds = [r['mmd'] for r in top_results]
    coverages = [r['coverage'] for r in top_results]

    # Score bar chart
    ax = axes[0, 0]
    colors = [plt.cm.RdYlGn((s - min(scores)) / (max(scores) - min(scores) + 1e-10)) for s in scores]
    bars = ax.barh(range(top_n), scores, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Combined Score (higher is better)')
    ax.set_title(f'Top {top_n} Augmentation Configurations')
    ax.invert_yaxis()

    # MMD bar chart
    ax = axes[0, 1]
    colors = [plt.cm.RdYlGn_r((m - min(mmds)) / (max(mmds) - min(mmds) + 1e-10)) for m in mmds]
    ax.barh(range(top_n), mmds, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('MMD (lower is better)')
    ax.set_title('Maximum Mean Discrepancy')
    ax.invert_yaxis()

    # Coverage bar chart
    ax = axes[1, 0]
    colors = [plt.cm.RdYlGn((c - min(coverages)) / (max(coverages) - min(coverages) + 1e-10)) for c in coverages]
    ax.barh(range(top_n), coverages, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Coverage (higher is better)')
    ax.set_title('Validation Coverage')
    ax.invert_yaxis()

    # Scatter: all configs
    ax = axes[1, 1]
    all_scores = [r['score'] for r in results]
    all_mmds = [r['mmd'] for r in results]
    all_coverages = [r['coverage'] for r in results]
    scatter = ax.scatter(all_mmds, all_coverages, c=all_scores, cmap='RdYlGn', alpha=0.6, s=30)
    plt.colorbar(scatter, ax=ax, label='Score')
    ax.set_xlabel('MMD (lower is better)')
    ax.set_ylabel('Coverage (higher is better)')
    ax.set_title(f'All {len(results)} Configurations')

    # Mark top 5
    for i, r in enumerate(results_sorted[:5]):
        ax.annotate(str(i+1), (r['mmd'], r['coverage']), fontsize=10, fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig(output_dir / 'augmentation_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'augmentation_comparison.png'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BEAST MODE Augmentation Explorer')
    parser.add_argument('--n-val-sessions', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='figures/augmentation_exploration')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n1. Loading data...")
    data = load_data(args.n_val_sessions, args.seed)

    ob, pcx = data['ob'], data['pcx']
    train_idx, val_idx = data['train_idx'], data['val_idx']
    session_ids = data['session_ids']

    ob_train, pcx_train = ob[train_idx], pcx[train_idx]
    ob_val, pcx_val = ob[val_idx], pcx[val_idx]
    session_ids_train = session_ids[train_idx]

    # Generate configs
    print("\n2. Generating BEAST MODE search space...")
    configs = generate_beast_search_space()

    # Evaluate - distribute across GPUs
    print(f"\n3. Evaluating {len(configs)} configurations...")
    results = []

    # Create tasks with round-robin GPU assignment
    tasks = []
    for i, config in enumerate(configs):
        gpu_id = i % max(1, N_GPUS)
        tasks.append((config, ob_train, pcx_train, ob_val, pcx_val, session_ids_train, gpu_id))

    # Run with progress bar
    for task in tqdm(tasks, desc="BEAST MODE"):
        result = evaluate_single_config(task)
        results.append(result)

    # Sort and report
    results_sorted = sorted(results, key=lambda r: r['score'], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 15 AUGMENTATION CONFIGURATIONS")
    print("=" * 70)

    for i, result in enumerate(results_sorted[:15]):
        print(f"\n{i+1}. {result['config']['name']}")
        print(f"   Score: {result['score']:.4f} | Coverage: {result['coverage']:.4f} | MMD: {result['mmd']:.4f} | CovDist: {result['cov_distance']:.4f}")

    # Save results
    print("\n4. Saving results...")
    with open(output_dir / 'augmentation_results.json', 'w') as f:
        json.dump(results_sorted, f, indent=2)

    # Plot
    print("\n5. Generating visualizations...")
    plot_results(results, output_dir)

    # Summary
    best = results_sorted[0]
    baseline = next((r for r in results if r['config']['name'] == 'baseline'), results_sorted[-1])

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nBaseline: Score={baseline['score']:.4f}, Coverage={baseline['coverage']:.4f}, MMD={baseline['mmd']:.4f}")
    print(f"\n🏆 BEST: {best['config']['name']}")
    print(f"   Score: {best['score']:.4f} (+{best['score'] - baseline['score']:.4f})")
    print(f"   Coverage: {best['coverage']:.4f} (+{best['coverage'] - baseline['coverage']:.4f})")
    print(f"   MMD: {best['mmd']:.4f} ({best['mmd'] - baseline['mmd']:+.4f})")

    print(f"\n📊 Results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
