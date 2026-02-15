#!/usr/bin/env python3
"""Calibration experiment: fit post-hoc corrections on TRAIN, evaluate on TEST.

Tries multiple calibration strategies and prints a comparison table.

Usage:
    python -m phase_four.calibration_experiment --dataset pcx1
    python -m phase_four.calibration_experiment --dataset pcx1 --max-train 5000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sig
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase_four.config import DATASETS, FREQUENCY_BANDS, Phase4Config


# ═══════════════════════════════════════════════════════════════════════════
# Frequency band definitions
# ═══════════════════════════════════════════════════════════════════════════

BANDS_COARSE: Dict[str, Tuple[float, float]] = FREQUENCY_BANDS  # 6 bands

BANDS_FINE: Dict[str, Tuple[float, float]] = {
    "delta_lo":   (1.0, 2.0),
    "delta_hi":   (2.0, 4.0),
    "theta_lo":   (4.0, 6.0),
    "theta_hi":   (6.0, 8.0),
    "alpha":      (8.0, 12.0),
    "beta_lo":    (12.0, 20.0),
    "beta_hi":    (20.0, 30.0),
    "gamma_lo":   (30.0, 50.0),
    "gamma_mid":  (50.0, 70.0),
    "gamma_hi":   (70.0, 100.0),
}


def make_uniform_bands(
    width_hz: float, lo: float = 1.0, hi: float = 500.0,
) -> Dict[str, Tuple[float, float]]:
    bands = {}
    f = lo
    while f < hi:
        f2 = min(f + width_hz, hi)
        bands[f"{f:.0f}-{f2:.0f}Hz"] = (f, f2)
        f = f2
    return bands


BANDS_2HZ = make_uniform_bands(2.0, 1.0, 500.0)
BANDS_5HZ = make_uniform_bands(5.0, 1.0, 500.0)


# ═══════════════════════════════════════════════════════════════════════════
# Calibration primitives
# ═══════════════════════════════════════════════════════════════════════════

def _build_masks(
    freqs: np.ndarray,
    bands: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """Smooth sigmoid masks [n_bands, n_freq]."""
    band_list = list(bands.values())
    masks = np.zeros((len(band_list), len(freqs)), dtype=np.float64)
    for i, (lo, hi) in enumerate(band_list):
        rise = 1.0 / (1.0 + np.exp(np.clip(-5.0 * (freqs - lo), -500, 500)))
        fall = 1.0 / (1.0 + np.exp(np.clip(-5.0 * (hi - freqs), -500, 500)))
        masks[i] = rise * fall
    mask_sum = masks.sum(axis=0, keepdims=True).clip(min=1e-6)
    masks /= mask_sum
    return masks


class SpectralBiasCalibrator:
    """Fit per-channel, per-band amplitude correction on train data."""

    def __init__(
        self,
        fs: int,
        bands: Dict[str, Tuple[float, float]],
        max_log_amp: float = 2.0,
    ):
        self.fs = fs
        self.bands = bands
        self.max_log_amp = max_log_amp
        self.log_amp_ratio: Optional[np.ndarray] = None  # [C, K]
        self._masks_cache: Dict[int, np.ndarray] = {}

    def _masks(self, T: int) -> np.ndarray:
        if T not in self._masks_cache:
            freqs = np.fft.rfftfreq(T, d=1.0 / self.fs)
            self._masks_cache[T] = _build_masks(freqs, self.bands)
        return self._masks_cache[T]

    def fit(self, pred_train: np.ndarray, real_train: np.ndarray):
        """Compute optimal correction from training predictions vs targets."""
        N, C, T = pred_train.shape
        masks = self._masks(T)  # [K, F]

        pred_fft = np.fft.rfft(pred_train.astype(np.float64), axis=-1)
        real_fft = np.fft.rfft(real_train.astype(np.float64), axis=-1)

        pred_power = np.abs(pred_fft) ** 2
        real_power = np.abs(real_fft) ** 2

        pred_bp = np.einsum("ncf,kf->nck", pred_power, masks).mean(axis=0)  # [C, K]
        real_bp = np.einsum("ncf,kf->nck", real_power, masks).mean(axis=0)

        ratio = 0.5 * (np.log(real_bp + 1e-20) - np.log(pred_bp + 1e-20))
        self.log_amp_ratio = np.clip(ratio, -self.max_log_amp, self.max_log_amp)

        db = self.log_amp_ratio * 20.0 / np.log(10)
        print(f"    [SpectralBias] bands={len(self.bands)}, "
              f"clamp=±{self.max_log_amp:.1f}, "
              f"mean_correction={db.mean():.1f}dB, "
              f"range=[{db.min():.1f}, {db.max():.1f}]dB")

    def transform(self, pred: np.ndarray) -> np.ndarray:
        """Apply fitted correction to predictions."""
        assert self.log_amp_ratio is not None, "Call fit() first"
        N, C, T = pred.shape
        masks = self._masks(T)

        amp_scales = np.exp(np.einsum("ck,kf->cf", self.log_amp_ratio, masks))
        coverage = masks.sum(axis=0)
        amp_scales[:, coverage < 0.01] = 1.0

        pred_fft = np.fft.rfft(pred.astype(np.float64), axis=-1)
        cal_fft = pred_fft * amp_scales[np.newaxis, :, :]
        return np.fft.irfft(cal_fft, n=T, axis=-1).astype(pred.dtype)


class EnvelopeCalibrator:
    """Fit per-channel envelope mean/std matching on train data."""

    def __init__(self):
        self.tgt_mean: Optional[np.ndarray] = None  # [C]
        self.tgt_std: Optional[np.ndarray] = None

    def fit(self, real_train: np.ndarray):
        """Compute target envelope statistics from real training data."""
        N, C, T = real_train.shape
        self.tgt_mean = np.empty(C)
        self.tgt_std = np.empty(C)
        for ch in range(C):
            env = np.abs(hilbert(real_train[:, ch, :].astype(np.float64), axis=-1))
            self.tgt_mean[ch] = env.mean()
            self.tgt_std[ch] = max(env.std(), 1e-8)
        print(f"    [Envelope] mean_range=[{self.tgt_mean.min():.3f}, {self.tgt_mean.max():.3f}], "
              f"std_range=[{self.tgt_std.min():.3f}, {self.tgt_std.max():.3f}]")

    def transform(self, pred: np.ndarray) -> np.ndarray:
        """Apply envelope correction to predictions."""
        assert self.tgt_mean is not None, "Call fit() first"
        N, C, T = pred.shape
        calibrated = np.empty_like(pred)
        for ch in range(C):
            analytic = hilbert(pred[:, ch, :].astype(np.float64), axis=-1)
            env = np.abs(analytic)
            phase = np.angle(analytic)
            p_mean = env.mean()
            p_std = max(env.std(), 1e-8)
            corrected = (env - p_mean) * (self.tgt_std[ch] / p_std) + self.tgt_mean[ch]
            calibrated[:, ch, :] = corrected * np.cos(phase)
        return calibrated.astype(pred.dtype)


class PerChannelGainCalibrator:
    """Simple per-channel scalar gain + bias (OLS fit on train)."""

    def __init__(self):
        self.gain: Optional[np.ndarray] = None  # [C]
        self.bias: Optional[np.ndarray] = None

    def fit(self, pred_train: np.ndarray, real_train: np.ndarray):
        N, C, T = pred_train.shape
        self.gain = np.empty(C)
        self.bias = np.empty(C)
        for ch in range(C):
            p = pred_train[:, ch, :].ravel().astype(np.float64)
            r = real_train[:, ch, :].ravel().astype(np.float64)
            # OLS: r = gain * p + bias
            cov = np.mean(p * r) - np.mean(p) * np.mean(r)
            var = np.mean(p ** 2) - np.mean(p) ** 2
            self.gain[ch] = cov / max(var, 1e-20)
            self.bias[ch] = np.mean(r) - self.gain[ch] * np.mean(p)
        print(f"    [ChannelGain] gain_range=[{self.gain.min():.3f}, {self.gain.max():.3f}], "
              f"bias_range=[{self.bias.min():.3f}, {self.bias.max():.3f}]")

    def transform(self, pred: np.ndarray) -> np.ndarray:
        assert self.gain is not None
        cal = pred.copy().astype(np.float64)
        for ch in range(pred.shape[1]):
            cal[:, ch, :] = self.gain[ch] * cal[:, ch, :] + self.bias[ch]
        return cal.astype(pred.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def bandpass(x, lo, hi, fs, order=4):
    nyq = fs / 2.0
    lo_n, hi_n = max(lo / nyq, 1e-5), min(hi / nyq, 0.9999)
    if lo_n >= hi_n:
        return np.zeros_like(x)
    sos = sig.butter(order, [lo_n, hi_n], btype="band", output="sos")
    return sig.sosfiltfilt(sos, x, axis=-1)


def compute_metrics(
    real: np.ndarray,
    pred: np.ndarray,
    fs: int,
    nperseg: int = 1024,
) -> Dict[str, float]:
    """Quick metrics: PSD corr, PSD MSE, per-band R², Pearson r."""
    # PSD
    _, psd_r = sig.welch(real, fs=fs, nperseg=nperseg, axis=-1)
    _, psd_p = sig.welch(pred, fs=fs, nperseg=nperseg, axis=-1)
    psd_r = psd_r.mean(axis=(0, 1))
    psd_p = psd_p.mean(axis=(0, 1))
    log_r = 10 * np.log10(psd_r + 1e-20)
    log_p = 10 * np.log10(psd_p + 1e-20)

    metrics: Dict[str, float] = {
        "psd_corr": float(np.corrcoef(log_r, log_p)[0, 1]),
        "psd_mse_db": float(np.mean((log_r - log_p) ** 2)),
    }

    # Pearson (global)
    metrics["pearson_r"] = float(np.corrcoef(real.ravel(), pred.ravel())[0, 1])

    # Per-band R²
    for name, (lo, hi) in FREQUENCY_BANDS.items():
        rb = bandpass(real, lo, hi, fs).ravel()
        pb = bandpass(pred, lo, hi, fs).ravel()
        ss_res = np.sum((rb - pb) ** 2)
        ss_tot = np.sum((rb - rb.mean()) ** 2)
        metrics[f"r2_{name}"] = float(1.0 - ss_res / (ss_tot + 1e-20))

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (reuses generate.py machinery)
# ═══════════════════════════════════════════════════════════════════════════

def load_train_test(
    dataset_key: str,
    cfg: Phase4Config,
    device,
    max_train: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load model + data, run inference on train & test splits.

    Returns:
        pred_train, real_train, pred_test, real_test
    """
    import torch
    from phase_four.generate import _load_model, run_inference

    ds = DATASETS[dataset_key]
    ckpt_path = cfg.get_checkpoint_path(dataset_key)

    print(f"Loading checkpoint: {ckpt_path}")
    model, train_config, ckpt = _load_model(ckpt_path, device)
    split_info = ckpt.get("split_info", {})

    # ── Build both train and test loaders ─────────────────────────────────
    if ds.train_name == "pcx1":
        pred_train, real_train, pred_test, real_test = _pcx1_train_test(
            model, train_config, cfg, device,
        )
    else:
        pred_train, real_train, pred_test, real_test = _indexed_train_test(
            model, dataset_key, ds, train_config, split_info, cfg, device,
        )

    if max_train and pred_train.shape[0] > max_train:
        rng = np.random.default_rng(42)
        idx = rng.choice(pred_train.shape[0], max_train, replace=False)
        pred_train, real_train = pred_train[idx], real_train[idx]

    print(f"  Train: {pred_train.shape}, Test: {pred_test.shape}")
    return pred_train, real_train, pred_test, real_test


def _pcx1_train_test(model, train_config, cfg, device):
    """PCx1: session-based splits → train + val loaders."""
    import torch
    from data import create_pcx1_dataloaders, get_pcx1_session_splits
    from phase_four.generate import run_inference

    seed = train_config.get("seed", 42)
    window_size = train_config.get("pcx1_window_size", 5000)
    stride = train_config.get("pcx1_stride", 2500)

    n_val = train_config.get("pcx1_n_val", 4)
    train_sessions, val_sessions, _ = get_pcx1_session_splits(
        seed=seed, n_val=n_val, n_test=0,
    )
    print(f"  PCx1 train_sessions={train_sessions}, val_sessions={val_sessions}")

    loaders = create_pcx1_dataloaders(
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        window_size=window_size,
        stride=stride,
        val_stride=stride,
        batch_size=cfg.batch_size,
        zscore_per_window=True,
        num_workers=4,
        separate_val_sessions=False,
        persistent_workers=False,
    )

    print("  Running inference on train split...")
    train_res = run_inference(model, loaders["train"], device)
    print("  Running inference on val/test split...")
    test_res = run_inference(model, loaders["val"], device)

    return train_res["predicted"], train_res["target"], test_res["predicted"], test_res["target"]


def _indexed_train_test(model, dataset_key, ds, train_config, split_info, cfg, device):
    """Index-based datasets: rebuild data + splits, run inference on both."""
    import torch
    from torch.utils.data import DataLoader
    from phase_four.generate import run_inference

    # Try checkpoint split_info first
    train_idx = np.array(split_info.get("train_idx", []))
    test_idx = np.array(split_info.get("test_idx", split_info.get("val_idx", [])))

    # If indices missing from checkpoint, rebuild splits from the data module
    if len(train_idx) == 0 or len(test_idx) == 0:
        print(f"  No train/test indices in checkpoint — rebuilding splits from data...")
        train_idx, test_idx, data_dict = _rebuild_splits(dataset_key, ds, train_config)
    else:
        data_dict = None

    # Build loaders
    from phase_four.generate import _make_test_loader

    if data_dict is not None:
        # We already have the full dataset loaded — build loaders directly
        from data import PairedNeuralDataset
        reverse = "--pfc-reverse" in ds.extra_train_args

        if ds.train_name == "pfc":
            if reverse:
                source, target = data_dict["ca1"], data_dict["pfc"]
            else:
                source, target = data_dict["pfc"], data_dict["ca1"]
            labels = data_dict["trial_types"]
        elif ds.train_name == "olfactory":
            source, target, labels = data_dict["ob"], data_dict["pcx"], data_dict["labels"]
        else:
            # Fallback for other datasets: try the generate.py factory
            train_loader = _make_test_loader(dataset_key, ds, train_config, train_idx, cfg)
            test_loader = _make_test_loader(dataset_key, ds, train_config, test_idx, cfg)
            print("  Running inference on train split...")
            train_res = run_inference(model, train_loader, device)
            print("  Running inference on test split...")
            test_res = run_inference(model, test_loader, device)
            return train_res["predicted"], train_res["target"], test_res["predicted"], test_res["target"]

        train_ds = PairedNeuralDataset(source, target, labels, train_idx)
        test_ds = PairedNeuralDataset(source, target, labels, test_idx)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    else:
        train_loader = _make_test_loader(dataset_key, ds, train_config, train_idx, cfg)
        test_loader = _make_test_loader(dataset_key, ds, train_config, test_idx, cfg)

    print(f"  Train: {len(train_idx)} trials, Test: {len(test_idx)} trials")
    print("  Running inference on train split...")
    train_res = run_inference(model, train_loader, device)
    print("  Running inference on test split...")
    test_res = run_inference(model, test_loader, device)

    return train_res["predicted"], train_res["target"], test_res["predicted"], test_res["target"]


def _rebuild_splits(dataset_key, ds, train_config):
    """Rebuild train/test splits from data when checkpoint doesn't store indices."""
    if ds.train_name == "pfc":
        from data import prepare_pfc_data
        data = prepare_pfc_data(split_by_session=True)
    elif ds.train_name == "olfactory":
        from data import prepare_data
        data = prepare_data(split_by_session=True)
    elif ds.train_name == "dandi":
        from data import prepare_dandi_data
        data = prepare_dandi_data(
            source_region=train_config.get("dandi_source_region", "amygdala"),
            target_region=train_config.get("dandi_target_region", "hippocampus"),
        )
    elif ds.train_name == "ecog":
        from data import prepare_ecog_data
        data = prepare_ecog_data(
            experiment=train_config.get("ecog_experiment", "motor_imagery"),
            source_region=train_config.get("ecog_source_region", "frontal"),
            target_region=train_config.get("ecog_target_region", "temporal"),
        )
    elif ds.train_name == "boran":
        from data import prepare_boran_data
        data = prepare_boran_data(
            source_region=train_config.get("boran_source_region", "hippocampus"),
            target_region=train_config.get("boran_target_region", "entorhinal_cortex"),
        )
    else:
        raise RuntimeError(f"Don't know how to rebuild splits for {dataset_key} ({ds.train_name})")

    train_idx = np.array(data["train_idx"])
    test_idx = np.array(data["test_idx"])
    # Fall back to val_idx when test split is empty (e.g. PFC session splits
    # use val as the held-out set with 0 dedicated test sessions)
    if len(test_idx) == 0:
        test_idx = np.array(data["val_idx"])
        print(f"  test_idx empty — using val_idx ({len(test_idx)} trials) as test set")
    return train_idx, test_idx, data


# ═══════════════════════════════════════════════════════════════════════════
# Experiment runner
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
    pred_train: np.ndarray,
    real_train: np.ndarray,
    pred_test: np.ndarray,
    real_test: np.ndarray,
    fs: int,
) -> List[Dict[str, Any]]:
    """Try all calibration strategies, return comparison rows."""

    rows: List[Dict[str, Any]] = []

    # ── 0. Baseline (no calibration) ─────────────────────────────────────
    print("\n  [0] Baseline (raw predictions)")
    m = compute_metrics(real_test, pred_test, fs)
    rows.append({"name": "raw", **m})

    # ── 1. Per-channel gain (OLS) ────────────────────────────────────────
    print("\n  [1] Per-channel OLS gain+bias")
    cal_gain = PerChannelGainCalibrator()
    cal_gain.fit(pred_train, real_train)
    p_gain = cal_gain.transform(pred_test)
    rows.append({"name": "ch_gain", **compute_metrics(real_test, p_gain, fs)})

    # ── 2. Spectral bias: coarse (6 bands) ──────────────────────────────
    print("\n  [2] Spectral bias (6 coarse bands)")
    cal_s6 = SpectralBiasCalibrator(fs, BANDS_COARSE, max_log_amp=2.0)
    cal_s6.fit(pred_train, real_train)
    p_s6 = cal_s6.transform(pred_test)
    rows.append({"name": "spec_6band", **compute_metrics(real_test, p_s6, fs)})

    # ── 3. Spectral bias: fine (10 neuro bands) ─────────────────────────
    print("\n  [3] Spectral bias (10 neuro bands)")
    cal_s10 = SpectralBiasCalibrator(fs, BANDS_FINE, max_log_amp=2.0)
    cal_s10.fit(pred_train, real_train)
    p_s10 = cal_s10.transform(pred_test)
    rows.append({"name": "spec_10band", **compute_metrics(real_test, p_s10, fs)})

    # ── 4. Spectral bias: uniform 2 Hz ──────────────────────────────────
    print("\n  [4] Spectral bias (uniform 2Hz bands)")
    cal_u2 = SpectralBiasCalibrator(fs, BANDS_2HZ, max_log_amp=2.0)
    cal_u2.fit(pred_train, real_train)
    p_u2 = cal_u2.transform(pred_test)
    rows.append({"name": "spec_2Hz", **compute_metrics(real_test, p_u2, fs)})

    # ── 5. Spectral bias: uniform 5 Hz ──────────────────────────────────
    print("\n  [5] Spectral bias (uniform 5Hz bands)")
    cal_u5 = SpectralBiasCalibrator(fs, BANDS_5HZ, max_log_amp=2.0)
    cal_u5.fit(pred_train, real_train)
    p_u5 = cal_u5.transform(pred_test)
    rows.append({"name": "spec_5Hz", **compute_metrics(real_test, p_u5, fs)})

    # ── 6. Envelope only ─────────────────────────────────────────────────
    print("\n  [6] Envelope matching only")
    cal_env = EnvelopeCalibrator()
    cal_env.fit(real_train)
    p_env = cal_env.transform(pred_test)
    rows.append({"name": "envelope", **compute_metrics(real_test, p_env, fs)})

    # ── 7. Best spectral + envelope ──────────────────────────────────────
    # (chain: spectral first, then envelope)
    print("\n  [7] Spectral (10 band) + envelope")
    p_both_10 = cal_env.transform(p_s10)
    rows.append({"name": "spec10+env", **compute_metrics(real_test, p_both_10, fs)})

    print("\n  [8] Spectral (2Hz) + envelope")
    p_both_2hz = cal_env.transform(p_u2)
    rows.append({"name": "spec2Hz+env", **compute_metrics(real_test, p_both_2hz, fs)})

    # ── 9. Gain + envelope ───────────────────────────────────────────────
    print("\n  [9] Channel gain + envelope")
    p_ge = cal_env.transform(p_gain)
    rows.append({"name": "gain+env", **compute_metrics(real_test, p_ge, fs)})

    # ── 10. Spectral (conservative clamp) ────────────────────────────────
    print("\n  [10] Spectral (10 band, clamp=0.5)")
    cal_cons = SpectralBiasCalibrator(fs, BANDS_FINE, max_log_amp=0.5)
    cal_cons.fit(pred_train, real_train)
    p_cons = cal_cons.transform(pred_test)
    rows.append({"name": "spec10_cons", **compute_metrics(real_test, p_cons, fs)})

    print("\n  [11] Spectral (10 band, clamp=0.5) + envelope")
    p_cons_env = cal_env.transform(p_cons)
    rows.append({"name": "spec10c+env", **compute_metrics(real_test, p_cons_env, fs)})

    return rows


def print_table(rows: List[Dict[str, Any]]):
    """Pretty-print comparison table."""
    band_cols = [k for k in rows[0] if k.startswith("r2_")]

    print("\n" + "=" * 120)
    print("CALIBRATION COMPARISON")
    print("=" * 120)

    # Header
    hdr = f"{'strategy':<16} {'psd_corr':>9} {'psd_mse':>9} {'pearson':>9}"
    for bc in band_cols:
        hdr += f" {bc.replace('r2_',''):>10}"
    print(hdr)
    print("-" * 120)

    # Baseline row for delta computation
    base = rows[0]

    for row in rows:
        line = f"{row['name']:<16}"
        # PSD corr
        d = row['psd_corr'] - base['psd_corr']
        line += f" {row['psd_corr']:>8.4f}{'+' if d >= 0 else ''}"
        # PSD MSE
        d = row['psd_mse_db'] - base['psd_mse_db']
        line += f" {row['psd_mse_db']:>8.2f}"
        # Pearson
        d = row['pearson_r'] - base['pearson_r']
        line += f" {row['pearson_r']:>8.4f}"
        # Band R²
        for bc in band_cols:
            line += f" {row[bc]:>10.4f}"
        print(line)

    # Delta row
    print("-" * 120)
    print("Δ vs raw:")
    for row in rows[1:]:
        line = f"  {row['name']:<14}"
        line += f" {row['psd_corr'] - base['psd_corr']:>+8.4f} "
        line += f" {row['psd_mse_db'] - base['psd_mse_db']:>+8.2f}"
        line += f" {row['pearson_r'] - base['pearson_r']:>+8.4f}"
        for bc in band_cols:
            line += f" {row[bc] - base[bc]:>+10.4f}"
        print(line)
    print("=" * 120)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Post-hoc calibration experiment")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASETS.keys()))
    parser.add_argument("--max-train", type=int, default=None,
                        help="Subsample training set (for speed)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--synth-root", type=str, default="/data/synth")
    parser.add_argument("--output-dir", type=str, default="results/phase4")
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Phase4Config(
        batch_size=args.batch_size,
        synth_root=Path(args.synth_root),
        output_dir=Path(args.output_dir),
    )

    ds = DATASETS[args.dataset]
    fs = ds.sampling_rate

    print(f"\n{'='*60}")
    print(f"Calibration Experiment: {ds.display_name}")
    print(f"{'='*60}")

    # 1. Load data + run inference
    t0 = time.time()
    pred_train, real_train, pred_test, real_test = load_train_test(
        args.dataset, cfg, device, max_train=args.max_train,
    )
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # 2. Run all calibration strategies
    t0 = time.time()
    rows = run_experiment(pred_train, real_train, pred_test, real_test, fs)
    print(f"\n  Experiments completed in {time.time() - t0:.1f}s")

    # 3. Print comparison table
    print_table(rows)

    # 4. Save results
    import json
    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration_experiment.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
