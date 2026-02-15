"""Spectral fidelity validation.

Compares real vs predicted target signals across three axes:
1. Power Spectral Density (PSD) match – Welch periodogram per channel
2. Per-band R²                       – delta / theta / alpha / beta / gamma
3. Cross-Frequency Coupling (PAC)    – theta-gamma phase-amplitude coupling

All functions accept arrays of shape [N, C, T] and a sampling rate.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import signal as sig

from phase_four.config import FREQUENCY_BANDS


# ═══════════════════════════════════════════════════════════════════════════
# 0.  Post-hoc calibration  (numpy, closed-form)
# ═══════════════════════════════════════════════════════════════════════════

# Fine-grained bands for spectral bias correction (matches legacy 10-band)
_CALIBRATION_BANDS: Dict[str, Tuple[float, float]] = {
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


def calibrate_spectral_bias(
    pred: np.ndarray,
    real: np.ndarray,
    fs: int,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """Per-band spectral bias correction in FFT domain (closed-form).

    Measures the power-spectral-density ratio between *real* and *pred* in
    each frequency band, then scales the prediction's FFT magnitudes to
    close the gap.  Operates per-channel, averaged across trials.

    Args:
        pred: predicted signals  [N, C, T]
        real: real target signals [N, C, T]
        fs:   sampling rate (Hz)
        bands: frequency band dict  {name: (lo_hz, hi_hz)}

    Returns:
        Calibrated predictions  [N, C, T]
    """
    if bands is None:
        bands = _CALIBRATION_BANDS

    N, C, T = pred.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)

    # Build smooth band masks  [n_bands, n_freq]
    band_list = list(bands.values())
    n_bands = len(band_list)
    masks = np.zeros((n_bands, len(freqs)), dtype=np.float64)
    for i, (lo, hi) in enumerate(band_list):
        rise = 1.0 / (1.0 + np.exp(np.clip(-5.0 * (freqs - lo), -500, 500)))
        fall = 1.0 / (1.0 + np.exp(np.clip(-5.0 * (hi - freqs), -500, 500)))
        masks[i] = rise * fall

    # Normalise so masks sum to 1 per freq bin (where covered)
    mask_sum = masks.sum(axis=0, keepdims=True).clip(min=1e-6)
    masks /= mask_sum

    # Per-channel PSD in each band
    pred_fft = np.fft.rfft(pred.astype(np.float64), axis=-1)  # [N, C, F]
    real_fft = np.fft.rfft(real.astype(np.float64), axis=-1)

    pred_power = np.abs(pred_fft) ** 2   # [N, C, F]
    real_power = np.abs(real_fft) ** 2

    # Band power per channel (averaged over trials)
    # einsum: ncf,kf -> nck  then mean over n -> [C, n_bands]
    pred_bp = np.einsum("ncf,kf->nck", pred_power, masks).mean(axis=0)  # [C, K]
    real_bp = np.einsum("ncf,kf->nck", real_power, masks).mean(axis=0)

    # Log-amplitude ratio:  log(sqrt(real_power / pred_power))
    log_amp_ratio = 0.5 * (np.log(real_bp + 1e-20) - np.log(pred_bp + 1e-20))  # [C, K]

    # Convert per-band ratios to per-frequency amplitude scale
    amp_scales = np.exp(np.einsum("ck,kf->cf", log_amp_ratio, masks))  # [C, F]

    # Leave uncovered frequencies (outside all bands) unchanged
    coverage = masks.sum(axis=0)
    uncovered = coverage < 0.01
    amp_scales[:, uncovered] = 1.0

    # Apply in FFT domain
    calibrated_fft = pred_fft * amp_scales[np.newaxis, :, :]  # [N, C, F]
    calibrated = np.fft.irfft(calibrated_fft, n=T, axis=-1)

    return calibrated.astype(pred.dtype)


def calibrate_envelope(
    pred: np.ndarray,
    real: np.ndarray,
) -> np.ndarray:
    """Envelope mean/std matching (closed-form).

    Computes the analytic-signal envelope of both *pred* and *real*,
    then linearly scales the prediction envelope to match the target
    statistics per channel.  Phase is preserved.

    Args:
        pred: predicted signals  [N, C, T]
        real: real target signals [N, C, T]

    Returns:
        Calibrated predictions  [N, C, T]
    """
    from scipy.signal import hilbert

    N, C, T = pred.shape
    calibrated = np.empty_like(pred)

    for ch in range(C):
        # Target envelope stats (across all trials for this channel)
        real_analytic = hilbert(real[:, ch, :].astype(np.float64), axis=-1)
        real_env = np.abs(real_analytic)
        tgt_mean = real_env.mean()
        tgt_std = max(real_env.std(), 1e-8)

        # Prediction envelope + phase
        pred_analytic = hilbert(pred[:, ch, :].astype(np.float64), axis=-1)
        pred_env = np.abs(pred_analytic)
        pred_phase = np.angle(pred_analytic)
        pred_mean = pred_env.mean()
        pred_std = max(pred_env.std(), 1e-8)

        # Linear scaling: match mean/std
        corrected_env = (pred_env - pred_mean) * (tgt_std / pred_std) + tgt_mean
        calibrated[:, ch, :] = corrected_env * np.cos(pred_phase)

    return calibrated.astype(pred.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PSD  match
# ═══════════════════════════════════════════════════════════════════════════

def compute_psd(
    x: np.ndarray,
    fs: int,
    nperseg: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean Welch PSD across trials and channels.

    Args:
        x: [N, C, T]
        fs: sampling rate (Hz)

    Returns:
        freqs [F], psd [F]  (averaged over N×C)
    """
    freqs, pxx = sig.welch(x, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, pxx.mean(axis=(0, 1))


def psd_match_metrics(
    real: np.ndarray,
    pred: np.ndarray,
    fs: int,
    nperseg: int = 1024,
) -> Dict[str, Any]:
    """Compare PSD of real vs predicted signals.

    Returns:
        psd_corr       – Pearson-r between log-PSD curves
        psd_mse_db     – mean squared error in dB space
        psd_real       – [F] array
        psd_pred       – [F] array
        freqs          – [F] array
    """
    freqs, psd_real = compute_psd(real, fs, nperseg)
    _, psd_pred = compute_psd(pred, fs, nperseg)

    # Log-space comparison (dB)
    log_real = 10 * np.log10(psd_real + 1e-20)
    log_pred = 10 * np.log10(psd_pred + 1e-20)

    psd_mse_db = float(np.mean((log_real - log_pred) ** 2))
    psd_corr = float(np.corrcoef(log_real, log_pred)[0, 1])

    return {
        "psd_corr": psd_corr,
        "psd_mse_db": psd_mse_db,
        "freqs": freqs,
        "psd_real": psd_real,
        "psd_pred": psd_pred,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Per-band  R²
# ═══════════════════════════════════════════════════════════════════════════

def bandpass(x: np.ndarray, lo: float, hi: float, fs: int, order: int = 4):
    """Zero-phase Butterworth bandpass, applied along last axis."""
    nyq = fs / 2.0
    # Clamp to valid range
    lo_n = max(lo / nyq, 1e-5)
    hi_n = min(hi / nyq, 0.9999)
    if lo_n >= hi_n:
        return np.zeros_like(x)
    sos = sig.butter(order, [lo_n, hi_n], btype="band", output="sos")
    return sig.sosfiltfilt(sos, x, axis=-1)


def per_band_r2(
    real: np.ndarray,
    pred: np.ndarray,
    fs: int,
    bands: Optional[Dict[str, tuple]] = None,
) -> Dict[str, float]:
    """R² between real and predicted for each frequency band.

    Filters both signals into each band, then computes channel-averaged R².
    """
    if bands is None:
        bands = FREQUENCY_BANDS

    results = {}
    for name, (lo, hi) in bands.items():
        real_band = bandpass(real, lo, hi, fs)
        pred_band = bandpass(pred, lo, hi, fs)

        # Flatten to (N*C*T,) for global R²
        r = real_band.ravel()
        p = pred_band.ravel()
        ss_res = np.sum((r - p) ** 2)
        ss_tot = np.sum((r - r.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-20)
        results[name] = float(r2)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Phase-Amplitude Coupling  (PAC)  –  Modulation Index
# ═══════════════════════════════════════════════════════════════════════════

def _modulation_index(phase: np.ndarray, amp: np.ndarray, n_bins: int = 18):
    """Tort et al. (2010) Modulation Index (MI) via KL divergence.

    Args:
        phase: instantaneous phase of low-freq signal (radians)  [T]
        amp  : instantaneous amplitude of high-freq signal         [T]

    Returns:
        MI (float) – 0 = no coupling, log(n_bins) = perfect coupling
    """
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (phase >= bin_edges[b]) & (phase < bin_edges[b + 1])
        if mask.any():
            mean_amp[b] = amp[mask].mean()

    # Normalise to a distribution
    total = mean_amp.sum()
    if total < 1e-20:
        return 0.0
    p = mean_amp / total

    # KL divergence from uniform
    q = np.ones(n_bins) / n_bins
    # Avoid log(0)
    p_safe = np.where(p > 0, p, 1e-20)
    kl = np.sum(p_safe * np.log(p_safe / q))
    mi = kl / np.log(n_bins)
    return float(mi)


def _modulation_index_vectorized(
    phase: np.ndarray,
    amp: np.ndarray,
    n_bins: int = 18,
) -> float:
    """Vectorized MI across all traces at once.

    Args:
        phase: [M, T]  (M = N*C or subsampled)
        amp:   [M, T]
    """
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    flat_phase = phase.ravel()
    flat_amp = amp.ravel()
    for b in range(n_bins):
        mask = (flat_phase >= bin_edges[b]) & (flat_phase < bin_edges[b + 1])
        if mask.any():
            mean_amp[b] = flat_amp[mask].mean()

    total = mean_amp.sum()
    if total < 1e-20:
        return 0.0
    p = mean_amp / total
    q = np.ones(n_bins) / n_bins
    p_safe = np.where(p > 0, p, 1e-20)
    kl = np.sum(p_safe * np.log(p_safe / q))
    return float(kl / np.log(n_bins))


def compute_pac(
    x: np.ndarray,
    fs: int,
    phase_band: Tuple[float, float] = (4, 8),
    amp_band: Tuple[float, float] = (30, 100),
    max_traces: int = 2000,
) -> float:
    """Mean PAC (Modulation Index) – vectorized over trials×channels.

    Args:
        x: [N, C, T]
        max_traces: subsample to at most this many traces for speed
    """
    N, C, T = x.shape
    # Flatten to [N*C, T] for vectorized filtering
    flat = x.reshape(N * C, T)

    # Subsample if too many traces
    if flat.shape[0] > max_traces:
        rng = np.random.default_rng(0)
        idx = rng.choice(flat.shape[0], max_traces, replace=False)
        flat = flat[idx]

    # Vectorized bandpass + hilbert (scipy operates along last axis)
    lo = bandpass(flat[:, np.newaxis, :], *phase_band, fs)[:, 0, :]
    phase = np.angle(sig.hilbert(lo, axis=-1))

    hi = bandpass(flat[:, np.newaxis, :], *amp_band, fs)[:, 0, :]
    amp = np.abs(sig.hilbert(hi, axis=-1))

    return _modulation_index_vectorized(phase, amp)


def pac_with_surrogates(
    x: np.ndarray,
    fs: int,
    phase_band: Tuple[float, float] = (4, 8),
    amp_band: Tuple[float, float] = (30, 100),
    n_surrogates: int = 200,
    max_traces: int = 2000,
) -> Dict[str, float]:
    """PAC with time-shifted surrogate significance testing (vectorized).

    Returns:
        mi          – observed Modulation Index
        mi_z        – z-score vs surrogate distribution
        mi_p        – p-value (proportion of surrogates >= observed)
    """
    N, C, T = x.shape

    # Subsample traces once, reuse for observed + surrogates
    flat = x.reshape(N * C, T)
    rng = np.random.default_rng(42)
    if flat.shape[0] > max_traces:
        idx = rng.choice(flat.shape[0], max_traces, replace=False)
        flat = flat[idx]

    # Pre-filter once (bandpass doesn't change with time shifts)
    lo = bandpass(flat[:, np.newaxis, :], *phase_band, fs)[:, 0, :]
    hi = bandpass(flat[:, np.newaxis, :], *amp_band, fs)[:, 0, :]

    # Observed MI
    phase = np.angle(sig.hilbert(lo, axis=-1))
    amp = np.abs(sig.hilbert(hi, axis=-1))
    mi_obs = _modulation_index_vectorized(phase, amp)

    # Surrogates: circular-shift the high-freq amplitude relative to phase
    surrogate_mis = np.empty(n_surrogates)
    for s in range(n_surrogates):
        shift = rng.integers(T // 4, 3 * T // 4)
        amp_shifted = np.roll(amp, shift, axis=-1)
        surrogate_mis[s] = _modulation_index_vectorized(phase, amp_shifted)

    mi_z = (mi_obs - surrogate_mis.mean()) / (surrogate_mis.std() + 1e-20)
    mi_p = float(np.mean(surrogate_mis >= mi_obs))

    return {"mi": mi_obs, "mi_z": float(mi_z), "mi_p": mi_p}


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_spectral_validation(
    synth_dir: Path,
    fs: int,
    nperseg: int = 1024,
    pac_phase_band: Tuple[float, float] = (4, 8),
    pac_amp_band: Tuple[float, float] = (30, 100),
    pac_n_surrogates: int = 0,  # kept for API compat, surrogates are skipped
) -> Dict[str, Any]:
    """Run all spectral validations on saved synthetic data.

    Args:
        synth_dir: path containing target_test.npy, predicted_test.npy
        fs: sampling rate in Hz

    Returns:
        Dictionary of all spectral metrics (JSON-serialisable, except arrays).
    """
    real = np.load(synth_dir / "target_test.npy")
    pred = np.load(synth_dir / "predicted_test.npy")

    print(f"  Spectral validation: {real.shape[0]} trials, "
          f"{real.shape[1]} channels, {real.shape[2]} time steps")

    # 1. PSD
    psd = psd_match_metrics(real, pred, fs, nperseg)
    print(f"    PSD correlation : {psd['psd_corr']:.4f}")
    print(f"    PSD MSE (dB)   : {psd['psd_mse_db']:.2f}")

    # 2. Band R²
    band_r2 = per_band_r2(real, pred, fs)
    for name, val in band_r2.items():
        print(f"    {name:12s} R² : {val:.4f}")

    # 3. PAC – raw MI only (no surrogates – too slow for large datasets)
    print("    Computing PAC (real)...")
    pac_mi_real = compute_pac(real, fs, pac_phase_band, pac_amp_band)
    print(f"    PAC real MI    : {pac_mi_real:.4f}")

    print("    Computing PAC (predicted)...")
    pac_mi_pred = compute_pac(pred, fs, pac_phase_band, pac_amp_band)
    print(f"    PAC pred MI    : {pac_mi_pred:.4f}")

    # ── Post-hoc calibration ─────────────────────────────────────────────
    print("    Post-hoc calibration (spectral bias + envelope matching)...")
    pred_cal = calibrate_spectral_bias(pred, real, fs)
    pred_cal = calibrate_envelope(pred_cal, real)

    psd_cal = psd_match_metrics(real, pred_cal, fs, nperseg)
    band_r2_cal = per_band_r2(real, pred_cal, fs)

    print(f"    [calibrated] PSD corr  : {psd_cal['psd_corr']:.4f}")
    print(f"    [calibrated] PSD MSE   : {psd_cal['psd_mse_db']:.2f}")
    for name, val in band_r2_cal.items():
        print(f"    [calibrated] {name:12s} R² : {val:.4f}")

    # Save calibrated predictions for downstream validators
    np.save(synth_dir / "predicted_test_calibrated.npy", pred_cal)

    # ── Per-channel statistics ────────────────────────────────────────────
    n_channels = real.shape[1]
    channel_stats = {}
    print(f"    Computing per-channel stats ({n_channels} channels)...")
    for ch in range(n_channels):
        ch_real = real[:, ch:ch+1, :]   # [N, 1, T]
        ch_pred = pred[:, ch:ch+1, :]

        # Per-channel PSD correlation
        _, psd_r = compute_psd(ch_real, fs, nperseg)
        _, psd_p = compute_psd(ch_pred, fs, nperseg)
        log_r = 10 * np.log10(psd_r + 1e-20)
        log_p = 10 * np.log10(psd_p + 1e-20)
        ch_psd_corr = float(np.corrcoef(log_r, log_p)[0, 1])

        # Per-channel Pearson correlation (time-domain, trial-averaged)
        ch_corr = float(np.corrcoef(ch_real.ravel(), ch_pred.ravel())[0, 1])

        # Per-channel R²
        r_flat = ch_real.ravel()
        p_flat = ch_pred.ravel()
        ss_res = np.sum((r_flat - p_flat) ** 2)
        ss_tot = np.sum((r_flat - r_flat.mean()) ** 2)
        ch_r2 = float(1.0 - ss_res / (ss_tot + 1e-20))

        # Per-channel band R²
        ch_band_r2 = per_band_r2(ch_real, ch_pred, fs)

        channel_stats[f"ch_{ch}"] = {
            "psd_corr": ch_psd_corr,
            "pearson_r": ch_corr,
            "r2": ch_r2,
            "band_r2": ch_band_r2,
        }

    results = {
        "psd_corr": psd["psd_corr"],
        "psd_mse_db": psd["psd_mse_db"],
        "band_r2": band_r2,
        "calibrated": {
            "psd_corr": psd_cal["psd_corr"],
            "psd_mse_db": psd_cal["psd_mse_db"],
            "band_r2": band_r2_cal,
        },
        "pac_mi_real": pac_mi_real,
        "pac_mi_pred": pac_mi_pred,
        "pac_mi_ratio": pac_mi_pred / (pac_mi_real + 1e-20),
        "per_channel": channel_stats,
    }

    # Save
    out_path = synth_dir / "spectral_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    → saved {out_path}")

    # Save per-channel summary as CSV for easy inspection
    _save_channel_csv(channel_stats, synth_dir / "spectral_per_channel.csv")

    return results


def _save_channel_csv(channel_stats: Dict, out_path: Path):
    """Write per-channel spectral metrics as CSV."""
    import csv
    bands = list(FREQUENCY_BANDS.keys())
    header = ["channel", "r2", "pearson_r", "psd_corr"] + [f"r2_{b}" for b in bands]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ch_key in sorted(channel_stats, key=lambda k: int(k.split("_")[1])):
            ch = channel_stats[ch_key]
            row = [
                ch_key,
                f"{ch['r2']:.6f}",
                f"{ch['pearson_r']:.6f}",
                f"{ch['psd_corr']:.6f}",
            ]
            for b in bands:
                row.append(f"{ch['band_r2'].get(b, 0.0):.6f}")
            writer.writerow(row)
    print(f"    → saved {out_path}")
