"""Partial Information Decomposition (PID) of translated signals.

Decomposes the information that real and predicted target signals carry
about an external variable (trial label or future activity) into four
atoms, following the Williams-Beer (2010) framework with I_min (MMI)
redundancy:

    Redundancy  – information both real AND predicted carry (successfully translated)
    Unique_real – information only the real target carries  (lost in translation)
    Unique_pred – information only the prediction carries   (hallucinated / artifact)
    Synergy     – information that emerges only when combining both (complementary)

This directly answers: "what KIND of neural information does the
translation preserve vs lose?"

Two decomposition modes:
1. **Label-based PID** – sources = {real_target, predicted_target},
   target = trial label.  Works for datasets with discrete labels.
   Uses class-conditional Gaussian MI (discriminant approach) so
   discrete labels are handled correctly without one-hot hacks.
2. **Temporal PID** – sources = {real_target_t, predicted_target_t},
   target = real_target_{t+1}.  Works for all datasets (no labels needed).
   Respects trial boundaries (no cross-trial lag pairs).

MI estimation: Gaussian copula (Ince et al. 2017, Entropy 19(10):502)
for continuous variables; class-conditional Gaussian for discrete targets.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from phase_four.config import FREQUENCY_BANDS
from phase_four.validation.spectral import bandpass


# ═══════════════════════════════════════════════════════════════════════════
# Gaussian mutual information estimators
# ═══════════════════════════════════════════════════════════════════════════

def _mi_gaussian(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information between x and y assuming joint Gaussian.

    Uses I(X;Y) = 0.5 * log( det(Σ_X) * det(Σ_Y) / det(Σ_XY) ).
    Regularisation scales with the trace of the covariance to avoid
    numerical issues without distorting estimates.

    Args:
        x: [N, d1] or [N]
        y: [N, d2] or [N]

    Returns:
        MI in nats (non-negative by clamp).
    """
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    d1, d2 = x.shape[1], y.shape[1]
    d = d1 + d2

    xy = np.concatenate([x, y], axis=1)
    cov_xy = np.cov(xy, rowvar=False)
    if cov_xy.ndim == 0:
        cov_xy = cov_xy.reshape(1, 1)

    # Adaptive regularisation: 1e-10 * trace to scale with signal variance
    reg = 1e-10 * max(np.trace(cov_xy), 1e-20) * np.eye(d)
    cov_xy = cov_xy + reg

    cov_x = cov_xy[:d1, :d1]
    cov_y = cov_xy[d1:, d1:]

    _, logdet_x = np.linalg.slogdet(cov_x)
    _, logdet_y = np.linalg.slogdet(cov_y)
    _, logdet_xy = np.linalg.slogdet(cov_xy)

    mi = 0.5 * (logdet_x + logdet_y - logdet_xy)
    return max(float(mi), 0.0)


def _mi_class_conditional(x: np.ndarray, labels: np.ndarray) -> float:
    """MI between continuous x and discrete labels via class-conditional Gaussians.

    I(X; Y_discrete) = H(X) - H(X|Y) where H(X|Y) = Σ_c p(c) H(X|Y=c).
    For Gaussians: H(X) = 0.5 * log((2πe)^d * det(Σ)) so the (2πe)^d
    terms cancel and I = 0.5 * (logdet(Σ_total) - Σ_c p(c) logdet(Σ_c)).

    This correctly handles discrete targets without one-hot encoding.
    """
    if x.ndim == 1:
        x = x[:, None]

    n, d = x.shape
    unique_labels = np.unique(labels)

    cov_total = np.cov(x, rowvar=False)
    if cov_total.ndim == 0:
        cov_total = cov_total.reshape(1, 1)
    reg = 1e-10 * max(np.trace(cov_total), 1e-20) * np.eye(d)
    cov_total = cov_total + reg

    _, logdet_total = np.linalg.slogdet(cov_total)

    # Weighted sum of class-conditional log-determinants
    cond_logdet = 0.0
    for c in unique_labels:
        mask = labels == c
        n_c = mask.sum()
        if n_c < 2:
            continue
        p_c = n_c / n
        cov_c = np.cov(x[mask], rowvar=False)
        if cov_c.ndim == 0:
            cov_c = cov_c.reshape(1, 1)
        cov_c = cov_c + reg
        _, logdet_c = np.linalg.slogdet(cov_c)
        cond_logdet += p_c * logdet_c

    mi = 0.5 * (logdet_total - cond_logdet)
    return max(float(mi), 0.0)


def _copula_transform(x: np.ndarray) -> np.ndarray:
    """Gaussian copula (rank -> normal quantile) for each column.

    Following Ince et al. (2017): rank data, map to uniform (0,1) via
    r/(n+1), then invert the standard normal CDF.  This makes the
    marginals exactly Gaussian so that Gaussian MI estimation is valid
    for arbitrary marginal distributions.
    """
    if x.ndim == 1:
        x = x[:, None]
    out = np.zeros_like(x, dtype=np.float64)
    n = x.shape[0]
    for j in range(x.shape[1]):
        ranks = sp_stats.rankdata(x[:, j])
        u = ranks / (n + 1)
        out[:, j] = sp_stats.norm.ppf(u)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# PID decomposition (Williams-Beer with MMI redundancy)
# ═══════════════════════════════════════════════════════════════════════════

def _pid_from_mis(mi_s1: float, mi_s2: float, mi_joint: float) -> Dict[str, float]:
    """Compute PID atoms from the three MI values.

    Uses I_min (MMI) redundancy measure (Williams & Beer 2010):
        RED = min(I(S1;T), I(S2;T))

    The full decomposition:
        I(S1,S2;T) = RED + UNQ_S1 + UNQ_S2 + SYN
        UNQ_S1 = I(S1;T) - RED
        UNQ_S2 = I(S2;T) - RED
        SYN    = I(S1,S2;T) - I(S1;T) - I(S2;T) + RED

    Co-information:
        CI = I(S1;T) + I(S2;T) - I(S1,S2;T)
        CI > 0 => redundancy-dominated; CI < 0 => synergy-dominated.
    """
    redundancy = min(mi_s1, mi_s2)
    unique_s1 = mi_s1 - redundancy
    unique_s2 = mi_s2 - redundancy
    synergy = mi_joint - mi_s1 - mi_s2 + redundancy
    co_info = mi_s1 + mi_s2 - mi_joint

    return {
        "redundancy": float(redundancy),
        "unique_real": float(unique_s1),
        "unique_pred": float(unique_s2),
        "synergy": float(synergy),
        "mi_real": float(mi_s1),
        "mi_pred": float(mi_s2),
        "mi_joint": float(mi_joint),
        "co_information": float(co_info),
        "translation_efficiency": float(redundancy / (mi_s1 + 1e-20)),
        "hallucination_ratio": float(unique_s2 / (mi_s2 + 1e-20)),
        "complementarity_ratio": float(synergy / (mi_joint + 1e-20)),
    }


def pid_mmi_continuous(
    s1: np.ndarray,
    s2: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """PID for continuous target using Gaussian copula MI.

    Used for temporal PID where target = future real activity.
    """
    s1 = _copula_transform(s1)
    s2 = _copula_transform(s2)
    target = _copula_transform(target)

    mi_s1 = _mi_gaussian(s1, target)
    mi_s2 = _mi_gaussian(s2, target)
    s_joint = np.concatenate([s1, s2], axis=1)
    mi_joint = _mi_gaussian(s_joint, target)

    return _pid_from_mis(mi_s1, mi_s2, mi_joint)


def pid_mmi_discrete(
    s1: np.ndarray,
    s2: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """PID for discrete target (class labels) using class-conditional Gaussian MI.

    Properly handles discrete targets without one-hot encoding artifacts.
    Sources are copula-transformed for valid Gaussian MI estimation.
    """
    s1 = _copula_transform(s1)
    s2 = _copula_transform(s2)

    mi_s1 = _mi_class_conditional(s1, labels)
    mi_s2 = _mi_class_conditional(s2, labels)
    s_joint = np.concatenate([s1, s2], axis=1)
    mi_joint = _mi_class_conditional(s_joint, labels)

    return _pid_from_mis(mi_s1, mi_s2, mi_joint)


# ═══════════════════════════════════════════════════════════════════════════
# Feature extractors for PID sources
# ═══════════════════════════════════════════════════════════════════════════

def _band_power_features(x: np.ndarray, fs: int) -> np.ndarray:
    """Mean band power per channel per trial -> [N, n_bands * C].

    Layout: [band0_ch0, band0_ch1, ..., band1_ch0, band1_ch1, ...].
    """
    N, C, T = x.shape
    feats = []
    for name, (lo, hi) in FREQUENCY_BANDS.items():
        filtered = bandpass(x, lo, hi, fs)
        power = np.mean(filtered ** 2, axis=-1)  # [N, C]
        feats.append(power)
    return np.concatenate(feats, axis=1)


def _single_channel_band_powers(x_ch: np.ndarray, fs: int) -> np.ndarray:
    """Band powers for a single channel: x_ch [N, 1, T] -> [N, n_bands]."""
    feats = []
    for name, (lo, hi) in FREQUENCY_BANDS.items():
        filtered = bandpass(x_ch, lo, hi, fs)
        power = np.mean(filtered ** 2, axis=-1)  # [N, 1]
        feats.append(power)
    return np.concatenate(feats, axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Label-based PID
# ═══════════════════════════════════════════════════════════════════════════

def label_based_pid(
    real: np.ndarray,
    pred: np.ndarray,
    labels: np.ndarray,
    fs: int,
    bands: Optional[Dict[str, tuple]] = None,
) -> Dict[str, Any]:
    """PID decomposition: {real_target, predicted_target} -> trial label.

    Uses class-conditional Gaussian MI so discrete labels are handled
    correctly (no one-hot encoding needed).

    Args:
        real: [N, C, T] real target signals
        pred: [N, C, T] predicted target signals
        labels: [N] trial labels (int or str)
        fs: sampling rate in Hz

    Returns:
        Overall PID + per-band + per-channel PID breakdown.
    """
    if bands is None:
        bands = FREQUENCY_BANDS

    # Overall PID using band power features
    feats_real = _band_power_features(real, fs)
    feats_pred = _band_power_features(pred, fs)

    overall = pid_mmi_discrete(feats_real, feats_pred, labels)

    # Per-band PID
    per_band = {}
    for name, (lo, hi) in bands.items():
        real_band = bandpass(real, lo, hi, fs)
        pred_band = bandpass(pred, lo, hi, fs)

        real_bp = np.mean(real_band ** 2, axis=-1)  # [N, C]
        pred_bp = np.mean(pred_band ** 2, axis=-1)

        per_band[name] = pid_mmi_discrete(real_bp, pred_bp, labels)

    # Per-channel PID
    per_channel = []
    n_channels = real.shape[1]
    for ch in range(n_channels):
        ch_feats_r = _single_channel_band_powers(real[:, ch:ch+1, :], fs)
        ch_feats_p = _single_channel_band_powers(pred[:, ch:ch+1, :], fs)

        ch_pid = pid_mmi_discrete(ch_feats_r, ch_feats_p, labels)
        ch_pid["channel"] = ch
        per_channel.append(ch_pid)

    return {
        "overall": overall,
        "per_band": per_band,
        "per_channel": per_channel,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Temporal PID (no labels needed)
# ═══════════════════════════════════════════════════════════════════════════

def temporal_pid(
    real: np.ndarray,
    pred: np.ndarray,
    fs: int,
    lag_ms: int = 50,
) -> Dict[str, Any]:
    """PID decomposition: {real_t, pred_t} -> real_{t+lag}.

    Measures how well the predicted signal, combined with the current
    real signal, predicts the future real signal.

    Trial boundaries are respected: lag pairs are only constructed within
    each trial (no cross-trial contamination).

    Args:
        real: [N, C, T]
        pred: [N, C, T]
        fs: sampling rate in Hz
        lag_ms: prediction horizon in milliseconds

    Returns:
        Overall temporal PID + per-channel breakdown.
    """
    lag_samples = max(1, int(lag_ms * fs / 1000))
    N, C, T = real.shape
    T_eff = T - lag_samples

    if T_eff < 1:
        return {
            "lag_ms": lag_ms,
            "lag_samples": lag_samples,
            "n_samples": 0,
            "overall": {},
            "per_channel": [],
            "error": "lag exceeds trial length",
        }

    # Build lag pairs within each trial, then stack across trials
    # real_t[i] and real_future[i] always come from the same trial
    real_t_all = real[:, :, :T_eff].reshape(N, C, T_eff)
    pred_t_all = pred[:, :, :T_eff].reshape(N, C, T_eff)
    real_future = real[:, :, lag_samples:].reshape(N, C, T_eff)

    # Transpose to [N, T_eff, C] then reshape to [N*T_eff, C]
    # This preserves trial boundaries: rows 0..T_eff-1 are trial 0, etc.
    real_t_flat = real_t_all.transpose(0, 2, 1).reshape(-1, C)
    pred_t_flat = pred_t_all.transpose(0, 2, 1).reshape(-1, C)
    future_flat = real_future.transpose(0, 2, 1).reshape(-1, C)

    # Sub-sample for tractability
    max_samples = 100_000
    rng = np.random.default_rng(42)
    n_total = real_t_flat.shape[0]
    if n_total > max_samples:
        idx = rng.choice(n_total, max_samples, replace=False)
        real_t_flat = real_t_flat[idx]
        pred_t_flat = pred_t_flat[idx]
        future_flat = future_flat[idx]

    overall = pid_mmi_continuous(real_t_flat, pred_t_flat, future_flat)

    # Per-channel temporal PID
    per_channel = []
    for ch in range(C):
        r_t = real_t_flat[:, ch:ch+1]
        p_t = pred_t_flat[:, ch:ch+1]
        r_f = future_flat[:, ch:ch+1]
        ch_pid = pid_mmi_continuous(r_t, p_t, r_f)
        ch_pid["channel"] = ch
        per_channel.append(ch_pid)

    return {
        "lag_ms": lag_ms,
        "lag_samples": lag_samples,
        "n_samples": int(real_t_flat.shape[0]),
        "overall": overall,
        "per_channel": per_channel,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_pid_validation(
    synth_dir: Path,
    fs: int,
    lag_ms: int = 50,
) -> Dict[str, Any]:
    """Run PID analysis on saved synthetic data.

    Args:
        synth_dir: contains target_test.npy, predicted_test.npy, labels_test.npy
        fs: sampling rate in Hz

    Returns:
        Dict with label_pid and temporal_pid results.
    """
    real = np.load(synth_dir / "target_test.npy")
    pred = np.load(synth_dir / "predicted_test.npy")
    labels = np.load(synth_dir / "labels_test.npy")

    print(f"  PID validation: {real.shape[0]} trials, "
          f"{real.shape[1]} channels, {real.shape[2]} time steps")

    results = {}

    # ── Label-based PID ───────────────────────────────────────────────────
    unique_labels = np.unique(labels)
    if len(unique_labels) >= 2:
        print(f"    Label-based PID ({len(unique_labels)} classes)...")
        label_pid = label_based_pid(real, pred, labels, fs)
        results["label_pid"] = label_pid

        ovr = label_pid["overall"]
        print(f"    Overall:")
        print(f"      Redundancy (translated)    : {ovr['redundancy']:.4f} nats")
        print(f"      Unique_real (lost)          : {ovr['unique_real']:.4f} nats")
        print(f"      Unique_pred (hallucinated)  : {ovr['unique_pred']:.4f} nats")
        print(f"      Synergy (complementary)     : {ovr['synergy']:.4f} nats")
        print(f"      Translation efficiency      : {ovr['translation_efficiency']:.1%}")
        print(f"      Hallucination ratio          : {ovr['hallucination_ratio']:.1%}")

        print(f"    Per-band PID:")
        for band_name, band_pid in label_pid["per_band"].items():
            print(f"      {band_name:12s}: RED={band_pid['redundancy']:.4f}  "
                  f"UNQ_R={band_pid['unique_real']:.4f}  "
                  f"UNQ_P={band_pid['unique_pred']:.4f}  "
                  f"SYN={band_pid['synergy']:.4f}  "
                  f"eff={band_pid['translation_efficiency']:.1%}")
    else:
        print(f"    Label-based PID skipped (only {len(unique_labels)} class)")
        results["label_pid"] = None

    # ── Temporal PID ──────────────────────────────────────────────────────
    print(f"    Temporal PID (lag={lag_ms}ms)...")
    temporal = temporal_pid(real, pred, fs, lag_ms)
    results["temporal_pid"] = temporal

    ovr_t = temporal["overall"]
    print(f"    Overall temporal PID:")
    print(f"      Redundancy (translated)    : {ovr_t['redundancy']:.4f} nats")
    print(f"      Unique_real (lost)          : {ovr_t['unique_real']:.4f} nats")
    print(f"      Unique_pred (hallucinated)  : {ovr_t['unique_pred']:.4f} nats")
    print(f"      Synergy (complementary)     : {ovr_t['synergy']:.4f} nats")
    print(f"      Translation efficiency      : {ovr_t['translation_efficiency']:.1%}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_path = synth_dir / "pid_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    → saved {out_path}")

    # ── Save per-channel CSV ──────────────────────────────────────────────
    _save_pid_channel_csv(results, synth_dir)

    # ── Save per-band CSV ─────────────────────────────────────────────────
    _save_pid_band_csv(results, synth_dir)

    return results


def _save_pid_channel_csv(results: Dict, synth_dir: Path):
    """Export per-channel PID as CSV."""
    rows = []
    header = [
        "channel", "source", "redundancy", "unique_real", "unique_pred",
        "synergy", "mi_real", "mi_pred", "mi_joint",
        "translation_efficiency", "hallucination_ratio", "complementarity_ratio",
    ]

    # Label-based per-channel
    if results.get("label_pid") and results["label_pid"].get("per_channel"):
        for ch in results["label_pid"]["per_channel"]:
            rows.append(["label"] + [ch.get(k, "") for k in header[1:]])
            rows[-1][1] = f"ch_{ch['channel']}"

    # Temporal per-channel
    if results.get("temporal_pid") and results["temporal_pid"].get("per_channel"):
        for ch in results["temporal_pid"]["per_channel"]:
            rows.append(["temporal"] + [ch.get(k, "") for k in header[1:]])
            rows[-1][1] = f"ch_{ch['channel']}"

    if not rows:
        return

    out_path = synth_dir / "pid_per_channel.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pid_type"] + header)
        for row in rows:
            writer.writerow([row[0]] + [
                f"{v:.6f}" if isinstance(v, float) else str(v) for v in row[1:]
            ])
    print(f"    → saved {out_path}")


def _save_pid_band_csv(results: Dict, synth_dir: Path):
    """Export per-band PID as CSV."""
    label_pid = results.get("label_pid")
    if not label_pid or not label_pid.get("per_band"):
        return

    out_path = synth_dir / "pid_per_band.csv"
    header = [
        "band", "redundancy", "unique_real", "unique_pred", "synergy",
        "mi_real", "mi_pred", "mi_joint",
        "translation_efficiency", "hallucination_ratio", "complementarity_ratio",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for band_name, band_pid in label_pid["per_band"].items():
            writer.writerow([band_name] + [
                f"{band_pid.get(k, 0.0):.6f}" for k in header[1:]
            ])
    print(f"    → saved {out_path}")
