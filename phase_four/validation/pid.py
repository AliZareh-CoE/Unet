"""Partial Information Decomposition (PID) of translated signals.

Decomposes the information that real and predicted target signals carry
about an external variable (trial label or future activity) into four
atoms, following the Williams-Beer framework:

    Redundancy  – information both real AND predicted carry (successfully translated)
    Unique_real – information only the real target carries  (lost in translation)
    Unique_pred – information only the prediction carries   (hallucinated / artifact)
    Synergy     – information that emerges only when combining both (complementary)

This directly answers: "what KIND of neural information does the
translation preserve vs lose?"

Two decomposition modes:
1. **Label-based PID** – sources = {real_target, predicted_target},
   target = trial label.  Works for datasets with discrete labels.
2. **Temporal PID** – sources = {real_target_t, predicted_target_t},
   target = real_target_{t+1}.  Works for all datasets (no labels needed).

All estimates use a Gaussian copula for tractability (Ince et al. 2017).
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

    Args:
        x: [N, d1] or [N]
        y: [N, d2] or [N]

    Returns:
        MI in nats
    """
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    n = x.shape[0]
    d1, d2 = x.shape[1], y.shape[1]

    # Regularise for numerical stability
    eps = 1e-8 * np.eye(d1 + d2)

    xy = np.concatenate([x, y], axis=1)
    cov_xy = np.cov(xy, rowvar=False) + eps
    cov_x = cov_xy[:d1, :d1]
    cov_y = cov_xy[d1:, d1:]

    # I(X;Y) = 0.5 * log( det(Σ_X) * det(Σ_Y) / det(Σ_XY) )
    sign_x, logdet_x = np.linalg.slogdet(cov_x)
    sign_y, logdet_y = np.linalg.slogdet(cov_y)
    sign_xy, logdet_xy = np.linalg.slogdet(cov_xy)

    mi = 0.5 * (logdet_x + logdet_y - logdet_xy)
    return max(float(mi), 0.0)


def _copula_transform(x: np.ndarray) -> np.ndarray:
    """Gaussian copula (rank → normal quantile) for each column."""
    if x.ndim == 1:
        x = x[:, None]
    out = np.zeros_like(x, dtype=np.float64)
    n = x.shape[0]
    for j in range(x.shape[1]):
        ranks = sp_stats.rankdata(x[:, j])
        # Map to (0, 1) then to standard normal quantile
        u = ranks / (n + 1)
        out[:, j] = sp_stats.norm.ppf(u)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# PID decomposition (Williams-Beer with MMI redundancy)
# ═══════════════════════════════════════════════════════════════════════════

def pid_mmi(
    s1: np.ndarray,
    s2: np.ndarray,
    target: np.ndarray,
    use_copula: bool = True,
) -> Dict[str, float]:
    """Partial information decomposition using minimum MI (MMI) redundancy.

    Sources: s1, s2  (e.g., real target features, predicted target features)
    Target:  target  (e.g., trial label embedding, future activity)

    Returns dict with: redundancy, unique_s1, unique_s2, synergy,
                       mi_s1, mi_s2, mi_joint, co_information
    """
    if use_copula:
        s1 = _copula_transform(s1)
        s2 = _copula_transform(s2)
        target = _copula_transform(target)

    mi_s1 = _mi_gaussian(s1, target)         # I(real; label)
    mi_s2 = _mi_gaussian(s2, target)         # I(pred; label)

    s_joint = np.concatenate([s1, s2], axis=1) if s1.ndim > 1 else np.stack([s1, s2], axis=1)
    mi_joint = _mi_gaussian(s_joint, target)  # I(real, pred; label)

    # MMI redundancy (Minimum Mutual Information)
    redundancy = min(mi_s1, mi_s2)

    # Unique information
    unique_s1 = mi_s1 - redundancy   # lost in translation
    unique_s2 = mi_s2 - redundancy   # hallucinated / noise
    synergy = mi_joint - mi_s1 - mi_s2 + redundancy

    # Co-information (positive = redundancy-dominated, negative = synergy-dominated)
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
        # Derived ratios
        "translation_efficiency": float(redundancy / (mi_s1 + 1e-20)),
        "hallucination_ratio": float(unique_s2 / (mi_s2 + 1e-20)),
        "complementarity_ratio": float(synergy / (mi_joint + 1e-20)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Feature extractors for PID sources
# ═══════════════════════════════════════════════════════════════════════════

def _band_power_features(x: np.ndarray, fs: int) -> np.ndarray:
    """Mean band power per channel per trial → [N, C*n_bands]."""
    N, C, T = x.shape
    feats = []
    for name, (lo, hi) in FREQUENCY_BANDS.items():
        filtered = bandpass(x, lo, hi, fs)
        power = np.mean(filtered ** 2, axis=-1)  # [N, C]
        feats.append(power)
    return np.concatenate(feats, axis=1)


def _label_to_embedding(labels: np.ndarray) -> np.ndarray:
    """Convert integer labels to one-hot embedding [N, n_classes]."""
    unique = np.unique(labels)
    n_classes = len(unique)
    label_map = {v: i for i, v in enumerate(unique)}
    one_hot = np.zeros((len(labels), n_classes))
    for i, lbl in enumerate(labels):
        one_hot[i, label_map[lbl]] = 1.0
    return one_hot


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
    """PID decomposition: {real_target, predicted_target} → trial label.

    Args:
        real: [N, C, T] real target signals
        pred: [N, C, T] predicted target signals
        labels: [N] integer trial labels

    Returns:
        Overall PID + per-band PID breakdown.
    """
    if bands is None:
        bands = FREQUENCY_BANDS

    target_emb = _label_to_embedding(labels)

    # Overall PID using band power features
    feats_real = _band_power_features(real, fs)
    feats_pred = _band_power_features(pred, fs)

    overall = pid_mmi(feats_real, feats_pred, target_emb)

    # Per-band PID
    per_band = {}
    for name, (lo, hi) in bands.items():
        real_band = bandpass(real, lo, hi, fs)
        pred_band = bandpass(pred, lo, hi, fs)

        # Use mean power per channel as features
        real_bp = np.mean(real_band ** 2, axis=-1)  # [N, C]
        pred_bp = np.mean(pred_band ** 2, axis=-1)

        per_band[name] = pid_mmi(real_bp, pred_bp, target_emb)

    # Per-channel PID
    per_channel = []
    n_channels = real.shape[1]
    n_bands = len(bands)
    for ch in range(n_channels):
        # All band powers for this channel
        ch_feats_real = feats_real[:, ch::n_channels]  # every n_channels-th column
        ch_feats_pred = feats_pred[:, ch::n_channels]

        # Simpler: just use band powers for this channel
        ch_bp_real = []
        ch_bp_pred = []
        for bname, (lo, hi) in bands.items():
            r_band = bandpass(real[:, ch:ch+1, :], lo, hi, fs)
            p_band = bandpass(pred[:, ch:ch+1, :], lo, hi, fs)
            ch_bp_real.append(np.mean(r_band ** 2, axis=-1))  # [N, 1]
            ch_bp_pred.append(np.mean(p_band ** 2, axis=-1))
        ch_feats_r = np.concatenate(ch_bp_real, axis=1)  # [N, n_bands]
        ch_feats_p = np.concatenate(ch_bp_pred, axis=1)

        ch_pid = pid_mmi(ch_feats_r, ch_feats_p, target_emb)
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
    """PID decomposition: {real_t, pred_t} → real_{t+lag}.

    Measures how well the predicted signal, combined with the current
    real signal, predicts the future real signal.

    Args:
        real: [N, C, T]
        pred: [N, C, T]
        lag_ms: prediction horizon in milliseconds

    Returns:
        Overall temporal PID + per-channel breakdown.
    """
    lag_samples = max(1, int(lag_ms * fs / 1000))

    # Use channel means for tractability
    # real_t, pred_t → real_{t+lag}
    N, C, T = real.shape
    T_eff = T - lag_samples

    # Flatten trials × time → samples (sub-sample for speed)
    max_samples = 100_000
    rng = np.random.default_rng(42)

    real_t_all = real[:, :, :T_eff].reshape(-1, C)     # [N*T_eff, C]
    pred_t_all = pred[:, :, :T_eff].reshape(-1, C)
    real_future = real[:, :, lag_samples:].reshape(-1, C)

    n_total = real_t_all.shape[0]
    if n_total > max_samples:
        idx = rng.choice(n_total, max_samples, replace=False)
        real_t_all = real_t_all[idx]
        pred_t_all = pred_t_all[idx]
        real_future = real_future[idx]

    overall = pid_mmi(real_t_all, pred_t_all, real_future)

    # Per-channel temporal PID
    per_channel = []
    for ch in range(C):
        r_t = real_t_all[:, ch:ch+1]
        p_t = pred_t_all[:, ch:ch+1]
        r_f = real_future[:, ch:ch+1]
        ch_pid = pid_mmi(r_t, p_t, r_f)
        ch_pid["channel"] = ch
        per_channel.append(ch_pid)

    return {
        "lag_ms": lag_ms,
        "lag_samples": lag_samples,
        "n_samples": int(real_t_all.shape[0]),
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
