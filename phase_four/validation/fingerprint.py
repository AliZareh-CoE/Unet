"""Session / subject fingerprinting from translated signals.

Measures whether NeuroGate-translated signals preserve individual
session or subject identity — the neural translation analogue of
the "identifiability index" from Amico & Goñi (2018).

Protocol
--------
For each trial, extract a feature vector (channel correlation matrix +
band powers).  Then compute:

    identifiability = mean(within-session similarity)
                    - mean(between-session similarity)

If the predicted signals are identifiable, the translation preserves
session-specific neural signatures — not just population-level structure.

Outputs
-------
- Identifiability index for: source, real target, predicted target
- Per-session similarity matrices (for source, real, predicted)
- Confusion matrix: given a predicted trial, which session is it most
  similar to?
- All results as JSON + CSV (no figures)
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction for fingerprinting
# ═══════════════════════════════════════════════════════════════════════════

def _channel_correlation_features(x: np.ndarray) -> np.ndarray:
    """Extract upper-triangle of channel correlation matrix per trial.

    Args:
        x: [N, C, T]

    Returns:
        features: [N, C*(C-1)/2]
    """
    N, C, T = x.shape
    n_features = C * (C - 1) // 2
    features = np.zeros((N, n_features))

    for trial in range(N):
        corr = np.corrcoef(x[trial])  # [C, C]
        # Handle NaN from constant channels
        corr = np.nan_to_num(corr, nan=0.0)
        idx = np.triu_indices(C, k=1)
        features[trial] = corr[idx]

    return features


def _band_power_features(x: np.ndarray, fs: int) -> np.ndarray:
    """Mean band power per channel per trial → [N, C * n_bands]."""
    from phase_four.validation.spectral import bandpass
    from phase_four.config import FREQUENCY_BANDS

    N, C, T = x.shape
    feats = []
    for name, (lo, hi) in FREQUENCY_BANDS.items():
        filtered = bandpass(x, lo, hi, fs)
        power = np.mean(filtered ** 2, axis=-1)  # [N, C]
        feats.append(power)
    return np.concatenate(feats, axis=1)


def extract_fingerprint_features(x: np.ndarray, fs: int) -> np.ndarray:
    """Combined feature vector: channel correlations + band powers."""
    cc = _channel_correlation_features(x)
    bp = _band_power_features(x, fs)
    return np.concatenate([cc, bp], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# Identifiability computation
# ═══════════════════════════════════════════════════════════════════════════

def _cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity [N, N] from features [N, D]."""
    # Normalise rows
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-20
    normed = features / norms
    return normed @ normed.T


def compute_identifiability(
    features: np.ndarray,
    session_ids: np.ndarray,
) -> Dict[str, Any]:
    """Compute the identifiability index (Amico & Goñi 2018).

    identifiability = mean(within-session similarity) - mean(between-session similarity)

    Also computes:
    - Per-session mean feature (template)
    - Identification accuracy: for each trial, is the nearest-template
      the correct session?

    Args:
        features: [N, D]
        session_ids: [N] integer session labels

    Returns:
        Dict with identifiability, accuracy, and per-session stats.
    """
    N = features.shape[0]
    sim = _cosine_similarity_matrix(features)

    unique_sessions = np.unique(session_ids)
    n_sessions = len(unique_sessions)

    # Within vs between session similarity
    within_sims = []
    between_sims = []
    for i in range(N):
        for j in range(i + 1, N):
            if session_ids[i] == session_ids[j]:
                within_sims.append(sim[i, j])
            else:
                between_sims.append(sim[i, j])

    within_mean = float(np.mean(within_sims)) if within_sims else 0.0
    between_mean = float(np.mean(between_sims)) if between_sims else 0.0
    identifiability = within_mean - between_mean

    # Template-based identification
    # Build session templates (mean feature per session)
    templates = {}
    for sess in unique_sessions:
        mask = session_ids == sess
        templates[int(sess)] = features[mask].mean(axis=0)

    template_matrix = np.array([templates[int(s)] for s in unique_sessions])  # [n_sess, D]
    template_norms = np.linalg.norm(template_matrix, axis=1, keepdims=True) + 1e-20
    template_normed = template_matrix / template_norms

    # For each trial, find nearest template
    feat_norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-20
    feat_normed = features / feat_norms
    sims_to_templates = feat_normed @ template_normed.T  # [N, n_sess]

    predicted_sessions = unique_sessions[np.argmax(sims_to_templates, axis=1)]
    correct = (predicted_sessions == session_ids).astype(float)
    identification_accuracy = float(correct.mean())

    # Per-session stats
    per_session = []
    for sess in unique_sessions:
        mask = session_ids == sess
        n_trials = int(mask.sum())
        sess_acc = float(correct[mask].mean()) if n_trials > 0 else 0.0

        # Within-session similarity (mean pairwise)
        sess_idx = np.where(mask)[0]
        if len(sess_idx) > 1:
            ws = [sim[sess_idx[a], sess_idx[b]]
                  for a in range(len(sess_idx)) for b in range(a + 1, len(sess_idx))]
            ws_mean = float(np.mean(ws))
        else:
            ws_mean = 0.0

        per_session.append({
            "session": int(sess),
            "n_trials": n_trials,
            "identification_accuracy": sess_acc,
            "within_session_similarity": ws_mean,
        })

    return {
        "identifiability": identifiability,
        "within_session_similarity": within_mean,
        "between_session_similarity": between_mean,
        "identification_accuracy": identification_accuracy,
        "n_sessions": n_sessions,
        "n_trials": N,
        "chance_level": 1.0 / n_sessions if n_sessions > 0 else 0.0,
        "per_session": per_session,
    }


def compute_cross_domain_identification(
    features_real: np.ndarray,
    features_pred: np.ndarray,
    session_ids: np.ndarray,
) -> Dict[str, float]:
    """Cross-domain identification: build templates from real, identify predicted.

    This is the strongest test: can we identify the session of a *predicted*
    trial using templates built from *real* trials?

    Args:
        features_real: [N, D] features from real target signals
        features_pred: [N, D] features from predicted target signals
        session_ids: [N]

    Returns:
        Dict with cross-domain accuracy and stats.
    """
    unique_sessions = np.unique(session_ids)
    n_sessions = len(unique_sessions)

    # Build templates from real
    templates = {}
    for sess in unique_sessions:
        mask = session_ids == sess
        templates[int(sess)] = features_real[mask].mean(axis=0)

    template_matrix = np.array([templates[int(s)] for s in unique_sessions])
    template_norms = np.linalg.norm(template_matrix, axis=1, keepdims=True) + 1e-20
    template_normed = template_matrix / template_norms

    # Identify predicted trials
    pred_norms = np.linalg.norm(features_pred, axis=1, keepdims=True) + 1e-20
    pred_normed = features_pred / pred_norms
    sims = pred_normed @ template_normed.T
    predicted_sessions = unique_sessions[np.argmax(sims, axis=1)]
    accuracy = float((predicted_sessions == session_ids).mean())

    # Confusion matrix
    confusion = np.zeros((n_sessions, n_sessions), dtype=int)
    sess_to_idx = {int(s): i for i, s in enumerate(unique_sessions)}
    for true_s, pred_s in zip(session_ids, predicted_sessions):
        confusion[sess_to_idx[int(true_s)], sess_to_idx[int(pred_s)]] += 1

    return {
        "cross_domain_accuracy": accuracy,
        "chance_level": 1.0 / n_sessions,
        "n_sessions": n_sessions,
        "confusion_matrix": confusion.tolist(),
        "session_labels": [int(s) for s in unique_sessions],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_fingerprint_validation(
    synth_dir: Path,
    fs: int,
    session_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run session fingerprinting on saved synthetic data.

    If session_ids is not provided, attempts to load from metadata or
    uses labels as a proxy.

    Args:
        synth_dir: contains source_test.npy, target_test.npy,
                   predicted_test.npy, labels_test.npy
        fs: sampling rate in Hz
        session_ids: optional [N] array of session/subject identifiers

    Returns:
        Dict with fingerprinting metrics for source, real, predicted.
    """
    source = np.load(synth_dir / "source_test.npy")
    real = np.load(synth_dir / "target_test.npy")
    pred = np.load(synth_dir / "predicted_test.npy")
    labels = np.load(synth_dir / "labels_test.npy")

    # Use labels as session proxy if no session_ids provided
    if session_ids is None:
        # Try loading from metadata
        meta_path = synth_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if "session_ids" in meta:
                session_ids = np.array(meta["session_ids"])

    if session_ids is None:
        # Fall back to labels
        session_ids = labels.copy()

    unique_sessions = np.unique(session_ids)
    if len(unique_sessions) < 2:
        msg = f"  Fingerprinting skipped: only {len(unique_sessions)} session(s)"
        print(msg)
        results = {"skipped": True, "reason": msg}
        with open(synth_dir / "fingerprint_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    print(f"  Fingerprinting: {real.shape[0]} trials, "
          f"{len(unique_sessions)} sessions, {real.shape[1]} channels")

    # ── Extract features ──────────────────────────────────────────────────
    print("    Extracting features (source)...")
    feats_source = extract_fingerprint_features(source, fs)
    print("    Extracting features (real target)...")
    feats_real = extract_fingerprint_features(real, fs)
    print("    Extracting features (predicted target)...")
    feats_pred = extract_fingerprint_features(pred, fs)
    print(f"    Feature dim: {feats_source.shape[1]}")

    # ── Identifiability for each signal type ──────────────────────────────
    print("    Computing identifiability (source)...")
    id_source = compute_identifiability(feats_source, session_ids)
    print(f"      Source identifiability : {id_source['identifiability']:.4f}  "
          f"(acc={id_source['identification_accuracy']:.1%})")

    print("    Computing identifiability (real target)...")
    id_real = compute_identifiability(feats_real, session_ids)
    print(f"      Real identifiability   : {id_real['identifiability']:.4f}  "
          f"(acc={id_real['identification_accuracy']:.1%})")

    print("    Computing identifiability (predicted target)...")
    id_pred = compute_identifiability(feats_pred, session_ids)
    print(f"      Pred identifiability   : {id_pred['identifiability']:.4f}  "
          f"(acc={id_pred['identification_accuracy']:.1%})")

    # ── Cross-domain identification ───────────────────────────────────────
    print("    Cross-domain: real templates → identify predicted...")
    cross = compute_cross_domain_identification(feats_real, feats_pred, session_ids)
    print(f"      Cross-domain accuracy  : {cross['cross_domain_accuracy']:.1%}  "
          f"(chance={cross['chance_level']:.1%})")

    # ── Preservation ratio ────────────────────────────────────────────────
    preservation = id_pred["identifiability"] / (id_real["identifiability"] + 1e-20)

    results = {
        "source": id_source,
        "real_target": id_real,
        "predicted_target": id_pred,
        "cross_domain": cross,
        "preservation_ratio": float(preservation),
        "n_sessions": len(unique_sessions),
        "n_trials": int(real.shape[0]),
        "feature_dim": int(feats_source.shape[1]),
    }

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_path = synth_dir / "fingerprint_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    → saved {out_path}")

    # ── Save per-session CSV ──────────────────────────────────────────────
    _save_session_csv(results, synth_dir)

    # ── Save summary CSV ──────────────────────────────────────────────────
    _save_summary_csv(results, synth_dir)

    return results


def _save_session_csv(results: Dict, synth_dir: Path):
    """Export per-session fingerprint stats as CSV."""
    header = [
        "signal_type", "session", "n_trials",
        "identification_accuracy", "within_session_similarity",
    ]

    out_path = synth_dir / "fingerprint_per_session.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for sig_type in ("source", "real_target", "predicted_target"):
            data = results.get(sig_type, {})
            if not data or "per_session" not in data:
                continue
            for sess in data["per_session"]:
                writer.writerow([
                    sig_type,
                    sess["session"],
                    sess["n_trials"],
                    f"{sess['identification_accuracy']:.6f}",
                    f"{sess['within_session_similarity']:.6f}",
                ])

    print(f"    → saved {out_path}")


def _save_summary_csv(results: Dict, synth_dir: Path):
    """Export summary identifiability comparison as CSV."""
    header = [
        "signal_type", "identifiability", "within_sim", "between_sim",
        "identification_accuracy", "chance_level",
    ]

    out_path = synth_dir / "fingerprint_summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for sig_type in ("source", "real_target", "predicted_target"):
            data = results.get(sig_type, {})
            if not data or "identifiability" not in data:
                continue
            writer.writerow([
                sig_type,
                f"{data['identifiability']:.6f}",
                f"{data['within_session_similarity']:.6f}",
                f"{data['between_session_similarity']:.6f}",
                f"{data['identification_accuracy']:.6f}",
                f"{data['chance_level']:.6f}",
            ])

        # Cross-domain row
        cross = results.get("cross_domain", {})
        if cross and "cross_domain_accuracy" in cross:
            writer.writerow([
                "cross_domain_real2pred",
                "",  # no identifiability for cross-domain
                "", "",
                f"{cross['cross_domain_accuracy']:.6f}",
                f"{cross['chance_level']:.6f}",
            ])

    print(f"    → saved {out_path}")
