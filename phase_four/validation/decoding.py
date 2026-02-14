"""Single-trial decoding validation.

The key question: does the translated signal preserve enough task-relevant
information to decode the experimental condition?

Protocol
--------
For each dataset with discrete labels (odor, trial type, …):

1. Extract features from *real* target signals  → train classifier → accuracy_real
2. Extract features from *predicted* target signals → test **same** classifier → accuracy_pred
3. Train classifier on *predicted* → test on *real* → accuracy_cross

If accuracy_pred ≈ accuracy_real, the UNet preserves task-relevant structure.

Feature extraction
------------------
- Mean band power in each FREQUENCY_BAND per channel  → C × n_bands features
- Temporal PCA (top-k components per channel)          → C × k features
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from phase_four.config import FREQUENCY_BANDS
from phase_four.validation.spectral import bandpass


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_band_power_features(
    x: np.ndarray,
    fs: int,
    bands: Optional[Dict[str, tuple]] = None,
) -> np.ndarray:
    """Extract mean band power per channel per trial.

    Args:
        x: [N, C, T]
        fs: sampling rate

    Returns:
        features [N, C * n_bands]
    """
    if bands is None:
        bands = FREQUENCY_BANDS

    N, C, T = x.shape
    feats = []
    for name, (lo, hi) in bands.items():
        filtered = bandpass(x, lo, hi, fs)
        # Mean power per channel per trial
        power = np.mean(filtered ** 2, axis=-1)  # [N, C]
        feats.append(power)

    return np.concatenate(feats, axis=1)  # [N, C * n_bands]


def extract_temporal_features(
    x: np.ndarray,
    n_components: int = 5,
) -> np.ndarray:
    """Extract temporal PCA features per trial.

    Concatenates the first n_components of SVD across all channels.

    Args:
        x: [N, C, T]

    Returns:
        features [N, C * n_components]
    """
    N, C, T = x.shape
    n_components = min(n_components, T)
    feats = np.zeros((N, C * n_components))

    for trial in range(N):
        for ch in range(C):
            trace = x[trial, ch]
            # Simple: use the first n_components of the DFT magnitude
            fft_mag = np.abs(np.fft.rfft(trace))[:n_components]
            if len(fft_mag) < n_components:
                fft_mag = np.pad(fft_mag, (0, n_components - len(fft_mag)))
            feats[trial, ch * n_components:(ch + 1) * n_components] = fft_mag

    return feats


def extract_features(x: np.ndarray, fs: int) -> np.ndarray:
    """Combined feature extraction: band powers + temporal."""
    bp = extract_band_power_features(x, fs)
    tp = extract_temporal_features(x, n_components=5)
    return np.concatenate([bp, tp], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# Classifiers
# ═══════════════════════════════════════════════════════════════════════════

CLASSIFIERS = {
    "lda": lambda: LinearDiscriminantAnalysis(),
    "svm": lambda: SVC(kernel="rbf", C=1.0, gamma="scale"),
    "rf": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
}


def _run_cv(
    clf_factory,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> float:
    """Stratified k-fold cross-validated balanced accuracy."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = clf_factory()
        clf.fit(X_train, y[train_idx])
        pred = clf.predict(X_test)
        accs.append(balanced_accuracy_score(y[test_idx], pred))
    return float(np.mean(accs))


def _run_cross_domain(
    clf_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Train on one domain, test on another."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = clf_factory()
    clf.fit(X_tr, y_train)
    pred = clf.predict(X_te)
    return float(balanced_accuracy_score(y_test, pred))


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_decoding_validation(
    synth_dir: Path,
    fs: int,
    classifier_names: List[str] = ("lda", "svm", "rf"),
    n_cv_folds: int = 5,
) -> Dict[str, Any]:
    """Run single-trial decoding on real vs predicted targets.

    Args:
        synth_dir: contains target_test.npy, predicted_test.npy, labels_test.npy
        fs: sampling rate

    Returns:
        Nested dict: {classifier: {acc_real, acc_pred, acc_cross_real2pred, ...}}
    """
    target = np.load(synth_dir / "target_test.npy")
    pred = np.load(synth_dir / "predicted_test.npy")
    labels = np.load(synth_dir / "labels_test.npy")

    # Check if labels are usable
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        msg = (f"  Decoding skipped: only {len(unique_labels)} unique label(s). "
               "Need >= 2 classes for classification.")
        print(msg)
        results = {"skipped": True, "reason": msg}
        with open(synth_dir / "decoding_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    n_classes = len(unique_labels)
    chance = 1.0 / n_classes
    print(f"  Decoding validation: {target.shape[0]} trials, "
          f"{n_classes} classes (chance = {chance:.1%})")

    # Extract features
    print("    Extracting features (real)...")
    feats_real = extract_features(target, fs)
    print("    Extracting features (predicted)...")
    feats_pred = extract_features(pred, fs)
    print(f"    Feature dim: {feats_real.shape[1]}")

    results = {"chance_level": chance, "n_classes": n_classes, "classifiers": {}}

    for clf_name in classifier_names:
        if clf_name not in CLASSIFIERS:
            print(f"    Unknown classifier: {clf_name}, skipping")
            continue

        factory = CLASSIFIERS[clf_name]
        print(f"    {clf_name.upper()}:")

        # 1. CV on real targets
        acc_real = _run_cv(factory, feats_real, labels, n_cv_folds)
        print(f"      Real targets (CV)       : {acc_real:.1%}")

        # 2. CV on predicted targets
        acc_pred = _run_cv(factory, feats_pred, labels, n_cv_folds)
        print(f"      Predicted targets (CV)   : {acc_pred:.1%}")

        # 3. Train on real, test on predicted
        acc_real2pred = _run_cross_domain(factory, feats_real, labels, feats_pred, labels)
        print(f"      Real → Predicted         : {acc_real2pred:.1%}")

        # 4. Train on predicted, test on real
        acc_pred2real = _run_cross_domain(factory, feats_pred, labels, feats_real, labels)
        print(f"      Predicted → Real         : {acc_pred2real:.1%}")

        results["classifiers"][clf_name] = {
            "acc_real_cv": acc_real,
            "acc_pred_cv": acc_pred,
            "acc_real_to_pred": acc_real2pred,
            "acc_pred_to_real": acc_pred2real,
            "preservation_ratio": acc_pred / (acc_real + 1e-20),
        }

    # ── Per-channel decoding (single-channel ablation) ────────────────────
    # For each channel, decode using only that channel's features to measure
    # per-channel contribution to task decoding.
    n_channels = target.shape[1]
    n_bands = len(FREQUENCY_BANDS)
    n_temporal = 5  # matches extract_temporal_features n_components
    feats_per_ch = n_bands + n_temporal

    print(f"    Per-channel decoding ({n_channels} channels)...")
    best_clf_name = max(
        [c for c in classifier_names if c in CLASSIFIERS],
        key=lambda c: results["classifiers"].get(c, {}).get("acc_real_cv", 0),
    )
    best_factory = CLASSIFIERS[best_clf_name]

    per_channel_decoding = []
    for ch in range(n_channels):
        start = ch * n_bands
        end_bp = start + n_bands
        # Band power features for this channel
        start_tp = n_channels * n_bands + ch * n_temporal
        end_tp = start_tp + n_temporal
        ch_feats_real = np.concatenate([feats_real[:, start:end_bp], feats_real[:, start_tp:end_tp]], axis=1)
        ch_feats_pred = np.concatenate([feats_pred[:, start:end_bp], feats_pred[:, start_tp:end_tp]], axis=1)

        acc_r = _run_cv(best_factory, ch_feats_real, labels, n_cv_folds)
        acc_p = _run_cv(best_factory, ch_feats_pred, labels, n_cv_folds)
        per_channel_decoding.append({
            "channel": ch,
            "acc_real": acc_r,
            "acc_pred": acc_p,
            "preservation": acc_p / (acc_r + 1e-20),
        })

    results["per_channel_decoding"] = per_channel_decoding
    results["per_channel_classifier"] = best_clf_name

    out_path = synth_dir / "decoding_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    → saved {out_path}")

    # CSV export
    _save_decoding_channel_csv(per_channel_decoding, synth_dir / "decoding_per_channel.csv")

    return results


def _save_decoding_channel_csv(per_channel: List[Dict], out_path: Path):
    """Write per-channel decoding metrics as CSV."""
    import csv
    header = ["channel", "acc_real", "acc_pred", "preservation_ratio"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ch in per_channel:
            writer.writerow([
                f"ch_{ch['channel']}",
                f"{ch['acc_real']:.6f}",
                f"{ch['acc_pred']:.6f}",
                f"{ch['preservation']:.6f}",
            ])
    print(f"    → saved {out_path}")
