#!/usr/bin/env python3
"""Analyze the natural relationship between brain region signals.

This script computes baseline metrics WITHOUT any model - just comparing
raw source vs target signals to understand:
1. What's the natural correlation between brain regions?
2. How much does this vary across sessions/subjects?
3. What does the model need to beat?

Supports: olfactory, pfc, dandi, pcx1, ecog
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path("results/baseline")


# =============================================================================
# Core Statistics Functions
# =============================================================================

def per_channel_normalize(x: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel independently."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std[std < 1e-8] = 1e-8
    return (x - mean) / std


def compute_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mean Pearson correlation across channels. x, y: (C, T)."""
    x_norm = per_channel_normalize(x)
    y_norm = per_channel_normalize(y)
    corrs = []
    for c in range(x.shape[0]):
        corr = np.corrcoef(x_norm[c], y_norm[c])[0, 1]
        if not np.isnan(corr):
            corrs.append(corr)
    return np.mean(corrs) if corrs else 0.0


def compute_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mean R2 across channels. x, y: (C, T)."""
    x_norm = per_channel_normalize(x)
    y_norm = per_channel_normalize(y)
    r2s = []
    for c in range(x.shape[0]):
        ss_res = np.sum((y_norm[c] - x_norm[c]) ** 2)
        ss_tot = np.sum((y_norm[c] - y_norm[c].mean()) ** 2)
        if ss_tot > 1e-8:
            r2s.append(1 - ss_res / ss_tot)
    return np.mean(r2s) if r2s else 0.0


def compute_snr(signal: np.ndarray) -> float:
    """Compute mean per-channel SNR (diff-based). signal: (C, T)."""
    snrs = []
    for c in range(signal.shape[0]):
        s = signal[c].astype(np.float64)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        sig_var = np.var(s)
        noise_var = np.var(np.diff(s))
        if noise_var > 1e-10 and np.isfinite(sig_var) and np.isfinite(noise_var):
            snrs.append(sig_var / noise_var)
    return float(np.mean(snrs)) if snrs else 0.0


def analyze_continuous_pair(
    source: np.ndarray,
    target: np.ndarray,
    label: str,
    srate: float = 1000.0,
    window_size: int = 5000,
    stride: int = 2500,
) -> Optional[Dict]:
    """Analyze baseline relationship between a source/target signal pair.

    Args:
        source: Source signals (C, T)
        target: Target signals (C, T)
        label: Label for this recording (session/subject name)
        srate: Sampling rate in Hz
        window_size: Window size in samples
        stride: Stride between windows

    Returns:
        Dict of results, or None if too short.
    """
    # Clean data
    source = np.nan_to_num(source.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    n_ch = min(source.shape[0], target.shape[0])
    n_samples = min(source.shape[1], target.shape[1])
    source = source[:n_ch, :n_samples]
    target = target[:n_ch, :n_samples]

    n_windows = (n_samples - window_size) // stride + 1
    if n_windows < 1:
        return None

    window_corrs = []
    window_r2s = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window_corrs.append(compute_correlation(source[:, start:end], target[:, start:end]))
        window_r2s.append(compute_r2(source[:, start:end], target[:, start:end]))

    return {
        "label": label,
        "n_source_channels": int(source.shape[0]),
        "n_target_channels": int(target.shape[0]),
        "n_samples": int(n_samples),
        "duration_sec": float(n_samples / srate),
        "n_windows": int(n_windows),
        "source_std": float(source.std()),
        "target_std": float(target.std()),
        "source_snr": compute_snr(source),
        "target_snr": compute_snr(target),
        "corr_mean": float(np.mean(window_corrs)),
        "corr_std": float(np.std(window_corrs)),
        "corr_min": float(np.min(window_corrs)),
        "corr_max": float(np.max(window_corrs)),
        "r2_mean": float(np.mean(window_r2s)),
        "r2_std": float(np.std(window_r2s)),
    }


def print_summary(dataset_name: str, pair_name: str, results: List[Dict]):
    """Print a compact summary table and save JSON."""
    if not results:
        print("No valid results.")
        return

    all_corrs = [r["corr_mean"] for r in results]
    all_r2s = [r["r2_mean"] for r in results]

    print(f"\n{'Label':<16} {'Src Ch':<8} {'Tgt Ch':<8} {'Corr':<16} {'R2':<16} {'SNR(s/t)':<14} {'Win'}")
    print("-" * 95)
    for r in results:
        print(f"{r['label']:<16} {r['n_source_channels']:<8} {r['n_target_channels']:<8} "
              f"{r['corr_mean']:.3f} +/- {r['corr_std']:.3f}   "
              f"{r['r2_mean']:.3f} +/- {r['r2_std']:.3f}   "
              f"{r['source_snr']:.1f}/{r['target_snr']:.1f}       {r['n_windows']}")
    print("-" * 95)
    print(f"{'OVERALL':<16} {'':8} {'':8} "
          f"{np.mean(all_corrs):.3f} +/- {np.std(all_corrs):.3f}   "
          f"{np.mean(all_r2s):.3f} +/- {np.std(all_r2s):.3f}")

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{dataset_name}.json"
    save_data = {
        "dataset": dataset_name,
        "pair": pair_name,
        "n_sessions": len(results),
        "overall_corr": float(np.mean(all_corrs)),
        "overall_r2": float(np.mean(all_r2s)),
        "per_session": results,
    }
    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {output_file}")


# =============================================================================
# Dataset-Specific Runners
# =============================================================================

def run_olfactory_analysis(window_size: int, stride: int):
    """Run baseline analysis for the olfactory dataset (OB -> PCx)."""
    from data import load_signals, load_session_ids, extract_window, DATA_PATH, ODOR_CSV_PATH

    print("Olfactory dataset: OB -> PCx")
    signals = load_signals(DATA_PATH)  # [N, 2, C, T_full]
    windowed = extract_window(signals)  # [N, 2, C, T_window]
    ob_all = windowed[:, 0]   # [N, C, T]
    pcx_all = windowed[:, 1]  # [N, C, T]

    session_ids, session_to_idx, idx_to_session = load_session_ids(ODOR_CSV_PATH, len(signals))
    unique_sessions = sorted(set(session_ids.tolist()))

    print(f"  {len(unique_sessions)} sessions, {len(signals)} trials")

    results = []
    for sess_int in unique_sessions:
        sess_name = idx_to_session[sess_int]
        mask = session_ids == sess_int
        ob_sess = ob_all[mask]   # [N_sess, C, T]
        pcx_sess = pcx_all[mask] # [N_sess, C, T]

        # Concatenate trials along time for continuous analysis
        ob_cat = ob_sess.reshape(ob_sess.shape[1], -1)    # [C, N*T]
        pcx_cat = pcx_sess.reshape(pcx_sess.shape[1], -1)  # [C, N*T]

        r = analyze_continuous_pair(ob_cat, pcx_cat, sess_name,
                                    srate=1000.0, window_size=window_size, stride=stride)
        if r:
            results.append(r)

    print_summary("olfactory", "OB -> PCx", results)


def run_pfc_analysis(window_size: int, stride: int):
    """Run baseline analysis for the PFC dataset (PFC -> CA1)."""
    from data import load_pfc_signals, load_pfc_session_ids, PFC_DATA_PATH, PFC_META_PATH

    print("PFC dataset: PFC -> CA1")
    pfc_all, ca1_all = load_pfc_signals(PFC_DATA_PATH)  # [N, C, T] each

    session_ids, session_to_idx, idx_to_session = load_pfc_session_ids(PFC_META_PATH, len(pfc_all))
    unique_sessions = sorted(set(session_ids.tolist()))

    print(f"  {len(unique_sessions)} sessions, {len(pfc_all)} trials")

    results = []
    for sess_int in unique_sessions:
        sess_name = idx_to_session[sess_int]
        mask = session_ids == sess_int
        pfc_sess = pfc_all[mask]  # [N_sess, C, T]
        ca1_sess = ca1_all[mask]  # [N_sess, C, T]

        pfc_cat = pfc_sess.reshape(pfc_sess.shape[1], -1)  # [C, N*T]
        ca1_cat = ca1_sess.reshape(ca1_sess.shape[1], -1)  # [C, N*T]

        r = analyze_continuous_pair(pfc_cat, ca1_cat, sess_name,
                                    srate=1250.0, window_size=window_size, stride=stride)
        if r:
            results.append(r)

    print_summary("pfc", "PFC -> CA1", results)


def run_dandi_analysis(window_size: int, stride: int):
    """Run baseline analysis for the DANDI dataset (amygdala -> hippocampus)."""
    from data import DANDI_RAW_PATH, list_dandi_nwb_files, load_dandi_subject

    print("DANDI dataset: amygdala -> hippocampus")
    nwb_files = list_dandi_nwb_files(DANDI_RAW_PATH)

    subject_ids = []
    for f in nwb_files:
        stem = f.stem
        subj_id = stem.split("_")[0] if "sub-" in stem else stem
        if subj_id not in subject_ids:
            subject_ids.append(subj_id)
    subject_ids = sorted(subject_ids)
    print(f"  {len(subject_ids)} subjects")

    results = []
    for subj_id in subject_ids:
        try:
            data = load_dandi_subject(subj_id, DANDI_RAW_PATH,
                                      source_region="amygdala",
                                      target_region="hippocampus",
                                      zscore=False)
            if data["n_source_channels"] < 4 or data["n_target_channels"] < 4:
                continue
            r = analyze_continuous_pair(data["source"], data["target"], subj_id,
                                        srate=1000.0, window_size=window_size, stride=stride)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  Skipping {subj_id}: {e}")

    print_summary("dandi", "amygdala -> hippocampus", results)


def run_pcx1_analysis(window_size: int, stride: int):
    """Run baseline analysis for the PCx1 dataset (OB -> PCx)."""
    from data import list_pcx1_sessions, load_pcx1_session

    print("PCx1 dataset: OB -> PCx (continuous)")
    sessions = list_pcx1_sessions()
    print(f"  {len(sessions)} sessions")

    results = []
    for session in sessions:
        try:
            data = load_pcx1_session(session)
            ob = data["ob"]    # [C, T]
            pcx = data["pcx"]  # [C, T]
            r = analyze_continuous_pair(ob, pcx, session,
                                        srate=1000.0, window_size=window_size, stride=stride)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  Skipping {session}: {e}")

    print_summary("pcx1", "OB -> PCx", results)


def run_ecog_analysis(
    experiment: str = "motor_imagery",
    source_region: str = "frontal",
    target_region: str = "parietal",
    window_size: int = 5000,
    stride: int = 2500,
):
    """Run baseline analysis for the ECoG dataset."""
    from data import (
        _ECOG_DATA_DIR,
        _enumerate_ecog_recordings,
        _get_ecog_region_channels,
        _parse_ecog_srate,
        ECOG_BRAIN_LOBES,
    )

    print(f"ECoG dataset ({experiment}): {source_region} -> {target_region}")

    npz_path = _ECOG_DATA_DIR / f"{experiment}.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run: python scripts/download_ecog.py")
        return

    alldat = np.load(npz_path, allow_pickle=True)["dat"]
    recordings = _enumerate_ecog_recordings(alldat)
    print(f"  {len(recordings)} recordings")

    def _analyze_pair(src_reg, tgt_reg):
        pair_results = []
        for row, col, rec_id in recordings:
            dat = alldat[row, col]
            if dat is None or not isinstance(dat, dict):
                continue
            V = dat.get("V")
            if V is None or V.size == 0:
                continue
            src_chs, tgt_chs = _get_ecog_region_channels(dat, src_reg, tgt_reg)
            if len(src_chs) < 4 or len(tgt_chs) < 4:
                continue
            srate = _parse_ecog_srate(dat)
            source = V[:, src_chs].T.astype(np.float64)
            target = V[:, tgt_chs].T.astype(np.float64)
            label = f"s{row:02d}_b{col:02d}"
            r = analyze_continuous_pair(source, target, label,
                                        srate=srate, window_size=window_size, stride=stride)
            if r:
                r["n_source_channels"] = len(src_chs)
                r["n_target_channels"] = len(tgt_chs)
                pair_results.append(r)
        return pair_results

    # Primary pair
    primary_results = _analyze_pair(source_region, target_region)
    if not primary_results:
        print("No valid recordings found! Try different region pairs.")
        return

    # All pairs comparison
    all_pairs = {}
    for src in ECOG_BRAIN_LOBES:
        for tgt in ECOG_BRAIN_LOBES:
            if src == tgt:
                continue
            if (src, tgt) == (source_region, target_region):
                pr = primary_results
            else:
                pr = _analyze_pair(src, tgt)
            if pr:
                corrs = [r["corr_mean"] for r in pr]
                r2s = [r["r2_mean"] for r in pr]
                all_pairs[f"{src} -> {tgt}"] = {
                    "corr": float(np.mean(corrs)),
                    "r2": float(np.mean(r2s)),
                    "n": len(pr),
                }

    # Print primary pair summary
    print_summary(f"ecog_{experiment}_{source_region}_{target_region}",
                  f"{source_region} -> {target_region}", primary_results)

    # Print cross-pair comparison
    if all_pairs:
        primary_key = f"{source_region} -> {target_region}"
        print(f"\nAll region pair baselines for {experiment}:")
        print(f"{'Pair':<30} {'Corr':<8} {'R2':<8} {'N'}")
        print("-" * 50)
        for pair, s in sorted(all_pairs.items(), key=lambda x: x[1]["corr"], reverse=True):
            marker = " <--" if pair == primary_key else ""
            print(f"{pair:<30} {s['corr']:.3f}    {s['r2']:.3f}    {s['n']}{marker}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze natural baseline relationship between brain regions (no model)"
    )
    parser.add_argument("--dataset", type=str, default="pcx1",
                        choices=["olfactory", "pfc", "dandi", "pcx1", "ecog"],
                        help="Dataset to analyze")
    parser.add_argument("--ecog-experiment", type=str, default="motor_imagery",
                        help="ECoG experiment name")
    parser.add_argument("--ecog-source-region", type=str, default="frontal",
                        help="ECoG source brain lobe")
    parser.add_argument("--ecog-target-region", type=str, default="parietal",
                        help="ECoG target brain lobe")
    parser.add_argument("--window-size", type=int, default=5000,
                        help="Window size in samples (default: 5000)")
    parser.add_argument("--stride", type=int, default=2500,
                        help="Stride between windows (default: 2500)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"BASELINE ANALYSIS: {args.dataset}")
    print("=" * 70)
    print("Natural inter-region correlation WITHOUT any model.\n")

    if args.dataset == "olfactory":
        run_olfactory_analysis(args.window_size, args.stride)
    elif args.dataset == "pfc":
        run_pfc_analysis(args.window_size, args.stride)
    elif args.dataset == "dandi":
        run_dandi_analysis(args.window_size, args.stride)
    elif args.dataset == "pcx1":
        run_pcx1_analysis(args.window_size, args.stride)
    elif args.dataset == "ecog":
        run_ecog_analysis(
            experiment=args.ecog_experiment,
            source_region=args.ecog_source_region,
            target_region=args.ecog_target_region,
            window_size=args.window_size,
            stride=args.stride,
        )


if __name__ == "__main__":
    main()
