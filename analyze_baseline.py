#!/usr/bin/env python3
"""Analyze the natural relationship between brain region signals.

This script computes baseline metrics WITHOUT any model - just comparing
raw source vs target signals to understand:
1. What's the natural correlation between brain regions?
2. How much does this vary across sessions/subjects?
3. What does the model need to beat?

Supports: pcx1 (OB->PCx), ecog (any inter-region pair)
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data import (
    list_pcx1_sessions,
    load_pcx1_session,
    PCX1_CONTINUOUS_PATH,
)


def per_channel_normalize(x: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel independently."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std[std < 1e-8] = 1e-8
    return (x - mean) / std


def compute_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mean Pearson correlation across channels."""
    # x, y: (channels, time)
    x_norm = per_channel_normalize(x)
    y_norm = per_channel_normalize(y)

    # Correlation per channel
    corrs = []
    for c in range(x.shape[0]):
        corr = np.corrcoef(x_norm[c], y_norm[c])[0, 1]
        if not np.isnan(corr):
            corrs.append(corr)

    return np.mean(corrs) if corrs else 0.0


def compute_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mean R² (explained variance) across channels."""
    x_norm = per_channel_normalize(x)
    y_norm = per_channel_normalize(y)

    r2s = []
    for c in range(x.shape[0]):
        ss_res = np.sum((y_norm[c] - x_norm[c]) ** 2)
        ss_tot = np.sum((y_norm[c] - y_norm[c].mean()) ** 2)
        if ss_tot > 1e-8:
            r2 = 1 - ss_res / ss_tot
            r2s.append(r2)

    return np.mean(r2s) if r2s else 0.0


def analyze_session(session_name: str, path: Path = PCX1_CONTINUOUS_PATH) -> Dict:
    """Analyze baseline OB-PCx relationship for one session."""
    print(f"\nAnalyzing session: {session_name}")

    # Load session data
    data = load_pcx1_session(session_name, path)
    ob = data['ob']   # (channels, time)
    pcx = data['pcx'] # (channels, time)

    print(f"  OB shape: {ob.shape}, PCx shape: {pcx.shape}")
    print(f"  OB range: [{ob.min():.2f}, {ob.max():.2f}], mean={ob.mean():.2f}, std={ob.std():.2f}")
    print(f"  PCx range: [{pcx.min():.2f}, {pcx.max():.2f}], mean={pcx.mean():.2f}, std={pcx.std():.2f}")

    # Ensure same length
    min_len = min(ob.shape[1], pcx.shape[1])
    ob = ob[:, :min_len]
    pcx = pcx[:, :min_len]

    # Compute windowed statistics (like training)
    window_size = 5000  # 5 seconds at 1kHz
    stride = 2500       # 50% overlap

    n_windows = (min_len - window_size) // stride + 1
    print(f"  Windows: {n_windows} (size={window_size}, stride={stride})")

    window_corrs = []
    window_r2s = []

    for i in range(n_windows):
        start = i * stride
        end = start + window_size

        ob_win = ob[:, start:end]
        pcx_win = pcx[:, start:end]

        corr = compute_correlation(ob_win, pcx_win)
        r2 = compute_r2(ob_win, pcx_win)

        window_corrs.append(corr)
        window_r2s.append(r2)

    results = {
        'session': session_name,
        'n_windows': n_windows,
        'duration_sec': min_len / 1000,
        'ob_std': ob.std(),
        'pcx_std': pcx.std(),
        'corr_mean': np.mean(window_corrs),
        'corr_std': np.std(window_corrs),
        'corr_min': np.min(window_corrs),
        'corr_max': np.max(window_corrs),
        'r2_mean': np.mean(window_r2s),
        'r2_std': np.std(window_r2s),
    }

    print(f"  Baseline correlation: r={results['corr_mean']:.3f} ± {results['corr_std']:.3f}")
    print(f"  Baseline R²: {results['r2_mean']:.3f} ± {results['r2_std']:.3f}")
    print(f"  Correlation range: [{results['corr_min']:.3f}, {results['corr_max']:.3f}]")

    return results


def analyze_ecog_recording(
    subject_idx: int,
    block_idx: int,
    alldat: np.ndarray,
    experiment: str,
    source_region: str,
    target_region: str,
    window_size: int = 5000,
    stride: int = 2500,
    min_channels: int = 4,
    verbose: bool = True,
) -> Optional[Dict]:
    """Analyze baseline inter-region relationship for one ECoG recording.

    Args:
        subject_idx: Row index in alldat
        block_idx: Column index in alldat
        alldat: Pre-loaded NPZ data array
        experiment: Experiment name
        source_region: Source brain lobe
        target_region: Target brain lobe
        window_size: Window size in samples
        stride: Stride between windows
        min_channels: Minimum channels per region
        verbose: Whether to print per-recording details

    Returns:
        Dict of results, or None if recording has insufficient channels.
    """
    from data import (
        _parse_ecog_lobe,
        _parse_ecog_srate,
        _get_ecog_region_channels,
    )

    dat = alldat[subject_idx, block_idx]
    if dat is None or not isinstance(dat, dict):
        return None

    V = dat.get("V")
    if V is None or V.size == 0:
        return None

    source_chs, target_chs = _get_ecog_region_channels(dat, source_region, target_region)
    if len(source_chs) < min_channels or len(target_chs) < min_channels:
        return None

    # Use min channels for consistent comparison
    n_ch = min(len(source_chs), len(target_chs))
    source = V[:, source_chs[:n_ch]].T.astype(np.float64)  # (C, T) - float64 to avoid overflow
    target = V[:, target_chs[:n_ch]].T.astype(np.float64)  # (C, T)

    # Clean inf/nan values
    source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    srate = _parse_ecog_srate(dat)
    subject_id = f"s{subject_idx:02d}_b{block_idx:02d}"

    if verbose:
        print(f"\nAnalyzing ECoG {experiment} {subject_id}: "
              f"{source_region}({len(source_chs)}ch) -> {target_region}({len(target_chs)}ch)")
        print(f"  Using {n_ch} matched channels, {V.shape[0]} samples ({V.shape[0]/srate:.1f}s), srate={srate}Hz")
        print(f"  Source range: [{source.min():.2f}, {source.max():.2f}], std={source.std():.2f}")
        print(f"  Target range: [{target.min():.2f}, {target.max():.2f}], std={target.std():.2f}")

    n_samples = source.shape[1]
    n_windows = (n_samples - window_size) // stride + 1

    if n_windows < 1:
        if verbose:
            print(f"  Too short for windowed analysis (need >= {window_size} samples)")
        return None

    if verbose:
        print(f"  Windows: {n_windows} (size={window_size}, stride={stride})")

    window_corrs = []
    window_r2s = []

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        corr = compute_correlation(source[:, start:end], target[:, start:end])
        r2 = compute_r2(source[:, start:end], target[:, start:end])
        window_corrs.append(corr)
        window_r2s.append(r2)

    # Spectral analysis
    freqs = np.fft.rfftfreq(window_size, 1 / srate)
    bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 100)}

    # Average PSD across channels for one representative window (middle)
    mid_win = n_windows // 2
    mid_start = mid_win * stride
    src_win = source[:, mid_start:mid_start + window_size]
    tgt_win = target[:, mid_start:mid_start + window_size]

    src_psd = np.mean(np.abs(np.fft.rfft(src_win, axis=1)) ** 2, axis=0)
    tgt_psd = np.mean(np.abs(np.fft.rfft(tgt_win, axis=1)) ** 2, axis=0)
    total_src = np.sum(src_psd) + 1e-10
    total_tgt = np.sum(tgt_psd) + 1e-10

    band_powers = {}
    for band_name, (f_low, f_high) in bands.items():
        mask = (freqs >= f_low) & (freqs < f_high)
        band_powers[f"source_{band_name}_rel"] = float(np.sum(src_psd[mask]) / total_src)
        band_powers[f"target_{band_name}_rel"] = float(np.sum(tgt_psd[mask]) / total_tgt)

    # SNR estimation (use per-channel to avoid cross-channel variance inflation)
    src_snrs = []
    tgt_snrs = []
    for c in range(n_ch):
        s_var = np.var(source[c])
        s_noise = np.var(np.diff(source[c]))
        if s_noise > 1e-10 and np.isfinite(s_var) and np.isfinite(s_noise):
            src_snrs.append(s_var / s_noise)
        t_var = np.var(target[c])
        t_noise = np.var(np.diff(target[c]))
        if t_noise > 1e-10 and np.isfinite(t_var) and np.isfinite(t_noise):
            tgt_snrs.append(t_var / t_noise)
    src_snr = float(np.mean(src_snrs)) if src_snrs else 0.0
    tgt_snr = float(np.mean(tgt_snrs)) if tgt_snrs else 0.0

    results = {
        "subject_id": subject_id,
        "experiment": experiment,
        "source_region": source_region,
        "target_region": target_region,
        "n_source_channels": len(source_chs),
        "n_target_channels": len(target_chs),
        "n_matched_channels": n_ch,
        "n_samples": n_samples,
        "duration_sec": n_samples / srate,
        "srate": srate,
        "n_windows": n_windows,
        "source_std": float(source.std()),
        "target_std": float(target.std()),
        "source_snr": src_snr,
        "target_snr": tgt_snr,
        "corr_mean": float(np.mean(window_corrs)),
        "corr_std": float(np.std(window_corrs)),
        "corr_min": float(np.min(window_corrs)),
        "corr_max": float(np.max(window_corrs)),
        "r2_mean": float(np.mean(window_r2s)),
        "r2_std": float(np.std(window_r2s)),
        **band_powers,
    }

    if verbose:
        print(f"  Baseline correlation: r={results['corr_mean']:.3f} +/- {results['corr_std']:.3f}")
        print(f"  Baseline R2: {results['r2_mean']:.3f} +/- {results['r2_std']:.3f}")
        print(f"  SNR: source={src_snr:.2f}, target={tgt_snr:.2f}")

    return results


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
        ECOG_EXPERIMENTS,
    )

    print("=" * 70)
    print(f"BASELINE ANALYSIS: ECoG {experiment} ({source_region} -> {target_region})")
    print("=" * 70)
    print("\nThis shows what correlation exists between brain regions")
    print("WITHOUT any model - just raw signal comparison.")
    print("Your model needs to BEAT these numbers to be useful.\n")

    npz_path = _ECOG_DATA_DIR / f"{experiment}.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run: python scripts/download_ecog.py")
        return

    alldat = np.load(npz_path, allow_pickle=True)["dat"]
    recordings = _enumerate_ecog_recordings(alldat)
    print(f"Found {len(recordings)} recordings in {experiment}")

    all_results = []
    for row, col, rec_id in recordings:
        try:
            results = analyze_ecog_recording(
                row, col, alldat, experiment,
                source_region, target_region,
                window_size, stride,
            )
            if results is not None:
                all_results.append(results)
        except Exception as e:
            print(f"  ERROR on {rec_id}: {e}")

    if not all_results:
        print("\nNo valid recordings found! Try different region pairs.")
        return

    # Summary table
    print("\n" + "=" * 70)
    print(f"SUMMARY: Baseline {source_region} -> {target_region} (no model)")
    print("=" * 70)
    print(f"\n{'Subject':<12} {'Src Ch':<8} {'Tgt Ch':<8} {'Corr (r)':<16} {'R2':<16} {'Windows':<8} {'SNR(s/t)'}")
    print("-" * 90)

    for r in all_results:
        print(f"{r['subject_id']:<12} {r['n_source_channels']:<8} {r['n_target_channels']:<8} "
              f"{r['corr_mean']:.3f} +/- {r['corr_std']:.3f}   "
              f"{r['r2_mean']:.3f} +/- {r['r2_std']:.3f}   "
              f"{r['n_windows']:<8} {r['source_snr']:.1f}/{r['target_snr']:.1f}")

    all_corrs = [r["corr_mean"] for r in all_results]
    all_r2s = [r["r2_mean"] for r in all_results]

    print("-" * 90)
    print(f"{'OVERALL':<12} {'':8} {'':8} "
          f"{np.mean(all_corrs):.3f} +/- {np.std(all_corrs):.3f}   "
          f"{np.mean(all_r2s):.3f} +/- {np.std(all_r2s):.3f}")

    # Spectral summary
    print(f"\n{'Band':<10} {'Source Power':<14} {'Target Power':<14}")
    print("-" * 38)
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        src_vals = [r[f"source_{band}_rel"] for r in all_results]
        tgt_vals = [r[f"target_{band}_rel"] for r in all_results]
        print(f"{band:<10} {np.mean(src_vals):.4f}         {np.mean(tgt_vals):.4f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"""
Baseline correlation (raw {source_region} vs {target_region}): r = {np.mean(all_corrs):.3f}
Baseline R2: {np.mean(all_r2s):.3f}

If model r > baseline r: Model is learning useful transformations
If model r ~ baseline r: Model might just be passing through signal
If model r < baseline r: Something is wrong
""")

    # Also test other region pairs for comparison
    from data import ECOG_BRAIN_LOBES
    other_pairs = []
    for src in ECOG_BRAIN_LOBES:
        for tgt in ECOG_BRAIN_LOBES:
            if src != tgt and (src, tgt) != (source_region, target_region):
                other_pairs.append((src, tgt))

    print("=" * 70)
    print("QUICK COMPARISON: Other region pairs")
    print("=" * 70)
    print(f"\n{'Pair':<30} {'Corr (r)':<12} {'R2':<12} {'N_valid'}")
    print("-" * 65)
    print(f"{source_region} -> {target_region:<20} {np.mean(all_corrs):.3f}        {np.mean(all_r2s):.3f}        {len(all_results)}")

    for src, tgt in other_pairs:
        pair_results = []
        for row, col, rec_id in recordings:
            r = analyze_ecog_recording(
                row, col, alldat, experiment, src, tgt,
                window_size, stride, verbose=False,
            )
            if r is not None:
                pair_results.append(r)
        if pair_results:
            pair_corrs = [r["corr_mean"] for r in pair_results]
            pair_r2s = [r["r2_mean"] for r in pair_results]
            print(f"{src} -> {tgt:<20} {np.mean(pair_corrs):.3f}        {np.mean(pair_r2s):.3f}        {len(pair_results)}")

    # Save results
    import json
    output_dir = Path("results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"ecog_{experiment}_{source_region}_{target_region}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def run_pcx1_analysis():
    """Run baseline analysis for the PCx1 dataset (original behavior)."""
    print("=" * 70)
    print("BASELINE ANALYSIS: Natural OB-PCx Relationship")
    print("=" * 70)
    print("\nThis shows what correlation exists between OB and PCx")
    print("WITHOUT any model - just raw signal comparison.")
    print("Your model needs to BEAT these numbers to be useful.\n")

    # Get all sessions
    sessions = list_pcx1_sessions()
    print(f"Found {len(sessions)} sessions: {sessions}")

    # Analyze each session
    all_results = []
    for session in sessions:
        try:
            results = analyze_session(session)
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Baseline OB-PCx Correlation (no model)")
    print("=" * 70)
    print(f"\n{'Session':<15} {'Corr (r)':<15} {'R2':<15} {'Windows':<10}")
    print("-" * 55)

    for r in all_results:
        print(f"{r['session']:<15} {r['corr_mean']:.3f} +/- {r['corr_std']:.3f}    "
              f"{r['r2_mean']:.3f} +/- {r['r2_std']:.3f}    {r['n_windows']}")

    # Overall
    all_corrs = [r['corr_mean'] for r in all_results]
    all_r2s = [r['r2_mean'] for r in all_results]

    print("-" * 55)
    print(f"{'OVERALL':<15} {np.mean(all_corrs):.3f} +/- {np.std(all_corrs):.3f}    "
          f"{np.mean(all_r2s):.3f} +/- {np.std(all_r2s):.3f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"""
Baseline correlation (raw OB vs PCx): r = {np.mean(all_corrs):.3f}
Baseline R2: {np.mean(all_r2s):.3f}

If model r > baseline r: Model is learning useful transformations
If model r ~ baseline r: Model might just be passing through signal
If model r < baseline r: Something is wrong
""")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze natural baseline relationship between brain regions (no model)"
    )
    parser.add_argument("--dataset", type=str, default="pcx1",
                        choices=["pcx1", "ecog"],
                        help="Dataset to analyze")
    parser.add_argument("--ecog-experiment", type=str, default="motor_imagery",
                        help="ECoG experiment name")
    parser.add_argument("--ecog-source-region", type=str, default="frontal",
                        help="ECoG source brain lobe")
    parser.add_argument("--ecog-target-region", type=str, default="parietal",
                        help="ECoG target brain lobe")
    parser.add_argument("--window-size", type=int, default=5000,
                        help="Window size in samples (default: 5000 = 5s at 1kHz)")
    parser.add_argument("--stride", type=int, default=2500,
                        help="Stride between windows (default: 2500)")
    args = parser.parse_args()

    if args.dataset == "ecog":
        run_ecog_analysis(
            experiment=args.ecog_experiment,
            source_region=args.ecog_source_region,
            target_region=args.ecog_target_region,
            window_size=args.window_size,
            stride=args.stride,
        )
    else:
        run_pcx1_analysis()


if __name__ == "__main__":
    main()
