#!/usr/bin/env python3
"""Analyze the natural relationship between OB and PCx signals.

This script computes baseline metrics WITHOUT any model - just comparing
raw OB vs PCx signals to understand:
1. What's the natural correlation between brain regions?
2. How much does this vary across sessions?
3. What does the model need to beat?
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
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


def main():
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
    print(f"\n{'Session':<15} {'Corr (r)':<15} {'R²':<15} {'Windows':<10}")
    print("-" * 55)

    for r in all_results:
        print(f"{r['session']:<15} {r['corr_mean']:.3f} ± {r['corr_std']:.3f}    "
              f"{r['r2_mean']:.3f} ± {r['r2_std']:.3f}    {r['n_windows']}")

    # Overall
    all_corrs = [r['corr_mean'] for r in all_results]
    all_r2s = [r['r2_mean'] for r in all_results]

    print("-" * 55)
    print(f"{'OVERALL':<15} {np.mean(all_corrs):.3f} ± {np.std(all_corrs):.3f}    "
          f"{np.mean(all_r2s):.3f} ± {np.std(all_r2s):.3f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"""
Baseline correlation (raw OB vs PCx): r = {np.mean(all_corrs):.3f}
Baseline R²: {np.mean(all_r2s):.3f}

Your model's validation metrics:
  Fwd: r=0.661, r²=0.439  (from Epoch 7)

Model improvement over baseline:
  Correlation: {0.661:.3f} vs {np.mean(all_corrs):.3f} baseline = {0.661 - np.mean(all_corrs):+.3f}
  R²: {0.439:.3f} vs {np.mean(all_r2s):.3f} baseline = {0.439 - np.mean(all_r2s):+.3f}

If model r > baseline r: Model is learning useful transformations
If model r ≈ baseline r: Model might just be passing through signal
If model r < baseline r: Something is wrong
""")


if __name__ == "__main__":
    main()
