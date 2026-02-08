#!/usr/bin/env python3
"""Preprocess Boran MTL dataset: downsample from 2kHz to 1kHz and save as NPZ.

Reads NIX/HDF5 files, extracts iEEG trial data per subject, downsamples
using scipy.signal.decimate (anti-alias filtered), and saves one NPZ per
subject with all metadata needed for training.

Usage:
    python scripts/preprocess_boran.py
    python scripts/preprocess_boran.py --input-dir /data/boran_mtl_wm/data_nix --output-dir /data/boran_mtl_wm/processed_1khz
"""

import argparse
import gc
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    print("ERROR: h5py required. pip install h5py")
    sys.exit(1)

try:
    from scipy.signal import decimate
except ImportError:
    print("ERROR: scipy required. pip install scipy")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Boran NIX/HDF5 helpers (copied from data.py to keep script self-contained)
# =============================================================================

def decode_bytes(val):
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    return str(val)


def extract_first_string(val):
    """Extract first string from NIX metadata compound field (numpy.void)."""
    if isinstance(val, np.void):
        try:
            first = val[0]
            if isinstance(first, (bytes, np.bytes_)):
                return first.decode("utf-8", errors="replace")
            return str(first)
        except (IndexError, TypeError):
            pass
    if isinstance(val, (tuple, list)) and len(val) > 0:
        first = val[0]
        if isinstance(first, (bytes, np.bytes_)):
            return first.decode("utf-8", errors="replace")
        return str(first)
    if isinstance(val, np.ndarray):
        if val.dtype.names:
            try:
                first = val.flat[0][0]
                if isinstance(first, (bytes, np.bytes_)):
                    return first.decode("utf-8", errors="replace")
                return str(first)
            except (IndexError, TypeError):
                pass
        elif val.size >= 1:
            first = val.flat[0]
            if isinstance(first, (bytes, np.bytes_)):
                return first.decode("utf-8", errors="replace")
            return str(first)
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, str):
        return val
    return str(val)


def get_depth_probes(f):
    """Extract depth electrode probe list from metadata."""
    probes = []
    if "metadata" not in f:
        return probes

    def _scan(group, depth=0):
        if depth > 6:
            return
        try:
            for key in group.keys():
                item = group[key]
                k_lower = key.lower()
                if "depth electrode" in k_lower:
                    if hasattr(item, 'shape'):
                        try:
                            val = item[()]
                            raw = extract_first_string(val).strip()
                            if raw:
                                probes.extend(
                                    p.strip().upper()
                                    for p in raw.split(",")
                                    if p.strip()
                                )
                                return
                        except Exception:
                            pass
                try:
                    if hasattr(item, 'keys'):
                        _scan(item, depth + 1)
                except Exception:
                    pass
        except Exception:
            pass

    _scan(f["metadata"])
    return probes


def get_soz_probes(f):
    """Extract Seizure Onset Zone probe names from metadata."""
    soz_probes = []
    if "metadata" not in f:
        return soz_probes

    def _scan(group, depth=0):
        if depth > 6:
            return
        try:
            for key in group.keys():
                item = group[key]
                k_lower = key.lower()
                if "seizure onset" in k_lower or "soz" in k_lower:
                    if hasattr(item, 'shape'):
                        try:
                            val = item[()]
                            raw = extract_first_string(val).strip()
                            if raw:
                                soz_probes.extend(
                                    p.strip().upper()
                                    for p in raw.split(",")
                                    if p.strip()
                                )
                                return
                        except Exception:
                            pass
                try:
                    if hasattr(item, 'keys'):
                        _scan(item, depth + 1)
                except Exception:
                    pass
        except Exception:
            pass

    _scan(f["metadata"])
    return soz_probes


def load_ieeg_trials(f):
    """Load all iEEG trial arrays from a NIX file.

    Returns list of arrays, each shape (n_channels, n_timepoints).
    """
    trials = []
    if "data" not in f:
        return trials

    data_grp = f["data"]
    for block_key in data_grp.keys():
        block = data_grp[block_key]
        if not hasattr(block, 'keys'):
            continue

        da_group = None
        if "data_arrays" in block:
            da_group = block["data_arrays"]
        elif block_key == "data_arrays":
            da_group = block

        if da_group is None:
            continue

        for da_key in da_group.keys():
            da = da_group[da_key]
            da_name = ""
            try:
                if hasattr(da, 'attrs') and "name" in da.attrs:
                    da_name = decode_bytes(da.attrs["name"])
                elif hasattr(da, 'keys') and "name" in da:
                    d = da["name"]
                    if hasattr(d, 'shape'):
                        da_name = decode_bytes(d[()])
            except Exception:
                da_name = da_key

            name_lower = da_name.lower()
            if "ieeg" in name_lower and "trial" in name_lower:
                try:
                    if hasattr(da, 'keys') and "data" in da:
                        arr = da["data"][()]
                    elif hasattr(da, 'shape'):
                        arr = da[()]
                    else:
                        continue

                    if arr.ndim == 2:
                        trials.append(arr)
                except Exception:
                    continue

    return trials


# =============================================================================
# Main preprocessing
# =============================================================================

def preprocess_subject(subject_id, h5_files, output_dir, downsample_factor=2):
    """Preprocess all sessions for a subject: load, downsample, save NPZ."""
    print(f"\n  {subject_id}: {len(h5_files)} session files")

    # Get metadata from first file
    with h5py.File(h5_files[0], "r") as f:
        probes = get_depth_probes(f)
        soz_probes = get_soz_probes(f)

    if not probes:
        print(f"    WARNING: No depth probes found, skipping")
        return None

    # Load all trials from all sessions
    all_trials = []
    n_sessions = 0
    n_channels = None

    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                trials = load_ieeg_trials(f)
                if trials:
                    n_sessions += 1
                    for trial in trials:
                        if n_channels is None:
                            n_channels = trial.shape[0]
                        elif trial.shape[0] != n_channels:
                            print(f"    WARNING: Channel mismatch in {h5_path.name}: "
                                  f"expected {n_channels}, got {trial.shape[0]}, skipping trial")
                            continue
                        all_trials.append(trial)
        except Exception as e:
            print(f"    ERROR reading {h5_path.name}: {e}")
            continue

    if not all_trials:
        print(f"    WARNING: No valid trials loaded, skipping")
        return None

    n_trials = len(all_trials)
    original_timepoints = all_trials[0].shape[1]
    print(f"    {n_sessions} sessions, {n_trials} trials, "
          f"{n_channels} channels, {original_timepoints} timepoints/trial @ 2kHz")

    # Downsample each trial: 2kHz -> 1kHz using scipy.signal.decimate
    # decimate applies anti-aliasing filter before downsampling
    downsampled_trials = []
    for i, trial in enumerate(all_trials):
        # trial shape: (n_channels, n_timepoints)
        # decimate along axis=1 (time)
        ds_trial = decimate(trial.astype(np.float64), downsample_factor, axis=1)
        downsampled_trials.append(ds_trial.astype(np.float32))

        if (i + 1) % 50 == 0:
            print(f"    Downsampled {i + 1}/{n_trials} trials...", flush=True)

    # Free original data
    del all_trials
    gc.collect()

    new_timepoints = downsampled_trials[0].shape[1]
    print(f"    Downsampled: {original_timepoints} -> {new_timepoints} timepoints/trial @ 1kHz")

    # Concatenate all trials: (n_channels, total_timepoints)
    concatenated = np.concatenate(downsampled_trials, axis=1)
    print(f"    Concatenated shape: {concatenated.shape} "
          f"({concatenated.shape[1] / 1000:.1f}s total)")

    del downsampled_trials
    gc.collect()

    # Compute per-probe channel assignment
    n_probes = len(probes)
    base_contacts = n_channels // n_probes
    remainder = n_channels % n_probes
    probe_n_contacts = np.array([
        base_contacts + (1 if i < remainder else 0)
        for i in range(n_probes)
    ], dtype=np.int32)

    # Save NPZ
    output_path = output_dir / f"{subject_id}.npz"
    np.savez_compressed(
        output_path,
        ieeg=concatenated,
        probes=np.array(probes, dtype='U10'),
        probe_n_contacts=probe_n_contacts,
        soz_probes=np.array(soz_probes, dtype='U10') if soz_probes else np.array([], dtype='U10'),
        n_sessions=np.int32(n_sessions),
        n_trials=np.int32(n_trials),
        n_channels=np.int32(n_channels),
        sampling_rate=np.int32(1000),
        original_sampling_rate=np.int32(2000),
        downsample_factor=np.int32(downsample_factor),
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {output_path} ({file_size_mb:.1f} MB)")

    return {
        "subject_id": subject_id,
        "n_sessions": n_sessions,
        "n_trials": n_trials,
        "n_channels": n_channels,
        "probes": probes,
        "soz_probes": soz_probes,
        "total_samples": concatenated.shape[1],
        "file_size_mb": file_size_mb,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Boran MTL dataset: downsample 2kHz -> 1kHz, save as NPZ")
    parser.add_argument("--input-dir", type=str, default="/data/boran_mtl_wm/data_nix",
                        help="Directory containing H5 files")
    parser.add_argument("--output-dir", type=str, default="/data/boran_mtl_wm/processed_1khz",
                        help="Output directory for NPZ files")
    parser.add_argument("--downsample-factor", type=int, default=2,
                        help="Downsample factor (default: 2 for 2kHz->1kHz)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("  BORAN MTL PREPROCESSING: 2kHz -> 1kHz")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Downsample factor: {args.downsample_factor}")
    print("=" * 70)

    if not input_dir.exists():
        print(f"\nERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find and group H5 files by subject
    h5_files = sorted(input_dir.glob("*.h5"))
    print(f"\n  Found {len(h5_files)} H5 files")

    subjects = defaultdict(list)
    for f in h5_files:
        m = re.search(r"Subject[_\s]*(\d+)", f.stem, re.IGNORECASE)
        if m:
            subj_id = f"S{int(m.group(1)):02d}"
            subjects[subj_id].append(f)

    print(f"  Found {len(subjects)} subjects: {sorted(subjects.keys())}")

    # Process each subject
    results = []
    for subj_id in sorted(subjects.keys()):
        h5_list = sorted(subjects[subj_id])
        result = preprocess_subject(subj_id, h5_list, output_dir, args.downsample_factor)
        if result:
            results.append(result)
        gc.collect()

    # Summary
    print(f"\n\n{'=' * 70}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Subjects processed: {len(results)}/{len(subjects)}")

    total_trials = sum(r["n_trials"] for r in results)
    total_samples = sum(r["total_samples"] for r in results)
    total_size = sum(r["file_size_mb"] for r in results)

    print(f"  Total trials: {total_trials}")
    print(f"  Total samples: {total_samples} ({total_samples / 1000:.1f}s at 1kHz)")
    print(f"  Total disk: {total_size:.1f} MB")

    print(f"\n  Per-subject summary:")
    print(f"  {'Subject':>8s}  {'Sessions':>8s}  {'Trials':>7s}  {'Channels':>8s}  {'Samples':>10s}  {'Duration':>10s}  {'Probes'}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*30}")
    for r in results:
        dur = r["total_samples"] / 1000
        probes_str = ",".join(r["probes"])
        soz_str = f" (SOZ: {','.join(r['soz_probes'])})" if r["soz_probes"] else ""
        print(f"  {r['subject_id']:>8s}  {r['n_sessions']:>8d}  {r['n_trials']:>7d}  "
              f"{r['n_channels']:>8d}  {r['total_samples']:>10d}  {dur:>8.1f}s  "
              f"{probes_str}{soz_str}")

    print(f"\n  Output files in: {output_dir}")
    print()


if __name__ == "__main__":
    main()
