#!/usr/bin/env python3
"""Convert Boran NPZ files to per-region NPY files for faster loading.

Reads each subject's NPZ, extracts source/target channels for all 3 region
pairs, z-scores per channel, and saves as individual NPY files. This avoids
repeated NPZ decompression and channel extraction at training time.

Output structure:
    {output_dir}/{subject_id}/
        hippocampus.npy          # (n_channels, n_samples) float32, z-scored
        entorhinal_cortex.npy
        amygdala.npy
        metadata.npz             # probes, soz_probes, n_sessions, etc.

Usage:
    python scripts/convert_boran_npz_to_npy.py
    python scripts/convert_boran_npz_to_npy.py --input-dir /data/boran_mtl_wm/processed_1khz --output-dir /data/boran_mtl_wm/processed_npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def convert_subject(npz_path: Path, output_dir: Path) -> dict:
    """Convert one subject's NPZ to per-region NPY files."""
    subject_id = npz_path.stem  # e.g., "S01"
    subj_dir = output_dir / subject_id
    subj_dir.mkdir(parents=True, exist_ok=True)

    # Load NPZ
    npz = np.load(npz_path, allow_pickle=False)
    ieeg = npz["ieeg"]  # (n_channels, total_samples) at 1kHz
    probes = list(npz["probes"])
    probe_n_contacts = npz["probe_n_contacts"]
    soz_probes_arr = npz["soz_probes"]
    soz_probes = list(soz_probes_arr) if len(soz_probes_arr) > 0 else []

    # Import region channel mapping from data.py
    from data import _boran_get_region_channels, BORAN_MTL_REGIONS

    # Extract and save each region
    region_info = {}
    for region in BORAN_MTL_REGIONS:
        chs, region_probes = _boran_get_region_channels(
            probes, probe_n_contacts, region, exclude_probes=None  # Don't exclude SOZ here, keep all
        )
        if len(chs) == 0:
            print(f"    {subject_id}: no channels for {region}, skipping region")
            continue

        # Extract and ensure contiguous memory
        signal = np.ascontiguousarray(ieeg[chs, :], dtype=np.float32)

        # Save raw (no z-score — let the loader handle normalization)
        npy_path = subj_dir / f"{region}.npy"
        np.save(npy_path, signal)

        region_info[region] = {
            "n_channels": len(chs),
            "n_samples": signal.shape[1],
            "probes": region_probes,
        }

    # Save metadata
    np.savez_compressed(
        subj_dir / "metadata.npz",
        probes=np.array(probes, dtype="U10"),
        probe_n_contacts=probe_n_contacts,
        soz_probes=np.array(soz_probes, dtype="U10") if soz_probes else np.array([], dtype="U10"),
        n_sessions=npz.get("n_sessions", np.int32(0)),
        n_trials=npz.get("n_trials", np.int32(0)),
        n_channels=npz.get("n_channels", np.int32(ieeg.shape[0])),
        sampling_rate=npz.get("sampling_rate", np.int32(1000)),
    )

    return {
        "subject_id": subject_id,
        "regions": region_info,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Boran NPZ files to per-region NPY for faster loading")
    parser.add_argument("--input-dir", type=str,
                        default="/data/boran_mtl_wm/processed_1khz",
                        help="Directory containing subject NPZ files")
    parser.add_argument("--output-dir", type=str,
                        default="/data/boran_mtl_wm/processed_npy",
                        help="Output directory for NPY files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("  BORAN NPZ -> NPY CONVERSION")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    if not input_dir.exists():
        print(f"\nERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*.npz"))
    print(f"\n  Found {len(npz_files)} NPZ files")

    results = []
    for npz_path in npz_files:
        print(f"\n  Converting {npz_path.stem}...")
        result = convert_subject(npz_path, output_dir)
        results.append(result)
        for region, info in result["regions"].items():
            print(f"    {region}: {info['n_channels']}ch x {info['n_samples']} samples")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  CONVERSION COMPLETE: {len(results)} subjects")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
