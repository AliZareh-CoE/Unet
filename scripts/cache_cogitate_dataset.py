#!/usr/bin/env python3
"""Pre-cache COGITATE dataset as memory-mapped numpy arrays for fast loading.

This script pre-computes all sliding windows and saves them as memory-mapped
files, eliminating the data loading bottleneck during training.

Usage:
    python scripts/cache_cogitate_dataset.py \
        --source temporal --target frontal \
        --source-channels 23 --target-channels 17 \
        --window-size 5120 --stride 2560 \
        --output /data/COGITATE_cached

The output can then be used with:
    python LOSO/runner.py --dataset cogitate_temp_front --cached-data /data/COGITATE_cached
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import (
    list_cogitate_subjects,
    load_cogitate_subject,
    _COGITATE_DATA_DIR,
)


def cache_cogitate_dataset(
    source_regions: list,
    target_regions: list,
    n_source_channels: int,
    n_target_channels: int,
    window_size: int,
    stride: int,
    output_dir: Path,
    data_dir: Path = _COGITATE_DATA_DIR,
    zscore: bool = True,
):
    """Pre-cache COGITATE dataset as memory-mapped arrays.

    Creates:
        - source_windows.npy: Memory-mapped [N, source_channels, window_size]
        - target_windows.npy: Memory-mapped [N, target_channels, window_size]
        - metadata.json: Subject info, window mapping, config
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all subjects and filter by channel requirements
    all_subjects = list_cogitate_subjects(data_dir)
    print(f"Found {len(all_subjects)} total subjects")

    viable_subjects = []
    subjects_data = []

    for subj_id in all_subjects:
        data = load_cogitate_subject(
            subject_id=subj_id,
            data_dir=data_dir,
            source_regions=source_regions,
            target_regions=target_regions,
            n_source_channels=n_source_channels,
            n_target_channels=n_target_channels,
            zscore=zscore,
        )
        if data is not None:
            viable_subjects.append(subj_id)
            subjects_data.append(data)
            print(f"  ✓ {subj_id}: {data['n_samples']} samples")
        else:
            print(f"  ✗ {subj_id}: insufficient channels")

    print(f"\nViable subjects: {len(viable_subjects)}/{len(all_subjects)}")

    if not viable_subjects:
        print("ERROR: No viable subjects found!")
        return

    # Calculate total windows per subject and overall
    subject_windows = []
    total_windows = 0

    for subj_idx, data in enumerate(subjects_data):
        n_samples = data["n_samples"]
        n_windows = max(0, (n_samples - window_size) // stride + 1)
        subject_windows.append({
            "subject_id": viable_subjects[subj_idx],
            "subject_idx": subj_idx,
            "n_samples": n_samples,
            "n_windows": n_windows,
            "window_start_idx": total_windows,
            "window_end_idx": total_windows + n_windows,
        })
        total_windows += n_windows

    print(f"Total windows: {total_windows}")
    print(f"Estimated size: {total_windows * (n_source_channels + n_target_channels) * window_size * 4 / 1e9:.2f} GB")

    # Create memory-mapped arrays
    source_path = output_dir / "source_windows.npy"
    target_path = output_dir / "target_windows.npy"

    # Create files with proper headers
    source_shape = (total_windows, n_source_channels, window_size)
    target_shape = (total_windows, n_target_channels, window_size)

    print(f"\nCreating memory-mapped arrays...")
    print(f"  Source: {source_shape}")
    print(f"  Target: {target_shape}")

    # Use memmap in write mode
    source_mmap = np.lib.format.open_memmap(
        source_path, mode='w+', dtype=np.float32, shape=source_shape
    )
    target_mmap = np.lib.format.open_memmap(
        target_path, mode='w+', dtype=np.float32, shape=target_shape
    )

    # Fill the arrays
    print("\nCaching windows...")
    window_idx = 0

    for subj_idx, data in enumerate(subjects_data):
        subj_id = viable_subjects[subj_idx]
        n_samples = data["n_samples"]
        n_windows = subject_windows[subj_idx]["n_windows"]

        source = data["source"][:n_source_channels]  # [channels, samples]
        target = data["target"][:n_target_channels]

        for w in range(n_windows):
            start = w * stride
            end = start + window_size

            source_mmap[window_idx] = source[:, start:end]
            target_mmap[window_idx] = target[:, start:end]
            window_idx += 1

        # Flush periodically
        if subj_idx % 5 == 0:
            source_mmap.flush()
            target_mmap.flush()

        print(f"  {subj_id}: {n_windows} windows (total: {window_idx})")

    # Final flush
    source_mmap.flush()
    target_mmap.flush()
    del source_mmap, target_mmap

    # Save metadata
    metadata = {
        "source_regions": source_regions,
        "target_regions": target_regions,
        "n_source_channels": n_source_channels,
        "n_target_channels": n_target_channels,
        "window_size": window_size,
        "stride": stride,
        "total_windows": total_windows,
        "n_subjects": len(viable_subjects),
        "subjects": viable_subjects,
        "subject_windows": subject_windows,
        "zscore": zscore,
        "source_shape": list(source_shape),
        "target_shape": list(target_shape),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Cached dataset saved to {output_dir}")
    print(f"  - source_windows.npy: {source_path.stat().st_size / 1e9:.2f} GB")
    print(f"  - target_windows.npy: {target_path.stat().st_size / 1e9:.2f} GB")
    print(f"  - metadata.json")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Pre-cache COGITATE dataset as memory-mapped arrays"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_COGITATE_DATA_DIR,
        help="Path to preprocessed COGITATE data",
    )
    parser.add_argument(
        "--source",
        nargs="+",
        default=["temporal"],
        help="Source region(s)",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        default=["frontal"],
        help="Target region(s)",
    )
    parser.add_argument(
        "--source-channels",
        type=int,
        required=True,
        help="Number of source channels",
    )
    parser.add_argument(
        "--target-channels",
        type=int,
        required=True,
        help="Number of target channels",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5120,
        help="Window size in samples (default: 5120 = 5s at 1024Hz)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2560,
        help="Stride between windows (default: 2560 = 50%% overlap)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for cached dataset",
    )
    parser.add_argument(
        "--no-zscore",
        action="store_true",
        help="Don't z-score the data",
    )

    args = parser.parse_args()

    cache_cogitate_dataset(
        source_regions=args.source,
        target_regions=args.target,
        n_source_channels=args.source_channels,
        n_target_channels=args.target_channels,
        window_size=args.window_size,
        stride=args.stride,
        output_dir=args.output,
        data_dir=args.data_dir,
        zscore=not args.no_zscore,
    )


if __name__ == "__main__":
    main()
