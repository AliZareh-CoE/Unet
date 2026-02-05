#!/usr/bin/env python3
"""Download and explore Kai Miller's ECoG Library dataset.

Downloads preprocessed ECoG data from the Neuromatch Academy OSF mirrors
(NumPy .npz format) of the Stanford ECoG Library (Miller, 2019).

Reference:
    Miller, K.J. (2019). A library of human electrocorticographic data and
    analyses. Nature Human Behaviour, 3(11), 1225-1235.
    DOI: 10.1038/s41562-019-0678-3

Usage:
    # Download all experiments
    python scripts/download_ecog.py

    # Download specific experiments
    python scripts/download_ecog.py --experiments fingerflex faceshouses

    # Download and show detailed exploration
    python scripts/download_ecog.py --explore

    # Specify output directory
    python scripts/download_ecog.py --output-dir /data/ECoG
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Experiment metadata: name -> (OSF URL, description, n_subjects)
ECOG_EXPERIMENTS = {
    "fingerflex": {
        "url": "https://osf.io/5m47z/download",
        "description": "Cued individual finger flexion with dataglove",
        "n_subjects": 1,  # 1 subject row, 3 blocks (different patients packed as blocks)
        "n_blocks": 3,
        "task_type": "motor",
        "has_behavioral": True,
    },
    "faceshouses": {
        "url": "https://osf.io/argh7/download",
        "description": "Face vs. house visual perception (400ms stimuli)",
        "n_subjects": 7,  # 7 subjects, 2 blocks each (session 1: basic, session 2: noise)
        "n_blocks": 2,
        "task_type": "visual",
        "has_behavioral": False,
    },
    "motor_imagery": {
        "url": "https://osf.io/ksqv8/download",
        "description": "Motor execution, motor imagery, and imagery feedback",
        "n_subjects": 7,  # 7 subjects, 2 blocks each (execution, imagery)
        "n_blocks": 2,
        "task_type": "motor",
        "has_behavioral": False,
    },
    "joystick_track": {
        "url": "https://osf.io/6jncm/download",
        "description": "2D joystick tracking of moving target",
        "n_subjects": 1,  # 1 subject row, 4 blocks (different patients packed as blocks)
        "n_blocks": 4,
        "task_type": "motor",
        "has_behavioral": True,
    },
    "memory_nback": {
        "url": "https://osf.io/xfc7e/download",
        "description": "N-back working memory task with house stimuli",
        "n_subjects": 3,
        "task_type": "cognitive",
        "has_behavioral": True,
    },
}


def download_file(url: str, filepath: Path, max_retries: int = 3) -> bool:
    """Download a file with retry logic and progress display."""
    import requests

    for attempt in range(max_retries):
        try:
            print(f"  Downloading from {url}...")
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = 100.0 * downloaded / total_size
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        print(
                            f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)",
                            end="",
                            flush=True,
                        )

            print(f"\n  Saved to: {filepath}")
            return True

        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"\n  Download failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)

    return False


def load_ecog_experiment(filepath: Path) -> np.ndarray:
    """Load an ECoG experiment .npz file.

    Returns:
        alldat: Array of shape (n_subjects,) where each element is an array
                of blocks. Each block is a dict with keys like 'V', 'srate', etc.
    """
    data = np.load(filepath, allow_pickle=True)
    return data["dat"]


def explore_experiment(name: str, filepath: Path) -> Dict:
    """Explore the structure and contents of a downloaded experiment.

    Returns a summary dict with experiment metadata.
    """
    info = ECOG_EXPERIMENTS[name]
    alldat = load_ecog_experiment(filepath)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*70}")
    print(f"Description: {info['description']}")
    print(f"Task type:   {info['task_type']}")
    print(f"Subjects:    {len(alldat)}")

    summary = {
        "name": name,
        "n_subjects": len(alldat),
        "subjects": [],
    }

    for subj_idx in range(len(alldat)):
        n_blocks = len(alldat[subj_idx])
        print(f"\n  Subject {subj_idx}: {n_blocks} block(s)")

        subj_info = {"n_blocks": n_blocks, "blocks": []}

        for block_idx in range(n_blocks):
            dat = alldat[subj_idx][block_idx]
            block_info = {"keys": list(dat.keys())}

            # Core signal info
            if "V" in dat:
                V = np.float32(dat["V"])
                n_samples, n_channels = V.shape
                # srate can be int, float, or nested ndarray like [[1000]]
                srate_raw = dat.get("srate", 1000)
                if isinstance(srate_raw, np.ndarray):
                    srate = int(srate_raw.flat[0])
                else:
                    srate = int(srate_raw)
                duration_s = n_samples / srate

                block_info.update({
                    "n_samples": n_samples,
                    "n_channels": n_channels,
                    "srate": srate,
                    "duration_s": float(duration_s),
                })

                print(f"    Block {block_idx}: {n_channels} channels, "
                      f"{n_samples} samples ({duration_s:.1f}s), "
                      f"srate={srate}Hz")
                print(f"      Keys: {list(dat.keys())}")

            # Anatomical info - parse lobe names to canonical form
            if "lobe" in dat:
                lobes = dat["lobe"]
                # Count channels per lobe (normalized)
                lobe_counts = {}
                for l in lobes:
                    lname = str(l).strip().lower()
                    for known in ("frontal", "temporal", "parietal", "occipital", "limbic"):
                        if known in lname:
                            lname = known
                            break
                    lobe_counts[lname] = lobe_counts.get(lname, 0) + 1
                block_info["lobe_counts"] = lobe_counts
                lobes_str = ", ".join(f"{k}: {v}ch" for k, v in sorted(lobe_counts.items()))
                print(f"      Lobes: {lobes_str}")

            if "gyrus" in dat:
                gyri = dat["gyrus"]
                unique_gyri = list(set(str(g) for g in gyri if str(g).strip()))
                block_info["gyri"] = unique_gyri
                if len(unique_gyri) <= 10:
                    print(f"      Gyri: {unique_gyri}")
                else:
                    print(f"      Gyri: {len(unique_gyri)} unique regions")

            if "Brodmann_Area" in dat:
                bas = dat["Brodmann_Area"]
                unique_bas = sorted(set(int(b) for b in bas if not np.isnan(float(b))))
                block_info["brodmann_areas"] = unique_bas
                print(f"      Brodmann Areas: {unique_bas}")

            # Stimulus info
            if "t_on" in dat:
                n_events = len(dat["t_on"])
                block_info["n_events"] = n_events
                print(f"      Events: {n_events} stimulus onsets")

            if "stim_id" in dat:
                stim_ids = dat["stim_id"]
                unique_stim = sorted(set(int(s) for s in stim_ids))
                block_info["unique_stim_ids"] = unique_stim
                print(f"      Stimulus IDs: {unique_stim}")

            # Finger flexion specific
            if "dg" in dat:
                dg = np.float32(dat["dg"])
                block_info["dg_shape"] = list(dg.shape)
                print(f"      Finger data: {dg.shape} (samples x 5 fingers)")

            # Joystick specific
            if "targetX" in dat:
                print(f"      Joystick tracking data present")

            # Memory n-back specific
            if "target" in dat:
                targets = dat["target"]
                n_targets = int(np.sum(targets))
                block_info["n_targets"] = n_targets
                print(f"      Targets: {n_targets} target events")

            subj_info["blocks"].append(block_info)

        summary["subjects"].append(subj_info)

    return summary


def analyze_region_coverage(name: str, filepath: Path) -> Dict:
    """Analyze brain region coverage across subjects for region-pairing.

    This helps determine which region pairs can be used for
    inter-region neural signal translation.
    """
    alldat = load_ecog_experiment(filepath)

    print(f"\n{'='*70}")
    print(f"REGION COVERAGE ANALYSIS: {name}")
    print(f"{'='*70}")

    region_map = {}  # subject -> {lobe: [channel_indices]}

    for subj_idx in range(len(alldat)):
        dat = alldat[subj_idx][0]  # Use first block
        if "lobe" not in dat or "V" not in dat:
            continue

        lobes = [str(l).strip() for l in dat["lobe"]]
        n_channels = np.float32(dat["V"]).shape[1]

        region_channels = {}
        for ch_idx, lobe in enumerate(lobes):
            if ch_idx >= n_channels:
                break
            if lobe and lobe != "nan":
                if lobe not in region_channels:
                    region_channels[lobe] = []
                region_channels[lobe].append(ch_idx)

        region_map[subj_idx] = region_channels

        regions_str = ", ".join(
            f"{lobe}: {len(chs)}ch" for lobe, chs in sorted(region_channels.items())
        )
        print(f"  Subject {subj_idx}: {regions_str}")

    # Find which region pairs are available across most subjects
    print(f"\nViable region pairs (for source->target translation):")
    all_regions = set()
    for subj_regions in region_map.values():
        all_regions.update(subj_regions.keys())

    all_regions = sorted(all_regions)
    for src in all_regions:
        for tgt in all_regions:
            if src >= tgt:
                continue
            n_subjects_with_both = sum(
                1 for subj_regions in region_map.values()
                if src in subj_regions and tgt in subj_regions
                and len(subj_regions[src]) >= 4 and len(subj_regions[tgt]) >= 4
            )
            if n_subjects_with_both >= 2:
                print(f"  {src} -> {tgt}: {n_subjects_with_both} subjects "
                      f"(>= 4 channels each)")

    return region_map


def main():
    parser = argparse.ArgumentParser(
        description="Download and explore Kai Miller's ECoG Library dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_ecog.py                          # Download all
    python scripts/download_ecog.py --experiments fingerflex  # Download one
    python scripts/download_ecog.py --explore                 # Full exploration
    python scripts/download_ecog.py --regions                 # Region analysis
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: $UNET_DATA_DIR/ECoG or /data/ECoG)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        choices=list(ECOG_EXPERIMENTS.keys()),
        default=None,
        help="Specific experiments to download (default: all)",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Show detailed exploration of downloaded data",
    )
    parser.add_argument(
        "--regions",
        action="store_true",
        help="Analyze brain region coverage for inter-region translation",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit",
    )

    args = parser.parse_args()

    # List experiments
    if args.list:
        print("Available ECoG Library experiments:")
        print(f"{'Name':<20} {'Subjects':<10} {'Type':<10} {'Description'}")
        print("-" * 80)
        for name, info in ECOG_EXPERIMENTS.items():
            print(f"{name:<20} {info['n_subjects']:<10} {info['task_type']:<10} {info['description']}")
        return 0

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        data_dir = Path(os.environ.get("UNET_DATA_DIR", "/data"))
        output_dir = data_dir / "ECoG"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"ECoG data directory: {output_dir}")

    # Select experiments
    experiments = args.experiments or list(ECOG_EXPERIMENTS.keys())

    # Download
    print(f"\nDownloading {len(experiments)} experiment(s)...")
    downloaded = []
    failed = []

    for name in experiments:
        info = ECOG_EXPERIMENTS[name]
        filepath = output_dir / f"{name}.npz"

        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"\n[{name}] Already exists ({size_mb:.1f} MB), skipping")
            downloaded.append(name)
            continue

        print(f"\n[{name}] {info['description']} ({info['n_subjects']} subjects)")
        if download_file(info["url"], filepath):
            downloaded.append(name)
        else:
            failed.append(name)
            print(f"  FAILED to download {name}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Download complete: {len(downloaded)}/{len(experiments)} experiments")
    if failed:
        print(f"Failed: {failed}")
    print(f"Data directory: {output_dir}")

    # Explore
    if args.explore:
        print("\n\n" + "#" * 70)
        print("# DATA EXPLORATION")
        print("#" * 70)

        for name in downloaded:
            filepath = output_dir / f"{name}.npz"
            if filepath.exists():
                explore_experiment(name, filepath)

    # Region analysis
    if args.regions:
        print("\n\n" + "#" * 70)
        print("# BRAIN REGION COVERAGE ANALYSIS")
        print("#" * 70)

        for name in downloaded:
            filepath = output_dir / f"{name}.npz"
            if filepath.exists():
                analyze_region_coverage(name, filepath)

    return 0


if __name__ == "__main__":
    sys.exit(main())
