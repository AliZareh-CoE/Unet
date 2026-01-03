#!/usr/bin/env python3
"""
Allen Institute Visual Coding Neuropixels Data Download & Preparation
======================================================================

Downloads the Brain Observatory 1.1 stimulus set data from Allen Institute
and prepares it for neural translation training.

Dataset: Visual Coding - Neuropixels
Source: https://portal.brain-map.org/circuits-behavior/visual-coding-neuropixels

The data includes simultaneous recordings from multiple brain regions:
- Visual cortex areas (V1, LM, AL, PM, AM, RL)
- Thalamus (LGN, LP)
- Hippocampus (CA1, CA3, DG)
- Other areas (SUB, ProS, etc.)

Usage:
    # Install dependencies first
    pip install allensdk pandas numpy tqdm

    # Download all Brain Observatory 1.1 sessions
    python download_allen_neuropixels.py --all

    # Download specific number of sessions
    python download_allen_neuropixels.py --n-sessions 10

    # Download specific session
    python download_allen_neuropixels.py --session-id 715093703

Author: Neural Translation Project
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Output directory
OUTPUT_DIR = Path("/data/allen_neuropixels")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

# Target sampling rate (downsample from 2500 Hz to 1000 Hz)
TARGET_FS = 1000
ORIGINAL_FS = 2500  # Allen LFP is at 2500 Hz


def install_allensdk():
    """Install AllenSDK if not available."""
    try:
        import allensdk
        print(f"AllenSDK version: {allensdk.__version__}")
        return True
    except ImportError:
        print("AllenSDK not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "allensdk"])
        return True


def get_cache():
    """Create EcephysProjectCache for data access."""
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize cache - will download manifest automatically
    cache = EcephysProjectCache.from_warehouse(manifest=str(MANIFEST_PATH))
    return cache


def get_brain_observatory_sessions(cache) -> pd.DataFrame:
    """Get all sessions with Brain Observatory 1.1 stimulus set."""
    sessions = cache.get_session_table()

    # Filter for Brain Observatory 1.1 stimulus set
    bo_sessions = sessions[sessions['session_type'] == 'brain_observatory_1.1']

    print(f"Found {len(bo_sessions)} Brain Observatory 1.1 sessions")
    return bo_sessions


def get_region_mapping() -> Dict[str, List[str]]:
    """Define brain region groupings for translation tasks."""
    return {
        # Visual cortex hierarchy
        'V1': ['VISp'],  # Primary visual cortex
        'LM': ['VISl'],  # Lateromedial
        'AL': ['VISal'],  # Anterolateral
        'PM': ['VISpm'],  # Posteromedial
        'AM': ['VISam'],  # Anteromedial
        'RL': ['VISrl'],  # Rostrolateral

        # Thalamus
        'LGN': ['LGd', 'LGv'],  # Lateral geniculate nucleus
        'LP': ['LP'],  # Lateral posterior

        # Hippocampus
        'CA1': ['CA1'],
        'CA3': ['CA3'],
        'DG': ['DG'],

        # Other
        'SUB': ['SUB', 'ProS'],  # Subiculum
    }


def downsample_lfp(lfp_data: np.ndarray, orig_fs: int = 2500, target_fs: int = 1000) -> np.ndarray:
    """Downsample LFP data from original to target sampling rate."""
    from scipy.signal import resample_poly

    # Calculate resampling factors
    # 2500 -> 1000 = downsample by 5/2
    gcd = np.gcd(orig_fs, target_fs)
    up = target_fs // gcd
    down = orig_fs // gcd

    # Resample along time axis (last axis)
    return resample_poly(lfp_data, up, down, axis=-1).astype(np.float32)


def extract_session_lfp(cache, session_id: int, output_path: Path) -> Optional[Dict]:
    """
    Extract LFP data from a session and organize by brain region.

    Returns metadata dict or None if failed.
    """
    from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession

    try:
        # Load session
        session = cache.get_session_data(session_id)

        # Get channel info
        channels = session.channels
        probes = session.probes

        # Get unique structures
        structures = channels['ecephys_structure_acronym'].unique()
        structures = [s for s in structures if s is not None and s != 'none']

        print(f"  Structures found: {structures}")

        # Load LFP for each probe
        probe_ids = probes.index.tolist()

        region_data = {}
        region_channels = {}

        for probe_id in probe_ids:
            try:
                # Get LFP for this probe
                lfp = session.get_lfp(probe_id)

                if lfp is None:
                    continue

                # Get channels for this probe
                probe_channels = channels[channels['probe_id'] == probe_id]

                # Organize by structure
                for structure in probe_channels['ecephys_structure_acronym'].unique():
                    if structure is None or structure == 'none':
                        continue

                    # Get channel indices for this structure
                    struct_channels = probe_channels[
                        probe_channels['ecephys_structure_acronym'] == structure
                    ]
                    channel_ids = struct_channels.index.tolist()

                    # Extract LFP for these channels
                    # lfp is xarray DataArray with dims (time, channel)
                    try:
                        struct_lfp = lfp.sel(channel=channel_ids).values.T  # (channels, time)
                    except:
                        # Channel IDs might not match exactly
                        available_channels = [c for c in channel_ids if c in lfp.channel.values]
                        if not available_channels:
                            continue
                        struct_lfp = lfp.sel(channel=available_channels).values.T

                    if structure not in region_data:
                        region_data[structure] = []
                        region_channels[structure] = 0

                    region_data[structure].append(struct_lfp)
                    region_channels[structure] += struct_lfp.shape[0]

            except Exception as e:
                print(f"    Warning: Failed to load LFP for probe {probe_id}: {e}")
                continue

        if not region_data:
            print(f"  No LFP data extracted for session {session_id}")
            return None

        # Concatenate channels per region and save
        output_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            'session_id': session_id,
            'original_fs': ORIGINAL_FS,
            'target_fs': TARGET_FS,
            'regions': {},
            'stimulus_set': 'brain_observatory_1.1',
            'processed_at': datetime.now().isoformat(),
        }

        for structure, lfp_list in region_data.items():
            # Different probes may have different lengths - truncate to minimum
            min_len = min(lfp.shape[1] for lfp in lfp_list)
            lfp_list_truncated = [lfp[:, :min_len] for lfp in lfp_list]

            # Concatenate all channels for this structure
            combined = np.concatenate(lfp_list_truncated, axis=0)

            # Downsample to 1kHz
            downsampled = downsample_lfp(combined, ORIGINAL_FS, TARGET_FS)

            # Save
            np.save(output_path / f'{structure}.npy', downsampled)

            metadata['regions'][structure] = {
                'n_channels': downsampled.shape[0],
                'n_samples': downsampled.shape[1],
                'duration_sec': downsampled.shape[1] / TARGET_FS,
            }

            print(f"    {structure}: {downsampled.shape[0]} channels, "
                  f"{downsampled.shape[1] / TARGET_FS:.1f} sec")

        # Save metadata
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    except Exception as e:
        print(f"  Error processing session {session_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_sessions(n_sessions: Optional[int] = None,
                     session_ids: Optional[List[int]] = None,
                     skip_existing: bool = True):
    """Download and process sessions."""

    # Install AllenSDK if needed
    install_allensdk()

    # Get cache
    print("Initializing EcephysProjectCache...")
    cache = get_cache()

    # Get Brain Observatory 1.1 sessions
    sessions = get_brain_observatory_sessions(cache)

    # Filter session list
    if session_ids:
        sessions = sessions[sessions.index.isin(session_ids)]
    elif n_sessions:
        sessions = sessions.head(n_sessions)

    print(f"\nWill process {len(sessions)} sessions")
    print(f"Output directory: {OUTPUT_DIR}")

    # Process each session
    results = []
    for idx, (session_id, row) in enumerate(tqdm(sessions.iterrows(), total=len(sessions))):
        output_path = OUTPUT_DIR / f"session_{session_id}"

        # Skip if exists
        if skip_existing and (output_path / 'metadata.json').exists():
            print(f"\n[{idx+1}/{len(sessions)}] Session {session_id}: Already exists, skipping")
            continue

        print(f"\n[{idx+1}/{len(sessions)}] Processing session {session_id}...")

        metadata = extract_session_lfp(cache, session_id, output_path)
        if metadata:
            results.append(metadata)

    # Save summary
    summary = {
        'n_sessions': len(results),
        'sessions': [r['session_id'] for r in results],
        'stimulus_set': 'brain_observatory_1.1',
        'sampling_rate': TARGET_FS,
        'downloaded_at': datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / 'download_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Sessions processed: {len(results)}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}")


def create_paired_dataset(source_region: str = 'VISp',
                         target_region: str = 'CA1',
                         window_size: int = 5000,
                         stride: int = 2500):
    """
    Create paired dataset for training (similar to PCx1 format).

    Finds sessions that have both source and target regions,
    then creates sliding windows.
    """
    paired_output = OUTPUT_DIR / f"paired_{source_region}_{target_region}"
    paired_output.mkdir(parents=True, exist_ok=True)

    # Find sessions with both regions
    sessions = sorted(OUTPUT_DIR.glob("session_*"))

    valid_sessions = []
    for sess_path in sessions:
        source_file = sess_path / f'{source_region}.npy'
        target_file = sess_path / f'{target_region}.npy'

        if source_file.exists() and target_file.exists():
            valid_sessions.append(sess_path)

    print(f"Found {len(valid_sessions)} sessions with both {source_region} and {target_region}")

    # Process each valid session
    for sess_path in tqdm(valid_sessions):
        session_name = sess_path.name

        # Load data
        source = np.load(sess_path / f'{source_region}.npy')
        target = np.load(sess_path / f'{target_region}.npy')

        # Align lengths
        min_len = min(source.shape[1], target.shape[1])
        source = source[:, :min_len]
        target = target[:, :min_len]

        # Save in PCx1 format
        out_path = paired_output / session_name
        out_path.mkdir(parents=True, exist_ok=True)

        np.save(out_path / 'source.npy', source)
        np.save(out_path / 'target.npy', target)

        # Metadata
        meta = {
            'session': session_name,
            'source_region': source_region,
            'target_region': target_region,
            'source_channels': source.shape[0],
            'target_channels': target.shape[0],
            'n_samples': min_len,
            'duration_sec': min_len / TARGET_FS,
            'sampling_rate': TARGET_FS,
        }

        with open(out_path / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

    print(f"\nPaired dataset saved to: {paired_output}")
    return paired_output


def main():
    parser = argparse.ArgumentParser(
        description="Download Allen Visual Coding Neuropixels data"
    )

    parser.add_argument('--all', action='store_true',
                       help='Download all Brain Observatory 1.1 sessions')
    parser.add_argument('--n-sessions', type=int, default=None,
                       help='Number of sessions to download')
    parser.add_argument('--session-id', type=int, nargs='+', default=None,
                       help='Specific session IDs to download')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip sessions that already exist')
    parser.add_argument('--no-skip', action='store_false', dest='skip_existing',
                       help='Reprocess existing sessions')

    # Paired dataset creation
    parser.add_argument('--create-paired', action='store_true',
                       help='Create paired dataset for training')
    parser.add_argument('--source-region', type=str, default='VISp',
                       help='Source brain region (default: VISp = V1)')
    parser.add_argument('--target-region', type=str, default='CA1',
                       help='Target brain region (default: CA1)')

    args = parser.parse_args()

    if args.create_paired:
        create_paired_dataset(args.source_region, args.target_region)
    elif args.all:
        download_sessions(skip_existing=args.skip_existing)
    elif args.n_sessions:
        download_sessions(n_sessions=args.n_sessions, skip_existing=args.skip_existing)
    elif args.session_id:
        download_sessions(session_ids=args.session_id, skip_existing=args.skip_existing)
    else:
        # Default: download 5 sessions as a test
        print("No options specified. Downloading 5 sessions as a test...")
        print("Use --all for all sessions, --n-sessions N for N sessions")
        download_sessions(n_sessions=5, skip_existing=args.skip_existing)


if __name__ == '__main__':
    main()
