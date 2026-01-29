#!/usr/bin/env python3
"""
BIDS iEEG Data Explorer
Analyzes the COGITATE ECOG BIDS dataset structure and contents.

Usage:
    python explore_bids_data.py /path/to/COG_ECOG_EXP1_BIDS
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd


def explore_bids_dataset(bids_root: str):
    """Comprehensive exploration of BIDS iEEG dataset."""
    bids_path = Path(bids_root)

    print("=" * 80)
    print("COGITATE BIDS iEEG DATA EXPLORATION")
    print("=" * 80)

    # 1. Dataset description
    print("\n" + "=" * 40)
    print("1. DATASET DESCRIPTION")
    print("=" * 40)
    desc_file = bids_path / "dataset_description.json"
    if desc_file.exists():
        with open(desc_file) as f:
            desc = json.load(f)
        for key, val in desc.items():
            print(f"  {key}: {val}")
    else:
        print("  dataset_description.json not found")

    # 2. Participants info
    print("\n" + "=" * 40)
    print("2. PARTICIPANTS")
    print("=" * 40)
    participants_file = bids_path / "participants.tsv"
    if participants_file.exists():
        participants = pd.read_csv(participants_file, sep='\t')
        print(f"  Total participants: {len(participants)}")
        print(f"  Columns: {list(participants.columns)}")
        print(f"\n  First 5 participants:")
        print(participants.head().to_string(index=False))
        if 'age' in participants.columns:
            print(f"\n  Age range: {participants['age'].min()} - {participants['age'].max()}")
        if 'sex' in participants.columns:
            print(f"  Sex distribution: {participants['sex'].value_counts().to_dict()}")
    else:
        print("  participants.tsv not found")

    # Check epilepsy-specific info
    epilepsy_file = bids_path / "participants_epilepsy.tsv"
    if epilepsy_file.exists():
        print("\n  Epilepsy info available (participants_epilepsy.tsv)")
        epi = pd.read_csv(epilepsy_file, sep='\t')
        print(f"  Epilepsy columns: {list(epi.columns)}")

    # 3. Subject directories
    print("\n" + "=" * 40)
    print("3. SUBJECT STRUCTURE")
    print("=" * 40)
    subject_dirs = sorted([d for d in bids_path.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    print(f"  Total subject directories: {len(subject_dirs)}")
    print(f"  Subject IDs: {[d.name for d in subject_dirs]}")

    # Group by prefix
    prefixes = defaultdict(list)
    for d in subject_dirs:
        prefix = d.name.split('-')[1][:2]  # e.g., CE, CF, CG
        prefixes[prefix].append(d.name)
    print(f"\n  By site prefix:")
    for prefix, subs in sorted(prefixes.items()):
        print(f"    {prefix}: {len(subs)} subjects")

    # 4. Analyze one subject in detail
    print("\n" + "=" * 40)
    print("4. DETAILED SUBJECT ANALYSIS")
    print("=" * 40)

    if subject_dirs:
        example_sub = subject_dirs[0]
        print(f"\n  Analyzing: {example_sub.name}")

        # List all contents recursively
        print(f"\n  Directory structure:")
        for item in sorted(example_sub.rglob('*')):
            rel_path = item.relative_to(example_sub)
            indent = "    " * len(rel_path.parts)
            if item.is_file():
                size_kb = item.stat().st_size / 1024
                print(f"  {indent}{item.name} ({size_kb:.1f} KB)")
            else:
                print(f"  {indent}{item.name}/")

        # Look for ieeg folder
        ieeg_dir = example_sub / "ieeg"
        if ieeg_dir.exists():
            print(f"\n  iEEG files found:")
            for f in sorted(ieeg_dir.iterdir()):
                print(f"    - {f.name}")

            # Analyze channels.tsv
            channels_files = list(ieeg_dir.glob("*_channels.tsv"))
            if channels_files:
                print(f"\n  Channels info ({channels_files[0].name}):")
                channels = pd.read_csv(channels_files[0], sep='\t')
                print(f"    Total channels: {len(channels)}")
                print(f"    Columns: {list(channels.columns)}")
                if 'type' in channels.columns:
                    print(f"    Channel types: {channels['type'].value_counts().to_dict()}")
                if 'status' in channels.columns:
                    print(f"    Channel status: {channels['status'].value_counts().to_dict()}")
                if 'group' in channels.columns or 'group_name' in channels.columns:
                    grp_col = 'group' if 'group' in channels.columns else 'group_name'
                    print(f"    Electrode groups: {channels[grp_col].nunique()} unique")
                print(f"\n    First 10 channels:")
                print(channels.head(10).to_string(index=False))

            # Analyze events.tsv
            events_files = list(ieeg_dir.glob("*_events.tsv"))
            if events_files:
                print(f"\n  Events info ({events_files[0].name}):")
                events = pd.read_csv(events_files[0], sep='\t')
                print(f"    Total events: {len(events)}")
                print(f"    Columns: {list(events.columns)}")
                if 'trial_type' in events.columns:
                    print(f"    Trial types: {events['trial_type'].value_counts().to_dict()}")
                elif 'value' in events.columns:
                    print(f"    Event values: {events['value'].unique()[:20]}")
                if 'onset' in events.columns:
                    print(f"    Duration: {events['onset'].max() - events['onset'].min():.1f} seconds")
                print(f"\n    First 10 events:")
                print(events.head(10).to_string(index=False))

            # Check for electrode positions
            electrodes_files = list(ieeg_dir.glob("*_electrodes.tsv"))
            if electrodes_files:
                print(f"\n  Electrode positions ({electrodes_files[0].name}):")
                electrodes = pd.read_csv(electrodes_files[0], sep='\t')
                print(f"    Total electrodes: {len(electrodes)}")
                print(f"    Columns: {list(electrodes.columns)}")
                if all(c in electrodes.columns for c in ['x', 'y', 'z']):
                    print(f"    Has 3D coordinates: Yes")
                print(f"\n    First 10 electrodes:")
                print(electrodes.head(10).to_string(index=False))

            # Check ieeg.json sidecar
            json_files = list(ieeg_dir.glob("*_ieeg.json"))
            if json_files:
                print(f"\n  iEEG metadata ({json_files[0].name}):")
                with open(json_files[0]) as f:
                    ieeg_meta = json.load(f)
                for key, val in ieeg_meta.items():
                    if not isinstance(val, (dict, list)):
                        print(f"    {key}: {val}")

            # Check data file format
            print(f"\n  Data file formats:")
            for ext in ['.edf', '.vhdr', '.nwb', '.set', '.fif', '.bdf']:
                files = list(ieeg_dir.glob(f"*{ext}"))
                if files:
                    print(f"    {ext}: {len(files)} files")
                    for f in files[:3]:
                        size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"      - {f.name} ({size_mb:.1f} MB)")

    # 5. Derivatives
    print("\n" + "=" * 40)
    print("5. DERIVATIVES")
    print("=" * 40)
    derivatives_dir = bids_path / "derivatives"
    if derivatives_dir.exists():
        print(f"  Derivatives folder exists")
        for item in sorted(derivatives_dir.iterdir()):
            if item.is_dir():
                print(f"    - {item.name}/")
                # Count files
                n_files = len(list(item.rglob('*')))
                print(f"      ({n_files} items)")
            else:
                print(f"    - {item.name}")
    else:
        print("  No derivatives folder")

    # 6. Summary statistics across all subjects
    print("\n" + "=" * 40)
    print("6. CROSS-SUBJECT SUMMARY")
    print("=" * 40)

    channel_counts = []
    event_counts = []
    data_formats = defaultdict(int)

    for sub_dir in subject_dirs:
        ieeg_dir = sub_dir / "ieeg"
        if ieeg_dir.exists():
            # Count channels
            ch_files = list(ieeg_dir.glob("*_channels.tsv"))
            if ch_files:
                ch = pd.read_csv(ch_files[0], sep='\t')
                channel_counts.append((sub_dir.name, len(ch)))

            # Count events
            ev_files = list(ieeg_dir.glob("*_events.tsv"))
            if ev_files:
                ev = pd.read_csv(ev_files[0], sep='\t')
                event_counts.append((sub_dir.name, len(ev)))

            # Data formats
            for ext in ['.edf', '.vhdr', '.nwb', '.set', '.fif', '.bdf']:
                if list(ieeg_dir.glob(f"*{ext}")):
                    data_formats[ext] += 1

    if channel_counts:
        counts = [c[1] for c in channel_counts]
        print(f"  Channel counts across subjects:")
        print(f"    Min: {min(counts)}, Max: {max(counts)}, Mean: {sum(counts)/len(counts):.1f}")

    if event_counts:
        counts = [c[1] for c in event_counts]
        print(f"  Event counts across subjects:")
        print(f"    Min: {min(counts)}, Max: {max(counts)}, Mean: {sum(counts)/len(counts):.1f}")

    if data_formats:
        print(f"  Data formats used:")
        for fmt, count in sorted(data_formats.items(), key=lambda x: -x[1]):
            print(f"    {fmt}: {count} subjects")

    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore BIDS iEEG dataset")
    parser.add_argument("bids_root", help="Path to BIDS dataset root directory")
    args = parser.parse_args()

    if not Path(args.bids_root).exists():
        print(f"Error: Path does not exist: {args.bids_root}")
        exit(1)

    explore_bids_dataset(args.bids_root)
