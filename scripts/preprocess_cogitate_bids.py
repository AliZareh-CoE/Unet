#!/usr/bin/env python3
"""
Preprocess COGITATE BIDS iEEG Data to Unified NPY Format
=========================================================

Converts COGITATE EXP1 BIDS dataset to a unified .npy format for neural
signal translation experiments.

Input: /data/COG_ECOG_EXP1_BIDS (BIDS format with BrainVision files)
Output: /data/COGITATEDataset/ (unified .npy format)

Output Structure:
    /data/COGITATEDataset/
    ├── sub-CE103/
    │   ├── continuous.npy       # (n_channels, n_samples) continuous iEEG
    │   ├── channels.csv         # Channel metadata with region labels
    │   └── events.csv           # Stimulus events with timing
    ├── sub-CE106/
    │   └── ...
    ├── metadata.csv             # All subjects metadata
    ├── region_mapping.json      # Region name -> channel indices per subject
    └── dataset_info.json        # Dataset summary (sampling rate, etc.)

Usage:
    python scripts/preprocess_cogitate_bids.py /data/COG_ECOG_EXP1_BIDS

    # With options
    python scripts/preprocess_cogitate_bids.py /data/COG_ECOG_EXP1_BIDS \
        --output-dir /data/COGITATEDataset \
        --target-sfreq 1000 \
        --regions hippocampus amygdala temporal

Requirements:
    pip install mne mne-bids pandas numpy
"""

import argparse
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Suppress MNE verbose output by default
os.environ.setdefault('MNE_LOGGING_LEVEL', 'WARNING')

try:
    import mne
    mne.set_log_level('WARNING')
except ImportError:
    print("ERROR: MNE-Python is required. Install with: pip install mne")
    sys.exit(1)


# =============================================================================
# Region Mapping from Atlas Labels
# =============================================================================

# Map Desikan atlas labels to simplified region names
DESIKAN_REGION_MAP = {
    # Hippocampus
    'hippocampus': ['Left-Hippocampus', 'Right-Hippocampus'],
    # Amygdala
    'amygdala': ['Left-Amygdala', 'Right-Amygdala'],
    # Temporal cortex
    'temporal': [
        'ctx-lh-superiortemporal', 'ctx-rh-superiortemporal',
        'ctx-lh-middletemporal', 'ctx-rh-middletemporal',
        'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal',
        'ctx-lh-transversetemporal', 'ctx-rh-transversetemporal',
        'ctx-lh-bankssts', 'ctx-rh-bankssts',
    ],
    # Fusiform (visual word form area)
    'fusiform': ['ctx-lh-fusiform', 'ctx-rh-fusiform'],
    # Insula
    'insula': ['ctx-lh-insula', 'ctx-rh-insula'],
    # Frontal
    'frontal': [
        'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal',
        'ctx-lh-rostralmiddlefrontal', 'ctx-rh-rostralmiddlefrontal',
        'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal',
        'ctx-lh-parsopercularis', 'ctx-rh-parsopercularis',
        'ctx-lh-parstriangularis', 'ctx-rh-parstriangularis',
        'ctx-lh-parsorbitalis', 'ctx-rh-parsorbitalis',
        'ctx-lh-lateralorbitofrontal', 'ctx-rh-lateralorbitofrontal',
        'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal',
        'ctx-lh-frontalpole', 'ctx-rh-frontalpole',
    ],
    # Parietal
    'parietal': [
        'ctx-lh-superiorparietal', 'ctx-rh-superiorparietal',
        'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',
        'ctx-lh-supramarginal', 'ctx-rh-supramarginal',
        'ctx-lh-postcentral', 'ctx-rh-postcentral',
        'ctx-lh-precuneus', 'ctx-rh-precuneus',
    ],
    # Occipital (visual)
    'occipital': [
        'ctx-lh-lateraloccipital', 'ctx-rh-lateraloccipital',
        'ctx-lh-lingual', 'ctx-rh-lingual',
        'ctx-lh-cuneus', 'ctx-rh-cuneus',
        'ctx-lh-pericalcarine', 'ctx-rh-pericalcarine',
    ],
    # Cingulate
    'cingulate': [
        'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',
        'ctx-lh-caudalanteriorcingulate', 'ctx-rh-caudalanteriorcingulate',
        'ctx-lh-posteriorcingulate', 'ctx-rh-posteriorcingulate',
        'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate',
    ],
    # Entorhinal (memory)
    'entorhinal': ['ctx-lh-entorhinal', 'ctx-rh-entorhinal'],
    # Parahippocampal
    'parahippocampal': ['ctx-lh-parahippocampal', 'ctx-rh-parahippocampal'],
}

# Also map electrode naming conventions (backup if atlas fails)
ELECTRODE_NAME_REGION_MAP = {
    'hippocampus': ['LAH', 'RAH', 'LMH', 'RMH', 'LPH', 'RPH', 'HIP', 'HC'],
    'amygdala': ['LA', 'RA', 'LAM', 'RAM', 'AMY', 'AM'],
    'temporal': ['LT', 'RT', 'LST', 'RST', 'LIT', 'RIT', 'LMT', 'RMT'],
    'insula': ['LIN', 'RIN', 'INS'],
    'frontal': ['LF', 'RF', 'LOF', 'ROF', 'LPF', 'RPF'],
    'occipital': ['LO', 'RO', 'LOC', 'ROC'],
}


def parse_atlas_labels(labels_str: str) -> Set[str]:
    """Parse atlas label string into set of region names.

    Atlas labels are formatted as: "Region1/Region2/Region3"
    where each channel may span multiple regions.
    """
    if pd.isna(labels_str) or labels_str in ['Unknown', 'n/a', '']:
        return set()
    return set(labels_str.split('/'))


def map_channel_to_region(
    channel_name: str,
    atlas_labels: str,
    region_map: Dict[str, List[str]] = DESIKAN_REGION_MAP,
) -> Optional[str]:
    """Map a channel to a simplified region name.

    First tries atlas labels, then falls back to electrode naming convention.

    Args:
        channel_name: Electrode name (e.g., 'LAH1', 'LA3')
        atlas_labels: Atlas label string from BIDS (e.g., 'Left-Hippocampus/Left-Cerebral-White-Matter')
        region_map: Mapping from region name to atlas labels

    Returns:
        Region name (e.g., 'hippocampus') or None if unmapped
    """
    labels = parse_atlas_labels(atlas_labels)

    # Try atlas labels first
    for region_name, atlas_patterns in region_map.items():
        for pattern in atlas_patterns:
            if pattern in labels:
                return region_name

    # Fall back to electrode naming convention
    ch_upper = channel_name.upper()
    for region_name, prefixes in ELECTRODE_NAME_REGION_MAP.items():
        for prefix in prefixes:
            # Match prefix followed by numbers (e.g., LAH1, LAH2)
            if re.match(f'^{prefix}\\d*$', ch_upper):
                return region_name

    return None


# =============================================================================
# BIDS Data Loading
# =============================================================================

def load_bids_subject(
    bids_root: Path,
    subject_id: str,
    task: str = "Dur",
    session: str = "1",
) -> Dict[str, Any]:
    """Load a single subject's data from BIDS format.

    Args:
        bids_root: Path to BIDS root directory
        subject_id: Subject ID (e.g., 'sub-CE103' or 'CE103')
        task: Task name (default: 'Dur')
        session: Session number (default: '1')

    Returns:
        Dictionary with raw data, channels, events, and metadata
    """
    # Normalize subject ID
    if not subject_id.startswith('sub-'):
        subject_id = f'sub-{subject_id}'

    subject_dir = bids_root / subject_id / f'ses-{session}' / 'ieeg'

    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    # Find the BrainVision header file
    vhdr_files = list(subject_dir.glob(f'*_task-{task}_ieeg.vhdr'))
    if not vhdr_files:
        raise FileNotFoundError(f"No .vhdr file found for {subject_id} task-{task}")
    vhdr_path = vhdr_files[0]

    # Load raw data with MNE
    print(f"  Loading {vhdr_path.name}...")
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

    # Load channels metadata
    channels_file = subject_dir / f'{subject_id}_ses-{session}_task-{task}_channels.tsv'
    if channels_file.exists():
        channels_df = pd.read_csv(channels_file, sep='\t')
    else:
        # Create basic channels info from raw
        channels_df = pd.DataFrame({
            'name': raw.ch_names,
            'type': ['SEEG'] * len(raw.ch_names),
            'status': ['good'] * len(raw.ch_names),
        })

    # Load atlas labels (Desikan)
    atlas_file = subject_dir / f'{subject_id}_ses-{session}_atlas-desikan_labels.tsv'
    if atlas_file.exists():
        atlas_df = pd.read_csv(atlas_file, sep='\t')
        # Merge atlas labels with channels
        if 'channel' in atlas_df.columns and 'region' in atlas_df.columns:
            atlas_map = dict(zip(atlas_df['channel'], atlas_df['region']))
            channels_df['atlas_region'] = channels_df['name'].map(atlas_map)
    else:
        channels_df['atlas_region'] = None

    # Load events
    events_file = subject_dir / f'{subject_id}_ses-{session}_task-{task}_events.tsv'
    if events_file.exists():
        events_df = pd.read_csv(events_file, sep='\t')
    else:
        events_df = pd.DataFrame()

    return {
        'subject_id': subject_id,
        'raw': raw,
        'channels': channels_df,
        'events': events_df,
        'sampling_rate': raw.info['sfreq'],
        'n_samples': raw.n_times,
        'duration_s': raw.n_times / raw.info['sfreq'],
    }


def get_channels_by_region(
    channels_df: pd.DataFrame,
    target_regions: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Group channels by brain region.

    Args:
        channels_df: Channels DataFrame with 'name' and 'atlas_region' columns
        target_regions: List of regions to extract (None = all found regions)

    Returns:
        Dict mapping region name -> list of channel names
    """
    region_channels = defaultdict(list)

    for _, row in channels_df.iterrows():
        ch_name = row['name']
        atlas_region = row.get('atlas_region', '')

        # Skip bad channels
        if row.get('status', 'good') == 'bad':
            continue

        # Map to simplified region
        region = map_channel_to_region(ch_name, atlas_region)

        if region is not None:
            if target_regions is None or region in target_regions:
                region_channels[region].append(ch_name)

    return dict(region_channels)


def parse_trial_type(trial_type_str: str) -> Dict[str, str]:
    """Parse COGITATE trial_type string into components.

    Format: "stimulus onset/block_1/miniblock_1/face/face_02/Center/1000ms/Relevant target/Hit"

    Returns dict with: event_type, block, miniblock, category, stimulus, position, duration, relevance, response
    """
    parts = trial_type_str.split('/')

    result = {
        'event_type': parts[0] if len(parts) > 0 else '',
        'block': parts[1] if len(parts) > 1 else '',
        'miniblock': parts[2] if len(parts) > 2 else '',
        'category': parts[3] if len(parts) > 3 else '',  # face, object, letter, false
        'stimulus': parts[4] if len(parts) > 4 else '',
        'position': parts[5] if len(parts) > 5 else '',
        'duration': parts[6] if len(parts) > 6 else '',
        'relevance': parts[7] if len(parts) > 7 else '',
        'response': parts[8] if len(parts) > 8 else '',
    }

    return result


# =============================================================================
# Preprocessing Pipeline
# =============================================================================

def preprocess_subject(
    bids_root: Path,
    subject_id: str,
    output_dir: Path,
    target_sfreq: Optional[float] = None,
    l_freq: Optional[float] = 0.5,
    h_freq: Optional[float] = 200.0,
    notch_freqs: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Preprocess a single subject and save to output directory.

    Args:
        bids_root: Path to BIDS root
        subject_id: Subject ID
        output_dir: Output directory for processed data
        target_sfreq: Target sampling frequency (Hz), None to keep original
        l_freq: High-pass filter frequency (Hz), None to skip
        h_freq: Low-pass filter frequency (Hz), None to skip
        notch_freqs: List of notch filter frequencies (e.g., [60, 120])

    Returns:
        Dictionary with processing summary
    """
    # Load subject data
    data = load_bids_subject(bids_root, subject_id)
    raw = data['raw']
    original_sfreq = raw.info['sfreq']

    print(f"  Original: {original_sfreq} Hz, {raw.n_times} samples, {len(raw.ch_names)} channels")

    # Get good SEEG channels only
    channels_df = data['channels']
    good_channels = channels_df[
        (channels_df['status'] == 'good') &
        (channels_df['type'].isin(['SEEG', 'ECoG', 'ECOG']))
    ]['name'].tolist()

    # Filter to channels that exist in raw
    good_channels = [ch for ch in good_channels if ch in raw.ch_names]

    if len(good_channels) == 0:
        print(f"  WARNING: No good SEEG channels found for {subject_id}")
        return None

    # Pick only good SEEG channels
    raw.pick_channels(good_channels)
    print(f"  After channel selection: {len(raw.ch_names)} good SEEG channels")

    # Apply filters (only if specified)
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        print(f"  Filtered: {l_freq or 0}-{h_freq or 'inf'} Hz")
    else:
        print(f"  No filtering applied")

    # Apply notch filter for line noise (only if specified)
    if notch_freqs:
        raw.notch_filter(notch_freqs, verbose=False)
        print(f"  Notch filtered: {notch_freqs} Hz")

    # Resample only if target_sfreq is specified AND different from original
    if target_sfreq is not None and raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
        print(f"  Resampled: {target_sfreq} Hz, {raw.n_times} samples")
    else:
        target_sfreq = original_sfreq
        print(f"  No resampling (keeping {original_sfreq} Hz)")

    # Get data as numpy array (channels x time)
    signal_data = raw.get_data()  # Shape: (n_channels, n_samples)

    # Update channels DataFrame for good channels only
    channels_df = channels_df[channels_df['name'].isin(raw.ch_names)].copy()
    channels_df = channels_df.reset_index(drop=True)

    # Map channels to regions
    region_channels = get_channels_by_region(channels_df)
    channels_df['region'] = channels_df['name'].apply(
        lambda x: map_channel_to_region(x, channels_df[channels_df['name'] == x]['atlas_region'].values[0]
                                        if len(channels_df[channels_df['name'] == x]) > 0 else '')
    )

    # Create output directory for this subject
    subject_output_dir = output_dir / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    # Save continuous data
    np.save(subject_output_dir / 'continuous.npy', signal_data.astype(np.float32))

    # Save channel info
    channels_df.to_csv(subject_output_dir / 'channels.csv', index=False)

    # Process and save events
    events_df = data['events']
    if len(events_df) > 0:
        # Parse trial types
        if 'trial_type' in events_df.columns:
            parsed = events_df['trial_type'].apply(parse_trial_type)
            for key in ['event_type', 'category', 'stimulus', 'position', 'duration', 'relevance', 'response']:
                events_df[key] = parsed.apply(lambda x: x.get(key, ''))

        # Convert onset to samples
        events_df['sample'] = (events_df['onset'] * target_sfreq).astype(int)

        # Save events
        events_df.to_csv(subject_output_dir / 'events.csv', index=False)

    # Save region mapping
    region_mapping = {}
    for region, ch_list in region_channels.items():
        # Get indices of these channels in the data
        ch_indices = [list(raw.ch_names).index(ch) for ch in ch_list if ch in raw.ch_names]
        region_mapping[region] = {
            'channels': ch_list,
            'indices': ch_indices,
            'n_channels': len(ch_indices),
        }

    with open(subject_output_dir / 'region_mapping.json', 'w') as f:
        json.dump(region_mapping, f, indent=2)

    summary = {
        'subject_id': subject_id,
        'n_channels': len(raw.ch_names),
        'n_samples': raw.n_times,
        'duration_s': raw.n_times / target_sfreq,
        'sampling_rate': target_sfreq,
        'regions': {r: len(v['indices']) for r, v in region_mapping.items()},
        'n_events': len(events_df) if len(events_df) > 0 else 0,
    }

    print(f"  Saved: {signal_data.shape[0]} channels x {signal_data.shape[1]} samples")
    print(f"  Regions found: {summary['regions']}")

    return summary


def preprocess_all_subjects(
    bids_root: Path,
    output_dir: Path,
    target_sfreq: float = 1000.0,
    l_freq: Optional[float] = 0.5,
    h_freq: Optional[float] = 200.0,
    notch_freqs: Optional[List[float]] = None,
    subjects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Preprocess all subjects in the BIDS dataset.

    Args:
        bids_root: Path to BIDS root
        output_dir: Output directory
        target_sfreq: Target sampling frequency
        l_freq: High-pass filter frequency
        h_freq: Low-pass filter frequency
        notch_freqs: Notch filter frequencies
        subjects: Optional list of subjects to process (None = all)

    Returns:
        Dataset summary
    """
    bids_root = Path(bids_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all subjects
    if subjects is None:
        subject_dirs = sorted([d for d in bids_root.iterdir()
                               if d.is_dir() and d.name.startswith('sub-')])
        subjects = [d.name for d in subject_dirs]
    else:
        subjects = [s if s.startswith('sub-') else f'sub-{s}' for s in subjects]

    print(f"Processing {len(subjects)} subjects from {bids_root}")
    print(f"Output directory: {output_dir}")
    print(f"Target sampling rate: {target_sfreq} Hz")
    print(f"Filters: {l_freq}-{h_freq} Hz, notch={notch_freqs}")
    print("=" * 60)

    # Process each subject
    summaries = []
    failed_subjects = []

    for i, subject_id in enumerate(subjects):
        print(f"\n[{i+1}/{len(subjects)}] Processing {subject_id}...")

        try:
            summary = preprocess_subject(
                bids_root=bids_root,
                subject_id=subject_id,
                output_dir=output_dir,
                target_sfreq=target_sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                notch_freqs=notch_freqs,
            )
            if summary is not None:
                summaries.append(summary)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed_subjects.append((subject_id, str(e)))

    print("\n" + "=" * 60)
    print(f"Processed {len(summaries)}/{len(subjects)} subjects successfully")

    if failed_subjects:
        print(f"Failed subjects ({len(failed_subjects)}):")
        for subj, error in failed_subjects:
            print(f"  {subj}: {error}")

    # Create metadata CSV
    if summaries:
        metadata_df = pd.DataFrame(summaries)
        metadata_df.to_csv(output_dir / 'metadata.csv', index=False)

        # Aggregate region statistics
        all_regions = set()
        for s in summaries:
            all_regions.update(s['regions'].keys())

        region_stats = {}
        for region in sorted(all_regions):
            counts = [s['regions'].get(region, 0) for s in summaries]
            region_stats[region] = {
                'n_subjects_with_region': sum(1 for c in counts if c > 0),
                'min_channels': min(c for c in counts if c > 0) if any(c > 0 for c in counts) else 0,
                'max_channels': max(counts),
                'mean_channels': np.mean([c for c in counts if c > 0]) if any(c > 0 for c in counts) else 0,
            }

        # Save dataset info
        dataset_info = {
            'bids_source': str(bids_root),
            'n_subjects': len(summaries),
            'sampling_rate': target_sfreq,
            'filter_l_freq': l_freq,
            'filter_h_freq': h_freq,
            'notch_freqs': notch_freqs,
            'total_duration_hours': sum(s['duration_s'] for s in summaries) / 3600,
            'region_stats': region_stats,
            'subjects': [s['subject_id'] for s in summaries],
        }

        with open(output_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)

        # Print summary
        print("\nDataset Summary:")
        print(f"  Total subjects: {len(summaries)}")
        print(f"  Total duration: {dataset_info['total_duration_hours']:.1f} hours")
        print(f"\nRegion coverage:")
        for region, stats in sorted(region_stats.items()):
            print(f"  {region}: {stats['n_subjects_with_region']} subjects, "
                  f"{stats['min_channels']}-{stats['max_channels']} channels "
                  f"(mean: {stats['mean_channels']:.1f})")

    return {
        'n_processed': len(summaries),
        'n_failed': len(failed_subjects),
        'summaries': summaries,
        'failed': failed_subjects,
    }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess COGITATE BIDS iEEG data to unified NPY format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "bids_root",
        type=str,
        help="Path to BIDS root directory (e.g., /data/COG_ECOG_EXP1_BIDS)",
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="/data/COGITATEDataset",
        help="Output directory for processed data",
    )

    parser.add_argument(
        "--target-sfreq",
        type=float,
        default=None,
        help="Target sampling frequency in Hz (default: keep original)",
    )

    parser.add_argument(
        "--l-freq",
        type=float,
        default=0.5,
        help="High-pass filter cutoff (Hz). Set to 0 to disable.",
    )

    parser.add_argument(
        "--h-freq",
        type=float,
        default=200.0,
        help="Low-pass filter cutoff (Hz). Set to 0 to disable.",
    )

    parser.add_argument(
        "--notch",
        type=float,
        nargs='+',
        default=[60.0, 120.0],
        help="Notch filter frequencies (Hz) for line noise removal",
    )

    parser.add_argument(
        "--subjects",
        type=str,
        nargs='+',
        default=None,
        help="Specific subjects to process (default: all)",
    )

    parser.add_argument(
        "--raw",
        action="store_true",
        help="Keep completely raw data (no filtering, no resampling)",
    )

    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip all filtering (use raw data)",
    )

    args = parser.parse_args()

    # Handle --raw flag (no filtering, no resampling)
    if args.raw:
        l_freq = None
        h_freq = None
        notch_freqs = None
        target_sfreq = None  # Keep original
        print("RAW MODE: No filtering, no resampling - keeping original data")
    else:
        # Handle filter arguments
        l_freq = None if args.no_filter or args.l_freq == 0 else args.l_freq
        h_freq = None if args.no_filter or args.h_freq == 0 else args.h_freq
        notch_freqs = None if args.no_filter else args.notch
        target_sfreq = args.target_sfreq

    # Run preprocessing
    result = preprocess_all_subjects(
        bids_root=Path(args.bids_root),
        output_dir=Path(args.output_dir),
        target_sfreq=target_sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        notch_freqs=notch_freqs,
        subjects=args.subjects,
    )

    # Exit with error code if any subjects failed
    if result['n_failed'] > 0:
        sys.exit(1)

    print(f"\nDone! Processed data saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
