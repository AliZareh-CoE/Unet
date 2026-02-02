#!/usr/bin/env python3
"""
Build COGITATE Dataset for Neural Translation Training
=======================================================

Creates training datasets from preprocessed COGITATE data with support for:
1. Many-to-many: Multiple source regions → multiple target regions
2. Many-to-one: Multiple source regions → single target region
3. Within-region: Channel interpolation within a single region

The output format is compatible with existing Unet DataLoaders.

Usage Examples:
    # Region-to-region translation (temporal → frontal)
    python scripts/build_cogitate_dataset.py /data/COGITATEDataset_1024hz \
        --source temporal --target frontal \
        --source-channels 23 --target-channels 17 \
        --output /data/COGITATE_temp_front

    # Many-to-one (temporal+parietal → hippocampus)
    python scripts/build_cogitate_dataset.py /data/COGITATEDataset_1024hz \
        --source temporal parietal --target hippocampus \
        --source-channels 30 --target-channels 7 \
        --output /data/COGITATE_multi_hipp

    # Within-region interpolation (temporal channels)
    python scripts/build_cogitate_dataset.py /data/COGITATEDataset_1024hz \
        --source temporal --target temporal \
        --mode within-region \
        --source-channels 20 --target-channels 10 \
        --output /data/COGITATE_temp_interp

Output Structure:
    /output/
    ├── sub-CE103/
    │   ├── source.npy          # (n_source_channels, n_samples)
    │   ├── target.npy          # (n_target_channels, n_samples)
    │   └── metadata.json       # Channel mappings and info
    ├── sub-CE106/
    │   └── ...
    └── dataset_config.json     # Full configuration
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_subject_data(subject_dir: Path) -> Dict[str, Any]:
    """Load preprocessed subject data."""
    data = {
        'continuous': np.load(subject_dir / 'continuous.npy'),
        'channels': pd.read_csv(subject_dir / 'channels.csv'),
    }

    with open(subject_dir / 'region_mapping.json', 'r') as f:
        data['region_mapping'] = json.load(f)

    if (subject_dir / 'events.csv').exists():
        data['events'] = pd.read_csv(subject_dir / 'events.csv')

    return data


def get_region_indices(
    region_mapping: Dict,
    regions: List[str],
) -> Tuple[List[int], Dict[str, List[int]]]:
    """Get channel indices for specified regions.

    Returns:
        all_indices: Combined list of all channel indices
        region_indices: Dict mapping each region to its indices
    """
    all_indices = []
    region_indices = {}

    for region in regions:
        if region in region_mapping:
            indices = region_mapping[region]['indices']
            region_indices[region] = indices
            all_indices.extend(indices)

    return all_indices, region_indices


def select_channels(
    data: np.ndarray,
    available_indices: List[int],
    n_channels: int,
    strategy: str = 'first',
    exclude_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Select specific number of channels from available indices.

    Args:
        data: Full continuous data (n_channels, n_samples)
        available_indices: Channel indices available for selection
        n_channels: Number of channels to select
        strategy: 'first', 'last', 'random', 'even' (evenly spaced)
        exclude_indices: Indices to exclude (for within-region mode)

    Returns:
        selected_data: (n_channels, n_samples)
        selected_indices: Which indices were selected
    """
    if exclude_indices:
        available_indices = [i for i in available_indices if i not in exclude_indices]

    if len(available_indices) < n_channels:
        raise ValueError(
            f"Not enough channels: need {n_channels}, have {len(available_indices)}"
        )

    if strategy == 'first':
        selected = available_indices[:n_channels]
    elif strategy == 'last':
        selected = available_indices[-n_channels:]
    elif strategy == 'random':
        selected = list(np.random.choice(available_indices, n_channels, replace=False))
    elif strategy == 'even':
        # Evenly spaced selection
        step = len(available_indices) / n_channels
        selected = [available_indices[int(i * step)] for i in range(n_channels)]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return data[selected, :], selected


def analyze_subjects(
    input_dir: Path,
    source_regions: List[str],
    target_regions: List[str],
    source_channels: int,
    target_channels: int,
    mode: str = 'region-to-region',
) -> List[Dict]:
    """Analyze which subjects have enough channels for the configuration.

    Returns:
        List of viable subjects with their channel info
    """
    viable_subjects = []

    for subject_dir in sorted(input_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith('sub-'):
            continue

        try:
            with open(subject_dir / 'region_mapping.json', 'r') as f:
                region_mapping = json.load(f)
        except FileNotFoundError:
            continue

        # Count available channels
        source_available = sum(
            region_mapping.get(r, {}).get('n_channels', 0)
            for r in source_regions
        )
        target_available = sum(
            region_mapping.get(r, {}).get('n_channels', 0)
            for r in target_regions
        )

        # For within-region mode, we need source + target channels from same regions
        if mode == 'within-region':
            total_needed = source_channels + target_channels
            if source_available >= total_needed:
                viable_subjects.append({
                    'subject_id': subject_dir.name,
                    'source_available': source_available,
                    'target_available': source_available,  # Same region
                    'region_breakdown': {
                        r: region_mapping.get(r, {}).get('n_channels', 0)
                        for r in source_regions
                    }
                })
        else:
            # Standard region-to-region
            if source_available >= source_channels and target_available >= target_channels:
                viable_subjects.append({
                    'subject_id': subject_dir.name,
                    'source_available': source_available,
                    'target_available': target_available,
                    'source_breakdown': {
                        r: region_mapping.get(r, {}).get('n_channels', 0)
                        for r in source_regions
                    },
                    'target_breakdown': {
                        r: region_mapping.get(r, {}).get('n_channels', 0)
                        for r in target_regions
                    }
                })

    return viable_subjects


def build_subject_dataset(
    subject_dir: Path,
    output_dir: Path,
    source_regions: List[str],
    target_regions: List[str],
    source_channels: int,
    target_channels: int,
    mode: str = 'region-to-region',
    channel_strategy: str = 'first',
) -> Dict[str, Any]:
    """Build source/target arrays for a single subject.

    Args:
        subject_dir: Path to preprocessed subject data
        output_dir: Where to save the built dataset
        source_regions: List of source region names
        target_regions: List of target region names
        source_channels: Number of source channels to select
        target_channels: Number of target channels to select
        mode: 'region-to-region' or 'within-region'
        channel_strategy: How to select channels ('first', 'even', 'random')

    Returns:
        Metadata dict for this subject
    """
    data = load_subject_data(subject_dir)
    continuous = data['continuous']
    region_mapping = data['region_mapping']

    # Get source channel indices
    source_indices, source_by_region = get_region_indices(region_mapping, source_regions)

    if mode == 'within-region':
        # Source and target come from same pool, but different channels
        all_indices = source_indices.copy()

        # Select source channels
        source_data, selected_source = select_channels(
            continuous, all_indices, source_channels, channel_strategy
        )

        # Select target channels from remaining
        target_data, selected_target = select_channels(
            continuous, all_indices, target_channels, channel_strategy,
            exclude_indices=selected_source
        )

        target_by_region = source_by_region  # Same regions

    else:
        # Standard region-to-region translation
        target_indices, target_by_region = get_region_indices(region_mapping, target_regions)

        # Select channels
        source_data, selected_source = select_channels(
            continuous, source_indices, source_channels, channel_strategy
        )
        target_data, selected_target = select_channels(
            continuous, target_indices, target_channels, channel_strategy
        )

    # Create output directory
    subject_output = output_dir / subject_dir.name
    subject_output.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.save(subject_output / 'source.npy', source_data.astype(np.float32))
    np.save(subject_output / 'target.npy', target_data.astype(np.float32))

    # Build metadata
    channels_df = data['channels']

    def get_channel_names(indices):
        names = []
        for idx in indices:
            if idx < len(channels_df):
                names.append(channels_df.iloc[idx]['name'])
            else:
                names.append(f'ch_{idx}')
        return names

    metadata = {
        'subject_id': subject_dir.name,
        'source_shape': list(source_data.shape),
        'target_shape': list(target_data.shape),
        'source_regions': source_regions,
        'target_regions': target_regions,
        'source_channels_selected': selected_source,
        'target_channels_selected': selected_target,
        'source_channel_names': get_channel_names(selected_source),
        'target_channel_names': get_channel_names(selected_target),
        'mode': mode,
        'n_samples': continuous.shape[1],
    }

    with open(subject_output / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Copy events if present
    if 'events' in data:
        data['events'].to_csv(subject_output / 'events.csv', index=False)

    return metadata


def build_dataset(
    input_dir: Path,
    output_dir: Path,
    source_regions: List[str],
    target_regions: List[str],
    source_channels: int,
    target_channels: int,
    mode: str = 'region-to-region',
    channel_strategy: str = 'first',
    subjects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build complete dataset from preprocessed COGITATE data.

    Args:
        input_dir: Path to preprocessed data (e.g., /data/COGITATEDataset_1024hz)
        output_dir: Where to save built dataset
        source_regions: List of source region names
        target_regions: List of target region names
        source_channels: Number of source channels
        target_channels: Number of target channels
        mode: 'region-to-region' or 'within-region'
        channel_strategy: 'first', 'even', or 'random'
        subjects: Optional list of subjects to include

    Returns:
        Dataset summary
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset info
    with open(input_dir / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)

    sampling_rate = dataset_info['sampling_rate']

    # Analyze viable subjects
    print(f"Analyzing subjects for configuration:")
    print(f"  Source regions: {source_regions}")
    print(f"  Target regions: {target_regions}")
    print(f"  Source channels: {source_channels}")
    print(f"  Target channels: {target_channels}")
    print(f"  Mode: {mode}")
    print()

    viable = analyze_subjects(
        input_dir, source_regions, target_regions,
        source_channels, target_channels, mode
    )

    if subjects:
        # Filter to specified subjects
        subjects_set = set(subjects)
        viable = [v for v in viable if v['subject_id'] in subjects_set]

    print(f"Found {len(viable)} viable subjects")

    if len(viable) == 0:
        print("ERROR: No subjects meet the channel requirements!")
        print("\nTry reducing channel counts or using different regions.")
        return {'n_subjects': 0, 'error': 'No viable subjects'}

    # Build dataset for each subject
    print(f"\nBuilding dataset...")
    summaries = []

    for i, subj_info in enumerate(viable):
        subject_id = subj_info['subject_id']
        subject_dir = input_dir / subject_id

        print(f"  [{i+1}/{len(viable)}] {subject_id}...")

        try:
            metadata = build_subject_dataset(
                subject_dir=subject_dir,
                output_dir=output_dir,
                source_regions=source_regions,
                target_regions=target_regions,
                source_channels=source_channels,
                target_channels=target_channels,
                mode=mode,
                channel_strategy=channel_strategy,
            )
            summaries.append(metadata)
        except Exception as e:
            print(f"    ERROR: {e}")

    # Save dataset configuration
    config = {
        'source_dir': str(input_dir),
        'source_regions': source_regions,
        'target_regions': target_regions,
        'source_channels': source_channels,
        'target_channels': target_channels,
        'mode': mode,
        'channel_strategy': channel_strategy,
        'sampling_rate': sampling_rate,
        'n_subjects': len(summaries),
        'subjects': [s['subject_id'] for s in summaries],
        'total_samples': sum(s['n_samples'] for s in summaries),
        'total_duration_hours': sum(s['n_samples'] for s in summaries) / sampling_rate / 3600,
    }

    with open(output_dir / 'dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset built successfully!")
    print(f"  Output: {output_dir}")
    print(f"  Subjects: {len(summaries)}")
    print(f"  Source: {source_channels} channels from {source_regions}")
    print(f"  Target: {target_channels} channels from {target_regions}")
    print(f"  Total duration: {config['total_duration_hours']:.1f} hours")
    print(f"  Sampling rate: {sampling_rate} Hz")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Build COGITATE dataset for neural translation training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Temporal → Frontal translation
  %(prog)s /data/COGITATEDataset_1024hz --source temporal --target frontal \\
      --source-channels 23 --target-channels 17 --output /data/COGITATE_temp_front

  # Multi-region → Hippocampus
  %(prog)s /data/COGITATEDataset_1024hz --source temporal parietal --target hippocampus \\
      --source-channels 30 --target-channels 7 --output /data/COGITATE_multi_hipp

  # Within-region channel interpolation
  %(prog)s /data/COGITATEDataset_1024hz --source temporal --target temporal \\
      --mode within-region --source-channels 20 --target-channels 10 \\
      --output /data/COGITATE_temp_interp
        """
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to preprocessed COGITATE data",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for built dataset",
    )

    parser.add_argument(
        "--source",
        type=str,
        nargs='+',
        required=True,
        help="Source region(s): temporal, frontal, hippocampus, amygdala, parietal, etc.",
    )

    parser.add_argument(
        "--target",
        type=str,
        nargs='+',
        required=True,
        help="Target region(s)",
    )

    parser.add_argument(
        "--source-channels",
        type=int,
        required=True,
        help="Number of source channels to select",
    )

    parser.add_argument(
        "--target-channels",
        type=int,
        required=True,
        help="Number of target channels to select",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=['region-to-region', 'within-region'],
        default='region-to-region',
        help="Translation mode (default: region-to-region)",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=['first', 'last', 'even', 'random'],
        default='first',
        help="Channel selection strategy (default: first)",
    )

    parser.add_argument(
        "--subjects",
        type=str,
        nargs='+',
        help="Specific subjects to include (default: all viable)",
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze viable subjects, don't build dataset",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    if args.analyze_only:
        # Just show analysis
        viable = analyze_subjects(
            input_dir,
            args.source,
            args.target,
            args.source_channels,
            args.target_channels,
            args.mode,
        )

        print(f"\nConfiguration: {args.source} → {args.target}")
        print(f"Required channels: {args.source_channels} → {args.target_channels}")
        print(f"Mode: {args.mode}")
        print(f"\nViable subjects: {len(viable)}")

        for subj in viable[:10]:
            print(f"  {subj['subject_id']}: source={subj['source_available']}, target={subj['target_available']}")

        if len(viable) > 10:
            print(f"  ... and {len(viable) - 10} more")

        return 0

    # Build the dataset
    result = build_dataset(
        input_dir=input_dir,
        output_dir=Path(args.output),
        source_regions=args.source,
        target_regions=args.target,
        source_channels=args.source_channels,
        target_channels=args.target_channels,
        mode=args.mode,
        channel_strategy=args.strategy,
        subjects=args.subjects,
    )

    if result.get('error'):
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
