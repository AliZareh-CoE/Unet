#!/usr/bin/env python3
"""
Create COGITATE Dataset Configurations for Neural Translation
==============================================================

Analyzes the preprocessed data and creates 3 uniform configurations:
1. temporal → frontal
2. temporal → hippocampus
3. hippocampus → amygdala

For each configuration:
- Finds subjects with BOTH regions
- Determines uniform channel counts (minimum across subjects)
- Filters by sampling rate threshold
- Outputs qualifying subjects and exact preprocessing command

Usage:
    python scripts/create_cogitate_configs.py /data/COGITATEDataset
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Configuration pairs to analyze
TRANSLATION_PAIRS = [
    ('temporal', 'frontal', 'Perception → Executive'),
    ('temporal', 'hippocampus', 'Perception → Memory'),
    ('hippocampus', 'amygdala', 'Memory → Emotion'),
]

# Minimum channels per region to be considered viable
MIN_CHANNELS_THRESHOLD = 2


def load_subject_data(dataset_dir: Path) -> List[Dict]:
    """Load all subject data from preprocessed dataset."""
    subjects = []

    for subject_dir in sorted(dataset_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith('sub-'):
            continue

        subject_id = subject_dir.name

        # Load region mapping
        region_file = subject_dir / 'region_mapping.json'
        if not region_file.exists():
            continue

        with open(region_file) as f:
            region_data = json.load(f)

        # Get channel counts per region
        regions = {}
        for region, data in region_data.items():
            regions[region] = data['n_channels']

        # Get sampling rate from channels.csv
        channels_file = subject_dir / 'channels.csv'
        sampling_rate = None
        if channels_file.exists():
            df = pd.read_csv(channels_file)
            if 'sampling_frequency' in df.columns:
                sampling_rate = df['sampling_frequency'].iloc[0]

        # Get data shape
        continuous_file = subject_dir / 'continuous.npy'
        n_samples = 0
        if continuous_file.exists():
            data = np.load(continuous_file, mmap_mode='r')
            n_samples = data.shape[1]

        subjects.append({
            'subject_id': subject_id,
            'regions': regions,
            'sampling_rate': sampling_rate,
            'n_samples': n_samples,
        })

    return subjects


def analyze_configuration(
    subjects: List[Dict],
    source_region: str,
    target_region: str,
    min_rate: float = 512.0,
    min_channels: int = MIN_CHANNELS_THRESHOLD,
) -> Dict:
    """Analyze a single source→target configuration."""

    # Filter subjects that have BOTH regions with minimum channels
    qualifying = []

    for s in subjects:
        # Check sampling rate
        rate = s.get('sampling_rate')
        if rate is None or rate < min_rate:
            continue

        # Check both regions exist with minimum channels
        src_ch = s['regions'].get(source_region, 0)
        tgt_ch = s['regions'].get(target_region, 0)

        if src_ch >= min_channels and tgt_ch >= min_channels:
            qualifying.append({
                'subject_id': s['subject_id'],
                'source_channels': src_ch,
                'target_channels': tgt_ch,
                'sampling_rate': rate,
                'n_samples': s['n_samples'],
            })

    if not qualifying:
        return {
            'source': source_region,
            'target': target_region,
            'n_subjects': 0,
            'subjects': [],
            'uniform_source_ch': 0,
            'uniform_target_ch': 0,
            'viable': False,
        }

    # Find minimum channels across all qualifying subjects
    min_source = min(s['source_channels'] for s in qualifying)
    min_target = min(s['target_channels'] for s in qualifying)

    # Get unique sampling rates
    rates = set(s['sampling_rate'] for s in qualifying)

    return {
        'source': source_region,
        'target': target_region,
        'n_subjects': len(qualifying),
        'subjects': qualifying,
        'uniform_source_ch': min_source,
        'uniform_target_ch': min_target,
        'sampling_rates': sorted(rates),
        'viable': len(qualifying) >= 10,  # Need at least 10 for LOSO
        'total_duration_hours': sum(s['n_samples'] / s['sampling_rate'] / 3600 for s in qualifying),
    }


def print_configuration_analysis(config: Dict, rationale: str):
    """Print detailed analysis of a configuration."""
    print(f"\n{'='*70}")
    print(f"  {config['source'].upper()} → {config['target'].upper()}")
    print(f"  ({rationale})")
    print(f"{'='*70}")

    if not config['viable']:
        print(f"  ❌ NOT VIABLE - Only {config['n_subjects']} subjects qualify")
        if config['n_subjects'] > 0:
            print(f"  Subjects: {[s['subject_id'] for s in config['subjects']]}")
        return

    print(f"  ✅ VIABLE CONFIGURATION")
    print(f"  ")
    print(f"  Qualifying subjects: {config['n_subjects']}")
    print(f"  Total duration: {config['total_duration_hours']:.1f} hours")
    print(f"  Sampling rates present: {config['sampling_rates']}")
    print(f"  ")
    print(f"  UNIFORM DIMENSIONS:")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  Source ({config['source']}): {config['uniform_source_ch']} channels")
    print(f"  │  Target ({config['target']}): {config['uniform_target_ch']} channels")
    print(f"  │  Translation: {config['uniform_source_ch']} → {config['uniform_target_ch']}")
    print(f"  └─────────────────────────────────────────┘")

    print(f"\n  Subject details:")
    print(f"  {'Subject':<15} {'Src Ch':>8} {'Tgt Ch':>8} {'Rate':>8} {'Duration':>10}")
    print(f"  {'-'*55}")

    for s in sorted(config['subjects'], key=lambda x: x['subject_id']):
        duration_min = s['n_samples'] / s['sampling_rate'] / 60
        print(f"  {s['subject_id']:<15} {s['source_channels']:>8} {s['target_channels']:>8} "
              f"{s['sampling_rate']:>8.0f} {duration_min:>8.1f} min")


def analyze_with_thresholds(
    subjects: List[Dict],
    source_region: str,
    target_region: str,
) -> None:
    """Show how subject count changes with different thresholds."""
    print(f"\n  Threshold analysis for {source_region} → {target_region}:")
    print(f"  {'Min Ch':>8} {'Min Rate':>10} {'Subjects':>10} {'Src Ch':>8} {'Tgt Ch':>8}")
    print(f"  {'-'*50}")

    for min_ch in [1, 2, 3, 4, 5]:
        for min_rate in [512, 1024]:
            config = analyze_configuration(
                subjects, source_region, target_region,
                min_rate=min_rate, min_channels=min_ch
            )
            if config['n_subjects'] > 0:
                print(f"  {min_ch:>8} {min_rate:>10} {config['n_subjects']:>10} "
                      f"{config['uniform_source_ch']:>8} {config['uniform_target_ch']:>8}")


def generate_preprocessing_command(
    config: Dict,
    output_suffix: str,
    target_sfreq: int = 1024,
) -> str:
    """Generate the preprocessing command for a configuration."""
    subjects = ' '.join(s['subject_id'] for s in config['subjects'])

    cmd = f"""
# Configuration: {config['source']} → {config['target']}
# Subjects: {config['n_subjects']}
# Dimensions: {config['uniform_source_ch']} → {config['uniform_target_ch']}

python scripts/preprocess_cogitate_bids.py /data/COG_ECOG_EXP1_BIDS \\
    --output-dir /data/COGITATEDataset_{output_suffix} \\
    --target-sfreq {target_sfreq} \\
    --no-filter \\
    --subjects {subjects}
"""
    return cmd


def save_configuration(config: Dict, output_path: Path):
    """Save configuration to JSON file."""
    # Convert to serializable format
    config_out = {
        'source_region': config['source'],
        'target_region': config['target'],
        'n_subjects': config['n_subjects'],
        'uniform_source_channels': config['uniform_source_ch'],
        'uniform_target_channels': config['uniform_target_ch'],
        'subjects': [s['subject_id'] for s in config['subjects']],
        'subject_details': config['subjects'],
    }

    with open(output_path, 'w') as f:
        json.dump(config_out, f, indent=2)

    print(f"  Saved configuration to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create COGITATE dataset configurations for neural translation"
    )
    parser.add_argument("dataset_dir", type=str, help="Path to preprocessed dataset")
    parser.add_argument("--min-channels", type=int, default=2,
                        help="Minimum channels per region (default: 2)")
    parser.add_argument("--min-rate", type=float, default=512,
                        help="Minimum sampling rate in Hz (default: 512)")
    parser.add_argument("--save-configs", action="store_true",
                        help="Save configuration JSONs")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    print("=" * 70)
    print("COGITATE DATASET CONFIGURATION ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {dataset_dir}")
    print(f"Min channels per region: {args.min_channels}")
    print(f"Min sampling rate: {args.min_rate} Hz")

    # Load all subject data
    print("\nLoading subject data...")
    subjects = load_subject_data(dataset_dir)
    print(f"Found {len(subjects)} subjects")

    # Analyze each configuration
    configs = []
    for source, target, rationale in TRANSLATION_PAIRS:
        config = analyze_configuration(
            subjects, source, target,
            min_rate=args.min_rate,
            min_channels=args.min_channels
        )
        config['rationale'] = rationale
        configs.append(config)

        print_configuration_analysis(config, rationale)

        # Show threshold analysis
        if config['n_subjects'] < 20:
            analyze_with_thresholds(subjects, source, target)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    viable_configs = [c for c in configs if c['viable']]

    print(f"\nViable configurations: {len(viable_configs)}/3")

    for c in configs:
        status = "✅" if c['viable'] else "❌"
        print(f"  {status} {c['source']} → {c['target']}: "
              f"{c['n_subjects']} subjects, {c['uniform_source_ch']}→{c['uniform_target_ch']} channels")

    # Generate commands
    if viable_configs:
        print("\n" + "=" * 70)
        print("PREPROCESSING COMMANDS")
        print("=" * 70)

        for c in viable_configs:
            suffix = f"{c['source'][:4]}_{c['target'][:4]}"
            cmd = generate_preprocessing_command(c, suffix)
            print(cmd)

            if args.save_configs:
                config_path = dataset_dir / f"config_{suffix}.json"
                save_configuration(c, config_path)

    # Final recommendation
    if viable_configs:
        best = max(viable_configs, key=lambda x: x['n_subjects'])
        print("\n" + "=" * 70)
        print("RECOMMENDED CONFIGURATION")
        print("=" * 70)
        print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  BEST OPTION: {best['source']} → {best['target']:<38} │
│                                                                     │
│  Subjects:     {best['n_subjects']:<50} │
│  Source:       {best['uniform_source_ch']} {best['source']} channels{' '*(36-len(best['source']))} │
│  Target:       {best['uniform_target_ch']} {best['target']} channels{' '*(36-len(best['target']))} │
│  Duration:     {best['total_duration_hours']:.1f} hours{' '*43} │
│  Rationale:    {best['rationale']:<50} │
└─────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
