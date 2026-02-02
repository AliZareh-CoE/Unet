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
    channel_strategy: str = 'min',  # 'min', 'median', 'percentile_25'
) -> Dict:
    """Analyze a single source→target configuration.

    Args:
        subjects: List of subject data dicts
        source_region: Source region name
        target_region: Target region name
        min_rate: Minimum sampling rate to include
        min_channels: Minimum channels per region to include subject
        channel_strategy: How to determine uniform channel count:
            - 'min': Use minimum (include all subjects, fewer channels)
            - 'median': Use median (include ~50% subjects, more channels)
            - 'percentile_25': Use 25th percentile (include ~75% subjects)
    """

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

    # Get channel counts
    src_channels = [s['source_channels'] for s in qualifying]
    tgt_channels = [s['target_channels'] for s in qualifying]

    # Compute statistics
    stats = {
        'min_source': min(src_channels),
        'min_target': min(tgt_channels),
        'max_source': max(src_channels),
        'max_target': max(tgt_channels),
        'median_source': int(np.median(src_channels)),
        'median_target': int(np.median(tgt_channels)),
        'p25_source': int(np.percentile(src_channels, 25)),
        'p25_target': int(np.percentile(tgt_channels, 25)),
    }

    # Determine uniform channel count based on strategy
    if channel_strategy == 'min':
        uniform_src = stats['min_source']
        uniform_tgt = stats['min_target']
        final_subjects = qualifying  # All qualify
    elif channel_strategy == 'median':
        uniform_src = stats['median_source']
        uniform_tgt = stats['median_target']
        # Filter to subjects meeting median threshold
        final_subjects = [s for s in qualifying
                         if s['source_channels'] >= uniform_src and s['target_channels'] >= uniform_tgt]
    elif channel_strategy == 'percentile_25':
        uniform_src = stats['p25_source']
        uniform_tgt = stats['p25_target']
        final_subjects = [s for s in qualifying
                         if s['source_channels'] >= uniform_src and s['target_channels'] >= uniform_tgt]
    else:
        raise ValueError(f"Unknown strategy: {channel_strategy}")

    # Get unique sampling rates
    rates = set(s['sampling_rate'] for s in final_subjects) if final_subjects else set()

    return {
        'source': source_region,
        'target': target_region,
        'n_subjects': len(final_subjects),
        'n_subjects_initial': len(qualifying),  # Before strategy filtering
        'subjects': final_subjects,
        'uniform_source_ch': uniform_src,
        'uniform_target_ch': uniform_tgt,
        'stats': stats,  # All statistics for reference
        'strategy': channel_strategy,
        'sampling_rates': sorted(rates) if rates else [],
        'viable': len(final_subjects) >= 10,  # Need at least 10 for LOSO
        'total_duration_hours': sum(s['n_samples'] / s['sampling_rate'] / 3600 for s in final_subjects) if final_subjects else 0,
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

    print(f"  ✅ VIABLE CONFIGURATION (strategy: {config.get('strategy', 'min')})")

    # Show channel statistics
    stats = config.get('stats', {})
    if stats:
        print(f"\n  Channel Statistics:")
        print(f"  {'':>20} {'Source':>12} {'Target':>12}")
        print(f"  {'-'*46}")
        print(f"  {'Min':>20} {stats.get('min_source', 0):>12} {stats.get('min_target', 0):>12}")
        print(f"  {'25th Percentile':>20} {stats.get('p25_source', 0):>12} {stats.get('p25_target', 0):>12}")
        print(f"  {'Median':>20} {stats.get('median_source', 0):>12} {stats.get('median_target', 0):>12}")
        print(f"  {'Max':>20} {stats.get('max_source', 0):>12} {stats.get('max_target', 0):>12}")
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
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["min", "median", "percentile_25", "all"],
                        help="Channel count strategy: min (all subjects), median (~50%%), percentile_25 (~75%%), all (compare all)")
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
    print(f"Strategy: {args.strategy}")

    # Load all subject data
    print("\nLoading subject data...")
    subjects = load_subject_data(dataset_dir)
    print(f"Found {len(subjects)} subjects")

    # Strategies to analyze
    if args.strategy == "all":
        strategies = ["min", "percentile_25", "median"]
    else:
        strategies = [args.strategy]

    # Analyze each configuration with each strategy
    all_configs = {}

    for strategy in strategies:
        print(f"\n\n{'#'*70}")
        print(f"# STRATEGY: {strategy.upper()}")
        print(f"# {'(Include all subjects, use minimum channels)' if strategy == 'min' else ''}")
        print(f"# {'(Include ~75% subjects, use 25th percentile channels)' if strategy == 'percentile_25' else ''}")
        print(f"# {'(Include ~50% subjects, use median channels)' if strategy == 'median' else ''}")
        print(f"{'#'*70}")

        configs = []
        for source, target, rationale in TRANSLATION_PAIRS:
            config = analyze_configuration(
                subjects, source, target,
                min_rate=args.min_rate,
                min_channels=args.min_channels,
                channel_strategy=strategy,
            )
            config['rationale'] = rationale
            configs.append(config)

            print_configuration_analysis(config, rationale)

        all_configs[strategy] = configs

    # Summary comparison table
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON TABLE")
    print("=" * 70)

    # Header
    print(f"\n{'Pair':<25} {'Strategy':<15} {'Subjects':>10} {'Src Ch':>8} {'Tgt Ch':>8} {'Viable'}")
    print("-" * 80)

    for source, target, rationale in TRANSLATION_PAIRS:
        pair_name = f"{source} → {target}"
        for strategy in strategies:
            for c in all_configs[strategy]:
                if c['source'] == source and c['target'] == target:
                    status = "✅" if c['viable'] else "❌"
                    print(f"{pair_name:<25} {strategy:<15} {c['n_subjects']:>10} "
                          f"{c['uniform_source_ch']:>8} {c['uniform_target_ch']:>8} {status}")
                    pair_name = ""  # Only show pair name once

    # Find best config across all strategies
    all_viable = []
    for strategy, configs in all_configs.items():
        for c in configs:
            if c['viable']:
                all_viable.append(c)

    if all_viable:
        print("\n" + "=" * 70)
        print("RECOMMENDED CONFIGURATIONS")
        print("=" * 70)

        # Best by subject count
        best_subjects = max(all_viable, key=lambda x: x['n_subjects'])
        # Best by channel count (product of src and tgt)
        best_channels = max(all_viable, key=lambda x: x['uniform_source_ch'] * x['uniform_target_ch'])

        print(f"""
  MOST SUBJECTS:
    {best_subjects['source']} → {best_subjects['target']} (strategy: {best_subjects['strategy']})
    {best_subjects['n_subjects']} subjects, {best_subjects['uniform_source_ch']}→{best_subjects['uniform_target_ch']} channels

  MOST CHANNELS:
    {best_channels['source']} → {best_channels['target']} (strategy: {best_channels['strategy']})
    {best_channels['n_subjects']} subjects, {best_channels['uniform_source_ch']}→{best_channels['uniform_target_ch']} channels
""")

        # Generate commands for viable configs with best strategy
        print("=" * 70)
        print("PREPROCESSING COMMANDS (for min strategy - most subjects)")
        print("=" * 70)

        for c in all_configs.get('min', []):
            if c['viable']:
                suffix = f"{c['source'][:4]}_{c['target'][:4]}"
                cmd = generate_preprocessing_command(c, suffix)
                print(cmd)

                if args.save_configs:
                    config_path = dataset_dir / f"config_{suffix}.json"
                    save_configuration(c, config_path)


if __name__ == "__main__":
    main()
