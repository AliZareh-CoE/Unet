#!/usr/bin/env python3
"""
Analyze COGITATE Dataset Coverage for Neural Translation
=========================================================

Analyzes the preprocessed COGITATE data to determine:
1. Sampling rate distribution
2. Region coverage across subjects
3. Viable source→target region pairs for LOSO
4. Recommended preprocessing strategy

Usage:
    python scripts/analyze_cogitate_coverage.py /data/COGITATEDataset
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def load_subject_info(subject_dir: Path) -> Dict:
    """Load subject's region mapping and metadata."""
    region_file = subject_dir / 'region_mapping.json'
    channels_file = subject_dir / 'channels.csv'
    continuous_file = subject_dir / 'continuous.npy'

    info = {
        'subject_id': subject_dir.name,
        'regions': {},
        'total_channels': 0,
        'n_samples': 0,
        'sampling_rate': None,
    }

    if region_file.exists():
        with open(region_file) as f:
            region_data = json.load(f)
        for region, data in region_data.items():
            info['regions'][region] = data['n_channels']

    if continuous_file.exists():
        # Just get shape without loading full array
        data = np.load(continuous_file, mmap_mode='r')
        info['total_channels'] = data.shape[0]
        info['n_samples'] = data.shape[1]

    if channels_file.exists():
        channels_df = pd.read_csv(channels_file)
        if 'sampling_frequency' in channels_df.columns:
            info['sampling_rate'] = channels_df['sampling_frequency'].iloc[0]

    return info


def analyze_dataset(dataset_dir: Path) -> Dict:
    """Comprehensive analysis of the dataset."""
    dataset_dir = Path(dataset_dir)

    # Load all subjects
    subject_dirs = sorted([d for d in dataset_dir.iterdir()
                          if d.is_dir() and d.name.startswith('sub-')])

    print(f"Found {len(subject_dirs)} subjects\n")

    subjects = []
    for sd in subject_dirs:
        info = load_subject_info(sd)
        subjects.append(info)

    # Also try to get sampling rate from dataset_info.json
    dataset_info_file = dataset_dir / 'dataset_info.json'
    if dataset_info_file.exists():
        with open(dataset_info_file) as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    return {
        'subjects': subjects,
        'dataset_info': dataset_info,
    }


def print_sampling_rate_analysis(subjects: List[Dict]):
    """Analyze sampling rate distribution."""
    print("=" * 70)
    print("1. SAMPLING RATE ANALYSIS")
    print("=" * 70)

    # Group by sampling rate
    by_rate = defaultdict(list)
    for s in subjects:
        # Infer rate from samples if not directly available
        rate = s.get('sampling_rate')
        if rate is None and s['n_samples'] > 0:
            # Try to infer from common rates
            duration = s['n_samples'] / 1024  # Assume 1024 first
            if duration > 100:  # Reasonable recording length
                rate = 1024
        by_rate[rate].append(s['subject_id'])

    print("\nSampling Rate Distribution:")
    print("-" * 40)
    for rate in sorted(by_rate.keys(), key=lambda x: x or 0):
        subs = by_rate[rate]
        print(f"  {rate} Hz: {len(subs)} subjects")
        print(f"    {subs}")

    # Recommendation
    print("\n📊 RECOMMENDATION:")
    rates = [r for r in by_rate.keys() if r is not None]
    if rates:
        # Count subjects at each rate threshold
        at_512 = sum(len(v) for k, v in by_rate.items() if k and k >= 512)
        at_1024 = sum(len(v) for k, v in by_rate.items() if k and k >= 1024)
        at_2048 = sum(len(v) for k, v in by_rate.items() if k and k >= 2048)

        print(f"  - At ≥512 Hz: {at_512} subjects available")
        print(f"  - At ≥1024 Hz: {at_1024} subjects available")
        print(f"  - At ≥2048 Hz: {at_2048} subjects available")

        if at_1024 >= 30:
            print(f"  → Recommend: 1024 Hz (good balance, {at_1024} subjects)")
        elif at_512 >= 30:
            print(f"  → Recommend: 512 Hz (maximize subjects, {at_512} subjects)")

    return by_rate


def print_region_coverage(subjects: List[Dict]):
    """Analyze region coverage across subjects."""
    print("\n" + "=" * 70)
    print("2. REGION COVERAGE ANALYSIS")
    print("=" * 70)

    # Collect all regions
    all_regions = set()
    for s in subjects:
        all_regions.update(s['regions'].keys())

    # Count subjects and channels per region
    region_stats = {}
    for region in sorted(all_regions):
        counts = []
        subjects_with = []
        for s in subjects:
            if region in s['regions'] and s['regions'][region] > 0:
                counts.append(s['regions'][region])
                subjects_with.append(s['subject_id'])

        region_stats[region] = {
            'n_subjects': len(subjects_with),
            'subjects': subjects_with,
            'min_ch': min(counts) if counts else 0,
            'max_ch': max(counts) if counts else 0,
            'mean_ch': np.mean(counts) if counts else 0,
            'median_ch': np.median(counts) if counts else 0,
        }

    print("\nRegion Coverage Summary:")
    print("-" * 70)
    print(f"{'Region':<20} {'Subjects':>10} {'Min Ch':>10} {'Max Ch':>10} {'Mean':>10}")
    print("-" * 70)

    for region in sorted(region_stats.keys(), key=lambda x: -region_stats[x]['n_subjects']):
        stats = region_stats[region]
        print(f"{region:<20} {stats['n_subjects']:>10} {stats['min_ch']:>10} "
              f"{stats['max_ch']:>10} {stats['mean_ch']:>10.1f}")

    return region_stats


def print_region_pair_analysis(subjects: List[Dict], region_stats: Dict):
    """Analyze viable source→target region pairs."""
    print("\n" + "=" * 70)
    print("3. VIABLE REGION PAIRS FOR NEURAL TRANSLATION")
    print("=" * 70)

    # Define potentially interesting pairs (scientifically motivated)
    interesting_pairs = [
        ('hippocampus', 'amygdala', 'Memory ↔ Emotion'),
        ('hippocampus', 'temporal', 'Memory ↔ Perception'),
        ('temporal', 'frontal', 'Perception ↔ Cognition'),
        ('amygdala', 'temporal', 'Emotion ↔ Perception'),
        ('amygdala', 'frontal', 'Emotion ↔ Cognition'),
        ('temporal', 'parietal', 'Auditory ↔ Association'),
        ('hippocampus', 'frontal', 'Memory ↔ Executive'),
        ('insula', 'amygdala', 'Interoception ↔ Emotion'),
        ('temporal', 'occipital', 'Temporal ↔ Visual'),
        ('fusiform', 'temporal', 'Face/Object ↔ Temporal'),
    ]

    print("\nPotential Translation Pairs:")
    print("-" * 70)
    print(f"{'Source':<15} {'Target':<15} {'Subjects':>10} {'Min Src':>10} {'Min Tgt':>10} {'Rationale'}")
    print("-" * 70)

    pair_stats = []
    for source, target, rationale in interesting_pairs:
        # Count subjects with BOTH regions
        subjects_with_both = []
        min_source_ch = float('inf')
        min_target_ch = float('inf')

        for s in subjects:
            src_ch = s['regions'].get(source, 0)
            tgt_ch = s['regions'].get(target, 0)
            if src_ch > 0 and tgt_ch > 0:
                subjects_with_both.append(s['subject_id'])
                min_source_ch = min(min_source_ch, src_ch)
                min_target_ch = min(min_target_ch, tgt_ch)

        if subjects_with_both:
            min_source_ch = int(min_source_ch)
            min_target_ch = int(min_target_ch)
        else:
            min_source_ch = 0
            min_target_ch = 0

        pair_stats.append({
            'source': source,
            'target': target,
            'n_subjects': len(subjects_with_both),
            'subjects': subjects_with_both,
            'min_source_ch': min_source_ch,
            'min_target_ch': min_target_ch,
            'rationale': rationale,
        })

        print(f"{source:<15} {target:<15} {len(subjects_with_both):>10} "
              f"{min_source_ch:>10} {min_target_ch:>10} {rationale}")

    # Sort by number of subjects
    pair_stats.sort(key=lambda x: -x['n_subjects'])

    print("\n📊 TOP RECOMMENDATIONS:")
    for i, p in enumerate(pair_stats[:3]):
        print(f"\n  {i+1}. {p['source']} → {p['target']}")
        print(f"     Subjects: {p['n_subjects']}")
        print(f"     Min channels: {p['min_source_ch']} (src) → {p['min_target_ch']} (tgt)")
        print(f"     Rationale: {p['rationale']}")

    return pair_stats


def print_subject_inclusion_matrix(subjects: List[Dict], top_pairs: List[Dict], by_rate: Dict):
    """Show which subjects qualify for each configuration."""
    print("\n" + "=" * 70)
    print("4. SUBJECT INCLUSION MATRIX")
    print("=" * 70)

    # Get subjects at each sampling rate threshold
    subs_1024plus = set()
    subs_512plus = set()
    for rate, subs in by_rate.items():
        if rate and rate >= 1024:
            subs_1024plus.update(subs)
        if rate and rate >= 512:
            subs_512plus.update(subs)

    print("\nSubjects qualifying for each configuration:")
    print("-" * 70)

    for p in top_pairs[:3]:
        pair_subs = set(p['subjects'])

        # At different rate thresholds
        at_1024 = pair_subs & subs_1024plus
        at_512 = pair_subs & subs_512plus

        print(f"\n{p['source']} → {p['target']}:")
        print(f"  Total with both regions: {len(pair_subs)}")
        print(f"  At ≥1024 Hz: {len(at_1024)} subjects")
        print(f"  At ≥512 Hz:  {len(at_512)} subjects")

        if len(at_1024) >= 20:
            print(f"  ✓ VIABLE at 1024 Hz with {len(at_1024)} subjects")
        elif len(at_512) >= 20:
            print(f"  ✓ VIABLE at 512 Hz with {len(at_512)} subjects")
        else:
            print(f"  ✗ May need to relax constraints")


def print_final_recommendation(subjects: List[Dict], pair_stats: List[Dict], by_rate: Dict):
    """Print final strategic recommendation."""
    print("\n" + "=" * 70)
    print("5. FINAL STRATEGIC RECOMMENDATION")
    print("=" * 70)

    # Find best configuration
    best_config = None
    best_n_subjects = 0

    subs_1024plus = set()
    for rate, subs in by_rate.items():
        if rate and rate >= 1024:
            subs_1024plus.update(subs)

    for p in pair_stats:
        pair_subs = set(p['subjects'])
        at_1024 = pair_subs & subs_1024plus

        if len(at_1024) > best_n_subjects and p['min_source_ch'] >= 3 and p['min_target_ch'] >= 3:
            best_n_subjects = len(at_1024)
            best_config = {
                'source': p['source'],
                'target': p['target'],
                'n_subjects': len(at_1024),
                'subjects': sorted(at_1024),
                'min_source_ch': p['min_source_ch'],
                'min_target_ch': p['min_target_ch'],
                'sampling_rate': 1024,
            }

    if best_config:
        print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED CONFIGURATION                         │
├─────────────────────────────────────────────────────────────────────┤
│  Source Region:     {best_config['source']:<46} │
│  Target Region:     {best_config['target']:<46} │
│  Sampling Rate:     {best_config['sampling_rate']} Hz{' '*42} │
│  Subjects:          {best_config['n_subjects']:<46} │
│  Min Source Ch:     {best_config['min_source_ch']:<46} │
│  Min Target Ch:     {best_config['min_target_ch']:<46} │
└─────────────────────────────────────────────────────────────────────┘
""")
        print("Qualifying subjects:")
        print(f"  {best_config['subjects']}")

        print("\n📋 PREPROCESSING COMMAND:")
        excluded = set(s['subject_id'] for s in subjects) - set(best_config['subjects'])
        print(f"""
python scripts/preprocess_cogitate_bids.py /data/COG_ECOG_EXP1_BIDS \\
    --output-dir /data/COGITATEDataset \\
    --target-sfreq {best_config['sampling_rate']} \\
    --no-filter \\
    --subjects {' '.join(best_config['subjects'])}
""")

        if excluded:
            print(f"\nExcluded subjects ({len(excluded)}): {sorted(excluded)}")

    return best_config


def main():
    parser = argparse.ArgumentParser(description="Analyze COGITATE dataset coverage")
    parser.add_argument("dataset_dir", type=str, help="Path to preprocessed dataset")
    args = parser.parse_args()

    print("=" * 70)
    print("COGITATE DATASET STRATEGIC ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {args.dataset_dir}\n")

    # Load data
    data = analyze_dataset(args.dataset_dir)
    subjects = data['subjects']

    # Run analyses
    by_rate = print_sampling_rate_analysis(subjects)
    region_stats = print_region_coverage(subjects)
    pair_stats = print_region_pair_analysis(subjects, region_stats)
    print_subject_inclusion_matrix(subjects, pair_stats, by_rate)
    best_config = print_final_recommendation(subjects, pair_stats, by_rate)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
