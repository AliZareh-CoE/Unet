#!/usr/bin/env python3
"""
Quick script to check Allen Neuropixels dataset statistics.
Shows how many sessions have each brain region pair.
"""

import sys

def main():
    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    except ImportError:
        print("Installing allensdk...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "allensdk", "-q"])
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

    from pathlib import Path
    import pandas as pd
    from collections import defaultdict

    # Initialize cache
    output_dir = Path("/data/allen_neuropixels")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    print("Fetching Allen Neuropixels metadata...")
    cache = EcephysProjectCache.from_warehouse(manifest=str(manifest_path))

    # Get sessions
    sessions = cache.get_session_table()
    print(f"\nTotal sessions: {len(sessions)}")

    # Filter Brain Observatory 1.1
    bo_sessions = sessions[sessions['session_type'] == 'brain_observatory_1.1']
    print(f"Brain Observatory 1.1 sessions: {len(bo_sessions)}")

    # Get unique specimens (mice)
    if 'specimen_id' in bo_sessions.columns:
        n_mice = bo_sessions['specimen_id'].nunique()
        print(f"Unique mice: {n_mice}")

    # Get channels to check region availability
    channels = cache.get_channels()
    print(f"\nTotal channels: {len(channels)}")

    # Get units to check unit counts per region
    units = cache.get_units()
    print(f"Total units: {len(units)}")

    # Filter to BO 1.1 sessions
    bo_session_ids = set(bo_sessions.index.tolist())

    # Check which regions are available
    print("\n" + "="*70)
    print("BRAIN REGION STATISTICS (Brain Observatory 1.1)")
    print("="*70)

    # Get channels for BO sessions
    bo_channels = channels[channels['ecephys_session_id'].isin(bo_session_ids)]

    # Count sessions per region
    region_stats = defaultdict(lambda: {'sessions': set(), 'channels': 0, 'units': 0})

    for _, row in bo_channels.iterrows():
        region = row.get('ecephys_structure_acronym')
        if region and region != 'none':
            session_id = row['ecephys_session_id']
            region_stats[region]['sessions'].add(session_id)
            region_stats[region]['channels'] += 1

    # Get unit counts per region
    bo_units = units[units['ecephys_session_id'].isin(bo_session_ids)]
    for _, row in bo_units.iterrows():
        region = row.get('ecephys_structure_acronym')
        if region and region != 'none':
            region_stats[region]['units'] += 1

    # Print stats sorted by session count
    print(f"\n{'Region':<10} {'Sessions':<10} {'Channels':<12} {'Units':<10}")
    print("-" * 45)

    sorted_regions = sorted(region_stats.items(),
                           key=lambda x: len(x[1]['sessions']),
                           reverse=True)

    for region, stats in sorted_regions:
        n_sess = len(stats['sessions'])
        n_chan = stats['channels']
        n_units = stats['units']
        print(f"{region:<10} {n_sess:<10} {n_chan:<12} {n_units:<10}")

    # Check pairings
    print("\n" + "="*70)
    print("REGION PAIRINGS - Sessions with BOTH regions")
    print("="*70)

    key_pairs = [
        ('LGd', 'VISp'),   # LGN -> V1
        ('VISp', 'VISl'),  # V1 -> LM
        ('VISp', 'CA1'),   # V1 -> Hippocampus
        ('VISp', 'VISal'), # V1 -> AL
        ('VISp', 'VISpm'), # V1 -> PM
        ('LP', 'VISp'),    # LP (thalamus) -> V1
    ]

    for src, tgt in key_pairs:
        src_sessions = region_stats[src]['sessions']
        tgt_sessions = region_stats[tgt]['sessions']
        overlap = src_sessions & tgt_sessions
        print(f"{src} → {tgt}: {len(overlap)} sessions")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    # Find best pairing
    best_pair = None
    best_count = 0
    for src, tgt in key_pairs:
        overlap = len(region_stats[src]['sessions'] & region_stats[tgt]['sessions'])
        if overlap > best_count:
            best_count = overlap
            best_pair = (src, tgt)

    if best_pair:
        print(f"Best pairing: {best_pair[0]} → {best_pair[1]} ({best_count} sessions)")
        print(f"\nThis gives you {best_count} independent sessions from different mice!")

    return region_stats


if __name__ == '__main__':
    main()
