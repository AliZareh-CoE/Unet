#!/usr/bin/env python3
"""Probe ECoG experiments: region pairs, electrode metadata, and spatial coverage.

Outputs detailed info about each experiment's electrode placement, lobe channels,
gyrus breakdown, Brodmann areas, 3D coordinate availability, and viable LOSO runs.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (
    _ECOG_DATA_DIR, _enumerate_ecog_recordings, _parse_ecog_lobe,
    ECOG_BRAIN_LOBES,
)
import numpy as np

EXPERIMENTS = ["fingerflex", "faceshouses", "motor_imagery", "joystick_track", "memory_nback"]
MIN_SUBJECTS = 3
MIN_CHANNELS = 4


def inspect_recording(dat, rec_id):
    """Inspect a single recording's electrode metadata."""
    info = {"rec_id": rec_id, "lobes": {}, "has_locs": False, "has_gyrus": False}

    lobes_raw = dat.get("lobe", [])
    n_channels = np.float32(dat["V"]).shape[1] if "V" in dat else 0
    n_samples = np.float32(dat["V"]).shape[0] if "V" in dat else 0

    if len(lobes_raw) == 0:
        return info

    # Parse lobes
    lobe_counts = {}
    excluded = 0
    for lobe_raw in lobes_raw:
        lobe = _parse_ecog_lobe(str(lobe_raw))
        if lobe is None:
            excluded += 1
            continue
        lobe_counts[lobe] = lobe_counts.get(lobe, 0) + 1
    info["lobes"] = lobe_counts
    info["excluded"] = excluded
    info["n_channels"] = n_channels
    info["n_samples"] = n_samples

    # 3D electrode coordinates
    if "locs" in dat:
        locs = np.array(dat["locs"], dtype=np.float64)
        if locs.ndim == 2 and locs.shape[1] >= 3:
            info["has_locs"] = True
            info["locs_shape"] = locs.shape
            # Compute spatial extent per lobe
            lobe_list = [_parse_ecog_lobe(str(l)) for l in lobes_raw]
            lobe_extents = {}
            for lobe_name in set(lobe_list):
                if lobe_name is None:
                    continue
                ch_idxs = [i for i, l in enumerate(lobe_list) if l == lobe_name and i < locs.shape[0]]
                if len(ch_idxs) >= 2:
                    lobe_locs = locs[ch_idxs, :3]
                    extent = lobe_locs.max(axis=0) - lobe_locs.min(axis=0)
                    spread = np.linalg.norm(extent)
                    lobe_extents[lobe_name] = {
                        "spread_mm": float(f"{spread:.1f}"),
                        "centroid": lobe_locs.mean(axis=0).tolist(),
                    }
            info["lobe_extents"] = lobe_extents

    # Gyrus breakdown per lobe
    if "gyrus" in dat:
        info["has_gyrus"] = True
        lobe_list = [_parse_ecog_lobe(str(l)) for l in lobes_raw]
        gyri_raw = [str(g) for g in dat["gyrus"]]
        lobe_gyri = {}
        for i, (lobe, gyrus) in enumerate(zip(lobe_list, gyri_raw)):
            if lobe is None or i >= n_channels:
                continue
            if lobe not in lobe_gyri:
                lobe_gyri[lobe] = {}
            gyrus_clean = gyrus.strip()
            lobe_gyri[lobe][gyrus_clean] = lobe_gyri[lobe].get(gyrus_clean, 0) + 1
        info["lobe_gyri"] = lobe_gyri

    # Brodmann areas per lobe
    if "Brodmann_Area" in dat:
        lobe_list = [_parse_ecog_lobe(str(l)) for l in lobes_raw]
        ba_raw = dat["Brodmann_Area"]
        lobe_ba = {}
        for i, (lobe, ba) in enumerate(zip(lobe_list, ba_raw)):
            if lobe is None or i >= n_channels:
                continue
            if lobe not in lobe_ba:
                lobe_ba[lobe] = set()
            try:
                val = float(ba)
                if not np.isnan(val):
                    lobe_ba[lobe].add(int(val))
            except (ValueError, TypeError):
                pass
        info["lobe_brodmann"] = {k: sorted(v) for k, v in lobe_ba.items()}

    return info


def main():
    print("=" * 100)
    print("  ECoG EXPERIMENT / REGION PAIR VIABILITY PROBE")
    print(f"  Data dir: {_ECOG_DATA_DIR}")
    print(f"  Min subjects for LOSO: {MIN_SUBJECTS}")
    print(f"  Min channels per region: {MIN_CHANNELS}")
    print("=" * 100)

    viable_runs = []
    by_experiment = {}

    for experiment in EXPERIMENTS:
        npz_path = _ECOG_DATA_DIR / f"{experiment}.npz"
        if not npz_path.exists():
            print(f"\n  {experiment}: DATA NOT FOUND ({npz_path})")
            continue

        alldat = np.load(npz_path, allow_pickle=True)["dat"]
        recordings = _enumerate_ecog_recordings(alldat)

        print(f"\n{'─' * 100}")
        print(f"  {experiment.upper()} — {len(recordings)} recordings, shape: {alldat.shape}")
        print(f"{'─' * 100}")

        recording_infos = {}
        for subj_idx, block_idx, rec_id in recordings:
            dat = alldat[subj_idx, block_idx]
            info = inspect_recording(dat, rec_id)
            recording_infos[rec_id] = info

            # Print compact summary
            lobes_str = ", ".join(f"{k}:{v}" for k, v in sorted(info["lobes"].items()))
            excl_str = f" (excl {info.get('excluded', 0)})" if info.get("excluded", 0) else ""
            locs_str = " [3D locs]" if info["has_locs"] else " [NO locs]"
            samp_str = f" {info.get('n_samples', 0)/1000:.0f}k samples" if info.get("n_samples") else ""

            print(f"    {rec_id}: {info.get('n_channels', 0)}ch{excl_str}{locs_str}{samp_str}")
            print(f"      Lobes: {lobes_str}")

            # Gyrus detail
            if "lobe_gyri" in info:
                for lobe_name, gyri in sorted(info["lobe_gyri"].items()):
                    gyri_str = ", ".join(f"{g}({c})" for g, c in sorted(gyri.items(), key=lambda x: -x[1]))
                    print(f"      {lobe_name} gyri: {gyri_str}")

            # Brodmann areas
            if "lobe_brodmann" in info:
                for lobe_name, bas in sorted(info["lobe_brodmann"].items()):
                    if bas:
                        print(f"      {lobe_name} Brodmann: {bas}")

            # Spatial extent
            if "lobe_extents" in info:
                for lobe_name, ext in sorted(info["lobe_extents"].items()):
                    print(f"      {lobe_name} spatial spread: {ext['spread_mm']}mm")

        # Region pair viability
        print(f"\n  Region pair viability (>= {MIN_SUBJECTS} subjects with >= {MIN_CHANNELS} ch each):")
        for src in ECOG_BRAIN_LOBES:
            for tgt in ECOG_BRAIN_LOBES:
                if src == tgt:
                    continue

                valid_subjects = []
                for rec_id, info in recording_infos.items():
                    src_ch = info["lobes"].get(src, 0)
                    tgt_ch = info["lobes"].get(tgt, 0)
                    if src_ch >= MIN_CHANNELS and tgt_ch >= MIN_CHANNELS:
                        valid_subjects.append((rec_id, src_ch, tgt_ch))

                n_valid = len(valid_subjects)
                mark = "  OK" if n_valid >= MIN_SUBJECTS else "  --"

                if n_valid > 0:
                    subj_str = ", ".join(f"{r[0]}({r[1]}/{r[2]}ch)" for r in valid_subjects)
                    min_src = min(r[1] for r in valid_subjects)
                    min_tgt = min(r[2] for r in valid_subjects)
                    print(f"  {mark} {src:>10} -> {tgt:<10}  {n_valid} subj  "
                          f"min_ch=({min_src}/{min_tgt})  [{subj_str}]")

                    if n_valid >= MIN_SUBJECTS:
                        viable_runs.append({
                            "experiment": experiment,
                            "source": src,
                            "target": tgt,
                            "n_subjects": n_valid,
                            "subjects": valid_subjects,
                            "min_src_ch": min_src,
                            "min_tgt_ch": min_tgt,
                        })

        by_experiment[experiment] = recording_infos

    # Summary
    print(f"\n\n{'=' * 100}")
    print(f"  VIABLE LOSO RUNS ({len(viable_runs)} total)")
    print(f"{'=' * 100}")
    print(f"  {'Experiment':<18} {'Direction':<25} {'Subj':>5} {'MinSrc':>7} {'MinTgt':>7}")
    print(f"  {'─'*18} {'─'*25} {'─'*5} {'─'*7} {'─'*7}")

    for r in viable_runs:
        direction = f"{r['source']} -> {r['target']}"
        print(f"  {r['experiment']:<18} {direction:<25} {r['n_subjects']:>5} "
              f"{r['min_src_ch']:>7} {r['min_tgt_ch']:>7}")

    # Print commands
    print(f"\n\n{'=' * 100}")
    print(f"  COMMANDS TO RUN ({len(viable_runs)} runs)")
    print(f"{'=' * 100}")
    for r in viable_runs:
        exp = r["experiment"]
        src = r["source"]
        tgt = r["target"]
        out_dir = f"results/ECoG/{exp}/{src}_to_{tgt}"
        print(f"python LOSO/runner.py --dataset ecog --ecog-experiment {exp} "
              f"--ecog-source-region {src} --ecog-target-region {tgt} "
              f"--fsdp --output-dir {out_dir}")

    # Summary by experiment
    exp_groups = {}
    for r in viable_runs:
        exp_groups.setdefault(r["experiment"], []).append(r)

    print(f"\n\n{'=' * 100}")
    print(f"  SUMMARY BY EXPERIMENT")
    print(f"{'=' * 100}")
    for exp in EXPERIMENTS:
        runs = exp_groups.get(exp, [])
        if runs:
            regions = set()
            for r in runs:
                regions.add(r["source"])
                regions.add(r["target"])
            print(f"  {exp:<18}  {len(runs):>3} runs  regions: {sorted(regions)}")
        else:
            print(f"  {exp:<18}    0 runs  (insufficient subjects or channels)")

    print(f"\n  TOTAL: {len(viable_runs)} viable LOSO runs across {len(exp_groups)} experiments")


if __name__ == "__main__":
    main()
