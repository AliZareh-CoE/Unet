#!/usr/bin/env python3
"""Probe ECoG experiments to find which region pairs have >= 3 subjects.

Outputs a table showing all viable LOSO runs and the exact commands to run them.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (
    list_ecog_subjects, load_ecog_subject, _ECOG_DATA_DIR,
    ECOG_BRAIN_LOBES, _enumerate_ecog_recordings,
)
import numpy as np

EXPERIMENTS = ["fingerflex", "faceshouses", "motor_imagery", "joystick_track", "memory_nback"]
MIN_SUBJECTS = 3
MIN_CHANNELS = 4


def main():
    print("=" * 90)
    print("  ECoG EXPERIMENT / REGION PAIR VIABILITY PROBE")
    print(f"  Data dir: {_ECOG_DATA_DIR}")
    print(f"  Min subjects for LOSO: {MIN_SUBJECTS}")
    print(f"  Min channels per region: {MIN_CHANNELS}")
    print("=" * 90)

    viable_runs = []
    all_experiment_info = {}

    for experiment in EXPERIMENTS:
        npz_path = _ECOG_DATA_DIR / f"{experiment}.npz"
        if not npz_path.exists():
            print(f"\n  {experiment}: DATA NOT FOUND ({npz_path})")
            continue

        alldat = np.load(npz_path, allow_pickle=True)["dat"]
        recordings = _enumerate_ecog_recordings(alldat)

        print(f"\n{'─' * 90}")
        print(f"  {experiment.upper()} — {len(recordings)} recordings")
        print(f"{'─' * 90}")

        # For each recording, catalog which lobes have >= MIN_CHANNELS channels
        recording_lobes = {}  # rec_id -> {lobe: n_channels}
        for subj_idx, block_idx, rec_id in recordings:
            dat = alldat[subj_idx, block_idx]
            lobes = dat.get("lobe", [])
            if len(lobes) == 0:
                continue

            from data import _parse_ecog_lobe
            lobe_counts = {}
            for lobe_raw in lobes:
                lobe = _parse_ecog_lobe(str(lobe_raw))
                lobe_counts[lobe] = lobe_counts.get(lobe, 0) + 1

            recording_lobes[rec_id] = {
                "subj_idx": subj_idx,
                "block_idx": block_idx,
                "lobes": lobe_counts,
            }

            print(f"    {rec_id}: {dict(sorted(lobe_counts.items()))}")

        # Now test all region pairs
        print(f"\n  Region pair viability (>= {MIN_SUBJECTS} subjects with >= {MIN_CHANNELS} ch each):")
        region_pairs_tested = 0
        for src in ECOG_BRAIN_LOBES:
            for tgt in ECOG_BRAIN_LOBES:
                if src == tgt:
                    continue

                valid_subjects = []
                for rec_id, info in recording_lobes.items():
                    src_ch = info["lobes"].get(src, 0)
                    tgt_ch = info["lobes"].get(tgt, 0)
                    if src_ch >= MIN_CHANNELS and tgt_ch >= MIN_CHANNELS:
                        valid_subjects.append((rec_id, src_ch, tgt_ch))

                n_valid = len(valid_subjects)
                status = "OK" if n_valid >= MIN_SUBJECTS else "SKIP"
                mark = "  ✓" if n_valid >= MIN_SUBJECTS else "  ✗"
                region_pairs_tested += 1

                if n_valid > 0:
                    subj_str = ", ".join(f"{r[0]}({r[1]}/{r[2]}ch)" for r in valid_subjects)
                    print(f"  {mark} {src:>10} → {tgt:<10}  {n_valid} subjects  [{subj_str}]")

                    if n_valid >= MIN_SUBJECTS:
                        viable_runs.append({
                            "experiment": experiment,
                            "source": src,
                            "target": tgt,
                            "n_subjects": n_valid,
                            "subjects": valid_subjects,
                        })

        all_experiment_info[experiment] = recording_lobes

    # Summary
    print(f"\n\n{'=' * 90}")
    print(f"  VIABLE LOSO RUNS ({len(viable_runs)} total)")
    print(f"{'=' * 90}")
    print(f"  {'Experiment':<18} {'Direction':<25} {'Subjects':>8}")
    print(f"  {'─'*18} {'─'*25} {'─'*8}")

    by_experiment = {}
    for r in viable_runs:
        exp = r["experiment"]
        if exp not in by_experiment:
            by_experiment[exp] = []
        by_experiment[exp].append(r)
        print(f"  {exp:<18} {r['source']} → {r['target']:<12}  {r['n_subjects']:>8}")

    # Print commands
    print(f"\n\n{'=' * 90}")
    print(f"  COMMANDS TO RUN ({len(viable_runs)} runs)")
    print(f"{'=' * 90}")
    for r in viable_runs:
        exp = r["experiment"]
        src = r["source"]
        tgt = r["target"]
        out_dir = f"results/ECoG/{exp}/{src}_to_{tgt}"
        print(f"python LOSO/runner.py --dataset ecog --ecog-experiment {exp} "
              f"--ecog-source-region {src} --ecog-target-region {tgt} "
              f"--fsdp --output-dir {out_dir}")

    # Summary by experiment
    print(f"\n\n{'=' * 90}")
    print(f"  SUMMARY BY EXPERIMENT")
    print(f"{'=' * 90}")
    for exp in EXPERIMENTS:
        runs = by_experiment.get(exp, [])
        n_runs = len(runs)
        if n_runs > 0:
            regions_involved = set()
            for r in runs:
                regions_involved.add(r["source"])
                regions_involved.add(r["target"])
            print(f"  {exp:<18}  {n_runs:>3} runs  regions: {sorted(regions_involved)}")
        else:
            print(f"  {exp:<18}    0 runs  (insufficient subjects or channels)")

    print(f"\n  TOTAL: {len(viable_runs)} viable LOSO runs across {len(by_experiment)} experiments")


if __name__ == "__main__":
    main()
