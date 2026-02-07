#!/usr/bin/env python3
"""Explore the PFC dataset: sessions, rats, trial types, and possible runs.

Usage:
    python explore_pfc.py                           # auto-detect data location
    python explore_pfc.py /path/to/pfc/processed_data  # explicit path
    UNET_DATA_DIR=/data python explore_pfc.py       # via env var
"""

import os
import sys
import glob as globmod
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("ERROR: numpy and pandas required. Install with: pip install numpy pandas")
    sys.exit(1)


def find_data_dir(explicit_path=None):
    """Try multiple locations to find the PFC data."""
    candidates = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    # From env var (same as data.py)
    env_dir = os.environ.get("UNET_DATA_DIR", "/data")
    candidates.append(Path(env_dir) / "pfc" / "processed_data")

    # Common relative/absolute paths
    script_dir = Path(__file__).resolve().parent
    candidates += [
        script_dir / "data" / "pfc" / "processed_data",
        Path.home() / "data" / "pfc" / "processed_data",
        Path("/data/pfc/processed_data"),
        Path("/home/user/Unet/data/pfc/processed_data"),
    ]

    # Also search for metadata.csv anywhere under common roots
    for root in [script_dir, Path.home(), Path("/data")]:
        for csv in root.rglob("metadata.csv") if root.exists() else []:
            if "pfc" in str(csv).lower():
                candidates.append(csv.parent)

    for p in candidates:
        if p.exists() and (p / "metadata.csv").exists():
            return p

    # Last resort: find any metadata.csv
    for p in candidates:
        if p.exists():
            return p

    return None


def print_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main():
    explicit = sys.argv[1] if len(sys.argv) > 1 else None
    data_dir = find_data_dir(explicit)

    print_header("PFC DATASET EXPLORATION")

    if data_dir is None:
        print("\n  DATA NOT FOUND!")
        print("  Searched:")
        print("    - $UNET_DATA_DIR/pfc/processed_data")
        print("    - ./data/pfc/processed_data")
        print("    - /data/pfc/processed_data")
        print("    - ~/data/pfc/processed_data")
        print()
        print("  To use: python explore_pfc.py /path/to/pfc/processed_data")
        print("  Or set: export UNET_DATA_DIR=/your/data/root")
        print()
        print("  Expected files in that directory:")
        print("    - neural_data.npy   (494 trials x 6250 time x 96 channels)")
        print("    - metadata.csv      (session, rat, trial_id, trial_type, ...)")
        sys.exit(1)

    print(f"\n  Data directory: {data_dir}")
    print(f"\n  Files:")
    for f in sorted(data_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name:40s} {size_mb:8.2f} MB")

    # -------------------------------------------------------------------------
    # Neural data
    # -------------------------------------------------------------------------
    neural_path = data_dir / "neural_data.npy"
    if neural_path.exists():
        print_header("NEURAL DATA")
        data = np.load(neural_path)
        print(f"  Raw shape: {data.shape}  (trials, time, channels)")
        print(f"  Dtype: {data.dtype}")
        print(f"  Value range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"  Mean: {data.mean():.4f}, Std: {data.std():.4f}")
        n_trials, n_time, n_channels = data.shape
        print(f"\n  Trials:     {n_trials}")
        print(f"  Time pts:   {n_time} ({n_time / 1250:.1f}s at 1250 Hz)")
        print(f"  Channels:   {n_channels}")
        print(f"    PFC:  channels 0-63   ({min(64, n_channels)} channels)")
        print(f"    CA1:  channels 64-95  ({max(0, n_channels - 64)} channels)")
    else:
        print(f"\n  neural_data.npy not found")
        data = None

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        print(f"\n  metadata.csv not found — cannot explore sessions/rats")
        sys.exit(1)

    print_header("METADATA")
    df = pd.read_csv(meta_path)
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Sample rows:")
    print(df.head(10).to_string(index=False))

    # -------------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------------
    sessions = []
    if "session" in df.columns:
        sessions = sorted(df["session"].unique())
        print_header(f"SESSIONS ({len(sessions)} unique)")
        for s in sessions:
            sub = df[df["session"] == s]
            rats = sub["rat"].unique().tolist() if "rat" in df.columns else ["?"]
            types = sub["trial_type"].unique().tolist() if "trial_type" in df.columns else ["?"]
            print(f"  {str(s):20s}  {len(sub):4d} trials  rats={rats}  types={types}")

    # -------------------------------------------------------------------------
    # Rats / Subjects
    # -------------------------------------------------------------------------
    rats = []
    if "rat" in df.columns:
        rats = sorted(df["rat"].unique())
        print_header(f"RATS / SUBJECTS ({len(rats)} unique)")
        for r in rats:
            sub = df[df["rat"] == r]
            sess = sorted(sub["session"].unique()) if "session" in df.columns else ["?"]
            print(f"  Rat '{r}':  {len(sub)} trials  sessions={sess}")

    # -------------------------------------------------------------------------
    # Trial types
    # -------------------------------------------------------------------------
    if "trial_type" in df.columns:
        print_header("TRIAL TYPES")
        for tt in sorted(df["trial_type"].unique(), key=str):
            print(f"  {str(tt):20s}  {len(df[df['trial_type'] == tt])} trials")

    # -------------------------------------------------------------------------
    # Cross-tabs
    # -------------------------------------------------------------------------
    if "session" in df.columns and "trial_type" in df.columns:
        print_header("CROSS-TAB: Sessions x Trial Types")
        ct = pd.crosstab(df["session"], df["trial_type"], margins=True)
        print(ct.to_string())

    if "rat" in df.columns and "session" in df.columns:
        print_header("CROSS-TAB: Rats x Sessions")
        ct = pd.crosstab(df["rat"], df["session"], margins=True)
        print(ct.to_string())

    # -------------------------------------------------------------------------
    # All column stats
    # -------------------------------------------------------------------------
    print_header("ALL COLUMNS - UNIQUE VALUES")
    for col in df.columns:
        uniques = df[col].unique()
        if len(uniques) <= 30:
            print(f"  {col} ({len(uniques)} unique): {sorted(uniques, key=str)}")
        else:
            print(f"  {col} ({len(uniques)} unique): "
                  f"range=[{df[col].min()}, {df[col].max()}]")

    # =========================================================================
    #  POSSIBLE RUNS
    # =========================================================================
    print_header("POSSIBLE REGION-TO-REGION RUNS")

    print("""
  The PFC dataset has 2 brain regions recorded simultaneously:
    - PFC (Prefrontal Cortex): 64 channels  (indices 0-63)
    - CA1 (Hippocampus):       32 channels  (indices 64-95)

  Possible translation directions:
  ┌─────────────────────────────────────────────────────────────┐
  │  #   Source  →  Target    In-Ch   Out-Ch   Status          │
  ├─────────────────────────────────────────────────────────────┤
  │  1   PFC     →  CA1       64      32       SUPPORTED       │
  │  2   CA1     →  PFC       32      64       needs code mod  │
  └─────────────────────────────────────────────────────────────┘
""")

    # -------------------------------------------------------------------------
    # LOSO folds
    # -------------------------------------------------------------------------
    if sessions:
        print_header(f"LOSO FOLDS — Leave-One-Session-Out ({len(sessions)} folds)")
        print(f"  Direction: PFC → CA1\n")
        for i, s in enumerate(sessions):
            train_s = [x for x in sessions if x != s]
            test_n = len(df[df["session"] == s])
            train_n = len(df[df["session"].isin(train_s)])
            print(f"  Fold {i+1:2d}: hold-out={str(s):15s} "
                  f"test={test_n:3d} trials, train={train_n:3d} trials "
                  f"(from {len(train_s)} sessions)")

    # -------------------------------------------------------------------------
    # LORO folds
    # -------------------------------------------------------------------------
    if rats and len(rats) > 1:
        print_header(f"LORO FOLDS — Leave-One-Rat-Out ({len(rats)} folds)")
        print(f"  Direction: PFC → CA1\n")
        for i, r in enumerate(rats):
            train_r = [x for x in rats if x != r]
            sub = df[df["rat"] == r]
            test_n = len(sub)
            train_n = len(df[df["rat"].isin(train_r)])
            test_sess = sorted(sub["session"].unique()) if "session" in df.columns else []
            print(f"  Fold {i+1:2d}: hold-out=rat '{r}' "
                  f"({len(test_sess)} sessions: {test_sess})  "
                  f"test={test_n:3d} trials, train={train_n:3d} trials")

    # -------------------------------------------------------------------------
    # Total run count
    # -------------------------------------------------------------------------
    print_header("TOTAL RUN COUNT SUMMARY")
    n_directions = 1  # only PFC→CA1 supported out of the box
    n_loso = len(sessions) if sessions else 0
    n_loro = len(rats) if rats and len(rats) > 1 else 0

    print(f"  Region directions:         {n_directions} (PFC → CA1)")
    print(f"  LOSO folds (session):      {n_loso}")
    print(f"  LORO folds (rat/subject):  {n_loro}")
    print(f"  Random-split runs:         1 (70/15/15)")
    print(f"  Inter-session split runs:  1 (random session hold-out)")
    print(f"")
    total = n_directions * (n_loso + n_loro + 2)
    print(f"  ── Total distinct runs:    {total}")
    print(f"     = {n_directions} direction x "
          f"({n_loso} LOSO + {n_loro} LORO + 1 random + 1 inter-session)")


if __name__ == "__main__":
    main()
