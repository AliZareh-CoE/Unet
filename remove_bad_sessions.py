#!/usr/bin/env python3
"""
Remove Bad Sessions from Olfactory Dataset
===========================================
Permanently removes sessions 141208-1, 141208-2, 141209 from the dataset.

These sessions have been identified as outliers with poor cross-session
generalization performance (likely due to different recording conditions,
equipment issues, or experimental factors from December 2014).

This script:
1. Loads the original signals and metadata
2. Filters out the bad sessions
3. Saves cleaned data with '_cleaned' suffix
4. Optionally replaces original files (with backup)

Usage:
    python remove_bad_sessions.py           # Preview what will be removed
    python remove_bad_sessions.py --apply   # Actually remove and save
    python remove_bad_sessions.py --replace # Replace original files (makes backup)
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Sessions to remove (December 2014 recordings - outliers)
BAD_SESSIONS = ['141208-1', '141208-2', '141209']

# Data paths
DATA_DIR = Path("/data")
SIGNALS_PATH = DATA_DIR / "signal_windows_1khz.npy"
META_PATH = DATA_DIR / "signal_windows_meta_1khz.csv"

# Output paths
CLEANED_SIGNALS_PATH = DATA_DIR / "signal_windows_1khz_cleaned.npy"
CLEANED_META_PATH = DATA_DIR / "signal_windows_meta_1khz_cleaned.csv"

# Backup paths
BACKUP_SIGNALS_PATH = DATA_DIR / "signal_windows_1khz_original.npy"
BACKUP_META_PATH = DATA_DIR / "signal_windows_meta_1khz_original.csv"


def main():
    parser = argparse.ArgumentParser(description="Remove bad sessions from olfactory dataset")
    parser.add_argument("--apply", action="store_true", help="Save cleaned data (with _cleaned suffix)")
    parser.add_argument("--replace", action="store_true", help="Replace original files (makes backup first)")
    args = parser.parse_args()

    print("=" * 60)
    print("REMOVE BAD SESSIONS FROM OLFACTORY DATASET")
    print("=" * 60)
    print(f"\nSessions to remove: {BAD_SESSIONS}")

    # Load metadata
    print(f"\nLoading metadata from {META_PATH}...")
    df = pd.read_csv(META_PATH)
    print(f"  Total trials: {len(df)}")

    # Check session column
    if 'recording_id' not in df.columns:
        raise ValueError("Metadata must have 'recording_id' column")

    # Show session distribution before
    print("\nSession distribution BEFORE removal:")
    session_counts = df['recording_id'].value_counts().sort_index()
    for session, count in session_counts.items():
        marker = " <-- REMOVE" if session in BAD_SESSIONS else ""
        print(f"  {session}: {count} trials{marker}")

    # Find indices to remove
    mask_remove = df['recording_id'].isin(BAD_SESSIONS)
    n_remove = mask_remove.sum()
    print(f"\nTrials to remove: {n_remove}")

    # Filter metadata
    df_clean = df[~mask_remove].reset_index(drop=True)
    print(f"Trials remaining: {len(df_clean)}")

    # Load signals
    print(f"\nLoading signals from {SIGNALS_PATH}...")
    signals = np.load(SIGNALS_PATH)
    print(f"  Original shape: {signals.shape}")

    if len(signals) != len(df):
        raise ValueError(f"Signal count ({len(signals)}) != metadata count ({len(df)})")

    # Filter signals
    keep_indices = np.where(~mask_remove)[0]
    signals_clean = signals[keep_indices]
    print(f"  Cleaned shape: {signals_clean.shape}")

    # Show session distribution after
    print("\nSession distribution AFTER removal:")
    clean_counts = df_clean['recording_id'].value_counts().sort_index()
    for session, count in clean_counts.items():
        print(f"  {session}: {count} trials")

    print(f"\nSummary:")
    print(f"  Sessions removed: {BAD_SESSIONS}")
    print(f"  Trials removed: {n_remove}")
    print(f"  Trials remaining: {len(df_clean)} ({len(df_clean)/len(df)*100:.1f}%)")
    print(f"  Sessions remaining: {df_clean['recording_id'].nunique()}")

    if not args.apply and not args.replace:
        print("\n" + "=" * 60)
        print("DRY RUN - No files modified")
        print("Use --apply to save cleaned data (new files)")
        print("Use --replace to replace original files (with backup)")
        print("=" * 60)
        return

    if args.apply or args.replace:
        # Save cleaned data
        print(f"\nSaving cleaned signals to {CLEANED_SIGNALS_PATH}...")
        np.save(CLEANED_SIGNALS_PATH, signals_clean)

        print(f"Saving cleaned metadata to {CLEANED_META_PATH}...")
        df_clean.to_csv(CLEANED_META_PATH, index=False)

        print("Cleaned files saved!")

    if args.replace:
        # Backup original files
        print(f"\nBacking up original files...")

        if not BACKUP_SIGNALS_PATH.exists():
            print(f"  Backing up signals to {BACKUP_SIGNALS_PATH}...")
            shutil.copy2(SIGNALS_PATH, BACKUP_SIGNALS_PATH)
        else:
            print(f"  Backup already exists: {BACKUP_SIGNALS_PATH}")

        if not BACKUP_META_PATH.exists():
            print(f"  Backing up metadata to {BACKUP_META_PATH}...")
            shutil.copy2(META_PATH, BACKUP_META_PATH)
        else:
            print(f"  Backup already exists: {BACKUP_META_PATH}")

        # Replace original files
        print(f"\nReplacing original files...")
        shutil.copy2(CLEANED_SIGNALS_PATH, SIGNALS_PATH)
        shutil.copy2(CLEANED_META_PATH, META_PATH)

        print("Original files replaced!")
        print("\nTo restore original files:")
        print(f"  cp {BACKUP_SIGNALS_PATH} {SIGNALS_PATH}")
        print(f"  cp {BACKUP_META_PATH} {META_PATH}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
