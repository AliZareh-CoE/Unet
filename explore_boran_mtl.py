#!/usr/bin/env python3
"""Explore the Boran et al. MTL Working Memory dataset.

Dataset: Human medial temporal lobe neurons, scalp and intracranial EEG
         during a verbal working memory task (Sternberg task).
DOI: 10.12751/g-node.d76994
Paper: Boran et al., Sci Data 2020; 7:30

Expected structure: HDF5 (.h5) files in G-Node/NIX format, one per session.
Regions: hippocampus, entorhinal cortex, amygdala (depth electrodes) + scalp EEG

Usage:
    python explore_boran_mtl.py                       # default: /data/boran_mtl_wm
    python explore_boran_mtl.py /path/to/boran_data   # explicit path
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required. pip install numpy")
    sys.exit(1)

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("WARNING: h5py not installed. pip install h5py")


def print_header(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def explore_h5_structure(f, path="", max_depth=4, indent=0):
    """Recursively print HDF5 file structure."""
    items = []
    try:
        for key in f.keys():
            item = f[key]
            full_path = f"{path}/{key}"
            if isinstance(item, h5py.Group):
                n_children = len(item.keys())
                items.append(("group", key, full_path, n_children, None, None))
                if indent < max_depth:
                    sub_items = explore_h5_structure(item, full_path, max_depth, indent + 1)
                    items.extend([("  " + t[0], t[1], t[2], t[3], t[4], t[5]) for t in sub_items])
            elif isinstance(item, h5py.Dataset):
                items.append(("dataset", key, full_path, None, item.shape, item.dtype))
    except Exception as e:
        items.append(("error", str(e), path, None, None, None))
    return items


def print_h5_tree(f, path="", max_depth=3, indent=0, max_items=50):
    """Print HDF5 tree structure nicely."""
    count = 0
    try:
        for key in sorted(f.keys()):
            if count >= max_items:
                print(f"{'  ' * indent}  ... ({len(f.keys()) - count} more items)")
                break
            count += 1
            item = f[key]
            if isinstance(item, h5py.Group):
                n_children = len(item.keys())
                print(f"{'  ' * indent}[G] {key}/ ({n_children} children)")
                if indent < max_depth:
                    print_h5_tree(item, f"{path}/{key}", max_depth, indent + 1, max_items=20)
            elif isinstance(item, h5py.Dataset):
                shape_str = str(item.shape)
                dtype_str = str(item.dtype)
                size_mb = item.nbytes / (1024 * 1024) if hasattr(item, 'nbytes') else 0
                print(f"{'  ' * indent}[D] {key}: shape={shape_str}, dtype={dtype_str}"
                      f"{f', {size_mb:.1f}MB' if size_mb > 0.1 else ''}")
    except Exception as e:
        print(f"{'  ' * indent}  ERROR: {e}")


def extract_electrode_info(f):
    """Try to extract electrode locations and region labels from H5 file."""
    electrode_info = []

    # Common paths in NIX format for electrode metadata
    search_paths = [
        "metadata", "data", "electrodes", "channels",
        "intracranial", "iEEG", "scalp", "depth",
    ]

    def _search_for_labels(group, path=""):
        """Recursively search for region/label datasets."""
        results = {}
        try:
            for key in group.keys():
                item = group[key]
                k_lower = key.lower()
                if isinstance(item, h5py.Dataset):
                    # Look for region/label/anatomy/lobe related datasets
                    if any(kw in k_lower for kw in
                           ["label", "region", "anatomy", "lobe", "location",
                            "electrode", "channel", "mni", "coord"]):
                        try:
                            data = item[()]
                            results[f"{path}/{key}"] = data
                        except Exception:
                            pass
                elif isinstance(item, h5py.Group) and len(path.split("/")) < 5:
                    sub = _search_for_labels(item, f"{path}/{key}")
                    results.update(sub)
        except Exception:
            pass
        return results

    return _search_for_labels(f)


def extract_signal_datasets(f):
    """Find all signal/voltage/EEG datasets in the H5 file."""
    signals = []

    def _search(group, path=""):
        try:
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    # Look for large numerical datasets (likely signals)
                    if item.ndim >= 1 and item.shape[0] > 100:
                        if np.issubdtype(item.dtype, np.number):
                            signals.append({
                                "path": f"{path}/{key}",
                                "shape": item.shape,
                                "dtype": str(item.dtype),
                                "size_mb": item.nbytes / (1024 * 1024),
                            })
                elif isinstance(item, h5py.Group):
                    _search(item, f"{path}/{key}")
        except Exception:
            pass

    _search(f)
    return signals


def main():
    parser = argparse.ArgumentParser(description="Explore Boran MTL WM dataset")
    parser.add_argument("data_dir", nargs="?", default="/data/boran_mtl_wm",
                        help="Path to dataset directory")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Max depth for H5 tree printing (default: 3)")
    parser.add_argument("--full", action="store_true",
                        help="Print full H5 structure (deeper, more items)")
    args = parser.parse_args()

    root = Path(args.data_dir)

    print_header("BORAN MTL WORKING MEMORY DATASET EXPLORATION")
    print(f"  Data directory: {root}")

    if not root.exists():
        print(f"\n  ERROR: Directory not found: {root}")
        sys.exit(1)

    # List all files
    print_header("FILES")
    all_files = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            rel = p.relative_to(root)
            all_files.append((rel, size_mb, p))
            if size_mb > 0.01:
                print(f"  {str(rel):60s} {size_mb:10.2f} MB")

    # Summary by extension
    by_ext = defaultdict(lambda: {"count": 0, "size_mb": 0})
    for rel, size, _ in all_files:
        ext = rel.suffix.lower() or "(no ext)"
        by_ext[ext]["count"] += 1
        by_ext[ext]["size_mb"] += size

    print(f"\n  File type summary:")
    for ext in sorted(by_ext.keys()):
        info = by_ext[ext]
        print(f"    {ext:10s}  {info['count']:>4} files  {info['size_mb']:>10.1f} MB total")

    # Find H5 files
    h5_files = sorted([p for _, _, p in all_files if p.suffix.lower() in (".h5", ".hdf5", ".nix")])

    if not h5_files and not HAS_H5PY:
        print("\n  No .h5 files found AND h5py not installed.")
        print("  Install h5py: pip install h5py")
        # Try to find other data formats
        mat_files = [p for _, _, p in all_files if p.suffix.lower() in (".mat", ".npy", ".npz")]
        csv_files = [p for _, _, p in all_files if p.suffix.lower() in (".csv", ".tsv")]
        print(f"\n  .mat files: {len(mat_files)}")
        print(f"  .csv/.tsv files: {len(csv_files)}")
        for f in csv_files[:5]:
            print(f"    {f.relative_to(root)}")
        sys.exit(0)

    if not HAS_H5PY:
        print(f"\n  Found {len(h5_files)} H5 files but h5py is not installed.")
        print("  Install: pip install h5py")
        sys.exit(1)

    print_header(f"HDF5 FILES ({len(h5_files)} found)")

    # Explore each H5 file
    subjects_info = {}
    all_regions = set()
    all_signals_summary = []

    for h5_path in h5_files:
        rel = h5_path.relative_to(root)
        size_mb = h5_path.stat().st_size / (1024 * 1024)

        print(f"\n{'─' * 80}")
        print(f"  File: {rel} ({size_mb:.1f} MB)")
        print(f"{'─' * 80}")

        try:
            with h5py.File(h5_path, "r") as f:
                # Print top-level structure
                print(f"\n  Top-level keys: {list(f.keys())}")

                # Print tree
                max_d = 5 if args.full else args.max_depth
                print(f"\n  Structure (depth={max_d}):")
                print_h5_tree(f, max_depth=max_d, max_items=30 if args.full else 15)

                # Print all attributes at top level
                if f.attrs:
                    print(f"\n  Root attributes:")
                    for attr_name in sorted(f.attrs.keys()):
                        val = f.attrs[attr_name]
                        if isinstance(val, bytes):
                            val = val.decode("utf-8", errors="replace")
                        print(f"    {attr_name}: {val}")

                # Search for electrode/region info
                label_data = extract_electrode_info(f)
                if label_data:
                    print(f"\n  Electrode/Region metadata found:")
                    for path, data in sorted(label_data.items()):
                        if isinstance(data, np.ndarray):
                            if data.dtype.kind in ('U', 'S', 'O'):
                                unique_vals = list(set(
                                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                                    for v in data.flat
                                ))[:20]
                                print(f"    {path}: {len(data)} entries, unique={unique_vals}")
                                all_regions.update(unique_vals)
                            else:
                                print(f"    {path}: shape={data.shape}, dtype={data.dtype}")
                        else:
                            s = str(data)
                            if isinstance(data, bytes):
                                s = data.decode("utf-8", errors="replace")
                            print(f"    {path}: {s[:100]}")

                # Find signal datasets
                signals = extract_signal_datasets(f)
                if signals:
                    print(f"\n  Signal datasets (numerical, >100 samples):")
                    for sig in sorted(signals, key=lambda x: -x["size_mb"])[:20]:
                        print(f"    {sig['path']}: shape={sig['shape']}, "
                              f"dtype={sig['dtype']}, {sig['size_mb']:.1f} MB")
                    all_signals_summary.append({
                        "file": str(rel),
                        "signals": signals,
                    })

                # Try to identify subject from filename or metadata
                fname = h5_path.stem.lower()
                subjects_info[str(rel)] = {
                    "size_mb": size_mb,
                    "n_signals": len(signals),
                    "n_label_fields": len(label_data),
                }

        except Exception as e:
            print(f"  ERROR reading file: {e}")

    # Global summary
    print_header("GLOBAL SUMMARY")
    print(f"  Total H5 files: {len(h5_files)}")
    print(f"  Total data size: {sum(s['size_mb'] for s in subjects_info.values()):.1f} MB")

    if all_regions:
        print(f"\n  All region/electrode labels found across files:")
        for r in sorted(all_regions):
            print(f"    - {r}")

    # LOSO viability assessment
    print_header("LOSO VIABILITY ASSESSMENT")
    print(f"  Subjects/sessions: {len(h5_files)}")
    if all_regions:
        print(f"  Regions found: {sorted(all_regions)}")
    print(f"\n  For inter-region translation LOSO, we need:")
    print(f"    - >= 3 subjects")
    print(f"    - >= 2 brain regions with multiple channels each")
    print(f"    - Simultaneous recordings from both regions")

    if len(h5_files) >= 3:
        print(f"\n  ✓ Subject count ({len(h5_files)}) meets minimum for LOSO")
    else:
        print(f"\n  ✗ Only {len(h5_files)} files — may need >= 3 for LOSO")

    print(f"\n  Next steps:")
    print(f"    1. Run this script and share the output")
    print(f"    2. Identify which iEEG channels map to which MTL subregions")
    print(f"    3. Check channel counts per region per subject")
    print(f"    4. Determine sampling rate and trial structure")


if __name__ == "__main__":
    main()
