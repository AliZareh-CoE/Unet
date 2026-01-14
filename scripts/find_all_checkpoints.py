#!/usr/bin/env python3
"""Find all checkpoint/model files that might contain trained models."""

import os
import sys
from pathlib import Path

def main():
    search_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    print(f"Searching in: {search_dir.absolute()}")
    print("="*60)

    # Find all potential model files
    extensions = ['.pkl', '.pt', '.pth', '.ckpt', '.checkpoint']

    found = []
    for ext in extensions:
        for f in search_dir.rglob(f"*{ext}"):
            size_mb = f.stat().st_size / (1024*1024)
            found.append((f, size_mb))

    # Group by directory
    by_dir = {}
    for f, size in sorted(found):
        d = str(f.parent)
        if d not in by_dir:
            by_dir[d] = []
        by_dir[d].append((f.name, size))

    for d, files in sorted(by_dir.items()):
        print(f"\n{d}/")
        for name, size in files:
            print(f"  {name:<50} {size:>8.2f} MB")

    print(f"\n{'='*60}")
    print(f"Total files found: {len(found)}")

    # Look specifically for fnet
    print(f"\n{'='*60}")
    print("Files containing 'fnet':")
    for f, size in found:
        if 'fnet' in str(f).lower():
            print(f"  {f} ({size:.2f} MB)")

if __name__ == "__main__":
    main()
