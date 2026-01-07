#!/usr/bin/env python3
"""Download DANDI 000623 dataset (Human iEEG movie watching).

This script downloads the NWB files from the DANDI Archive using their API.

Usage:
    python experiment/download_dandi.py
    python experiment/download_dandi.py --output /path/to/output

Environment Variables:
    UNET_DATA_DIR: Base data directory (default: /data)
                   NWB files will be saved to $UNET_DATA_DIR/movie/

Reference:
    Keles et al., 2024, Scientific Data
    "Multimodal single-neuron, intracranial EEG, and fMRI brain responses
    during movie watching in human patients"
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError
import ssl


# DANDI API endpoints
DANDI_API_URL = "https://api.dandiarchive.org/api"
DANDISET_ID = "000623"


def get_dandiset_info(dandiset_id: str = DANDISET_ID) -> Dict[str, Any]:
    """Get dandiset metadata from DANDI API."""
    url = f"{DANDI_API_URL}/dandisets/{dandiset_id}/"

    # Handle SSL context
    context = ssl.create_default_context()

    try:
        with urlopen(url, context=context) as response:
            return json.loads(response.read().decode())
    except URLError as e:
        print(f"Error fetching dandiset info: {e}")
        raise


def list_assets(dandiset_id: str = DANDISET_ID, version: str = "draft") -> List[Dict[str, Any]]:
    """List all assets (files) in the dandiset."""
    assets = []
    url = f"{DANDI_API_URL}/dandisets/{dandiset_id}/versions/{version}/assets/"

    context = ssl.create_default_context()

    while url:
        try:
            with urlopen(url, context=context) as response:
                data = json.loads(response.read().decode())
                assets.extend(data.get("results", []))
                url = data.get("next")  # Pagination
        except URLError as e:
            print(f"Error listing assets: {e}")
            raise

    return assets


def get_asset_download_url(dandiset_id: str, version: str, asset_id: str) -> str:
    """Get the download URL for an asset."""
    url = f"{DANDI_API_URL}/dandisets/{dandiset_id}/versions/{version}/assets/{asset_id}/download/"
    return url


def download_file(url: str, output_path: Path, show_progress: bool = True) -> None:
    """Download a file with progress indication."""
    import urllib.request

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if show_progress:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                mb_downloaded = (count * block_size) / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urlretrieve(url, output_path, reporthook=progress_hook)
        print()  # New line after progress
    else:
        urlretrieve(url, output_path)


def download_dandi_dataset(
    output_dir: Path,
    dandiset_id: str = DANDISET_ID,
    version: str = "draft",
    file_pattern: str = ".nwb",
    max_files: int = None,
    skip_existing: bool = True,
) -> List[Path]:
    """Download DANDI dataset files.

    Args:
        output_dir: Directory to save downloaded files
        dandiset_id: DANDI dataset ID
        version: Dataset version ("draft" or specific version)
        file_pattern: Only download files matching this pattern
        max_files: Maximum number of files to download (None = all)
        skip_existing: Skip files that already exist

    Returns:
        List of downloaded file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching DANDI {dandiset_id} info...")
    info = get_dandiset_info(dandiset_id)
    print(f"  Name: {info.get('draft_version', {}).get('name', 'Unknown')}")

    print(f"\nListing assets...")
    assets = list_assets(dandiset_id, version)

    # Filter for NWB files
    nwb_assets = [a for a in assets if file_pattern in a.get("path", "")]
    print(f"  Found {len(nwb_assets)} NWB files (total: {len(assets)} assets)")

    if max_files:
        nwb_assets = nwb_assets[:max_files]
        print(f"  Limiting to first {max_files} files")

    downloaded = []
    total_size = sum(a.get("size", 0) for a in nwb_assets)
    print(f"\nTotal download size: {total_size / (1024**3):.2f} GB")

    for i, asset in enumerate(nwb_assets, 1):
        asset_path = asset.get("path", "")
        asset_id = asset.get("asset_id", "")
        asset_size = asset.get("size", 0)

        # Preserve directory structure
        output_path = output_dir / asset_path

        print(f"\n[{i}/{len(nwb_assets)}] {asset_path}")
        print(f"  Size: {asset_size / (1024**2):.1f} MB")

        if skip_existing and output_path.exists():
            existing_size = output_path.stat().st_size
            if existing_size == asset_size:
                print(f"  Skipping (already exists)")
                downloaded.append(output_path)
                continue
            else:
                print(f"  Re-downloading (size mismatch: {existing_size} vs {asset_size})")

        # Get download URL and download
        download_url = get_asset_download_url(dandiset_id, version, asset_id)

        try:
            download_file(download_url, output_path)
            downloaded.append(output_path)
            print(f"  Downloaded: {output_path}")
        except Exception as e:
            print(f"  Error downloading: {e}")

    print(f"\n{'='*60}")
    print(f"Downloaded {len(downloaded)} files to {output_dir}")

    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download DANDI 000623 dataset (Human iEEG movie watching)"
    )
    # Default output path uses UNET_DATA_DIR environment variable
    default_output = Path(os.environ.get("UNET_DATA_DIR", "/data")) / "movie"
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=default_output,
        help=f"Output directory for downloaded files (default: $UNET_DATA_DIR/movie = {default_output})"
    )
    parser.add_argument(
        "--max-files", "-n",
        type=int,
        default=None,
        help="Maximum number of files to download (default: all)"
    )
    parser.add_argument(
        "--version", "-v",
        default="draft",
        help="Dataset version (default: draft)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-download files even if they exist"
    )

    args = parser.parse_args()

    print("="*60)
    print("DANDI 000623 Dataset Downloader")
    print("Human iEEG during Movie Watching")
    print("="*60)

    download_dandi_dataset(
        output_dir=args.output,
        version=args.version,
        max_files=args.max_files,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
