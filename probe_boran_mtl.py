#!/usr/bin/env python3
"""Probe Boran MTL dataset: extract electrode regions, channel counts, trial counts.

Memory-efficient: reads ONLY metadata and electrode info, never loads signal data.

Dataset: Boran et al., Sci Data 2020; 7:30
DOI: 10.12751/g-node.d76994
Regions: hippocampus, entorhinal cortex, amygdala (depth electrodes) + scalp EEG

Electrode naming convention from the paper:
  AH  = Anterior Hippocampus       PH  = Posterior Hippocampus
  EC  = Entorhinal Cortex          A   = Amygdala
  L/R = Left/Right hemisphere
  _N  = contact number on the depth electrode

  Examples: AHL_5 = Anterior Hippocampus Left contact 5
            ECR_3 = Entorhinal Cortex Right contact 3

Usage:
    python probe_boran_mtl.py                          # default: /data/boran_mtl_wm
    python probe_boran_mtl.py /path/to/boran_data      # explicit path
"""

import os
import sys
import re
import gc
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import h5py
except ImportError:
    print("ERROR: h5py required. pip install h5py")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required. pip install numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
#  Region parsing
# ---------------------------------------------------------------------------
# Map electrode name prefixes to canonical region names
# Prefix matching is done longest-first to avoid 'A' swallowing 'AH', 'AL', etc.
REGION_MAP = {
    "AH":  "hippocampus",      # Anterior Hippocampus
    "PH":  "hippocampus",      # Posterior Hippocampus
    "EC":  "entorhinal_cortex",
    "AL":  "amygdala",         # Amygdala Left (match before generic 'A')
    "AR":  "amygdala",         # Amygdala Right
    "A":   "amygdala",         # Amygdala (generic, if no L/R suffix)
    "TB":  "temporal_basal",   # sometimes present
    "PHC": "parahippocampal",  # sometimes present
    "DR":  "other",            # e.g. DRR in S05 — possibly a non-MTL probe
    "LR":  "other",            # e.g. LR in S01 — possibly a non-MTL probe
}

# More specific sub-region mapping (for detailed analysis)
SUBREGION_MAP = {
    "AHL": "ant_hippocampus_L",
    "AHR": "ant_hippocampus_R",
    "PHL": "post_hippocampus_L",
    "PHR": "post_hippocampus_R",
    "ECL": "entorhinal_cortex_L",
    "ECR": "entorhinal_cortex_R",
    "AL":  "amygdala_L",
    "AR":  "amygdala_R",
    "TBL": "temporal_basal_L",
    "TBR": "temporal_basal_R",
    "DRR": "other_DRR",
    "LR":  "other_LR",
}


def parse_electrode_region(name):
    """Parse an electrode/probe name like 'uAHL_5', 'AHL', 'ECR' into region info.

    Handles both individual electrode names (with contact number) and bare
    probe abbreviations from the metadata "Depth electrodes" field.

    Returns (canonical_region, subregion, hemisphere, contact_num, raw_prefix).
    """
    # Strip leading 'u' (micro-electrode marker) or 'i' prefix
    clean = name.strip()
    if clean.startswith("u"):
        clean = clean[1:]

    # Try to match pattern: PREFIX or PREFIX_NUMBER
    m = re.match(r"([A-Za-z]+)[_\s]*(\d+)?", clean)
    if not m:
        return ("unknown", "unknown", "?", None, clean)

    prefix = m.group(1).upper()
    contact = int(m.group(2)) if m.group(2) else None

    # Hemisphere from last char
    hemisphere = "?"
    if prefix.endswith("L"):
        hemisphere = "L"
    elif prefix.endswith("R"):
        hemisphere = "R"

    # Sub-region: try exact match first
    subregion = SUBREGION_MAP.get(prefix, f"unknown_{prefix}")

    # Canonical region: match longest prefix first to avoid 'A' swallowing 'AH'/'AL'/'AR'
    # Also try exact match first (e.g. 'AL' -> amygdala, not 'A' -> amygdala)
    region = "unknown"
    if prefix in REGION_MAP:
        region = REGION_MAP[prefix]
    else:
        for pfx in sorted(REGION_MAP.keys(), key=len, reverse=True):
            if prefix.startswith(pfx):
                region = REGION_MAP[pfx]
                break

    return (region, subregion, hemisphere, contact, prefix)


def decode_bytes(val):
    """Decode bytes to string if needed."""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.bytes_):
        return val.decode("utf-8", errors="replace")
    return str(val)


def extract_first_string(val):
    """Extract the first string/bytes value from a metadata field.

    NIX metadata properties are stored as HDF5 compound datasets. When read
    with h5py, they come back as numpy void scalars (np.void) with fields like
    (b'AHL,AL,ECL,...', 0., b'', b'', b'', b'').

    The actual content is always in the first field. This function robustly
    extracts it regardless of the numpy type.
    """
    # numpy void (structured scalar from compound dataset)
    if isinstance(val, np.void):
        try:
            first = val[0]
            if isinstance(first, (bytes, np.bytes_)):
                return first.decode("utf-8", errors="replace")
            return str(first)
        except (IndexError, TypeError):
            pass

    # Regular tuple/list
    if isinstance(val, (tuple, list)):
        if len(val) > 0:
            first = val[0]
            if isinstance(first, (bytes, np.bytes_)):
                return first.decode("utf-8", errors="replace")
            return str(first)

    # numpy array
    if isinstance(val, np.ndarray):
        if val.dtype.names:
            # Structured array — get first field of first element
            try:
                first = val.flat[0][0]
                if isinstance(first, (bytes, np.bytes_)):
                    return first.decode("utf-8", errors="replace")
                return str(first)
            except (IndexError, TypeError):
                pass
        elif val.size >= 1:
            first = val.flat[0]
            if isinstance(first, (bytes, np.bytes_)):
                return first.decode("utf-8", errors="replace")
            return str(first)

    # Plain bytes
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")

    # Plain string
    if isinstance(val, str):
        return val

    # Fallback
    return str(val)


# ---------------------------------------------------------------------------
#  NIX H5 metadata extraction (no signal data loaded)
# ---------------------------------------------------------------------------

def extract_subject_metadata(f):
    """Extract subject demographics from NIX metadata sections."""
    info = {}

    def _scan_attrs(group, path=""):
        """Scan group attributes for subject info."""
        try:
            for attr_name in group.attrs:
                val = group.attrs[attr_name]
                val_str = decode_bytes(val) if isinstance(val, (bytes, np.bytes_)) else val
                key_lower = attr_name.lower()
                if any(k in key_lower for k in ["age", "sex", "hand", "pathol", "soz",
                                                  "depth", "subject", "patient"]):
                    info[attr_name] = val_str
        except Exception:
            pass

    def _scan_metadata(group, path="", depth=0):
        """Recursively scan metadata sections."""
        if depth > 6:
            return
        try:
            _scan_attrs(group, path)
            # In NIX format, metadata properties are stored as datasets
            for key in group.keys():
                item = group[key]
                k_lower = key.lower()
                if isinstance(item, h5py.Dataset):
                    if any(kw in k_lower for kw in ["age", "sex", "hand", "pathol",
                                                     "soz", "depth", "subject",
                                                     "patient", "hemisphere"]):
                        try:
                            val = item[()]
                            if isinstance(val, np.ndarray) and val.size == 1:
                                val = val.flat[0]
                            info[f"{path}/{key}"] = decode_bytes(val) if isinstance(val, (bytes, np.bytes_)) else val
                        except Exception:
                            pass
                elif isinstance(item, h5py.Group):
                    _scan_metadata(item, f"{path}/{key}", depth + 1)
        except Exception:
            pass

    # Scan metadata section
    if "metadata" in f:
        _scan_metadata(f["metadata"], "metadata")

    return info


def extract_electrode_names_from_sources(f):
    """Extract iEEG electrode names from NIX sources.

    In NIX format, sources represent recorded entities. Each iEEG electrode
    is a source with a name like 'iEEG electrode 1' and type 'neo.recordingchannel'.
    The actual electrode label (e.g., 'AHL_5') is stored in properties.
    """
    electrodes = []

    def _scan_sources(group, path="", depth=0):
        if depth > 5:
            return
        try:
            for key in group.keys():
                item = group[key]
                if not isinstance(item, h5py.Group):
                    continue

                # Check if this is an electrode source
                item_name = ""
                item_type = ""
                try:
                    if "name" in item.attrs:
                        item_name = decode_bytes(item.attrs["name"])
                    elif "name" in item:
                        d = item["name"]
                        if isinstance(d, h5py.Dataset):
                            item_name = decode_bytes(d[()])
                except Exception:
                    pass

                try:
                    if "type" in item.attrs:
                        item_type = decode_bytes(item.attrs["type"])
                    elif "type" in item:
                        d = item["type"]
                        if isinstance(d, h5py.Dataset):
                            item_type = decode_bytes(d[()])
                except Exception:
                    pass

                # Look for electrode label in properties or sub-datasets
                label = ""
                props = {}
                try:
                    # NIX stores properties in a 'properties' sub-group
                    # or as direct datasets
                    for sub_key in item.keys():
                        sub_item = item[sub_key]
                        if isinstance(sub_item, h5py.Dataset):
                            sub_lower = sub_key.lower()
                            if sub_lower in ("label", "name", "electrode_label"):
                                try:
                                    val = sub_item[()]
                                    if isinstance(val, np.ndarray) and val.size == 1:
                                        val = val.flat[0]
                                    label = decode_bytes(val) if isinstance(val, (bytes, np.bytes_)) else str(val)
                                except Exception:
                                    pass
                            # Collect other small properties
                            try:
                                if sub_item.size <= 10:
                                    val = sub_item[()]
                                    if isinstance(val, np.ndarray) and val.size == 1:
                                        val = val.flat[0]
                                    props[sub_key] = decode_bytes(val) if isinstance(val, (bytes, np.bytes_)) else val
                            except Exception:
                                pass
                        elif isinstance(sub_item, h5py.Group):
                            # Recurse one more level for properties
                            try:
                                for pk in sub_item.keys():
                                    pi = sub_item[pk]
                                    if isinstance(pi, h5py.Dataset) and pi.size <= 10:
                                        val = pi[()]
                                        if isinstance(val, np.ndarray) and val.size == 1:
                                            val = val.flat[0]
                                        props[pk] = decode_bytes(val) if isinstance(val, (bytes, np.bytes_)) else val
                            except Exception:
                                pass
                except Exception:
                    pass

                # Check attrs for properties too
                try:
                    for attr_name in item.attrs:
                        props[attr_name] = decode_bytes(item.attrs[attr_name]) if isinstance(
                            item.attrs[attr_name], (bytes, np.bytes_)) else item.attrs[attr_name]
                except Exception:
                    pass

                is_electrode = ("ieeg" in item_name.lower() or
                                "electrode" in item_name.lower() or
                                "electrode" in item_type.lower() or
                                "recording" in item_type.lower())

                if is_electrode:
                    electrodes.append({
                        "h5_name": item_name,
                        "h5_type": item_type,
                        "label": label,
                        "path": f"{path}/{key}",
                        "props": props,
                    })
                else:
                    # Recurse into sub-sources
                    _scan_sources(item, f"{path}/{key}", depth + 1)
        except Exception:
            pass

    if "data" in f:
        data_grp = f["data"]
        # NIX sources are under data/sources or data/*/sources
        if "sources" in data_grp:
            _scan_sources(data_grp["sources"], "data/sources")
        # Also check top-level blocks
        for block_key in data_grp.keys():
            block = data_grp[block_key]
            if isinstance(block, h5py.Group) and "sources" in block:
                _scan_sources(block["sources"], f"data/{block_key}/sources")

    # Also check root-level sources
    if "sources" in f:
        _scan_sources(f["sources"], "sources")

    return electrodes


def extract_electrode_names_from_multitags(f):
    """Extract electrode names from multi_tag names (spike data).

    Multi-tags for spike sorting often have names like 'uAHL_5' which
    directly encode the electrode region.
    """
    names = set()
    try:
        if "data" not in f:
            return names
        data_grp = f["data"]

        # Look for multi_tags
        for key in data_grp.keys():
            item = data_grp[key]
            if not isinstance(item, h5py.Group):
                continue

            # Check if this group has multi_tags
            mt_group = None
            if "multi_tags" in item:
                mt_group = item["multi_tags"]
            elif key == "multi_tags":
                mt_group = item

            if mt_group is None:
                continue

            for mt_key in mt_group.keys():
                mt = mt_group[mt_key]
                mt_name = ""
                try:
                    if "name" in mt.attrs:
                        mt_name = decode_bytes(mt.attrs["name"])
                    elif isinstance(mt, h5py.Group) and "name" in mt:
                        d = mt["name"]
                        if isinstance(d, h5py.Dataset):
                            mt_name = decode_bytes(d[()])
                except Exception:
                    pass

                if not mt_name:
                    mt_name = mt_key

                # Check if name looks like an electrode name (e.g. uAHL_5)
                if re.match(r"[ui]?[A-Z]{1,4}[LR]?[_\s]?\d+", mt_name):
                    names.add(mt_name)

    except Exception:
        pass
    return names


def count_trials(f):
    """Count number of trials by looking at data array names (without loading data)."""
    n_scalp_trials = 0
    n_ieeg_trials = 0
    scalp_shape = None
    ieeg_shape = None
    sampling_rate = None

    try:
        if "data" not in f:
            return n_scalp_trials, n_ieeg_trials, scalp_shape, ieeg_shape, sampling_rate

        data_grp = f["data"]

        # Look for data_arrays in NIX blocks
        for block_key in data_grp.keys():
            block = data_grp[block_key]
            if not isinstance(block, h5py.Group):
                continue

            da_group = None
            if "data_arrays" in block:
                da_group = block["data_arrays"]
            elif block_key == "data_arrays":
                da_group = block

            if da_group is None:
                continue

            for da_key in da_group.keys():
                da = da_group[da_key]
                da_name = ""
                try:
                    if "name" in da.attrs:
                        da_name = decode_bytes(da.attrs["name"])
                    elif isinstance(da, h5py.Group) and "name" in da:
                        d = da["name"]
                        if isinstance(d, h5py.Dataset):
                            da_name = decode_bytes(d[()])
                except Exception:
                    da_name = da_key

                name_lower = da_name.lower()

                if "scalp" in name_lower and "trial" in name_lower:
                    n_scalp_trials += 1
                    # Get shape without loading data
                    if scalp_shape is None:
                        try:
                            if isinstance(da, h5py.Group) and "data" in da:
                                scalp_shape = da["data"].shape
                            elif isinstance(da, h5py.Dataset):
                                scalp_shape = da.shape
                        except Exception:
                            pass
                elif "ieeg" in name_lower and "trial" in name_lower:
                    n_ieeg_trials += 1
                    if ieeg_shape is None:
                        try:
                            if isinstance(da, h5py.Group) and "data" in da:
                                ieeg_shape = da["data"].shape
                            elif isinstance(da, h5py.Dataset):
                                ieeg_shape = da.shape
                        except Exception:
                            pass

                # Look for sampling rate in dimensions
                if sampling_rate is None:
                    try:
                        if isinstance(da, h5py.Group) and "dimensions" in da:
                            dims = da["dimensions"]
                            for dk in dims.keys():
                                dim = dims[dk]
                                if isinstance(dim, h5py.Group):
                                    for prop_key in dim.keys():
                                        if "sampling" in prop_key.lower() or "rate" in prop_key.lower():
                                            val = dim[prop_key][()]
                                            if isinstance(val, np.ndarray) and val.size == 1:
                                                val = val.flat[0]
                                            if isinstance(val, (int, float, np.integer, np.floating)) and val > 0:
                                                sampling_rate = float(val)
                                    # Also check attrs
                                    for attr in dim.attrs:
                                        if "sampling" in attr.lower() or "interval" in attr.lower():
                                            val = dim.attrs[attr]
                                            if isinstance(val, (int, float, np.integer, np.floating)):
                                                if "interval" in attr.lower() and val > 0:
                                                    sampling_rate = 1.0 / val
                                                else:
                                                    sampling_rate = float(val)
                    except Exception:
                        pass

    except Exception:
        pass

    return n_scalp_trials, n_ieeg_trials, scalp_shape, ieeg_shape, sampling_rate


def parse_depth_electrodes_metadata(meta_dict):
    """Extract depth electrode probe list from metadata.

    The metadata contains entries like:
      'metadata/Subject/properties/Depth electrodes':
          np.void(b'AHL,AL,ECL,LR,PHL,PHR', 0., b'', ...)

    The actual content string is always in the first field of the compound type.
    Returns a list of probe abbreviations, e.g. ['AHL', 'AL', 'ECL', 'LR', 'PHL', 'PHR'].
    """
    for key, val in meta_dict.items():
        if "depth electrode" in key.lower():
            raw = extract_first_string(val).strip()
            if raw:
                probes = [p.strip().upper() for p in raw.split(",") if p.strip()]
                return probes
    return []


def assign_channels_to_probes(probes, n_channels):
    """Distribute iEEG channels across depth electrode probes.

    Boran dataset uses 8 contacts per probe typically.
    Returns dict: {probe_abbreviation: n_contacts}.
    """
    if not probes or n_channels == 0:
        return {}

    n_probes = len(probes)
    base_contacts = n_channels // n_probes
    remainder = n_channels % n_probes

    assignment = {}
    for i, probe in enumerate(probes):
        # Distribute remainder to first probes
        assignment[probe] = base_contacts + (1 if i < remainder else 0)

    return assignment


def probe_single_file(h5_path):
    """Probe a single H5 file for electrode and trial metadata.

    Returns a dict with all extracted info, or None on error.
    """
    result = {
        "path": str(h5_path),
        "filename": h5_path.name,
        "size_mb": h5_path.stat().st_size / (1024 * 1024),
        "subject": None,
        "session": None,
        "subject_meta": {},
        "electrodes": [],
        "multitag_electrodes": set(),
        "depth_probes": [],       # e.g. ['AHL', 'AL', 'ECL', ...]
        "probe_channels": {},     # e.g. {'AHL': 8, 'AL': 8, ...}
        "n_scalp_trials": 0,
        "n_ieeg_trials": 0,
        "scalp_shape": None,
        "ieeg_shape": None,
        "sampling_rate": None,
        "region_counts": defaultdict(int),
        "subregion_counts": defaultdict(int),
        "error": None,
    }

    # Parse subject/session from filename: Data_Subject_XX_Session_YY.h5
    m = re.match(r".*Subject[_\s]*(\d+).*Session[_\s]*(\d+)", h5_path.stem, re.IGNORECASE)
    if m:
        result["subject"] = f"S{int(m.group(1)):02d}"
        result["session"] = int(m.group(2))

    try:
        with h5py.File(h5_path, "r") as f:
            # Subject metadata
            result["subject_meta"] = extract_subject_metadata(f)

            # Electrodes from sources (for counting total iEEG channels)
            result["electrodes"] = extract_electrode_names_from_sources(f)

            # Electrodes from multitag names (spike sorting)
            result["multitag_electrodes"] = extract_electrode_names_from_multitags(f)

            # Trial counts and signal shapes
            (result["n_scalp_trials"], result["n_ieeg_trials"],
             result["scalp_shape"], result["ieeg_shape"],
             result["sampling_rate"]) = count_trials(f)

    except Exception as e:
        result["error"] = str(e)
        return result

    # --- KEY FIX: Parse region labels from "Depth electrodes" metadata ---
    result["depth_probes"] = parse_depth_electrodes_metadata(result["subject_meta"])

    # Get total iEEG channel count from data shape or source count
    n_ieeg_channels = 0
    if result["ieeg_shape"]:
        n_ieeg_channels = result["ieeg_shape"][0]  # shape is (channels, timepoints)
    elif result["electrodes"]:
        n_ieeg_channels = len(result["electrodes"])

    # Distribute channels across probes
    if result["depth_probes"] and n_ieeg_channels > 0:
        result["probe_channels"] = assign_channels_to_probes(
            result["depth_probes"], n_ieeg_channels)

        # Map probes to regions
        for probe, n_contacts in result["probe_channels"].items():
            region, subregion, hemi, _, _ = parse_electrode_region(probe)
            result["region_counts"][region] += n_contacts
            result["subregion_counts"][subregion] += n_contacts

    # Force garbage collection after closing file
    gc.collect()

    return result


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Probe Boran MTL dataset for electrode regions and LOSO viability")
    parser.add_argument("data_dir", nargs="?", default="/data/boran_mtl_wm",
                        help="Path to dataset directory")
    args = parser.parse_args()

    root = Path(args.data_dir)

    print("=" * 90)
    print("  BORAN MTL WORKING MEMORY — ELECTRODE & REGION PROBE")
    print(f"  Data directory: {root}")
    print("=" * 90)

    if not root.exists():
        print(f"\n  ERROR: Directory not found: {root}")
        sys.exit(1)

    # Find H5 files
    h5_files = sorted(root.rglob("*.h5"))
    if not h5_files:
        h5_files = sorted(root.rglob("*.hdf5")) + sorted(root.rglob("*.nix"))

    print(f"\n  Found {len(h5_files)} H5 files")

    if not h5_files:
        print("  No H5 files found!")
        sys.exit(1)

    # Probe each file
    all_results = []
    subjects = defaultdict(list)  # subject_id -> list of session results

    for i, h5_path in enumerate(h5_files):
        rel = h5_path.relative_to(root) if h5_path.is_relative_to(root) else h5_path
        print(f"\n  [{i+1}/{len(h5_files)}] Probing {rel} ...", end="", flush=True)

        result = probe_single_file(h5_path)
        all_results.append(result)

        if result["error"]:
            print(f" ERROR: {result['error']}")
            continue

        subj = result["subject"] or f"unknown_{i}"
        subjects[subj].append(result)

        n_ch = result["ieeg_shape"][0] if result["ieeg_shape"] else 0
        probes_str = ",".join(result["depth_probes"]) if result["depth_probes"] else "?"
        region_str = {k: v for k, v in result["region_counts"].items() if k != "unknown"}
        print(f" OK  (subj={subj}, sess={result['session']}, "
              f"ieeg_ch={n_ch}, probes=[{probes_str}], "
              f"trials={result['n_ieeg_trials']}, "
              f"regions={region_str or dict(result['region_counts'])})")

    # =========================================================================
    #  PER-SUBJECT SUMMARY
    # =========================================================================
    print(f"\n\n{'=' * 90}")
    print(f"  PER-SUBJECT SUMMARY ({len(subjects)} subjects)")
    print(f"{'=' * 90}")

    subject_region_counts = {}  # subject -> {region: max_channels}

    for subj in sorted(subjects.keys()):
        sessions = subjects[subj]
        print(f"\n  {subj}:")
        print(f"    Sessions: {len(sessions)}")

        # Aggregate region counts across sessions (use max per session)
        merged_regions = defaultdict(int)
        merged_subregions = defaultdict(int)
        total_trials_scalp = 0
        total_trials_ieeg = 0
        sr = None

        for sess in sessions:
            for region, count in sess["region_counts"].items():
                merged_regions[region] = max(merged_regions[region], count)
            for subregion, count in sess["subregion_counts"].items():
                merged_subregions[subregion] = max(merged_subregions[subregion], count)
            total_trials_scalp += sess["n_scalp_trials"]
            total_trials_ieeg += sess["n_ieeg_trials"]
            if sess["sampling_rate"] and sr is None:
                sr = sess["sampling_rate"]

            # Print session details
            print(f"    Session {sess['session']}: "
                  f"scalp_trials={sess['n_scalp_trials']}, "
                  f"ieeg_trials={sess['n_ieeg_trials']}, "
                  f"ieeg_shape={sess['ieeg_shape']}, "
                  f"sr={sess['sampling_rate']}")

        subject_region_counts[subj] = dict(merged_regions)

        print(f"    Total trials: scalp={total_trials_scalp}, ieeg={total_trials_ieeg}")
        if sr:
            print(f"    Sampling rate: {sr} Hz")

        print(f"    Region channels (max across sessions):")
        for region in sorted(merged_regions.keys()):
            print(f"      {region:25s}  {merged_regions[region]:3d} channels")

        if merged_subregions:
            print(f"    Sub-region detail:")
            for subr in sorted(merged_subregions.keys()):
                print(f"      {subr:25s}  {merged_subregions[subr]:3d} channels")

        # Print depth electrode probes
        probes = sessions[0].get("depth_probes", [])
        if probes:
            print(f"    Depth probes: {', '.join(probes)}")

        # Print key subject metadata
        meta = sessions[0].get("subject_meta", {})
        if meta:
            print(f"    Metadata:")
            for k, v in sorted(meta.items()):
                short_key = k.split("/")[-1] if "/" in k else k
                clean_v = extract_first_string(v)
                print(f"      {short_key}: {clean_v}")

    # =========================================================================
    #  DEPTH ELECTRODE PROBE MAPPING (from first session of each subject)
    # =========================================================================
    print(f"\n\n{'=' * 90}")
    print(f"  DEPTH ELECTRODE PROBE → REGION MAPPING")
    print(f"{'=' * 90}")

    for subj in sorted(subjects.keys()):
        sess = subjects[subj][0]  # first session
        n_ch = sess["ieeg_shape"][0] if sess["ieeg_shape"] else len(sess["electrodes"])
        probes = sess["depth_probes"]
        probe_ch = sess["probe_channels"]

        print(f"\n  {subj}: {n_ch} total iEEG channels, {len(probes)} probes")
        print(f"    Depth electrodes field: {','.join(probes)}")

        if probe_ch:
            ch_per = n_ch / len(probes) if probes else 0
            print(f"    Contacts per probe: ~{ch_per:.1f}")
            print(f"    {'Probe':<8s} {'Contacts':>10s}  {'Region':<22s} {'Sub-region':<25s} {'Hemi'}")
            print(f"    {'─'*8} {'─'*10}  {'─'*22} {'─'*25} {'─'*4}")
            for probe in probes:
                nc = probe_ch.get(probe, 0)
                region, subregion, hemi, _, _ = parse_electrode_region(probe)
                print(f"    {probe:<8s} {nc:>10d}  {region:<22s} {subregion:<25s} {hemi}")

        # Also show multitag electrode names if found (spike-sorted units)
        if sess["multitag_electrodes"]:
            print(f"    Spike-sorted units from multitags ({len(sess['multitag_electrodes'])}):")
            for name in sorted(sess["multitag_electrodes"]):
                region, subregion, hemi, contact, prefix = parse_electrode_region(name)
                print(f"      {name:20s}  -> {region} ({subregion})")

    # =========================================================================
    #  REGION-TO-REGION VIABILITY TABLE
    # =========================================================================
    MIN_CHANNELS = 2  # depth electrodes have fewer channels than ECoG
    MIN_SUBJECTS = 3

    # Canonical regions of interest
    REGIONS_OF_INTEREST = ["hippocampus", "entorhinal_cortex", "amygdala"]

    # Collect all regions found (including 'other')
    all_found_regions = set()
    for rc in subject_region_counts.values():
        all_found_regions.update(rc.keys())
    extra_regions = sorted(all_found_regions - set(REGIONS_OF_INTEREST) - {"unknown"})
    display_regions = REGIONS_OF_INTEREST + extra_regions

    print(f"\n\n{'=' * 90}")
    print(f"  REGION-TO-REGION VIABILITY")
    print(f"  Min channels/region: {MIN_CHANNELS}  |  Min subjects for LOSO: {MIN_SUBJECTS}")
    print(f"{'=' * 90}")

    # Build table: subject x region -> channel count
    print(f"\n  Channel count matrix (subjects x regions):")
    print(f"  {'Subject':>10s}", end="")
    for r in display_regions:
        rname = r[:18]
        print(f"  {rname:>18s}", end="")
    print(f"  {'total':>8s}  {'sessions':>8s}")
    print(f"  {'─'*10}", end="")
    for _ in display_regions:
        print(f"  {'─'*18}", end="")
    print(f"  {'─'*8}  {'─'*8}")

    for subj in sorted(subject_region_counts.keys()):
        rc = subject_region_counts[subj]
        n_sess = len(subjects[subj])
        total_ch = sum(rc.values())
        print(f"  {subj:>10s}", end="")
        for r in display_regions:
            cnt = rc.get(r, 0)
            mark = " ✓" if cnt >= MIN_CHANNELS else " ✗" if cnt == 0 else " ~"
            print(f"  {cnt:>15d}{mark:>2s}", end="")
        print(f"  {total_ch:>8d}  {n_sess:>8d}")

    # Test all region pairs
    print(f"\n  Region pair viability (>= {MIN_SUBJECTS} subjects with >= {MIN_CHANNELS} channels each):")
    viable_runs = []

    for src in REGIONS_OF_INTEREST:
        for tgt in REGIONS_OF_INTEREST:
            if src == tgt:
                continue

            valid_subjects = []
            for subj, rc in subject_region_counts.items():
                src_ch = rc.get(src, 0)
                tgt_ch = rc.get(tgt, 0)
                if src_ch >= MIN_CHANNELS and tgt_ch >= MIN_CHANNELS:
                    valid_subjects.append((subj, src_ch, tgt_ch))

            n_valid = len(valid_subjects)
            mark = "✓" if n_valid >= MIN_SUBJECTS else "✗"

            subj_str = ", ".join(f"{s[0]}({s[1]}/{s[2]}ch)" for s in valid_subjects)
            print(f"  {mark} {src:>20s} -> {tgt:<20s}  {n_valid} subjects  [{subj_str}]")

            if n_valid >= MIN_SUBJECTS:
                viable_runs.append({
                    "source": src,
                    "target": tgt,
                    "n_subjects": n_valid,
                    "subjects": valid_subjects,
                })

    # =========================================================================
    #  LOSO FOLD DETAILS
    # =========================================================================
    if viable_runs:
        print(f"\n\n{'=' * 90}")
        print(f"  VIABLE LOSO RUNS ({len(viable_runs)} total)")
        print(f"{'=' * 90}")

        for run in viable_runs:
            print(f"\n  {run['source']} -> {run['target']} ({run['n_subjects']} subjects):")
            for subj, src_ch, tgt_ch in run["subjects"]:
                n_sess = len(subjects.get(subj, []))
                total_ieeg_trials = sum(s["n_ieeg_trials"] for s in subjects.get(subj, []))
                print(f"    {subj}: {src_ch} src ch, {tgt_ch} tgt ch, "
                      f"{n_sess} sessions, {total_ieeg_trials} iEEG trials")
    else:
        print(f"\n\n  NO VIABLE LOSO RUNS FOUND")
        print(f"  This may mean:")
        print(f"    1. Electrode labels couldn't be parsed (check electrode label output above)")
        print(f"    2. Regions are stored differently in this dataset")
        print(f"    3. Too few subjects have channels in multiple regions")

    # =========================================================================
    #  SIGNAL SHAPE SUMMARY
    # =========================================================================
    print(f"\n\n{'=' * 90}")
    print(f"  SIGNAL SHAPE SUMMARY")
    print(f"{'=' * 90}")

    seen_shapes = defaultdict(list)
    for result in all_results:
        if result["ieeg_shape"]:
            seen_shapes[str(result["ieeg_shape"])].append(result["filename"])
        if result["scalp_shape"]:
            seen_shapes[f"scalp:{result['scalp_shape']}"].append(result["filename"])
        if result["sampling_rate"]:
            seen_shapes[f"sr:{result['sampling_rate']}"].append(result["filename"])

    for shape_key in sorted(seen_shapes.keys()):
        files = seen_shapes[shape_key]
        print(f"  {shape_key}: {len(files)} files")
        for fname in files[:5]:
            print(f"    {fname}")
        if len(files) > 5:
            print(f"    ... ({len(files)-5} more)")

    # =========================================================================
    #  SOZ ANALYSIS — exclude seizure onset zone electrodes
    # =========================================================================
    print(f"\n\n{'=' * 90}")
    print(f"  SEIZURE ONSET ZONE (SOZ) ANALYSIS")
    print(f"  SOZ electrodes may have pathological activity — consider excluding")
    print(f"{'=' * 90}")

    subject_soz = {}  # subject -> list of SOZ probe abbreviations
    subject_region_counts_no_soz = {}  # subject -> {region: channels} excluding SOZ probes

    for subj in sorted(subjects.keys()):
        sess = subjects[subj][0]
        meta = sess.get("subject_meta", {})
        soz_probes = []

        for key, val in meta.items():
            if "seizure onset" in key.lower() or "soz" in key.lower():
                raw = extract_first_string(val).strip()
                if raw:
                    soz_probes = [p.strip().upper() for p in raw.split(",") if p.strip()]

        subject_soz[subj] = soz_probes

        # Compute region counts excluding SOZ probes
        probe_ch = sess.get("probe_channels", {})
        region_counts_clean = defaultdict(int)
        for probe, n_contacts in probe_ch.items():
            if probe not in soz_probes:
                region, _, _, _, _ = parse_electrode_region(probe)
                region_counts_clean[region] += n_contacts
        subject_region_counts_no_soz[subj] = dict(region_counts_clean)

        probes_all = sess.get("depth_probes", [])
        clean_probes = [p for p in probes_all if p not in soz_probes]
        n_ch_clean = sum(probe_ch.get(p, 0) for p in clean_probes)
        n_ch_total = sum(probe_ch.values())

        soz_str = ", ".join(soz_probes) if soz_probes else "(none)"
        print(f"  {subj}: SOZ = [{soz_str}]  "
              f"channels: {n_ch_total} total -> {n_ch_clean} after SOZ exclusion  "
              f"probes: {len(probes_all)} -> {len(clean_probes)}")

    # Viability table excluding SOZ
    print(f"\n  Region pair viability EXCLUDING SOZ probes:")
    viable_runs_no_soz = []
    for src in REGIONS_OF_INTEREST:
        for tgt in REGIONS_OF_INTEREST:
            if src == tgt:
                continue
            valid_subjects = []
            for subj, rc in subject_region_counts_no_soz.items():
                src_ch = rc.get(src, 0)
                tgt_ch = rc.get(tgt, 0)
                if src_ch >= MIN_CHANNELS and tgt_ch >= MIN_CHANNELS:
                    valid_subjects.append((subj, src_ch, tgt_ch))
            n_valid = len(valid_subjects)
            mark = "✓" if n_valid >= MIN_SUBJECTS else "✗"
            subj_str = ", ".join(f"{s[0]}({s[1]}/{s[2]}ch)" for s in valid_subjects)
            print(f"  {mark} {src:>20s} -> {tgt:<20s}  {n_valid} subjects  [{subj_str}]")
            if n_valid >= MIN_SUBJECTS:
                viable_runs_no_soz.append({
                    "source": src, "target": tgt,
                    "n_subjects": n_valid, "subjects": valid_subjects,
                })

    # =========================================================================
    #  OVERALL VERDICT
    # =========================================================================
    all_regions_found = set()
    for rc in subject_region_counts.values():
        all_regions_found.update(rc.keys())

    print(f"\n\n{'=' * 90}")
    print(f"  OVERALL VERDICT")
    print(f"{'=' * 90}")
    print(f"  Subjects:              {len(subjects)}")
    print(f"  Total sessions/files:  {len(all_results)}")
    total_trials = sum(sum(s["n_ieeg_trials"] for s in sessions)
                       for sessions in subjects.values())
    print(f"  Total iEEG trials:     {total_trials}")
    print(f"  Sampling rate:         2000 Hz (all files)")
    print(f"  Regions found:         {sorted(all_regions_found)}")
    print(f"  Viable LOSO runs:      {len(viable_runs)} (all probes)")
    print(f"  Viable LOSO runs:      {len(viable_runs_no_soz)} (excluding SOZ)")

    best_runs = viable_runs if viable_runs else viable_runs_no_soz
    if best_runs:
        print(f"\n  DATASET IS VIABLE for inter-region neural translation!")
        print(f"  Best region pairs (including SOZ):")
        for run in sorted(viable_runs, key=lambda x: -x["n_subjects"]):
            print(f"    {run['source']} -> {run['target']}: {run['n_subjects']} subjects")
        if viable_runs_no_soz:
            print(f"  Best region pairs (excluding SOZ):")
            for run in sorted(viable_runs_no_soz, key=lambda x: -x["n_subjects"]):
                print(f"    {run['source']} -> {run['target']}: {run['n_subjects']} subjects")
    else:
        print(f"\n  DATASET MAY NOT BE VIABLE — check probe mapping above")
        print(f"  If regions show as 'unknown', the naming convention may differ")

    print()


if __name__ == "__main__":
    main()
