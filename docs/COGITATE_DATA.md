# COGITATE Dataset Integration Guide

## Overview

This document describes the exploration and preprocessing of the COGITATE EXP1 BIDS dataset for neural signal translation experiments.

**Dataset**: COGITATE (Consciousness & Global Intelligence Assessment for Testing Emergent Intelligence)
**Format**: BIDS (Brain Imaging Data Structure)
**Modality**: iEEG (intracranial EEG / SEEG depth electrodes)
**Task**: "Dur" - Duration/visual cognitive task with faces, objects, letters, false-fonts

---

## 1. Dataset Structure

### Subjects
- **Total subjects**: 38
- **Sites**: CE (16), CF (11), CG (11)
- **Electrode type**: SEEG (stereo-EEG depth electrodes)

### File Format
- BrainVision format (`.eeg`, `.vhdr`, `.vmrk`)
- Electrode positions in MNI coordinates
- Channel metadata with Desikan atlas region labels

### Sampling Rate Distribution
| Rate (Hz) | Subjects |
|-----------|----------|
| 128       | 1        |
| 512       | 4        |
| 1023      | 1        |
| 1024      | 26       |
| 2000      | 1        |
| 2048      | 5        |

**Note**: Heterogeneous sampling rates require resampling for uniform dataset.

---

## 2. Brain Region Coverage

Electrodes are mapped to anatomical regions via the Desikan atlas. Key regions for neural translation:

| Region | Subjects | Min Channels | Max Channels | Mean |
|--------|----------|--------------|--------------|------|
| temporal | 37 | 7 | 95 | 42.6 |
| frontal | 35 | 3 | 79 | 35.1 |
| hippocampus | 35 | 2 | 22 | 7.4 |
| amygdala | 27 | 2 | 12 | 4.5 |
| parietal | 29 | 1 | 42 | 13.8 |
| insula | 26 | 1 | 14 | 4.9 |
| occipital | 14 | 1 | 20 | 6.4 |

---

## 3. Neural Translation Configurations

### Configuration Analysis

Three scientifically motivated translation pairs were analyzed:

1. **Temporal → Frontal** (Perception → Executive)
2. **Temporal → Hippocampus** (Perception → Memory)
3. **Hippocampus → Amygdala** (Memory → Emotion)

### Channel Selection Strategies

To create uniform datasets, three strategies were evaluated:

| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| `min` | Use minimum channel count | All subjects included, fewer channels |
| `percentile_25` | Use 25th percentile | ~75% subjects, more channels |
| `median` | Use median channel count | ~50% subjects, most channels |

### Results Summary

| Pair | Strategy | Subjects | Source Ch | Target Ch | Viable |
|------|----------|----------|-----------|-----------|--------|
| temporal → frontal | min | 32 | 7 | 3 | ✅ |
| temporal → frontal | percentile_25 | 20 | 23 | 17 | ✅ |
| temporal → frontal | median | 10 | 33 | 24 | ✅ |
| temporal → hippocampus | min | 33 | 7 | 2 | ✅ |
| temporal → hippocampus | percentile_25 | 19 | 23 | 4 | ✅ |
| temporal → hippocampus | median | 11 | 33 | 7 | ✅ |
| hippocampus → amygdala | min | 22 | 2 | 2 | ✅ |
| hippocampus → amygdala | percentile_25 | 17 | 4 | 2 | ✅ |
| hippocampus → amygdala | median | 10 | 7 | 4 | ✅ |

---

## 4. Recommended Configurations

### Best Balance: Temporal → Frontal (percentile_25)

- **Subjects**: 20
- **Dimensions**: 23 → 17 channels
- **Total duration**: ~45 hours
- **Rationale**: Good subject count with meaningful channel depth

### Most Subjects: Temporal → Frontal (min)

- **Subjects**: 32
- **Dimensions**: 7 → 3 channels
- **Rationale**: Maximum statistical power, minimal channels

### Alternative: Temporal → Hippocampus (percentile_25)

- **Subjects**: 19
- **Dimensions**: 23 → 4 channels
- **Rationale**: Memory-perception relationship

---

## 5. Scripts Created

### `scripts/explore_bids_data.py`
Initial exploration of BIDS structure, subjects, channels, and events.

```bash
python scripts/explore_bids_data.py /data/COG_ECOG_EXP1_BIDS
```

### `scripts/preprocess_cogitate_bids.py`
Converts BIDS data to NumPy format with optional filtering/resampling.

```bash
# Basic preprocessing (keep original rate)
python scripts/preprocess_cogitate_bids.py /data/COG_ECOG_EXP1_BIDS \
    --output-dir /data/COGITATEDataset \
    --no-filter

# With resampling to uniform 1024 Hz
python scripts/preprocess_cogitate_bids.py /data/COG_ECOG_EXP1_BIDS \
    --output-dir /data/COGITATEDataset \
    --target-sfreq 1024 \
    --no-filter
```

### `scripts/create_cogitate_configs.py`
Analyzes preprocessed data and determines viable configurations.

```bash
python scripts/create_cogitate_configs.py /data/COGITATEDataset \
    --strategy all \
    --min-channels 2 \
    --min-rate 512
```

### `scripts/analyze_cogitate_coverage.py`
Strategic analysis of region coverage and sampling rates.

```bash
python scripts/analyze_cogitate_coverage.py /data/COGITATEDataset
```

### `scripts/build_cogitate_dataset.py`
Builds final training datasets with uniform dimensions. Supports three modes:

**Region-to-Region (many-to-many):**
```bash
python scripts/build_cogitate_dataset.py /data/COGITATEDataset_1024hz \
    --source temporal --target frontal \
    --source-channels 23 --target-channels 17 \
    --output /data/COGITATE_temp_front
```

**Multi-Region Source (many-to-one):**
```bash
python scripts/build_cogitate_dataset.py /data/COGITATEDataset_1024hz \
    --source temporal parietal --target hippocampus \
    --source-channels 30 --target-channels 7 \
    --output /data/COGITATE_multi_hipp
```

**Within-Region Interpolation:**
```bash
python scripts/build_cogitate_dataset.py /data/COGITATEDataset_1024hz \
    --source temporal --target temporal \
    --mode within-region \
    --source-channels 20 --target-channels 10 \
    --output /data/COGITATE_temp_interp
```

---

## 6. Training Modes

### Mode 1: Region-to-Region Translation
Translate neural signals between different brain regions.
- **Use case**: Predict frontal activity from temporal activity
- **Example**: Perception (temporal) → Executive control (frontal)

### Mode 2: Many-to-One Translation
Combine multiple source regions to predict a single target region.
- **Use case**: Integrate information from multiple areas
- **Example**: Temporal + Parietal → Hippocampus (memory encoding)

### Mode 3: Within-Region Interpolation
Learn to interpolate missing channels within a region.
- **Use case**: Reconstruct failed/missing electrodes
- **Example**: Use 20 temporal channels to reconstruct 10 others

---

## 7. Quick Start Commands

### Step 1: Preprocess (already done)

```bash
python scripts/preprocess_cogitate_bids.py /data/COG_ECOG_EXP1_BIDS \
    --output-dir /data/COGITATEDataset_1024hz \
    --target-sfreq 1024 \
    --no-filter
```

### Step 2: Build Training Dataset

**Recommended - Temporal → Frontal (20 subjects):**
```bash
python scripts/build_cogitate_dataset.py /data/COGITATEDataset_1024hz \
    --source temporal --target frontal \
    --source-channels 23 --target-channels 17 \
    --output /data/COGITATE_temp_front_p25
```

**Alternative - More subjects, fewer channels (32 subjects):**
```bash
python scripts/build_cogitate_dataset.py /data/COGITATEDataset_1024hz \
    --source temporal --target frontal \
    --source-channels 7 --target-channels 3 \
    --output /data/COGITATE_temp_front_min
```

### Step 3: Training

```bash
python LOSO/runner.py --dataset COGITATE_temp_front_p25 \
    --source-channels 23 --target-channels 17 \
    --batch-size 32 --epochs 100
```

---

## 8. Key Differences from Existing Datasets

| Aspect | PCX1 (OB→PCx) | COGITATE |
|--------|---------------|----------|
| Species | Rat | Human |
| Recording | Surface array | Depth electrodes |
| Channels | 32 → 32 (fixed) | Variable per region |
| Sampling | 1000 Hz (uniform) | 128-2048 Hz (heterogeneous) |
| Subjects | Multiple | 38 |
| Regions | OB, PCx | Multiple cortical/subcortical |

---

## 9. Technical Notes

### Nyquist Considerations
- Original 1024 Hz → Nyquist: 512 Hz
- Resampling to 1024 Hz from higher rates preserves up to 512 Hz content
- For high-gamma analysis (>200 Hz), subjects at 2048 Hz have full coverage

### Region Mapping
Electrodes are automatically mapped to Desikan atlas regions based on MNI coordinates in the electrodes.tsv files.

### MNE-Python Requirements
```bash
pip install mne mne-bids pandas numpy
```

---

*Document created: 2026-02-02*
*Last updated: 2026-02-02*
