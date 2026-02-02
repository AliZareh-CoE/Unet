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

---

## 6. Next Steps

### Step 1: Preprocess with Uniform Sampling Rate

```bash
python scripts/preprocess_cogitate_bids.py /data/COG_ECOG_EXP1_BIDS \
    --output-dir /data/COGITATEDataset_1024hz \
    --target-sfreq 1024 \
    --no-filter
```

### Step 2: Verify Configuration

```bash
python scripts/create_cogitate_configs.py /data/COGITATEDataset_1024hz --strategy all
```

### Step 3: Create Final Dataset

A new script is needed to:
1. Extract source/target region channels per subject
2. Apply uniform channel selection (first N channels or by signal quality)
3. Segment continuous data into training windows
4. Create train/val splits compatible with existing DataLoaders

### Step 4: Training

Once the dataset is prepared with uniform dimensions:
```bash
python LOSO/runner.py --dataset COGITATEDataset_temp_front \
    --source-channels 23 --target-channels 17 \
    --batch-size 32 --epochs 100
```

---

## 7. Key Differences from Existing Datasets

| Aspect | PCX1 (OB→PCx) | COGITATE |
|--------|---------------|----------|
| Species | Rat | Human |
| Recording | Surface array | Depth electrodes |
| Channels | 32 → 32 (fixed) | Variable per region |
| Sampling | 1000 Hz (uniform) | 128-2048 Hz (heterogeneous) |
| Subjects | Multiple | 38 |
| Regions | OB, PCx | Multiple cortical/subcortical |

---

## 8. Technical Notes

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
