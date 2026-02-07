# ECoG Library: Biological Validation Guide

## Dataset Overview

The Miller ECoG Library (Miller, 2019) is a curated collection of human
electrocorticographic (ECoG) recordings from 34 epilepsy patients across 16
behavioral experiments. All data were recorded with subdural platinum electrode
arrays at 1kHz using the same amplifier (Synamps2) and acquisition software
(BCI2000).

**Reference:** Miller, K.J. (2019). A library of human electrocorticographic data
and analyses. *Nature Human Behaviour*, 3(11), 1225-1235.
[DOI: 10.1038/s41562-019-0678-3](https://doi.org/10.1038/s41562-019-0678-3)

**Repository:** [Stanford Digital Repository](https://purl.stanford.edu/zk881ps0522)
(CC BY-SA 4.0)

## Why This Dataset Matters for Biological Validation

Our UNet model was developed for translating neural signals between brain
regions (e.g., OB→PCx, PFC→CA1). The ECoG Library provides an opportunity to
validate this approach on **human cortical data** with known neuroanatomical
ground truth, testing whether the model captures biologically meaningful
inter-region relationships.

### Key Validation Properties

| Property | Our Existing Datasets | Miller ECoG Library |
|---|---|---|
| Species | Rodent (mouse/rat) | Human |
| Signal type | LFP (depth electrodes) | ECoG (subdural surface) |
| Sampling rate | 1kHz / 1.25kHz | 1kHz |
| Brain coverage | 2 regions per dataset | Multiple lobes per subject |
| Subjects | Sessions | 34 patients |
| Task paradigms | Olfaction / Navigation | Motor, visual, cognitive, memory |

## Available Experiments

| Experiment | Subjects | Task | Useful For |
|---|---|---|---|
| `fingerflex` | 9 | Cued finger flexion | Motor cortex→somatosensory, cortico-behavioral |
| `faceshouses` | 7 | Face vs. house perception | Visual→temporal cortex translation |
| `motor_imagery` | 1 | Motor execution + imagery | Execution vs. imagery comparison |
| `joystick_track` | 4 | 2D joystick tracking | Continuous motor decoding |
| `memory_nback` | 3 | N-back working memory | Frontal→temporal during cognition |

## Biological Validation Strategies

### 1. Inter-Region Translation Fidelity

**Goal:** Test whether the model learns biologically meaningful mappings
between cortical regions.

**Approach:** Group electrodes by anatomical lobe (frontal, temporal, parietal)
and train the UNet to translate signals from one region to another.

```bash
# Download the data
python scripts/download_ecog.py --experiments fingerflex --explore --regions

# Train: frontal → temporal translation
python train.py --dataset ecog \
    --ecog-experiment fingerflex \
    --ecog-source-region frontal \
    --ecog-target-region temporal \
    --epochs 80

# LOSO cross-validation (Leave-One-Subject-Out)
python -m LOSO.runner --dataset ecog \
    --ecog-experiment fingerflex \
    --ecog-source-region frontal \
    --ecog-target-region temporal
```

**Validation criteria:**
- R² > 0 indicates the model captures inter-region signal structure
- Compare against a Wiener filter baseline (linear translation)
- Higher R² for anatomically connected regions (e.g., pre-central↔post-central)
  vs. distant regions validates biological plausibility

### 2. Task-Modulated Translation

**Goal:** Test whether translation quality changes during task-relevant epochs
(e.g., movement onset) vs. rest.

**Approach with fingerflex:**
1. Train the model on full continuous data
2. At evaluation, segment outputs by trial events (using `t_on`/`t_off`)
3. Compare R² during movement epochs vs. rest epochs
4. If R² is higher during movement, the model has learned task-relevant
   inter-region dynamics

**Expected biological result:** Motor cortex (frontal) → somatosensory (parietal)
translation should improve during active movement because:
- Efference copies from motor cortex reach somatosensory cortex
- Cortico-cortical coupling increases during voluntary movement
- Sensorimotor integration strengthens the frontal-parietal signal relationship

### 3. Frequency Band Analysis

**Goal:** Validate that the model preserves known ECoG spectral features.

**Known ECoG spectral properties (from Miller's work):**
- Power law: P ∝ 1/f^χ (χ ≈ 2-4 depending on region)
- High-gamma band (70-150 Hz): Correlates with local neural population activity
- Beta band (13-30 Hz): Modulated by motor planning and execution
- 1/f broadband shift during task engagement

**Validation approach:**
1. Compute PSD of model outputs vs. ground truth targets
2. Verify the model preserves the 1/f power law slope
3. Check that task-modulated high-gamma changes are captured
4. Use `--compute-psd-validation` flag to track spectral metrics during training

```bash
python train.py --dataset ecog \
    --ecog-experiment fingerflex \
    --ecog-source-region frontal \
    --ecog-target-region parietal \
    --compute-psd-validation \
    --epochs 80
```

### 4. Cross-Subject Generalization (LOSO)

**Goal:** Test whether the model generalizes across human subjects with different
electrode placements and cortical anatomy.

**This is the strongest biological validation test.** Because each patient has a
unique electrode grid placement over different cortical areas, successful
Leave-One-Subject-Out cross-validation demonstrates that the model has learned
general principles of cortical signal propagation rather than
subject-specific artifact patterns.

```bash
# Full LOSO cross-validation
python -m LOSO.runner --dataset ecog \
    --ecog-experiment fingerflex \
    --ecog-source-region frontal \
    --ecog-target-region temporal \
    --epochs 80 --batch-size 64

# Run directly from train.py
python train.py --dataset ecog --loso \
    --ecog-experiment fingerflex \
    --ecog-source-region frontal \
    --ecog-target-region temporal
```

**Expected results:**
- Positive mean R² across folds = model captures general inter-region structure
- High variance across folds = expected due to electrode placement differences
- Compare to within-subject R² to quantify the generalization gap

### 5. Directional Asymmetry

**Goal:** Test whether translation performance is direction-dependent
(A→B vs. B→A), validating that the model captures anatomical connectivity
asymmetries.

**Biological basis:** Cortical connections are not symmetric. For example:
- Motor cortex → somatosensory cortex: strong efferent projections
- Temporal cortex → frontal cortex: feedforward sensory pathways
- Frontal cortex → temporal cortex: top-down modulatory pathways

**Approach:** Train two models with swapped source/target and compare R²:

```bash
# Direction 1: frontal → temporal
python train.py --dataset ecog \
    --ecog-source-region frontal --ecog-target-region temporal

# Direction 2: temporal → frontal
python train.py --dataset ecog \
    --ecog-source-region temporal --ecog-target-region frontal
```

If the model achieves different R² in each direction, this reflects asymmetric
cortical connectivity—a biologically meaningful result.

### 6. Anatomical Distance Effect

**Goal:** Validate that translation difficulty scales with anatomical distance
between regions.

**Approach:** Train separate models for different region pairs and rank by R²:

| Source | Target | Expected Difficulty | Biological Reasoning |
|---|---|---|---|
| frontal (motor) | parietal (somatosensory) | Easiest | Direct cortico-cortical connections |
| frontal | temporal | Medium | Multi-synaptic pathways |
| parietal | temporal | Hard | Indirect connectivity |

Lower R² for more distant/disconnected regions would validate that the model
is learning genuine neural signal relationships rather than trivial correlations.

## Data Characteristics and Preprocessing Notes

### Signal Properties
- **Format:** Preprocessed NumPy `.npz` files (via Neuromatch/OSF)
- **Preprocessing applied:** Notch filtered at 60, 120, 180, 240, 250 Hz;
  z-scored across recording; stored as float16
- **Our pipeline:** Additional per-channel z-score and sliding window extraction

### Electrode Metadata
Each electrode has anatomical annotations:
- `hemisphere`: Left or right
- `lobe`: Frontal, temporal, parietal, occipital
- `gyrus`: Specific gyrus name
- `Brodmann_Area`: Cytoarchitectonic area number
- `locs`: 3D coordinates on brain surface

### Subject Variability
Unlike our rodent datasets where electrode placement is standardized, ECoG
electrode placement varies across subjects based on clinical need. This means:
- Channel counts per region vary across subjects (handled by taking the minimum)
- Some subjects may lack coverage in certain lobes
- The `prepare_ecog_data()` function automatically filters subjects that don't
  have sufficient channels (≥4) in both source and target regions

## Quick Start

```bash
# 1. Download the dataset
python scripts/download_ecog.py

# 2. Explore what's available
python scripts/download_ecog.py --explore --regions

# 3. Train a model
python train.py --dataset ecog --epochs 60

# 4. Run LOSO cross-validation
python -m LOSO.runner --dataset ecog --output-dir results/ecog_loso

# 5. Try different experiments and region pairs
python train.py --dataset ecog \
    --ecog-experiment faceshouses \
    --ecog-source-region temporal \
    --ecog-target-region parietal
```

## Interpreting Results

### Positive R² (model better than mean prediction)
The model has learned meaningful inter-region signal structure. Compare the
magnitude to our existing datasets:
- Olfactory (OB→PCx): Our baseline performance
- PFC→CA1: Cross-region LFP translation

### Near-Zero R²
The selected region pair may not have strong functional coupling in the chosen
task. Try:
- A different region pair (e.g., frontal↔parietal for motor tasks)
- A different experiment that engages the regions of interest
- Examining per-frequency-band performance (low freq may be easier than high)

### Negative R²
Possible causes:
- Insufficient data (too few subjects with coverage in both regions)
- Regions genuinely not coupled in this task
- Model overfitting to training subjects' specific electrode patterns

## Comparison to Existing Datasets

Running the model on the ECoG Library alongside our existing datasets enables
a multi-scale, cross-species comparison:

| Level | Dataset | Species | Translation |
|---|---|---|---|
| Subcortical LFP | Olfactory (OB→PCx) | Rodent | Bulb → cortex |
| Subcortical LFP | PFC→CA1 | Rodent | Prefrontal → hippocampus |
| Cortical iEEG | DANDI Movie | Human | Amygdala → hippocampus |
| Cortical ECoG | **Miller ECoG** | **Human** | **Cortex → cortex** |

This progression tests whether the same architecture and training procedure
can learn neural signal translation at different spatial scales, across
species, and between different brain systems.
