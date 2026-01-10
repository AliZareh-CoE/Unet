# Nature Methods Publication Plan: Neural Signal Translation

## Project Overview

**Title (working):** "Cross-Regional Neural Signal Translation Using Conditional U-Net Architecture"

**Core Contribution:** A deep learning method for translating neural activity between brain regions, enabling prediction of downstream neural responses from upstream recordings.

---

## 1. MAIN FIGURES (4-6 figures typical for Nature Methods)

### Figure 1: Method Overview & Architecture
**Purpose:** Introduce the problem and our solution

**Panels:**
- **(A)** Schematic: Brain regions with electrodes, showing translation concept (OB→PCx, PFC→CA1, AMY→HPC)
- **(B)** CondUNet1D architecture diagram:
  - Encoder-decoder with skip connections
  - Cross-frequency coupling attention (SE attention)
  - Spectro-temporal conditioning module
  - Bidirectional training (forward + reverse)
- **(C)** Training pipeline: Data flow from raw LFP → preprocessing → model → evaluation
- **(D)** Loss function components: Huber + Wavelet multi-scale loss

### Figure 2: Primary Results - Olfactory System (OB→PCx)
**Purpose:** Demonstrate core performance on primary dataset

**Panels:**
- **(A)** Example traces: Ground truth vs predicted PCx signals (multiple channels, 5s windows)
- **(B)** Correlation heatmap: Per-channel prediction accuracy
- **(C)** Spectral fidelity: PSD comparison (predicted vs actual) across frequency bands
- **(D)** Cross-session generalization: Performance on held-out recording sessions
- **(E)** Per-odor performance: Does prediction quality vary by stimulus?
- **(F)** Reverse direction (PCx→OB): Bidirectional validation

### Figure 3: Cross-Dataset Generalization
**Purpose:** Show method works across species, brain regions, recording types

**Panels:**
- **(A)** PFC→CA1 (rodent hippocampus): Performance metrics + example traces
- **(B)** PCx1 continuous dataset: Long recording translation
- **(C)** DANDI 000623 (human iEEG): AMY→HPC during movie watching
- **(D)** Summary bar plot: Correlation across all datasets
- **(E)** Comparison table: Channel counts, sampling rates, recording types

### Figure 4: Ablation Studies
**Purpose:** Justify architectural decisions

**Panels:**
- **(A)** Attention mechanism comparison:
  - No attention vs Basic self-attention vs Cross-frequency coupling (V1 vs V2)
- **(B)** Loss function ablation:
  - L1 only vs Huber only vs Wavelet only vs Combined
- **(C)** Conditioning ablation:
  - No conditioning vs Odor labels vs Spectro-temporal (self-conditioning)
- **(D)** Model capacity:
  - Base channels: 32 vs 64 vs 128
- **(E)** Data augmentation impact:
  - No augmentation vs Full augmentation suite
- **(F)** Bidirectional training:
  - Unidirectional vs Bidirectional with cycle consistency

### Figure 5: Interpretability & Neuroscience Validation
**Purpose:** Show the model learns biologically meaningful representations

**Panels:**
- **(A)** Channel importance: Which source channels drive predictions?
- **(B)** Frequency band contributions: Theta, beta, gamma importance
- **(C)** Temporal dynamics: When in the trial does prediction matter most?
- **(D)** Cross-frequency coupling preservation: Does predicted signal maintain theta-gamma coupling?
- **(E)** Phase-amplitude coupling (PAC) analysis: Predicted vs actual
- **(F)** Coherence matrices: Inter-regional coherence preservation

### Figure 6: Comparison to Baselines
**Purpose:** Benchmark against alternative methods

**Panels:**
- **(A)** Classical methods:
  - Linear regression
  - Ridge regression
  - Wiener filter
  - PCA + regression
- **(B)** Deep learning baselines:
  - Basic CNN
  - LSTM/GRU
  - Transformer
  - Standard U-Net (no conditioning)
- **(C)** Statistical comparison: Paired tests across sessions
- **(D)** Computational efficiency: Training time, inference speed, memory

---

## 2. SUPPLEMENTARY FIGURES

### Supp. Fig 1: Extended Architecture Details
- Full layer specifications
- Attention mechanism internals
- Wavelet loss decomposition

### Supp. Fig 2: Dataset Details
- Recording setup schematics for each dataset
- Electrode placement diagrams
- Session/trial statistics

### Supp. Fig 3: Hyperparameter Sensitivity
- Learning rate sweeps
- Batch size effects
- Window size optimization

### Supp. Fig 4: Additional Ablations
- Dropout rates
- Skip connection variants
- Encoder depth

### Supp. Fig 5: Per-Session Breakdown
- Individual session performance
- Session difficulty analysis
- Outlier investigation

### Supp. Fig 6: Failure Cases
- When does the model fail?
- Low-SNR scenarios
- Edge cases

### Supp. Fig 7: Extended Human iEEG Results
- All region pairs for DANDI dataset
- Subject variability
- Electrode count effects

---

## 3. ABLATION STUDIES TO RUN

### 3.1 Architecture Ablations

```bash
# Attention type comparison
for attn in none basic cross_freq cross_freq_v2; do
  python train.py --dataset pcx1 --attention-type $attn --epochs 60 --seed 42
done

# Base channel scaling
for ch in 32 64 128; do
  python train.py --dataset pcx1 --base-channels $ch --epochs 60 --seed 42
done

# With/without conditioning
python train.py --dataset pcx1 --conditioning none --epochs 60
python train.py --dataset pcx1 --conditioning spectro_temporal --epochs 60
python train.py --dataset pcx1 --conditioning odor_onehot --epochs 60
```

### 3.2 Loss Function Ablations

```bash
for loss in l1 huber wavelet l1_wavelet huber_wavelet; do
  python train.py --dataset pcx1 --loss $loss --epochs 60 --seed 42
done
```

### 3.3 Training Ablations

```bash
# Augmentation impact
python train.py --dataset pcx1 --no-aug --epochs 60
python train.py --dataset pcx1 --epochs 60  # with aug (default)

# Bidirectional training
python train.py --dataset pcx1 --no-bidirectional --epochs 60
python train.py --dataset pcx1 --epochs 60  # bidirectional (default)
```

### 3.4 Cross-Dataset Validation

```bash
# All datasets with same settings
for ds in olfactory pcx1 pfc dandi; do
  python train.py --dataset $ds --epochs 60 --seed 42
done
```

---

## 4. STATISTICAL TESTS REQUIRED

### 4.1 Primary Metrics
- **Pearson correlation (r):** Primary accuracy metric
- **R² (coefficient of determination):** Variance explained
- **NRMSE:** Normalized root mean square error
- **PSD error (dB):** Spectral fidelity

### 4.2 Statistical Comparisons
- **Paired t-tests:** Model vs baselines (per session)
- **Wilcoxon signed-rank:** Non-parametric alternative
- **Bootstrap confidence intervals:** 95% CI for all metrics
- **Multiple comparison correction:** Bonferroni or FDR for ablations

### 4.3 Cross-Validation
- **Session-based holdout:** Train on N-k sessions, test on k
- **Leave-one-session-out:** For smaller datasets
- **Stratified by condition:** Ensure balanced odor/trial types

### 4.4 Effect Sizes
- **Cohen's d:** For pairwise comparisons
- **Eta-squared:** For ANOVA-style ablations

---

## 5. METHODS SECTION OUTLINE

### 5.1 Datasets
- Olfactory bulb / Piriform cortex (rodent, N sessions, N trials)
- PFC / CA1 hippocampus (rodent, N sessions)
- PCx1 continuous recordings (rodent, N hours)
- DANDI 000623 human iEEG (N subjects, N electrodes)

### 5.2 Preprocessing
- Bandpass filtering (0.5-200 Hz)
- Downsampling to 1000 Hz
- Z-score normalization per window
- Artifact rejection criteria

### 5.3 Model Architecture
- CondUNet1D: encoder-decoder with skip connections
- Cross-frequency coupling attention mechanism
- Spectro-temporal conditioning module
- Output scaling layer

### 5.4 Training Procedure
- Loss: Huber + multi-scale wavelet
- Optimizer: AdamW, lr=1e-4, weight_decay=0.01
- Scheduler: Cosine annealing with warmup
- Batch size: 32-64, epochs: 60
- Data augmentation: time shift, noise, channel dropout, amplitude scaling

### 5.5 Evaluation Protocol
- Session-based holdout (cross-session generalization)
- Metrics: Pearson r, R², NRMSE, PSD error
- Per-channel and aggregate statistics

### 5.6 Baseline Methods
- Linear regression, Ridge, Wiener filter
- CNN, LSTM, Transformer baselines
- Implementation details for each

### 5.7 Statistical Analysis
- Bootstrap CIs (10,000 iterations)
- Paired statistical tests with correction
- Effect size reporting

---

## 6. CODE & DATA AVAILABILITY

### 6.1 Code Release
- GitHub repository with:
  - Training scripts
  - Model implementations
  - Evaluation notebooks
  - Pre-trained model weights

### 6.2 Data Availability
- Olfactory dataset: [repository TBD]
- DANDI 000623: https://dandiarchive.org/dandiset/000623
- PFC dataset: [repository TBD]

---

## 7. EXPERIMENTS TO RUN (PRIORITY ORDER)

### Phase 1: Core Results (Week 1-2)
1. [ ] Train final model on PCx1 with best hyperparameters
2. [ ] Generate main Figure 2 panels (example traces, metrics)
3. [ ] Run cross-dataset validation (olfactory, pfc, dandi)
4. [ ] Compute all primary metrics with CIs

### Phase 2: Ablations (Week 2-3)
5. [ ] Attention ablation (4 conditions × 3 seeds)
6. [ ] Loss function ablation (5 conditions × 3 seeds)
7. [ ] Conditioning ablation (3 conditions × 3 seeds)
8. [ ] Augmentation ablation (2 conditions × 3 seeds)

### Phase 3: Baselines (Week 3-4)
9. [ ] Implement classical baselines (linear, ridge, wiener)
10. [ ] Implement DL baselines (CNN, LSTM, Transformer)
11. [ ] Run all baselines on all datasets
12. [ ] Statistical comparisons

### Phase 4: Analysis (Week 4-5)
13. [ ] Interpretability analysis (channel importance, frequency bands)
14. [ ] Neuroscience validation (PAC, coherence)
15. [ ] Failure case analysis
16. [ ] Human iEEG deep dive

### Phase 5: Figures & Writing (Week 5-6)
17. [ ] Generate all main figures
18. [ ] Generate supplementary figures
19. [ ] Write methods section
20. [ ] Write results section

---

## 8. KEY CLAIMS TO SUPPORT

1. **"CondUNet achieves state-of-the-art neural signal translation"**
   - Evidence: Correlation > 0.6 on held-out sessions, beats all baselines

2. **"Cross-frequency attention is critical for performance"**
   - Evidence: Ablation showing significant drop without it

3. **"Method generalizes across brain regions and species"**
   - Evidence: Works on rodent (OB, PCx, PFC, CA1) and human (AMY, HPC, MFC)

4. **"Spectral content is faithfully preserved"**
   - Evidence: PSD error < 2dB across frequency bands

5. **"Model captures biologically relevant features"**
   - Evidence: PAC preservation, coherence maintenance

---

## 9. POTENTIAL REVIEWER CONCERNS

1. **"Is this just curve fitting?"**
   - Response: Cross-session generalization proves learned mapping is robust

2. **"Why not use simpler linear methods?"**
   - Response: Ablation shows nonlinear model significantly outperforms

3. **"Does it work on human data?"**
   - Response: DANDI results demonstrate cross-species generalization

4. **"What about different electrode configurations?"**
   - Response: Works with variable channel counts (8-64 channels)

5. **"Computational requirements?"**
   - Response: Trains in X hours on single GPU, inference in real-time

---

## 10. TIMELINE

| Week | Tasks |
|------|-------|
| 1-2  | Core results, cross-dataset validation |
| 2-3  | Ablation experiments |
| 3-4  | Baseline comparisons |
| 4-5  | Analysis & interpretation |
| 5-6  | Figure generation, writing |
| 7-8  | Revision, supplementary materials |
| 9    | Internal review, final polish |
| 10   | Submission |

