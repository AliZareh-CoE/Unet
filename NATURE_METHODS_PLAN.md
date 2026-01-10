# Nature Methods Publication Plan: Neural Signal Translation

## Executive Summary

**Title:** "CondUNet: Cross-Regional Neural Signal Translation via Conditional U-Net with Cross-Frequency Attention"

**Datasets:**
| Dataset | Species | Regions | Channels | Sessions | Duration |
|---------|---------|---------|----------|----------|----------|
| Olfactory | Rodent | OB → PCx | 32 → 32 | Multiple | 5s trials, 7 odors |
| PFC-CA1 | Rodent | PFC → CA1 | 64 → 32 | 494 trials | 5s, Left/Right task |
| PCx1 | Rodent | OB → PCx | 32 → 32 | Continuous | Hours of recording |
| DANDI 000623 | Human | AMY/HPC/MFC | Variable | 18 subjects | 8min movie |

**Baselines Available:**
- Classical: Wiener Filter, Ridge Regression, VAR Model (7 variants)
- Neural: Linear, SimpleCNN, WaveNet, FNet, ViT1D, Performer, Mamba (7 architectures)

---

# PHASE 1: CLASSICAL BASELINE FLOOR
## "How hard is this problem?"

### Objective
Establish performance floor with classical methods. All neural methods must beat this.

### Experiments to Run

```bash
# Run all classical baselines on all datasets
cd experiments/tier0_classical

# Olfactory dataset
python run_tier0.py --dataset olfactory --output results/tier0_olfactory/

# PCx1 continuous
python run_tier0.py --dataset pcx1 --output results/tier0_pcx1/

# PFC-CA1
python run_tier0.py --dataset pfc --output results/tier0_pfc/

# DANDI human
python run_tier0.py --dataset dandi --output results/tier0_dandi/
```

### Baselines to Compare
| Method | Description | Variants |
|--------|-------------|----------|
| Wiener Filter | Optimal linear filter | Single-channel, MIMO |
| Ridge Regression | L2-regularized | Standard, Temporal, CV |
| VAR Model | Autoregressive | Standard, Exogenous |

### Phase 1 Figures

#### Figure P1.1: Classical Baseline Performance Matrix
```
┌─────────────────────────────────────────────────────────────┐
│  CLASSICAL BASELINE COMPARISON ACROSS DATASETS              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) Heatmap: R² scores                                     │
│      Rows: 7 baselines                                      │
│      Cols: 4 datasets                                       │
│      Color: Performance (0 to 1)                            │
│                                                             │
│  (B) Bar plot: Best classical per dataset                   │
│      X: Datasets                                            │
│      Y: R² with 95% CI                                      │
│      Annotations: Method name on each bar                   │
│                                                             │
│  (C) Radar plot: Per-frequency-band performance             │
│      5 axes: Delta, Theta, Alpha, Beta, Gamma               │
│      Lines: Best classical method per dataset               │
│                                                             │
│  (D) Computation time comparison                            │
│      Training time vs inference time                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Figure P1.2: Example Predictions - Classical Methods
```
┌─────────────────────────────────────────────────────────────┐
│  CLASSICAL METHOD PREDICTIONS                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) Olfactory: 3 channels × 5 seconds                      │
│      Row 1: Ground truth                                    │
│      Row 2: Wiener prediction                               │
│      Row 3: Ridge prediction                                │
│      Row 4: Residual (error)                                │
│                                                             │
│  (B) PSD comparison: Predicted vs Actual                    │
│      Overlay plots per frequency band                       │
│                                                             │
│  (C) Scatter: Predicted vs Actual amplitude                 │
│      Per dataset, with R² annotation                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 1 Deliverables
- [ ] Table: All baseline results with 95% CI
- [ ] Best classical baseline identified per dataset
- [ ] Performance floor established (GATE for Phase 2)
- [ ] 2 figures generated

---

# PHASE 2: NEURAL ARCHITECTURE SCREENING
## "Which neural architecture works best?"

### Objective
Screen 7 neural architectures against classical floor. Identify top performers.

### Experiments to Run

```bash
# Run architecture screening
cd experiments/tier1_screening

# Test all architectures on each dataset
for arch in linear simplecnn wavenet fnet vit performer mamba conunet; do
  for dataset in olfactory pcx1 pfc dandi; do
    python run_tier1.py --arch $arch --dataset $dataset --seeds 42,43,44
  done
done
```

### Architectures to Compare
| Architecture | Type | Key Feature |
|--------------|------|-------------|
| Linear | Baseline | Simple linear mapping |
| SimpleCNN | Convolutional | Basic conv layers |
| WaveNet1D | Dilated Conv | Causal dilated convolutions |
| FNet1D | Fourier | FFT-based mixing |
| ViT1D | Transformer | Self-attention |
| Performer1D | Efficient Transformer | Linear attention |
| Mamba1D | State Space | SSM-based |
| **CondUNet** | **U-Net** | **Skip connections + conditioning** |

### Phase 2 Figures

#### Figure P2.1: Architecture Comparison
```
┌─────────────────────────────────────────────────────────────┐
│  NEURAL ARCHITECTURE SCREENING                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) Grouped bar plot: R² by architecture                   │
│      Groups: 4 datasets                                     │
│      Bars: 8 architectures                                  │
│      Error bars: 95% CI from 3 seeds                        │
│      Horizontal line: Classical baseline floor              │
│                                                             │
│  (B) Improvement over classical baseline                    │
│      Delta R² with significance stars                       │
│      Color: Green (significant) / Gray (not)                │
│                                                             │
│  (C) Ranking plot: Mean rank across datasets                │
│      Box plot showing rank distribution                     │
│                                                             │
│  (D) Parameter count vs Performance                         │
│      Scatter: X=params, Y=R², size=inference time           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Figure P2.2: Architecture Deep Dive
```
┌─────────────────────────────────────────────────────────────┐
│  TOP 3 ARCHITECTURES DETAILED COMPARISON                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) Learning curves: Loss over epochs                      │
│      3 architectures × 4 datasets                           │
│      Train (solid) vs Val (dashed)                          │
│                                                             │
│  (B) Per-frequency performance                              │
│      Grouped bars: Delta/Theta/Alpha/Beta/Gamma             │
│      Groups: Top 3 architectures                            │
│                                                             │
│  (C) Example predictions: Best vs Worst architecture        │
│      Side-by-side trace comparison                          │
│                                                             │
│  (D) Statistical test matrix                                │
│      Pairwise Wilcoxon tests                                │
│      Bonferroni corrected p-values                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2 Deliverables
- [ ] Full comparison table with all metrics
- [ ] Top 3 architectures identified
- [ ] Statistical significance tests (vs baseline, pairwise)
- [ ] 2 figures generated
- [ ] GATE: At least one architecture beats classical by R² ≥ 0.10

---

# PHASE 3: CONDUNET ABLATION STUDIES
## "What makes CondUNet work?"

### Objective
Systematic ablation of CondUNet components to justify design decisions.

### Experiments to Run

```bash
# Ablation studies (Tier 3)
cd experiments/tier3_ablation

# Attention ablation
python run_ablation.py --ablation attention \
  --variants none,basic,cross_freq,cross_freq_v2 \
  --datasets olfactory,pcx1,pfc,dandi --seeds 42,43,44

# Conditioning ablation
python run_ablation.py --ablation conditioning \
  --variants none,odor_onehot,spectro_temporal,cpc,vqvae \
  --datasets olfactory,pcx1,pfc,dandi --seeds 42,43,44

# Loss function ablation
python run_ablation.py --ablation loss \
  --variants l1,huber,wavelet,l1_wavelet,huber_wavelet \
  --datasets olfactory,pcx1,pfc,dandi --seeds 42,43,44

# Depth ablation
python run_ablation.py --ablation depth \
  --variants 3,4,5 \
  --datasets olfactory,pcx1,pfc,dandi --seeds 42,43,44

# Bidirectional ablation
python run_ablation.py --ablation bidirectional \
  --variants unidirectional,bidirectional \
  --datasets olfactory,pcx1,pfc,dandi --seeds 42,43,44

# Augmentation ablation
python run_ablation.py --ablation augmentation \
  --variants none,full \
  --datasets olfactory,pcx1,pfc,dandi --seeds 42,43,44
```

### Ablation Components
| Component | Variants | Hypothesis |
|-----------|----------|------------|
| Attention | none, basic, cross_freq, cross_freq_v2 | Cross-freq attention captures theta-gamma coupling |
| Conditioning | none, odor, spectro_temporal | Self-conditioning enables label-free training |
| Loss | L1, Huber, Wavelet, Combined | Multi-scale wavelet preserves spectral content |
| Depth | 3, 4, 5 layers | 4 layers optimal for 5s windows |
| Bidirectional | uni, bi | Cycle consistency improves both directions |
| Augmentation | none, full | Augmentation critical for generalization |

### Phase 3 Figures

#### Figure P3.1: Attention Mechanism Ablation
```
┌─────────────────────────────────────────────────────────────┐
│  ATTENTION MECHANISM ABLATION                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) Bar plot: R² by attention type                         │
│      4 variants × 4 datasets                                │
│      Significance stars vs "none"                           │
│                                                             │
│  (B) Attention weight visualization                         │
│      Heatmap: Query vs Key positions                        │
│      For cross_freq_v2 on example input                     │
│                                                             │
│  (C) Frequency band preservation                            │
│      Improvement in gamma band with cross_freq attention    │
│                                                             │
│  (D) Effect size (Cohen's d) summary                        │
│      Forest plot with CI                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Figure P3.2: Loss Function Ablation
```
┌─────────────────────────────────────────────────────────────┐
│  LOSS FUNCTION ABLATION                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) Overall R² comparison                                  │
│      5 loss functions × 4 datasets                          │
│                                                             │
│  (B) PSD error comparison                                   │
│      Wavelet losses should have lower PSD error             │
│                                                             │
│  (C) Per-frequency breakdown                                │
│      Which loss works best for which band?                  │
│                                                             │
│  (D) Trade-off plot: R² vs PSD error                        │
│      Pareto frontier visualization                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Figure P3.3: Component Removal Impact
```
┌─────────────────────────────────────────────────────────────┐
│  COMPONENT REMOVAL SUMMARY                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) Waterfall chart: Performance drop per removal          │
│      Full model → -attention → -conditioning → etc.         │
│                                                             │
│  (B) Interaction effects                                    │
│      Heatmap: Pairwise removal effects                      │
│      Are effects additive or synergistic?                   │
│                                                             │
│  (C) Critical components identified                         │
│      Ranked by Cohen's d effect size                        │
│                                                             │
│  (D) Final recipe: Minimal effective configuration          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3 Deliverables
- [ ] Complete ablation table (6 ablations × 4 datasets × 3 seeds)
- [ ] Effect sizes for all components
- [ ] Justified architectural decisions
- [ ] 3 figures generated

---

# PHASE 4: CROSS-SESSION GENERALIZATION
## "Does it generalize to unseen recordings?"

### Objective
Prove the model learns generalizable mappings, not session-specific patterns.

### Experiments to Run

```bash
# Session holdout experiments
for dataset in olfactory pcx1 pfc; do
  # Leave-one-session-out
  python train.py --dataset $dataset \
    --split-by-session --no-test-set \
    --separate-val-sessions \
    --epochs 60 --seed 42

  # Multiple held-out sessions
  for n_val in 1 2 4 6; do
    python train.py --dataset $dataset \
      --split-by-session --n-val-sessions $n_val \
      --epochs 60 --seed 42
  done
done

# DANDI: Leave-one-subject-out
python train.py --dataset dandi \
  --split-by-session --no-test-set \
  --epochs 60 --seed 42
```

### Phase 4 Figures

#### Figure P4.1: Cross-Session Performance
```
┌─────────────────────────────────────────────────────────────┐
│  CROSS-SESSION GENERALIZATION                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) Per-session performance heatmap                        │
│      Rows: Training configuration                           │
│      Cols: Test sessions                                    │
│      Color: R² score                                        │
│                                                             │
│  (B) Within-session vs Cross-session comparison             │
│      Paired bar plot with significance test                 │
│      Quantify generalization gap                            │
│                                                             │
│  (C) Session difficulty analysis                            │
│      Which sessions are hard? Why?                          │
│      Scatter: Session R² vs session characteristics         │
│                                                             │
│  (D) Learning with more sessions                            │
│      Line plot: R² vs number of training sessions           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Figure P4.2: Human Subject Generalization (DANDI)
```
┌─────────────────────────────────────────────────────────────┐
│  HUMAN iEEG: CROSS-SUBJECT TRANSFER                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) Leave-one-subject-out results                          │
│      18 subjects, per-subject R²                            │
│      Box plot with individual points                        │
│                                                             │
│  (B) Subject similarity analysis                            │
│      Dendrogram: Which subjects cluster together?           │
│      Does anatomical similarity predict transfer?           │
│                                                             │
│  (C) Channel count effect                                   │
│      Scatter: #channels vs R²                               │
│      Does more electrodes help?                             │
│                                                             │
│  (D) Brain region comparison                                │
│      AMY→HPC vs HPC→MFC vs AMY→MFC                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 4 Deliverables
- [ ] Cross-session generalization quantified
- [ ] Generalization gap measured (within vs across)
- [ ] Human subject leave-one-out validation
- [ ] 2 figures generated

---

# PHASE 5: SPECTRAL & PHASE FIDELITY
## "Does it preserve neural dynamics?"

### Objective
Verify the model preserves biologically meaningful spectral and phase relationships.

### Experiments to Run

```bash
# Compute comprehensive neural metrics
python evaluate.py --checkpoint best_model.pt \
  --metrics spectral,phase,pac \
  --datasets olfactory,pcx1,pfc,dandi
```

### Metrics to Compute
| Metric | Description | Biological Relevance |
|--------|-------------|---------------------|
| PSD Error | Power spectral density match | Oscillation strength preserved |
| PLV | Phase locking value | Inter-regional synchrony |
| PLI | Phase lag index | Directed coupling |
| PAC | Phase-amplitude coupling | Theta-gamma nesting |
| Coherence | Frequency-resolved correlation | Band-specific coupling |

### Phase 5 Figures

#### Figure P5.1: Spectral Fidelity
```
┌─────────────────────────────────────────────────────────────┐
│  SPECTRAL CONTENT PRESERVATION                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) PSD overlay: Predicted vs Actual                       │
│      Log-log plot, 1-200 Hz                                 │
│      Shaded: 95% CI across trials                           │
│      Per dataset subplot                                    │
│                                                             │
│  (B) PSD error by frequency band                            │
│      Bar plot: Delta/Theta/Alpha/Beta/Gamma                 │
│      Grouped by dataset                                     │
│                                                             │
│  (C) Spectrogram comparison                                 │
│      Time-frequency representation                          │
│      Actual | Predicted | Difference                        │
│                                                             │
│  (D) 1/f slope preservation                                 │
│      Scatter: Actual vs Predicted slope                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Figure P5.2: Phase Relationship Preservation
```
┌─────────────────────────────────────────────────────────────┐
│  PHASE DYNAMICS PRESERVATION                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) PLV comparison: Actual vs Predicted                    │
│      Scatter plot with identity line                        │
│      Per frequency band                                     │
│                                                             │
│  (B) Phase-amplitude coupling (PAC)                         │
│      Comodulogram: Actual | Predicted                       │
│      Theta phase × Gamma amplitude                          │
│                                                             │
│  (C) PAC preservation ratio                                 │
│      Bar plot: How much PAC is preserved?                   │
│      Comparison to baseline methods                         │
│                                                             │
│  (D) Coherence matrices                                     │
│      Channel × Channel coherence                            │
│      Actual | Predicted | Correlation                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 5 Deliverables
- [ ] All spectral metrics computed
- [ ] Phase metrics (PLV, PLI, PAC) computed
- [ ] Comparison to baselines on phase preservation
- [ ] 2 figures generated

---

# PHASE 6: FULL BASELINE COMPARISON
## "How does CondUNet compare to everything?"

### Objective
Comprehensive comparison against all classical and neural baselines.

### Experiments to Run

```bash
# Run all methods on all datasets
# Classical (7) + Neural (7) + CondUNet variants (3) = 17 methods

# All baselines
for method in wiener ridge_cv var linear simplecnn wavenet fnet vit performer mamba; do
  for dataset in olfactory pcx1 pfc dandi; do
    python run_baseline.py --method $method --dataset $dataset --seeds 42,43,44
  done
done

# CondUNet variants
for variant in conunet conunet_small conunet_large; do
  for dataset in olfactory pcx1 pfc dandi; do
    python train.py --dataset $dataset --model $variant --seeds 42,43,44
  done
done
```

### Phase 6 Figures

#### Figure P6.1: Grand Comparison
```
┌─────────────────────────────────────────────────────────────┐
│  COMPREHENSIVE METHOD COMPARISON                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) Overall ranking                                        │
│      Stacked bar: Mean R² across all datasets               │
│      17 methods ordered by performance                      │
│      Error bars: 95% CI                                     │
│      Color: Classical (blue) / Neural (orange) / Ours (red) │
│                                                             │
│  (B) Per-dataset comparison                                 │
│      4 subplots (one per dataset)                           │
│      Top 5 methods highlighted                              │
│                                                             │
│  (C) Metric trade-offs                                      │
│      Scatter: R² vs PSD error                               │
│      Scatter: R² vs Inference time                          │
│      Pareto frontier marked                                 │
│                                                             │
│  (D) Statistical significance matrix                        │
│      17×17 pairwise comparison                              │
│      Significant wins/losses marked                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Figure P6.2: Best Methods Deep Dive
```
┌─────────────────────────────────────────────────────────────┐
│  TOP 5 METHODS DETAILED ANALYSIS                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) Example predictions side-by-side                       │
│      5 methods on same trial                                │
│      Ground truth + 5 predictions + residuals               │
│                                                             │
│  (B) Per-frequency performance                              │
│      Line plot: R² across frequency bands                   │
│      5 methods overlaid                                     │
│                                                             │
│  (C) Failure case analysis                                  │
│      When does each method fail?                            │
│      Low-SNR, artifacts, specific conditions                │
│                                                             │
│  (D) Computational requirements                             │
│      Table: Params / Train time / Inference time / Memory   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 6 Deliverables
- [ ] Full comparison table (17 methods × 4 datasets)
- [ ] Statistical tests (all pairwise)
- [ ] CondUNet shown to be best or among best
- [ ] 2 figures generated

---

# PHASE 7: INTERPRETABILITY & BIOLOGICAL VALIDATION
## "What has the model learned?"

### Objective
Interpret model behavior and validate biological relevance.

### Experiments to Run

```bash
# Channel importance analysis
python analyze.py --checkpoint best_model.pt \
  --analysis channel_importance,frequency_importance,temporal_importance

# Biological validation
python analyze.py --checkpoint best_model.pt \
  --analysis odor_decoding,phase_preservation,information_transfer
```

### Phase 7 Figures

#### Figure P7.1: What the Model Learns
```
┌─────────────────────────────────────────────────────────────┐
│  MODEL INTERPRETABILITY                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) Channel importance map                                 │
│      Which source channels drive predictions?               │
│      Heatmap: Source channel × Target channel               │
│                                                             │
│  (B) Frequency band importance                              │
│      Bar plot: Contribution of each band                    │
│      Ablation: Mask each band, measure drop                 │
│                                                             │
│  (C) Temporal importance                                    │
│      Line plot: When in trial matters most?                 │
│      Gradient-based saliency over time                      │
│                                                             │
│  (D) Attention pattern analysis                             │
│      What does cross-frequency attention focus on?          │
│      Theta-gamma coupling visualization                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Figure P7.2: Biological Validation
```
┌─────────────────────────────────────────────────────────────┐
│  BIOLOGICAL RELEVANCE                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  (A) Odor information preservation                          │
│      Train classifier on predicted signals                  │
│      Compare: Actual vs Predicted decoding accuracy         │
│                                                             │
│  (B) Negative control: Shuffled labels                      │
│      Model trained with shuffled correspondences            │
│      Should fail → proves real mapping learned              │
│                                                             │
│  (C) Temporal structure preservation                        │
│      Cross-correlation: Actual vs Predicted                 │
│      Lag analysis: Is timing preserved?                     │
│                                                             │
│  (D) Information theoretic analysis                         │
│      Mutual information: Source-Target vs Source-Predicted  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 7 Deliverables
- [ ] Interpretability analysis complete
- [ ] Negative controls passed
- [ ] Biological relevance demonstrated
- [ ] 2 figures generated

---

# MAIN PAPER FIGURES (6 figures)

After all phases, combine into main paper figures:

## Figure 1: Method Overview
- Architecture schematic (from Phase 3)
- Training pipeline
- Dataset overview table

## Figure 2: Main Results
- CondUNet performance on all 4 datasets (from Phase 6)
- Example predictions (best cases)
- Key metrics summary

## Figure 3: Comparison to Baselines
- Classical baseline comparison (from Phase 1 & 6)
- Neural baseline comparison (from Phase 2 & 6)
- Statistical significance

## Figure 4: Ablation Studies
- Component importance (from Phase 3)
- Loss function comparison
- Attention mechanism impact

## Figure 5: Generalization
- Cross-session (from Phase 4)
- Cross-subject/human data (from Phase 4)
- Cross-region (DANDI variants)

## Figure 6: Biological Validation
- Spectral fidelity (from Phase 5)
- Phase preservation (from Phase 5)
- Interpretability (from Phase 7)

---

# SUPPLEMENTARY MATERIALS

## Supp. Tables
1. Full hyperparameter table
2. Complete baseline results (all metrics)
3. Per-session breakdown
4. Statistical test results

## Supp. Figures
1. Extended architecture details
2. All ablation results
3. Learning curves for all experiments
4. Failure cases analysis
5. Additional DANDI region pairs
6. Computational benchmarks

---

# EXECUTION COMMANDS SUMMARY

```bash
# Phase 1: Classical Baselines
python experiments/tier0_classical/run_tier0.py --all-datasets

# Phase 2: Architecture Screening
python experiments/tier1_screening/run_tier1.py --all-datasets

# Phase 3: Ablations
python experiments/tier3_ablation/run_tier3.py --all-datasets

# Phase 4: Cross-Session
python train.py --dataset pcx1 --split-by-session --no-test-set --separate-val-sessions

# Phase 5: Spectral Analysis
python evaluate.py --comprehensive --metrics spectral,phase,pac

# Phase 6: Full Comparison
python experiments/run_full_comparison.py --all-methods --all-datasets

# Phase 7: Interpretability
python analyze.py --interpretability --biological-validation
```

---

# TIMELINE

| Phase | Duration | Experiments | Figures |
|-------|----------|-------------|---------|
| Phase 1 | 2-3 days | 7 baselines × 4 datasets | 2 |
| Phase 2 | 3-4 days | 8 architectures × 4 datasets | 2 |
| Phase 3 | 5-7 days | 6 ablations × multiple variants | 3 |
| Phase 4 | 3-4 days | Session holdout experiments | 2 |
| Phase 5 | 2-3 days | Spectral/phase analysis | 2 |
| Phase 6 | 3-4 days | Full comparison | 2 |
| Phase 7 | 3-4 days | Interpretability | 2 |
| **Total** | **~4 weeks** | | **15 figures** |

---

# STATISTICAL REQUIREMENTS

For each comparison:
- [ ] 3+ random seeds
- [ ] 95% bootstrap confidence intervals
- [ ] Paired statistical tests (Wilcoxon signed-rank)
- [ ] Multiple comparison correction (Bonferroni)
- [ ] Effect sizes (Cohen's d)
- [ ] All p-values reported
