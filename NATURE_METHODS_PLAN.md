# Nature Methods Publication Plan: Neural Signal Translation

## Study Structure Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STUDY STRUCTURE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PHASE 1-3: CORE METHOD DEVELOPMENT                                      │
│  ══════════════════════════════════                                      │
│  Dataset: Olfactory (OB→PCx, fixed 5s windows, trial-based)              │
│  • Phase 1: Classical Baseline Floor                                     │
│  • Phase 2: Neural Architecture Screening                                │
│  • Phase 3: CondUNet Ablation Studies                                    │
│                                                                          │
│  PHASE 4: GENERALIZATION - INTER vs INTRA SESSION                        │
│  ════════════════════════════════════════════════                        │
│  Dataset: Olfactory (same dataset, different splits)                     │
│  • Intra-session: Random split within sessions                           │
│  • Inter-session: Held-out entire sessions (cross-session)               │
│                                                                          │
│  PHASE 5: GENERALIZATION - CROSS-DATASET                                 │
│  ═══════════════════════════════════════                                 │
│  Datasets: PFC→CA1 (rodent), DANDI (human iEEG)                          │
│  • All fixed window (same approach, different brain regions/species)     │
│                                                                          │
│  PHASE 6: REAL-TIME FEASIBILITY - CONTINUOUS                             │
│  ═══════════════════════════════════════════                             │
│  Dataset: PCx1 continuous recordings                                     │
│  • Sliding window training and inference                                 │
│  • Latency and throughput benchmarks                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# PRIMARY DATASET: Olfactory (OB → PCx)

| Property | Value |
|----------|-------|
| Species | Rodent |
| Source Region | Olfactory Bulb (OB) |
| Target Region | Piriform Cortex (PCx) |
| Channels | 32 → 32 |
| Sampling Rate | 1000 Hz |
| Window | 5 seconds (5000 samples) |
| Trial Types | 7 odors |
| Format | Fixed windows, trial-based |

---

# PHASE 1: CLASSICAL BASELINE FLOOR
## Dataset: Olfactory | Mode: Fixed Window

### Objective
Establish performance floor. All neural methods must beat this.

### Experiments

```bash
# Run all classical baselines on olfactory dataset
cd experiments/tier0_classical

python run_tier0.py --dataset olfactory --output results/phase1/

# Individual baselines for debugging
python run_baseline.py --method wiener --dataset olfactory
python run_baseline.py --method wiener_mimo --dataset olfactory
python run_baseline.py --method ridge --dataset olfactory
python run_baseline.py --method ridge_temporal --dataset olfactory
python run_baseline.py --method ridge_cv --dataset olfactory
python run_baseline.py --method var --dataset olfactory
python run_baseline.py --method var_exog --dataset olfactory
```

### Methods (7 total)
| Method | Type | Description |
|--------|------|-------------|
| Wiener | Linear | Single-channel optimal filter |
| Wiener MIMO | Linear | Multi-input multi-output |
| Ridge | Linear | L2-regularized regression |
| Ridge Temporal | Linear | With temporal features |
| Ridge CV | Linear | Cross-validated regularization |
| VAR | Autoregressive | Vector autoregressive model |
| VAR Exogenous | Autoregressive | VAR with input signals |

### Figure P1.1: Classical Baseline Comparison
```
┌──────────────────────────────────────────────────────────────┐
│  CLASSICAL BASELINE PERFORMANCE (Olfactory Dataset)          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  (A) Bar plot: R² for each method                            │
│      7 bars with 95% CI                                      │
│      Sorted by performance                                   │
│      Best method highlighted                                 │
│                                                              │
│  (B) Per-frequency-band breakdown                            │
│      Grouped bars: Delta/Theta/Alpha/Beta/Gamma              │
│      Top 3 classical methods                                 │
│                                                              │
│  (C) Example predictions                                     │
│      3 channels × 5 seconds                                  │
│      Ground truth vs Best classical                          │
│      Residual plot below                                     │
│                                                              │
│  (D) PSD comparison                                          │
│      Actual vs Predicted power spectrum                      │
│      Log-log scale, 1-200 Hz                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Deliverables
- [ ] All 7 baselines evaluated (5-fold CV, 3 seeds)
- [ ] Best classical method identified: ____________
- [ ] Best R²: ______, Best Correlation: ______
- [ ] Performance floor established for GATE

### GATE Criterion
**Neural methods must beat best classical R² by ≥ 0.05**

---

# PHASE 2: NEURAL ARCHITECTURE SCREENING
## Dataset: Olfactory | Mode: Fixed Window

### Objective
Screen neural architectures. Identify best performers vs classical floor.

### Experiments

```bash
# Run all neural architectures
cd experiments/tier1_screening

# Full screening (all architectures, 3 seeds each)
for arch in linear simplecnn wavenet fnet vit performer mamba condunet; do
  for seed in 42 43 44; do
    python train.py --dataset olfactory \
      --arch $arch \
      --epochs 60 \
      --seed $seed \
      --output results/phase2/${arch}_seed${seed}/
  done
done
```

### Architectures (8 total)
| Architecture | Type | Parameters | Key Feature |
|--------------|------|------------|-------------|
| Linear | Baseline | ~100K | Simple linear mapping |
| SimpleCNN | CNN | ~500K | Basic conv layers |
| WaveNet1D | Dilated CNN | ~1M | Causal dilated convolutions |
| FNet1D | Fourier | ~800K | FFT-based token mixing |
| ViT1D | Transformer | ~2M | Full self-attention |
| Performer1D | Efficient Trans | ~1.5M | Linear attention (FAVOR+) |
| Mamba1D | State Space | ~1M | Selective SSM |
| **CondUNet** | **U-Net** | **~2M** | **Skip connections + conditioning** |

### Figure P2.1: Architecture Comparison
```
┌──────────────────────────────────────────────────────────────┐
│  NEURAL ARCHITECTURE SCREENING (Olfactory Dataset)           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  (A) Bar plot: R² by architecture                            │
│      8 architectures, sorted by performance                  │
│      Error bars: std across 3 seeds                          │
│      Horizontal line: Classical baseline floor               │
│      Color: Beat baseline (green) / Below (gray)             │
│                                                              │
│  (B) Improvement over classical baseline                     │
│      ΔR² with significance stars (* p<0.05, ** p<0.01)       │
│      Only show methods that beat baseline                    │
│                                                              │
│  (C) Learning curves                                         │
│      Loss vs Epoch for top 3 architectures                   │
│      Train (solid) vs Val (dashed)                           │
│                                                              │
│  (D) Params vs Performance trade-off                         │
│      Scatter: X=parameters, Y=R²                             │
│      Size: inference time                                    │
│      Pareto frontier marked                                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Figure P2.2: Top Architectures Deep Dive
```
┌──────────────────────────────────────────────────────────────┐
│  TOP 3 ARCHITECTURES DETAILED COMPARISON                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  (A) Example predictions side-by-side                        │
│      Same trial: Ground truth + 3 predictions                │
│      3 channels shown                                        │
│                                                              │
│  (B) Per-frequency R²                                        │
│      Line plot across frequency bands                        │
│      3 architectures + classical baseline                    │
│                                                              │
│  (C) Pairwise statistical comparison                         │
│      3×3 matrix: Wilcoxon p-values                           │
│      Bonferroni corrected                                    │
│                                                              │
│  (D) Computational comparison table                          │
│      Params | Train time | Inference | Memory                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Deliverables
- [ ] All 8 architectures evaluated (3 seeds each = 24 runs)
- [ ] Top 3 architectures identified: 1.______ 2.______ 3.______
- [ ] Statistical tests vs classical baseline
- [ ] CondUNet ranking: ______

### GATE Criterion
**At least one architecture beats classical by R² ≥ 0.05 (p < 0.05)**

---

# PHASE 3: CONDUNET ABLATION STUDIES
## Dataset: Olfactory | Mode: Fixed Window

### Objective
Justify every CondUNet design decision with ablation evidence.

### Experiments

```bash
# Ablation experiments (Tier 3)
cd experiments/tier3_ablation

# 1. Attention mechanism ablation
for attn in none basic cross_freq cross_freq_v2; do
  for seed in 42 43 44; do
    python train.py --dataset olfactory \
      --attention-type $attn \
      --epochs 60 --seed $seed
  done
done

# 2. Conditioning ablation
for cond in none odor_onehot spectro_temporal; do
  for seed in 42 43 44; do
    python train.py --dataset olfactory \
      --conditioning $cond \
      --epochs 60 --seed $seed
  done
done

# 3. Loss function ablation
for loss in l1 huber wavelet l1_wavelet huber_wavelet; do
  for seed in 42 43 44; do
    python train.py --dataset olfactory \
      --loss $loss \
      --epochs 60 --seed $seed
  done
done

# 4. Model capacity ablation
for channels in 32 64 128; do
  for seed in 42 43 44; do
    python train.py --dataset olfactory \
      --base-channels $channels \
      --epochs 60 --seed $seed
  done
done

# 5. Bidirectional training ablation
for seed in 42 43 44; do
  python train.py --dataset olfactory --epochs 60 --seed $seed  # bidirectional (default)
  python train.py --dataset olfactory --no-bidirectional --epochs 60 --seed $seed
done

# 6. Data augmentation ablation
for seed in 42 43 44; do
  python train.py --dataset olfactory --epochs 60 --seed $seed  # with aug (default)
  python train.py --dataset olfactory --no-aug --epochs 60 --seed $seed
done
```

### Ablation Summary
| Ablation | Variants | Hypothesis |
|----------|----------|------------|
| Attention | none, basic, cross_freq, cross_freq_v2 | Cross-freq captures theta-gamma coupling |
| Conditioning | none, odor, spectro_temporal | Self-conditioning enables unsupervised learning |
| Loss | L1, Huber, Wavelet, Combined | Multi-scale wavelet preserves spectrum |
| Capacity | 32, 64, 128 channels | 64 is sweet spot |
| Bidirectional | uni, bi | Cycle consistency regularizes |
| Augmentation | none, full | Aug critical for generalization |

**Total: 6 ablations × 3-5 variants × 3 seeds = ~60 runs**

### Figure P3.1: Attention Ablation
```
┌──────────────────────────────────────────────────────────────┐
│  ATTENTION MECHANISM ABLATION                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  (A) R² by attention type                                    │
│      4 bars: none → basic → cross_freq → cross_freq_v2       │
│      Error bars from 3 seeds                                 │
│      Significance stars vs "none"                            │
│                                                              │
│  (B) Per-frequency improvement                               │
│      Which bands benefit most from attention?                │
│      Grouped bars: Δ improvement per band                    │
│                                                              │
│  (C) Attention visualization                                 │
│      Heatmap: What does cross_freq_v2 attend to?             │
│      Example from validation set                             │
│                                                              │
│  (D) Effect size forest plot                                 │
│      Cohen's d with 95% CI for each variant vs none          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Figure P3.2: Loss Function Ablation
```
┌──────────────────────────────────────────────────────────────┐
│  LOSS FUNCTION ABLATION                                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  (A) R² comparison                                           │
│      5 loss functions                                        │
│                                                              │
│  (B) PSD error comparison                                    │
│      Wavelet losses should have lower spectral error         │
│                                                              │
│  (C) Trade-off: R² vs PSD error                              │
│      Scatter plot, Pareto frontier                           │
│      huber_wavelet should be on frontier                     │
│                                                              │
│  (D) Per-frequency breakdown                                 │
│      Which loss best for which band?                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Figure P3.3: Component Importance Summary
```
┌──────────────────────────────────────────────────────────────┐
│  ABLATION SUMMARY: COMPONENT IMPORTANCE                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  (A) Waterfall chart                                         │
│      Full model R² → remove each component                   │
│      Shows contribution of each part                         │
│                                                              │
│  (B) Ranked importance                                       │
│      Bar plot: Cohen's d effect size                         │
│      Sorted by importance                                    │
│      1. Attention  2. Loss  3. Conditioning  etc.            │
│                                                              │
│  (C) Interaction effects (optional)                          │
│      Does removing A+B hurt more than A alone + B alone?     │
│                                                              │
│  (D) Final recipe table                                      │
│      Recommended configuration with justification            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Deliverables
- [ ] 6 ablation studies completed (~60 runs)
- [ ] Effect sizes computed for all comparisons
- [ ] Optimal configuration identified
- [ ] All design decisions justified with p-values

---

# PHASE 4: GENERALIZATION - INTER vs INTRA SESSION
## Dataset: Olfactory | Mode: Fixed Window

### Objective
Compare within-session vs cross-session generalization.
**Key question: Does the model learn session-specific patterns or generalizable mappings?**

### Experimental Design

```
┌─────────────────────────────────────────────────────────────┐
│  INTRA-SESSION (Within Session)                             │
│  ═══════════════════════════════                            │
│  • Random 80/20 split WITHIN each session                   │
│  • Train and test data from SAME recording sessions         │
│  • Easier: Same electrode positions, same day               │
├─────────────────────────────────────────────────────────────┤
│  INTER-SESSION (Cross Session)                              │
│  ═══════════════════════════════                            │
│  • Hold out ENTIRE sessions for validation                  │
│  • Train on sessions A,B,C → Test on session D              │
│  • Harder: Different days, potential electrode drift        │
└─────────────────────────────────────────────────────────────┘
```

### Experiments

```bash
# INTRA-SESSION: Random split within sessions
for seed in 42 43 44; do
  python train.py --dataset olfactory \
    --no-split-by-session \
    --epochs 60 --seed $seed \
    --output results/phase4/intra_session/seed${seed}/
done

# INTER-SESSION: Hold out entire sessions
for seed in 42 43 44; do
  python train.py --dataset olfactory \
    --split-by-session \
    --no-test-set \
    --separate-val-sessions \
    --epochs 60 --seed $seed \
    --output results/phase4/inter_session/seed${seed}/
done

# Vary number of held-out sessions
for n_val in 1 2 3 4; do
  python train.py --dataset olfactory \
    --split-by-session \
    --n-val-sessions $n_val \
    --no-test-set \
    --epochs 60 --seed 42 \
    --output results/phase4/inter_session_n${n_val}/
done
```

### Figure P4.1: Intra vs Inter Session Comparison
```
┌──────────────────────────────────────────────────────────────┐
│  GENERALIZATION: INTRA vs INTER SESSION                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  (A) Bar plot comparison                                     │
│      Two bars: Intra-session vs Inter-session                │
│      Error bars from 3 seeds                                 │
│      Significance test between them                          │
│      QUANTIFY THE GENERALIZATION GAP                         │
│                                                              │
│  (B) Per-session performance (Inter-session)                 │
│      Box plot: Each held-out session                         │
│      Shows variance across sessions                          │
│      Identify easy vs hard sessions                          │
│                                                              │
│  (C) Learning curve by # training sessions                   │
│      X: Number of training sessions                          │
│      Y: Val R² on held-out sessions                          │
│      Does more data help?                                    │
│                                                              │
│  (D) Classical vs Neural on both settings                    │
│      Grouped bars: Intra/Inter × Classical/CondUNet          │
│      Does neural advantage hold for cross-session?           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Figure P4.2: Session Difficulty Analysis
```
┌──────────────────────────────────────────────────────────────┐
│  SESSION DIFFICULTY ANALYSIS                                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  (A) Session performance heatmap                             │
│      Rows: Training configuration                            │
│      Cols: Test session                                      │
│      Color: R² score                                         │
│                                                              │
│  (B) Session characteristics vs performance                  │
│      Scatter plots:                                          │
│      - # trials vs R²                                        │
│      - Recording date vs R²                                  │
│      - Signal quality vs R²                                  │
│                                                              │
│  (C) Transfer matrix                                         │
│      Train on session i, test on session j                   │
│      Which sessions transfer well to each other?             │
│                                                              │
│  (D) Failure case examples                                   │
│      Worst predictions from hardest session                  │
│      What went wrong?                                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Deliverables
- [ ] Intra-session R²: ______ ± ______
- [ ] Inter-session R²: ______ ± ______
- [ ] Generalization gap: ______ (should be small if model generalizes)
- [ ] Per-session breakdown
- [ ] Session difficulty analysis

### Key Metrics
| Metric | Intra-Session | Inter-Session | Gap |
|--------|---------------|---------------|-----|
| R² | | | |
| Correlation | | | |
| PSD Error | | | |

---

# PHASE 5: GENERALIZATION - CROSS-DATASET
## Datasets: PFC→CA1, DANDI | Mode: Fixed Window

### Objective
Test if method generalizes to different brain regions and species.
**No retraining of hyperparameters - use same config as olfactory.**

### Experiments

```bash
# PFC → CA1 (Rodent hippocampus)
for seed in 42 43 44; do
  python train.py --dataset pfc \
    --split-by-session \
    --no-test-set \
    --epochs 60 --seed $seed \
    --output results/phase5/pfc/seed${seed}/
done

# DANDI Human iEEG: AMY → HPC
for seed in 42 43 44; do
  python train.py --dataset dandi \
    --dandi-source-region amygdala \
    --dandi-target-region hippocampus \
    --split-by-session \
    --no-test-set \
    --epochs 60 --seed $seed \
    --output results/phase5/dandi_amy_hpc/seed${seed}/
done

# DANDI Human iEEG: Other region pairs
python train.py --dataset dandi \
  --dandi-source-region hippocampus \
  --dandi-target-region medial_frontal_cortex \
  --epochs 60 --seed 42

python train.py --dataset dandi \
  --dandi-source-region amygdala \
  --dandi-target-region medial_frontal_cortex \
  --epochs 60 --seed 42
```

### Dataset Comparison
| Dataset | Species | Regions | Channels | Challenge |
|---------|---------|---------|----------|-----------|
| Olfactory | Rodent | OB→PCx | 32→32 | Primary dataset |
| PFC-CA1 | Rodent | PFC→CA1 | 64→32 | Different regions |
| DANDI | **Human** | AMY→HPC | Variable | Cross-species |

### Figure P5.1: Cross-Dataset Generalization
```
┌──────────────────────────────────────────────────────────────┐
│  CROSS-DATASET GENERALIZATION                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  (A) R² across datasets                                      │
│      3 datasets: Olfactory, PFC, DANDI                       │
│      CondUNet vs Best Classical                              │
│      Error bars from seeds                                   │
│                                                              │
│  (B) Per-frequency breakdown by dataset                      │
│      Heatmap: Dataset × Frequency band                       │
│      Does gamma transfer across datasets?                    │
│                                                              │
│  (C) Human iEEG detailed results                             │
│      Per-subject performance (18 subjects)                   │
│      Box plot with individual points                         │
│                                                              │
│  (D) Region pair comparison (DANDI)                          │
│      AMY→HPC vs HPC→MFC vs AMY→MFC                           │
│      Which direction is easier?                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Figure P5.2: Human iEEG Deep Dive
```
┌──────────────────────────────────────────────────────────────┐
│  DANDI 000623: HUMAN iEEG RESULTS                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  (A) Leave-one-subject-out                                   │
│      18 subjects, cross-subject transfer                     │
│      Performance distribution                                │
│                                                              │
│  (B) Channel count effect                                    │
│      Scatter: # electrodes vs R²                             │
│      Does more channels help?                                │
│                                                              │
│  (C) Example human predictions                               │
│      Best and worst subject                                  │
│      AMY→HPC translation example                             │
│                                                              │
│  (D) Comparison to rodent                                    │
│      Same method, different species                          │
│      What transfers, what doesn't?                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Deliverables
- [ ] PFC→CA1 results: R² = ______ ± ______
- [ ] DANDI AMY→HPC results: R² = ______ ± ______
- [ ] Cross-species generalization demonstrated
- [ ] Human data validates method

---

# PHASE 6: REAL-TIME FEASIBILITY - CONTINUOUS
## Dataset: PCx1 Continuous | Mode: Sliding Window

### Objective
Demonstrate real-time applicability with continuous sliding window.
**Different from fixed trials - streaming data simulation.**

### Experimental Design

```
┌─────────────────────────────────────────────────────────────┐
│  FIXED WINDOW (Phases 1-5)        SLIDING WINDOW (Phase 6)  │
│  ════════════════════════         ════════════════════════  │
│                                                             │
│  |--trial 1--|--trial 2--|        |====|                    │
│                                     |====|                  │
│  Discrete, non-overlapping            |====|                │
│  Trial boundaries known                  |====|             │
│                                             |====|          │
│                                   Continuous, overlapping   │
│                                   No trial boundaries       │
│                                   Simulates real-time       │
└─────────────────────────────────────────────────────────────┘
```

### Experiments

```bash
# Sliding window with different strides
for stride_ratio in 0.25 0.5 0.75; do
  python train.py --dataset pcx1 \
    --pcx1-window-size 5000 \
    --pcx1-stride-ratio $stride_ratio \
    --split-by-session \
    --no-test-set \
    --epochs 60 --seed 42 \
    --output results/phase6/stride_${stride_ratio}/
done

# Window size ablation
for window in 1000 2500 5000 10000; do
  python train.py --dataset pcx1 \
    --pcx1-window-size $window \
    --pcx1-stride-ratio 0.5 \
    --split-by-session \
    --epochs 60 --seed 42 \
    --output results/phase6/window_${window}/
done

# Latency benchmark
python benchmark.py --model best_model.pt \
  --batch-sizes 1 4 16 64 \
  --measure-latency
```

### Sliding Window Parameters
| Parameter | Values to Test | Purpose |
|-----------|---------------|---------|
| Window Size | 1s, 2.5s, 5s, 10s | Temporal context |
| Stride Ratio | 25%, 50%, 75% | Overlap vs speed |
| Batch Size | 1, 4, 16, 64 | Throughput vs latency |

### Figure P6.1: Continuous Processing Performance
```
┌──────────────────────────────────────────────────────────────┐
│  CONTINUOUS SLIDING WINDOW PERFORMANCE                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  (A) R² vs Window Size                                       │
│      Line plot: 1s → 2.5s → 5s → 10s                         │
│      Longer windows = more context = better?                 │
│                                                              │
│  (B) R² vs Stride (overlap)                                  │
│      More overlap = more compute but maybe better?           │
│                                                              │
│  (C) Fixed window vs Sliding window comparison               │
│      Same model, different evaluation                        │
│      Does continuous hurt performance?                       │
│                                                              │
│  (D) Long recording example                                  │
│      60 seconds of continuous prediction                     │
│      Ground truth vs Predicted overlay                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Figure P6.2: Real-Time Feasibility
```
┌──────────────────────────────────────────────────────────────┐
│  REAL-TIME FEASIBILITY ANALYSIS                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  (A) Latency benchmark                                       │
│      Inference time vs batch size                            │
│      CPU vs GPU comparison                                   │
│      Target: <100ms for real-time                            │
│                                                              │
│  (B) Throughput analysis                                     │
│      Samples/second at different batch sizes                 │
│      Can we keep up with 1kHz sampling?                      │
│                                                              │
│  (C) Memory footprint                                        │
│      GPU memory vs batch size                                │
│      Can run on edge device?                                 │
│                                                              │
│  (D) Quality vs Speed trade-off                              │
│      Scatter: R² vs Latency                                  │
│      Pareto frontier for deployment                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Deliverables
- [ ] Optimal window size: ______ ms
- [ ] Optimal stride: ______%
- [ ] Inference latency: ______ ms (batch=1)
- [ ] Throughput: ______ samples/sec
- [ ] Real-time feasible: YES / NO

### Real-Time Requirements
| Requirement | Target | Achieved |
|-------------|--------|----------|
| Latency (batch=1) | < 100 ms | |
| Throughput | > 1000 Hz | |
| GPU Memory | < 4 GB | |
| R² (continuous) | > 0.5 | |

---

# MAIN PAPER FIGURES (6 figures)

## Figure 1: Method Overview
- (A) Problem schematic: Brain regions, translation concept
- (B) CondUNet architecture diagram
- (C) Training pipeline
- (D) Dataset summary table

## Figure 2: Baseline Comparison
- From Phase 1: Classical baselines
- From Phase 2: Neural architectures
- (A) Classical floor (7 methods)
- (B) Neural screening (8 architectures)
- (C) CondUNet vs all others
- (D) Statistical significance matrix

## Figure 3: Ablation Studies
- From Phase 3
- (A) Attention ablation
- (B) Loss function ablation
- (C) Component importance waterfall
- (D) Optimal configuration

## Figure 4: Generalization - Sessions
- From Phase 4
- (A) Intra vs Inter session
- (B) Per-session breakdown
- (C) Generalization gap analysis
- (D) Session transfer matrix

## Figure 5: Generalization - Datasets
- From Phase 5
- (A) Cross-dataset comparison (3 datasets)
- (B) Human iEEG results (DANDI)
- (C) Per-frequency transfer
- (D) Species comparison

## Figure 6: Real-Time Feasibility
- From Phase 6
- (A) Continuous processing performance
- (B) Window/stride optimization
- (C) Latency benchmarks
- (D) Deployment feasibility

---

# EXECUTION SUMMARY

```bash
# ═══════════════════════════════════════════════════════════
# PHASE 1: Classical Baselines (Olfactory)
# Runs: 7 methods × 5-fold × 3 seeds = ~105 evaluations
# Time: ~1 day
# ═══════════════════════════════════════════════════════════
python experiments/tier0_classical/run_tier0.py --dataset olfactory

# ═══════════════════════════════════════════════════════════
# PHASE 2: Architecture Screening (Olfactory)
# Runs: 8 architectures × 3 seeds = 24 training runs
# Time: ~2-3 days (60 epochs each)
# ═══════════════════════════════════════════════════════════
python experiments/tier1_screening/run_tier1.py --dataset olfactory

# ═══════════════════════════════════════════════════════════
# PHASE 3: Ablations (Olfactory)
# Runs: ~60 training runs
# Time: ~4-5 days
# ═══════════════════════════════════════════════════════════
python experiments/tier3_ablation/run_tier3.py --dataset olfactory

# ═══════════════════════════════════════════════════════════
# PHASE 4: Inter vs Intra Session (Olfactory)
# Runs: ~12 training runs
# Time: ~1-2 days
# ═══════════════════════════════════════════════════════════
python train.py --dataset olfactory --split-by-session ...
python train.py --dataset olfactory --no-split-by-session ...

# ═══════════════════════════════════════════════════════════
# PHASE 5: Cross-Dataset (PFC, DANDI)
# Runs: ~10 training runs
# Time: ~2 days
# ═══════════════════════════════════════════════════════════
python train.py --dataset pfc ...
python train.py --dataset dandi ...

# ═══════════════════════════════════════════════════════════
# PHASE 6: Continuous (PCx1)
# Runs: ~12 training runs + benchmarks
# Time: ~2 days
# ═══════════════════════════════════════════════════════════
python train.py --dataset pcx1 --pcx1-stride-ratio 0.5 ...
python benchmark.py ...
```

---

# TIMELINE

| Phase | Focus | Dataset | Duration | Runs |
|-------|-------|---------|----------|------|
| 1 | Classical Floor | Olfactory | 1 day | ~100 |
| 2 | Architecture Screen | Olfactory | 2-3 days | 24 |
| 3 | Ablations | Olfactory | 4-5 days | ~60 |
| 4 | Inter/Intra Session | Olfactory | 1-2 days | ~12 |
| 5 | Cross-Dataset | PFC, DANDI | 2 days | ~10 |
| 6 | Continuous | PCx1 | 2 days | ~12 |
| **Total** | | | **~2 weeks** | **~220 runs** |

---

# STATISTICAL CHECKLIST

For ALL comparisons:
- [ ] Minimum 3 random seeds
- [ ] 95% bootstrap confidence intervals (10,000 resamples)
- [ ] Paired Wilcoxon signed-rank tests
- [ ] Bonferroni correction for multiple comparisons
- [ ] Cohen's d effect sizes
- [ ] All p-values reported (exact, not just < 0.05)
