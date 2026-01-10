# Nature Methods Publication Plan: Neural Signal Translation

## Datasets & Region Mappings

| Dataset | Species | Source → Target | Channels | Format |
|---------|---------|-----------------|----------|--------|
| **Olfactory** | Rodent | OB → PCx | 32 → 32 | Trial-based, 5s |
| **PFC** | Rodent | PFC → CA1 | 64 → 32 | Trial-based, 5s |
| **DANDI** | Human | AMY → HPC | Variable | Movie, 8min |

---

## Study Structure (5 Phases)

```
┌────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: CLASSICAL BASELINES                                          │
│  ─────────────────────────────                                         │
│  All 3 datasets (Olfactory, PFC, DANDI)                                │
│  7 classical methods, establish performance floor                      │
├────────────────────────────────────────────────────────────────────────┤
│  PHASE 2: ARCHITECTURE SCREENING                                       │
│  ───────────────────────────────                                       │
│  All 3 datasets                                                        │
│  8 neural architectures, identify top performers                       │
├────────────────────────────────────────────────────────────────────────┤
│  PHASE 3: CONDUNET ABLATIONS                                           │
│  ───────────────────────────                                           │
│  Olfactory only (primary dataset)                                      │
│  6 ablation studies, justify design decisions                          │
├────────────────────────────────────────────────────────────────────────┤
│  PHASE 4: INTER vs INTRA SESSION                                       │
│  ──────────────────────────────                                        │
│  All 3 datasets                                                        │
│  Cross-session vs within-session generalization                        │
├────────────────────────────────────────────────────────────────────────┤
│  PHASE 5: REAL-TIME CONTINUOUS                                         │
│  ─────────────────────────────                                         │
│  All 3 datasets (sliding window mode)                                  │
│  Latency, throughput, real-time feasibility                            │
└────────────────────────────────────────────────────────────────────────┘
```

---

# PHASE 1: CLASSICAL BASELINES
## All Datasets | Establish Performance Floor

### 1A: Olfactory (OB → PCx)

```bash
python run_tier0.py --dataset olfactory --output results/p1/olfactory/
```

### 1B: PFC (PFC → CA1)

```bash
python run_tier0.py --dataset pfc --output results/p1/pfc/
```

### 1C: DANDI (AMY → HPC)

```bash
python run_tier0.py --dataset dandi \
  --source-region amygdala --target-region hippocampus \
  --output results/p1/dandi/
```

### Methods (7 total)
| Method | Description |
|--------|-------------|
| Wiener | Single-channel optimal filter |
| Wiener MIMO | Multi-input multi-output |
| Ridge | L2-regularized regression |
| Ridge Temporal | With temporal features |
| Ridge CV | Cross-validated |
| VAR | Vector autoregressive |
| VAR Exogenous | VAR with input signals |

### Figure 1.1: Classical Baselines Across Datasets
```
┌─────────────────────────────────────────────────────────┐
│  (A) R² by method - Olfactory                           │
│  (B) R² by method - PFC                                 │
│  (C) R² by method - DANDI                               │
│  (D) Cross-dataset comparison: best classical per dataset│
└─────────────────────────────────────────────────────────┘
```

### Deliverables
- [ ] Best classical R² (Olfactory): ______
- [ ] Best classical R² (PFC): ______
- [ ] Best classical R² (DANDI): ______

---

# PHASE 2: ARCHITECTURE SCREENING
## All Datasets | Neural Methods

### 2A: Olfactory

```bash
for arch in linear simplecnn wavenet fnet vit performer mamba condunet; do
  for seed in 42 43 44; do
    python train.py --dataset olfactory --arch $arch --epochs 60 --seed $seed
  done
done
```

### 2B: PFC

```bash
for arch in linear simplecnn wavenet fnet vit performer mamba condunet; do
  for seed in 42 43 44; do
    python train.py --dataset pfc --arch $arch --epochs 60 --seed $seed
  done
done
```

### 2C: DANDI

```bash
for arch in linear simplecnn wavenet fnet vit performer mamba condunet; do
  for seed in 42 43 44; do
    python train.py --dataset dandi \
      --dandi-source-region amygdala --dandi-target-region hippocampus \
      --arch $arch --epochs 60 --seed $seed
  done
done
```

### Architectures (8 total)
| Architecture | Type | Key Feature |
|--------------|------|-------------|
| Linear | Baseline | Simple linear mapping |
| SimpleCNN | CNN | Basic conv layers |
| WaveNet1D | Dilated CNN | Causal dilated convolutions |
| FNet1D | Fourier | FFT-based mixing |
| ViT1D | Transformer | Full self-attention |
| Performer1D | Efficient Trans | Linear attention |
| Mamba1D | State Space | Selective SSM |
| **CondUNet** | **U-Net** | **Skip + conditioning** |

### Figure 2.1: Architecture Comparison Across Datasets
```
┌─────────────────────────────────────────────────────────┐
│  (A) R² by architecture - Olfactory                     │
│  (B) R² by architecture - PFC                           │
│  (C) R² by architecture - DANDI                         │
│  (D) CondUNet rank across all 3 datasets                │
└─────────────────────────────────────────────────────────┘
```

### Figure 2.2: Neural vs Classical
```
┌─────────────────────────────────────────────────────────┐
│  (A) ΔR² improvement over classical (per dataset)       │
│  (B) Which architectures beat classical baseline?       │
│  (C) Params vs Performance trade-off                    │
│  (D) Consistent winners across datasets                 │
└─────────────────────────────────────────────────────────┘
```

### Deliverables
- [ ] Top architecture per dataset
- [ ] CondUNet ranking across datasets
- [ ] Runs: 8 × 3 seeds × 3 datasets = 72 runs

---

# PHASE 3: CONDUNET ABLATIONS
## Olfactory Only | Justify Design Decisions

### Ablation Studies

```bash
# 3A: Attention mechanism
for attn in none basic cross_freq cross_freq_v2; do
  python train.py --dataset olfactory --attention-type $attn --epochs 60 --seed 42
done

# 3B: Conditioning
for cond in none odor spectro_temporal; do
  python train.py --dataset olfactory --conditioning $cond --epochs 60 --seed 42
done

# 3C: Loss function
for loss in l1 huber wavelet huber_wavelet; do
  python train.py --dataset olfactory --loss $loss --epochs 60 --seed 42
done

# 3D: Model capacity
for channels in 32 64 128; do
  python train.py --dataset olfactory --base-channels $channels --epochs 60 --seed 42
done

# 3E: Bidirectional training
python train.py --dataset olfactory --epochs 60 --seed 42  # bidirectional
python train.py --dataset olfactory --no-bidirectional --epochs 60 --seed 42

# 3F: Augmentation
python train.py --dataset olfactory --epochs 60 --seed 42  # with aug
python train.py --dataset olfactory --no-aug --epochs 60 --seed 42
```

### Summary Table
| Ablation | Variants | Runs |
|----------|----------|------|
| Attention | none, basic, cross_freq, cross_freq_v2 | 4 |
| Conditioning | none, odor, spectro_temporal | 3 |
| Loss | L1, Huber, Wavelet, Combined | 4 |
| Capacity | 32, 64, 128 channels | 3 |
| Bidirectional | uni, bi | 2 |
| Augmentation | none, full | 2 |
| **Total** | | **18** |

### Figure 3.1: Ablation Results
```
┌─────────────────────────────────────────────────────────┐
│  (A) Attention ablation: R² by type                     │
│  (B) Loss ablation: R² by loss function                 │
│  (C) Waterfall: contribution of each component          │
│  (D) Final optimal configuration                        │
└─────────────────────────────────────────────────────────┘
```

---

# PHASE 4: INTER vs INTRA SESSION
## All Datasets | Generalization Study

### Definition
```
INTRA-SESSION: Train/test split WITHIN each session (random)
INTER-SESSION: Train on some sessions, test on HELD-OUT sessions
```

### 4A: Olfactory

```bash
# Intra-session (random split within sessions)
python train.py --dataset olfactory --no-split-by-session --epochs 60 --seed 42

# Inter-session (hold out entire sessions)
python train.py --dataset olfactory --split-by-session --epochs 60 --seed 42
```

### 4B: PFC

```bash
# Intra-session
python train.py --dataset pfc --no-split-by-session --epochs 60 --seed 42

# Inter-session
python train.py --dataset pfc --split-by-session --epochs 60 --seed 42
```

### 4C: DANDI

```bash
# Intra-session (random within subjects)
python train.py --dataset dandi --dandi-source-region amygdala --dandi-target-region hippocampus \
  --no-split-by-session --epochs 60 --seed 42

# Inter-session (leave-one-subject-out)
python train.py --dataset dandi --dandi-source-region amygdala --dandi-target-region hippocampus \
  --split-by-session --epochs 60 --seed 42
```

### Figure 4.1: Intra vs Inter Session
```
┌─────────────────────────────────────────────────────────┐
│  (A) Olfactory: Intra vs Inter session R²               │
│  (B) PFC: Intra vs Inter session R²                     │
│  (C) DANDI: Intra vs Inter session R²                   │
│  (D) Generalization gap across datasets                 │
└─────────────────────────────────────────────────────────┘
```

### Figure 4.2: Session/Subject Analysis
```
┌─────────────────────────────────────────────────────────┐
│  (A) Per-session performance breakdown                  │
│  (B) Easy vs hard sessions identified                   │
│  (C) DANDI per-subject (18 subjects)                    │
│  (D) What predicts session difficulty?                  │
└─────────────────────────────────────────────────────────┘
```

### Deliverables
| Dataset | Intra R² | Inter R² | Gap |
|---------|----------|----------|-----|
| Olfactory | | | |
| PFC | | | |
| DANDI | | | |

---

# PHASE 5: REAL-TIME CONTINUOUS
## All Datasets | Sliding Window Mode

### Definition
```
FIXED WINDOW: Discrete trials, no overlap
SLIDING WINDOW: Continuous, overlapping windows for real-time simulation
```

### 5A: Olfactory (PCx1 continuous)

```bash
# Window size ablation
for window in 1000 2500 5000; do
  python train.py --dataset pcx1 --pcx1-window-size $window \
    --pcx1-stride-ratio 0.5 --epochs 60 --seed 42
done

# Stride ablation
for stride in 0.25 0.5 0.75; do
  python train.py --dataset pcx1 --pcx1-window-size 5000 \
    --pcx1-stride-ratio $stride --epochs 60 --seed 42
done
```

### 5B: PFC (sliding window from trials)

```bash
for window in 1000 2500 5000; do
  python train.py --dataset pfc --sliding-window \
    --window-size $window --stride-ratio 0.5 --epochs 60 --seed 42
done
```

### 5C: DANDI (sliding window from movie)

```bash
for window in 1000 2500 5000; do
  python train.py --dataset dandi --dandi-source-region amygdala --dandi-target-region hippocampus \
    --sliding-window --window-size $window --stride-ratio 0.5 --epochs 60 --seed 42
done
```

### Latency Benchmarks (All Datasets)

```bash
python benchmark.py --model best_olfactory.pt --batch-sizes 1 4 16 64
python benchmark.py --model best_pfc.pt --batch-sizes 1 4 16 64
python benchmark.py --model best_dandi.pt --batch-sizes 1 4 16 64
```

### Figure 5.1: Continuous Performance
```
┌─────────────────────────────────────────────────────────┐
│  (A) R² vs window size (all 3 datasets)                 │
│  (B) R² vs stride ratio                                 │
│  (C) Fixed vs Sliding comparison                        │
│  (D) Long recording example (60s)                       │
└─────────────────────────────────────────────────────────┘
```

### Figure 5.2: Real-Time Feasibility
```
┌─────────────────────────────────────────────────────────┐
│  (A) Inference latency vs batch size                    │
│  (B) Throughput (samples/sec)                           │
│  (C) Memory footprint                                   │
│  (D) R² vs Latency trade-off (Pareto)                   │
└─────────────────────────────────────────────────────────┘
```

### Real-Time Requirements
| Metric | Target | Olfactory | PFC | DANDI |
|--------|--------|-----------|-----|-------|
| Latency (batch=1) | < 100 ms | | | |
| Throughput | > 1000 Hz | | | |
| GPU Memory | < 4 GB | | | |

---

# MAIN PAPER FIGURES

| Figure | Content | Source Phase |
|--------|---------|--------------|
| **Fig 1** | Method overview, architecture | - |
| **Fig 2** | Classical baselines (3 datasets) | Phase 1 |
| **Fig 3** | Architecture screening (3 datasets) | Phase 2 |
| **Fig 4** | CondUNet ablations | Phase 3 |
| **Fig 5** | Inter vs Intra session (3 datasets) | Phase 4 |
| **Fig 6** | Real-time continuous (3 datasets) | Phase 5 |

---

# EXECUTION CHECKLIST

| Phase | Focus | Datasets | Est. Runs |
|-------|-------|----------|-----------|
| 1 | Classical Baselines | All 3 | ~21 (7×3) |
| 2 | Architecture Screening | All 3 | 72 (8×3×3) |
| 3 | CondUNet Ablations | Olfactory | 18 |
| 4 | Inter vs Intra Session | All 3 | 18 (2×3×3 seeds) |
| 5 | Real-time Continuous | All 3 | ~30 |
| **Total** | | | **~160 runs** |

---

# STATISTICAL REQUIREMENTS

For ALL comparisons:
- [ ] 3 random seeds minimum
- [ ] 95% CI (bootstrap, 10k resamples)
- [ ] Paired Wilcoxon signed-rank tests
- [ ] Bonferroni correction
- [ ] Cohen's d effect sizes
- [ ] Exact p-values reported
