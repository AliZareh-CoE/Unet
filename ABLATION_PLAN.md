# Nature Methods Publication Plan: Neural Signal Translation (OB → PCx)

## Executive Summary
Comprehensive ablation study to demonstrate that each component of our neural signal translation model contributes meaningfully to performance.

---

## PHASE 0: VALIDATE CURRENT FIX (Today)
**Goal:** Confirm spectral bias fix resolves IF distribution mismatch

- [ ] Run current best config with fixed spectral bias (250 bands @ 2Hz)
- [ ] Verify IF distribution matches between predicted and target
- [ ] Document baseline performance: target r > 0.78, R² > 0.60

---

## PHASE 1: INFRASTRUCTURE (1-2 days)

### 1.1 Ablation CLI Flags
Add flags to `train.py` for easy component toggling:
```bash
# Augmentation control
--no-augmentation          # Disable ALL augmentation
--aug-only TYPE            # Enable ONLY specific augmentation

# Architecture control
--no-attention             # Disable attention mechanism
--no-conditioning          # Disable odor conditioning (cond_mode=none)
--no-se-blocks             # Disable squeeze-excitation
--no-spectral-shift        # Disable learnable spectral shift

# Training control
--no-two-stage             # End-to-end training (no post-hoc calibration)
--no-spectral-bias         # Disable OptimalSpectralBias in Stage 2
--no-histogram-matching    # Disable EnvelopeHistogramMatching in Stage 2
```

### 1.2 Baseline Models
Create `baselines.py` with:

| Model | Description | Parameters |
|-------|-------------|------------|
| **Ridge Regression** | Linear baseline, per-channel | ~1K |
| **MLP** | 3-layer, shared across channels | ~100K |
| **Vanilla CNN** | Conv1D stack, no conditioning | ~500K |
| **UNet-Simple** | UNet without attention/SE/conditioning | ~2M |
| **UNet-Full** | Your full model | ~8M |

### 1.3 Experiment Runner
Create `run_ablations.py`:
- Automated experiment queue
- 3 seeds per config (for error bars)
- Results logging to CSV/JSON
- GPU scheduling for parallel runs

---

## PHASE 2: EXPERIMENTS (3-5 days with 8x A100)

### Experiment 1: Model Architecture Comparison
**Question:** Does our architecture outperform simpler alternatives?

| Run | Model | Expected r |
|-----|-------|------------|
| 1.1 | Ridge Regression | ~0.30-0.40 |
| 1.2 | MLP | ~0.45-0.55 |
| 1.3 | Vanilla CNN | ~0.55-0.65 |
| 1.4 | UNet-Simple | ~0.65-0.70 |
| 1.5 | UNet-Full (ours) | ~0.78+ |

**Runs:** 5 models × 3 seeds = **15 runs**

---

### Experiment 2: Augmentation Ablation
**Question:** Which augmentations help? Is the combination synergistic?

| Run | Config | Tests |
|-----|--------|-------|
| 2.0 | No augmentation | Baseline |
| 2.1 | Time shift only | Temporal invariance |
| 2.2 | Noise only | Noise robustness |
| 2.3 | Channel dropout only | Channel redundancy |
| 2.4 | Amplitude scale only | Amplitude invariance |
| 2.5 | Time mask only | Temporal robustness |
| 2.6 | Mixup only | Interpolation smoothness |
| 2.7 | Freq mask only | Frequency robustness |
| 2.8 | All augmentations | Full regularization |

**Runs:** 9 configs × 3 seeds = **27 runs**

---

### Experiment 3: Conditioning Mechanism Ablation
**Question:** How important is odor-specific conditioning?

| Run | cond_mode | Description |
|-----|-----------|-------------|
| 3.1 | none | No conditioning |
| 3.2 | concat | Simple concatenation |
| 3.3 | film | FiLM modulation |
| 3.4 | cross_attn | Cross-attention |
| 3.5 | cross_attn_gated | Gated cross-attention (ours) |

**Runs:** 5 modes × 3 seeds = **15 runs**

---

### Experiment 4: Attention & Architecture Ablation
**Question:** Do attention and SE blocks contribute?

| Run | Attention | SE Blocks | Description |
|-----|-----------|-----------|-------------|
| 4.1 | ✗ | ✗ | Baseline UNet |
| 4.2 | ✓ | ✗ | + Attention only |
| 4.3 | ✗ | ✓ | + SE only |
| 4.4 | ✓ | ✓ | Full model |

**Runs:** 4 configs × 3 seeds = **12 runs**

---

### Experiment 5: Spectral Correction Ablation
**Question:** Is post-hoc spectral correction necessary?

| Run | Spectral Shift | Optimal Bias | Histogram Match | Description |
|-----|---------------|--------------|-----------------|-------------|
| 5.1 | ✗ | ✗ | ✗ | Raw UNet output |
| 5.2 | ✓ | ✗ | ✗ | Learnable shift only |
| 5.3 | ✗ | ✓ | ✗ | Post-hoc bias only |
| 5.4 | ✗ | ✗ | ✓ | Histogram match only |
| 5.5 | ✓ | ✓ | ✓ | Full pipeline (ours) |

**Runs:** 5 configs × 3 seeds = **15 runs**

---

### Experiment 6: Training Strategy Ablation
**Question:** Does two-stage training help?

| Run | Strategy | Description |
|-----|----------|-------------|
| 6.1 | End-to-end | Train everything jointly |
| 6.2 | Two-stage | Stage 1 → Stage 2 (ours) |
| 6.3 | Stage 2 longer | More calibration epochs |

**Runs:** 3 configs × 3 seeds = **9 runs**

---

### Experiment 7: Regularization Ablation
**Question:** How much does regularization help?

| Run | Dropout | Weight Decay | Augmentation |
|-----|---------|--------------|--------------|
| 7.1 | 0.0 | 0.0 | ✗ | No regularization |
| 7.2 | 0.2 | 0.0 | ✗ | Dropout only |
| 7.3 | 0.0 | 0.01 | ✗ | Weight decay only |
| 7.4 | 0.0 | 0.0 | ✓ | Augmentation only |
| 7.5 | 0.2 | 0.01 | ✓ | Full regularization |

**Runs:** 5 configs × 3 seeds = **15 runs**

---

## TOTAL EXPERIMENT RUNS: ~108 runs
With early stopping ~50-80 epochs each, ~15-20 min/run on 8x A100
**Estimated time: 27-36 GPU-hours** (parallelizable)

---

## PHASE 3: ANALYSIS & VISUALIZATION (2-3 days)

### 3.1 Statistical Analysis
- Paired t-tests between conditions
- Effect sizes (Cohen's d)
- 95% confidence intervals
- Multiple comparison correction (Bonferroni)

### 3.2 Main Figures

**Figure 1: Model Comparison**
- Bar plot: r and R² across architectures
- Error bars from 3 seeds
- Statistical significance brackets

**Figure 2: Ablation Heatmap**
- Rows: Components (augmentation, attention, spectral, etc.)
- Columns: Metrics (r, R², train/val gap)
- Color intensity = performance

**Figure 3: Augmentation Analysis**
- Individual vs combined effect
- Synergy visualization

**Figure 4: Spectral Correction**
- IF distribution before/after correction
- Per-frequency-band correlation improvement
- Spectrogram comparison (pred vs target)

**Figure 5: Odor-Specific Performance**
- Per-odor correlation breakdown
- Conditioning effect visualization
- t-SNE of learned odor embeddings

### 3.3 Supplementary Figures
- Learning curves for all experiments
- Attention map visualizations
- Per-channel performance breakdown
- Failure case analysis

---

## PHASE 4: BIOLOGICAL VALIDATION (Optional but strengthens paper)

### 4.1 Neuroscience Sanity Checks
- [ ] Verify predicted signals have realistic neural dynamics
- [ ] Check oscillation frequencies match known PCx rhythms
- [ ] Validate odor response latencies are physiologically plausible

### 4.2 Decoding Analysis
- [ ] Can downstream decoder extract odor identity from predictions?
- [ ] Compare decoding accuracy: real vs predicted PCx

### 4.3 Perturbation Analysis
- [ ] What happens when OB input is silenced?
- [ ] Gradual input degradation → graceful output degradation?

---

## PHASE 5: PAPER WRITING

### Structure for Nature Methods
1. **Abstract** (150 words)
2. **Introduction** - Why OB→PCx translation matters
3. **Results**
   - Model outperforms baselines (Exp 1)
   - Each component contributes (Exp 2-7)
   - Spectral correction is critical (Exp 5)
   - Biological validation
4. **Discussion** - Limitations, future work
5. **Methods** - Full technical details
6. **Supplementary** - All ablation tables, extra figures

---

## TIMELINE ESTIMATE

| Phase | Duration | Parallelizable |
|-------|----------|----------------|
| Phase 0: Validate fix | 1 day | No |
| Phase 1: Infrastructure | 2 days | No |
| Phase 2: Experiments | 3-5 days | Yes (multi-GPU) |
| Phase 3: Analysis | 2-3 days | Partially |
| Phase 4: Bio validation | 3-5 days | Yes |
| Phase 5: Writing | 5-7 days | No |

**Total: ~3-4 weeks to submission-ready**

---

## FILES TO CREATE

```
Unet/
├── baselines.py              # Baseline models (Ridge, MLP, CNN)
├── run_ablations.py          # Experiment runner
├── analyze_ablations.py      # Results aggregation
├── plot_ablations.py         # Publication figures
├── configs/
│   └── ablations/
│       ├── exp1_architectures.yaml
│       ├── exp2_augmentation.yaml
│       ├── exp3_conditioning.yaml
│       ├── exp4_attention.yaml
│       ├── exp5_spectral.yaml
│       ├── exp6_training.yaml
│       └── exp7_regularization.yaml
└── results/
    └── ablations/
        ├── raw/              # Raw experiment outputs
        ├── aggregated/       # Processed results
        └── figures/          # Publication figures
```

---

## NEXT ACTION
Start with Phase 0: Run current config to validate spectral bias fix works.
