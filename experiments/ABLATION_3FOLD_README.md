# 3-Fold Cross-Validation Ablation Study

## Overview

This ablation study uses a proper 3-fold cross-validation approach where:
- 9 sessions are divided into 3 groups of 3 sessions each
- Each fold holds out a different group for testing
- All sessions get evaluated as test data across the 3 folds

## Key Design Decisions

### Baseline: `depth_deep` (n_downsample=4)
Previous results showed depth_deep outperforms other depths:
- baseline (n_downsample=2): 0.6679
- depth_medium (n_downsample=3): 0.6808 (+0.0129)
- depth_deep (n_downsample=4): 0.6821 (+0.0143)

### 3-Fold Session Splits
For 9 sessions [s0, s1, ..., s8]:
- Fold 0: test=[s0,s1,s2], train=[s3-s8]
- Fold 1: test=[s3,s4,s5], train=[s0-s2,s6-s8]
- Fold 2: test=[s6,s7,s8], train=[s0-s5]

## Ablation Configurations

| Name | Description | Key Change |
|------|-------------|------------|
| `baseline` | depth_deep with all best components | n_downsample=4 |
| `conv_type_standard` | Standard convolutions | conv_type=standard |
| `conditioning_none` | No auto-conditioning | conditioning=none, cond_mode=none |
| `adaptive_scaling_off` | No adaptive scaling | use_adaptive_scaling=False |
| `depth_medium` | Medium depth | n_downsample=3 |
| `depth_shallow` | Original shallow | n_downsample=2 |
| `attention_none` | No attention | attention_type=none |
| `attention_basic` | Basic attention | attention_type=basic |
| `skip_type_concat` | Concat skips | skip_type=concat |
| `bidirectional_on` | Bidirectional training | use_bidirectional=True |
| `dropout_01` | 10% dropout | dropout=0.1 |
| `dropout_02` | 20% dropout | dropout=0.2 |

Note: `cond_mode` only supports `"none"` and `"cross_attn_gated"` in the model.
The `conditioning_none` ablation tests disabling conditioning via `cond_mode=none`.

## Usage

### Run complete ablation study
```bash
python experiments/run_ablation_3fold.py --output-dir results/ablation_3fold
```

### Run specific ablations
```bash
python experiments/run_ablation_3fold.py --ablations baseline conv_type_standard conditioning_none
```

### Run specific folds
```bash
python experiments/run_ablation_3fold.py --folds 0 1  # Only run first two folds
```

### List all ablation configurations
```bash
python experiments/run_ablation_3fold.py --list-ablations
```

### Multi-GPU with FSDP
```bash
python experiments/run_ablation_3fold.py --fsdp
```

### Dry run (verify commands without executing)
```bash
python experiments/run_ablation_3fold.py --dry-run
```

## Output Structure

```
results/ablation_3fold/
├── ablation_summary.json       # Main summary with all results
├── statistical_analysis.json   # Statistical comparisons vs baseline
├── per_session_results.json    # Per-session R² for all ablations
├── training_curves.json        # Full training history curves
├── results_table.csv           # Publication-ready table
├── ablation_run_*.log          # Run logs
├── baseline/
│   ├── baseline_fold0_results.json  # Full metrics per fold
│   ├── baseline_fold1_results.json
│   ├── baseline_fold2_results.json
│   └── ablation_result.json    # Aggregated results
├── conv_type_standard/
│   └── ...
└── ...
```

## Saved Metrics (Per Fold)

From each `train.py` run, we capture:
- **Primary metrics**: val_r2, val_loss, val_corr, val_mae
- **Training metadata**: epochs_trained, n_parameters, best_epoch, total_time
- **Training curves**: train_losses, val_losses, val_r2s, val_corrs (full history)
- **Per-session metrics**: per_session_r2, per_session_corr, per_session_loss
- **Test metrics** (if available): test_avg_r2, per_session_test_results

## Statistical Analysis

For each ablation vs baseline, we compute:
- **Effect sizes**: Cohen's d, Hedges' g (bias-corrected)
- **Statistical tests**: Paired t-test, Wilcoxon signed-rank
- **Confidence intervals**: 95% CI for mean difference
- **Multiple comparison correction**: Holm-Bonferroni
- **Normality tests**: Shapiro-Wilk (recommends parametric vs non-parametric)

## Results Interpretation

The summary shows R² performance for each ablation:
- **mean_r2 ± std_r2**: Average performance across 3 folds
- **delta**: Difference from baseline (positive = better)

Example output:
```
Ablation                  R2 (mean ± std)      Delta
------------------------------------------------------------
baseline                  0.6821 +/- 0.0038       --
bidirectional_on          0.6850 +/- 0.0042   +0.0029
dropout_01                0.6810 +/- 0.0035   -0.0011
conv_type_standard        0.5924 +/- 0.0062   -0.0897
```
