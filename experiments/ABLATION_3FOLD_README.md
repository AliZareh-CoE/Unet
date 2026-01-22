# 3-Fold Cross-Validation Ablation Study

## Overview

This ablation study uses a proper 3-fold cross-validation approach where:
- 9 sessions are divided into 3 groups of 3 sessions each
- Each fold holds out a different group for testing
- All sessions get evaluated as test data across the 3 folds

## Key Design Decisions

### Baseline: Original default (n_downsample=2)
Using the original default configuration as the reference point:
- baseline (n_downsample=2): Reference
- depth_medium (n_downsample=3): Tests deeper network
- depth_deep (n_downsample=4): Tests deepest network

### 3-Fold Session Splits
For 9 sessions [s0, s1, ..., s8]:
- Fold 0: test=[s0,s1,s2], train=[s3-s8]
- Fold 1: test=[s3,s4,s5], train=[s0-s2,s6-s8]
- Fold 2: test=[s6,s7,s8], train=[s0-s5]

## Ablation Configurations

| Name | Description | Key Change |
|------|-------------|------------|
| `baseline` | Original default with all components | n_downsample=2 |
| `conv_type_standard` | Standard convolutions | conv_type=standard |
| `conditioning_none` | No auto-conditioning | conditioning=none, cond_mode=none |
| `adaptive_scaling_off` | No adaptive scaling | use_adaptive_scaling=False |
| `depth_medium` | Medium depth | n_downsample=3 |
| `depth_deep` | Deep network | n_downsample=4 |
| `attention_none` | No attention | attention_type=none |
| `attention_basic` | Basic attention | attention_type=basic |
| `skip_type_concat` | Concat skips | skip_type=concat |
| `bidirectional_on` | Bidirectional training | use_bidirectional=True |
| `dropout_01` | 10% dropout | dropout=0.1 |
| `dropout_02` | 20% dropout | dropout=0.2 |
| `width_narrow` | Narrow network | base_channels=64 |
| `width_wide` | Wide network | base_channels=256 |

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

### Disable sweep mode (don't apply permanent decisions)
```bash
python experiments/run_ablation_3fold.py --no-sweep
```

## Output Structure

```
results/ablation_3fold/
├── ablation_summary.json       # Main summary with all results
├── statistical_analysis.json   # Statistical comparisons vs baseline
├── per_session_results.json    # Per-session R² for all ablations
├── training_curves.json        # Full training history curves
├── results_table.csv           # Publication-ready table
├── recommendations.json        # Automated recommendations
├── sweep_state.json            # Persistent sweep decisions
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
baseline                  0.6679 +/- 0.0038       --
depth_deep                0.6821 +/- 0.0042   +0.0142
dropout_01                0.6670 +/- 0.0035   -0.0009
conv_type_standard        0.5924 +/- 0.0062   -0.0755
```

## Automated Recommendations

The system automatically analyzes results and provides recommendations based on:

### Decision Logic

| Condition | Action | Rationale |
|-----------|--------|-----------|
| Significantly better (p < 0.05, delta > 0) | **UPGRADE** | Switch to this config |
| Significantly worse (p < 0.05, delta < 0) | **ELIMINATE** | Remove from consideration |
| Equivalent performance + cheaper | **PREFER** | Same quality, lower cost |
| Equivalent performance + more expensive | **KEEP BASELINE** | No benefit to switch |

### Output Files

The recommendations are saved to:
- `recommendations.json` - Full decision analysis with reasoning
- `ablation_summary.json` - Includes recommendations in the summary

### Example Recommendation Output
```
================================================================================
RECOMMENDATIONS
================================================================================

[UPGRADE] Significantly BETTER than baseline:
  * depth_deep: +0.0142 R² (p=0.0231)

[ELIMINATE] Significantly WORSE than baseline:
  * conv_type_standard: -0.0755 R² (p=0.0012)

[PREFER] Equivalent performance but CHEAPER:
  * width_narrow: -0.0008 R², 50% fewer params

------------------------------------------------------------
FINAL RECOMMENDATION: depth_deep
  Reason: UPGRADE: depth_deep is significantly better (+0.0142 R², p=0.0231)
------------------------------------------------------------
```

## Persistent Sweep State

The system maintains a **persistent sweep state** that tracks decisions across runs. This enables automatic progressive elimination of poor configurations and upgrades to better ones.

### How It Works

1. **First Run**: All ablations are run, recommendations are computed
2. **Sweep Decisions Applied**:
   - **ELIMINATE**: Configs significantly worse than baseline are permanently marked for exclusion
   - **UPGRADE**: If a config is significantly better, it becomes the new baseline
3. **Subsequent Runs**: Eliminated configs are automatically skipped

### Sweep State File

The sweep state is saved to `sweep_state.json` in the output directory:

```json
{
  "current_baseline": "depth_deep",
  "eliminated": ["conv_type_standard", "attention_none"],
  "upgrade_history": [
    {
      "timestamp": "2024-01-15T10:30:00",
      "from": "baseline",
      "to": "depth_deep",
      "delta": 0.0142,
      "p_value": 0.0231
    }
  ],
  "decision_log": [
    {
      "timestamp": "2024-01-15T10:30:00",
      "action": "ELIMINATE",
      "config": "conv_type_standard",
      "delta": -0.0755,
      "p_value": 0.0012,
      "reason": "Significantly worse (-0.0755 R², p=0.0012)"
    }
  ]
}
```

### Disabling Sweep Mode

To run without applying permanent decisions (useful for re-running eliminated configs):

```bash
python experiments/run_ablation_3fold.py --no-sweep
```

### Resetting Sweep State

To start fresh, delete the sweep state file:

```bash
rm results/ablation_3fold/sweep_state.json
```

### Workflow Example

```bash
# Run 1: All 14 ablations run
python experiments/run_ablation_3fold.py
# -> Eliminates conv_type_standard (significantly worse)
# -> Upgrades to depth_deep (significantly better)

# Run 2: Only 13 ablations run (conv_type_standard skipped)
python experiments/run_ablation_3fold.py
# -> Further refinement based on new baseline

# Run with sweep disabled (test eliminated configs again)
python experiments/run_ablation_3fold.py --no-sweep
```
