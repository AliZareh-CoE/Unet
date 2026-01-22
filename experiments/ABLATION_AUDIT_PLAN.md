# Ablation Study Audit Plan

## Overview

This document outlines a comprehensive audit plan to verify the correctness of the 3-fold cross-validation ablation study. Each section identifies what needs to be verified and how to verify it.

---

## 1. ARGUMENT PASSING AUDIT

### What to Verify
- All ablation configurations correctly translate to train.py CLI arguments
- Arguments are spelled correctly (matching train.py's argparse)
- Default values are appropriate
- Conditional arguments are handled correctly (e.g., cond_mode=none)

### Verification Method
```bash
python experiments/run_ablation_3fold.py --dry-run --ablations baseline conditioning_none conv_type_standard
```

### Checklist
| Argument | Expected Format | Verified |
|----------|----------------|----------|
| `--n-downsample` | Integer (2, 3, or 4) | [ ] |
| `--conv-type` | "standard" or "modern" | [ ] |
| `--attention-type` | "none", "basic", "cross_freq", "cross_freq_v2" | [ ] |
| `--cond-mode` | "none" or "cross_attn_gated" | [ ] |
| `--conditioning` | Only passed when cond_mode != "none" | [ ] |
| `--skip-type` | "add" or "concat" | [ ] |
| `--use-adaptive-scaling` | Flag (present or absent) | [ ] |
| `--no-bidirectional` | Flag (present or absent) | [ ] |
| `--dropout` | Float, only passed when > 0 | [ ] |
| `--optimizer` | "adamw" | [ ] |
| `--lr-schedule` | "cosine_warmup" | [ ] |
| `--val-sessions` | List of session names | [ ] |

### Critical Checks
1. **conditioning_none ablation**: Must pass `--cond-mode none` WITHOUT `--conditioning`
2. **bidirectional_on ablation**: Must NOT pass `--no-bidirectional`
3. **dropout ablations**: Must pass `--dropout 0.1` or `--dropout 0.2`

---

## 2. SESSION SPLIT AUDIT

### What to Verify
- 3-fold splits correctly partition all sessions
- Each session appears in exactly ONE test fold
- All sessions appear in exactly TWO training folds
- No session overlap between train and test within a fold

### Verification Method
```python
from experiments.run_ablation_3fold import get_3fold_session_splits

sessions = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
splits = get_3fold_session_splits(sessions)

# Verify each session is tested exactly once
test_counts = {}
for split in splits:
    for s in split['test_sessions']:
        test_counts[s] = test_counts.get(s, 0) + 1

assert all(c == 1 for c in test_counts.values()), "Each session should be tested exactly once"

# Verify no overlap between train and test
for split in splits:
    train_set = set(split['train_sessions'])
    test_set = set(split['test_sessions'])
    assert train_set.isdisjoint(test_set), f"Overlap in fold {split['fold_idx']}"
```

### Expected Split Structure
| Fold | Test Sessions | Train Sessions |
|------|---------------|----------------|
| 0 | [s0, s1, s2] | [s3, s4, s5, s6, s7, s8] |
| 1 | [s3, s4, s5] | [s0, s1, s2, s6, s7, s8] |
| 2 | [s6, s7, s8] | [s0, s1, s2, s3, s4, s5] |

---

## 3. METRIC CAPTURE AUDIT

### What to Verify
- All metrics from train.py output are captured in FoldResult
- No metrics are lost or misnamed
- Data types are correct

### train.py Output Fields (from `--output-results-file`)
```json
{
  "architecture": "condunet",
  "fold": 0,
  "best_val_r2": 0.6821,
  "best_val_mae": 0.0234,
  "best_val_corr": 0.8265,
  "best_val_loss": 0.0156,
  "best_epoch": 45,
  "train_losses": [0.05, 0.04, ...],
  "val_losses": [0.06, 0.05, ...],
  "val_r2s": [0.5, 0.55, ...],
  "val_corrs": [0.7, 0.72, ...],
  "per_session_r2": {"session1": 0.68, "session2": 0.71},
  "per_session_corr": {"session1": 0.82, "session2": 0.84},
  "per_session_loss": {"session1": 0.015, "session2": 0.014},
  "per_session_test_results": [...],
  "test_avg_r2": null,
  "test_avg_corr": null,
  "test_avg_delta": null,
  "total_time": 3600.0,
  "epochs_trained": 80,
  "n_parameters": 12500000,
  "completed_successfully": true
}
```

### FoldResult Mapping Checklist
| train.py Field | FoldResult Field | Captured | Verified |
|----------------|------------------|----------|----------|
| `best_val_r2` | `val_r2` | Yes | [ ] |
| `best_val_loss` | `val_loss` | Yes | [ ] |
| `best_val_corr` | `val_corr` | Yes | [ ] |
| `best_val_mae` | `val_mae` | Yes | [ ] |
| `best_epoch` | `best_epoch` | Yes | [ ] |
| `train_losses` | `train_losses` | Yes | [ ] |
| `val_losses` | `val_losses` | Yes | [ ] |
| `val_r2s` | `val_r2s` | Yes | [ ] |
| `val_corrs` | `val_corrs` | Yes | [ ] |
| `per_session_r2` | `per_session_r2` | Yes | [ ] |
| `per_session_corr` | `per_session_corr` | Yes | [ ] |
| `per_session_loss` | `per_session_loss` | Yes | [ ] |
| `per_session_test_results` | `per_session_test_results` | Yes | [ ] |
| `test_avg_r2` | `test_avg_r2` | Yes | [ ] |
| `test_avg_corr` | `test_avg_corr` | Yes | [ ] |
| `epochs_trained` | `epochs_trained` | Yes | [ ] |
| `n_parameters` | `n_parameters` | Yes | [ ] |
| `total_time` | `total_time` | Via elapsed | [ ] |

---

## 4. STATISTICAL ANALYSIS AUDIT

### What to Verify
- Effect sizes computed correctly (Cohen's d, Hedges' g)
- Statistical tests use correct formulas
- Confidence intervals are valid
- Multiple comparison correction is applied correctly

### 4.1 Cohen's d (Paired Samples)
**Formula**: `d = mean(diff) / std(diff)`

**Verification**:
```python
import numpy as np

baseline = np.array([0.68, 0.69, 0.67])
ablation = np.array([0.65, 0.66, 0.64])
diff = ablation - baseline

expected_d = np.mean(diff) / np.std(diff, ddof=1)
# Should be negative (ablation worse than baseline)
```

### 4.2 Hedges' g (Bias Correction)
**Formula**: `g = d * (1 - 3/(4*n - 1))`

**Verification**:
```python
n = len(diff)
correction = 1 - (3 / (4 * n - 1))
expected_g = expected_d * correction
```

### 4.3 Paired t-test
**Verification**:
```python
from scipy import stats
t_stat, p_value = stats.ttest_rel(ablation, baseline)
```

### 4.4 Wilcoxon Signed-Rank Test
**Verification**:
```python
from scipy import stats
w_stat, w_p = stats.wilcoxon(diff[diff != 0])
```

### 4.5 Confidence Interval for Mean Difference
**Formula**: `CI = mean(diff) ± t_crit * sem(diff)`

**Verification**:
```python
from scipy import stats
se = stats.sem(diff)
t_crit = stats.t.ppf(0.975, len(diff) - 1)
ci_lower = np.mean(diff) - t_crit * se
ci_upper = np.mean(diff) + t_crit * se
```

### 4.6 Holm-Bonferroni Correction
**Algorithm**:
1. Sort p-values in ascending order
2. For i-th p-value, threshold = α / (n - i + 1)
3. Reject if p < threshold, stop at first non-rejection

**Verification**:
```python
p_values = [0.01, 0.03, 0.04, 0.06]
n = len(p_values)
sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])

significant = [False] * n
for i, (orig_idx, p) in enumerate(sorted_p):
    threshold = 0.05 / (n - i)
    if p < threshold:
        significant[orig_idx] = True
    else:
        break
```

---

## 5. OUTPUT FILE AUDIT

### What to Verify
- All output files are created
- JSON files have valid structure
- CSV file is properly formatted
- No data loss during serialization

### Expected Output Files
| File | Purpose | Verified |
|------|---------|----------|
| `ablation_summary.json` | Main summary with all results | [ ] |
| `statistical_analysis.json` | Statistical comparisons | [ ] |
| `per_session_results.json` | Per-session breakdown | [ ] |
| `training_curves.json` | Full training history | [ ] |
| `results_table.csv` | Publication-ready table | [ ] |
| `{ablation}/ablation_result.json` | Per-ablation aggregated | [ ] |
| `{ablation}/{ablation}_fold{N}_results.json` | Raw train.py output | [ ] |

### JSON Schema Verification
```python
import json
from jsonschema import validate

# Define expected schema
summary_schema = {
    "type": "object",
    "required": ["metadata", "results", "statistical_comparisons", "summary_table"],
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["timestamp", "n_ablations", "n_folds", "baseline"]
        },
        "results": {"type": "object"},
        "statistical_comparisons": {"type": "object"},
        "summary_table": {"type": "array"}
    }
}

with open("ablation_summary.json") as f:
    data = json.load(f)
validate(data, summary_schema)
```

---

## 6. END-TO-END VERIFICATION

### Dry Run Test
```bash
# Run dry-run for all ablations
python experiments/run_ablation_3fold.py --dry-run --folds 0

# Check output for each ablation command
# Verify arguments are correct
```

### Mini Integration Test
```bash
# Run single ablation, single fold with minimal epochs
python experiments/run_ablation_3fold.py \
    --ablations baseline \
    --folds 0 \
    --output-dir results/audit_test

# Verify output files exist and have correct structure
ls results/audit_test/
cat results/audit_test/baseline/baseline_fold0_results.json
```

### Numerical Verification
After a real run, manually verify:
1. Pick one ablation result
2. Recompute statistics from raw fold values
3. Compare with saved statistics

```python
import json
import numpy as np
from scipy import stats

# Load results
with open("results/ablation_3fold/ablation_summary.json") as f:
    data = json.load(f)

# Get baseline fold values
baseline_r2s = data["results"]["baseline"]["fold_r2_values"]
ablation_r2s = data["results"]["conv_type_standard"]["fold_r2_values"]

# Manually compute statistics
diff = np.array(ablation_r2s) - np.array(baseline_r2s)
manual_d = np.mean(diff) / np.std(diff, ddof=1)
manual_t, manual_p = stats.ttest_rel(ablation_r2s, baseline_r2s)

# Compare with saved values
saved_d = data["statistical_comparisons"]["conv_type_standard"]["cohens_d"]
saved_p = data["statistical_comparisons"]["conv_type_standard"]["t_pvalue"]

print(f"Cohen's d: manual={manual_d:.4f}, saved={saved_d:.4f}")
print(f"p-value: manual={manual_p:.4f}, saved={saved_p:.4f}")

assert abs(manual_d - saved_d) < 1e-6, "Cohen's d mismatch!"
assert abs(manual_p - saved_p) < 1e-6, "p-value mismatch!"
```

---

## 7. COMMON PITFALLS CHECKLIST

### Argument Passing
- [ ] `--cond-mode` vs `--conditioning` confusion
- [ ] Missing `--no-bidirectional` for non-bidirectional runs
- [ ] Forgetting `--force-recreate-splits` for new sessions
- [ ] Wrong session names passed to `--val-sessions`

### Session Splits
- [ ] Sessions not properly sorted
- [ ] Uneven session distribution (remainder handling)
- [ ] Train/test overlap

### Metric Capture
- [ ] Using `val_r2` instead of `best_val_r2`
- [ ] Missing per-session metrics
- [ ] Incorrect data type conversions

### Statistical Analysis
- [ ] Using independent t-test instead of paired
- [ ] Forgetting to handle ties in Wilcoxon test
- [ ] Wrong degrees of freedom for CI
- [ ] Not applying multiple comparison correction

### File Output
- [ ] JSON serialization of numpy types
- [ ] Missing directories (not using makedirs)
- [ ] File encoding issues

---

## 8. AUTOMATED AUDIT SCRIPT

Run the automated audit script to verify everything:

```bash
python experiments/verify_ablation_setup.py
```

This script will:
1. Verify all argument mappings
2. Test session split logic
3. Validate statistical formulas
4. Check output file schemas
5. Run a minimal dry-run test

---

## Sign-Off

| Audit Section | Verified By | Date | Notes |
|---------------|-------------|------|-------|
| Argument Passing | | | |
| Session Splits | | | |
| Metric Capture | | | |
| Statistical Analysis | | | |
| Output Files | | | |
| End-to-End | | | |
