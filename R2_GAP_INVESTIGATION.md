# R² Gap Investigation Report

## The Problem
There's a significant gap between overall validation R² and per-session R² metrics during LOSO training:
- Overall: `r=0.703, r²=0.479`
- Per-session 160819: `r=0.603, r²=0.313`

This ~35% relative difference in R² is unexpectedly large when both should be measuring the same data.

## Root Cause Analysis

### Finding 1: Missing Metric Synchronization in Distributed Training (BUG)

**Location:** `train.py:2894-2900`

```python
# Only val_loss is synchronized!
if dist.is_initialized():
    val_loss_tensor = torch.tensor(val_loss, device=device)
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
    val_loss = val_loss_tensor.item()
    val_metrics["loss"] = val_loss
# MISSING: No synchronization for corr, r2, mse, mae!
```

**Impact:** With FSDP and 8 GPUs, each GPU only sees 1/8th of the validation data. The printed metrics come from rank 0 only, which means:
- Only ~53 out of 420 validation samples are used for the reported metrics
- Small sample size increases metric variance

### Finding 2: Per-batch R² Averaging vs Pooled R² (Mathematical Issue)

**Location:** `train.py:1522, 1622`

```python
r2_list.append(explained_variance_torch(pred_f32, pcx_f32).item())  # Per-batch
...
"r2": float(np.mean(r2_list))  # Simple average
```

**The Math Problem:**
R² = 1 - (MSE / Var(target))

When averaging R² across batches:
- Each batch has different target variance
- Average of ratios ≠ ratio of averages (Jensen's inequality)

**Example:**
- Batch A: MSE=100, Var=1000 → R²=0.90
- Batch B: MSE=100, Var=100 → R²=0.00
- **Averaged R² = 0.45**

But if computed on pooled data:
- Pooled MSE = 100, Pooled Var ≈ 550
- **Pooled R² = 0.82**

This can cause significant discrepancies when batches have heterogeneous variance.

### Finding 3: Dataset Creation Difference (Potential Bug)

**Combined validation loader** (`data.py:1810-1815`):
```python
val_dataset = PairedConditionalDataset(
    data["ob"], data["pcx"], data["odors"], data["val_idx"],
    session_ids=session_ids,  # <-- Passes session_ids
    ...
)
```

**Per-session validation loader** (`data.py:1944-1946`):
```python
dataset = PairedConditionalDataset(
    data["ob"], data["pcx"], data["odors"], indices,
    # session_ids NOT passed - defaults to using odors!
)
```

When `session_ids` is not passed, `PairedConditionalDataset` falls back to using odor IDs:
```python
if session_ids is not None:
    self.session_ids = torch.from_numpy(session_ids[indices]).long()
else:
    self.session_ids = self.odors  # Fallback to odors
```

**Impact:** While this shouldn't affect R² directly (since use_session_embedding=False), it's an inconsistency that could cause issues in other scenarios.

### Finding 4: Per-session R² Not Saved to Results JSON (Missing Feature)

**Location:** `train.py:4467-4516`

The `output_results` dictionary does NOT include per-session R² values:
```python
output_results = {
    "best_val_r2": best_val_r2,  # Only overall
    "val_r2s": val_r2s,          # Only overall
    # Missing: per_session_r2 dict!
}
```

And in LOSO runner (`LOSO/runner.py:372`):
```python
per_session_r2=results.get("per_session_r2", {}),  # Always empty!
```

## Recommendations

### Fix 1: Synchronize All Metrics Across GPUs

```python
# In train.py, after line 2900, add:
if dist.is_initialized():
    for metric_name in ["corr", "r2", "mae", "mse"]:
        if metric_name in val_metrics:
            metric_tensor = torch.tensor(val_metrics[metric_name], device=device)
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
            val_metrics[metric_name] = metric_tensor.item()

    # Also sync per-session metrics
    for sess_name, sess_m in per_session_metrics.items():
        for metric_name in ["corr", "r2", "mae", "mse"]:
            if metric_name in sess_m:
                metric_tensor = torch.tensor(sess_m[metric_name], device=device)
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
                sess_m[metric_name] = metric_tensor.item()
```

### Fix 2: Use Pooled R² Calculation

Instead of averaging per-batch R², accumulate MSE and Var separately, then compute R² at the end:

```python
# In evaluate(), replace per-batch R² with:
total_mse = 0.0
total_var = 0.0
total_samples = 0

for batch in loader:
    ...
    mse = ((pred_f32 - pcx_f32) ** 2).sum().item()
    var = ((pcx_f32 - pcx_f32.mean()) ** 2).sum().item()
    n = pred_f32.numel()

    total_mse += mse
    total_var += var
    total_samples += n

# After loop:
pooled_r2 = 1.0 - (total_mse / total_samples) / (total_var / total_samples)
```

### Fix 3: Pass session_ids Consistently

In `create_single_session_dataloader`:
```python
dataset = PairedConditionalDataset(
    data["ob"], data["pcx"], data["odors"], indices,
    session_ids=data.get("session_ids"),  # Add this line
)
```

### Fix 4: Save Per-Session Metrics to Results

In `train.py` around line 4500:
```python
# Add per-session metrics
if history:
    per_session_r2 = {}
    per_session_corr = {}
    for sess_name in per_session_metrics.keys():
        r2_key = f"val_{sess_name}_r2"
        corr_key = f"val_{sess_name}_corr"
        if r2_key in history[-1]:
            per_session_r2[sess_name] = history[-1][r2_key]
            per_session_corr[sess_name] = history[-1][corr_key]

    output_results["per_session_r2"] = per_session_r2
    output_results["per_session_corr"] = per_session_corr
```

## Diagnostic Steps

Run the diagnostic script `debug_r2_gap.py` to verify:
1. Whether val_idx == val_idx_per_session["160819"]
2. Whether the data tensors are identical
3. Whether per-batch averaging causes the discrepancy

```bash
python debug_r2_gap.py
```

## Summary

The R² gap is most likely caused by:
1. **Missing distributed metric synchronization** - only rank 0's metrics on 1/8th of data are printed
2. **Per-batch R² averaging** - mathematically different from pooled R²
3. **Batch variance heterogeneity** - different batches have different target variances

The recommended fixes focus on proper metric synchronization across GPUs and using pooled R² calculation for more accurate results.
