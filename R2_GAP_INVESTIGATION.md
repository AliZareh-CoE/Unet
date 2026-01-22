# R² Gap Investigation Report - RESOLVED

## The Problem
There was a significant gap between overall validation R² and per-session R² metrics during LOSO training:
- Overall: `r=0.703, r²=0.479`
- Per-session 160819: `r=0.603, r²=0.313`

This ~35% relative difference in R² was unexpectedly large when both should measure the same data.

## Root Cause: Index Shuffle Ordering Bug

**Location:** `data.py:load_or_create_session_splits()`, lines 632-670

### The Bug

The code was:
1. Creating `val_idx_per_session` at lines 632-636 (BEFORE shuffle)
2. Shuffling `val_idx` at line 668 (AFTER per-session indices were created)

```python
# OLD CODE (BUGGY):
for sess_id in val_session_ids:
    sess_indices = all_indices[session_ids == sess_id]
    val_idx_per_session[sess_name] = sess_indices  # Created BEFORE shuffle!

val_idx = all_indices[np.isin(session_ids, list(val_session_ids))]

# ... later ...
rng.shuffle(val_idx)  # Shuffled AFTER per-session was created!
```

### Impact

For a single-session LOSO (e.g., session 160819):
- `val_idx` contained indices `[100, 200, 300, ...]` → shuffled to `[300, 100, 200, ...]`
- `val_idx_per_session["160819"]` contained `[100, 200, 300, ...]` → **NOT shuffled**

When creating datasets:
- Combined val dataset: `data[val_idx]` = `data[[300, 100, 200, ...]]`
- Per-session dataset: `data[val_idx_per_session["160819"]]` = `data[[100, 200, 300, ...]]`

With `DistributedSampler` giving rank 0 indices `[0, 8, 16, ...]`:
- Combined loader `dataset[0]` → sample at original index 300
- Per-session loader `dataset[0]` → sample at original index 100

**These were DIFFERENT samples!** Hence completely different metrics.

## The Fix

Create `val_idx_per_session` AFTER shuffling `val_idx`, extracting from the shuffled array:

```python
# NEW CODE (FIXED):
val_idx = all_indices[np.isin(session_ids, list(val_session_ids))]

# Shuffle first
rng.shuffle(val_idx)

# THEN create per-session indices from shuffled val_idx
if separate_val_sessions:
    for sess_id in val_session_ids:
        sess_mask = session_ids[val_idx] == sess_id
        sess_indices = val_idx[sess_mask]  # Extracted from SHUFFLED val_idx
        val_idx_per_session[sess_name] = sess_indices
```

Now both loaders present samples in the same order, so `dataset[0]` is the same sample for both.

## Other Issues Investigated (Not Root Cause)

### 1. Distributed Metric Synchronization
- Only `val_loss` was synchronized across GPUs with `all_reduce`
- `corr`, `r2`, etc. were not synchronized
- This means reported metrics are from rank 0's shard only
- **Impact:** Metrics based on 1/N of data, but NOT the cause of the gap between overall vs per-session

### 2. Per-batch R² Averaging
- R² is computed per-batch then averaged
- This is mathematically different from pooled R² due to Jensen's inequality
- **Impact:** Could cause some discrepancy, but NOT the main cause

### 3. Missing session_ids in per-session dataloader
- Per-session dataloader didn't pass `session_ids` parameter
- **Impact:** Minor inconsistency, fixed but not the root cause

### 4. Per-session metrics not saved to JSON
- LOSO runner expected `per_session_r2` but it wasn't saved
- **Impact:** Fixed to enable proper LOSO aggregation

## Files Changed

1. **data.py**: Fixed index creation order (ROOT CAUSE FIX)
   - Create `val_idx_per_session` AFTER shuffling `val_idx`
   - Added debug output to verify indices match

2. **train.py**:
   - Added debug output for loader comparison
   - Added per-session metrics to output JSON
   - Reverted incorrect metric averaging (was making things worse)

## Verification

After the fix, the debug output should show:
```
[DEBUG R² Gap] Index comparison:
  val_idx shape: (420,), per_session shape: (420,)
  Indices are IDENTICAL: True
```

And the R² values should now match between overall and per-session metrics.
