# Train.py CLI Reference

Quick reference for command-line arguments. All arguments are optional.

## Essential Arguments

```bash
--dataset DATASET     # Dataset: olfactory, pfc, pcx1, dandi (default: olfactory)
--epochs N            # Training epochs (default: 60)
--batch-size N        # Batch size (default: 32)
--lr RATE             # Learning rate (default: 1e-4)
--seed N              # Random seed for reproducibility
```

## Distributed Training (Multi-GPU)

```bash
--fsdp                # Enable FSDP for multi-GPU training
--fsdp-strategy STR   # full, grad_op, no_shard, hybrid (default: full)

# Example: 8-GPU training
torchrun --nproc_per_node=8 train.py --dataset pcx1 --fsdp --batch-size 64
```

## Dataset-Specific Arguments

### PCx1 Continuous Dataset
```bash
--pcx1-window-size N      # Window size in samples (default: 5000 = 5s)
--pcx1-stride N           # Stride between windows
--pcx1-stride-ratio R     # Stride as ratio of window (0.5 = 50% overlap)
--pcx1-train-sessions ... # Explicit training sessions
--pcx1-val-sessions ...   # Explicit validation sessions
--pcx1-n-val N            # Number of validation sessions (default: 4)
```

### DANDI Human iEEG Dataset
```bash
--dandi-source-region R   # Source: amygdala, hippocampus, medial_frontal_cortex
--dandi-target-region R   # Target region (default: hippocampus)
--dandi-window-size N     # Window size (default: 5000)
--dandi-data-dir PATH     # Path to NWB files (default: $UNET_DATA_DIR/movie)
```

### PFC-CA1 Dataset
```bash
--pfc-sliding-window      # Use sliding window instead of fixed trials
--pfc-window-size N       # Window size (default: 2500)
--resample-pfc            # Resample from 1250Hz to 1000Hz
```

## Session Splits (Cross-Session Generalization)

```bash
--split-by-session        # Hold out entire sessions for test/val
--no-test-set             # Use all held-out sessions for validation only
--n-val-sessions N        # Number of validation sessions
--val-sessions S1 S2 ...  # Explicit validation session names
--separate-val-sessions   # Report per-session metrics
```

## Training Options

```bash
--lr-scheduler TYPE       # none, cosine, cosine_warmup
--weight-decay W          # L2 regularization (default: 0)
--val-every N             # Validate every N epochs (default: 1)
--no-aug                  # Disable all data augmentation
--no-bidirectional        # Disable reverse direction training
--compile                 # Enable torch.compile optimization
```

## Model Options

```bash
--base-channels N         # Base channel count (default: 64)
--conditioning TYPE       # spectro_temporal, odor_onehot, cpc, vqvae
--loss TYPE               # l1, huber, wavelet, l1_wavelet, huber_wavelet
--no-output-scaling       # Disable learnable output scaling
```

## Output Options

```bash
--no-plots                # Skip validation plot generation
```

## Common Usage Examples

```bash
# Basic training on olfactory dataset
python train.py --dataset olfactory --epochs 60

# PCx1 with session holdout, no test set
python train.py --dataset pcx1 --split-by-session --no-test-set --epochs 60

# DANDI human iEEG: Amygdala -> Hippocampus
python train.py --dataset dandi --dandi-source-region amygdala --dandi-target-region hippocampus

# Multi-GPU with FSDP
torchrun --nproc_per_node=8 train.py --dataset pcx1 --fsdp --batch-size 64

# Fast training (less frequent validation)
python train.py --dataset pcx1 --val-every 5 --no-plots
```

## Environment Variables

```bash
UNET_DATA_DIR=/path/to/data   # Base data directory (default: /data)
```
