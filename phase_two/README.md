# Phase 2: Architecture Screening

## Overview

Phase 2 compares **6 neural network architectures** on the olfactory dataset to identify the top performers. Each architecture is evaluated with 5-fold cross-validation for statistical validity.

**Key Question**: Which neural architecture performs best for neural signal translation?

## Architectures (6 Total)

| Architecture | Type | Key Feature | Complexity |
|-------------|------|-------------|------------|
| `linear` | Baseline | Simple MLP per time point | O(T) |
| `simplecnn` | CNN | Standard conv-bn-relu blocks | O(T) |
| `wavenet` | Dilated CNN | Exponentially increasing dilation | O(T log T) |
| `fnet` | Fourier | FFT-based token mixing | O(T log T) |
| `vit` | Transformer | Full self-attention | O(T²) |
| `condunet` | **U-Net** | **Skip connections + conditioning** | O(T) |

## Quick Start

```bash
# Full screening (6 architectures × 5 folds = 30 runs)
python -m phase_two.runner --dataset olfactory --epochs 60

# Quick test
python -m phase_two.runner --dry-run

# Run specific architectures
python -m phase_two.runner --arch condunet wavenet vit

# With FSDP multi-GPU
python -m phase_two.runner --use-train-py --fsdp
```

## Directory Structure

```
phase_two/
├── __init__.py              # Package exports
├── config.py                # Configuration
├── runner.py                # Main execution script
├── trainer.py               # Training loop
├── README.md                # This file
└── architectures/
    ├── __init__.py          # Architecture exports
    ├── registry.py          # Factory and registry
    ├── linear.py            # Linear baseline
    ├── simplecnn.py         # Simple CNN
    ├── wavenet.py           # WaveNet1D
    ├── fnet.py              # FNet1D
    ├── vit.py               # Vision Transformer 1D
    └── condunet.py          # Conditional U-Net
```

## Training Configuration

```python
TrainingConfig(
    epochs=60,              # Training epochs
    batch_size=32,          # Batch size
    learning_rate=1e-3,     # Initial LR
    weight_decay=1e-4,      # L2 regularization
    warmup_epochs=5,        # LR warmup
    lr_scheduler="cosine",  # Cosine annealing
    patience=15,            # Early stopping patience
    grad_clip=1.0,          # Gradient clipping
    loss_fn="l1",           # L1 loss
    use_amp=True,           # Mixed precision
    use_augmentation=True,  # Data augmentation
)
```

## Output

### Results File (`phase2_results_YYYYMMDD_HHMMSS.json`)

```json
{
  "results": [...],           // Per-run results
  "aggregated": {             // Per-architecture summary
    "condunet": {"r2_mean": 0.52, "r2_std": 0.02, ...},
    ...
  },
  "best_architecture": "condunet",
  "best_r2": 0.52,
  "gate_passed": true,
  "classical_baseline_r2": 0.35
}
```

## Gate Condition

From Phase 1, neural methods must beat the classical baseline by R² ≥ 0.10:

```
GATE_THRESHOLD = classical_r2 + 0.10
```

If the best neural architecture passes this gate, we proceed to Phase 3 ablations.

## API Usage

```python
from phase_two.architectures import create_architecture, list_architectures

# List available architectures
print(list_architectures())
# ['linear', 'simplecnn', 'wavenet', 'fnet', 'vit', 'condunet']

# Create model
model = create_architecture(
    "condunet",
    in_channels=32,
    out_channels=32,
    base_channels=64,
)

# Forward pass
y_pred = model(x)  # x: [B, 32, 5000]
```

## Architecture Details

### Linear Baseline
Simple learned linear mapping. MLP applied per time point.

### SimpleCNN
Stack of conv-bn-relu blocks. No fancy components.

### WaveNet1D
Dilated causal convolutions with gated activations. Exponentially increasing dilation for large receptive field.

### FNet1D
Replaces attention with FFT-based mixing. O(N log N) complexity.

### ViT1D
Vision Transformer adapted for 1D. Patches signal into segments, applies full self-attention.

### NeuroGate
Our proposed architecture:
- U-Net encoder-decoder with skip connections
- FiLM conditioning for adaptive modulation
- Cross-frequency attention mechanisms
- Residual learning (predicts delta)

## Expected Results

Typical ranking on olfactory dataset:

| Rank | Architecture | R² (approx) |
|------|-------------|-------------|
| 1 | condunet | 0.50-0.55 |
| 2 | wavenet | 0.45-0.50 |
| 3 | vit | 0.38-0.43 |
| 4 | fnet | 0.35-0.40 |
| 5 | simplecnn | 0.32-0.38 |
| 6 | linear | 0.25-0.30 |

## Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **RAM**: 16GB+ system memory
- **Time**: ~2-4 hours for full screening (30 runs)

## Citation

See main repository README for citation information.
