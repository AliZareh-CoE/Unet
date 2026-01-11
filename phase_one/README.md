# Phase 1: Classical Baselines for Neural Signal Translation

## Overview

Phase 1 establishes the **performance floor** for neural signal translation by evaluating 7 classical signal processing methods. This baseline performance serves as the gate that neural network methods must beat in subsequent phases.

**Key Question**: What is the best performance achievable with classical linear methods?

## Methods (7 Total)

| Method | Description | Key Feature |
|--------|-------------|-------------|
| `wiener` | Single-channel Wiener filter | Optimal linear filter in frequency domain |
| `wiener_mimo` | Multi-input multi-output Wiener | Joint channel optimization |
| `ridge` | L2-regularized regression | Pointwise linear mapping |
| `ridge_temporal` | Ridge with temporal context | Sliding window features |
| `ridge_cv` | Cross-validated Ridge | Auto-tuned regularization |
| `var` | Vector Autoregressive | Temporal dynamics modeling |
| `var_exogenous` | VAR with exogenous inputs | Extended VARX model |

## Quick Start

```bash
# Run full evaluation (olfactory dataset)
python -m phase_one.runner --dataset olfactory

# Quick test with synthetic data
python -m phase_one.runner --dry-run

# Fast run (only ridge and wiener)
python -m phase_one.runner --fast

# Custom output directory
python -m phase_one.runner --output results/phase1/
```

## Directory Structure

```
phase_one/
├── __init__.py           # Package exports
├── config.py             # Configuration and constants
├── runner.py             # Main execution script
├── metrics.py            # Evaluation metrics
├── visualization.py      # Figure generation
├── baselines/
│   ├── __init__.py       # Baseline exports
│   ├── registry.py       # Factory and registry
│   ├── wiener.py         # Wiener filter implementations
│   ├── ridge.py          # Ridge regression variants
│   └── var.py            # VAR model implementations
└── README.md             # This file
```

## Output

### Results File (`phase1_results_YYYYMMDD_HHMMSS.json`)

```json
{
  "metrics": [...],
  "best_method": "wiener_mimo",
  "best_r2": 0.4523,
  "gate_threshold": 0.5523,
  "statistical_comparisons": [...],
  "config": {...},
  "timestamp": "2024-01-15T10:30:00"
}
```

### Figure 1.1: Classical Baselines

Four-panel figure:
- **(A)** R² by method (sorted bar chart with 95% CI)
- **(B)** Per-frequency-band R² heatmap
- **(C)** Example prediction vs ground truth
- **(D)** PSD comparison (spectral fidelity)

## Metrics

### Primary Metrics
- **R²**: Coefficient of determination (per-channel, then averaged)
- **95% CI**: Bootstrap confidence interval (10,000 samples)
- **Pearson correlation**: Linear correlation coefficient

### Secondary Metrics
- **MAE**: Mean absolute error
- **PSD Error (dB)**: Spectral fidelity in decibels
- **Per-band R²**: R² for delta, theta, alpha, beta, gamma bands

### Statistical Tests
- Paired t-tests (same CV folds)
- Bonferroni correction for multiple comparisons
- Cohen's d effect sizes

## Gate Condition

Neural methods (Phase 2+) must beat the classical baseline by **R² ≥ 0.10**:

```
GATE_THRESHOLD = best_classical_R² + 0.10
```

## API Usage

```python
from phase_one import run_phase1, Phase1Config
from phase_one.baselines import create_baseline, list_baselines

# List available baselines
print(list_baselines())
# ['wiener', 'wiener_mimo', 'ridge', 'ridge_temporal', 'ridge_cv', 'var', 'var_exogenous']

# Create and use a baseline
model = create_baseline("wiener_mimo")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Run full evaluation
config = Phase1Config(dataset="olfactory", n_folds=5)
result = run_phase1(config, X, y)
print(f"Best: {result.best_method} with R²={result.best_r2:.4f}")
```

## Configuration Options

```python
from phase_one.config import Phase1Config

config = Phase1Config(
    dataset="olfactory",      # Dataset: olfactory, pfc, dandi
    n_folds=5,                # CV folds
    seed=42,                  # Random seed
    n_bootstrap=10000,        # Bootstrap samples for CI
    ci_level=0.95,            # Confidence level
    sample_rate=1000.0,       # Hz
    output_dir="results/",    # Output directory
    baselines=[...],          # Methods to evaluate
)
```

## Neural Frequency Bands

```python
NEURAL_BANDS = {
    "delta": (1, 4),      # Hz - Deep sleep, slow oscillations
    "theta": (4, 8),      # Hz - Memory, navigation
    "alpha": (8, 12),     # Hz - Relaxation, attention
    "beta": (12, 30),     # Hz - Active thinking
    "gamma": (30, 100),   # Hz - Cognition, perception
}
```

## Expected Results

Typical performance on olfactory dataset:

| Method | R² (approx) |
|--------|-------------|
| wiener_mimo | 0.35-0.45 |
| ridge_cv | 0.30-0.40 |
| var_exogenous | 0.25-0.35 |
| wiener | 0.25-0.35 |
| ridge_temporal | 0.25-0.35 |
| ridge | 0.20-0.30 |
| var | 0.20-0.30 |

*Note: Actual values depend on data quality and preprocessing.*

## Dependencies

- NumPy
- SciPy
- Matplotlib

## Citation

If you use this code, please cite our Nature Methods paper:

```bibtex
@article{neural_signal_translation_2024,
  title={Neural Signal Translation with Conditional U-Net},
  journal={Nature Methods},
  year={2024}
}
```

## License

MIT License - See repository root for details.
