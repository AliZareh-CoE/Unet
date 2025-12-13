#!/bin/bash
# =============================================================================
# Loss Function Ablation Experiments
# =============================================================================
# Runs 16 experiments with different loss function combinations
# Each experiment runs for 80 epochs with n_downsample=2
#
# Usage: ./run_loss_experiments.sh [start_exp] [end_exp]
#   start_exp: Starting experiment number (1-16, default: 1)
#   end_exp: Ending experiment number (1-16, default: 16)
#
# Example: ./run_loss_experiments.sh 1 4   # Run experiments 1-4 only
# =============================================================================

set -e  # Exit on error

# Configuration
EPOCHS=80
N_DOWNSAMPLE=2
BASE_DIR="artifacts/loss_ablation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${BASE_DIR}/logs_${TIMESTAMP}"

# Parse arguments
START_EXP=${1:-1}
END_EXP=${2:-16}

# Create directories
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Loss Function Ablation Study"
echo "=============================================="
echo "Epochs per experiment: $EPOCHS"
echo "Downsample levels: $N_DOWNSAMPLE"
echo "Running experiments: $START_EXP to $END_EXP"
echo "Log directory: $LOG_DIR"
echo "=============================================="

# Function to run a single experiment
run_experiment() {
    local exp_num=$1
    local exp_name=$2
    local config_overrides=$3

    echo ""
    echo "=============================================="
    echo "Experiment $exp_num/16: $exp_name"
    echo "=============================================="
    echo "Config: $config_overrides"
    echo ""

    # Create experiment-specific checkpoint dir
    export CHECKPOINT_DIR="${BASE_DIR}/checkpoints/${exp_name}"
    mkdir -p "$CHECKPOINT_DIR"

    # Run training with config overrides
    # Using Python to pass config as JSON
    python -c "
import sys
sys.path.insert(0, '.')
import json
import train

# Base config overrides
config = {
    'num_epochs': $EPOCHS,
    'n_downsample': $N_DOWNSAMPLE,
    $config_overrides
}

# Print config
print('Config overrides:', json.dumps(config, indent=2))

# Update default config and run
for k, v in config.items():
    train.DEFAULT_CONFIG[k] = v

# Run training
train.main()
" 2>&1 | tee "${LOG_DIR}/exp${exp_num}_${exp_name}.log"

    echo "Experiment $exp_num complete. Log: ${LOG_DIR}/exp${exp_num}_${exp_name}.log"
}

# =============================================================================
# EXPERIMENT DEFINITIONS (16 experiments)
# =============================================================================

# Experiment 1: Baseline - L1 + Wavelet only (no frequency-domain losses)
if [ $START_EXP -le 1 ] && [ $END_EXP -ge 1 ]; then
    run_experiment 1 "baseline_l1_wavelet" "
    'use_l1_loss': True,
    'use_wavelet_loss': True,
    'use_spectral_loss': False,
    'use_freq_l1_loss': False
"
fi

# Experiment 2: + Spectral loss (no scaling)
if [ $START_EXP -le 2 ] && [ $END_EXP -ge 2 ]; then
    run_experiment 2 "add_spectral_none" "
    'use_l1_loss': True,
    'use_wavelet_loss': True,
    'use_spectral_loss': True,
    'use_freq_l1_loss': False,
    'spectral_freq_scaling': 'none'
"
fi

# Experiment 3: + Spectral loss (power scaling - 1/f compensation)
if [ $START_EXP -le 3 ] && [ $END_EXP -ge 3 ]; then
    run_experiment 3 "add_spectral_power" "
    'use_l1_loss': True,
    'use_wavelet_loss': True,
    'use_spectral_loss': True,
    'use_freq_l1_loss': False,
    'spectral_freq_scaling': 'power',
    'spectral_power_exponent': 1.0
"
fi

# Experiment 4: + Spectral loss (focal - focus on hard frequencies)
if [ $START_EXP -le 4 ] && [ $END_EXP -ge 4 ]; then
    run_experiment 4 "add_spectral_focal" "
    'use_l1_loss': True,
    'use_wavelet_loss': True,
    'use_spectral_loss': True,
    'use_freq_l1_loss': False,
    'spectral_freq_scaling': 'focal',
    'spectral_focal_gamma': 2.0
"
fi

# Experiment 5: + FreqL1 loss (no scaling)
if [ $START_EXP -le 5 ] && [ $END_EXP -ge 5 ]; then
    run_experiment 5 "add_freql1_none" "
    'use_l1_loss': True,
    'use_wavelet_loss': True,
    'use_spectral_loss': False,
    'use_freq_l1_loss': True,
    'freq_l1_scaling': 'none'
"
fi

# Experiment 6: + FreqL1 loss (power scaling)
if [ $START_EXP -le 6 ] && [ $END_EXP -ge 6 ]; then
    run_experiment 6 "add_freql1_power" "
    'use_l1_loss': True,
    'use_wavelet_loss': True,
    'use_spectral_loss': False,
    'use_freq_l1_loss': True,
    'freq_l1_scaling': 'power',
    'freq_l1_power_exponent': 1.0
"
fi

# Experiment 7: All losses (no scaling)
if [ $START_EXP -le 7 ] && [ $END_EXP -ge 7 ]; then
    run_experiment 7 "all_losses_none" "
    'use_l1_loss': True,
    'use_wavelet_loss': True,
    'use_spectral_loss': True,
    'use_freq_l1_loss': True,
    'spectral_freq_scaling': 'none',
    'freq_l1_scaling': 'none'
"
fi

# Experiment 8: All losses (power scaling) - RECOMMENDED
if [ $START_EXP -le 8 ] && [ $END_EXP -ge 8 ]; then
    run_experiment 8 "all_losses_power" "
    'use_l1_loss': True,
    'use_wavelet_loss': True,
    'use_spectral_loss': True,
    'use_freq_l1_loss': True,
    'spectral_freq_scaling': 'power',
    'freq_l1_scaling': 'power',
    'spectral_power_exponent': 1.0,
    'freq_l1_power_exponent': 1.0
"
fi

# Experiment 9: All losses (spectral=focal, freql1=power)
if [ $START_EXP -le 9 ] && [ $END_EXP -ge 9 ]; then
    run_experiment 9 "all_focal_spectral" "
    'use_l1_loss': True,
    'use_wavelet_loss': True,
    'use_spectral_loss': True,
    'use_freq_l1_loss': True,
    'spectral_freq_scaling': 'focal',
    'freq_l1_scaling': 'power',
    'spectral_focal_gamma': 2.0
"
fi

# Experiment 10: No time-domain L1 (frequency-only reconstruction)
if [ $START_EXP -le 10 ] && [ $END_EXP -ge 10 ]; then
    run_experiment 10 "no_l1_freq_only" "
    'use_l1_loss': False,
    'use_wavelet_loss': True,
    'use_spectral_loss': True,
    'use_freq_l1_loss': True,
    'spectral_freq_scaling': 'power',
    'freq_l1_scaling': 'power'
"
fi

# Experiment 11: L1 + Spectral only (no wavelet, no freql1)
if [ $START_EXP -le 11 ] && [ $END_EXP -ge 11 ]; then
    run_experiment 11 "l1_spectral_only" "
    'use_l1_loss': True,
    'use_wavelet_loss': False,
    'use_spectral_loss': True,
    'use_freq_l1_loss': False,
    'spectral_freq_scaling': 'power'
"
fi

# Experiment 12: L1 + FreqL1 only (no wavelet, no spectral)
if [ $START_EXP -le 12 ] && [ $END_EXP -ge 12 ]; then
    run_experiment 12 "l1_freql1_only" "
    'use_l1_loss': True,
    'use_wavelet_loss': False,
    'use_spectral_loss': False,
    'use_freq_l1_loss': True,
    'freq_l1_scaling': 'power'
"
fi

# Experiment 13: All losses + phase penalty in FreqL1
if [ $START_EXP -le 13 ] && [ $END_EXP -ge 13 ]; then
    run_experiment 13 "all_with_phase" "
    'use_l1_loss': True,
    'use_wavelet_loss': True,
    'use_spectral_loss': True,
    'use_freq_l1_loss': True,
    'spectral_freq_scaling': 'power',
    'freq_l1_scaling': 'power',
    'freq_l1_use_phase': True,
    'freq_l1_phase_weight': 0.1
"
fi

# Experiment 14: All losses with higher power exponent (1.5 - stronger high-freq boost)
if [ $START_EXP -le 14 ] && [ $END_EXP -ge 14 ]; then
    run_experiment 14 "all_power_1.5" "
    'use_l1_loss': True,
    'use_wavelet_loss': True,
    'use_spectral_loss': True,
    'use_freq_l1_loss': True,
    'spectral_freq_scaling': 'power',
    'freq_l1_scaling': 'power',
    'spectral_power_exponent': 1.5,
    'freq_l1_power_exponent': 1.5
"
fi

# Experiment 15: All losses with linear scaling
if [ $START_EXP -le 15 ] && [ $END_EXP -ge 15 ]; then
    run_experiment 15 "all_linear" "
    'use_l1_loss': True,
    'use_wavelet_loss': True,
    'use_spectral_loss': True,
    'use_freq_l1_loss': True,
    'spectral_freq_scaling': 'linear',
    'freq_l1_scaling': 'linear',
    'spectral_scaling_factor': 2.0,
    'freq_l1_scaling_factor': 2.0
"
fi

# Experiment 16: All losses with log scaling
if [ $START_EXP -le 16 ] && [ $END_EXP -ge 16 ]; then
    run_experiment 16 "all_log" "
    'use_l1_loss': True,
    'use_wavelet_loss': True,
    'use_spectral_loss': True,
    'use_freq_l1_loss': True,
    'spectral_freq_scaling': 'log',
    'freq_l1_scaling': 'log',
    'spectral_scaling_factor': 2.0,
    'freq_l1_scaling_factor': 2.0
"
fi

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Results saved to: $BASE_DIR"
echo "Logs saved to: $LOG_DIR"
echo ""
echo "To compare results, check the validation metrics in each log file:"
echo "  grep -E 'val_corr|val_r2|psd_err' ${LOG_DIR}/*.log"
echo ""
