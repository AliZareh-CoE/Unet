#!/bin/bash
# =============================================================================
# Loss Function Ablation Experiments
# =============================================================================
# Runs 16 experiments with different loss function combinations
# Each experiment runs for 80 epochs with n_downsample=2
# Uses torchrun with 8 GPUs and FSDP
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
N_GPUS=8
BASE_DIR="artifacts/loss_ablation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse arguments
START_EXP=${1:-1}
END_EXP=${2:-16}

echo "=============================================="
echo "Loss Function Ablation Study"
echo "=============================================="
echo "Epochs per experiment: $EPOCHS"
echo "Downsample levels: $N_DOWNSAMPLE"
echo "GPUs: $N_GPUS"
echo "Running experiments: $START_EXP to $END_EXP"
echo "Output directory: $BASE_DIR"
echo "=============================================="

# Base command (common args for all experiments)
BASE_CMD="torchrun --nproc_per_node=$N_GPUS train.py \
    --fsdp --fsdp-strategy grad_op \
    --per-channel-norm --output-scaling \
    --epochs $EPOCHS --n-downsample $N_DOWNSAMPLE"

# Function to run a single experiment
run_experiment() {
    local exp_num=$1
    local exp_name=$2
    shift 2
    local extra_args="$@"

    echo ""
    echo "=============================================="
    echo "Experiment $exp_num/16: $exp_name"
    echo "=============================================="
    echo "Extra args: $extra_args"
    echo ""

    # Create experiment-specific output directories
    local exp_dir="${BASE_DIR}/${exp_name}"
    mkdir -p "$exp_dir"

    # Set ALL output directories via environment variables
    # This ensures checkpoints, logs, and plots all go to experiment folder
    export OUTPUT_DIR="$exp_dir"
    export CHECKPOINT_DIR="$exp_dir/checkpoints"
    export LOGS_DIR="$exp_dir/logs"
    mkdir -p "$CHECKPOINT_DIR" "$LOGS_DIR"

    # Run training
    local log_file="${exp_dir}/train.log"
    echo "Output directory: $OUTPUT_DIR"
    echo "Checkpoints: $CHECKPOINT_DIR"
    echo "Logs: $LOGS_DIR"
    echo "Training log: $log_file"
    echo ""

    $BASE_CMD $extra_args 2>&1 | tee "$log_file"

    # Extract final metrics
    echo ""
    echo "=============================================="
    echo "Results for $exp_name:"
    echo "=============================================="
    grep -E "Best:|val.*corr|val.*r2|psd_err|STAGE" "$log_file" | tail -15 || true
    echo ""
    echo "Output saved to: $exp_dir"
    echo ""
}

# =============================================================================
# EXPERIMENT DEFINITIONS (16 experiments)
# =============================================================================

# Experiment 1: Baseline - L1 + Wavelet only (no frequency-domain losses)
if [ $START_EXP -le 1 ] && [ $END_EXP -ge 1 ]; then
    run_experiment 1 "01_baseline_l1_wavelet" \
        --no-spectral-loss --no-freq-l1-loss
fi

# Experiment 2: + Spectral loss (no scaling)
if [ $START_EXP -le 2 ] && [ $END_EXP -ge 2 ]; then
    run_experiment 2 "02_spectral_none" \
        --use-spectral-loss --no-freq-l1-loss \
        --spectral-freq-scaling none
fi

# Experiment 3: + Spectral loss (power scaling - 1/f compensation)
if [ $START_EXP -le 3 ] && [ $END_EXP -ge 3 ]; then
    run_experiment 3 "03_spectral_power" \
        --use-spectral-loss --no-freq-l1-loss \
        --spectral-freq-scaling power
fi

# Experiment 4: + Spectral loss (focal - focus on hard frequencies)
if [ $START_EXP -le 4 ] && [ $END_EXP -ge 4 ]; then
    run_experiment 4 "04_spectral_focal" \
        --use-spectral-loss --no-freq-l1-loss \
        --spectral-freq-scaling focal
fi

# Experiment 5: + FreqL1 loss (no scaling)
if [ $START_EXP -le 5 ] && [ $END_EXP -ge 5 ]; then
    run_experiment 5 "05_freql1_none" \
        --no-spectral-loss --use-freq-l1-loss \
        --freq-l1-scaling none
fi

# Experiment 6: + FreqL1 loss (power scaling)
if [ $START_EXP -le 6 ] && [ $END_EXP -ge 6 ]; then
    run_experiment 6 "06_freql1_power" \
        --no-spectral-loss --use-freq-l1-loss \
        --freq-l1-scaling power
fi

# Experiment 7: All losses (no scaling)
if [ $START_EXP -le 7 ] && [ $END_EXP -ge 7 ]; then
    run_experiment 7 "07_all_none" \
        --use-spectral-loss --use-freq-l1-loss \
        --spectral-freq-scaling none --freq-l1-scaling none
fi

# Experiment 8: All losses (power scaling) - RECOMMENDED
if [ $START_EXP -le 8 ] && [ $END_EXP -ge 8 ]; then
    run_experiment 8 "08_all_power" \
        --use-spectral-loss --use-freq-l1-loss \
        --spectral-freq-scaling power --freq-l1-scaling power
fi

# Experiment 9: All losses (spectral=focal, freql1=power)
if [ $START_EXP -le 9 ] && [ $END_EXP -ge 9 ]; then
    run_experiment 9 "09_all_focal_spectral" \
        --use-spectral-loss --use-freq-l1-loss \
        --spectral-freq-scaling focal --freq-l1-scaling power
fi

# Experiment 10: No time-domain L1 (frequency-only reconstruction)
if [ $START_EXP -le 10 ] && [ $END_EXP -ge 10 ]; then
    run_experiment 10 "10_no_l1_freq_only" \
        --no-l1-loss --use-spectral-loss --use-freq-l1-loss \
        --spectral-freq-scaling power --freq-l1-scaling power
fi

# Experiment 11: L1 + Spectral only (no wavelet, no freql1)
if [ $START_EXP -le 11 ] && [ $END_EXP -ge 11 ]; then
    run_experiment 11 "11_l1_spectral_only" \
        --no-wavelet-loss --use-spectral-loss --no-freq-l1-loss \
        --spectral-freq-scaling power
fi

# Experiment 12: L1 + FreqL1 only (no wavelet, no spectral)
if [ $START_EXP -le 12 ] && [ $END_EXP -ge 12 ]; then
    run_experiment 12 "12_l1_freql1_only" \
        --no-wavelet-loss --no-spectral-loss --use-freq-l1-loss \
        --freq-l1-scaling power
fi

# Experiment 13: All losses + phase penalty in FreqL1
if [ $START_EXP -le 13 ] && [ $END_EXP -ge 13 ]; then
    run_experiment 13 "13_all_with_phase" \
        --use-spectral-loss --use-freq-l1-loss \
        --spectral-freq-scaling power --freq-l1-scaling power \
        --freq-l1-use-phase
fi

# Experiment 14: All losses with higher power exponent (1.5 - stronger high-freq boost)
if [ $START_EXP -le 14 ] && [ $END_EXP -ge 14 ]; then
    run_experiment 14 "14_all_power_1.5" \
        --use-spectral-loss --use-freq-l1-loss \
        --spectral-freq-scaling power --freq-l1-scaling power \
        --power-exponent 1.5
fi

# Experiment 15: All losses with linear scaling
if [ $START_EXP -le 15 ] && [ $END_EXP -ge 15 ]; then
    run_experiment 15 "15_all_linear" \
        --use-spectral-loss --use-freq-l1-loss \
        --spectral-freq-scaling linear --freq-l1-scaling linear
fi

# Experiment 16: All losses with log scaling
if [ $START_EXP -le 16 ] && [ $END_EXP -ge 16 ]; then
    run_experiment 16 "16_all_log" \
        --use-spectral-loss --use-freq-l1-loss \
        --spectral-freq-scaling log --freq-l1-scaling log
fi

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Results saved to: $BASE_DIR"
echo ""
echo "To compare results:"
echo "  for d in ${BASE_DIR}/*/; do echo \"\$d\"; grep 'Best:' \"\$d/train.log\" | tail -1; done"
echo ""
