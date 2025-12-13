#!/bin/bash
# =============================================================================
# Smoke Test for Loss Ablation Experiments
# =============================================================================
# Runs each experiment for just 1 epoch to verify:
# - All CLI arguments are valid
# - Models initialize correctly
# - Loss functions work
# - Output directories are created
# - No crashes occur
#
# Usage: ./smoke_test.sh
# =============================================================================

set -e  # Exit on first error

# Configuration - minimal settings for quick test
EPOCHS=1
N_DOWNSAMPLE=2
N_GPUS=8
BASE_DIR="artifacts/smoke_test"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "SMOKE TEST - Loss Function Ablation"
echo "=============================================="
echo "Testing all 16 experiments with 1 epoch each"
echo "Output: $BASE_DIR"
echo "=============================================="
echo ""

# Clean previous smoke test
rm -rf "$BASE_DIR"
mkdir -p "$BASE_DIR"

# Base command
BASE_CMD="torchrun --nproc_per_node=$N_GPUS train.py \
    --fsdp --fsdp-strategy grad_op \
    --per-channel-norm --output-scaling \
    --epochs $EPOCHS --n-downsample $N_DOWNSAMPLE"

# Track results
PASSED=0
FAILED=0
FAILED_EXPS=""

# Function to run smoke test for single experiment
smoke_test_experiment() {
    local exp_num=$1
    local exp_name=$2
    shift 2
    local extra_args="$@"

    echo -n "[$exp_num/16] $exp_name ... "

    # Create experiment directory
    local exp_dir="${BASE_DIR}/${exp_name}"
    mkdir -p "$exp_dir"

    # Set environment variables
    export OUTPUT_DIR="$exp_dir"
    export CHECKPOINT_DIR="$exp_dir/checkpoints"
    export LOGS_DIR="$exp_dir/logs"
    mkdir -p "$CHECKPOINT_DIR" "$LOGS_DIR"

    # Run with timeout (5 minutes should be enough for 1 epoch)
    local log_file="${exp_dir}/smoke.log"

    if timeout 300 $BASE_CMD $extra_args > "$log_file" 2>&1; then
        # Check if training actually ran
        if grep -q "Epoch 1/" "$log_file"; then
            echo "PASSED ✓"
            ((PASSED++))
            return 0
        else
            echo "FAILED ✗ (no epoch output)"
            ((FAILED++))
            FAILED_EXPS="$FAILED_EXPS $exp_name"
            return 1
        fi
    else
        echo "FAILED ✗ (crashed or timeout)"
        ((FAILED++))
        FAILED_EXPS="$FAILED_EXPS $exp_name"
        # Show last few lines of error
        echo "  Last error lines:"
        tail -5 "$log_file" 2>/dev/null | sed 's/^/    /'
        return 1
    fi
}

echo ""
echo "Running smoke tests..."
echo ""

# Test all 16 experiments
smoke_test_experiment 1 "01_baseline_l1_wavelet" \
    --no-spectral-loss --no-freq-l1-loss || true

smoke_test_experiment 2 "02_spectral_none" \
    --use-spectral-loss --no-freq-l1-loss \
    --spectral-freq-scaling none || true

smoke_test_experiment 3 "03_spectral_power" \
    --use-spectral-loss --no-freq-l1-loss \
    --spectral-freq-scaling power || true

smoke_test_experiment 4 "04_spectral_focal" \
    --use-spectral-loss --no-freq-l1-loss \
    --spectral-freq-scaling focal || true

smoke_test_experiment 5 "05_freql1_none" \
    --no-spectral-loss --use-freq-l1-loss \
    --freq-l1-scaling none || true

smoke_test_experiment 6 "06_freql1_power" \
    --no-spectral-loss --use-freq-l1-loss \
    --freq-l1-scaling power || true

smoke_test_experiment 7 "07_all_none" \
    --use-spectral-loss --use-freq-l1-loss \
    --spectral-freq-scaling none --freq-l1-scaling none || true

smoke_test_experiment 8 "08_all_power" \
    --use-spectral-loss --use-freq-l1-loss \
    --spectral-freq-scaling power --freq-l1-scaling power || true

smoke_test_experiment 9 "09_all_focal_spectral" \
    --use-spectral-loss --use-freq-l1-loss \
    --spectral-freq-scaling focal --freq-l1-scaling power || true

smoke_test_experiment 10 "10_no_l1_freq_only" \
    --no-l1-loss --use-spectral-loss --use-freq-l1-loss \
    --spectral-freq-scaling power --freq-l1-scaling power || true

smoke_test_experiment 11 "11_l1_spectral_only" \
    --no-wavelet-loss --use-spectral-loss --no-freq-l1-loss \
    --spectral-freq-scaling power || true

smoke_test_experiment 12 "12_l1_freql1_only" \
    --no-wavelet-loss --no-spectral-loss --use-freq-l1-loss \
    --freq-l1-scaling power || true

smoke_test_experiment 13 "13_all_with_phase" \
    --use-spectral-loss --use-freq-l1-loss \
    --spectral-freq-scaling power --freq-l1-scaling power \
    --freq-l1-use-phase || true

smoke_test_experiment 14 "14_all_power_1.5" \
    --use-spectral-loss --use-freq-l1-loss \
    --spectral-freq-scaling power --freq-l1-scaling power \
    --power-exponent 1.5 || true

smoke_test_experiment 15 "15_all_linear" \
    --use-spectral-loss --use-freq-l1-loss \
    --spectral-freq-scaling linear --freq-l1-scaling linear || true

smoke_test_experiment 16 "16_all_log" \
    --use-spectral-loss --use-freq-l1-loss \
    --spectral-freq-scaling log --freq-l1-scaling log || true

# Summary
echo ""
echo "=============================================="
echo "SMOKE TEST SUMMARY"
echo "=============================================="
echo "Passed: $PASSED/16"
echo "Failed: $FAILED/16"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed experiments:$FAILED_EXPS"
    echo ""
    echo "Check logs in: $BASE_DIR/<exp_name>/smoke.log"
    echo ""
    echo "❌ SMOKE TEST FAILED - DO NOT RUN FULL EXPERIMENTS"
    exit 1
else
    echo ""
    echo "✅ ALL SMOKE TESTS PASSED"
    echo ""
    echo "Safe to run full experiments:"
    echo "  ./run_loss_experiments.sh"
    echo ""
    # Cleanup smoke test artifacts
    echo "Cleaning up smoke test files..."
    rm -rf "$BASE_DIR"
    echo "Done!"
    exit 0
fi
