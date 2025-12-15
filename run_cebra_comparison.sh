#!/bin/bash
# =============================================================================
# CEBRA vs Baseline Comparison Experiment
# =============================================================================
# Tests whether CEBRA contrastive learning helps cross-session generalization
#
# Setup:
#   - 5 runs with CEBRA contrastive learning
#   - 5 runs without CEBRA (baseline)
#   - All with 4 test sessions for proper cross-session evaluation
#   - 80 epochs each
#   - Different seeds for statistical validity
#
# Usage:
#   ./run_cebra_comparison.sh          # Full experiment (10 runs x 80 epochs)
#   ./run_cebra_comparison.sh smoke    # Quick smoke test (2 runs x 2 epochs)
#   ./run_cebra_comparison.sh cebra    # Only CEBRA runs
#   ./run_cebra_comparison.sh baseline # Only baseline runs
# =============================================================================

set -e  # Exit on error

# Configuration
EPOCHS=80
N_TEST_SESSIONS=4
N_GPUS=8
SEEDS=(42 123 456 789 1024)
RESULTS_DIR="artifacts/cebra_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/results_${TIMESTAMP}.txt"

# Common training arguments
COMMON_ARGS="--fsdp --fsdp-strategy grad_op \
  --per-channel-norm --output-scaling \
  --split-by-session --n-test-sessions ${N_TEST_SESSIONS}"

# Parse arguments
MODE=${1:-"full"}

if [ "$MODE" == "smoke" ]; then
    echo "=== SMOKE TEST MODE ==="
    EPOCHS=2
    SEEDS=(42 123)  # Only 2 seeds for smoke test
fi

# Create results directory
mkdir -p ${RESULTS_DIR}

# Header for results file
echo "==============================================================================" | tee ${RESULTS_FILE}
echo "CEBRA vs Baseline Comparison - ${TIMESTAMP}" | tee -a ${RESULTS_FILE}
echo "Epochs: ${EPOCHS}, Test Sessions: ${N_TEST_SESSIONS}, GPUs: ${N_GPUS}" | tee -a ${RESULTS_FILE}
echo "Seeds: ${SEEDS[*]}" | tee -a ${RESULTS_FILE}
echo "==============================================================================" | tee -a ${RESULTS_FILE}

# =============================================================================
# SMOKE TEST - Quick validation that both configs work
# =============================================================================
run_smoke_test() {
    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "SMOKE TEST: Validating configurations (2 epochs each)" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}

    # Test CEBRA config
    echo "[SMOKE] Testing CEBRA configuration..." | tee -a ${RESULTS_FILE}
    torchrun --nproc_per_node=${N_GPUS} train.py ${COMMON_ARGS} \
        --epochs 2 --seed 42 --use-contrastive \
        --force-recreate-splits \
        2>&1 | tee ${RESULTS_DIR}/smoke_cebra.log

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[SMOKE] CEBRA config FAILED!" | tee -a ${RESULTS_FILE}
        exit 1
    fi
    echo "[SMOKE] CEBRA config OK" | tee -a ${RESULTS_FILE}

    # Test baseline config
    echo "[SMOKE] Testing baseline configuration..." | tee -a ${RESULTS_FILE}
    torchrun --nproc_per_node=${N_GPUS} train.py ${COMMON_ARGS} \
        --epochs 2 --seed 42 --no-contrastive \
        2>&1 | tee ${RESULTS_DIR}/smoke_baseline.log

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[SMOKE] Baseline config FAILED!" | tee -a ${RESULTS_FILE}
        exit 1
    fi
    echo "[SMOKE] Baseline config OK" | tee -a ${RESULTS_FILE}

    echo ""
    echo "[SMOKE] All configurations validated successfully!" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
}

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================
run_cebra_experiments() {
    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "CEBRA EXPERIMENTS (with contrastive learning)" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}

    for i in "${!SEEDS[@]}"; do
        SEED=${SEEDS[$i]}
        RUN_NUM=$((i + 1))
        LOG_FILE="${RESULTS_DIR}/cebra_run${RUN_NUM}_seed${SEED}.log"

        echo ""
        echo "[CEBRA ${RUN_NUM}/${#SEEDS[@]}] Seed=${SEED}, Epochs=${EPOCHS}" | tee -a ${RESULTS_FILE}
        echo "  Log: ${LOG_FILE}"

        torchrun --nproc_per_node=${N_GPUS} train.py ${COMMON_ARGS} \
            --epochs ${EPOCHS} --seed ${SEED} --use-contrastive \
            2>&1 | tee ${LOG_FILE}

        # Extract results from log
        CORR=$(grep "RESULT_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2)
        R2=$(grep "RESULT_R2=" ${LOG_FILE} | tail -1 | cut -d'=' -f2)
        AVG_CORR=$(grep "RESULT_PER_SESSION_AVG_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")

        echo "  [CEBRA ${RUN_NUM}] Corr=${CORR}, R2=${R2}, PerSessionAvg=${AVG_CORR}" | tee -a ${RESULTS_FILE}
    done
}

run_baseline_experiments() {
    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "BASELINE EXPERIMENTS (without contrastive learning)" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}

    for i in "${!SEEDS[@]}"; do
        SEED=${SEEDS[$i]}
        RUN_NUM=$((i + 1))
        LOG_FILE="${RESULTS_DIR}/baseline_run${RUN_NUM}_seed${SEED}.log"

        echo ""
        echo "[BASELINE ${RUN_NUM}/${#SEEDS[@]}] Seed=${SEED}, Epochs=${EPOCHS}" | tee -a ${RESULTS_FILE}
        echo "  Log: ${LOG_FILE}"

        torchrun --nproc_per_node=${N_GPUS} train.py ${COMMON_ARGS} \
            --epochs ${EPOCHS} --seed ${SEED} --no-contrastive \
            2>&1 | tee ${LOG_FILE}

        # Extract results from log
        CORR=$(grep "RESULT_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2)
        R2=$(grep "RESULT_R2=" ${LOG_FILE} | tail -1 | cut -d'=' -f2)
        AVG_CORR=$(grep "RESULT_PER_SESSION_AVG_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")

        echo "  [BASELINE ${RUN_NUM}] Corr=${CORR}, R2=${R2}, PerSessionAvg=${AVG_CORR}" | tee -a ${RESULTS_FILE}
    done
}

# =============================================================================
# SUMMARY
# =============================================================================
print_summary() {
    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "SUMMARY" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}

    echo ""
    echo "CEBRA Results:" | tee -a ${RESULTS_FILE}
    echo "  Run | Seed | Corr   | R²     | PerSession" | tee -a ${RESULTS_FILE}
    echo "  ----|------|--------|--------|------------" | tee -a ${RESULTS_FILE}

    for i in "${!SEEDS[@]}"; do
        SEED=${SEEDS[$i]}
        RUN_NUM=$((i + 1))
        LOG_FILE="${RESULTS_DIR}/cebra_run${RUN_NUM}_seed${SEED}.log"
        if [ -f "${LOG_FILE}" ]; then
            CORR=$(grep "RESULT_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
            R2=$(grep "RESULT_R2=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
            AVG=$(grep "RESULT_PER_SESSION_AVG_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
            printf "  %3d | %4d | %6s | %6s | %s\n" ${RUN_NUM} ${SEED} ${CORR} ${R2} ${AVG} | tee -a ${RESULTS_FILE}
        fi
    done

    echo ""
    echo "BASELINE Results:" | tee -a ${RESULTS_FILE}
    echo "  Run | Seed | Corr   | R²     | PerSession" | tee -a ${RESULTS_FILE}
    echo "  ----|------|--------|--------|------------" | tee -a ${RESULTS_FILE}

    for i in "${!SEEDS[@]}"; do
        SEED=${SEEDS[$i]}
        RUN_NUM=$((i + 1))
        LOG_FILE="${RESULTS_DIR}/baseline_run${RUN_NUM}_seed${SEED}.log"
        if [ -f "${LOG_FILE}" ]; then
            CORR=$(grep "RESULT_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
            R2=$(grep "RESULT_R2=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
            AVG=$(grep "RESULT_PER_SESSION_AVG_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
            printf "  %3d | %4d | %6s | %6s | %s\n" ${RUN_NUM} ${SEED} ${CORR} ${R2} ${AVG} | tee -a ${RESULTS_FILE}
        fi
    done

    echo ""
    echo "Results saved to: ${RESULTS_FILE}" | tee -a ${RESULTS_FILE}
    echo "Logs saved to: ${RESULTS_DIR}/" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
}

# =============================================================================
# MAIN
# =============================================================================
echo "Starting CEBRA vs Baseline comparison experiment..."
echo "Mode: ${MODE}"
echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

case ${MODE} in
    "smoke")
        run_smoke_test
        ;;
    "cebra")
        run_smoke_test
        run_cebra_experiments
        print_summary
        ;;
    "baseline")
        run_smoke_test
        run_baseline_experiments
        print_summary
        ;;
    "full")
        run_smoke_test
        run_cebra_experiments
        run_baseline_experiments
        print_summary
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Usage: $0 [smoke|cebra|baseline|full]"
        exit 1
        ;;
esac

echo ""
echo "Experiment complete!"
