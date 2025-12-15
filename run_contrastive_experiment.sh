#!/bin/bash
# =============================================================================
# Label-Supervised Contrastive Learning Experiment
# =============================================================================
# Tests different contrastive learning parameters for cross-session generalization
#
# Setup:
#   - 1 baseline run (no contrastive learning)
#   - 9 runs with label-supervised contrastive learning (different parameters)
#   - All with 4 test sessions, fixed seed, 80 epochs
#
# Parameters varied:
#   - contrastive_weight: 0.05, 0.1, 0.2
#   - contrastive_temperature: 0.05, 0.1, 0.2
#
# Usage:
#   ./run_contrastive_experiment.sh          # Full experiment (10 runs x 80 epochs)
#   ./run_contrastive_experiment.sh smoke    # Quick smoke test (2 epochs)
# =============================================================================

set -e  # Exit on error

# Configuration
EPOCHS=80
N_TEST_SESSIONS=4
N_GPUS=8
SEED=42  # FIXED SEED - no variation
RESULTS_DIR="artifacts/contrastive_experiment"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/results_${TIMESTAMP}.txt"

# Contrastive parameter grid
WEIGHTS=(0.05 0.1 0.2)
TEMPS=(0.05 0.1 0.2)

# Common training arguments
COMMON_ARGS="--fsdp --fsdp-strategy grad_op \
  --per-channel-norm --output-scaling \
  --split-by-session --n-test-sessions ${N_TEST_SESSIONS} \
  --seed ${SEED}"

# Parse arguments
MODE=${1:-"full"}

if [ "$MODE" == "smoke" ]; then
    echo "=== SMOKE TEST MODE ==="
    EPOCHS=2
fi

# Create results directory
mkdir -p ${RESULTS_DIR}

# Header for results file
echo "==============================================================================" | tee ${RESULTS_FILE}
echo "Label-Supervised Contrastive Learning Experiment - ${TIMESTAMP}" | tee -a ${RESULTS_FILE}
echo "Epochs: ${EPOCHS}, Test Sessions: ${N_TEST_SESSIONS}, GPUs: ${N_GPUS}, Seed: ${SEED}" | tee -a ${RESULTS_FILE}
echo "==============================================================================" | tee -a ${RESULTS_FILE}

# =============================================================================
# SMOKE TEST
# =============================================================================
run_smoke_test() {
    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "SMOKE TEST: Validating configurations (2 epochs each)" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}

    # Test baseline config
    echo "[SMOKE] Testing baseline (no contrastive)..." | tee -a ${RESULTS_FILE}
    torchrun --nproc_per_node=${N_GPUS} train.py ${COMMON_ARGS} \
        --epochs 2 --no-contrastive \
        --force-recreate-splits \
        2>&1 | tee ${RESULTS_DIR}/smoke_baseline.log

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[SMOKE] Baseline config FAILED!" | tee -a ${RESULTS_FILE}
        exit 1
    fi
    echo "[SMOKE] Baseline OK" | tee -a ${RESULTS_FILE}

    # Test contrastive config
    echo "[SMOKE] Testing contrastive (weight=0.1, temp=0.1)..." | tee -a ${RESULTS_FILE}
    torchrun --nproc_per_node=${N_GPUS} train.py ${COMMON_ARGS} \
        --epochs 2 --use-contrastive \
        --contrastive-weight 0.1 --contrastive-temperature 0.1 \
        2>&1 | tee ${RESULTS_DIR}/smoke_contrastive.log

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[SMOKE] Contrastive config FAILED!" | tee -a ${RESULTS_FILE}
        exit 1
    fi
    echo "[SMOKE] Contrastive OK" | tee -a ${RESULTS_FILE}

    echo ""
    echo "[SMOKE] All configurations validated!" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
}

# =============================================================================
# BASELINE RUN
# =============================================================================
run_baseline() {
    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "BASELINE (no contrastive learning)" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}

    LOG_FILE="${RESULTS_DIR}/baseline.log"
    echo "[BASELINE] Epochs=${EPOCHS}, Seed=${SEED}" | tee -a ${RESULTS_FILE}
    echo "  Log: ${LOG_FILE}"

    torchrun --nproc_per_node=${N_GPUS} train.py ${COMMON_ARGS} \
        --epochs ${EPOCHS} --no-contrastive \
        --force-recreate-splits \
        2>&1 | tee ${LOG_FILE}

    # Extract results
    CORR=$(grep "RESULT_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2)
    R2=$(grep "RESULT_R2=" ${LOG_FILE} | tail -1 | cut -d'=' -f2)
    AVG_CORR=$(grep "RESULT_PER_SESSION_AVG_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")

    echo "[BASELINE] Corr=${CORR}, R2=${R2}, PerSessionAvg=${AVG_CORR}" | tee -a ${RESULTS_FILE}
}

# =============================================================================
# CONTRASTIVE RUNS (9 parameter combinations)
# =============================================================================
run_contrastive_experiments() {
    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "LABEL-SUPERVISED CONTRASTIVE LEARNING (9 parameter combinations)" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}

    RUN_NUM=0
    for WEIGHT in "${WEIGHTS[@]}"; do
        for TEMP in "${TEMPS[@]}"; do
            RUN_NUM=$((RUN_NUM + 1))
            LOG_FILE="${RESULTS_DIR}/contrastive_w${WEIGHT}_t${TEMP}.log"

            echo ""
            echo "[CONTRASTIVE ${RUN_NUM}/9] Weight=${WEIGHT}, Temp=${TEMP}" | tee -a ${RESULTS_FILE}
            echo "  Log: ${LOG_FILE}"

            torchrun --nproc_per_node=${N_GPUS} train.py ${COMMON_ARGS} \
                --epochs ${EPOCHS} --use-contrastive \
                --contrastive-weight ${WEIGHT} \
                --contrastive-temperature ${TEMP} \
                2>&1 | tee ${LOG_FILE}

            # Extract results
            CORR=$(grep "RESULT_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2)
            R2=$(grep "RESULT_R2=" ${LOG_FILE} | tail -1 | cut -d'=' -f2)
            AVG_CORR=$(grep "RESULT_PER_SESSION_AVG_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")

            echo "  [CONTRASTIVE ${RUN_NUM}] Corr=${CORR}, R2=${R2}, PerSessionAvg=${AVG_CORR}" | tee -a ${RESULTS_FILE}
        done
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
    echo "BASELINE (no contrastive):" | tee -a ${RESULTS_FILE}
    LOG_FILE="${RESULTS_DIR}/baseline.log"
    if [ -f "${LOG_FILE}" ]; then
        CORR=$(grep "RESULT_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
        R2=$(grep "RESULT_R2=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
        AVG=$(grep "RESULT_PER_SESSION_AVG_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
        echo "  Corr=${CORR}, R2=${R2}, PerSessionAvg=${AVG}" | tee -a ${RESULTS_FILE}
    fi

    echo ""
    echo "LABEL-SUPERVISED CONTRASTIVE:" | tee -a ${RESULTS_FILE}
    echo "  Weight | Temp  | Corr   | R²     | PerSession | Δ vs Baseline" | tee -a ${RESULTS_FILE}
    echo "  -------|-------|--------|--------|------------|---------------" | tee -a ${RESULTS_FILE}

    BASELINE_CORR=$(grep "RESULT_CORR=" ${RESULTS_DIR}/baseline.log 2>/dev/null | tail -1 | cut -d'=' -f2 || echo "0")

    for WEIGHT in "${WEIGHTS[@]}"; do
        for TEMP in "${TEMPS[@]}"; do
            LOG_FILE="${RESULTS_DIR}/contrastive_w${WEIGHT}_t${TEMP}.log"
            if [ -f "${LOG_FILE}" ]; then
                CORR=$(grep "RESULT_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
                R2=$(grep "RESULT_R2=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
                AVG=$(grep "RESULT_PER_SESSION_AVG_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")

                # Calculate delta vs baseline
                if [ "${CORR}" != "N/A" ] && [ "${BASELINE_CORR}" != "0" ]; then
                    DELTA=$(echo "${CORR} - ${BASELINE_CORR}" | bc -l 2>/dev/null | xargs printf "%+.4f" 2>/dev/null || echo "N/A")
                else
                    DELTA="N/A"
                fi

                printf "  %6s | %5s | %6s | %6s | %10s | %s\n" ${WEIGHT} ${TEMP} ${CORR} ${R2} ${AVG} ${DELTA} | tee -a ${RESULTS_FILE}
            fi
        done
    done

    echo ""
    echo "Results saved to: ${RESULTS_FILE}" | tee -a ${RESULTS_FILE}
    echo "Logs saved to: ${RESULTS_DIR}/" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
}

# =============================================================================
# MAIN
# =============================================================================
echo "Starting Label-Supervised Contrastive Learning Experiment..."
echo "Mode: ${MODE}"
echo "Seed: ${SEED} (fixed)"
echo "Results: ${RESULTS_DIR}"
echo ""

case ${MODE} in
    "smoke")
        run_smoke_test
        ;;
    "full")
        run_smoke_test
        run_baseline
        run_contrastive_experiments
        print_summary
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Usage: $0 [smoke|full]"
        exit 1
        ;;
esac

echo ""
echo "Experiment complete!"
