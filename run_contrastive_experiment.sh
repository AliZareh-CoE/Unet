#!/bin/bash
# =============================================================================
# Label-Supervised Contrastive Learning Experiment (Grid Search)
# =============================================================================
# 12 runs total: 1 baseline + 9 grid points (3x3) + 2 additional points
#
# Grid points (weight x temperature):
#   weight:      0.01, 0.5, 1.0
#   temperature: 0.01, 0.5, 1.0
#
# Additional points:
#   - (0.1, 0.1)   - common default region
#   - (0.25, 0.25) - quarter point
#
# Usage:
#   ./run_contrastive_experiment.sh smoke   # Quick smoke test (2 epochs)
#   ./run_contrastive_experiment.sh         # Full 12-run experiment
# =============================================================================

set -e

# Configuration
EPOCHS=80
N_TEST_SESSIONS=4
N_GPUS=8
SEED=42
RESULTS_DIR="artifacts/contrastive_experiment"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/results_${TIMESTAMP}.txt"

# Common args
COMMON_ARGS="--fsdp --fsdp-strategy grad_op \
  --per-channel-norm --output-scaling \
  --split-by-session --n-test-sessions ${N_TEST_SESSIONS} \
  --seed ${SEED}"

MODE=${1:-"full"}

if [ "$MODE" == "smoke" ]; then
    echo "=== SMOKE TEST MODE ==="
    EPOCHS=2
fi

mkdir -p ${RESULTS_DIR}

# =============================================================================
# HELPER: Run single experiment
# =============================================================================
run_experiment() {
    local NAME=$1
    local EXTRA_ARGS=$2
    local LOG_FILE="${RESULTS_DIR}/${NAME}.log"

    echo ""
    echo "[${NAME}] Starting..." | tee -a ${RESULTS_FILE}
    echo "  Log: ${LOG_FILE}"

    torchrun --nproc_per_node=${N_GPUS} train.py ${COMMON_ARGS} \
        --epochs ${EPOCHS} ${EXTRA_ARGS} \
        2>&1 | tee ${LOG_FILE}

    # Extract results
    local CORR=$(grep "RESULT_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
    local R2=$(grep "RESULT_R2=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
    local AVG=$(grep "RESULT_PER_SESSION_AVG_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")

    echo "[${NAME}] Corr=${CORR}, R2=${R2}, PerSessionAvg=${AVG}" | tee -a ${RESULTS_FILE}

    # Return correlation for comparison
    echo ${CORR}
}

# =============================================================================
# SMOKE TEST
# =============================================================================
run_smoke_test() {
    echo "==============================================================================" | tee ${RESULTS_FILE}
    echo "SMOKE TEST - Validating configs" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}

    echo "[SMOKE] Testing baseline..." | tee -a ${RESULTS_FILE}
    torchrun --nproc_per_node=${N_GPUS} train.py ${COMMON_ARGS} \
        --epochs 2 --no-contrastive --force-recreate-splits \
        2>&1 | tee ${RESULTS_DIR}/smoke_baseline.log
    echo "[SMOKE] Baseline OK" | tee -a ${RESULTS_FILE}

    echo "[SMOKE] Testing contrastive (extreme: w=1.0, t=0.01)..." | tee -a ${RESULTS_FILE}
    torchrun --nproc_per_node=${N_GPUS} train.py ${COMMON_ARGS} \
        --epochs 2 --use-contrastive \
        --contrastive-weight 1.0 --contrastive-temperature 0.01 \
        2>&1 | tee ${RESULTS_DIR}/smoke_contrastive.log
    echo "[SMOKE] Contrastive OK" | tee -a ${RESULTS_FILE}

    echo ""
    echo "[SMOKE] All configs validated!" | tee -a ${RESULTS_FILE}
}

# =============================================================================
# FULL EXPERIMENT: Baseline + 3x3 Grid + 2 Additional Points (12 runs)
# =============================================================================
run_full_experiment() {
    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "FULL EXPERIMENT: 12 RUNS (1 baseline + 9 grid + 2 extra)" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}

    # 1. Baseline
    run_experiment "baseline" "--no-contrastive --force-recreate-splits"

    # 2-10. 3x3 Grid: weight in {0.01, 0.5, 1.0} x temp in {0.01, 0.5, 1.0}
    for W in 0.01 0.5 1.0; do
        for T in 0.01 0.5 1.0; do
            run_experiment "grid_w${W}_t${T}" "--use-contrastive --contrastive-weight ${W} --contrastive-temperature ${T}"
        done
    done

    # 11-12. Additional points
    run_experiment "extra_w0.1_t0.1" "--use-contrastive --contrastive-weight 0.1 --contrastive-temperature 0.1"
    run_experiment "extra_w0.25_t0.25" "--use-contrastive --contrastive-weight 0.25 --contrastive-temperature 0.25"

    print_final_summary
}

print_final_summary() {
    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "FINAL SUMMARY - ALL RUNS" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}

    # Get baseline for delta calculation
    BASELINE_CORR=$(grep "RESULT_CORR=" ${RESULTS_DIR}/baseline.log 2>/dev/null | tail -1 | cut -d'=' -f2 || echo "0")

    echo ""
    echo "Baseline Corr: ${BASELINE_CORR}" | tee -a ${RESULTS_FILE}
    echo ""
    echo "  Experiment                   | Corr   | R²     | PerSession | Δ Baseline" | tee -a ${RESULTS_FILE}
    echo "  -----------------------------|--------|--------|------------|------------" | tee -a ${RESULTS_FILE}

    # List all log files and extract results
    for LOG_FILE in ${RESULTS_DIR}/*.log; do
        NAME=$(basename ${LOG_FILE} .log)

        # Skip smoke tests
        [[ ${NAME} == smoke* ]] && continue

        if [ -f "${LOG_FILE}" ]; then
            CORR=$(grep "RESULT_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
            R2=$(grep "RESULT_R2=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
            AVG=$(grep "RESULT_PER_SESSION_AVG_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")

            # Calculate delta
            if [ "${CORR}" != "N/A" ] && [ "${BASELINE_CORR}" != "0" ]; then
                DELTA=$(echo "${CORR} - ${BASELINE_CORR}" | bc -l 2>/dev/null | xargs printf "%+.4f" 2>/dev/null || echo "N/A")
            else
                DELTA="N/A"
            fi

            printf "  %-28s | %6s | %6s | %10s | %s\n" ${NAME} ${CORR} ${R2} ${AVG} ${DELTA} | tee -a ${RESULTS_FILE}
        fi
    done

    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "Results saved to: ${RESULTS_FILE}" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
}

# =============================================================================
# MAIN
# =============================================================================
echo "Label-Supervised Contrastive Experiment (Grid Search)"
echo "Mode: ${MODE}, Seed: ${SEED}, Epochs: ${EPOCHS}"
echo "Results: ${RESULTS_DIR}"
echo ""

case ${MODE} in
    "smoke")
        run_smoke_test
        ;;
    "full")
        run_smoke_test
        run_full_experiment
        ;;
    *)
        echo "Usage: $0 [smoke|full]"
        exit 1
        ;;
esac

echo ""
echo "Experiment complete!"
