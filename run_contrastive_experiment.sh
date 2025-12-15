#!/bin/bash
# =============================================================================
# Label-Supervised Contrastive Learning Experiment (Binary Search Approach)
# =============================================================================
# Phase 1: Test 4 extreme corners + baseline (5 runs)
# Phase 2: Based on best extreme, narrow down with 5 more runs
#
# Extreme corners:
#   - (weight=0.01, temp=0.01) - minimal contrastive, sharp
#   - (weight=0.01, temp=1.0)  - minimal contrastive, soft
#   - (weight=1.0,  temp=0.01) - strong contrastive, sharp
#   - (weight=1.0,  temp=1.0)  - strong contrastive, soft
#
# Usage:
#   ./run_contrastive_experiment.sh smoke   # Quick smoke test (2 epochs)
#   ./run_contrastive_experiment.sh phase1  # Extreme corners only (5 runs)
#   ./run_contrastive_experiment.sh         # Full experiment (auto phase 2)
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
# PHASE 1: Baseline + 4 Extreme Corners
# =============================================================================
run_phase1() {
    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "PHASE 1: BASELINE + 4 EXTREME CORNERS" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}

    # Baseline
    run_experiment "baseline" "--no-contrastive --force-recreate-splits"

    # 4 Extreme corners
    run_experiment "extreme_w0.01_t0.01" "--use-contrastive --contrastive-weight 0.01 --contrastive-temperature 0.01"
    run_experiment "extreme_w0.01_t1.0" "--use-contrastive --contrastive-weight 0.01 --contrastive-temperature 1.0"
    run_experiment "extreme_w1.0_t0.01" "--use-contrastive --contrastive-weight 1.0 --contrastive-temperature 0.01"
    run_experiment "extreme_w1.0_t1.0" "--use-contrastive --contrastive-weight 1.0 --contrastive-temperature 1.0"

    print_phase1_summary
}

print_phase1_summary() {
    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "PHASE 1 SUMMARY" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo ""
    echo "  Name                  | Weight | Temp | Corr   | R²     | PerSession" | tee -a ${RESULTS_FILE}
    echo "  ----------------------|--------|------|--------|--------|------------" | tee -a ${RESULTS_FILE}

    for LOG in baseline extreme_w0.01_t0.01 extreme_w0.01_t1.0 extreme_w1.0_t0.01 extreme_w1.0_t1.0; do
        LOG_FILE="${RESULTS_DIR}/${LOG}.log"
        if [ -f "${LOG_FILE}" ]; then
            CORR=$(grep "RESULT_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
            R2=$(grep "RESULT_R2=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")
            AVG=$(grep "RESULT_PER_SESSION_AVG_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "N/A")

            case ${LOG} in
                baseline) W="-"; T="-" ;;
                extreme_w0.01_t0.01) W="0.01"; T="0.01" ;;
                extreme_w0.01_t1.0) W="0.01"; T="1.0" ;;
                extreme_w1.0_t0.01) W="1.0"; T="0.01" ;;
                extreme_w1.0_t1.0) W="1.0"; T="1.0" ;;
            esac

            printf "  %-21s | %6s | %4s | %6s | %6s | %s\n" ${LOG} ${W} ${T} ${CORR} ${R2} ${AVG} | tee -a ${RESULTS_FILE}
        fi
    done

    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "Based on these results, Phase 2 will explore the best region." | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
}

# =============================================================================
# PHASE 2: Narrow down based on Phase 1 results
# =============================================================================
run_phase2() {
    echo ""
    echo "==============================================================================" | tee -a ${RESULTS_FILE}
    echo "PHASE 2: NARROWING DOWN (5 additional runs)" | tee -a ${RESULTS_FILE}
    echo "==============================================================================" | tee -a ${RESULTS_FILE}

    # Find best extreme from Phase 1
    BEST_CORR=0
    BEST_W=0.1
    BEST_T=0.1

    for LOG in extreme_w0.01_t0.01 extreme_w0.01_t1.0 extreme_w1.0_t0.01 extreme_w1.0_t1.0; do
        LOG_FILE="${RESULTS_DIR}/${LOG}.log"
        if [ -f "${LOG_FILE}" ]; then
            CORR=$(grep "RESULT_CORR=" ${LOG_FILE} | tail -1 | cut -d'=' -f2 || echo "0")

            # Compare (using bc for float comparison)
            IS_BETTER=$(echo "${CORR} > ${BEST_CORR}" | bc -l 2>/dev/null || echo "0")
            if [ "${IS_BETTER}" == "1" ]; then
                BEST_CORR=${CORR}
                case ${LOG} in
                    extreme_w0.01_t0.01) BEST_W=0.01; BEST_T=0.01 ;;
                    extreme_w0.01_t1.0) BEST_W=0.01; BEST_T=1.0 ;;
                    extreme_w1.0_t0.01) BEST_W=1.0; BEST_T=0.01 ;;
                    extreme_w1.0_t1.0) BEST_W=1.0; BEST_T=1.0 ;;
                esac
            fi
        fi
    done

    echo ""
    echo "Best extreme: weight=${BEST_W}, temp=${BEST_T}, corr=${BEST_CORR}" | tee -a ${RESULTS_FILE}
    echo "Exploring region around best extreme..." | tee -a ${RESULTS_FILE}

    # Generate 5 points around the best extreme (binary search style)
    # Midpoints between best and center (0.5, 0.5)
    if [ "${BEST_W}" == "0.01" ]; then
        W_MID=0.05
        W_QUARTER=0.25
    else
        W_MID=0.5
        W_QUARTER=0.75
    fi

    if [ "${BEST_T}" == "0.01" ]; then
        T_MID=0.05
        T_QUARTER=0.25
    else
        T_MID=0.5
        T_QUARTER=0.75
    fi

    # Run 5 experiments around the best region
    run_experiment "phase2_w${BEST_W}_t${T_MID}" "--use-contrastive --contrastive-weight ${BEST_W} --contrastive-temperature ${T_MID}"
    run_experiment "phase2_w${W_MID}_t${BEST_T}" "--use-contrastive --contrastive-weight ${W_MID} --contrastive-temperature ${BEST_T}"
    run_experiment "phase2_w${W_MID}_t${T_MID}" "--use-contrastive --contrastive-weight ${W_MID} --contrastive-temperature ${T_MID}"
    run_experiment "phase2_w${W_QUARTER}_t${T_QUARTER}" "--use-contrastive --contrastive-weight ${W_QUARTER} --contrastive-temperature ${T_QUARTER}"
    run_experiment "phase2_w0.1_t0.1" "--use-contrastive --contrastive-weight 0.1 --contrastive-temperature 0.1"

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
echo "Label-Supervised Contrastive Experiment (Binary Search)"
echo "Mode: ${MODE}, Seed: ${SEED}, Epochs: ${EPOCHS}"
echo "Results: ${RESULTS_DIR}"
echo ""

case ${MODE} in
    "smoke")
        run_smoke_test
        ;;
    "phase1")
        run_smoke_test
        run_phase1
        ;;
    "full")
        run_smoke_test
        run_phase1
        run_phase2
        ;;
    *)
        echo "Usage: $0 [smoke|phase1|full]"
        exit 1
        ;;
esac

echo ""
echo "Experiment complete!"
