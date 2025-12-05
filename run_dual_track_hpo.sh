#!/bin/bash
#
# Dual-Track HPO Launcher for Nature Methods
# ===========================================
#
# Runs two independent optimization tracks in parallel:
#   - Track A (Conditioning): Improves Stage 1 R² directly
#   - Track B (SpectralShift): Fixes PSD while preserving R²
#
# Usage:
#   # Full run (both phases)
#   ./run_dual_track_hpo.sh
#
#   # Phase 1 only (coarse screening)
#   ./run_dual_track_hpo.sh --phase 1
#
#   # Phase 2 only (deep dive on top 3)
#   ./run_dual_track_hpo.sh --phase 2
#
#   # Dry-run test
#   ./run_dual_track_hpo.sh --dry-run
#
#   # Export results only
#   ./run_dual_track_hpo.sh --export

set -e

# =============================================================================
# Configuration
# =============================================================================

# GPU allocation (adjust based on available hardware)
TRACK_A_GPUS="0,1,2,3"  # GPUs for Track A (Conditioning)
TRACK_B_GPUS="4,5,6,7"  # GPUs for Track B (SpectralShift)

# Stage 1 checkpoint for Track B (must exist before running)
STAGE1_CHECKPOINT="artifacts/checkpoints/best_stage1.pt"

# HPO settings
TRIALS_PHASE1=32        # Trials per approach in Phase 1
TRIALS_PHASE2=96        # Trials per approach in Phase 2
N_BASELINE=20           # Baseline trials per study
SEED=42                 # Base random seed

# Logging
LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# Parse Arguments
# =============================================================================

PHASE="both"
DRY_RUN=false
EXPORT_ONLY=false
TRACK_A_APPROACHES=""
TRACK_B_APPROACHES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --export)
            EXPORT_ONLY=true
            shift
            ;;
        --track-a-approaches)
            TRACK_A_APPROACHES="$2"
            shift 2
            ;;
        --track-b-approaches)
            TRACK_B_APPROACHES="$2"
            shift 2
            ;;
        --stage1-checkpoint)
            STAGE1_CHECKPOINT="$2"
            shift 2
            ;;
        --track-a-gpus)
            TRACK_A_GPUS="$2"
            shift 2
            ;;
        --track-b-gpus)
            TRACK_B_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_track_a() {
    local phase=$1
    local gpu=$2
    local approaches=$3

    log "Starting Track A (Phase $phase) on GPU $gpu"

    if [ -n "$approaches" ]; then
        CUDA_VISIBLE_DEVICES=$gpu python experiments/conditioning/run_track_a.py \
            --phase "$phase" \
            --approaches $approaches \
            --trials-per-approach "$TRIALS_PHASE2" \
            --n-baseline "$N_BASELINE" \
            --seed "$SEED" \
            2>&1 | tee "${LOG_DIR}/track_a_phase${phase}_gpu${gpu}_${TIMESTAMP}.log"
    else
        CUDA_VISIBLE_DEVICES=$gpu python experiments/conditioning/run_track_a.py \
            --phase "$phase" \
            --trials-per-approach "$TRIALS_PHASE1" \
            --n-baseline "$N_BASELINE" \
            --seed "$SEED" \
            2>&1 | tee "${LOG_DIR}/track_a_phase${phase}_gpu${gpu}_${TIMESTAMP}.log"
    fi
}

run_track_b() {
    local phase=$1
    local gpu=$2
    local approaches=$3

    log "Starting Track B (Phase $phase) on GPU $gpu"

    if [ -n "$approaches" ]; then
        CUDA_VISIBLE_DEVICES=$gpu python experiments/spectral/run_track_b.py \
            --phase "$phase" \
            --approaches $approaches \
            --trials-per-approach "$TRIALS_PHASE2" \
            --n-baseline "$N_BASELINE" \
            --stage1-checkpoint "$STAGE1_CHECKPOINT" \
            --seed "$SEED" \
            2>&1 | tee "${LOG_DIR}/track_b_phase${phase}_gpu${gpu}_${TIMESTAMP}.log"
    else
        CUDA_VISIBLE_DEVICES=$gpu python experiments/spectral/run_track_b.py \
            --phase "$phase" \
            --trials-per-approach "$TRIALS_PHASE1" \
            --n-baseline "$N_BASELINE" \
            --stage1-checkpoint "$STAGE1_CHECKPOINT" \
            --seed "$SEED" \
            2>&1 | tee "${LOG_DIR}/track_b_phase${phase}_gpu${gpu}_${TIMESTAMP}.log"
    fi
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

log "=============================================="
log "  Dual-Track HPO for Nature Methods"
log "=============================================="
log ""
log "Configuration:"
log "  Track A GPUs: $TRACK_A_GPUS"
log "  Track B GPUs: $TRACK_B_GPUS"
log "  Stage 1 Checkpoint: $STAGE1_CHECKPOINT"
log "  Phase: $PHASE"
log "  Trials Phase 1: $TRIALS_PHASE1"
log "  Trials Phase 2: $TRIALS_PHASE2"
log "  Baseline Trials: $N_BASELINE"
log ""

# Export mode
if [ "$EXPORT_ONLY" = true ]; then
    log "Export mode - exporting results only"
    python experiments/conditioning/run_track_a.py --export --phase 1
    python experiments/conditioning/run_track_a.py --export --phase 2
    python experiments/spectral/run_track_b.py --export --phase 1
    python experiments/spectral/run_track_b.py --export --phase 2
    log "Export complete!"
    exit 0
fi

# Dry-run mode
if [ "$DRY_RUN" = true ]; then
    log "Dry-run mode - testing both tracks"

    # Get first GPU from each group
    TRACK_A_GPU=$(echo $TRACK_A_GPUS | cut -d',' -f1)
    TRACK_B_GPU=$(echo $TRACK_B_GPUS | cut -d',' -f1)

    log "Testing Track A on GPU $TRACK_A_GPU..."
    CUDA_VISIBLE_DEVICES=$TRACK_A_GPU python experiments/conditioning/run_track_a.py \
        --dry-run --approach cpc

    log "Testing Track B on GPU $TRACK_B_GPU..."
    CUDA_VISIBLE_DEVICES=$TRACK_B_GPU python experiments/spectral/run_track_b.py \
        --dry-run --approach phase_preserving \
        --stage1-checkpoint "$STAGE1_CHECKPOINT"

    log "Dry-run complete! Both tracks are functional."
    exit 0
fi

# Check Stage 1 checkpoint exists for Track B
if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    log "ERROR: Stage 1 checkpoint not found: $STAGE1_CHECKPOINT"
    log "Run: python train.py --epochs 80 to create checkpoint first"
    exit 1
fi

# Check GPUs are available
if ! command -v nvidia-smi &> /dev/null; then
    log "WARNING: nvidia-smi not found, GPU availability not verified"
fi

# =============================================================================
# Main Execution
# =============================================================================

# Get GPU arrays
IFS=',' read -ra GPUS_A <<< "$TRACK_A_GPUS"
IFS=',' read -ra GPUS_B <<< "$TRACK_B_GPUS"

# Use first GPU from each group for primary work
PRIMARY_GPU_A=${GPUS_A[0]}
PRIMARY_GPU_B=${GPUS_B[0]}

# Phase 1: Coarse Screening
if [ "$PHASE" = "1" ] || [ "$PHASE" = "both" ]; then
    log "=============================================="
    log "  PHASE 1: Coarse Screening"
    log "=============================================="

    # Launch both tracks in parallel
    run_track_a 1 "$PRIMARY_GPU_A" "" &
    PID_A=$!

    run_track_b 1 "$PRIMARY_GPU_B" "" &
    PID_B=$!

    # Wait for both to complete
    log "Waiting for Phase 1 to complete..."
    wait $PID_A
    STATUS_A=$?
    wait $PID_B
    STATUS_B=$?

    if [ $STATUS_A -ne 0 ]; then
        log "ERROR: Track A Phase 1 failed with status $STATUS_A"
    fi
    if [ $STATUS_B -ne 0 ]; then
        log "ERROR: Track B Phase 1 failed with status $STATUS_B"
    fi

    log "Phase 1 complete!"

    # Export Phase 1 results
    log "Exporting Phase 1 results..."
    python experiments/conditioning/run_track_a.py --export --phase 1
    python experiments/spectral/run_track_b.py --export --phase 1
fi

# Phase 2: Deep Dive
if [ "$PHASE" = "2" ] || [ "$PHASE" = "both" ]; then
    log "=============================================="
    log "  PHASE 2: Deep Dive (Top 3 Approaches)"
    log "=============================================="

    # Default top 3 approaches (can be overridden via CLI)
    if [ -z "$TRACK_A_APPROACHES" ]; then
        TRACK_A_APPROACHES="cpc vqvae vib"
    fi
    if [ -z "$TRACK_B_APPROACHES" ]; then
        TRACK_B_APPROACHES="phase_preserving adaptive_gated wavelet"
    fi

    log "Track A approaches: $TRACK_A_APPROACHES"
    log "Track B approaches: $TRACK_B_APPROACHES"

    # Launch both tracks in parallel
    run_track_a 2 "$PRIMARY_GPU_A" "$TRACK_A_APPROACHES" &
    PID_A=$!

    run_track_b 2 "$PRIMARY_GPU_B" "$TRACK_B_APPROACHES" &
    PID_B=$!

    # Wait for both to complete
    log "Waiting for Phase 2 to complete..."
    wait $PID_A
    STATUS_A=$?
    wait $PID_B
    STATUS_B=$?

    if [ $STATUS_A -ne 0 ]; then
        log "ERROR: Track A Phase 2 failed with status $STATUS_A"
    fi
    if [ $STATUS_B -ne 0 ]; then
        log "ERROR: Track B Phase 2 failed with status $STATUS_B"
    fi

    log "Phase 2 complete!"

    # Export Phase 2 results
    log "Exporting Phase 2 results..."
    python experiments/conditioning/run_track_a.py --export --phase 2
    python experiments/spectral/run_track_b.py --export --phase 2
fi

# =============================================================================
# Final Analysis
# =============================================================================

log "=============================================="
log "  Running Final Analysis"
log "=============================================="

# Run analysis scripts if they exist
if [ -f "experiments/analysis/phase1_screening.py" ]; then
    log "Running Phase 1 screening analysis..."
    python experiments/analysis/phase1_screening.py
fi

if [ -f "experiments/analysis/phase2_deep_dive.py" ]; then
    log "Running Phase 2 deep dive analysis..."
    python experiments/analysis/phase2_deep_dive.py
fi

log "=============================================="
log "  HPO Complete!"
log "=============================================="
log ""
log "Results saved to:"
log "  - artifacts/optuna_conditioning/"
log "  - artifacts/optuna_spectral/"
log "  - artifacts/logs/"
log ""
log "To generate Nature Methods figures, run:"
log "  python experiments/analysis/nature_methods_plots.py"
