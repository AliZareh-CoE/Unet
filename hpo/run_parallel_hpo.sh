#!/bin/bash
# =============================================================================
# PARALLEL HPO LAUNCHER FOR 8×A100 (Nature Methods Edition)
# =============================================================================
# Run 8 independent trials simultaneously (one per GPU)
# This gives 8× speedup over sequential trials!
#
# IMPORTANT: Always run --dry-run first before starting full sweep!
#
# Usage:
#   bash run_parallel_hpo.sh --dry-run              # Test first!
#   bash run_parallel_hpo.sh                        # Full sweep (256 trials)
#   bash run_parallel_hpo.sh --export               # Export results only
#   bash run_parallel_hpo.sh --validate-top 5       # Validate top 5 with train.py
#
# Options:
#   --dry-run         Test single trial before full sweep (ALWAYS RUN FIRST!)
#   --trials N        Number of trials per GPU (default: 32)
#   --epochs N        Max epochs per trial (default: 80)
#   --phase N         HPO phase: 1=explore, 2=focused (default: 1)
#   --export          Export results after completion
#   --validate-top N  Validate top N trials with train.py after HPO
#   --no-checkpoints  Disable checkpoint saving
#   --help            Show this help message

set -e

# =============================================================================
# Parse arguments
# =============================================================================
TRIALS_PER_GPU=32
MAX_EPOCHS=80
PHASE=1
DRY_RUN=false
EXPORT_ONLY=false
VALIDATE_TOP=0
SAVE_CHECKPOINTS=true

show_help() {
    head -30 "$0" | tail -25
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --trials)
            TRIALS_PER_GPU="$2"
            shift 2
            ;;
        --epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --export)
            EXPORT_ONLY=true
            shift
            ;;
        --validate-top)
            VALIDATE_TOP="$2"
            shift 2
            ;;
        --no-checkpoints)
            SAVE_CHECKPOINTS=false
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Configuration
STUDY_NAME="olfactory_hpo_phase${PHASE}"
STUDY_DIR="artifacts/optuna_parallel"
DB_PATH="${STUDY_DIR}/${STUDY_NAME}.db"
LOG_DIR="${STUDY_DIR}/logs_phase${PHASE}"
RESULTS_DIR="${STUDY_DIR}/results"
CHECKPOINT_DIR="${STUDY_DIR}/checkpoints"

# Create directories
mkdir -p "$STUDY_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$CHECKPOINT_DIR"

# =============================================================================
# Header
# =============================================================================
echo "============================================================"
echo "   PARALLEL HPO LAUNCHER (Nature Methods Edition)"
echo "============================================================"
echo "Phase: $PHASE"
echo "Study: $STUDY_NAME"
echo "Database: $DB_PATH"
echo ""

# =============================================================================
# Dry-run mode
# =============================================================================
if [ "$DRY_RUN" = true ]; then
    echo "Running PRE-FLIGHT TEST..."
    echo "============================================================"
    python3 optuna_hpo_parallel.py --dry-run --gpu-id 0 --seed 42

    if [ $? -eq 0 ]; then
        echo ""
        echo "============================================================"
        echo "   DRY-RUN PASSED! Ready for full sweep."
        echo "============================================================"
        echo ""
        echo "To start full HPO sweep, run:"
        echo "  bash run_parallel_hpo.sh --trials $TRIALS_PER_GPU --epochs $MAX_EPOCHS"
        echo ""
        echo "Estimated time: ~4 hours (256 trials across 8 GPUs)"
    else
        echo ""
        echo "============================================================"
        echo "   DRY-RUN FAILED! Fix issues before running full sweep."
        echo "============================================================"
        exit 1
    fi
    exit 0
fi

# =============================================================================
# Export-only mode
# =============================================================================
if [ "$EXPORT_ONLY" = true ]; then
    echo "Exporting results..."
    echo "============================================================"
    python3 optuna_hpo_parallel.py --export \
        --study-name "$STUDY_NAME" \
        --storage "sqlite:///${DB_PATH}" \
        --export-dir "$RESULTS_DIR"
    echo ""
    echo "Results exported to: $RESULTS_DIR"
    exit 0
fi

# =============================================================================
# Normal HPO mode
# =============================================================================
echo "Trials per GPU: $TRIALS_PER_GPU"
echo "Total trials: $((TRIALS_PER_GPU * 8))"
echo "Max epochs: $MAX_EPOCHS"
echo "Save checkpoints: $SAVE_CHECKPOINTS"
echo "============================================================"

# Create study with ASHA pruner (run once)
echo ""
echo "Creating/loading Optuna study..."
python3 << EOF
import optuna
import os

storage = "sqlite:///${DB_PATH}"
study_name = "${STUDY_NAME}"

try:
    study = optuna.load_study(study_name=study_name, storage=storage)
    print(f"Loaded existing study with {len(study.trials)} trials")
except:
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=5,
            max_resource=${MAX_EPOCHS},
            reduction_factor=3,
        ),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=20,
            multivariate=True,
            group=True,
        ),
    )
    print("Created new study with ASHA pruner")
EOF

echo ""
echo "Launching 8 parallel workers..."
echo ""

# Build checkpoint flag
CHECKPOINT_FLAG=""
if [ "$SAVE_CHECKPOINTS" = false ]; then
    CHECKPOINT_FLAG="--no-checkpoints"
fi

# Launch workers on all 8 GPUs
PIDS=()
for GPU_ID in {0..7}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 optuna_hpo_parallel.py \
        --study-name "$STUDY_NAME" \
        --storage "sqlite:///${DB_PATH}" \
        --n-trials $TRIALS_PER_GPU \
        --epochs $MAX_EPOCHS \
        --early-stop 10 \
        --gpu-id 0 \
        $CHECKPOINT_FLAG \
        > "${LOG_DIR}/gpu_${GPU_ID}.log" 2>&1 &

    PIDS+=($!)
    echo "  GPU $GPU_ID: PID ${PIDS[-1]}"
done

echo ""
echo "All workers launched!"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/gpu_*.log"
echo ""
echo "View study dashboard:"
echo "  optuna-dashboard sqlite:///${DB_PATH}"
echo ""

# Wait for all workers
echo "Waiting for all workers to complete..."
echo "(This will take approximately 4 hours)"
echo ""

for PID in ${PIDS[@]}; do
    wait $PID
done

echo ""
echo "============================================================"
echo "   ALL TRIALS COMPLETE!"
echo "============================================================"

# Print summary
python3 << EOF
import optuna
import json

storage = "sqlite:///${DB_PATH}"
study = optuna.load_study(study_name="${STUDY_NAME}", storage=storage)

print(f"\nStudy Summary:")
print(f"  Total trials: {len(study.trials)}")
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
print(f"  Completed: {len(completed)}")
print(f"  Pruned: {len(pruned)}")
print(f"  Failed: {len(failed)}")

if study.best_trial:
    print(f"\nBest Trial #{study.best_trial.number}")
    print(f"  Score: {study.best_trial.value:.4f}")
    print(f"  Correlation: {study.best_trial.user_attrs.get('best_corr', 'N/A')}")
    print(f"  R²: {study.best_trial.user_attrs.get('best_r2', 'N/A')}")
    print(f"  Training time: {study.best_trial.user_attrs.get('training_time_seconds', 'N/A')}s")
    print(f"\n  Parameters:")
    for k, v in study.best_trial.params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4e}")
        else:
            print(f"    {k}: {v}")

    # Save best config to JSON
    config_json = study.best_trial.user_attrs.get('full_config', '{}')
    with open("${RESULTS_DIR}/best_config.json", 'w') as f:
        f.write(config_json)
    print(f"\n  Best config saved to: ${RESULTS_DIR}/best_config.json")

# Top 5 trials
print("\nTop 5 Trials:")
trials = sorted(
    [t for t in study.trials if t.value is not None],
    key=lambda t: t.value,
    reverse=True
)[:5]
for i, t in enumerate(trials):
    corr = t.user_attrs.get('best_corr', 0)
    r2 = t.user_attrs.get('best_r2', 0)
    epochs = t.user_attrs.get('epochs_run', 0)
    time_s = t.user_attrs.get('training_time_seconds', 0)
    ckpt = "+" if t.user_attrs.get('checkpoint_path') else ""
    print(f"  {i+1}. Trial #{t.number}: score={t.value:.4f}, corr={corr:.4f}, r²={r2:.4f}, epochs={epochs}, time={time_s:.0f}s {ckpt}")

# Hyperparameter importance
try:
    importance = optuna.importance.get_param_importances(study)
    print("\nHyperparameter Importance:")
    for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {param}: {imp:.4f}")
except Exception as e:
    print(f"\nCould not compute importance: {e}")

EOF

# =============================================================================
# Auto-export results
# =============================================================================
echo ""
echo "============================================================"
echo "   EXPORTING RESULTS FOR NATURE METHODS"
echo "============================================================"

python3 optuna_hpo_parallel.py --export \
    --study-name "$STUDY_NAME" \
    --storage "sqlite:///${DB_PATH}" \
    --export-dir "$RESULTS_DIR"

echo ""
echo "Results exported to: $RESULTS_DIR"
echo "  - hpo_trials.csv"
echo "  - hpo_best_configs.json"
echo "  - hpo_training_curves.csv"
echo "  - hpo_importance.json"
echo "  - hpo_summary.json"
echo "  - best_config.json"

# =============================================================================
# Validate top trials (if requested)
# =============================================================================
if [ "$VALIDATE_TOP" -gt 0 ]; then
    echo ""
    echo "============================================================"
    echo "   VALIDATING TOP $VALIDATE_TOP TRIALS WITH train.py"
    echo "============================================================"
    echo ""
    echo "This will run full training for each trial to verify HPO results."
    echo "Estimated time: ~$((VALIDATE_TOP * 4)) hours"
    echo ""

    python3 validate_optuna_trial.py \
        --top $VALIDATE_TOP \
        --study-name "$STUDY_NAME" \
        --storage "sqlite:///${DB_PATH}" \
        --epochs $MAX_EPOCHS \
        --output "${RESULTS_DIR}/validation_report.json"

    echo ""
    echo "Validation complete! Report: ${RESULTS_DIR}/validation_report.json"
fi

# =============================================================================
# Final summary
# =============================================================================
echo ""
echo "============================================================"
echo "   HPO COMPLETE!"
echo "============================================================"
echo ""
echo "Files generated:"
echo "  Database:     ${DB_PATH}"
echo "  Logs:         ${LOG_DIR}/"
echo "  Results:      ${RESULTS_DIR}/"
echo "  Checkpoints:  ${CHECKPOINT_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Review results: optuna-dashboard sqlite:///${DB_PATH}"
echo "  2. Validate top configs: python validate_optuna_trial.py --top 5"
echo "  3. Train final model: python train.py --config ${RESULTS_DIR}/best_config.json"
echo ""
