#!/usr/bin/env bash
# Launch ECoG LOSO cross-validation for all region pathways.
# Each pathway gets its own GPU, log file, and output directory.
#
# Usage:
#   bash scripts/launch_ecog_loso.sh
#
# To check progress:
#   tail -f results/ECoG/motor_imagery/frontal_to_temporal/run.log
#
# To kill all runners:
#   pkill -f "LOSO.runner.*ecog"

set -euo pipefail

BASE="results/ECoG"
mkdir -p "$BASE"

launch() {
    local gpu=$1
    local experiment=$2
    local src=$3
    local tgt=$4
    local outdir="$BASE/$experiment/${src}_to_${tgt}"

    mkdir -p "$outdir"
    echo "[GPU $gpu] $experiment: $src -> $tgt  (log: $outdir/run.log)"

    CUDA_VISIBLE_DEVICES=$gpu nohup python -m LOSO.runner \
        --dataset ecog \
        --ecog-experiment "$experiment" \
        --ecog-source-region "$src" \
        --ecog-target-region "$tgt" \
        --output-dir "$outdir" \
        > "$outdir/run.log" 2>&1 &
}

echo "=== Launching ECoG LOSO runs ==="
echo ""

# Motor imagery: 6 pathways (GPUs 0-5)
launch 0 motor_imagery frontal  temporal
launch 1 motor_imagery frontal  parietal
launch 2 motor_imagery temporal frontal
launch 3 motor_imagery temporal parietal
launch 4 motor_imagery parietal frontal
launch 5 motor_imagery parietal temporal

# Faceshouses: 2 pathways (GPUs 6-7)
launch 6 faceshouses frontal  temporal
launch 7 faceshouses temporal frontal

echo ""
echo "=== All 8 runners launched ==="
echo "Monitor with: tail -f $BASE/*/run.log"
echo "Kill all:     pkill -f 'LOSO.runner.*ecog'"
