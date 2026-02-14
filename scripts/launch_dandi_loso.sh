#!/usr/bin/env bash
# Launch DANDI LOSO cross-validation — all 6 region pathways, 1 GPU each.
# Regions: amygdala (AMY), hippocampus (HPC), medial_frontal_cortex (MFC)
# Dataset: DANDI 000623 human iEEG movie watching (18 subjects)

set -euo pipefail

BASE="results/DANDI"
mkdir -p "$BASE"

launch() {
    local gpu=$1
    local src=$2
    local tgt=$3
    local outdir="$BASE/${src}_to_${tgt}"

    mkdir -p "$outdir"
    echo "[GPU $gpu] $src -> $tgt  (log: $outdir/run.log)"

    CUDA_VISIBLE_DEVICES=$gpu nohup python -m LOSO.runner \
        --dataset dandi_movie \
        --dandi-source-region "$src" \
        --dandi-target-region "$tgt" \
        --epochs 20 \
        --output-dir "$outdir" \
        > "$outdir/run.log" 2>&1 &
}

echo "=== Launching DANDI LOSO runs (18 subjects × 6 pathways) ==="

launch 0 amygdala              hippocampus
launch 1 amygdala              medial_frontal_cortex
launch 2 hippocampus           amygdala
launch 3 hippocampus           medial_frontal_cortex
launch 4 medial_frontal_cortex amygdala
launch 5 medial_frontal_cortex hippocampus

echo "=== All 6 pathway runners launched (GPUs 0-5) ==="
echo "Monitor with: tail -f $BASE/*/run.log"
echo "Kill all:     pkill -f 'LOSO.runner.*dandi'"
