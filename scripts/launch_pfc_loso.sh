#!/usr/bin/env bash
# Launch PFC/HPC LOSO cross-validation in both directions.
# Uses --parallel-folds to spread LOSO folds across 4 GPUs per direction.
#
# Usage:
#   bash scripts/launch_pfc_loso.sh
#
# To check progress:
#   tail -f results/PFC_HPC/pfc_to_ca1/run.log
#   tail -f results/PFC_HPC/ca1_to_pfc/run.log
#
# To kill all runners:
#   pkill -f "LOSO.runner.*pfc_hpc"

set -euo pipefail

BASE="results/PFC_HPC"
mkdir -p "$BASE"

echo "=== Launching PFC/HPC LOSO runs (both directions) ==="
echo ""

# Direction 1: PFC -> CA1 (GPUs 0-3)
DIR1="$BASE/pfc_to_ca1"
mkdir -p "$DIR1"
echo "[GPUs 0-3] PFC -> CA1  (log: $DIR1/run.log)"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m LOSO.runner \
    --dataset pfc_hpc \
    --parallel-folds --n-gpus 4 \
    --output-dir "$DIR1" \
    > "$DIR1/run.log" 2>&1 &

# Direction 2: CA1 -> PFC (GPUs 4-7)
DIR2="$BASE/ca1_to_pfc"
mkdir -p "$DIR2"
echo "[GPUs 4-7] CA1 -> PFC  (log: $DIR2/run.log)"
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m LOSO.runner \
    --dataset pfc_hpc \
    --pfc-reverse \
    --parallel-folds --n-gpus 4 \
    --output-dir "$DIR2" \
    > "$DIR2/run.log" 2>&1 &

echo ""
echo "=== Both PFC/HPC runners launched ==="
echo "Monitor with: tail -f $BASE/*/run.log"
echo "Kill all:     pkill -f 'LOSO.runner.*pfc_hpc'"
