#!/bin/bash
# Channel Shuffle Experiment
# Compares model performance with and without channel shuffling
# to test if model relies on channel position correspondence

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="experiment/logs/channel_shuffle_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Channel Shuffle Experiment"
echo "========================================"
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Experiment 1: WITH channel shuffle
echo "[1/2] Running WITH channel shuffle (mode=both)..."
echo "Start time: $(date)"
torchrun --nproc_per_node=8 train.py \
    --dataset pcx1 \
    --fsdp \
    --epochs 60 \
    --batch-size 64 \
    --aug-channel-shuffle \
    --aug-channel-shuffle-mode both \
    2>&1 | tee "${LOG_DIR}/with_shuffle.log"

echo ""
echo "[1/2] Completed at: $(date)"
echo ""

# Experiment 2: WITHOUT channel shuffle (baseline)
echo "[2/2] Running WITHOUT channel shuffle (baseline)..."
echo "Start time: $(date)"
torchrun --nproc_per_node=8 train.py \
    --dataset pcx1 \
    --fsdp \
    --epochs 60 \
    --batch-size 64 \
    2>&1 | tee "${LOG_DIR}/no_shuffle_baseline.log"

echo ""
echo "[2/2] Completed at: $(date)"
echo ""

echo "========================================"
echo "Experiment Complete!"
echo "========================================"
echo "Logs saved to:"
echo "  - ${LOG_DIR}/with_shuffle.log"
echo "  - ${LOG_DIR}/no_shuffle_baseline.log"
echo ""
echo "Compare results:"
echo "  grep -E 'val.*corr|diagonal_dominance' ${LOG_DIR}/*.log"
