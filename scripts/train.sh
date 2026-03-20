#!/bin/bash
set -euo pipefail

# Training script for cloud GPU runs.
# Expects HF_TOKEN and optionally WANDB_API_KEY to be set.
#
# Usage:
#   ./scripts/train.sh                          # default: disney, 100K frames
#   ./scripts/train.sh open-sora 500000         # different dataset, 500K frames

DATASET="${1:-disney}"
MAX_FRAMES="${2:-100000}"
BATCH_SIZE="${3:-8}"

OUTPUT_DIR="checkpoints/${DATASET}_$(date +%Y%m%d_%H%M%S)"

echo "dataset:    $DATASET"
echo "max_frames: $MAX_FRAMES"
echo "batch_size: $BATCH_SIZE"
echo "output:     $OUTPUT_DIR"
echo ""

uv run python -m worlds1k.train.world_model \
    --dataset "$DATASET" \
    --max-frames "$MAX_FRAMES" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate 3e-4 \
    --warmup-steps 100 \
    --eval-freq 50 \
    --output-dir "$OUTPUT_DIR"
