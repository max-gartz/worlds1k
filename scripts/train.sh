#!/bin/bash
set -euo pipefail

# Training script for cloud GPU runs.
# Expects HF_TOKEN to be set and uv to be installed.
#
# Usage:
#   ./scripts/train.sh              # default: disney, 500 clips
#   ./scripts/train.sh open-sora    # different dataset

DATASET="${1:-disney}"
MAX_SAMPLES="${2:-500}"
BATCH_SIZE="${3:-8}"
EPOCHS="${4:-50}"

OUTPUT_DIR="checkpoints/${DATASET}_$(date +%Y%m%d_%H%M%S)"

echo "dataset:     $DATASET"
echo "max_samples: $MAX_SAMPLES"
echo "batch_size:  $BATCH_SIZE"
echo "epochs:      $EPOCHS"
echo "output:      $OUTPUT_DIR"
echo ""

uv run python -m worlds1k.training.pretrain \
    --dataset "$DATASET" \
    --max-samples "$MAX_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$EPOCHS" \
    --learning-rate 3e-4 \
    --warmup-steps 100 \
    --eval-freq 50 \
    --output-dir "$OUTPUT_DIR"
