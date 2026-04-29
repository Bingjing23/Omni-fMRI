#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Configuration
CONFIG_FILE=${CONFIG_FILE:-configs/finetune.yaml}
NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-2953}

# Optional: Output directory
OUTPUT_DIR=${OUTPUT_DIR:-outputs/finetune}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

echo "Starting DDP fine-tuning with $NUM_GPUS GPUs..."
echo "Config: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

ARGS=(--config "$CONFIG_FILE" --output_dir "$OUTPUT_DIR")
if [ -n "$RESUME_CHECKPOINT" ]; then
    ARGS+=(--resume "$RESUME_CHECKPOINT")
fi
ARGS+=("$@")

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    finetune.py \
    "${ARGS[@]}"

echo "Fine-tuning completed!"



