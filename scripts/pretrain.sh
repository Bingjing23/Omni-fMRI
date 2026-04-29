#!/bin/bash

# Configuration
CONFIG_FILE=${CONFIG_FILE:-configs/pretrain.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/pretrain}
NUM_GPUS=${NUM_GPUS:-1}  # Number of GPUs to use
MASTER_PORT=${MASTER_PORT:-2955}

# Optional: Resume from checkpoint
RESUME=${RESUME:-}  # Set to checkpoint path to resume training

echo "Starting DDP training with $NUM_GPUS GPUs..."
echo "Config: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

export OMP_NUM_THREADS=1

ARGS=(--config "$CONFIG_FILE" --output_dir "$OUTPUT_DIR")
if [ -n "$RESUME" ]; then
    ARGS+=(--resume "$RESUME")
fi
ARGS+=("$@")

# Launch training with torchrun (recommended for PyTorch >= 1.10)
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    pretrain.py \
    "${ARGS[@]}"

echo "Training completed!"

