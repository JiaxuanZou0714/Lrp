#!/bin/bash
# NORA (Normalized Orthogonal Row Alignment) 60M Model Training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/train_universal.sh" \
    --model_size 60m \
    --optimizer NORA \
    --num_gpus 2 \
    --lr_matrix 0.005 \
    --lr_adam 0.005 \
    --r 1.9 \
    --num_steps 10000 \
    --batch_size 64 \
    --total_batch_size 512 \
    --warmup_steps 1000 \
    --weight_decay 0.0 \
    --save_every 5000 \
    --eval_every 1000 \
    "$@"
