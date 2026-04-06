#!/bin/bash
# SOAP 1B Model Training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/train_universal.sh" \
    --model_size 1b \
    --optimizer soap \
    --num_gpus 8 \
    --lr_matrix 0.003 \
    --lr_adam 0.003 \
    --num_steps 90000 \
    --batch_size 64 \
    --total_batch_size 512 \
    --warmup_steps 9000 \
    --weight_decay 0.1 \
    --save_every 5000 \
    --eval_every 1000 \
    "$@"
