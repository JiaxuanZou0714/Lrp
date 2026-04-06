#!/bin/bash
# SOAP 135M Model Training

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/train_universal.sh" \
    --model_size 135m \
    --optimizer soap \
    --num_gpus 4 \
    --lr_matrix 0.003 \
    --lr_adam 0.003 \
    --num_steps 20000 \
    --batch_size 128 \
    --total_batch_size 512 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --save_every 999999 \
    --eval_every 1000 \
    "$@"
