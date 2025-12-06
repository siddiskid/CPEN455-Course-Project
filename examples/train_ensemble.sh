#!/bin/bash

SEEDS=(0 42 123 456 789)
NUM_SEEDS=3

for i in $(seq 0 $((NUM_SEEDS-1))); do
    SEED=${SEEDS[$i]}
    echo "=========================================="
    echo "Training model with seed $SEED"
    echo "=========================================="
    
    uv run -m examples.bayes_inverse \
        --method full_finetune \
        --max_seq_len 256 \
        --batch_size 8 \
        --num_iterations 80 \
        --learning_rate 1e-5 \
        --seed $SEED \
        --ckpt_suffix "_seed${SEED}"
done

echo "=========================================="
echo "All ensemble models trained!"
echo "=========================================="
