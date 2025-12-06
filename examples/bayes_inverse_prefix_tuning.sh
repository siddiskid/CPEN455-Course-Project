#!/bin/bash
# Prefix tuning experiment
uv run -m examples.bayes_inverse \
    --method prefix_tuning \
    --max_seq_len 256 \
    --batch_size 8 \
    --num_iterations 100 \
    --prefix_length 16 \
    --learning_rate 1e-4
