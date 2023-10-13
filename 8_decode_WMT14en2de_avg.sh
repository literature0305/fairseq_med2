#!/usr/bin/env bash

python scripts/average_checkpoints.py \
    --inputs checkpoints \
    --checkpoint-upper-bound $1 \
    --num-epoch-checkpoints 10 \
    --output checkpoints/checkpoint.avg_tmp.pt

CUDA_VISIBLE_DEVICES=1 fairseq-generate \
    data-bin/wmt14_en_de \
    --path checkpoints/checkpoint.avg_tmp.pt \
    --beam 20 --lenpen 0.6 --remove-bpe --batch-size 32
    # > gen.out

# # # evaluation
# fairseq-generate data-bin/wmt14_en_de \
#     --path checkpoints/checkpoint_best.pt \
#     --beam 20 --remove-bpe