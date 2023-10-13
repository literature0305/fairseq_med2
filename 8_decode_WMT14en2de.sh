#!/usr/bin/env bash

# # evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-generate data-bin/wmt14_en_de \
    --path checkpoints/checkpoint_best.pt \
    --beam 20 --remove-bpe --batch-size 32