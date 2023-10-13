#!/usr/bin/env bash

# Binarize the dataset
TEXT=examples/translation/wmt14_en_de
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt14_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

fairseq-train \
    data-bin/wmt14_en_de \
    --arch transformer_vaswani_wmt_en_de_big --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion scheduled_sampling --label-smoothing 0.1 \
    --max-tokens 3584 \
    --fp16 \
    --distributed-world-size 4

# # train    
# mkdir -p checkpoints/transformer_wmt_en_de
# CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
#     data-bin/wmt14_en_de \
#     --arch transformer_wmt_en_de --share-decoder-input-output-embed \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --criterion segment_level_training --label-smoothing 0.1 \
#     --max-tokens 4096 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 3, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

# --max-tokens 4096 \

# # # evaluation
# fairseq-generate data-bin/wmt14_en_de \
#     --path checkpoints/checkpoint_best.pt \
#     --beam 5 --remove-bpe