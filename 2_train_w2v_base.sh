#!/bin/bash

fairseq-hydra-train \
    task.data=/home/Workspace/fairseq/data \
    --config-dir /home/Workspace/fairseq/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_base_librispeech