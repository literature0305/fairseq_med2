#!/bin/bash
stage=3
stop_stage=1000
datadir=/DB/LibriSpeech # dir to save data
tsvdir=data # dir to save tsv

# name LibriSpeech datasets
data_sets="train-clean-360 train-other-500"

python examples/wav2vec/wav2vec_manifest.py $datadir --dest $tsvdir --ext flac --valid-percent 0


# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#     echo "stage 3: Prepare tsv"
#     for part in $data_sets; do
#         python examples/wav2vec/wav2vec_manifest.py $datadir/LibriSpeech/$part --dest $tsvdir/$part --ext flac --valid-percent 0
#         mv $tsvdir/$part/train.tsv $tsvdir/$part/wav_dir.tsv
#     done
# fi