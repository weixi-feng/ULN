#!/bin/bash
gpu_id=$1
name=$2
log_file=explorer_$name
# name=VLNBERT-train-Prevalent
# --aug data/prevalent/prevalent_aug_merged_last.json

flag="--vlnbert prevalent

      --test_only 0

      --train explorer
      --load snap/VLNBERT-train-Prevalent-reproduce/state_dict/best_val_unseen

      --features places365
      --maxAction 15
      --batchSize 8
      --feedback teacher
      --lr 1e-4
      --iters 10000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5
      --e2e
      "

      # --speaker
      # --speaker_snap snap/speaker_transformer3/state_dict/best_val_unseen_bleu

# mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$gpu_id python r2r_src/train.py $flag --name $name
#  |& tee logs/$log_file.txt 2>&1 &