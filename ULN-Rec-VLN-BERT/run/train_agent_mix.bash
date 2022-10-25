#!/bin/bash
gpu_id=$1
name=$2
log_file=agent_$name
# name=VLNBERT-train-Prevalent
# --aug data/prevalent/prevalent_aug_merged_last.json

flag="--vlnbert prevalent

      --test_only 0

      --train listener

      --features places365
      --maxAction 15
      --batchSize 8
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$gpu_id python r2r_src/train.py $flag --name $name 
# |& tee logs/$log_file.txt 2>&1 &
