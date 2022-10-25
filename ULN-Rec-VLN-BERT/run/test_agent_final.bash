name=VLNBERT-test-base

suffix=$2

flag="--vlnbert prevalent

      --submit 1
      --test_only 0

      --train validlistener
      --load snap/VLNBERT-Prevalent-final/state_dict/best_val_unseen

      --features places365
      --maxAction 15
      --feedback argmax
      --lr 1e-5
      --iters 300000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5
      --suffix $suffix
      --batchSize 8
      "
      # --load_classifier snap/classifier/state_dict/best_val_unseen
      # --classify_first
      # --load_explorer snap/VLNBERT-train-Prevalent-reproduce-explorer/state_dict/best_val_unseen
      # --e2e
      # --state_freeze

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name 
