name=VLNBERT-test-Prevalent

suffix=$2

flag="--vlnbert prevalent

      --submit 0
      --test_only 0

      --train validlistener
      --load snap/VLNBERT-train-Prevalent-reproduce/state_dict/best_val_unseen

      --features places365
      --maxAction 15
      --feedback argmax
      --lr 1e-6
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

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name 