python3 train_imagenet_dbt.py \
  --rec-train ../data/imagenet/train.rec --rec-train-idx ../data/imagenet/train.idx \
  --rec-val ../data/imagenet/val.rec --rec-val-idx ../data/imagenet/val.idx \
  --model resnet50  --mode hybrid \
  --lr 0.4 --lr-mode cosine --num-epochs 120 --batch-size 16 --num-gpus 2 -j 12 \
  --warmup-epochs 5 --dtype float16 \
  --use-rec --no-wd --label-smoothing --last-gamma \
  --save-dir ../model/params_imagenet_dbt \
  --logging-file ../model/log/imagenet_dbt.log
