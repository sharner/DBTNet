python3 ft_cub_dbt.py \
  --rec-train ../data/n1_crops_dataset/n1_training_clean/n1_rec.rec --rec-train-idx ../data/n1_crops_dataset/n1_training_clean/n1_rec.idx \
  --rec-val ../data/n1_crops_dataset/n1_training_clean/n1_val_rec.rec --rec-val-idx ../data/n1_crops_dataset/n1_training_clean/n1_val_rec.idx \
  --model resnet50 --mode hybrid \
  --lr 0.05 --lr-mode cosine --num-epochs 1000 --batch-size 16 --num-gpus 2 -j 12 --crop-ratio 0.875\
  --warmup-epochs 0 --dtype float16 \
  --use-rec --no-wd --label-smoothing --last-gamma \
  --save-dir ../model/params_n1_training_clean_dbt \
  --logging-file ../model/log/n1_training_clean_dbt.log 


