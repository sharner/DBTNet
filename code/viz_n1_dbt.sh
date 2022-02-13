python3 viz_dbt.py \
  --data-dir ../data/n1_crops_dataset/n1_training_clean/ \
  --rec-train ../data/n1_crops_dataset/n1_training_clean/n1_rec.rec --rec-train-idx ../data/n1_crops_dataset/n1_training_clean/n1_rec.idx \
  --rec-val ../data/n1_crops_dataset/n1_training_clean/n1_val_rec.rec --rec-val-idx ../data/n1_crops_dataset/n1_training_clean/n1_val_rec.idx \
  --model resnet50 --mode hybrid \
  --parameters \
  ../model/params_n1_training_clean_dbt/imagenet-resnet50-999.params \
  --batch-size 16 --num-gpus 1 -j 12 --crop-ratio 0.875\
  --nclasses 457\
  --num-training-samples 15782\
  --dtype float16 \
  --use-rec --last-gamma --use-pretrained
