python3 export_dbt.py \
	--model resnet50 --mode hybrid \
	--load-params ../model/params_n1_training_clean_dbt.def.1K/imagenet-resnet50-999.params\
	--num-epochs 1000 --batch-size 16 --num-gpus 2 \
	--input-size 224\
	--nclasses 457\
	--dtype float16 \
	--last-gamma \
	--export-dir ../model/export/n1_training_clean_default_dbt \
	--logging-file ../model/log/n1_export_clean_dbt.log 
