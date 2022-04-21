python3 export_to_onnx.py \
	--load-params ../model/params_n1_training_clean_dbt/imagenet-resnet50-0009.params \
	--load-syms ../model/params_n1_training_clean_dbt/imagenet-resnet50-symbol.json \
	--onnx-file test.onnx \
	--input-size 224\
	--nclasses 457
