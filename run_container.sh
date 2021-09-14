#!/usr/bin/env bash

docker run \
	-p 8890:8888 \
	--gpus all \
	--mount type=bind,source="$(pwd)",target=/layerjot/DBTNet \
	--mount type=bind,source="${LAYERJOT_HOME}/TBDNet_Data",target=/layerjot/DataModels \
	--mount type=bind,source="/data",target=/data \
	--rm --network host -it dbtnet:latest
