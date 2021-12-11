#!/usr/bin/env bash

docker run \
	--gpus all \
	--mount type=bind,source="$(pwd)",target=/forest/DBTNet \
	--mount type=bind,source="/home",target=/home \
	--rm --network host -it dbtnet:latest
