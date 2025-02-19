#!/bin/sh
image=fastapi:latest

docker run --rm -it \
	--shm-size 16G \
	-p 8010:8010 \
	$image \
