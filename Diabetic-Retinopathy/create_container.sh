#!/usr/bin/env bash
# example command to run
# bash create_container.sh -g 0 -n pm-container
# -g: gpu number
# -n: name of the container

# get inputs
while getopts "g:n:" OPTION; do
	case $OPTION in
		g) gpu=$OPTARG;;
		n) name=$OPTARG;;
		*) exit 1 ;;
	esac
done

image=tritonserver/optha:1.0

docker run --rm -ti \
	--shm-size 16G \
	--name "$name" \
	-v $HOME/truefoundry_deployment/Diabetic-Retinopathy/src:/opt/tritonserver/src \
	-v $HOME/truefoundry_deployment/Diabetic-Retinopathy/config:/opt/tritonserver/config \
	-v $HOME/scratchj/ehealth/optha/eyepacs:/opt/tritonserver/scratchj/ehealth/optha/eyepacs \
	-v $HOME/truefoundry_deployment/Diabetic-Retinopathy/model_repository:/opt/tritonserver/model_repository \
	-p 8080:8000 \
    -p 8081:8001 \
    -p 8082:8002 \
    --ulimit stack=67108864 \
	--ipc host \
	$image