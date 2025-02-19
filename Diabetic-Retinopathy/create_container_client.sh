#!/bin/sh
image=nvcr.io/nvidia/tritonserver:23.01-py3-sdk

while getopts "i:" OPTION; do
	case $OPTION in
		i) img=$OPTARG;;
		*) exit 1 ;;
	esac
done

echo "($img)"

docker run --rm \
	--shm-size 16G \
	-v $HOME/truefoundry_deployment/Diabetic-Retinopathy/model_repository:/workspace/model_repository \
	--net host \
	$image \
	python model_repository/pipeline/1/client.py --image $img --url ophtha-deployment.apps.wadhwaniai.org/optha-inference-service-8000/