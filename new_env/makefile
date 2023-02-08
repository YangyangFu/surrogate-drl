IMA_NAME = test_env
IMA_NAME_D = test_env_d
HOST = shichao






DOCKERFILE_GPU_PY3 = Dockerfile

TAG_GPU_PY3 = gpu_py3


build_gpu_py3:
	docker build -f ${DOCKERFILE_GPU_PY3} --no-cache --rm -t ${IMA_NAME} .

remove_image:
	docker rmi ${IMA_NAME}

tag_gpu_py3:
	docker tag ${IMA_NAME} ${HOST}/${IMA_NAME}:${TAG_GPU_PY3}

push_gpu_py3:
	docker push ${HOST}/${IMA_NAME}:${TAG_GPU_PY3}