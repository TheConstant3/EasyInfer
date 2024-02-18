docker build . -t easy_infer
docker run --gpus all --net host -it --rm -v $(pwd):/app easy_infer