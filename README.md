# EasyInfer
Code to make running model inference easy in different backends. 
You don't have to worry about input/output names, dtypes and batch size. 
ONNX and Triton Inference Server are now available.

## Quick start

1. Run ```run_env.sh``` for building docker image and run docker container.
2. In docker environment from ```example``` directory run ```python3.8 prepare_model.py``` for export resnet18 to onnx with dynamic and static batch size
3. Out of docker environment from ```example``` directory run ```run_triton.sh``` for start Triton Inference Server with two exported models
4. In docker environment from workdir run ```python3.8 main.py``` for send batch with size 12 to models with size 8 in triton and onnx format. 