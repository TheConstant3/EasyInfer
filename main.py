from src import OnnxClient, TritonClient
import numpy as np


input = np.random.random((12, 3, 224, 224))

model_name = 'model_static'

model_onnx = OnnxClient(
    model_name=model_name, 
    model_path=f'example/triton_repo/{model_name}/1/model.onnx',
    device='cuda'
    )
onnx_result = model_onnx(input)


model_triton = TritonClient(
    url='localhost:8000',
    model_name=model_name
    )
triton_result = model_triton(input)

for onnx_out, triton_out in zip(onnx_result, triton_result):
    print('onnx result shape', onnx_out.shape)
    print('triton result shape', triton_out.shape)
    assert np.argmax(onnx_out[0]) == np.argmax(triton_out[0])
    

