from collections import defaultdict
from typing import List, Optional
import tritonclient.http as httpclient
from .base import BaseClient
import numpy as np
import time


class TritonClient(BaseClient):
    def __init__(self, url: str,
                 model_name: str,
                 max_batch_size: int = 0,
                 sample_inputs: Optional[List[np.ndarray]] = None,
                 need_pad_to_max_batch: bool = False,
                 is_async: bool = True
                 ):
        super().__init__()
        self.model_name = model_name
        self.url = url
        assert max_batch_size >= 0, 'max_batch_size must be > 0'
        self.max_batch_size = max_batch_size
        self.is_async = is_async

        self.triton_client: httpclient.InferenceServerClient = \
            httpclient.InferenceServerClient(
                url=self.url,
                verbose=False,
                ssl=False,
            )

        self.str_to_np_mapping = {
            "FP32": np.float32,
            "FP16": np.float16,
            "UINT8": np.uint8,
            "INT32": np.int32,
            "INT64": np.int64
        }
        self.triton_inputs_formats = None
        self.np_inputs_formats = None
        
        self.inputs_shapes = None
        self.need_pad_to_max_batch = need_pad_to_max_batch
        
        self.inputs_names = None
        self.outputs_names = None
        
        self.prepare_params()
        
        self.sample_inputs = sample_inputs or self.create_sample_input()
        
        self.validate_sample_inputs()
        
        self.warmup_model()
    
    def prepare_params(self):
        config = self.triton_client.get_model_config(self.model_name)
        inputs_params = config['input']
        outputs_params = config['output']
        config_max_batch_size = config['max_batch_size']
        
        self.triton_inputs_formats = [
            params['data_type'].replace('TYPE_', '') 
            for params in inputs_params
            ]
        self.np_inputs_formats = [
            self.str_to_np_mapping[triton_format]
            for triton_format in self.triton_inputs_formats
            ]
        
        if config_max_batch_size == 0:
            self.inputs_shapes = [
                params['dims'] for params in inputs_params
            ]
            self.max_batch_size = self.inputs_shapes[0][0]
            self.need_pad_to_max_batch = True
        else:
            if self.max_batch_size == 0:
                self.max_batch_size = config_max_batch_size
            else:
                self.max_batch_size = min(config_max_batch_size, self.max_batch_size)
            
            self.inputs_shapes = [self.max_batch_size] + [
                params['dims'] for params in inputs_params
            ]
            
        self.inputs_names = [params['name'] for params in inputs_params]
        self.outputs_names = [params['name'] for params in outputs_params]
        
    def postprocess(self, outputs, padding_size):
        result = dict()
        for output_name in self.outputs_names:
            result[output_name] = outputs.as_numpy(output_name)
            if padding_size != 0:
                result[output_name] = result[output_name][:-padding_size]
        return result
    
    def create_triton_batch(self, input_data: np.ndarray, input_name: str, triton_input_format: str):
        infer_input = httpclient.InferInput(input_name, input_data.shape, triton_input_format)
        infer_input.set_data_from_numpy(input_data)
        return infer_input
    
    @staticmethod
    def get_inputs_batches_container(input_batches: list, inputs_data: list):
        return [
            [None for _ in range(len(inputs_data))] 
            for _ in range(len(input_batches))
        ]
    
    def preprocess_inputs(self, *inputs_data: np.ndarray):
        inputs_batches = None
        batches_paddings = None
        
        for input_index in range(len(inputs_data)):
            input_data = inputs_data[input_index]
            input_name = self.inputs_names[input_index]
            
            np_input_format = self.np_inputs_formats[input_index]
            triton_input_format = self.triton_inputs_formats[input_index]
            
            if input_data.dtype != np_input_format:
                input_data = input_data.astype(np_input_format)
            input_batches, input_batches_paddings = self.split_on_batches(input_data)
            
            if inputs_batches is None:
                inputs_batches = self.get_inputs_batches_container(input_batches, inputs_data)
                batches_paddings = input_batches_paddings
            
            for batch_index, input_batch in enumerate(input_batches):
                inputs_batches[batch_index][input_index] = \
                    self.create_triton_batch(input_batch, input_name, triton_input_format)
                    
        return inputs_batches, batches_paddings
    
    def forward(self, *inputs_data: np.ndarray):
        assert len(inputs_data) == len(self.inputs_names), 'inputs number is not equal to model inputs'

        inputs_batches, batches_paddings = self.preprocess_inputs(*inputs_data)

        result = defaultdict(list)
        for infer_inputs, batch_padding in zip(inputs_batches, batches_paddings):
            infer_outputs = [
                httpclient.InferRequestedOutput(output_name) \
                    for output_name in self.outputs_names
                ]
            outputs = self.triton_client.infer(
                model_name=self.model_name, 
                inputs=infer_inputs, 
                outputs=infer_outputs
                )
            outputs = self.postprocess(outputs, batch_padding)
            for output_name, output_value in outputs.items():
                result[output_name].append(output_value)
        
        for output_name, output_values in result.items(): 
            result[output_name] = np.concatenate(output_values)
            
        return self.get_sorted_values(result)
    
    def async_forward(self, *inputs_data: np.ndarray):
        assert len(inputs_data) == len(self.inputs_names), 'inputs number is not equal to model inputs'

        inputs_batches, batches_paddings = self.preprocess_inputs(*inputs_data)
        
        result = defaultdict(list)
        requests = []
        for infer_inputs in inputs_batches:
            infer_outputs = [
                httpclient.InferRequestedOutput(output_name) \
                    for output_name in self.outputs_names
                ]
            
            request = self.triton_client.async_infer(
                model_name=self.model_name, 
                inputs=infer_inputs, 
                outputs=infer_outputs
                )
            
            requests.append(request)
        
        for request, batch_padding in zip(requests, batches_paddings):
            output = self.postprocess(request.get_result(), batch_padding)
            for output_name, output_value in output.items():
                result[output_name].append(output_value)
        
        for output_name, output_values in result.items(): 
            result[output_name] = np.concatenate(output_values)
        
        return self.get_sorted_values(result)

    def send_async_requests(self, *inputs_data: np.ndarray):
        assert len(inputs_data) == len(self.inputs_names), 'inputs number is not equal to model inputs'

        inputs_batches, batches_paddings = self.preprocess_inputs(*inputs_data)
        
        requests = []
        for infer_inputs in inputs_batches:
            infer_outputs = [
                httpclient.InferRequestedOutput(output_name) \
                    for output_name in self.outputs_names
                ]
            
            request = self.triton_client.async_infer(
                model_name=self.model_name, 
                inputs=infer_inputs, 
                outputs=infer_outputs
                )
            
            requests.append(request)
        
        return requests, batches_paddings
    
    def get_async_results(self, requests, batches_paddings):
        result = defaultdict(list)
        for request, batch_padding in zip(requests, batches_paddings):
            output = self.postprocess(request.get_result(), batch_padding)
            for output_name, output_value in output.items():
                result[output_name].append(output_value)
        
        for output_name, output_values in result.items(): 
            result[output_name] = np.concatenate(output_values, axis=1)
        
        return self.get_sorted_values(result)