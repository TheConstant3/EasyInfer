from collections import defaultdict
from typing import List, Optional
from .base import BaseClient
import onnxruntime as rt
import numpy as np
import time
import os


class OnnxClient(BaseClient):
    def __init__(self, model_path: str, 
                 model_name: str, 
                 max_batch_size: int = 1,
                 sample_inputs: Optional[List[np.ndarray]] = None,
                 need_pad_to_max_batch: bool = False,
                 device: str = 'cpu'):
        super().__init__()
        assert max_batch_size > 0, 'max_batch_size must be > 0'
        self.max_batch_size = max_batch_size
        self.need_pad_to_max_batch = need_pad_to_max_batch
        self.model_name = model_name
        
        sess_options = rt.SessionOptions()
        self.onnx_model = rt.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider' if device == 'cuda'
                       else 'CPUExecutionProvider'],
            sess_options=sess_options
        )

        self.onnx_to_np_types = {
            'tensor(float)': np.float32,
            'tensor(float16)': np.float16,
        }
        
        self.prepare_params()
        
        self.sample_inputs = sample_inputs or self.create_sample_input()
        
        self.validate_sample_inputs()
        
        self.warmup_model()
    
    def prepare_params(self):
        model_inputs = self.onnx_model.get_inputs()
        self.inputs_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.inputs_types = [
            self.onnx_to_np_types[model_inputs[i].type] 
            for i in range(len(model_inputs))
            ]
        
        self.inputs_shapes = []
        for i in range(len(model_inputs)):
            new_shape = [
                s if isinstance(s, int) else -1
                for s in model_inputs[i].shape
            ]
            # if model has dynamic batch shape, set max_batch_size
            if new_shape[0] == -1:
                new_shape[0] = self.max_batch_size
            # else pad to batch dim size
            else:
                self.need_pad_to_max_batch = True
                self.max_batch_size = new_shape[0]
            self.inputs_shapes.append(new_shape)

        model_outputs = self.onnx_model.get_outputs()
        self.outputs_names = [model_outputs[i].name for i in range(len(model_outputs))]
    
    def preprocess_inputs(self, *inputs_data: np.ndarray):
        inputs_batches = None
        batches_paddings = None
        
        for i_input, input_data in enumerate(inputs_data):
            input_name = self.inputs_names[i_input]
            input_type = self.inputs_types[i_input]
            
            input_batches, input_batches_paddings = self.split_on_batches(input_data)

            if inputs_batches is None:
                inputs_batches = [dict() for _ in range(len(input_batches))]
                batches_paddings = input_batches_paddings
            
            for i_batch, batch in enumerate(input_batches):
                inputs_batches[i_batch][input_name] = batch.astype(input_type)
        
        return inputs_batches, batches_paddings
    
    def forward(self, *inputs_data: np.ndarray):
        inputs_batches, batches_paddings = self.preprocess_inputs(*inputs_data)
        
        result = defaultdict(list)
        for batch, batch_padding in zip(inputs_batches, batches_paddings):
            outputs = self.onnx_model.run(self.outputs_names, batch)
            for output_name, output_values in zip(self.outputs_names, outputs):
                result[output_name].append(
                    output_values if batch_padding == 0
                    else output_values[:-batch_padding]
                    )
        
        for output_name, output_values in result.items(): 
            result[output_name] = np.concatenate(output_values)
            
        return self.get_sorted_values(result)
