import numpy as np
import time
import os


class BaseClient:
    def __init__(self, *args, **kwargs):
        self.show_fps = os.environ.get('SHOW_FPS') in {'yes', 'true'}
        self.model_name = ''
        self.sample_inputs = None
        self.need_pad_to_max_batch = False
        self.max_batch_size = 1
        self.is_async = False
        
        self.inputs_shapes = []
        self.inputs_names = []
        self.outputs_names = []
    
    def get_sorted_values(self, dict_answer):
        return [dict_answer[name] for name in self.outputs_names]
    
    def prepare_params(self):
        raise NotImplementedError()
    
    def validate_sample_inputs(self):
        # check each input
        for sample_input, input_shape in zip(self.sample_inputs, self.inputs_shapes):
            # compare each dim in input
            for i, (s_dim, t_dim) in enumerate(zip(sample_input.shape, input_shape)):
                # first dim is batch
                if i == 0:
                    # if model support only fixed batch size
                    if self.need_pad_to_max_batch:
                        assert s_dim == t_dim, \
                            f'model support fixed batch size {t_dim}, \
                                sample_inputs has batch size {s_dim}'
                    else:
                        assert s_dim <= t_dim, \
                            f'model support max batch size {t_dim}, \
                                sample_inputs has batch size {s_dim}'
                    continue
                # if dim in model config is -1, we can skip it
                assert ((t_dim != -1) and (int(s_dim) == int(t_dim))) or t_dim == -1, \
                    f'incorrect shape in sample_inputs {sample_input.shape}, must be {input_shape}'
    
    def create_sample_input(self):
        has_dynamic_shapes = any(-1 in x for x in self.inputs_shapes)
        if has_dynamic_shapes:
            return None
        
        sample_inputs = []
        for input_shape in self.inputs_shapes:
            sample_inputs.append(
                np.ones(input_shape)
            )
        return sample_inputs
    
    def log(self, text, warn=False, err=False):
        text = f'{self.__class__.__name__} ({self.model_name}) - {text}'
        if err:
            print('error', text)
        elif warn:
            print('warning',text)
        else:
            print('debug', text)

    def warmup_model(self):
        if self.sample_inputs is None:
            self.log('Model was not warmed up, because sample_inputs didn\'t set or shape is dynamic and cannot auto generate', warn=True)
            return
        exception = None
        for _ in range(5):
            try:
                _ = self.__call__(*self.sample_inputs)
                exception = None
            except Exception as e:
                self.log(f'{e} while warmup, repeat inference...', err=True)
                exception = e
                time.sleep(2)
        if exception is not None:
            raise exception
    
    @staticmethod
    def pad_batch(batch: np.ndarray, padding_size: int):
        padded_batch = np.pad(batch, ((0, padding_size), (0, 0), (0, 0), (0, 0)))
        return padded_batch
    
    def split_on_batches(self, input_data: np.ndarray):
        batches = []
        paddings = []
        for i in range(0, len(input_data), self.max_batch_size):
            batch = input_data[i:i+self.max_batch_size]
            
            padding_size = 0
            if self.need_pad_to_max_batch:
                padding_size = self.max_batch_size - batch.shape[0]
                if padding_size > 0:
                    batch = self.pad_batch(batch, padding_size)
            
            batches.append(batch)
            paddings.append(padding_size)
        return batches, paddings
    
    def forward(self, *input_data):
        raise NotImplementedError
    
    def async_forward(self, *input_data):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        t1 = time.time()
        forward_func = self.async_forward if self.is_async else self.forward
        output = forward_func(*args, **kwargs)
        t2 = time.time()
        if self.show_fps:
            self.log(f'fps {int(len(args[0])/(t2-t1))}')
        return output
