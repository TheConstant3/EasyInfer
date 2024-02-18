import torch
import torchvision.models as models


model = models.resnet18(pretrained=True)

model.eval()

dummy_input = torch.randn(8, 3, 224, 224)

input_names = [ "input" ]
output_names = [ "output" ]

torch.onnx.export(model, 
                  dummy_input,
                  "triton_repo/model_dynamic/1/model.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  dynamic_axes={
                      "input": {0: "batch_size"},
                      "output": {0: "batch_size"},
                  }
                  )

torch.onnx.export(model, 
                  dummy_input,
                  "triton_repo/model_static/1/model.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )

