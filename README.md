# onnx-pytorch


[![Build Status](https://travis-ci.com/fumihwh/onnx-pytorch.svg?branch=main)](https://travis-ci.com/fumihwh/onnx-pytorch)


Generating pytorch code from ONNX.
Currently support `onnx==1.9.0` and `torch==1.8.1`.

## Installation

- From PyPI
```
pip install onnx-pytorch
```

- From source
```
git clone https://github.com/fumihwh/onnx-pytorch.git
pip install -r requirements.txt
pip install -e .
```


## Usage
```
from onnx_pytorch import code_gen
code_gen.gen("/path/to/onnx_model", "/path/to/output_dir")
```
A `model.py` file and `variables` folder will be created under `output_dir`.

## Tutorial
- Download resnet18 onnx model
 
```wget https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v2-7.onnx```

- Use onnx-pytorch to generate pytorch code and variables.
```
from onnx_pytorch import code_gen
code_gen.gen("resnet18-v2-7.onnx", "./")
```

- Test result
```
import numpy as np
import onnx
import onnxruntime
import torch
torch.set_printoptions(8)

from model import Model

model = Model()
model.eval()
inp = np.random.randn(1, 3, 224, 224).astype(np.float32)
with torch.no_grad():
  torch_outputs = model(torch.from_numpy(inp))

onnx_model = onnx.load("resnet18-v2-7.onnx")
sess_options = onnxruntime.SessionOptions()
session = onnxruntime.InferenceSession(onnx_model.SerializeToString(),
                                       sess_options)
inputs = {"data": inp}
ort_outputs = session.run(None, inputs)

print(
    "Comparison result:",
    np.allclose(torch_outputs.detach().numpy(),
                ort_outputs[0],
                atol=1e-5,
                rtol=1e-5))
```