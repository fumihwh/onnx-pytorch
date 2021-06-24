# onnx-pytorch


![Build Status](https://github.com/fumihwh/onnx-pytorch/actions/workflows/main.yml/badge.svg?branch=main)


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

### By Command Line
```
python -m onnx_pytorch.code_gen -h

usage: code_gen.py [-h] [--onnx_model_path ONNX_MODEL_PATH] [--output_dir OUTPUT_DIR] [--overwrite OVERWRITE] [--tensor_inplace TENSOR_INPLACE] [--continue_on_error CONTINUE_ON_ERROR] [--simplify_names SIMPLIFY_NAMES]

optional arguments:
  -h, --help            show this help message and exit
  --onnx_model_path ONNX_MODEL_PATH
                        The onnx model path.
  --output_dir OUTPUT_DIR
                        The output dir
  --overwrite OVERWRITE
                        Should overwrite the output dir.
  --tensor_inplace TENSOR_INPLACE
                        Try best to inplace tensor.
  --continue_on_error CONTINUE_ON_ERROR
                        Continue on error.
  --simplify_names SIMPLIFY_NAMES
                        Use indexing shorten name instead of original name.
```

### By Python
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

## TEST
cd onnx_pytorch/tests/
pytest -s test_base.py
pytest -s test_base.py::TestBase::test_elu
pytest -s test_base.py::TestBase::test_tanh
pytest -s test_base.py::TestBase::test_sub

