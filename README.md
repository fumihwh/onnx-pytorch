# onnx-pytorch


![Build Status](https://github.com/fumihwh/onnx-pytorch/actions/workflows/main.yml/badge.svg?branch=main)


Generates PyTorch code from ONNX.

## Installation

- From PyPI
```bash
pip install onnx-pytorch
```

- From source
```bash
git clone https://github.com/fumihwh/onnx-pytorch.git
cd onnx-pytorch
pip install -r requirements.txt
pip install -e .
```


## Usage
### By Command Line
```bash
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
```python
from onnx_pytorch import code_gen
code_gen.gen("/path/to/onnx_model", "/path/to/output_dir")
```

A `model.py` file and `variables/` folder will be created under `output_dir/`.

## Tutorial
1. Download resnet18 ONNX model.

```bash
wget https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v2-7.onnx
```

2. Use `onnx-pytorch` to generate PyTorch code and variables.
```python
from onnx_pytorch import code_gen
code_gen.gen("resnet18-v2-7.onnx", "./")
```

3. Test result.
```python
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
inputs = {session.get_inputs()[0].name: inp}
ort_outputs = session.run(None, inputs)

print(
    "Comparison result:",
    np.allclose(torch_outputs.detach().numpy(),
                ort_outputs[0],
                atol=1e-5,
                rtol=1e-5))
```

## Test
```bash
pytest onnx_pytorch/tests
```
