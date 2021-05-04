# onnx-pytorch

Generating pytorch code from ONNX.
Currently support `onnx==1.9.0` and `torch==1.8.1`.

## Usage
```
from onnx_pytorch import code_gen
code_gen.gen("/path/to/onnx_model", "/path/to/output_dir")
```
A `model.py` file and `variables` folder will be created under `output_dir`.