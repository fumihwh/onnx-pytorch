from onnx_pytorch import code_gen
code_gen.gen("resnet18-v2-7.onnx", "./")

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
