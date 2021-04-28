from tempfile import TemporaryDirectory

import os
import importlib.util

import numpy as np
import onnx
import onnxruntime
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
import pytest
import torch

from onnx_model_maker import *
from onnx_model_maker.ops import *
from onnx_pytorch import code_gen

torch.set_printoptions(6)


class TestBase:

  def _run(self, inputs_np):
    model = onnx.ModelProto()
    model.CopyFrom(omm.model)
    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(model.SerializeToString(),
                                           sess_options)
    ort_outputs = session.run(None, {k: v for k, v in inputs_np})
    model.graph.ClearField("value_info")
    model = SymbolicShapeInference.infer_shapes(model, 2**31 - 1, True, True, 1)
    with TemporaryDirectory() as tmpdir:
      code_gen.gen(model, output_dir=tmpdir)
      spec = importlib.util.spec_from_file_location(
          "model", os.path.join(tmpdir, "model.py"))
      mod = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(mod)
      pt_outputs = mod.test_run_model(
          [torch.from_numpy(v) for _, v in inputs_np])
      assert np.allclose(ort_outputs, [o.detach().numpy() for o in pt_outputs],
                         atol=1e-4,
                         rtol=1e-4)

  def test_conv_flatten_relu(self):
    reset_model()
    inputs = Input(np.random.randn(1, 3, 224, 224).astype(np.float32))
    conv_node = Conv(inputs[0],
                     np.random.randn(32, 3, 3, 3).astype(np.float32),
                     np.random.randn(32).astype(np.float32))
    flatten_node = Flatten(conv_node)
    relu_node = Relu(flatten_node)
    Output(relu_node)

    self._run([(inputs[0], np.random.randn(1, 3, 224, 224).astype(np.float32))])

  def test_conv_maxpool_flatten_add_relu(self):
    reset_model()
    inputs = Input(np.random.randn(1, 3, 224, 224).astype(np.float32))
    conv_node = Conv(inputs[0],
                     np.random.randn(32, 3, 3, 3).astype(np.float32),
                     np.random.randn(32).astype(np.float32))
    max_pool_node = MaxPool(conv_node,
                            kernel_shape=(3, 3),
                            strides=(2, 2),
                            pads=(0, 0, 1, 1))
    flatten_node = Flatten(max_pool_node, axis=1)
    add_node = Add(flatten_node, np.random.randn(1).astype(np.float32))
    relu_node = Relu(add_node)
    Output(relu_node)

    self._run([(inputs[0], np.random.randn(1, 3, 224, 224).astype(np.float32))])


if __name__ == '__main__':
  pytest.main(['-s', 'test_base.py'])
