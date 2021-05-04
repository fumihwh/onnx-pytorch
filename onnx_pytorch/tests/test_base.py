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

torch.set_printoptions(8)


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
      if type(pt_outputs) == torch.Tensor:
        pt_outputs = [pt_outputs.detach().numpy()]
      elif type(pt_outputs) in (list, tuple):
        pt_outputs = [o.detach().numpy() for o in pt_outputs]
      for l, r in zip(ort_outputs, pt_outputs):
        assert np.allclose(l, r, atol=1e-4, rtol=1e-4, equal_nan=True)

  def test_conv_flatten_relu(self):
    reset_model()
    nps = [np.random.randn(1, 3, 224, 224).astype(np.float32)]
    inputs = Input(*nps)
    conv_node = Conv(inputs[0],
                     np.random.randn(32, 3, 3, 3).astype(np.float32),
                     np.random.randn(32).astype(np.float32))
    flatten_node = Flatten(conv_node)
    relu_node = Relu(flatten_node)
    Output(relu_node)
    self._run(list(zip(inputs, nps)))

  def test_conv_batchnorm_maxpool_flatten_add_relu(self):
    reset_model()
    nps = [np.random.randn(1, 3, 224, 224).astype(np.float32)]
    inputs = Input(*nps)
    conv_node = Conv(inputs[0],
                     np.random.randn(32, 3, 3, 3).astype(np.float32),
                     np.random.randn(32).astype(np.float32))
    bn_node = BatchNormalization(
        conv_node,
        np.ones(32,).astype(np.float32),
        np.zeros(32,).astype(np.float32),
        np.random.randn(32).astype(np.float32),
        np.random.randn(32).astype(np.float32),
    )
    max_pool_node = MaxPool(bn_node,
                            kernel_shape=(3, 3),
                            strides=(2, 2),
                            pads=(0, 0, 1, 1))
    flatten_node = Flatten(max_pool_node, axis=1)
    add_node = Add(flatten_node, np.random.randn(1).astype(np.float32))
    relu_node = Relu(add_node)
    Output(relu_node)
    self._run(list(zip(inputs, nps)))

  def test_reshape(self):
    reset_model()
    nps = [np.random.randn(4,).astype(np.float32)]
    inputs = Input(*nps)
    Output(Reshape(inputs[0], np.array((2, 2)).astype(np.int64)))
    self._run(list(zip(inputs, nps)))

  def test_maxpool(self):
    reset_model()
    nps = [np.random.randn(1, 1, 5, 5).astype(np.float32)]
    inputs = Input(*nps)
    Output(MaxPool(inputs[0], kernel_shape=(3, 3), pads=(0, 0, 1, 1)))
    self._run(list(zip(inputs, nps)))

  def test_sigmoid(self):
    reset_model()
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(Sigmoid(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_slice(self):
    reset_model()
    nps = [np.random.randn(5, 5).astype(np.float32)]
    inputs = Input(*nps)
    Output(
        Slice(inputs[0],
              np.array((2, 3)).astype(np.int64),
              np.array((4, 5)).astype(np.int64)))
    self._run(list(zip(inputs, nps)))

  def test_reduce_sum(self):
    reset_model()
    nps = [np.random.randn(1, 2, 3).astype(np.float32)]
    inputs = Input(*nps)
    Output(ReduceSum(inputs[0], np.array((1, 2)).astype(np.int64)))
    self._run(list(zip(inputs, nps)))

  def test_mul(self):
    reset_model()
    nps = [
        np.random.randn(1, 2, 3).astype(np.float32),
        np.random.randn(1, 2, 3).astype(np.float32)
    ]
    inputs = Input(*nps)
    Output(Mul(*inputs))
    self._run(list(zip(inputs, nps)))

  def test_mat_mul(self):
    reset_model()
    nps = [
        np.random.randn(5, 2, 3).astype(np.float32),
        np.random.randn(5, 3, 2).astype(np.float32)
    ]
    inputs = Input(*nps)
    Output(MatMul(*inputs))
    self._run(list(zip(inputs, nps)))

  def test_softmax(self):
    reset_model()
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(Softmax(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_reciprocal(self):
    reset_model()
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(Reciprocal(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_sqrt(self):
    reset_model()
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(Sqrt(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_max(self):
    reset_model()
    nps = [
        np.random.randn(1, 10).astype(np.float32),
        np.random.randn(2, 10).astype(np.float32)
    ]
    inputs = Input(*nps)
    Output(Max(inputs))
    self._run(list(zip(inputs, nps)))

  def test_concat(self):
    reset_model()
    nps = [
        np.random.randn(1, 10).astype(np.float32),
        np.random.randn(2, 10).astype(np.float32)
    ]
    inputs = Input(*nps)
    Output(Concat(inputs, axis=0))
    self._run(list(zip(inputs, nps)))

  def test_split(self):
    reset_model()
    nps = [
        np.random.randn(1, 10).astype(np.float32),
    ]
    inputs = Input(*nps)
    Output(Split(inputs, split=np.array([2, 8]), axis=1))
    self._run(list(zip(inputs, nps)))


if __name__ == '__main__':
  pytest.main(['-s', 'test_base.py'])
