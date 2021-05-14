import logging
from tempfile import TemporaryDirectory

import os
import importlib.util

import numpy as np
import onnx
import onnxruntime
from onnx.helper import make_tensor
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
    ort_outputs = session.run(None, {k: v for k, v in inputs_np if k != ""})
    model.graph.ClearField("value_info")
    try:
      model = SymbolicShapeInference.infer_shapes(model, 2**31 - 1, True, True,
                                                  1)
    except:
      logging.warning("Shape infer by onnxruntime failed.")
    with TemporaryDirectory() as tmpdir:
      code_gen.gen(model,
                   output_dir=tmpdir,
                   tensor_inplace=True,
                   simplify_names=True)
      spec = importlib.util.spec_from_file_location(
          "model", os.path.join(tmpdir, "model.py"))
      mod = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(mod)
      pt_outputs = mod.test_run_model(
          [torch.from_numpy(v) for k, v in inputs_np if k != ""])
      if type(pt_outputs) == torch.Tensor:
        pt_outputs = [pt_outputs.detach().numpy()]
      elif type(pt_outputs) in (list, tuple):
        pt_outputs = [o.detach().numpy() for o in pt_outputs]
      for l, r in zip(ort_outputs, pt_outputs):
        assert np.allclose(l, r, atol=1e-4, rtol=1e-4, equal_nan=True)

  def test_conv_flatten_relu(self):
    reset_model(13)
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
    reset_model(13)
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

  def test_abs(self):
    reset_model(13)
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(Abs(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_acos(self):
    reset_model(13)
    nps = [np.random.randn(5).astype(np.float32)]
    inputs = Input(*nps)
    Output(Acos(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_acosh(self):
    reset_model(13)
    nps = [np.random.randn(5).astype(np.float32)]
    inputs = Input(*nps)
    Output(Acosh(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_add(self):
    reset_model(13)
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(Add(inputs[0], np.random.randn(1, 10).astype(np.float32)))
    self._run(list(zip(inputs, nps)))

  def test_and(self):
    reset_model(13)
    nps = [
        np.random.randint(low=0, high=1, size=(5,)).astype(np.bool),
        np.random.randint(low=0, high=1, size=(5,)).astype(np.bool)
    ]
    inputs = Input(*nps)
    Output(And(*inputs))
    self._run(list(zip(inputs, nps)))

  def test_argmax(self):
    reset_model(13)
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(ArgMax(inputs, axis=1))
    self._run(list(zip(inputs, nps)))

  def test_argmin(self):
    reset_model(13)
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(ArgMin(inputs, axis=1))
    self._run(list(zip(inputs, nps)))

  def test_asin(self):
    reset_model(13)
    nps = [np.random.randn(5).astype(np.float32)]
    inputs = Input(*nps)
    Output(Asin(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_asinh(self):
    reset_model(13)
    nps = [np.random.randn(5).astype(np.float32)]
    inputs = Input(*nps)
    Output(Asinh(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_atan(self):
    reset_model(13)
    nps = [np.random.randn(5).astype(np.float32)]
    inputs = Input(*nps)
    Output(Atan(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_atanh(self):
    reset_model(13)
    nps = [np.random.randn(5).astype(np.float32)]
    inputs = Input(*nps)
    Output(Atanh(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_avg_pool(self):
    reset_model(13)
    nps = [np.random.randn(1, 1, 5, 5).astype(np.float32)]
    inputs = Input(*nps)
    Output(AveragePool(inputs, kernel_shape=(3, 3), pads=(0, 0, 1, 1)))
    self._run(list(zip(inputs, nps)))

  def test_avg_pool_no_pad(self):
    reset_model(13)
    nps = [np.random.randn(1, 1, 6, 6).astype(np.float32)]
    inputs = Input(*nps)
    Output(AveragePool(inputs, kernel_shape=(3, 3)))
    self._run(list(zip(inputs, nps)))

  def test_bitshift_left(self):
    reset_model(13)
    nps = [np.array([1, 2]).astype(np.uint8), np.array([1, 2]).astype(np.uint8)]
    inputs = Input(*nps)
    Output(BitShift(*inputs, direction="LEFT"))
    self._run(list(zip(inputs, nps)))

  def test_bitshift_right(self):
    reset_model(13)
    nps = [np.array([1, 4]).astype(np.uint8), np.array([1, 1]).astype(np.uint8)]
    inputs = Input(*nps)
    Output(BitShift(*inputs, direction="RIGHT"))
    self._run(list(zip(inputs, nps)))

  def test_batch_normalization(self):
    reset_model(13)
    nps = [np.random.randn(1, 32, 3, 3).astype(np.float32)]
    inputs = Input(*nps)
    Output(
        BatchNormalization(
            inputs[0],
            np.ones(32,).astype(np.float32),
            np.zeros(32,).astype(np.float32),
            np.random.randn(32).astype(np.float32),
            np.random.randn(32).astype(np.float32),
        ))
    self._run(list(zip(inputs, nps)))

  def test_cast(self):
    reset_model(13)
    nps = [
        np.random.randn(1, 10).astype(np.float32),
    ]
    inputs = Input(*nps)
    Output(Cast(inputs, to=6))
    self._run(list(zip(inputs, nps)))

  def test_ceil(self):
    reset_model(13)
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(Ceil(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_clip(self):
    reset_model(13)
    nps = [
        np.random.randn(1, 5).astype(np.float32),
        np.asarray(0).astype(np.float32)
    ]
    inputs = Input(*nps)
    Output(Clip(*inputs))
    self._run(list(zip(inputs, nps)))

  def test_concat(self):
    reset_model(13)
    nps = [
        np.random.randn(1, 10).astype(np.float32),
        np.random.randn(2, 10).astype(np.float32)
    ]
    inputs = Input(*nps)
    Output(Concat(inputs, axis=0))
    self._run(list(zip(inputs, nps)))

  def test_constant_add(self):
    reset_model(13)
    nps = [
        np.random.randn(1, 10).astype(np.float32),
    ]
    inputs = Input(*nps)
    constant_value = np.random.randn(1, 10).astype(np.float32)
    t = make_tensor("", 1, constant_value.shape, constant_value.flatten())
    Output(Add(inputs, Constant(value=t)))
    self._run(list(zip(inputs, nps)))

  def test_constant_of_shape(self):
    reset_model(13)
    nps = [
        np.array([2, 3]).astype(np.int64),
    ]
    inputs = Input(*nps)
    constant_value = np.random.randn(1).astype(np.float32)
    t = make_tensor("", 1, constant_value.shape, constant_value)
    Output(ConstantOfShape(inputs, value=t))
    self._run(list(zip(inputs, nps)))

  def test_conv(self):
    reset_model(13)
    nps = [np.random.randn(1, 3, 3, 3).astype(np.float32)]
    inputs = Input(*nps)
    Output(
        Conv(
            inputs[0],
            np.random.randn(32, 3, 3, 3).astype(np.float32),
            np.random.randn(32).astype(np.float32),
        ))
    self._run(list(zip(inputs, nps)))

  def test_conv_transpose(self):
    reset_model(13)
    nps = [np.random.randn(1, 3, 3, 3).astype(np.float32)]
    inputs = Input(*nps)
    Output(
        ConvTranspose(
            inputs[0],
            np.random.randn(3, 32, 3, 3).astype(np.float32),
            np.random.randn(32).astype(np.float32),
        ))
    self._run(list(zip(inputs, nps)))

  def test_conv_transpose_pads(self):
    reset_model(13)
    nps = [np.random.randn(1, 1, 3, 3).astype(np.float32)]
    inputs = Input(*nps)
    Output(
        ConvTranspose(inputs[0],
                      np.random.randn(1, 2, 3, 3).astype(np.float32),
                      strides=[3, 2],
                      pads=[1, 2, 1, 2]))
    self._run(list(zip(inputs, nps)))

  def test_conv_transpose_dilations(self):
    reset_model(13)
    nps = [np.random.randn(1, 1, 3, 3).astype(np.float32)]
    inputs = Input(*nps)
    Output(
        ConvTranspose(
            inputs[0],
            np.random.randn(1, 1, 2, 2).astype(np.float32),
            dilations=[2, 2],
        ))
    self._run(list(zip(inputs, nps)))

  def test_conv_transpose_attributes(self):
    reset_model(13)
    nps = [np.random.randn(1, 1, 3, 3).astype(np.float32)]
    inputs = Input(*nps)
    Output(
        ConvTranspose(
            inputs[0],
            np.random.randn(1, 2, 3, 3).astype(np.float32),
            strides=[3, 2],
            output_shape=[10, 8],
            output_padding=[1, 1],
        ))
    self._run(list(zip(inputs, nps)))

  def test_cos(self):
    reset_model(13)
    nps = [np.random.randn(5).astype(np.float32)]
    inputs = Input(*nps)
    Output(Cos(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_cosh(self):
    reset_model(13)
    nps = [np.random.randn(5).astype(np.float32)]
    inputs = Input(*nps)
    Output(Cosh(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_flatten(self):
    reset_model(13)
    nps = [np.random.randn(1, 3, 3, 3).astype(np.float32)]
    inputs = Input(*nps)
    Output(Flatten(inputs))
    self._run(list(zip(inputs, nps)))

  def test_gather(self):
    reset_model(13)
    nps = [
        np.random.randn(6, 8, 3).astype(np.float32),
        np.random.randn(*[3, 2]).astype(np.int64),
    ]
    inputs = Input(*nps)
    Output(Gather(*inputs, axis=0))
    self._run(list(zip(inputs, nps)))

  def test_gather_axis_1(self):
    reset_model(13)
    nps = [
        np.random.randn(6, 3, 3).astype(np.float32),
        np.array([[0, 2], [0, 1], [2, 0]]).astype(np.int64),
    ]
    inputs = Input(*nps)
    Output(Gather(*inputs, axis=1))
    self._run(list(zip(inputs, nps)))

  def test_gather_nd(self):
    reset_model(13)
    nps = [
        np.random.randn(6, 8, 3).astype(np.float32),
        np.random.randn(*[3, 2]).astype(np.int64),
    ]
    inputs = Input(*nps)
    Output(GatherND(*inputs, batch_dims=0))
    self._run(list(zip(inputs, nps)))

  @pytest.mark.skip(reason="Not implemented for batch_dims != 0")
  def test_gather_nd_batch_dims_1(self):
    reset_model(13)
    nps = [
        np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
        np.array([[1], [0]]).astype(np.int64),
    ]
    inputs = Input(*nps)
    Output(GatherND(*inputs, batch_dims=1))
    self._run(list(zip(inputs, nps)))

  def test_gemm(self):
    reset_model(13)
    nps = [
        np.random.randn(2, 3).astype(np.float32),
        np.random.randn(3, 4).astype(np.float32),
        np.random.randn(2, 4).astype(np.float32),
    ]
    inputs = Input(*nps)
    Output(Gemm(*inputs))
    self._run(list(zip(inputs, nps)))

  def test_global_average_pool(self):
    reset_model(13)
    nps = [np.random.randn(2, 3, 4).astype(np.float32)]
    inputs = Input(*nps)
    Output(GlobalAveragePool(*inputs))
    self._run(list(zip(inputs, nps)))

  def test_mat_mul(self):
    reset_model(13)
    nps = [
        np.random.randn(5, 2, 3).astype(np.float32),
        np.random.randn(5, 3, 2).astype(np.float32)
    ]
    inputs = Input(*nps)
    Output(MatMul(*inputs))
    self._run(list(zip(inputs, nps)))

  def test_max(self):
    reset_model(13)
    nps = [
        np.random.randn(1, 10).astype(np.float32),
        np.random.randn(2, 10).astype(np.float32)
    ]
    inputs = Input(*nps)
    Output(Max(inputs))
    self._run(list(zip(inputs, nps)))

  def test_max_pool(self):
    reset_model(13)
    nps = [np.random.randn(1, 1, 5, 5).astype(np.float32)]
    inputs = Input(*nps)
    Output(MaxPool(inputs, kernel_shape=(3, 3), pads=(0, 0, 1, 1)))
    self._run(list(zip(inputs, nps)))

  def test_mul(self):
    reset_model(13)
    nps = [
        np.random.randn(1, 2, 3).astype(np.float32),
        np.random.randn(1, 2, 3).astype(np.float32)
    ]
    inputs = Input(*nps)
    Output(Mul(*inputs))
    self._run(list(zip(inputs, nps)))

  def test_pad(self):
    reset_model(13)
    nps = [
        np.random.randn(1, 2, 3).astype(np.float32),
    ]
    inputs = Input(*nps)
    Output(Pad(*inputs, np.array([0, 0, 1, 0, 0, 2])))
    self._run(list(zip(inputs, nps)))

  def test_pad_5D(self):
    reset_model(13)
    nps = [
        np.random.randn(1, 2, 3, 4, 5).astype(np.float32),
    ]
    inputs = Input(*nps)
    Output(
        Pad(*inputs, np.array([0, 0, 1, 2, 3, 0, 0, 2, 3, 4]),
            np.array([1.0]).astype(np.float32)))
    self._run(list(zip(inputs, nps)))

  def test_pad_reflect(self):
    reset_model(13)
    nps = [
        np.random.randn(1, 2, 3, 4).astype(np.float32),
    ]
    inputs = Input(*nps)
    Output(
        Pad(*inputs,
            np.array([0, 0, 1, 2, 0, 0, 2, 3]),
            np.array([1.0]).astype(np.float32),
            mode="reflect"))
    self._run(list(zip(inputs, nps)))

  def test_reciprocal(self):
    reset_model(13)
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(Reciprocal(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_reduce_prod(self):
    reset_model(13)
    nps = [np.random.randn(1, 2, 3).astype(np.float32)]
    inputs = Input(*nps)
    Output(ReduceProd(inputs[0], axes=np.array((1, 2)).astype(np.int64)))
    self._run(list(zip(inputs, nps)))

  def test_reduce_sum(self):
    reset_model(13)
    nps = [np.random.randn(1, 2, 3).astype(np.float32)]
    inputs = Input(*nps)
    Output(ReduceSum(inputs[0], np.array((1, 2)).astype(np.int64)))
    self._run(list(zip(inputs, nps)))

  def test_relu(self):
    reset_model(13)
    nps = [np.random.randn(1, 5).astype(np.float32)]
    inputs = Input(*nps)
    Output(Relu(inputs))
    self._run(list(zip(inputs, nps)))

  def test_reshape(self):
    reset_model(13)
    nps = [np.random.randn(4,).astype(np.float32)]
    inputs = Input(*nps)
    Output(Reshape(inputs[0], np.array((2, 2)).astype(np.int64)))
    self._run(list(zip(inputs, nps)))

  def test_resize_scales_nearest(self):
    reset_model(13)
    nps = [
        np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]]], dtype=np.float32),
        np.array([], dtype=np.float32),
    ]
    inputs = Input(*nps)
    Output(
        Resize(
            *inputs,
            np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32),
            mode="nearest",
        ))
    self._run(list(zip(inputs, nps)))

  def test_resize_downsample_sizes_linear_pytorch_half_pixel(self):
    reset_model(13)
    nps = [
        np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]]],
                 dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
    ]
    inputs = Input(*nps)
    Output(
        Resize(*inputs,
               np.array([1, 1, 3, 1], dtype=np.int64),
               mode='linear',
               coordinate_transformation_mode='pytorch_half_pixel'))
    self._run(list(zip(inputs, nps)))

  def test_resize_pt_nearest(self):
    reset_model(13)
    nps = [
        np.array([[[[1., 2., 0.], [3., 4., 0.], [0., 0., 0.]]]],
                 dtype=np.float32),
        np.array([], dtype=np.float32),
    ]
    inputs = Input(*nps)
    Output(
        Resize(
            *inputs,
            np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
            mode="nearest",
        ))
    self._run(list(zip(inputs, nps)))

  def test_resize_pt_bilinear(self):
    reset_model(13)
    nps = [
        np.array([[[[1., 2., 0.], [3., 4., 0.], [0., 0., 0.]]]],
                 dtype=np.float32),
        np.array([], dtype=np.float32),
    ]
    inputs = Input(*nps)
    Output(
        Resize(
            *inputs,
            np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
            mode="linear",
        ))
    self._run(list(zip(inputs, nps)))

  def test_resize_pt_bilinear_align_corners(self):
    reset_model(13)
    nps = [
        np.array([[[[1., 2., 0.], [3., 4., 0.], [0., 0., 0.]]]],
                 dtype=np.float32),
        np.array([], dtype=np.float32),
    ]
    inputs = Input(*nps)
    Output(
        Resize(
            *inputs,
            np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
            mode="linear",
            coordinate_transformation_mode="align_corners",
        ))
    self._run(list(zip(inputs, nps)))

  def test_shape(self):
    reset_model(13)
    nps = [np.random.randn(1, 2, 3, 4).astype(np.float32)]
    inputs = Input(*nps)
    Output(Shape(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_sigmoid(self):
    reset_model(13)
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(Sigmoid(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_slice(self):
    reset_model(13)
    nps = [np.random.randn(5, 5).astype(np.float32)]
    inputs = Input(*nps)
    Output(
        Slice(inputs[0],
              np.array((2, 3)).astype(np.int64),
              np.array((4, 5)).astype(np.int64)))
    self._run(list(zip(inputs, nps)))

  def test_softmax(self):
    reset_model(13)
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(Softmax(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_split(self):
    reset_model(13)
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(Split(inputs, split=np.array([2, 8]), axis=1))
    self._run(list(zip(inputs, nps)))

  def test_sqrt(self):
    reset_model(13)
    nps = [np.random.randn(1, 10).astype(np.float32)]
    inputs = Input(*nps)
    Output(Sqrt(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_squeeze(self):
    reset_model(13)
    nps = [np.random.randn(1, 10, 1, 1).astype(np.float32)]
    inputs = Input(*nps)
    Output(Squeeze(inputs[0], np.array(([2, 3]))))
    self._run(list(zip(inputs, nps)))

  def test_squeeze_no_axes(self):
    reset_model(13)
    nps = [np.random.randn(1, 10, 1, 1).astype(np.float32)]
    inputs = Input(*nps)
    Output(Squeeze(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_transpose(self):
    reset_model(13)
    nps = [np.random.randn(1, 2, 3, 4).astype(np.float32)]
    inputs = Input(*nps)
    Output(Transpose(inputs[0], perm=[0, 2, 3, 1]))
    self._run(list(zip(inputs, nps)))

  def test_transpose_no_perm(self):
    reset_model(13)
    nps = [np.random.randn(1, 2, 3, 4).astype(np.float32)]
    inputs = Input(*nps)
    Output(Transpose(inputs[0]))
    self._run(list(zip(inputs, nps)))

  def test_unsqueeze(self):
    reset_model(13)
    nps = [np.random.randn(1, 2).astype(np.float32)]
    inputs = Input(*nps)
    Output(Unsqueeze(inputs[0], np.array(([2, 3]))))
    self._run(list(zip(inputs, nps)))


if __name__ == '__main__':
  pytest.main(['-s', 'test_base.py'])
