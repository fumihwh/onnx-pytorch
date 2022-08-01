# Autogenerated by onnx-model-maker. Don't modify it manually.

import onnx
import onnx.helper
import onnx.numpy_helper
from onnx_model_maker import omm
from onnx_model_maker import onnx_mm_export
from onnx_model_maker.ops.op_helper import _add_input


@onnx_mm_export("v17.STFT")
def STFT(signal, frame_step, window=None, frame_length=None, **kwargs):
  _inputs = []
  for i in (signal, frame_step, window, frame_length):
    _add_input(i, _inputs)

  idx = omm.op_counter["STFT"]
  omm.op_counter["STFT"] += 1
  node = onnx.helper.make_node("STFT",
                               _inputs, [f'_t_STFT_{idx}_output'],
                               name=f"STFT_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v17.MelWeightMatrix")
def MelWeightMatrix(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz, **kwargs):
  _inputs = []
  for i in (num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz):
    _add_input(i, _inputs)

  idx = omm.op_counter["MelWeightMatrix"]
  omm.op_counter["MelWeightMatrix"] += 1
  node = onnx.helper.make_node("MelWeightMatrix",
                               _inputs, [f'_t_MelWeightMatrix_{idx}_output'],
                               name=f"MelWeightMatrix_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v17.SequenceMap")
def SequenceMap(input_sequence, additional_inputs=None, **kwargs):
  _inputs = []
  for i in (input_sequence, additional_inputs):
    _add_input(i, _inputs)

  idx = omm.op_counter["SequenceMap"]
  omm.op_counter["SequenceMap"] += 1
  node = onnx.helper.make_node("SequenceMap",
                               _inputs, [f'_t_SequenceMap_{idx}_out_sequence'],
                               name=f"SequenceMap_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v17.HannWindow")
def HannWindow(size, **kwargs):
  _inputs = []
  for i in (size, ):
    _add_input(i, _inputs)

  idx = omm.op_counter["HannWindow"]
  omm.op_counter["HannWindow"] += 1
  node = onnx.helper.make_node("HannWindow",
                               _inputs, [f'_t_HannWindow_{idx}_output'],
                               name=f"HannWindow_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v17.BlackmanWindow")
def BlackmanWindow(size, **kwargs):
  _inputs = []
  for i in (size, ):
    _add_input(i, _inputs)

  idx = omm.op_counter["BlackmanWindow"]
  omm.op_counter["BlackmanWindow"] += 1
  node = onnx.helper.make_node("BlackmanWindow",
                               _inputs, [f'_t_BlackmanWindow_{idx}_output'],
                               name=f"BlackmanWindow_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v17.LayerNormalization")
def LayerNormalization(X, Scale, B=None, **kwargs):
  _inputs = []
  for i in (X, Scale, B):
    _add_input(i, _inputs)

  idx = omm.op_counter["LayerNormalization"]
  omm.op_counter["LayerNormalization"] += 1
  node = onnx.helper.make_node("LayerNormalization",
                               _inputs, [f'_t_LayerNormalization_{idx}_Y', f'_t_LayerNormalization_{idx}_Mean', f'_t_LayerNormalization_{idx}_InvStdDev'],
                               name=f"LayerNormalization_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v17.HammingWindow")
def HammingWindow(size, **kwargs):
  _inputs = []
  for i in (size, ):
    _add_input(i, _inputs)

  idx = omm.op_counter["HammingWindow"]
  omm.op_counter["HammingWindow"] += 1
  node = onnx.helper.make_node("HammingWindow",
                               _inputs, [f'_t_HammingWindow_{idx}_output'],
                               name=f"HammingWindow_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node


@onnx_mm_export("v17.DFT")
def DFT(input, dft_length=None, **kwargs):
  _inputs = []
  for i in (input, dft_length):
    _add_input(i, _inputs)

  idx = omm.op_counter["DFT"]
  omm.op_counter["DFT"] += 1
  node = onnx.helper.make_node("DFT",
                               _inputs, [f'_t_DFT_{idx}_output'],
                               name=f"DFT_{idx}",
                               **kwargs)
  onnx.checker.check_node(node, omm.ctx)
  omm.model.graph.node.append(node)
  return node