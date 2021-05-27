import logging

import onnx
import onnx.numpy_helper
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class ConvOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(ConvOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)

    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2
    assert (d in (1, 2, 3))

    nn_name = f"Conv{d}d"
    node_name = self.rename_helper.get_node_name(node.name, node.op_type)
    init_str, forward_str = [], []
    padding = 0
    if "pads" in attr_value_dict:
      padding = [attr_value_dict["pads"][i] for i in range(d)]
    elif attr_value_dict["auto_pad"] not in (b"NOTSET", b""):
      logging.warning(
          "auto_pad is a DEPRECATED attribute, will not guarantee the result.")
      forward_str.append(
          f"{inputs_str[0]} = self.compatible_auto_pad({inputs_str[0]}, self.{node_name}.weight.data.shape[2:], self.{node_name}, '{attr_value_dict['auto_pad'].decode('utf-8')}')"
      )
    weights = onnx.numpy_helper.to_array(initializers[node.input[1]])
    params_str = self.gen_params_str(
        groups=attr_value_dict["group"],
        dilation=attr_value_dict.get("dilations", 1),
        out_channels=weights.shape[0],
        padding=padding,
        kernel_size=weights.shape[2:].__repr__(),
        stride=attr_value_dict.get("strides", 1),
        in_channels=weights.shape[1] * attr_value_dict["group"],
        bias=len(node.input) > 2)

    init_str.append(f"self.{node_name} = nn.{nn_name}(**{{{params_str}}})")
    init_str.append(f"self.{node_name}.weight.data = {inputs_str[1]}")
    if len(node.input) > 2:
      init_str.append(f"self.{node_name}.bias.data = {inputs_str[2]}")

    forward_str.append(f"{outputs_str[0]} = self.{node_name}({inputs_str[0]})")

    return {"init": init_str, "forward": forward_str}

  @staticmethod
  def gen_method():
    return '''def compatible_auto_pad(self, input, kernel_spatial_shape, nn_mod, auto_pad=None, **kwargs):
    input_spatial_shape = input.shape[2:]
    d = len(input_spatial_shape)
    strides = nn_mod.stride
    dilations = nn_mod.dilation
    output_spatial_shape = [math.ceil(float(l) / float(r)) for l, r in zip(input.shape[2:], strides)]
    pt_padding = [0] * 2 * d
    pad_shape = [0] * d
    for i in range(d):
      pad_shape[i] = (output_spatial_shape[i] - 1) * strides[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
      mean = pad_shape[i] // 2
      if auto_pad == b"SAME_UPPER":
        l, r = pad_shape[i] - mean, mean
      else:
        l, r = mean, pad_shape[i] - mean
      pt_padding.insert(0, r)
      pt_padding.insert(0, l)
    return F.pad(input, pt_padding)
'''
