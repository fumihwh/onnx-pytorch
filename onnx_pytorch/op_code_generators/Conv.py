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

    init_str, forward_str = [], []
    padding = 0
    strides = attr_value_dict.get("strides", 1)
    if "pads" in attr_value_dict:
      padding = [attr_value_dict["pads"][i] for i in range(d)]
    elif attr_value_dict["auto_pad"] not in (b"NOTSET", b""):
      logging.warning(
          "auto_pad is a DEPRECATED attribute, will not guarantee the result.")
      assert node.input[0] in value_infos and node.input[
          1] in value_infos and node.output[
              0] in value_infos, "Calculate pad shape need value infos for input, kernel, output."
      pt_padding = [0] * 2 * d
      pad_shape = [0] * d
      strides_spatial_shape = [strides] * d if strides == 1 else strides
      input_spatial_shape = self.get_shape(node.input[0], value_infos)[2:]
      kernel_spatial_shape = self.get_shape(node.input[1], value_infos)[2:]
      output_spatial_shape = self.get_shape(node.output[0], value_infos)[2:]
      for i in range(d):
        pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[
            i] + kernel_spatial_shape[i] - input_spatial_shape[i]
        mean = pad_shape[i] // 2
        if attr_value_dict["auto_pad"] == b"SAME_UPPER":
          l, r = pad_shape[i] - mean, mean
        else:
          l, r = mean, pad_shape[i] - mean
        pt_padding.insert(0, r)
        pt_padding.insert(0, l)
      forward_str.append(
          f"{inputs_str[0]} = F.pad({inputs_str[0]}, {pt_padding.__repr__()})")
      padding = 0

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

    nn_name = f"Conv{d}d"
    node_name = self.rename_helper.get_node_name(node.name, node.op_type)
    init_str.append(f"self.{node_name} = nn.{nn_name}(**{{{params_str}}})")
    init_str.append(f"self.{node_name}.weight.data = {inputs_str[1]}")
    if len(node.input) > 2:
      init_str.append(f"self.{node_name}.bias.data = {inputs_str[2]}")

    forward_str.append(f"{outputs_str[0]} = self.{node_name}({inputs_str[0]})")

    return {"init": init_str, "forward": forward_str}
