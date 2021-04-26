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
        node, initializers, 1)

    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2
    assert (d in (1, 2, 3))

    padding = 0
    if "pads" in attr_value_dict:
      padding = [attr_value_dict["pads"][i] for i in range(d)]

    weights = onnx.numpy_helper.to_array(initializers[node.input[1]])
    params_str = self.gen_params_str(groups=attr_value_dict["group"],
                                     dilation=attr_value_dict.get(
                                         "dilations", 1),
                                     out_channels=weights.shape[0],
                                     padding=padding,
                                     kernel_size=weights.shape[2:].__repr__(),
                                     stride=attr_value_dict.get("strides", 1),
                                     in_channels=weights.shape[1],
                                     bias=len(node.input) > 2)

    nn_name = f"Conv{d}d"
    init_str, forward_str = [], []
    init_str.append(
        f"self.{node.name} = nn.{nn_name}(**{{{params_str}}})")
    init_str.append(
        f"self.{node.name}.weight.data = self.__variables[\"{node.input[1]}\"]")
    if len(node.input) > 2:
      init_str.append(
          f"self.{node.name}.bias.data = self.__variables[\"{node.input[2]}\"]")

    forward_str.append(f"{outputs_str[0]} = self.{node.name}({inputs_str[0]})")

    return {"init": init_str, "forward": forward_str}
