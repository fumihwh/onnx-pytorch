import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class MaxPoolOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(MaxPoolOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers, rename_helper, tensor_inplace):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, rename_helper, tensor_inplace)

    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2
    assert (d in (1, 2, 3))

    params_str = self.gen_params_str(
        dilation=attr_value_dict.get("dilations", 1),
        kernel_size=attr_value_dict["kernel_shape"][:].__repr__(),
        ceil_mode=bool(attr_value_dict["ceil_mode"]),
        stride=attr_value_dict.get("strides", 1),
        return_indices=(len(node.output) == 2))

    nn_name = f"MaxPool{d}d"
    node_name = rename_helper.get_node_name(node.name, node.op_type)
    init_str, forward_str = [], []
    init_str.append(f"self.{node_name} = nn.{nn_name}(**{{{params_str}}})")
    if "pads" in attr_value_dict:
      padding = []
      for i in range(d):
        padding.insert(0, attr_value_dict['pads'][i + d])
        padding.insert(0, attr_value_dict['pads'][i])
      forward_str.append(
          f"{inputs_str[0]} = F.pad({inputs_str[0]}, {padding.__repr__()}, value=float('-inf'))"
      )
    forward_str.append(f"{outputs_str[0]} = self.{node_name}({inputs_str[0]})")

    return {"init": init_str, "forward": forward_str}
