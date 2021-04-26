import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class MaxPoolOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(MaxPoolOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(node, initializers)

    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2
    assert (d in (1, 2, 3))

    params_str = self.gen_params_str(
        dilation=attr_value_dict.get("dilations", 1),
        kernel_size=attr_value_dict["kernel_shape"][:].__repr__(),
        ceil_mode=bool(attr_value_dict["ceil_mode"]),
        stride=attr_value_dict.get("strides", 1),
        return_indices=(len(node.output) == 2))

    nn_name = f"MaxPool{d}d"
    init_str, forward_str = [], []
    init_str.append(f"self.{node.name} = nn.{nn_name}(**{{{params_str}}})")
    if "pads" in attr_value_dict:
      forward_str.append(
          f"{inputs_str[0]} = torch.nn.functional.pad({inputs_str[0]}, {attr_value_dict['pads'].__repr__()})"
      )
    forward_str.append(f"{outputs_str[0]} = self.{node.name}({inputs_str[0]})")

    return {"init": init_str, "forward": forward_str}
