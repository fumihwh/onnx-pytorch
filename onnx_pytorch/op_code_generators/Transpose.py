import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class TransposeOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(TransposeOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    if "perm" in attr_value_dict:
      forward_str.append(
          f"{outputs_str[0]} = {inputs_str[0]}.permute(*{attr_value_dict['perm'].__repr__()})"
      )
    else:
      forward_str.append(f"{outputs_str[0]} = {inputs_str[0]}.T")
    return {"init": init_str, "forward": forward_str}
