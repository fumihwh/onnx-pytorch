import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class ClipOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(ClipOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    min = attr_value_dict.get("min", "float(\"-inf\")")
    max = attr_value_dict.get("max", "float(\"inf\")")
    if len(inputs_str) == 1:
      inputs_str.append(str(min))
    if len(inputs_str) < 3:
      inputs_str.append(str(max))
    init_str, forward_str = [], []
    forward_str.append(
        f"{outputs_str[0]} = torch.clip({', '.join(inputs_str)})")
    return {"init": init_str, "forward": forward_str}
