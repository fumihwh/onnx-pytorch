import onnx
import torch
from onnx.numpy_helper import to_array

from onnx_pytorch.op_code_generators import OpCodeGenerator


class ConstantOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(ConstantOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    if "value" in attr_value_dict:
      initializers[node.output[0]] = attr_value_dict["value"]
    else:
      raise NotImplementedError
    forward_str.append(f"{outputs_str[0]} = self._vars['{outputs_str[0]}']")
    return {"init": init_str, "forward": forward_str}
