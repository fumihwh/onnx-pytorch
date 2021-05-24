import onnx
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class CastOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(CastOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    forward_str.append(
        f"{outputs_str[0]} = {inputs_str[0]}.to(device={inputs_str[0]}.device, dtype=torch.{str(TENSOR_TYPE_TO_NP_TYPE[attr_value_dict['to']])}, copy=True)"
    )
    return {"init": init_str, "forward": forward_str}
