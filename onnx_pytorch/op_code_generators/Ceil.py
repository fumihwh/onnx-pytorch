import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class CeilOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(CeilOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers, rename_helper, tensor_inplace):
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, rename_helper, tensor_inplace)
    init_str, forward_str = [], []
    forward_str.append(
        f"{outputs_str[0]} = torch.ceil({', '.join(inputs_str)})")
    return {"init": init_str, "forward": forward_str}