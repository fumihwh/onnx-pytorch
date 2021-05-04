from uuid import uuid4

import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class ReshapeOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(ReshapeOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    inputs_str, outputs_str = self.gen_input_output_string(node, initializers)
    init_str, forward_str = [], []
    suffix = uuid4().hex[:4]
    forward_str.append(
        f"shape_{suffix} = [s if s != 0 else {inputs_str[0]}.shape[0] for s in {inputs_str[1]}.detach().numpy()]"
    )
    forward_str.append(
        f"{outputs_str[0]} = torch.reshape({inputs_str[0]}, shape_{suffix})")
    return {"init": init_str, "forward": forward_str}
