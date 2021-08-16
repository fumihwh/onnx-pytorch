import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class ReshapeOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(ReshapeOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    forward_str.append(
        f"{outputs_str[0]} = torch.reshape({inputs_str[0]}, [s if s != 0 else {inputs_str[0]}.shape[i] for i, s in enumerate({inputs_str[1]})])"
    )
    return {"init": init_str, "forward": forward_str}
