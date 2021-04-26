import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class GemmOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(GemmOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(node, initializers)

    if attr_value_dict["transA"] == 1:
      inputs_str[0] = "torch.transpose({}, 0, 1)".format(inputs_str[0])
    if attr_value_dict["transB"] == 1:
      inputs_str[1] = "torch.transpose({}, 0, 1)".format(inputs_str[1])

    init_str, forward_str = [], []
    forward_str.append(
        f"{outputs_str[0]} = {attr_value_dict['alpha']} * torch.matmul({', '.join(inputs_str[:2])}) + {attr_value_dict['beta']} * {inputs_str[2]}"
    )

    return {"init": init_str, "forward": forward_str}
