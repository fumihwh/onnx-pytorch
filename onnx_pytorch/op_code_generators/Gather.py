import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class GatherOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(GatherOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers, rename_helper, tensor_inplace):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, rename_helper, tensor_inplace)
    init_str, forward_str = [], []
    axis = attr_value_dict.get("axis", 0)
    forward_str.append(
        f"{outputs_str[0]} = {inputs_str[0]}.__getitem__([slice(None) for _ in range({axis})] + [{inputs_str[1]}.to(torch.int64)])"
    )
    return {"init": init_str, "forward": forward_str}
