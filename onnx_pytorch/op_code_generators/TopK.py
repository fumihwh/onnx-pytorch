import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class TopKOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(TopKOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []

    params_str = self.gen_params_str(
        dim=attr_value_dict.get("axis", -1),
        largest=bool(attr_value_dict.get("largest", 1)),
        sorted=bool(attr_value_dict.get("sorted", 1)))
    inputs_str[1] = f"int({inputs_str[1]})"
    forward_str.append(
        f"{', '.join(outputs_str)} = torch.topk({', '.join(inputs_str)}, **{{{params_str}}})"
    )
    return {"init": init_str, "forward": forward_str}
