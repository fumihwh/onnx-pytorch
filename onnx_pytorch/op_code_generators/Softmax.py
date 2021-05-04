import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class SoftmaxOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(SoftmaxOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(node, initializers)
    init_str, forward_str = [], []

    params_str = self.gen_params_str(dim=attr_value_dict["axis"])
    init_str.append(f"self.{node.name} = nn.{self.onnx_op}(**{{{params_str}}})")
    forward_str.append(f"{outputs_str[0]} = self.{node.name}({inputs_str[0]})")
    return {"init": init_str, "forward": forward_str}
