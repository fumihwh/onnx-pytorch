import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class IdentityOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(IdentityOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    nn_name = self.onnx_op
    node_name = self.rename_helper.get_node_name(node.name, node.op_type)
    init_str.append(f"self.{node_name} = nn.{nn_name}()")
    forward_str.append(f"{outputs_str[0]} = self.{node_name}({inputs_str[0]})")
    return {"init": init_str, "forward": forward_str}
