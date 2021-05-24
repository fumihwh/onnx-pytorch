import onnx
import torch
from onnx.numpy_helper import to_array

from onnx_pytorch.op_code_generators import OpCodeGenerator


class DropoutOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(DropoutOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    check_list = []
    ratio = attr_value_dict.get("ratio", 0.5)
    training_mode = attr_value_dict.get("training_mode", False)
    if len(node.input) > 1:
      check_list.append((node.input[1], "ratio"))
    if len(node.input) > 2:
      check_list.append((node.input[2], "training_mode"))
    inits = self.check_in_init(check_list, initializers)
    if len(node.input) > 1:
      ratio = to_array(inits[0])[0]
    if len(node.input) > 2:
      training_mode = bool(to_array(inits[1])[0])
    params_str = self.gen_params_str(p=ratio)
    nn_name = "Dropout"
    node_name = self.rename_helper.get_node_name(node.name, node.op_type)
    init_str.append(f"self.{node_name} = nn.{nn_name}(**{{{params_str}}})")
    forward_str.append(f"{outputs_str[0]} = self.{node_name}({inputs_str[0]})")
    return {"init": init_str, "forward": forward_str}
