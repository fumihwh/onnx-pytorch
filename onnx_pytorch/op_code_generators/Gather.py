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
        f'''shape_{inputs_str[0]}, shape_{inputs_str[1]} = list({inputs_str[0]}.shape), list({inputs_str[1]}.shape)
    {inputs_str[1]} = {inputs_str[1]}.flatten()
    for r in range(0, {axis}):
      {inputs_str[1]} = {inputs_str[1]}.unsqueeze(0)
    for r in range({axis}, len(shape_{inputs_str[0]}) - 1):
      {inputs_str[1]} = {inputs_str[1]}.unsqueeze(-1)
    {inputs_str[1]} = {inputs_str[1]}.expand(*(shape_{inputs_str[0]}[:{axis}] + [np.prod(shape_{inputs_str[1]})] + shape_{inputs_str[0]}[{axis} + 1:]))
    {inputs_str[1]} = torch.where({inputs_str[1]} >= 0, {inputs_str[1]}, {inputs_str[1]} + shape_{inputs_str[0]}[{axis}])
    {outputs_str[0]} = torch.gather({inputs_str[0]}, {axis}, {inputs_str[1]})
    {outputs_str[0]} = torch.reshape({inputs_str[0]}, shape_{inputs_str[0]}[:{axis}] + shape_{inputs_str[1]} + shape_{inputs_str[0]}[{axis} + 1:])
''')
    return {"init": init_str, "forward": forward_str}
