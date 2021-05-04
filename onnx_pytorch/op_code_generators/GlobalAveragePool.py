import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class GlobalAveragePoolOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(GlobalAveragePoolOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    inputs_str, outputs_str = self.gen_input_output_string(node, initializers)
    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2
    params_str = self.gen_params_str(
        kernel_size=f"self.{node.input[0]}.shape[{d}:]")

    nn_name = f"functional.avg_pool{d}d"
    init_str, forward_str = [], []
    init_str.append(f"self.{node.name} = nn.{nn_name}")
    forward_str.append(
        f"{outputs_str[0]} = self.{node.name}({inputs_str[0]}, **{{{params_str}}})"
    )

    return {"init": init_str, "forward": forward_str}
