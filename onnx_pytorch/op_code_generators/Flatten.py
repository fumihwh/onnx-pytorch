import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class FlattenOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(FlattenOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    params_str = self.gen_params_str(start_dim=attr_value_dict["axis"])
    nn_name = self.onnx_op
    node_name = self.rename_helper.get_node_name(node.name, node.op_type)
    init_str, forward_str = [], []
    init_str.append(f"self.{node_name} = nn.{nn_name}(**{{{params_str}}})")
    forward_str.append(
        f"{', '.join(outputs_str)} = self.{node_name}({', '.join(inputs_str)})")

    return {"init": init_str, "forward": forward_str}
