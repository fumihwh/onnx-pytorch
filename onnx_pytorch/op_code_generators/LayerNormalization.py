import onnx
import onnx.numpy_helper
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class LayerNormalizationOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(LayerNormalizationOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)

    axis = attr_value_dict["axis"]

    nn_name = f"LayerNorm"
    node_name = self.rename_helper.get_node_name(node.name, node.op_type)

    params_str = self.gen_params_str(
        normalized_shape=onnx.numpy_helper.to_array(
            initializers[node.input[1]]).shape,
        eps=attr_value_dict["epsilon"])

    init_str, forward_str = [], []
    init_str.append(f"self.{node_name} = nn.{nn_name}(**{{{params_str}}})")
    init_str.append(f"self.{node_name}.weight.data = {inputs_str[1]}")
    init_str.append(f"self.{node_name}.bias.data = {inputs_str[2]}")
    curr_input = inputs_str[0]

    forward_str.append(f"{outputs_str[0]} = self.{node_name}({curr_input})")

    return {"init": init_str, "forward": forward_str}
