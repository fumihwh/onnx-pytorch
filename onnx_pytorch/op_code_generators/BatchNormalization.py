import onnx
import onnx.numpy_helper
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class BatchNormalizationOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(BatchNormalizationOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers, rename_helper, tensor_inplace):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, rename_helper, tensor_inplace)

    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2
    assert (d in (1, 2, 3))

    nn_name = f"BatchNorm{d}d"
    node_name = rename_helper.get_node_name(node.name, node.op_type)

    params_str = self.gen_params_str(num_features=onnx.numpy_helper.to_array(
        initializers[node.input[1]]).shape[0],
                                     eps=attr_value_dict["epsilon"],
                                     momentum=attr_value_dict["momentum"])

    init_str, forward_str = [], []
    init_str.append(f"self.{node_name} = nn.{nn_name}(**{{{params_str}}})")
    init_str.append(f"self.{node_name}.weight.data = {inputs_str[1]}")
    init_str.append(f"self.{node_name}.bias.data = {inputs_str[2]}")
    init_str.append(f"self.{node_name}.running_mean.data = {inputs_str[3]}")
    init_str.append(f"self.{node_name}.running_var.data = {inputs_str[4]}")
    forward_str.append(f"{outputs_str[0]} = self.{node_name}({inputs_str[0]})")
    return {"init": init_str, "forward": forward_str}
