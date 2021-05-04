import onnx
import onnx.numpy_helper
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class BatchNormalizationOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(BatchNormalizationOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(node, initializers)

    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2
    assert (d in (1, 2, 3))

    nn_name = f"BatchNorm{d}d"

    params_str = self.gen_params_str(num_features=onnx.numpy_helper.to_array(
        initializers[node.input[1]]).shape[0],
                                     eps=attr_value_dict["epsilon"],
                                     momentum=attr_value_dict["momentum"])

    init_str, forward_str = [], []
    init_str.append(f"self.{node.name} = nn.{nn_name}(**{{{params_str}}})")
    init_str.append(
        f"self.{node.name}.weight.data = self.__variables[\"{node.input[1]}\"]")
    init_str.append(
        f"self.{node.name}.bias.data = self.__variables[\"{node.input[2]}\"]")
    init_str.append(
        f"self.{node.name}.running_mean.data = self.__variables[\"{node.input[3]}\"]"
    )
    init_str.append(
        f"self.{node.name}.running_var.data = self.__variables[\"{node.input[4]}\"]"
    )
    forward_str.append(f"{outputs_str[0]} = self.{node.name}({inputs_str[0]})")
    return {"init": init_str, "forward": forward_str}
