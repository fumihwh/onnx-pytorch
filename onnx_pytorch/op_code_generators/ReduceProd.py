import onnx
import torch

from onnx_pytorch.op_code_generators import ReduceOpCodeGenerator


class ReduceProdOpCodeGenerator(ReduceOpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(ReduceProdOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim)
    dim = self._get_dim(attr_value_dict, d, node, initializers)
    params_str = self.gen_params_str(keepdim=bool(attr_value_dict["keepdims"]))
    curr_input = inputs_str[0]
    for d in reversed(dim):
      forward_str.append(
          f"{outputs_str[0]} = torch.prod({curr_input}, {d}, **{{{params_str}}})"
      )
      curr_input = outputs_str[0]
    return {"init": init_str, "forward": forward_str}
