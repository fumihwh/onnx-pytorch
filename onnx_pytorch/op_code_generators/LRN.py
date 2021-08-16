import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class LRNOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(LRNOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    params_str = self.gen_params_str(alpha=attr_value_dict["alpha"],
                                     beta=attr_value_dict["beta"],
                                     k=attr_value_dict["bias"],
                                     size=attr_value_dict["size"])
    forward_str.append(
        f"{outputs_str[0]} = F.local_response_norm({inputs_str[0]}, **{{{params_str}}})"
    )
    return {"init": init_str, "forward": forward_str}
