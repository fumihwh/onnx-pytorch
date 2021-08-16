import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class ArgMaxOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(ArgMaxOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    select_last_index = attr_value_dict.get("select_last_index", 0)
    assert select_last_index == 0, NotImplementedError
    params_str = self.gen_params_str(keepdim=bool(
        attr_value_dict.get("keepdims", 1)),
                                     axis=attr_value_dict.get("axis", 0))
    forward_str.append(
        f"{outputs_str[0]} = torch.argmax({', '.join(inputs_str)}, **{{{params_str}}})"
    )
    return {"init": init_str, "forward": forward_str}
