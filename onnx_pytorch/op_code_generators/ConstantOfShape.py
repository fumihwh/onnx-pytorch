import onnx
import torch
from onnx.numpy_helper import to_array

from onnx_pytorch.op_code_generators import OpCodeGenerator


class ConstantOfShapeOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(ConstantOfShapeOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    dtype = "float32"
    val = 0.
    if "value" in attr_value_dict:
      array = to_array(attr_value_dict["value"])
      dtype = array.dtype
      val = array[0]
    forward_str.append(
        f"{outputs_str[0]} = torch.Tensor().new_full(size={inputs_str[0]}.tolist(), fill_value={val}, dtype=torch.{dtype})"
    )
    return {"init": init_str, "forward": forward_str}
