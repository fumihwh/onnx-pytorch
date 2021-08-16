import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class UpsampleOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(UpsampleOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    if node.input[1] in initializers:
      scales = tuple(
          onnx.numpy_helper.to_array(initializers[node.input[1]])[2:])
    else:
      scales = f"list({self.rename_helper.tensor_name_mapping.get(node.input[1], node.input[1])})[2:]"

    align_corners = None
    mode = attr_value_dict['mode'].decode()
    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2
    assert d < 4, "Currently temporal, spatial and volumetric sampling are supported."
    if mode == "linear":
      modes = ["linear", "bilinear", "trilinear"]
      mode = modes[d - 1]
    params_str = self.gen_params_str(
        scale_factor=scales,
        mode=f"'{mode}'",
        align_corners=align_corners,
        recompute_scale_factor=scales is not None,
    )
    init_str, forward_str = [], []

    forward_str.append(
        f"{outputs_str[0]} = F.interpolate({inputs_str[0]}, **{{{params_str}}})"
    )
    return {"init": init_str, "forward": forward_str}
