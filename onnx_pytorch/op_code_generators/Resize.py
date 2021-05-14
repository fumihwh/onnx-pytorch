import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class ResizeOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(ResizeOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers, rename_helper, tensor_inplace):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, rename_helper, tensor_inplace)
    scales, sizes = None, None
    if len(node.input) == 4:
      sizes = tuple(onnx.numpy_helper.to_array(initializers[node.input[3]])[2:])
    else:
      scales = tuple(
          onnx.numpy_helper.to_array(initializers[node.input[2]])[2:])

    align_corners = None
    if attr_value_dict["coordinate_transformation_mode"].decode(
    ) == "align_corners":
      align_corners = True
    mode = attr_value_dict['mode'].decode()
    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2
    if mode == "linear":
      if d == 1:
        pass
      elif d == 2:
        mode = "bilinear"
      elif d == 3:
        mode = "bicubic"
      elif d == 4:
        mode = "trilinear"
      else:
        raise NotImplementedError
    params_str = self.gen_params_str(
        size=sizes,
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
