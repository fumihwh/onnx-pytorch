import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class RoiAlignOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(RoiAlignOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    params_str = self.gen_params_str(
        output_size=(attr_value_dict["output_height"],
                     attr_value_dict["output_width"]),
        sampling_ratio=attr_value_dict["sampling_ratio"],
        spatial_scale=attr_value_dict["spatial_scale"],
    )
    forward_str.append(
        f"boxes = torch.cat((torch.unsqueeze({inputs_str[2]}, 1), {inputs_str[1]}), axis=1)"
    )
    forward_str.append(
        f"{outputs_str[0]} = torchvision.ops.roi_align({inputs_str[0]}, boxes, **{{{params_str}}})"
    )
    return {"init": init_str, "forward": forward_str}
