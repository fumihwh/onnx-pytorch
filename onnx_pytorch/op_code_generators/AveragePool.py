import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class AveragePoolOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(AveragePoolOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)

    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2
    assert (d in (1, 2, 3))

    nn_name = f"AvgPool{d}d"
    node_name = self.rename_helper.get_node_name(node.name, node.op_type)
    init_str, forward_str = [], []
    slice_str = [":", ":"]

    param = {
        "kernel_size": attr_value_dict["kernel_shape"][:].__repr__(),
        "ceil_mode": bool(attr_value_dict["ceil_mode"]),
        "stride": attr_value_dict.get("strides", 1),
        "count_include_pad": bool(attr_value_dict.get("count_include_pad", 0))
    }
    if "pads" in attr_value_dict:
      padding = []
      for i in range(d):
        padding_size = max(attr_value_dict['pads'][i],
                           attr_value_dict['pads'][i + d])
        padding.append(padding_size)
        slice_begin = "" if padding_size == attr_value_dict['pads'][i] else str(
            padding_size - attr_value_dict['pads'][i])
        slice_end = "" if padding_size == attr_value_dict['pads'][
            i + d] else str(attr_value_dict['pads'][i + d] - padding_size)
        slice_str.append(":".join([slice_begin, slice_end]))
      param["padding"] = padding.__repr__()
    params_str = self.gen_params_str(**param,)
    init_str.append(f"self.{node_name} = nn.{nn_name}(**{{{params_str}}})")
    forward_str.append(
        f"{outputs_str[0]} = self.{node_name}({inputs_str[0]})[{', '.join(slice_str)}]"
    )

    return {"init": init_str, "forward": forward_str}
