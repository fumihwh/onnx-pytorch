import logging

import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class MaxPoolOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(MaxPoolOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)

    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2
    assert (d in (1, 2, 3))

    params = {
        "dilation": attr_value_dict.get("dilations", 1),
        "kernel_size": attr_value_dict["kernel_shape"][:].__repr__(),
        "ceil_mode": bool(attr_value_dict["ceil_mode"]),
        "stride": attr_value_dict.get("strides", 1),
        "return_indices": len(node.output) == 2
    }

    nn_name = f"MaxPool{d}d"
    node_name = self.rename_helper.get_node_name(node.name, node.op_type)
    init_str, forward_str = [], []
    if "pads" in attr_value_dict:
      padding = []
      pt_padding = []
      for i in range(d):
        if attr_value_dict['pads'][i] == attr_value_dict['pads'][
            i + d] and pt_padding is not None:
          pt_padding.append(attr_value_dict['pads'][i])
        else:
          pt_padding = None
        padding.insert(0, attr_value_dict['pads'][i + d])
        padding.insert(0, attr_value_dict['pads'][i])
      if pt_padding is None:
        logging.warning(
            "MaxPool with asymmetric padding will get incorrect indices.")
        forward_str.append(
            f"{inputs_str[0]} = F.pad({inputs_str[0]}, {padding.__repr__()}, value=float('-inf'))"
        )
      else:
        params["padding"] = pt_padding.__repr__()
    params_str = self.gen_params_str(**params)
    init_str.append(f"self.{node_name} = nn.{nn_name}(**{{{params_str}}})")
    forward_str.append(
        f"{', '.join(outputs_str)} = self.{node_name}({inputs_str[0]})")
    return {"init": init_str, "forward": forward_str}
