import onnx
import torch
from onnx.numpy_helper import to_array

from onnx_pytorch.op_code_generators import OpCodeGenerator


class SliceOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(SliceOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper)
    init_str, forward_str = [], []
    d = len(value_infos[node.input[0]].type.tensor_type.shape.dim)
    starts, ends, axes, steps = self._get_starts_ends_axes_steps(
        attr_value_dict, d, node, initializers)
    slice_str = []
    for i in range(d):
      if i in axes:
        j = axes.index(i)
        s = ["", ""]
        if type(starts) == str and type(ends) == str:
          s[0] = f'{starts}[{j}] if {starts}[{j}]'
          s[1] = f'{ends}[{j}] if {ends}[{j}]'
        else:
          s = [
              str(starts[j]) if starts[j] != 0 else "",
              str(ends[j]) if ends[j] < 2**31 else ""
          ]
        if steps[j] != 1:
          s.append(str(steps[j]))
        slice_str.append(":".join(s))
      else:
        slice_str.append(":")

    forward_str.append(
        f"{outputs_str[0]} = {inputs_str[0]}[{', '.join(slice_str)}]")
    return {"init": init_str, "forward": forward_str}

  def _get_starts_ends_axes_steps(self, attr_value_dict, d, node, initializers):
    axes = list(range(d))
    steps = [1] * len(axes)
    if self.onnx_ver > 1 and len(node.input) > 1:
      starts = initializers.get(node.input[1], None)
      ends = initializers.get(node.input[2], None)
      if starts is None:
        starts = node.input[1]
      else:
        starts = to_array(starts)
      if ends is None:
        ends = node.input[2]
      else:
        ends = to_array(ends)
      if len(node.input) > 3:
        axes = initializers.get(node.input[3], None)
      if len(node.input) > 4:
        steps = initializers.get(node.input[4], None)
      assert starts is not None or ends is not None or axes is not None or steps is not None, "Currently SliceOpCodeGenerator only support all of [starts, ends, axes, steps] is in initializers."
      if len(node.input) > 3:
        axes = to_array(axes)
      if len(node.input) > 4:
        steps = to_array(steps)
    else:
      starts = attr_value_dict["starts"]
      ends = attr_value_dict["ends"]
      axes = attr_value_dict.get("axes", axes)
    return starts, ends, list(axes), list(steps)
