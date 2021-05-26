import logging

import onnx
import onnx.numpy_helper
from onnx.numpy_helper import to_array
import torch

import glob
import os

modules = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
__all__ = [
    os.path.basename(f)[:-3]
    for f in modules
    if os.path.isfile(f) and not f.endswith('__init__.py')
] + ["get_op_code_generator"]


class OpCodeGenerator:

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    self.onnx_ver = onnx_ver
    self.torch_ver = torch_ver
    self.onnx_op = self.__class__.__name__.replace("OpCodeGenerator", "")
    self.schema = onnx.defs.get_schema(self.onnx_op,
                                       max_inclusive_version=onnx_ver)

    # Should inherit from ModelCodeGenerator
    self.rename_helper = None
    self.tensor_inplace = None

    if self.schema is not None:
      self.attr_default = {}
      for a, i in self.schema.attributes.items():
        try:
          default_value = onnx.helper.get_attribute_value(i.default_value)
          self.attr_default[a] = default_value
        except Exception as e:
          logging.warning(
              f"Cannot get default value for {a} of {self.onnx_op}.")

  def gen(self, node, value_infos, initializers):
    raise Exception

  def get_attr_value_dict(self, node):
    attr_value_dict = {}
    for a in node.attribute:
      attr_value_dict[a.name] = onnx.helper.get_attribute_value(a)
    attr_value_dict = dict(
        list(self.attr_default.items()) + list(attr_value_dict.items()))
    return attr_value_dict

  def gen_input_output_string(self,
                              node,
                              initializers,
                              rename_helper,
                              tensor_inplace=False,
                              input_num=None,
                              output_num=None):
    inputs_str, outputs_str = [], []
    input_num, output_num = input_num or len(node.input), output_num or len(
        node.output)
    for idx, (num, f, ls) in enumerate(
        ((input_num, node.input, inputs_str), (output_num, node.output,
                                               outputs_str))):
      for i in range(num):
        # tensor_inplace condition:
        # idx == 1: output
        # i == 0: first output tensor (Currently only support first tensor inplace)
        # node.input[0] not in initializers: Could not inplace initializer
        # rename_helper.tensor_name_counter[f[i]] == 2: output tensor 0 should only be counted twice
        # rename_helper.tensor_name_counter[node.input[0]] == 2: input tensor 0 should only be counted twice
        if idx == 1 \
            and i == 0 \
            and tensor_inplace \
            and len(node.input) > 0 \
            and node.input[0] not in initializers \
            and rename_helper.tensor_name_counter[f[i]] == 2 \
            and rename_helper.tensor_name_counter[node.input[0]] == 2:
          tensor_name = node.input[0]
          rename_helper.tensor_name_mapping[
              f[i]] = rename_helper.get_tensor_name(tensor_name)
        else:
          tensor_name = f[i]
        formatter = "{}"
        if tensor_name in initializers:
          formatter = "self._vars[\"{}\"]"
        s = formatter.format(rename_helper.get_tensor_name(tensor_name))
        ls.append(s)

    return inputs_str, outputs_str

  def gen_params_str(self, **kwargs):
    params = []
    for k, v in kwargs.items():
      v_str = v if type(v) == str else v.__repr__()
      params.append(f"'{k}': {v_str}")
    return ', '.join(params).__repr__()[1:-1]

  def check_in_init(self, targets, initializers):
    lacks = []
    rs = [None] * len(targets)
    for i, (t, n) in enumerate(targets):
      init = initializers.get(n, None)
      if init is None:
        lacks.append(n)
      rs[i] = init
    if lacks:
      raise Exception(
          f"Currently {self.__class__} only support all of {lacks.__repr__()} is in initializers."
      )
    return rs

  def get_shape(self, value, value_infos):
    if value not in value_infos:
      return None
    shape = []
    for d in value_infos[value].type.tensor_type.shape.dim:
      if d.dim_param != "":
        shape.append(-1)
      else:
        shape.append(d.dim_value)
    return shape


class ReduceOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(ReduceOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def _get_dim(self, attr_value_dict, d, node, initializers):
    if "axes" in attr_value_dict:
      dim = attr_value_dict["axes"]
    else:
      dim = list(range(d))
      if len(node.input) > 1:
        dim = initializers.get(node.input[1], None)
        assert dim is not None, "Currently ReduceOpCodeGenerator only support all of [axes] is in initializers."
        dim = list(to_array(dim))
    return dim


__op_gen_dict = {}


def get_op_code_generator(op, **kwargs):
  op_code_gen_name = "{}OpCodeGenerator".format(op)
  if op_code_gen_name in __op_gen_dict:
    return __op_gen_dict[op_code_gen_name]
  mod = globals().get(op, None)
  if mod is None:
    return None
  __op_gen_dict[op_code_gen_name] = getattr(mod, op_code_gen_name)(**kwargs)
  return __op_gen_dict[op_code_gen_name]


def clear_op_code_generator():
  __op_gen_dict = {}
