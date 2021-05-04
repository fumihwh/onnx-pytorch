import argparse
import logging
import os
import re
import shutil

import numpy as np
import onnx
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx.numpy_helper import to_array
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from onnx_pytorch.code_gen_template import CodeGenTemplate
from onnx_pytorch.op_code_generators import *


class ModelCodeGenerator:

  def __init__(self, onnx_model=None, output_dir=None):
    self.onnx_model = onnx_model
    self.output_dir = output_dir

    self.init_parts = []
    self.forward_parts = []

  def add_init_part(self, m):
    if type(m) in (list, tuple, set):
      self.init_parts.extend(m)
    else:
      self.init_parts.append(m)

  def add_forward_part(self, m):
    if type(m) in (list, tuple, set):
      self.forward_parts.extend(m)
    else:
      self.forward_parts.append(m)

  def add_forward_return(self, outputs_value_infos):
    return_list = [f"self.{o.name}" for o in outputs_value_infos]
    self.forward_parts.append(f"return {', '.join(return_list)}")

  def add_forward_input(self, inputs_value_infos):
    initializer_names = {i.name for i in self.onnx_model.graph.initializer}
    return_list = [
        f"self.{i.name}" for i in inputs_value_infos
        if i.name not in initializer_names
    ]
    if len(return_list) == 1:
      self.forward_parts.append(f"{return_list[0]}, = inputs")
    else:
      self.forward_parts.append(f"{', '.join(return_list)} = inputs")

  def gen_model_code(self):
    return CodeGenTemplate.model(model_init='''
    '''.join(self.init_parts),
                                 model_forward='''
    '''.join(self.forward_parts),
                                 test_run_model=self.gen_test_run_model_code())

  def gen_test_run_model_code(self):
    numpy_input_str = []
    initializer_names = {i.name for i in self.onnx_model.graph.initializer}
    for i in self.onnx_model.graph.input:
      if i.name in initializer_names:
        continue
      dtype = TENSOR_TYPE_TO_NP_TYPE[i.type.tensor_type.elem_type]
      shape = []
      for d in i.type.tensor_type.shape.dim:
        if d.dim_param != "":
          shape.append(1)
        else:
          shape.append(d.dim_value)
      numpy_input_str.append(
          f"torch.from_numpy(np.random.randn(*{[s if s > 1 else 1 for s in shape].__repr__()}).astype(np.{dtype.name}))"
      )
    test_run_model = [
        f'''@torch.no_grad()
def test_run_model(inputs=[{', '.join(numpy_input_str)}]):''',
        "model = Model()", "model.eval()", "print(model)"
    ]
    test_run_model.extend(["rs = model(*inputs)", "print(rs)", "return rs"])
    return '''
  '''.join(test_run_model)

  def preprocess_onnx_model(self):
    self.onnx_model.graph.ClearField("value_info")
    for n in self.onnx_model.graph.node:
      inputs, outputs = [], []
      for ls, f in ((inputs, n.input), (outputs, n.output)):
        for i in f:
          if i.isnumeric():
            ls.append(f"__t_{i}")
          else:
            ls.append(re.sub("[:/.]", "_", i))
          if i != ls[-1]:
            logging.warning(f"Tensor name {i} is changed to {ls[-1]}.")

      n.ClearField("input")
      n.input.extend(inputs)
      n.ClearField("output")
      n.output.extend(outputs)

    for f in (self.onnx_model.graph.input, self.onnx_model.graph.output,
              self.onnx_model.graph.initializer, self.onnx_model.graph.node):
      for i in f:
        if i.name.isnumeric():
          i.name = f"__t_{i.name}"
        else:
          i.name = re.sub("[:/.]", "_", i.name)

    model = SymbolicShapeInference.infer_shapes(self.onnx_model, 2**31 - 1,
                                                True, True, 0)
    onnx.save(model, os.path.join(self.output_dir, "tmp_processed.onnx"))
    self.onnx_model = model

  def run(self):
    self.preprocess_onnx_model()
    initializers = {i.name: i for i in self.onnx_model.graph.initializer}
    input_value_infos = {i.name: i for i in self.onnx_model.graph.input}
    output_value_infos = {i.name: i for i in self.onnx_model.graph.output}
    value_infos = {}
    value_infos.update(input_value_infos)
    value_infos.update(output_value_infos)
    value_infos.update({i.name: i for i in self.onnx_model.graph.value_info})

    self.add_forward_input(self.onnx_model.graph.input)
    for n in self.onnx_model.graph.node:
      op_code_gen = get_op_code_generator(n.op_type)
      assert op_code_gen, "OpCodeGenerator is unimplemented for {}.".format(
          n.op_type)
      gened = op_code_gen.gen(n, value_infos, initializers)
      self.add_init_part(gened["init"])
      self.add_forward_part(gened["forward"])
    self.add_forward_return(self.onnx_model.graph.output)

    gened_code = self.gen_model_code()
    print(gened_code)
    with open(os.path.join(self.output_dir, "model.py"), "w") as f:
      f.write(gened_code)
    shutil.rmtree(os.path.join(self.output_dir, "variables"),
                  ignore_errors=True)
    os.makedirs(os.path.join(self.output_dir, "variables"))
    for k, v in initializers.items():
      np.save(os.path.join(self.output_dir, "variables", f"{k}.npy"),
              to_array(v))


def gen(onnx_model, output_dir, overwrite=False):
  kwargs = {"output_dir": output_dir}
  if type(onnx_model) == onnx.ModelProto:
    kwargs["onnx_model"] = onnx_model
  else:
    assert os.path.exists(
        onnx_model), f"ONNX model {onnx_model} does not exist."
    assert os.path.isfile(onnx_model), f"{onnx_model} is not a file."
    assert os.path.exists(
        output_dir
    ) and overwrite is not True, f"{output_dir} is not empty and overwrite is not True."
    assert os.path.isdir(output_dir), f"{output_dir} is not directory."
    kwargs["onnx_model"] = onnx.load(onnx_model)
  if overwrite:
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)
  ModelCodeGenerator(**kwargs).run()


def main():
  debug = True
  parser = argparse.ArgumentParser()
  parser.add_argument("--onnx_model_path",
                      default=None,
                      type=str,
                      required=not debug,
                      help="The onnx model path.")

  parser.add_argument("--output_dir",
                      default=None,
                      type=str,
                      required=not debug,
                      help="The output dir")

  parser.add_argument("--overwrite",
                      default=False,
                      type=bool,
                      help="Should overwrite the output dir.")
  args = parser.parse_args()

  gen(onnx_model=args.onnx_model_path,
      output_dir=args.output_dir,
      overwrite=args.overwrite)


if __name__ == '__main__':
  main()
