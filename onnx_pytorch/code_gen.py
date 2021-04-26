import argparse
import logging
import os
import re
import shutil

import numpy as np
import onnx
from onnx.numpy_helper import to_array
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
import torch
import torch.nn as nn

from onnx_pytorch.code_gen_template import CodeGenTemplate
from onnx_pytorch.op_code_generators import *

torch.set_printoptions(6)

nn_dict = {
    nn_name.lower(): getattr(nn, nn_name)
    for nn_name in dir(nn)
    if nn_name[0].isupper()
}

nn_name_dict = {
    nn_name.lower(): nn_name for nn_name in dir(nn) if nn_name[0].isupper()
}


class ModelCodeGenerator:

  def __init__(self, onnx_model_path=None, output_dir=None, overwrite=False):
    self.onnx_model_path = onnx_model_path
    assert os.path.exists(
        onnx_model_path), f"ONNX model {onnx_model_path} does not exist."
    assert os.path.isfile(onnx_model_path), f"{onnx_model_path} is not a file."

    self.output_dir = output_dir
    assert os.path.exists(
        output_dir
    ) and overwrite is not True, f"{output_dir} is not empty and overwrite is not True."
    assert os.path.isdir(output_dir), f"{output_dir} is not directory."
    if overwrite:
      shutil.rmtree(output_dir, ignore_errors=True)
      os.makedirs(output_dir)

    self.onnx_model = None

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
    return_list = [f"self.{i.name}" for i in inputs_value_infos]
    if len(inputs_value_infos) == 1:
      self.forward_parts.append(f"{return_list[0]}, = inputs")
    else:
      self.forward_parts.append(f"{', '.join(return_list)} = inputs")

  def gen_model_code(self):
    return CodeGenTemplate.model(model_init='''
    '''.join(self.init_parts),
                                 model_forward='''
    '''.join(self.forward_parts))

  def preprocess_onnx_model(self):
    model = onnx.load(self.onnx_model_path)
    model.graph.ClearField("value_info")
    for n in model.graph.node:
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

    for f in (model.graph.input, model.graph.output, model.graph.initializer):
      for i in f:
        if i.name.isnumeric():
          i.name = f"__t_{i.name}"
        else:
          i.name = re.sub("[:/.]", "_", i.name)

    model = SymbolicShapeInference.infer_shapes(model, 2**31 - 1, True, True, 0)
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


def gen(onnx_model_path, output_dir, overwrite=False):
  model_code_gen = ModelCodeGenerator(onnx_model_path=onnx_model_path,
                                      output_dir=output_dir,
                                      overwrite=overwrite)
  model_code_gen.run()


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

  # model_code_gen = ModelCodeGenerator(
  #     onnx_model_path="test/ort_train_resnet18.onnx", output_dir="./")
  model_code_gen = ModelCodeGenerator(onnx_model_path=args.onnx_model_path,
                                      output_dir=args.output_dir,
                                      overwrite=args.overwrite)
  model_code_gen.run()


if __name__ == '__main__':
  main()
