import argparse
import logging
import os
import re
import shutil
from collections import Counter

import numpy as np
import onnx
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx.numpy_helper import to_array
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from onnx_pytorch.code_gen_template import CodeGenTemplate
from onnx_pytorch.op_code_generators import *
from onnx_pytorch.utils.embedding_config_helper import load_embedding_config


class RenameHelper:

  def __init__(self, simplify_names=False):
    self.simplify_names = simplify_names

    self.tensor_name_mapping = {}
    self.tensor_name_counter = Counter()
    self.node_name_mapping = {}
    self.node_name_counter = Counter()

    self.tensor_counter = 0
    self.node_counter = Counter()

    self.init_name_set = set()
    self.sim_tensor_name_set = set()

  def get_tensor_name(self, tensor_name):
    if self.simplify_names:
      return self.get_simplify_tensor_name(tensor_name)
    if tensor_name.isnumeric():
      self.tensor_name_mapping[tensor_name] = f"t_{tensor_name}"
      return f"t_{tensor_name}"
    return tensor_name

  def get_node_name(self, node_name, op_type):
    if self.simplify_names or not node_name:
      return self.get_simplify_node_name(node_name, op_type)
    return f"n_{node_name}"

  def get_simplify_node_name(self, node_name, op_type):
    idx = self.node_counter[op_type]
    self.node_counter[op_type] += 1
    self.node_name_mapping[node_name] = f"n_{op_type}_{idx}"
    return self.node_name_mapping[node_name]

  def get_simplify_tensor_name(self, tensor_name):
    if tensor_name in self.tensor_name_mapping:
      return self.tensor_name_mapping[tensor_name]
    suffix = self.tensor_counter
    self.tensor_counter += 1
    sim_tensor_name = f"t_{suffix}"
    self.sim_tensor_name_set.add(sim_tensor_name)
    self.tensor_name_mapping[tensor_name] = sim_tensor_name
    return self.tensor_name_mapping[tensor_name]


class ModelCodeGenerator:

  def __init__(self,
               onnx_model=None,
               output_dir=None,
               simplify_names=False,
               tensor_inplace=False,
               continue_on_error=False,
               embedding_conf=None,
               shape_infer=True):
    self.onnx_model = onnx_model
    self.output_dir = output_dir
    self.rename_helper = RenameHelper(simplify_names)
    self.tensor_inplace = tensor_inplace
    self.continue_on_error = continue_on_error
    self.embedding_conf = embedding_conf
    self.shape_infer = shape_infer
    self.init_parts = []
    self.forward_parts = []
    self.method_parts = {}

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
    return_list = [
        f"{self.rename_helper.get_tensor_name(o.name)}"
        for o in outputs_value_infos
    ]
    self.forward_parts.append(f"return {', '.join(return_list)}")

  def add_forward_input(self, inputs_value_infos):
    initializer_names = {i.name for i in self.onnx_model.graph.initializer}
    return_list = [
        f"{self.rename_helper.get_tensor_name(i.name)}"
        for i in inputs_value_infos
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
                                 model_method='''
  '''.join(self.method_parts.values()),
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
      if shape:
        numpy_input_str.append(
            f"torch.from_numpy(np.random.randn(*{[s if s > 1 else 1 for s in shape].__repr__()}).astype(np.{dtype.name}))"
        )
      else:
        numpy_input_str.append(
            f"torch.from_numpy(np.random.randn(1).astype(np.{dtype.name}))")
    test_run_model = [
        f'''@torch.no_grad()
def test_run_model(inputs=[{', '.join(numpy_input_str)}]):''',
        "model = Model()", "model.eval()"
    ]
    test_run_model.extend(["rs = model(*inputs)", "print(rs)", "return rs"])
    return '''
  '''.join(test_run_model)

  def preprocess_onnx_model(self):
    for n in self.onnx_model.graph.node:
      inputs, outputs = [], []
      for ls, f in ((inputs, n.input), (outputs, n.output)):
        for i in f:
          new_i = re.sub("[:/.]", "_", i)
          ls.append(new_i)
          if i != ls[-1] and not self.rename_helper.simplify_names:
            logging.info(f"Tensor name {i} is changed to {ls[-1]}.")
          self.rename_helper.tensor_name_counter[ls[-1]] += 1

      n.ClearField("input")
      n.input.extend(inputs)
      n.ClearField("output")
      n.output.extend(outputs)

      old_name = n.name
      n.name = re.sub("[:/.]", "_", n.name)
      if old_name != n.name and not self.rename_helper.simplify_names:
        logging.info(f"Node name {old_name} is changed to {n.name}.")
      self.rename_helper.node_name_counter[n.name] += 1

    for f in (self.onnx_model.graph.input, self.onnx_model.graph.output,
              self.onnx_model.graph.initializer):
      for i in f:
        old_name = i.name
        i.name = re.sub("[:/.]", "_", i.name)
        if old_name != i.name and not self.rename_helper.simplify_names:
          logging.info(f"Tensor name {i.name} is changed to {i.name}.")
        self.rename_helper.tensor_name_counter[i.name] += 1

    model = self.onnx_model
    for f in (model.graph.input, model.graph.output):
      for i in f:
        for d in i.type.tensor_type.shape.dim:
          if d.dim_param != "":
            d.dim_param = ""
            d.dim_value = -1
          elif d.dim_value == 0:
            d.dim_value = -1
    # TODO how to deal with custom op?
    if self.shape_infer:
      try:
        model.graph.ClearField("value_info")
        model = SymbolicShapeInference.infer_shapes(model, 2**31 - 1, True,
                                                    True, 1)
      except:
        logging.warning("Shape infer by onnxruntime failed.")
    else:
      for f in (self.onnx_model.graph.value_info,):
        for i in f:
          old_name = i.name
          i.name = re.sub("[:/.]", "_", i.name)
          if old_name != i.name and not self.rename_helper.simplify_names:
            logging.info(f"Tensor name {i.name} is changed to {i.name}.")
          self.rename_helper.tensor_name_counter[i.name] += 1
    onnx.save(model, os.path.join(self.output_dir, "tmp_processed.onnx"))
    self.onnx_model = model

  def add_attr_to_op_code_generator(self, op_code_gen):
    for k, v in {
        "rename_helper": self.rename_helper,
        "tensor_inplace": self.tensor_inplace,
        "embedding_conf": self.embedding_conf
    }.items():
      if hasattr(op_code_gen, k):
        setattr(op_code_gen, k, v)

  def run(self):
    self.preprocess_onnx_model()
    initializers = {i.name: i for i in self.onnx_model.graph.initializer}
    input_value_infos = {i.name: i for i in self.onnx_model.graph.input}
    output_value_infos = {i.name: i for i in self.onnx_model.graph.output}
    value_infos = {}
    value_infos.update(input_value_infos)
    value_infos.update(output_value_infos)
    value_infos.update({i.name: i for i in self.onnx_model.graph.value_info})

    for i in self.onnx_model.graph.initializer:
      self.rename_helper.get_tensor_name(i.name)

    self.add_forward_input(self.onnx_model.graph.input)
    for n in self.onnx_model.graph.node:
      op_code_gen = get_op_code_generator(n.op_type)
      self.add_attr_to_op_code_generator(op_code_gen)
      if op_code_gen is None:
        if self.continue_on_error:
          self.add_forward_part(n.__repr__())
          logging.warning(f"OpCodeGenerator is unimplemented for {n.op_type}. "
                          "Please modify this part by manual later.")
        else:
          raise NotImplementedError(
              f"OpCodeGenerator is unimplemented for {n.op_type}.")
      else:
        try:
          if hasattr(op_code_gen,
                     "gen_method") and n.op_type not in self.method_parts:
            self.method_parts[n.op_type] = op_code_gen.gen_method()
          gened = op_code_gen.gen(n, value_infos, initializers)
          self.add_init_part(gened["init"])
          self.add_forward_part(gened["forward"])
        except BaseException as e:
          if self.continue_on_error:
            logging.warning(e)
            self.add_forward_part(n.__repr__())
          else:
            raise e
    self.add_forward_return(self.onnx_model.graph.output)

    gened_code = self.gen_model_code()
    print(gened_code)
    with open(os.path.join(self.output_dir, "model.py"), "w") as f:
      f.write(gened_code)
    shutil.rmtree(os.path.join(self.output_dir, "variables"),
                  ignore_errors=True)
    os.makedirs(os.path.join(self.output_dir, "variables"))
    for k, v in initializers.items():
      np.save(
          os.path.join(self.output_dir, "variables",
                       f"{self.rename_helper.get_tensor_name(k)}.npy"),
          to_array(v))


def gen(
    onnx_model,
    output_dir,
    overwrite=False,
    tensor_inplace=False,
    simplify_names=False,
    continue_on_error=False,
    embedding_conf_file=None,
    shape_infer=True,
):
  model_code_generator = get_model_code_generator(
      onnx_model, output_dir, overwrite, tensor_inplace, simplify_names,
      continue_on_error, embedding_conf_file, shape_infer)
  model_code_generator.run()


def get_model_code_generator(
    onnx_model,
    output_dir,
    overwrite=False,
    tensor_inplace=False,
    simplify_names=False,
    continue_on_error=False,
    embedding_conf_file=None,
    shape_infer=False,
):
  kwargs = {
      "output_dir": output_dir,
      "simplify_names": simplify_names,
      "tensor_inplace": tensor_inplace,
      "continue_on_error": continue_on_error,
      "shape_infer": shape_infer
  }
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
  if embedding_conf_file is not None:
    assert os.path.exists(
        embedding_conf_file
    ), f"Embedding config file {embedding_conf_file} does not exist."
    kwargs["embedding_conf"] = load_embedding_config(embedding_conf_file)
  return ModelCodeGenerator(**kwargs)


def main():
  debug = True
  parser = argparse.ArgumentParser()
  parser.add_argument("--onnx_model_path",
                      default=None,
                      type=str,
                      required=not debug,
                      help="The ONNX model path.")
  parser.add_argument("--output_dir",
                      default=None,
                      type=str,
                      required=not debug,
                      help="The output dir")
  parser.add_argument("--overwrite",
                      default=False,
                      type=bool,
                      help="Should overwrite the output dir.")
  parser.add_argument("--tensor_inplace",
                      default=False,
                      type=bool,
                      help="Try best to inplace tensor.")
  parser.add_argument("--continue_on_error",
                      default=False,
                      type=bool,
                      help="Continue on error.")
  parser.add_argument("--embedding_conf_file",
                      type=str,
                      help="Embedding config file path.")
  parser.add_argument(
      "--simplify_names",
      default=False,
      type=int,
      help="Use indexing shorten name instead of original name.")
  args = parser.parse_args()

  gen(onnx_model=args.onnx_model_path,
      output_dir=args.output_dir,
      overwrite=args.overwrite,
      tensor_inplace=args.tensor_inplace,
      simplify_names=args.simplify_names,
      continue_on_error=args.continue_on_error,
      embedding_conf_file=args.embedding_conf_file)


if __name__ == '__main__':
  main()
