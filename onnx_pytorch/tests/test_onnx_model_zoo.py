from tempfile import TemporaryDirectory

import os
import importlib.util
from urllib import request

import numpy as np
import onnx
import onnxruntime
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
import pytest
import torch
from tqdm import tqdm

from onnx_pytorch import code_gen

torch.set_printoptions(8)


class TqdmUpTo(tqdm):
  """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

  def update_to(self, b=1, bsize=1, tsize=None):
    """
    b  : int, optional
        Number of blocks transferred so far [default: 1].
    bsize  : int, optional
        Size of each block (in tqdm units) [default: 1].
    tsize  : int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
    """
    if tsize is not None:
      self.total = tsize
    return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


class TestModel:

  def _run(self, inputs_np, onnx_model, gen_kwargs={}):
    model = onnx.ModelProto()
    model.CopyFrom(onnx_model)
    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(model.SerializeToString(),
                                           sess_options)
    ort_outputs = session.run(None, {k: v for k, v in inputs_np})
    model.graph.ClearField("value_info")
    model = SymbolicShapeInference.infer_shapes(model, 2**31 - 1, True, True, 1)
    with TemporaryDirectory() as tmpdir:
      code_gen.gen(model,
                   output_dir=tmpdir,
                   tensor_inplace=False,
                   simplify_names=False,
                   **gen_kwargs)
      spec = importlib.util.spec_from_file_location(
          "model", os.path.join(tmpdir, "model.py"))
      mod = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(mod)
      pt_outputs = mod.test_run_model(
          [torch.from_numpy(v) for _, v in inputs_np])
      assert np.allclose(ort_outputs, [o.detach().numpy() for o in pt_outputs],
                         atol=1e-5,
                         rtol=1e-5,
                         equal_nan=True)

  def test_vision_body_analysis_age_gender_age_googlenet(self):
    dir_path = os.path.join(os.path.dirname(__file__), "onnx_model_zoo",
                            "vision", "body_analysis", "age_gender",
                            "age_googlenet")
    file_path = os.path.join(dir_path, "model.onnx")
    if os.path.exists(file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = "https://github.com/onnx/models/raw/master/vision/body_analysis/age_gender/models/age_googlenet.onnx"
      self._down_file([(url, file_path)])
    model = onnx.load(file_path)

    self._run([("input", np.random.randn(1, 3, 224, 224).astype(np.float32))],
              model)

  def test_vision_body_analysis_age_gender_gender_googlenet(self):
    dir_path = os.path.join(os.path.dirname(__file__), "onnx_model_zoo",
                            "vision", "body_analysis", "age_gender",
                            "gender_googlenet")
    file_path = os.path.join(dir_path, "model.onnx")
    if os.path.exists(file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = "https://github.com/onnx/models/raw/master/vision/body_analysis/age_gender/models/gender_googlenet.onnx"
      self._down_file([(url, file_path)])
    model = onnx.load(file_path)

    self._run([("input", np.random.randn(1, 3, 224, 224).astype(np.float32))],
              model)

  def test_vision_body_analysis_arcface_arcfaceresnet100_8(self):
    dir_path = os.path.join(os.path.dirname(__file__), "onnx_model_zoo",
                            "vision", "body_analysis", "arcface",
                            "arcfaceresnet100-8")
    file_path = os.path.join(dir_path, "model.onnx")
    if os.path.exists(file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = "https://github.com/onnx/models/raw/master/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
      self._down_file([(url, file_path)])
    model = onnx.load(file_path)

    # TODO: https://github.com/onnx/models/issues/242
    for node in model.graph.node:
      if node.op_type == "BatchNormalization":
        for attr in node.attribute:
          if attr.name == "spatial":
            attr.i = 1

    self._run([("data", np.random.randn(1, 3, 112, 112).astype(np.float32))],
              model)

  def _down_file(self, pairs):
    for url, path in pairs:
      if os.path.exists(path):
        continue
      downloadnow = '\t[-]Downloading:%s' % url[url.rfind('/') + 1:]
      with TqdmUpTo(unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                    desc=downloadnow,
                    ncols=120) as t:
        request.urlretrieve(url, path, reporthook=t.update_to, data=None)


if __name__ == '__main__':
  pytest.main(['-s', 'test_onnx_model_zoo.py'])
