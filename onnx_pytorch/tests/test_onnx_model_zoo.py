import logging
import tarfile
from tempfile import TemporaryDirectory

import os
import importlib.util
from urllib import request

import numpy as np
import onnx
import onnxruntime
from onnx.numpy_helper import to_array
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
import pytest
import torch
from tqdm import tqdm

from onnx_pytorch import code_gen

torch.set_printoptions(8)

ONNX_MODEL_ZOO_DIR = "~/onnx_model_zoo"
ONNX_MODELS_REPO = "https://github.com/onnx/models/raw/main"


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

  def _run(self, inputs_np, onnx_model, gen_kwargs=None, tol=None):
    inputs_np_dict = {k: v for k, v in inputs_np}
    model = onnx.ModelProto()
    model.CopyFrom(onnx_model)
    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(model.SerializeToString(),
                                           sess_options)
    ort_outputs = session.run(None, inputs_np_dict)
    model.graph.ClearField("value_info")
    initializers = {i.name: i for i in model.graph.initializer}
    for i in model.graph.input:
      if i.name in initializers:
        continue
      for idx, d in enumerate(i.type.tensor_type.shape.dim):
        if d.dim_param != "":
          d.ClearField("dim_param")
        d.dim_value = inputs_np_dict[i.name].shape[idx]
    try:
      model = SymbolicShapeInference.infer_shapes(model, 2**31 - 1, True, True,
                                                  1)
    except:
      logging.warning("Shape infer by onnxruntime failed.")
    with TemporaryDirectory() as tmpdir:
      if gen_kwargs is None:
        gen_kwargs = {}
      code_gen.gen(model,
                   output_dir=tmpdir,
                   tensor_inplace=False,
                   simplify_names=False,
                   shape_infer=False,
                   **gen_kwargs)
      spec = importlib.util.spec_from_file_location(
          "model", os.path.join(tmpdir, "model.py"))
      mod = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(mod)
      pt_outputs = mod.test_run_model(
          [torch.from_numpy(v) for _, v in inputs_np])
      if tol is None:
        tol = {"atol": 1e-5, "rtol": 1e-5}
      for l, r in zip(ort_outputs, [o.detach().numpy() for o in pt_outputs]):
        assert np.allclose(l, r, equal_nan=True, **tol)

  def test_vision_body_analysis_age_gender_age_googlenet(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "body_analysis",
                            "age_gender", "age_googlenet")
    file_path = os.path.join(dir_path, "model.onnx")
    if os.path.exists(file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/body_analysis/age_gender/models/age_googlenet.onnx"
      self._down_file([(url, file_path)])
    model = onnx.load(file_path)

    self._run([("input", np.random.randn(1, 3, 224, 224).astype(np.float32))],
              model)

  def test_vision_body_analysis_age_gender_gender_googlenet(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "body_analysis",
                            "age_gender", "gender_googlenet")
    file_path = os.path.join(dir_path, "model.onnx")
    if os.path.exists(file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/body_analysis/age_gender/models/gender_googlenet.onnx"
      self._down_file([(url, file_path)])
    model = onnx.load(file_path)

    self._run([("input", np.random.randn(1, 3, 224, 224).astype(np.float32))],
              model)

  @pytest.mark.skip(reason="torch==1.12 && onnx==1.12 break this test")
  def test_vision_body_analysis_arcface_arcfaceresnet100_8(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "body_analysis",
                            "arcface", "arcfaceresnet100-8")
    file_path = os.path.join(dir_path, "model.onnx")
    if os.path.exists(file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
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

  def test_vision_body_analysis_emotion_ferplus_emotion_ferplus_8(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "body_analysis",
                            "emotion_ferplus", "emotion-ferplus-8")
    file_path = os.path.join(dir_path, "model.onnx")
    if os.path.exists(file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
      self._down_file([(url, file_path)])
    model = onnx.load(file_path)

    self._run([("Input3", np.random.randn(1, 1, 64, 64).astype(np.float32))],
              model)

  def test_vision_body_analysis_ultraface_version_RFB_320(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "body_analysis",
                            "ultraface", "version-RFB-320")
    file_path = os.path.join(dir_path, "model.onnx")
    if os.path.exists(file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/body_analysis/ultraface/models/version-RFB-320.onnx"
      self._down_file([(url, file_path)])
    model = onnx.load(file_path)

    self._run([("input", np.random.randn(1, 3, 240, 320).astype(np.float32))],
              model)

  def test_vision_classification_alexnet(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "classification",
                            "alexnet")
    file_path = os.path.join(dir_path, "model.onnx")
    if os.path.exists(file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/classification/alexnet/model/bvlcalexnet-9.onnx"
      self._down_file([(url, file_path)])
    model = onnx.load(file_path)

    self._run([("data_0", np.random.randn(1, 3, 224, 224).astype(np.float32))],
              model)

  def test_vision_classification_mobilenet(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "classification",
                            "mobilenet")
    file_path = os.path.join(dir_path, "model.onnx")
    if os.path.exists(file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
      self._down_file([(url, file_path)])
    model = onnx.load(file_path)

    self._run([("input", np.random.randn(1, 3, 224, 224).astype(np.float32))],
              model)

  def test_vision_classification_resnet(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "classification",
                            "resnet")
    file_path = os.path.join(dir_path, "model.onnx")
    if os.path.exists(file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/classification/resnet/model/resnet18-v2-7.onnx"
      self._down_file([(url, file_path)])
    model = onnx.load(file_path)

    self._run([("data", np.random.randn(1, 3, 224, 224).astype(np.float32))],
              model)

  def test_vision_classification_shufflenet(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "classification",
                            "shufflenet")
    file_path = os.path.join(dir_path, "model.onnx")
    if os.path.exists(file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/classification/shufflenet/model/shufflenet-v2-10.onnx"
      self._down_file([(url, file_path)])
    model = onnx.load(file_path)

    self._run([("input", np.random.randn(1, 3, 224, 224).astype(np.float32))],
              model)

  def test_vision_classification_squeezenet(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "classification",
                            "squeezenet")
    file_path = os.path.join(dir_path, "model.onnx")
    if os.path.exists(file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
      self._down_file([(url, file_path)])
    model = onnx.load(file_path)

    self._run([("data", np.random.randn(1, 3, 224, 224).astype(np.float32))],
              model)

  def test_vision_object_detection_segmentation_faster_rcnn(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision",
                            "object_detection_segmentation", "faster-rcnn")
    tar_file_path = os.path.join(dir_path, "FasterRCNN-10.tar.gz")
    if os.path.exists(tar_file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.tar.gz"
      self._down_file([(url, tar_file_path)])
    tar = tarfile.open(tar_file_path)
    names = tar.getnames()
    dir_path_tar = f"{dir_path}/faster_rcnn_R_50_FPN_1x"
    for name in names:
      tar.extract(name, path=dir_path)
    tar.close()
    file_path = os.path.join(dir_path_tar, "faster_rcnn_R_50_FPN_1x.onnx")
    model = onnx.load(file_path)
    image = onnx.load_tensor(
        os.path.join(dir_path_tar, "test_data_set_0", "input_0.pb"))
    self._run([("image", to_array(image))], model)

  @pytest.mark.skip(reason="https://github.com/pytorch/pytorch/issues/62237")
  def test_vision_object_detection_segmentation_mask_rcnn(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision",
                            "object_detection_segmentation", "mask-rcnn")
    tar_file_path = os.path.join(dir_path, "MaskRCNN-10.tar.gz")
    if os.path.exists(tar_file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.tar.gz"
      self._down_file([(url, tar_file_path)])
    tar = tarfile.open(tar_file_path)
    names = tar.getnames()
    dir_path_tar = dir_path
    for name in names:
      tar.extract(name, path=dir_path)
    tar.close()
    file_path = os.path.join(dir_path_tar, "mask_rcnn_R_50_FPN_1x.onnx")
    model = onnx.load(file_path)
    image = onnx.load_tensor(
        os.path.join(dir_path_tar, "test_data_set_0", "input_0.pb"))
    self._run([("image", to_array(image))],
              model,
              tol={
                  "atol": 1e-4,
                  "rtol": 1e-5
              })

  def test_vision_object_detection_segmentation_ssd(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision",
                            "object_detection_segmentation", "ssd")
    tar_file_path = os.path.join(dir_path, "ssd-10.tar.gz")
    if os.path.exists(tar_file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/object_detection_segmentation/ssd/model/ssd-10.tar.gz"
      self._down_file([(url, tar_file_path)])
    tar = tarfile.open(tar_file_path)
    names = tar.getnames()
    dir_path_tar = dir_path
    for name in names:
      tar.extract(name, path=dir_path)
    tar.close()
    file_path = os.path.join(dir_path_tar, "model.onnx")
    model = onnx.load(file_path)
    image = onnx.load_tensor(
        os.path.join(dir_path_tar, "test_data_set_0", "input_0.pb"))
    self._run([("image", to_array(image))], model)

  def test_vision_style_transfer_fast_neural_style_candy(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "style_transfer",
                            "fast_neural_style", "candy")
    tar_file_path = os.path.join(dir_path, "candy-9.tar.gz")
    if os.path.exists(tar_file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/style_transfer/fast_neural_style/model/candy-9.tar.gz"
      self._down_file([(url, tar_file_path)])
    tar = tarfile.open(tar_file_path)
    names = tar.getnames()
    dir_path_tar = os.path.join(dir_path, "candy")
    for name in names:
      tar.extract(name, path=dir_path)
    tar.close()
    file_path = os.path.join(dir_path_tar, "candy.onnx")
    model = onnx.load(file_path)
    input1 = onnx.load_tensor(
        os.path.join(dir_path_tar, "test_data_set_0", "input_0.pb"))
    self._run([("input1", to_array(input1))],
              model,
              tol={
                  "atol": 1e-2,
                  "rtol": 1e-5
              })

  def test_vision_style_transfer_fast_neural_style_mosaic(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "style_transfer",
                            "fast_neural_style", "mosaic")
    tar_file_path = os.path.join(dir_path, "mosaic-9.tar.gz")
    if os.path.exists(tar_file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/style_transfer/fast_neural_style/model/mosaic-9.tar.gz"
      self._down_file([(url, tar_file_path)])
    tar = tarfile.open(tar_file_path)
    names = tar.getnames()
    dir_path_tar = os.path.join(dir_path, "mosaic")
    for name in names:
      tar.extract(name, path=dir_path)
    tar.close()
    file_path = os.path.join(dir_path_tar, "mosaic.onnx")
    model = onnx.load(file_path)
    input1 = onnx.load_tensor(
        os.path.join(dir_path_tar, "test_data_set_0", "input_0.pb"))
    self._run([("input1", to_array(input1))],
              model,
              tol={
                  "atol": 1e-2,
                  "rtol": 1e-5
              })

  def test_vision_style_transfer_fast_neural_style_pointilism(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "style_transfer",
                            "fast_neural_style", "pointilism")
    tar_file_path = os.path.join(dir_path, "pointilism-9.tar.gz")
    if os.path.exists(tar_file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/style_transfer/fast_neural_style/model/pointilism-9.tar.gz"
      self._down_file([(url, tar_file_path)])
    tar = tarfile.open(tar_file_path)
    names = tar.getnames()
    dir_path_tar = os.path.join(dir_path, "pointilism")
    for name in names:
      tar.extract(name, path=dir_path)
    tar.close()
    file_path = os.path.join(dir_path_tar, "pointilism.onnx")
    model = onnx.load(file_path)
    input1 = onnx.load_tensor(
        os.path.join(dir_path_tar, "test_data_set_0", "input_0.pb"))
    self._run([("input1", to_array(input1))],
              model,
              tol={
                  "atol": 1e-4,
                  "rtol": 1e-5
              })

  def test_vision_style_transfer_fast_neural_style_rain_princess(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "style_transfer",
                            "fast_neural_style", "rain_princess")
    tar_file_path = os.path.join(dir_path, "rain-princess-9.tar.gz")
    if os.path.exists(tar_file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/style_transfer/fast_neural_style/model/rain-princess-9.tar.gz"
      self._down_file([(url, tar_file_path)])
    tar = tarfile.open(tar_file_path)
    names = tar.getnames()
    dir_path_tar = os.path.join(dir_path, "rain_princess")
    for name in names:
      tar.extract(name, path=dir_path)
    tar.close()
    file_path = os.path.join(dir_path_tar, "rain_princess.onnx")
    model = onnx.load(file_path)
    input1 = onnx.load_tensor(
        os.path.join(dir_path_tar, "test_data_set_0", "input_0.pb"))
    self._run([("input1", to_array(input1))],
              model,
              tol={
                  "atol": 1e-3,
                  "rtol": 1e-5
              })

  def test_vision_style_transfer_fast_neural_style_udnie(self):
    dir_path = os.path.join(ONNX_MODEL_ZOO_DIR, "vision", "style_transfer",
                            "fast_neural_style", "udnie")
    tar_file_path = os.path.join(dir_path, "udnie-9.tar.gz")
    if os.path.exists(tar_file_path):
      pass
    else:
      os.makedirs(dir_path, exist_ok=True)
      url = f"{ONNX_MODELS_REPO}/vision/style_transfer/fast_neural_style/model/udnie-9.tar.gz"
      self._down_file([(url, tar_file_path)])
    tar = tarfile.open(tar_file_path)
    names = tar.getnames()
    dir_path_tar = os.path.join(dir_path, "udnie")
    for name in names:
      tar.extract(name, path=dir_path)
    tar.close()
    file_path = os.path.join(dir_path_tar, "udnie.onnx")
    model = onnx.load(file_path)
    input1 = onnx.load_tensor(
        os.path.join(dir_path_tar, "test_data_set_0", "input_0.pb"))
    self._run([("input1", to_array(input1))],
              model,
              tol={
                  "atol": 1e-3,
                  "rtol": 1e-5
              })

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
