import collections
import sys

import onnx
import onnx.onnx_cpp2py_export.checker as C
from onnx.helper import make_opsetid, make_model_gen_version

__all__ = ["omm", "mod_name", "reset_model", "set_model", "onnx_mm_export"]

# OPSET_VER = onnx.defs.onnx_opset_version()
OPSET_VER = 13


class OnnxModelMaker:

  def __init__(self, opset_ver=OPSET_VER):
    self.opset_import = make_opsetid("", opset_ver)
    self.model = make_model_gen_version(onnx.GraphProto(),
                                        opset_imports=[self.opset_import])
    self.op_counter = collections.Counter()
    self.ctx = C.CheckerContext()
    self.ctx.ir_version = self.model.ir_version
    self.ctx.opset_imports = {'': opset_ver}

  def reset_model(self, opset_ver=None):
    if opset_ver is not None:
      opset_imports = [make_opsetid("", opset_ver)]
      global OPSET_VER
      OPSET_VER = opset_ver
      self.ctx.opset_imports = {'': opset_ver}
    else:
      opset_imports = [self.opset_import]
    self.model = make_model_gen_version(onnx.GraphProto(),
                                        opset_imports=opset_imports)
    self.op_counter = collections.Counter()

  def set_model(self, model):
    self.model = model


omm = OnnxModelMaker()
mod_name = __name__
reset_model = omm.reset_model
set_model = omm.set_model
ctx = omm.ctx


class onnx_mm_export(object):

  def __init__(self, *args, **kwargs):
    self._names = args

  def __call__(self, func):
    for a in self._names:
      mod = sys.modules[f"{mod_name}.ops"]
      setattr(mod, a, func)
    pass

    return func
