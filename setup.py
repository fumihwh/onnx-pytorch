import os
from typing import List
from setuptools import setup, find_packages


def _version() -> str:
  from onnx_pytorch import _version
  return _version.__version__


def _parse_requirements() -> List[str]:
  file_path = "requirements.txt"
  if not os.path.exists(file_path):
    file_path = "onnx_pytorch.egg-info/requires.txt"
  with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         file_path)) as f:
    required = f.read().splitlines()
  return required


setup(name="onnx-pytorch",
      version=_version(),
      description="Convert ONNX to PyTorch code.",
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      author="fumihwh",
      author_email="fumihwh@gmail.com",
      url="https://github.com/fumihwh/onnx-pytorch",
      packages=find_packages(),
      license="Apache 2.0",
      scripts=["onnx_pytorch/code_gen.py"],
      install_requires=_parse_requirements(),
      classifiers=["Programming Language :: Python :: 3"])
