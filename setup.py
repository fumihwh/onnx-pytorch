from setuptools import setup, find_packages

version_str = open("version.txt", "r").read().strip()

setup(name="onnx-pytorch",
      version=version_str,
      description="Convert onnx to pytorch code.",
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      author="fumihwh",
      author_email="fumihwh@gmail.com",
      url="https://github.com/fumihwh/onnx-pytorch",
      packages=find_packages(),
      license="Apache 2.0",
      install_requires=["numpy", "torch", "onnx", "onnxruntime"],
      scripts=["onnx_pytorch/code_gen.py"],
      classifiers=["Programming Language :: Python :: 3"])
