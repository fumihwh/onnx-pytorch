from setuptools import setup, find_packages

exec(open('onnx_pytorch/_version.py').read())
version_str = __version__

setup(name="onnx-pytorch",
      version=version_str,
      description="Convert ONNX to PyTorch code.",
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      author="fumihwh",
      author_email="fumihwh@gmail.com",
      url="https://github.com/fumihwh/onnx-pytorch",
      packages=find_packages(),
      license="Apache 2.0",
      install_requires=[
          "numpy", "torch", "torchvision", "onnx", "onnxruntime", "PyYaml"
      ],
      scripts=["onnx_pytorch/code_gen.py"],
      classifiers=["Programming Language :: Python :: 3"])
