#!/bin/bash

pip install -r requirements.txt

onnx_pytorch_dir="$PWD"
pip install -e $onnx_pytorch_dir

python --version

python -m pytest $onnx_pytorch_dir/onnx_pytorch/tests