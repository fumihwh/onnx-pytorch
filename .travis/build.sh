#!/bin/bash

onnx_pytorch_dir="$PWD"

yapf -ri $onnx_pytorch_dir/onnx_pytorch --exclude $onnx_pytorch_dir/onnx_model_maker/ops
if [ $(git diff --cached --exit-code HEAD^ >/dev/null && (git ls-files --other --exclude-standard --directory | grep -c -v '/$')) ]; then
  echo "yapf formatter check failed."
  exit 1
fi

python -m pytest $onnx_pytorch_dir/onnx_pytorch/tests
