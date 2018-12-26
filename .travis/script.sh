#!/bin/bash

python -c "import onnx"
python -c "import onnx_tf"
export PYTHONPATH=$PWD
python test/lenet_caffe_test.py
