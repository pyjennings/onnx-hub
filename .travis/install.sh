#!/bin/bash

pushd onnx_hub/external/onnx/
sudo python setup.py install
popd

pushd onnx_hub/external/onnx-tensorflow/
sudo python setup.py install
popd
