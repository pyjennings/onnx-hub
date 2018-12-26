from onnx_tf.backend import prepare
import onnx_hub.caffe.caffe2onnx
import tensorflow as tf
import onnx
import cv2
import numpy as np

input_data = cv2.imread("test/data/mnist/6_1099.png", 0)
input_data = input_data.reshape(1, 1, 28, 28)
onnx_lenet = onnx_hub.caffe.caffe2onnx.load(
        "onnx_hub/external/models/caffe/lenet/lenet_iter_10000.caffemodel",
        "onnx_hub/external/models/caffe/lenet/lenet_workaround.prototxt")
tf_rep = prepare(onnx_lenet)
output = tf_rep.run(input_data)
if np.sum(output.prob - [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]) != 0.0:
  raise RuntimeError("Compare error!")
else:
  print("Result compare success.")
