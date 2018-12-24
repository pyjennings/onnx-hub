from onnx_tf.backend import prepare
import onnx_hub.tf.tf2onnx
import tensorflow as tf
import onnx
import cv2
import numpy as np

input_data = cv2.imread("../../../code/glow/tests/images/mnist/4_1059.png", 0)
input_data = input_data.reshape(1, 784)
onnx_lenet = onnx_hub.tf.tf2onnx.load("../../../code/models/tensorflow/lenet_fp.pb")
onnx.save(onnx_lenet, "/home/min/tf_lenet.onnx")
output = prepare(onnx_lenet).run(input_data)
print(np.argmax(output))
