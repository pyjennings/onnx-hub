from onnx_hub.caffe.handler.caffe2onnx_handler import Caffe2OnnxHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Relu")
@tf_op("ReLU")
class Relu(Caffe2OnnxHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.make_node_from_caffe_node(node, [node.bottom[0]])

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.make_node_from_caffe_node(node, [node.bottom[0]])
