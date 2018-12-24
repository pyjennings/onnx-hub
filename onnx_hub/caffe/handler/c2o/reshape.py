from onnx_tf.common import exception
from onnx_hub.caffe.handler.caffe2onnx_handler import Caffe2OnnxHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Reshape")
@tf_op("Reshape")
class Reshape(Caffe2OnnxHandler):

  @classmethod
  def version_5(cls, node, **kwargs):
    return cls.make_node_from_caffe_node(node, [node.bottom[0], node.name+'_0'])
