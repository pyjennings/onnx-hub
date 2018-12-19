from onnx_hub.caffe.handler.caffe2onnx_handler import Caffe2OnnxHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from .pool_mixin import PoolMixin


@onnx_op("MaxPool")
@tf_op("Pooling")
class MaxPool(PoolMixin, Caffe2OnnxHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.pool_op(node, **kwargs)

  @classmethod
  def version_8(cls, node, **kwargs):
    return cls.pool_op(node, **kwargs)
