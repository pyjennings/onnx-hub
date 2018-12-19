from onnx_tf.common import exception
from onnx_hub.caffe.handler.caffe2onnx_handler import Caffe2OnnxHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op
from .conv_mixin import ConvMixin


@onnx_op("Conv")
@tf_op("Convolution")
class Convolution(ConvMixin, Caffe2OnnxHandler):
  """
    Converts different convolutions
    Warning: Depthwise Conv is not supported by ONNX directly, so this generates
    a grouped convolution with n_groups = n_channels which is semantically the same.
    Make sure your backend knows about this special case in order
    to generate more optimal code.
  """

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.conv_op(node, **kwargs)
