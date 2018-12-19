from onnx_tf.common import get_unique_suffix
from onnx_hub.caffe.handler.caffe2onnx_handler import Caffe2OnnxHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("MatMul")
@tf_op("InnerProduct")
class Matmul(Caffe2OnnxHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    input_a = node.bottom[0]
    input_b = node.name + '_0'
    input_c = [node.name+'_1'] if node.inner_product_param.bias_term else []
    node_mul_proto = cls.make_node_from_caffe_node(
            node, [input_a, input_b], [node.top[0]+'_mul'])

    if input_c != []:
      node_bias_proto = cls.make_node(
              "Add", [node.top[0]+'_mul', input_c[0]], 
              [node.top[0]], node.name+'_bias')
    return [node_mul_proto, node_bias_proto]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)
