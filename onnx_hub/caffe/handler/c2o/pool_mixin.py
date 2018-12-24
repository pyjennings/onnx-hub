from .conv_mixin import ConvMixin


class PoolMixin(object):

  @staticmethod
  def args_check(node, **kwargs):
    if "count_include_pad" in kwargs:
      if cls.ONNX_OP != "AveragePool":
        raise RuntimeError("count_include_pad is only for AveragePool.")
      if cls.SINCE_VERSION < 7:
        raise RuntimeError("count_include_pad is added since version 7.")

  @classmethod
  def pool_op(cls, node, **kwargs):
    strides = cls.caffe2onnx_param([node.pooling_param.stride], 1)[:2]
    kernel_shape = [node.pooling_param.kernel_size] * 2
    pads = cls.caffe2onnx_param([node.pooling_param.pad], 0)

    node_kwargs = {}
    if "count_include_pad" in kwargs:
      node_kwargs["count_include_pad"] = kwargs["count_include_pad"]
    return cls.make_node_from_caffe_node(
        node, [node.bottom[0]],
        pads=pads,
        kernel_shape=kernel_shape,
        strides=strides,
        **node_kwargs)

  @classmethod
  def caffe2onnx_param(cls, caffe_param, default):
    if len(caffe_param) == 0:
      onnx_param = [default] * 4
    elif len(caffe_param) == 1:
      onnx_param = [caffe_param[0]] * 4
    elif len(caffe_param) == 2:
      onnx_param = [caffe_param[0],
                    caffe_param[1],
                    caffe_param[0],
                    caffe_param[1]]
    else:
      raise RuntimeError("Error caffe param.")

    return onnx_param
