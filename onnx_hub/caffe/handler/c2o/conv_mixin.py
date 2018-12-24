from onnx.helper import make_node


class ConvMixin(object):

  @classmethod
  def conv_op(cls, node, **kwargs):
    strides = cls.caffe2onnx_param(list(node.convolution_param.stride), 1)[:2]
    dilations = cls.caffe2onnx_param(list(node.convolution_param.dilation), 1)[:2]
    kernel_shape = list(node.convolution_param.kernel_size) * 2
    n_groups = node.convolution_param.group

    pads = cls.caffe2onnx_param(list(node.convolution_param.pad), 0)

    bias_name = [node.name+'_1'] if node.convolution_param.bias_term else []
    conv_node = cls.make_node_from_caffe_node(
        node, [node.bottom[0], node.name+'_0'] + bias_name,
        pads=pads,
        group=n_groups,
        kernel_shape=kernel_shape,
        strides=strides,
        dilations=dilations)

    if not isinstance(conv_node, list):
      conv_node = [conv_node]
    return conv_node

  @staticmethod
  def caffe2onnx_param(caffe_param, default):
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

