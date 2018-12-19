from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings

import onnx
from onnx import checker
from onnx import helper

from onnx_tf.handlers.handler import Handler
from onnx_tf.common import deprecated
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import get_unique_suffix


class Caffe2OnnxHandler(Handler):
  """ This class is base frontend handler class.
  All frontend operator handler class MUST inherit this class.
  In frontend, operator handler class's name should be pascal case of file name
  which should be snake case.
  It is best to use tf functions' names. e.g. tf.nn.avg_pool
  If there is a multiple mapping, e.g. tf.nn.conv1d, tf.nn.conv2d, tf.nn.conv3d,
  try find common one first. In this case, tf.nn.convolution.
  Finally, use ONNX operator name if above does not work.
  """

  @classmethod
  def check_cls(cls):
    super(Caffe2OnnxHandler, cls).check_cls()

  @classmethod
  def handle(cls, node, **kwargs):
    return super(Caffe2OnnxHandler, cls).handle(node, **kwargs)

  @classmethod
  def make_node(cls,
                op_type,
                inputs,
                outputs,
                name=None,
                doc_string=None,
                version=0,
                should_check=True,
                **kwargs):
    """ Make a NodeProto from scratch.
    The main api is same to onnx.helper.make_node without any default value.

    :param op_type: The name of the operator to construct.
    :param inputs: Inputs names.
    :param outputs: Outputs names.
    :param name: optional unique identifier.
    :param doc_string: optional documentation string.
    :param version: Version used for check node. Default is cls.VERSION.
    :param should_check: Should check flag.
    Should set to False if is an unimplemented customized op.
    :param kwargs: Other args.
    :return: NodeProto.
    """
    node = helper.make_node(op_type, inputs, outputs, name, doc_string,
                            **kwargs)
    if should_check:
      cls.check_node(node, version)
    else:
      warnings.warn("Skipped check for {}.".format(node.op_type))
    return node

  @classmethod
  def make_node_from_caffe_node(cls,
                               node,
                               inputs=None,
                               outputs=None,
                               op_type=None,
                               name=None,
                               doc_string=None,
                               version=0,
                               should_check=True,
                               **kwargs):
    """ Helper method to make node.
    The main api is almost same to onnx.helper.make_node with default value
    from IRNode given.

    :param node: IRNode object.
    :param inputs: Inputs names. Default is node.inputs.
    :param outputs: Outputs name. Default is node.outputs.
    :param op_type: ONNX op name. Default is cls.ONNX_OP.
    :param name: Node name. Default is node.name.
    :param doc_string: optional documentation string.
    :param version: Version used for check node. Default is cls.VERSION.
    :param should_check: Should check flag.
    Should set to False if is an unimplemented customized op.
    :param kwargs: Other args.
    :return: NodeProto.
    """
    inputs = inputs
    outputs = outputs if outputs is not None else list(node.top)
    onnx_node = helper.make_node(
        op_type if op_type is not None else cls.ONNX_OP,
        inputs,
        outputs,
        name=name if name is not None else node.name,
        doc_string=doc_string,
        **kwargs)

    if should_check:
      cls.check_node(onnx_node, version)
    else:
      warnings.warn("Skipped check for {}.".format(node.op_type))

    return onnx_node

  @classmethod
  def check_node(cls, node, version=0):
    version = version or cls.VERSION
    if version == 0:
      raise ValueError("version can not be 0.")
    ctx = checker.C.CheckerContext()
    ctx.ir_version = onnx.IR_VERSION
    ctx.opset_imports = {cls.DOMAIN: version}
    checker.check_node(node, ctx=ctx)

