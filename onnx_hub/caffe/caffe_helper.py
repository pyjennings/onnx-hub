from onnx import defs
from onnx.helper import make_model
from onnx.helper import make_opsetid
from onnx.optimizer import optimize

from onnx_tf.common import exception
from onnx_hub.caffe.handler.caffe2onnx_handler import Caffe2OnnxHandler
from onnx_hub.caffe.ir_wrapper import IRGraph
from onnx_hub.caffe.handler.c2o import *

def get_all_caffe2onnx_handlers(opset_dict):
  """ Get a dict of all caffe2onnx handler classes.
  e.g. {'domain': {'Abs': Abs handler class}, ...}, }.

  :param opset_dict: A dict of opset. e.g. {'domain': version, ...}
  :return: Dict.
  """
  handlers = {}
  for handler in Caffe2OnnxHandler.__subclasses__():
    handler.check_cls()

    domain = handler.DOMAIN
    version = opset_dict[domain]
    handler.VERSION = version

    since_version = 1
    if handler.ONNX_OP and defs.has(handler.ONNX_OP, domain=handler.DOMAIN):
      since_version = defs.get_schema(
          handler.ONNX_OP, domain=handler.DOMAIN,
          max_inclusive_version=version).since_version
    else:
      warnings.warn("Unknown op {} in domain `{}`. "
                    "Can't check specification by ONNX. "
                    "Please set should_check flag to False "
                    "when call make_node method in handler.".format(
                        handler.ONNX_OP or "Undefined", handler.DOMAIN or
                        "ai.onnx"))
    handler.SINCE_VERSION = since_version

    for caffe_layer in handler.TF_OP:
      handlers.setdefault(domain, {})[caffe_layer] = handler
  return handlers

def merge_caffe_model(weights, model):
  for layer in model.layer:
    for weight_layer in weights.layer:
      if layer.name == weight_layer.name:
        layer.blobs.MergeFrom(weight_layer.blobs)


def caffe_model_to_onnx_graph(caffemodel,
                              output,
                              opset=((defs.ONNX_DOMAIN,
                                      defs.onnx_opset_version()),),
                              name="graph",
                              ignore_unimplemented=False):
  """Converts a Caffe model Proto to an ONNX graph

  This function converts a Caffe model proto to an equivalent
  representation of ONNX graph.

  :param caffemodel: Caffe Proto object.
  :param output: List of Tensorflow NodeDef object specifying which nodes
    to be taken as outputs of the ONNX graph.
  :param opset: Opset, which should be ((str domain: int version number),).
  :param name: The name of the output ONNX Graph.
  :param ignore_unimplemented: Convert to ONNX model and ignore all the operators
    that are not currently supported by onnx-hub.
    This is an experimental feature. By enabling this feature,
    the graph would not be guaranteed to match the ONNX specifications.

  :returns: The equivalent ONNX Graph Proto object.
  """
  ir_graph = IRGraph(name)
  exception.IGNORE_UNIMPLEMENTED = ignore_unimplemented
  training_ops_to_remove = ["DropOut"]

  opset_dict = {}
  for domain, version in opset:
    if domain == "ai.onnx":
      domain = defs.ONNX_DOMAIN
    opset_dict[domain] = version

  handlers = get_all_caffe2onnx_handlers(opset_dict)

  for node in caffemodel.layer:
    if node.type in training_ops_to_remove:
      logger.info(
          "A training op with name {} type {} has been removed.".format(
              node.name, node.type))
    else:
      ir_graph.add_node(node)
      if node.type == "Input":
        continue
      handler = handlers.get(defs.ONNX_DOMAIN, {}).get(node.type, None)
      node_proto = None
      if handler:
        node_proto = handler.handle(
            node,
            consts=ir_graph.consts,
            data_type_cast_map=ir_graph.data_type_cast_map)
      else:
        exception.OP_UNIMPLEMENTED_EXCEPT(
            node.type,
            domain=None if defs.ONNX_DOMAIN in handlers else defs.ONNX_DOMAIN)

      if node_proto is None:
        node_proto = Caffe2OnnxHandler.make_node_from_caffe_layer(
            node, op_type=node.type, should_check=False)
      ir_graph.add_node_proto(node_proto)

  ir_graph.set_output(output)

  return ir_graph.make_graph_proto()


def caffe_model_to_onnx_model(weights,
                              model,
                              output,
                              opset=0,
                              producer_name="onnx-hub",
                              graph_name="graph",
                              ignore_unimplemented=False,
                              optimizer_passes=None):
  """Converts a Caffe model Proto to an ONNX model

  This function converts a Caffe model proto to an equivalent
  representation of ONNX model.

  :param weights: caffemodel Proto object.
  :param model: Proto object from prototxt file.
  :param output: List of string or a string specifying the name
    of the output graph node.
  :param opset: Opset version number, list or tuple.
    Default is 0 means using latest version with domain ''.
    List or tuple items should be (str domain, int version number).
  :param producer_name: The name of the producer.
  :param graph_name: The name of the output ONNX Graph.
  :param ignore_unimplemented: Convert to ONNX model and ignore all the operators
    that are not currently supported by onnx-hub.
    This is an experimental feature. By enabling this feature,
    the model would not be guaranteed to match the ONNX specifications.
  :param optimizer_passes: List of optimization names c.f.
    https://github.com/onnx/onnx/blob/master/onnx/optimizer.py for available
    optimization passes.

  :returns: The equivalent ONNX Model Proto object.
  """

  if not isinstance(opset, (int, long, list, tuple)):
    raise TypeError("opset is expected to int, list or tuple, but {}.".format(
        type(opset)))
  if isinstance(opset, (int, long)):
    opset = [(defs.ONNX_DOMAIN, opset or defs.onnx_opset_version())]
  opset_imports = [make_opsetid(item[0], item[1]) for item in opset]

  if not isinstance(output, (list, tuple)):
    output = [output]

  merge_caffe_model(weights, model)
  onnx_graph = caffe_model_to_onnx_graph(
      model, output, opset, graph_name, ignore_unimplemented)
  onnx_model = make_model(
      onnx_graph, producer_name=producer_name, opset_imports=opset_imports)

  if isinstance(optimizer_passes, (list, tuple)) and optimizer_passes:
    onnx_model = optimize(onnx_model, optimizer_passes)

  return onnx_model

