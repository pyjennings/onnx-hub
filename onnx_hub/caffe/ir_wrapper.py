import inspect
import warnings

import numpy as np
from onnx import NodeProto
from onnx import TensorProto
from onnx import ValueInfoProto
from onnx import numpy_helper
from onnx.helper import make_graph
from onnx.helper import make_tensor
from onnx.helper import make_tensor_value_info
from onnx.helper import mapping

from onnx_tf.common import attr_converter
from onnx_tf.common import attr_translator
from onnx_tf.common import data_type
from onnx_tf.common import IS_PYTHON3

from onnx_hub.caffe.proto.caffe_pb2 import LayerParameter


class IRNode(object):

  def __init__(self,
               node,
               name,
               op_type,
               inputs=None,
               outputs=None,
               domain=None):
    # storing a reference to the original protobuf object
    self.node = node
    self.name = name
    self.op_type = op_type
    self.inputs = inputs or []
    self.outputs = outputs or []
    self.domain = domain or ""

  def get_outputs_names(self, num=None):
    """ Helper method to get outputs names.
    e.g. tf.split: [Split, Split:1, Split:2]

    :param num: Force to get `num` outputs names.
    :return: List of outputs names.
    """
    if num is None:
      if len(self.outputs) > 0:
        num = len(self.outputs)
      else:
        raise RuntimeError("No output!")
    return [
        self.name + ":{}".format(i) if i > 0 else self.name for i in range(num)
    ]


class IRGraph(object):
  """ A helper class for making ONNX graph.
  This class holds all information ONNX graph needs.
  """

  def __init__(self, name=None, graph_proto=None):
    self._name = name or ""
    self._nodes = []
    self._var_names = []
    self._output_names = []
    self._placeholer_names = []
    self._consts = {}

    self._nodes_proto = []
    self._data_type_cast_map = {}

  # This list holds the protobuf objects of type ValueInfoProto
  # representing the input to the converted ONNX graph.
  @property
  def inputs(self):
    inputs = []
    for name, value in self._consts.items():
      inputs.append([name, value.dtype, value.shape])
    for ph in self._placeholer_names:
      # treat placeholder as fp32
      inputs.append([ph, np.dtype('f4'), [u'?']*4])
    return inputs

  @property
  def input_proto(self):
    in_proto = []
    for input_entry in self.inputs:
      input_name = input_entry[0]
      dtype = input_entry[1]
      shape = input_entry[2]
      onnx_dtype = mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
      proto = make_tensor_value_info(input_name, onnx_dtype, shape)
      in_proto.append(proto)
    return in_proto

  @property
  def output_names(self):
    if self._output_names == []:
      raise RuntimeError("No output set.")
    else:
      return self._output_names

  @property
  def output_proto(self):
    out_proto = []
    for output_name in self.output_names:
      proto = make_tensor_value_info(output_name, TensorProto.FLOAT, [u'?']*4)
      out_proto.append(proto)
    return out_proto

  # This list holds the protobuf objects of type NodeProto
  # representing the ops in the converted ONNX graph.
  @property
  def nodes(self):
    return self._nodes

  # This dictionary contains a map from the name of the constant
  # op to the array of values it holds. This is useful because
  # tensorflow is less eager to know about input values at
  # graph construction time than ONNX. That is to say, some ONNX
  # attributes are input tensors in TF. This dictionary extracts
  # those values of constant tensors that are known at graph
  # construction time.
  @property
  def consts(self):
    return self._consts

  @property
  def initializer_proto(self):
    init_proto = []
    for name, value in self._consts.items():
      onnx_dtype = mapping.NP_TYPE_TO_TENSOR_TYPE[value.dtype]
      proto = make_tensor(
        name=name, data_type=onnx_dtype, dims=np.shape(value), vals=value.flatten())
      init_proto.append(proto)
    return init_proto

  # A map holds nodes name and new data type. Will be used to
  # process protos to match ONNX type constraints.
  @property
  def data_type_cast_map(self):
    return self._data_type_cast_map

  # This list holds the protobuf objects of type ValueInfoProto
  # representing the all nodes' outputs to the converted ONNX graph.
  #@property
  #def value_info_proto(self):

  def add_node(self, node):
    if isinstance(node, LayerParameter):
      if node.type in ["Input", "Data"]:
        for top in node.top:
          self._placeholer_names.append(top)
        return
      elif node.type == 'Reshape':
        self.add_const(node.name+'_0',
                np.array(node.reshape_param.shape.dim, dtype="i4"))
        return

      ir_node = IRNode(node, node.name, node.type, node.bottom, node.top)
      self._nodes.append(ir_node)
      for blob_idx in range(len(node.blobs)):
        blob = node.blobs[blob_idx]
        np_blob = np.array(blob.data, dtype="f4")
        np_blob = np.reshape(np_blob, blob.shape.dim)
        if node.type == 'InnerProduct':
          np_blob = np_blob.transpose()
        self.add_const(node.name+'_'+str(blob_idx), np_blob)

      for bottom in node.bottom:
        self.add_var(bottom)
      for top in node.top:
        self.add_var(top)

    else:
      raise RuntimeError("Unsupported node type.")

  def add_var(self, var_name):
    if var_name not in self._var_names:
      self._var_names.append(var_name)

  def add_const(self, name, data):
    self._consts[name] = data

  def set_output(self, output_names):
    for output_name in output_names:
      if output_name in self._var_names:
        self._output_names.append(output_name)
      else:
        raise RuntimeError("No such var.")

  def add_node_proto(self, node_proto):
    if not isinstance(node_proto, (list, tuple)):
      node_proto = [node_proto]
    self._nodes_proto.extend(node_proto)

  def make_graph_proto(self):
    return make_graph(self._nodes_proto, self._name, self.input_proto,
                      self.output_proto, initializer=self.initializer_proto)
