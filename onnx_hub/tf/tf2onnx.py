from tensorflow.core.framework import graph_pb2

from onnx_tf.common import get_output_node_names
from onnx_tf.frontend import tensorflow_graph_to_onnx_model

def load(input_path):
    graph_def = graph_pb2.GraphDef()
    with open(input_path, "rb") as f:   # load tf graph def
        graph_def.ParseFromString(f.read())
    output = get_output_node_names(graph_def)  # get output node names

    # convert tf graph to onnx model
    model = tensorflow_graph_to_onnx_model(graph_def, output)

    return model
