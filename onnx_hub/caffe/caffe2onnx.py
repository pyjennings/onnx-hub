from caffe.proto import caffe_pb2
from google.protobuf import text_format
import onnx

import onnx_hub.caffe.handler
import caffe_helper

def load(weights_path, model_path):
    weights = caffe_pb2.NetParameter()
    weights.ParseFromString(open(weights_path, "rb").read())

    model = caffe_pb2.NetParameter()
    text_format.Merge(open(model_path).read(), model)
    onnx_model = caffe_helper.caffe_model_to_onnx_model(
            weights, model, 'prob')

    return onnx_model
