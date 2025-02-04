import os
import grpc
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from flask import Flask
from flask import request
from flask import jsonify

# Output classes in the order output by classifier
CLASSES = [
 'antelope_duiker',
 'bird',
 'blank',
 'civet_genet',
 'hog',
 'leopard',
 'monkey_prosimian',
 'rodent'
]

# gRPC port where service is listening
HOST = os.getenv('TF_SERVING_HOST', 'localhost:8500')

# Request timeout (in seconds)
TIMEOUT = 60.0

# Function to convert NumPy tensor to TensorProto (protocol buffer representing a tensor)
def np_to_protobuf(data):
    return tf.make_tensor_proto(data, shape=data.shape)

# Function to prepare a prediction request, given an image in NumPy representation
def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = 'species_model'
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs['input_layer_11'].CopyFrom(np_to_protobuf(X))
    return pb_request

# Function to postprocess prediction response by identifying model scores with class labels
def prepare_response(pb_response):
    preds = pb_response.outputs['output_0'].float_val
    return dict(zip(CLASSES, preds))

# Function to make a prediction, given a image URL
def predict(url):
    # Download image from URL and preprocess it
    img = Image.open(requests.get(url, stream=True).raw)
    img = img.resize((150, 150), Image.NEAREST)
    X = np.array([np.array(img)])
    X = np.float32(X)

    # Make prediction request and return formatted predictions
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=TIMEOUT)
    return prepare_response(pb_response)

# Create Flask app
app = Flask('gateway')

# Instantiate channel and prediction service stub
channel = grpc.insecure_channel(HOST)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)

if __name__ == '__main__':
    # # Image URL
    # url = 'https://drivendata-public-assets.s3.amazonaws.com/zjff-ZJ000097.jpg'

    # # Instantiate channel and prediction service stub
    # channel = grpc.insecure_channel(HOST)
    # stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # # Invoke the model and return predictions
    # response = predict(url)
    # print(response)

    app.run(debug=True, host='0.0.0.0', port=9696)