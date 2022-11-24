from __future__ import print_function
from PIL import Image
import numpy as np
import time
import requests


SERVER_URL1 = 'http://' + 'localhost' + ':8501/v1/models/rfcn:predict'
SERVER_URL = 'http://' + 'localhost' + ':8500/v1/models/rfcn:predict'
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

def numpys(batch_size):
    
    start = time.time()

    image  = Image.open(f'/home/bohyeok/dockertopython/1.jpg')
    (im_width, im_height) = image.size
    
    a = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    np_images = np.repeat(np.expand_dims(a, 0).tolist(), batch_size, axis=0).tolist()
    b = time.time() - start
    print("time :",f'{b}')
    options = [('grpc.max_message_length', 100 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 *1024)]
    channel = grpc.insecure_channel(SERVER_URL, options = options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    
    request.model_spec.signature_name = 'serving_default'
    request.inputs['input_tensor'].CopyFrom(tf.make_tensor_proto(np_images))
    return (stub, request)


def request(predict_request):
    ladybug=predict_request[0].Predict(predict_request[1])
    output_dict = ladybug.outputs
    nod = output_dict['detection_scores']
    nod = tf.make_ndarray(nod)
    my_num_detections = len(nod[nod >= 0.5])
    print(my_num_detections)


predict_request_cloud = numpys(1)
request(predict_request_cloud)