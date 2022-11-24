from __future__ import print_function
from PIL import Image
import numpy as np
import time
import requests
import random as rd
SERVER_URL1 = 'http://' + '192.168.1.103' + ':8501/v1/models/rfcn:predict'




def numpys(batch_size):
    
    start = time.time()
    num = rd.randrange(1, 14)
    image  = Image.open(f'{num}.jpg')
    (im_width, im_height) = image.size
    
    a = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    np_images = np.repeat(np.expand_dims(a, 0).tolist(), batch_size, axis=0).tolist()
    b = time.time() - start
    print("time :",f'{b}')
    return '{"instances" : %s}' % np_images
        


def request(predict_request):
    ladybug = requests.post(SERVER_URL1, data=predict_request)
    output_dict = ladybug.json()['predictions'][0]
    nod = np.array(output_dict['detection_scores'])
    my_num_detections = len(nod[nod>=0.5])
    print(ladybug,my_num_detections)



predict_request_cloud = numpys(1)
request(predict_request_cloud)


# batchsize = 1  