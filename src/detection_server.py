#!/usr/bin/python
import json
import http.server
import socketserver
import numpy as np
from numpy import ndarray
import skimage.io
from MaskRcnnDetector import MaskrcnnObjectDetector as Detector

from pathlib import Path


PORT = 8000
path_to_maskrcnn = Path(__file__).parent.parent.joinpath("Mask_RCNN-master")
verbose = False

detector = Detector(str(path_to_maskrcnn),
                    str(path_to_maskrcnn.joinpath('samples/coco')),
                   str(path_to_maskrcnn.joinpath('samples/mask_rcnn_coco.h5')),
                   str(path_to_maskrcnn.joinpath('logs'))
                     )



class image_detection_handler(http.server.BaseHTTPRequestHandler):
    def _set_headers(self):

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()


    # GET sends back a Hello world message
    def do_POST(self):
        length = int(self.headers.get('content-length'))
        body = self.rfile.read(length).decode()
        body = json.loads(body)

        image = np.array(body['image_array'])
        desired_classes = body["classes"]

        if verbose:
            skimage.io.imshow(image)
            skimage.io.show()

        detection_results = detector.detect(image, desired_classes)
        # convert ndarrays to regular lists
        converted_results = {key: (value if not isinstance(value, ndarray) else value.tolist())
                             for (key, value) in detection_results.items()}
        self._set_headers()
        self.wfile.write(json.dumps(converted_results).encode())

        if verbose:
            detector.display(image, detection_results)



with socketserver.TCPServer(("", PORT), image_detection_handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()