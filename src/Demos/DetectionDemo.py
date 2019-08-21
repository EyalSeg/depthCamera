# ## READ ME ### #
# clone from https://github.com/matterport/Mask_RCNN
# cd to master and pip install requirements.txt
# python setup.py install
# provide the required paths to detector class as seen in the demo below
# for the purposes of this demo: 'pip install opencv_contrib_python' might be required


from MaskRcnnDetector import MaskrcnnObjectDetector as Detector

import skimage.io as skio
import cv2 as cv

import numpy as np

path_to_img = '/home/eyal/PycharmProjects/depthCamera/Mask_RCNN-master/images/4782628554_668bc31826_z.jpg'


from pathlib import Path
path_to_maskrcnn = Path(__file__).parent.parent.parent.joinpath("Mask_RCNN-master")
print(path_to_maskrcnn)

print("Initializing MaskRCNN detector...")

detector = Detector(str(path_to_maskrcnn),
                    str(path_to_maskrcnn.joinpath('samples/coco')),
                    str(path_to_maskrcnn.joinpath('samples/mask_rcnn_coco.h5')),
                    str(path_to_maskrcnn.joinpath('logs')),
                        )

print("Init finished")
print("Reading image")

img = skio.imread(path_to_img)
img_input = cv.imread(path_to_img, cv.IMREAD_COLOR)
img_output = img_input.copy()

print("detecting stuff")
results = detector.detect(img)
print(results)

print("\n\n")
print("found the following classes: ")
for id in results['class_ids']:
    print("\t" + detector.class_names[id])
print("end of result classes \n")

print("Displaying results")
detector.display(img, results)

