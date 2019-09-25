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

#path_to_img = '/home/eyal/PycharmProjects/depthCamera/Mask_RCNN-master/images/4782628554_668bc31826_z.jpg'
path_to_img = '/home/eyal/Pictures/line_simulation.png'

#name the types of objects you'd like to detect. None for all known objects in the network
desired_classes = ['person']

from pathlib import Path
path_to_maskrcnn = Path(__file__).parent.parent.parent.joinpath("Mask_RCNN-master")
print(path_to_maskrcnn)

print("Initializing MaskRCNN detector...")

detector = Detector(str(path_to_maskrcnn),
                    str(path_to_maskrcnn.joinpath('samples/coco')),
                    str(path_to_maskrcnn.joinpath('samples/mask_rcnn_coco.h5')),
                    str(path_to_maskrcnn.joinpath('logs'))
                        )

print("Init finished")
print("Reading image")

img = skio.imread(path_to_img)

print("detecting stuff")
results = detector.detect(img, desired_classes=desired_classes)
print(results)

print("\n\n")
print("found the following classes: ")
for id in results['class_ids']:
    print("\t" + detector.class_names[id])
print("end of result classes \n")

print("Displaying results")
detector.display(img, results)

