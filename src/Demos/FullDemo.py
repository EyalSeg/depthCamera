from MaskRcnnDetector import MaskrcnnObjectDetector as Detector
from RealSenseLoop import AsyncRealSenseLoop, FramesConsumer
import LineDetector as ln

from os import listdir
from os.path import isfile, join
import skimage.io

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


from pathlib import Path


desired_classes = ["cup"]
distance = 0.1
path_to_maskrcnn = Path(__file__).parent.parent.parent.joinpath("Mask_RCNN-master")
path_to_bagfile_dir = '/home/eyal/Documents/line_bags'
verbose = False
#verbose=True
#path_to_bagfile_dir = None


def init_mrcnn(path_to_mrcnn):
    print(path_to_maskrcnn)

    print("Initializing MaskRCNN detector...")

    detector = Detector(str(path_to_mrcnn),
                        str(path_to_mrcnn.joinpath('samples/coco')),
                        str(path_to_mrcnn.joinpath('samples/mask_rcnn_coco.h5')),
                        str(path_to_mrcnn.joinpath('logs'))
                        )

    print("Init finished")

    return detector

def detect(detector, bagfile, verbose=False, last_n_people = 5):
    with AsyncRealSenseLoop(bagfile=bagfile) as loop:
        consumer = FramesConsumer(loop)
        frame = consumer.consume()

    skimage.io.imshow(frame.color_image)

    if verbose:
        skimage.io.imshow(frame.depth_image * frame.depth_scale)

    skimage.io.show()

    detection_results = detector.detect(frame.color_image, desired_classes)

    if verbose:
        detector.display(frame.color_image, detection_results)

    masks = [detection_results["masks"][:, :, i] for i in range(len(detection_results["rois"]))]
    coordinates = [frame.get_object_position(mask) for mask in masks]

    #distances = [np.linalg.norm(coord) for coord in coordinates]
    #dist2 = [frame.distance_to_object(mask) for mask in masks]

    # TODO: convert coordinates to a world-space s.t the ground is at y=0

    # we ignore most of the line except for the last few people
    coordinates.sort(key=lambda coord: coord[0])

    xs = [x for (x, y, z) in coordinates]
    zs = [z for (x, y, z) in coordinates]
    ys = [y for (x, y, z) in coordinates]
    # z_ = zs[-5:]

    x_ = xs[-last_n_people:]
    z_ = zs[-last_n_people:]

    result = ln.find_line_position(x_, ys, z_, distance)
    poly = result["line_polynomial"]
    new_pos = result["next_pos"]

    # plt.ylim(0, 2)
    # plt.xlim(-0.5, 0.5)
    plt.plot(xs, zs, 'bo')

    # plot line
    poly_x = np.linspace(xs[0] - 0.1, xs[-1] + distance, 100)
    poly_y = poly(poly_x)
    plt.plot(poly_x, poly_y)

    # plot the new pos
    plt.plot(*new_pos, "ro")
    plt.annotate('new position', xy=new_pos, xytext=(new_pos[0], new_pos[1] + 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )

    plt.show()


detector = init_mrcnn(path_to_maskrcnn)

if path_to_bagfile_dir is None:
    detect(detector, None, verbose=verbose)
else:
    bagfiles = [join(path_to_bagfile_dir, f) for f in listdir(path_to_bagfile_dir)]
    bagfiles = [bag for bag in bagfiles if isfile(bag)]
    for bag in bagfiles:
        print("detecting bag: " + bag)
        detect(detector, bag, verbose=verbose)





