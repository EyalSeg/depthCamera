import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import math

import importlib.util

class MaskrcnnObjectDetector:
    def __init__(self, pathToMrcnn, pathToCoco, pathToWeights, modelDir):
        sys.path.append(pathToMrcnn)
        from mrcnn import utils
        import mrcnn.model as modellib
        from mrcnn import visualize

        self.visualize = visualize

        sys.path.append(pathToCoco)
        import coco

        # Download COCO trained weights from Releases if needed
        if not os.path.exists(pathToWeights):
            utils.download_trained_weights(pathToWeights)

        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=modelDir, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(pathToWeights, by_name=True)

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']

    # results: {rois, masks, class_ids, scores}
    # each attribute is an array, with matching order. for each object detected of index i:
    #       count: number of results
    #       rois[i] is it's bounding box, [x1, y1, x2, y2]
    #       masks[j,k,i] is a boolean indicating if the object i is present at index [j, k] of the image
    #           to get the mask of the object i: masks[:, :, i]
    #       class_ids[i] is the id of the object class, matching self.class_names
    #       scores[i] is how certain the detection is
    def detect(self, image, verbose = 1):
        results = self.model.detect([image], verbose= verbose)

        r = results[0]
        r["count"] = r["rois"].shape[0]
        r["coordinates"] = [self.center_of_mass(r["masks"][:, :, i])
                            for i in range(r["count"])]
        return r

    def display(self, image, results):

        skimage.io.imshow(image)

        #ax = plt.subplot()

        self.visualize.display_instances(
            image,
            results["rois"],
            results["masks"],
            results["class_ids"],
            self.class_names,
            results["scores"],
            #ax=ax pass a subplot to ax in order to cancel visualize's auto display
        )

        # draw the center of masses in a new plot
        ax = plt.subplot()
        ax.axis('off')
        ax.imshow(image)
        for i in range(results["count"]):
            circle = plt.Circle(results["coordinates"][i], 5, color="blue")
            ax.add_artist(circle)


        plt.show()

    def center_of_mass(self, mask):
        ints = mask.astype(int)
        total_sum = ints.sum()

        weights_i = [ints[i, j] * i for i in range(ints.shape[0]) for j in range(ints.shape[1])]
        sum_i = math.ceil(sum(weights_i) / total_sum)

        weights_j = [ints[i, j] * j for i in range(ints.shape[0]) for j in range(ints.shape[1])]
        sum_j = math.ceil(sum(weights_j) / total_sum)

        return (sum_j, sum_i)
