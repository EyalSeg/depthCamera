import LineDetector as ln
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

line_samples = [
    {"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 8.5, 9, 9.4],
     "y": None,
     "z": [1, 1.2, 1.3, 1.7, 2.1, 2.8, 4, 6, 9, 13, 16, 20]},

    {"x": [0, 1, 2, 3, 4, 5],
     "y": None,
     "z": [1, 1.2, 1.1, 0.9, 1.2, 1.1]}
]

distance = 1

for line_sample in line_samples:
    x = line_sample["x"]
    y = line_sample["y"]
    z = line_sample["z"]

    if y is None:
        y = [0] * len(x)

    # we ignore most of the line except for the last few people
    x_ = x[-5: ]
    z_ = z[-5: ]

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    plt.plot(x, z, 'bo')

    result = ln.find_line_position(x_, y, z_, distance)
    poly = result["line_polynomial"]
    new_pos = result["next_pos"]

