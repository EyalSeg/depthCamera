from RealSenseLoop import AsyncRealSenseLoop, FramesConsumer
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imshow

def pixle_to_depthpoint(row, col, frames):
    depth = frames.depth_frame.get_distance(col, row)
    depth_point = rs.rs2_deproject_pixel_to_point(
        frames.depth_intrin, [col, row], depth)

    return depth_point


with AsyncRealSenseLoop() as loop:
    # Subscribing to the loop will call the callbacks on every frame

    # sometimes, however, it might be easier to use a consumer instead:
    consumer = FramesConsumer(loop)

    frames = consumer.consume()

    width = frames.depth_frame.get_width()
    height = frames.depth_frame.get_height()

    coordinates = np.array([pixle_to_depthpoint(height//2, col, frames) for col in range(width)])

    xs = coordinates[:, 0]
    ys = coordinates[:, 1]
    zs = coordinates[:, 2]
    plt.plot(xs, zs, '--b', label="raw coordinates")

    # notice how a few faulty reads skew the results

    mean = np.mean(zs, axis=0)
    sd = np.std(zs, axis=0)

    # replace everything above/below to standard deviations with the median
    zs = [z if (z > mean - 2 * sd) else mean for z in zs]
    zs = [z if (z < mean + 2 * sd) else mean for z in zs]
    print (zs)

    plt.plot(xs, zs, '--r', label="normalized z")
    plt.legend(loc="lower left")
    plt.show()



# note that the loop will end when we leave the "with" statement.