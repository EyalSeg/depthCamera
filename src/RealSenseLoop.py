import pyrealsense2 as rs
import threading
import numpy as np
import concurrent.futures
import time

import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# See the source example in :
# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py

def frames_to_images(frames):
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth = np.asanyarray(depth_frame.get_data())
    color = np.asanyarray(color_frame.get_data())

    return color, depth


class CameraFrame:
    def __init__(self, profile, frames):

        self.frames = frames
        self.profile = profile

        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        self.depth_frame = frames.get_depth_frame()
        self.color_frame = frames.get_color_frame()

        self.color_image = np.asanyarray(self.color_frame.get_data())
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.depth_scale = depth_scale

        self.depth_intrin = self.depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrin = self.color_frame.profile.as_video_stream_profile().intrinsics
        self.depth_to_color_extrin = self.depth_frame.profile.get_extrinsics_to(self.color_frame.profile)

    def get_object_position(self, mask):
        points = self.mask_to_points(mask)
        points = self.normalize_points(points)

        x = np.mean(points[:, 0])
        y = np.mean(points[:, 1])
        z = np.mean(points[:, 2])

        point = (x, y, z)

        return point


    def normalize_points(self, points, m=1.8):
        points = self.normalize_points_by_coordinate(points, 0)
        points = self.normalize_points_by_coordinate(points, 1)
        points = self.normalize_points_by_coordinate(points, 2)

        return points

    def normalize_points_by_coordinate(self, points, coordinate_index, m=1.8):
        mean = np.mean(points[:,coordinate_index])
        sd = np.std(points[:,coordinate_index])

        return points[abs(points[:, coordinate_index] - mean) < m * sd]


    def mask_to_points(self, mask):
        pixles = np.argwhere(mask)
        coordinates = np.array([self.pixle_to_point(*pixle) for pixle in pixles])

        # remove (0,0,0)
        coordinates = coordinates[np.any(coordinates != 0, axis=1)]

        return coordinates


    def pixle_to_point(self, row, col):
        depth = self.depth_frame.get_distance(col, row)
        depth_point = rs.rs2_deproject_pixel_to_point(
            self.depth_intrin, [col, row], depth)

        return depth_point





class AsyncRealSenseLoop:
    def __init__(self, bagfile=None):
        self.onNewImage = []
        self.onStop = []
        self.isRunning = False

        self.worker_thread = None
        self.config = self.createConfig(bagfile)


    def __enter__(self):
        self.isRunning = True
        self.worker_thread = threading.Thread(target=self.loop, daemon=False)
        self.worker_thread.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.isRunning:
            return

        self.isRunning = False

        self.worker_thread.join()
        self.worker_thread = None

        for callback in self.onStop:
            callback()

    def createConfig(self, path_to_bag = None):
        config = rs.config()
        if path_to_bag is not None:
            rs.config.enable_device_from_file(config, path_to_bag)

        return config

    def loop(self):
        pipe = rs.pipeline()
        profile = pipe.start(self.config)

        align_to = rs.stream.color
        align = rs.align(align_to)

        # Skip first frames to give the Auto-Exposure time to adjust
        for x in range(20):
            pipe.wait_for_frames()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            while self.isRunning:
                frames = pipe.wait_for_frames()
                aligned_frames = align.process(frames)

                result = CameraFrame(profile, aligned_frames)

                for subscirber in self.onNewImage:
                    executor.submit(subscirber, result)

        pipe.stop()

class FramesConsumer:
    def __init__(self, loop_obj):
        self.loop = loop_obj

        self.frame = None

        self.loop.onNewImage.append(self.onNewImage)

    def onNewImage(self, frame):
        self.frame = frame

    # WILL busy wait
    def consume(self):
        while self.frame is None:
            time.sleep(0.5)

        frame = self.frame

        return frame


    def __del__(self):
        self.loop.onNewImage.remove(self.onNewImage)



