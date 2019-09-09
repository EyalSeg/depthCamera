from RealSenseLoop import AsyncRealSenseLoop, FramesConsumer
from scipy.misc import imshow


def print_something(*args):
    print("Got new image!")

def display_images(frame):
    imshow(frame.color_image)
    imshow(frame.depth_image)

# Create a new loop object

with AsyncRealSenseLoop() as loop:
    # Subscribing to the loop will call the callbacks on every frame
    loop.onStop.append(lambda : print("Done."))
    loop.onNewImage.append(print_something)

    # sometimes, however, it might be easier to use a consumer instead:
    consumer = FramesConsumer(loop)

    # this will busy-wait until a new image is set
    # however, it will result with the newest image.
    for i in range(5):
        frame = consumer.consume()
        display_images(frame)

# note that the loop will end when we leave the "with" statement.
