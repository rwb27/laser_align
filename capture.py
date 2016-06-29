"""capture.py
Contains functions to capture images, either in compressed or raw BAYER
form."""

import io
import picamera.array
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def capture_compressed(cam):
    """Capture a compressed frame from a camera and output to a BGR numpy
    array.
    :param cam: The camera object.
    :return The 3-channel image from the numpy array."""
    stream = io.BytesIO()
    cam.capture(stream, format='jpeg')  # get an image, see
    # picamera.readthedocs.org/en/latest/recipes1.html
    data = np.fromstring(stream.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, 1)


def capture_raw(cam):
    """Capture a raw bayer image, de-mosaic it and output a BGR numpy array.
    :param cam: The camera object.
    :return The 3-channel image from the numpy array."""
    with picamera.array.PiBayerArray(cam) as output:
        # Normally bayer images are not processed via white balance, etc in
        # the camera and are thus of much worse quality if this were done.
        # But the combination of lenses in the Pi means that the reverse is
        # true.
        cam.capture(output, 'jpeg', bayer=True)
        data = np.fromstring(output.getvalue(), dtype=np.uint8)
        output = cv2.imdecode(data, 1)
        #raw_array = (output.demosaic() >> 2).astype(np.uint8)
        plt.ion()
        plt.imshow(output)
        time.sleep(200)
        return output

if __name__ == '__main__':
    with picamera.PiCamera() as camera:
        capture_raw(camera)
