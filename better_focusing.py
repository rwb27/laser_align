#!/usr/bin/env python

"""better_focusing.py
Auto-focuses the camera using an algorithm that compares the Laplacian of
the image sharpness."""

import io
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import picamera
import scipy.ndimage as sn
import scope_stage as s


def capture_to_bgr_array(cam):
    """Capture a frame from a camera and output to a numpy array.
    :param cam: The camera object.
    :return The 3-channel image from the numpy array."""
    stream = io.BytesIO()
    cam.capture(stream, format='jpeg')  # get an image, see
    # picamera.readthedocs.org/en/latest/recipes1.html
    data = np.fromstring(stream.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, 1)


def sharpness(rgb_image):
    """Calculate sharpness as the Laplacian of the black and white image.
    :param rgb_image: The 3-channel image to calculate sharpness for.
    :return: The mean Laplacian.
    """
    image_bw = np.mean(rgb_image, 2)
    image_lap = sn.filters.laplace(image_bw)    # Look up this filter.
    return np.mean(np.abs(image_lap))


if __name__ == "__main__":
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.start_preview()
        time.sleep(2)   # Let camera to receive image clearly before capturing.
        stage = s.ScopeStage()
        backlash = 128
        for step, n in [(1000, 20), (200, 10), (200, 10), (100, 12)]:
            sharpness_list = []
            positions = []
            stage.focus_rel(-step * n / 2 - backlash)
            stage.focus_rel(backlash)
            sharpness_list.append(sharpness(capture_to_bgr_array(camera)))
            positions.append(stage.position[2])
            for i in range(n):
                stage.focus_rel(step)
                sharpness_list.append(sharpness(capture_to_bgr_array(camera)))
                positions.append(stage.position[2])
            newposition = np.argmax(sharpness_list)
            stage.focus_rel(-(n - newposition) * step - backlash)
            stage.focus_rel(backlash)
            print sharpness_list
            plt.plot(positions, sharpness_list, 'o-')
        plt.xlabel('position (Microsteps)')
        plt.ylabel('Sharpness (a.u.)')
        time.sleep(5)

    plt.show()

    print "Done :)"

'''
plt.figure()

plt.imshow(image_bw)

plt.figure()

plt.imshow(image_lap)

plt.show()
'''
