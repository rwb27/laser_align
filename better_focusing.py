#!/usr/bin/env python

"""better_focusing.py
Auto-focuses the camera using an algorithm that compares the Laplacian of
the image sharpness, which is a better algorithm."""

import picamera
import cv2
import io
import scipy.ndimage as sn
import numpy as np
import time
import matplotlib.pyplot as plt
import scope_stage as s
from helpers import factors
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


def capture_to_bgr_array(cam):
    """Capture a frame from a camera and output to a numpy array.
    :param cam: The camera object.
    :return The 3-channel image from the numpy array."""
    stream = io.BytesIO()
    cam.capture(stream, format='jpeg')  # get an image, see
    # picamera.readthedocs.org/en/latest/recipes1.html
    data = np.fromstring(stream.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, 1)


def crop_bgr_array(bgr_arr, n):
    """Splits the bgr array into n equal sized chunks by array slicing.
    :param bgr_arr: The array to split, in the format (no. of row pixels in
    image, no. of column pixels in image, 3 BGR values).
    :param n: The number of equal sized chunks to split the array into.
    :return: A list in the format [tuple of resolution of main image,
    tuple of number of subimages per (row, column), list of lists of each
    sub-image array."""

    # Round n to the nearest factor of the total resolution so the image is
    # cropped while maintaining the same aspect ratio per crop.
    total_res = np.product(bgr_arr.shape[:2])
    res = (bgr_arr.shape[1], bgr_arr.shape[0])

    factor_list = factors(total_res)
    num_subimages = min(factor_list, key=lambda x: abs(x - n))
    print "Splitting image into {} sub-images (rounding ensures aspect " \
          "ratio is maintained).".format(num_subimages)

    # No. of subimages/row.
    x_split = np.sqrt(float(bgr_arr.shape[1]) / bgr_arr.shape[0] *
                      num_subimages)
    y_split = num_subimages / x_split
    # List have been used here instead of arrays because although it may be
    # slower, memory leaks are less likely.
    split_y = np.split(bgr_arr, y_split, axis=0)
    split_xy = []
    for row in split_y:
        split_x = np.split(row, x_split, axis=1)
        split_xy.append(split_x)

    return [res, (x_split, y_split), split_xy]


def sharpness_vs_position(main_img_res, no_sub_imgs, list_of_arrs):
    """Calculates the sharpness of a set of sub-images as a function of
    position.
    :param main_img_res: A tuple of the resolution of the main image,
    e.g. (640, 480), where measurements are in pixels.
    :param no_sub_imgs: A tuple of the number of sub-images along x and y,
    e.g. (4, 3).
    :param list_of_arrs: A list of lists containing the sub-image arrays,
    in the format returned by crop_bgr_array.
    :return: An array in the form [x_position, y_position, sharpness],
    where of the three are column vectors."""

    sharpness_arr = []
    for arr_list in list_of_arrs:
        sharpness_col = []
        for arr in arr_list:
            sharpness_col.append(sharpness(arr))
        sharpness_arr.append(sharpness_col)

    sharpness_arr = np.array(sharpness_arr)

    pixel_steps = (float(main_img_res[0])/no_sub_imgs[0], float(main_img_res[
        1])/no_sub_imgs[1])

    it = np.nditer(sharpness_arr, flags=['multi_index'])
    results = []
    while not it.finished:
        results.append([it.multi_index[0] * pixel_steps[0] + pixel_steps[0]/2,
                        it.multi_index[1] * pixel_steps[1] + pixel_steps[1]/2,
                        it[0]])
        it.iternext()

    return np.array(results)


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
        time.sleep(3)   # Let camera to receive image clearly before capturing.
        capture_to_bgr_array(camera)
        camera.stop_preview()   # Preview must be stopped afterwards to
        # prevent Pi freezing on camera screen.

        split_img_data = crop_bgr_array(capture_to_bgr_array(camera), 4800)
        plotting_data = sharpness_vs_position(*split_img_data)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        [X, Y, Z] = [plotting_data[:, 0], plotting_data[:, 1],
                     plotting_data[:, 2]]
        ax.plot_wireframe(X, Y, Z)

        plt.show()
        #stage = s.ScopeStage()
        #backlash = 128

        #for step, n in [(1000, 20), (200, 10), (200, 10), (100, 12)]:
        #    sharpness_list = []
        #    positions = []

        #    stage.focus_rel(-step * n / 2 - backlash)
        #    stage.focus_rel(backlash)
        #    sharpness_list.append(sharpness(capture_to_bgr_array(camera)))
        #    positions.append(stage.position[2])

        #    for i in range(n):
        #        stage.focus_rel(step)
        #        sharpness_list.append(sharpness(capture_to_bgr_array(camera)))
        #        positions.append(stage.position[2])

        #    newposition = np.argmax(sharpness_list)

        #    stage.focus_rel(-(n - newposition) * step - backlash)
        #    stage.focus_rel(backlash)

        #    print sharpness_list

        #    plt.plot(positions, sharpness_list, 'o-')

        #plt.xlabel('position (Microsteps)')
        #plt.ylabel('Sharpness (a.u.)')
        #time.sleep(5)

    #plt.show()

    print "Done :)"

'''
plt.figure()

plt.imshow(image_bw)

plt.figure()

plt.imshow(image_lap)

plt.show()
'''
