#!/usr/bin/env python

"""image_proc.py
Functions to process and measure image features."""

import numpy as np
from better_focusing import sharpness
import gen_helpers as h


def crop_img_into_n(bgr_arr, n):
    """Splits the bgr array into n equal sized chunks by array slicing.
    :param bgr_arr: The 3D array to split, in the format (no. of row pixels in
    image, no. of column pixels in image, 3 BGR values).
    :param n: The number of equal sized chunks to split the array into.
    :return: A list in the format [tuple of resolution of main image,
    tuple of number of subimages per (row, column), list of lists of each
    sub-image array."""

    [x_res, y_res, tot_res] = get_res(bgr_arr)

    # Round n to the nearest factor of the total resolution so the image is
    # cropped while maintaining the same aspect ratio per crop.
    num_subimages = h.closest_factor(tot_res, n)
    print "Splitting image into {} sub-images.".format(num_subimages)

    [x_subimgs, y_subimgs] = get_num_subimages((x_res, y_res), num_subimages)
    pixel_step = get_pixel_step((x_res, y_res), (x_subimgs, y_subimgs))

    # Split image along y, then x. Lists have been used here instead of
    # arrays because although it may be slower, memory leaks are less likely.
    split_y = np.split(bgr_arr, y_subimgs, axis=0)
    split_xy = []
    for row in split_y:
        split_x = np.split(row, x_subimgs, axis=1)
        split_xy.append(split_x)

    # split_xy is a list of lists containing subsections of the 3D array
    # bgr_arr.
    return [pixel_step, split_xy]


def sharpness_vs_position(pixel_step, list_of_arrs):
    """Calculates the sharpness of a set of sub-images as a function of
    position.
    :param pixel_step: A tuple of no. of pixels per sub-image along (x, y).
    :param list_of_arrs: A list of lists containing the sub-image arrays,
    in the format returned by crop_img_into_n.
    :return: An array in the form [x_position, y_position, sharpness],
    where each of the three are column vectors."""

    sharpness_arr = []
    for arr_list in list_of_arrs:
        sharpness_col = []
        for arr in arr_list:
            sharpness_col.append(sharpness(arr))
        sharpness_arr.append(sharpness_col)

    sharpness_arr = np.array(sharpness_arr)

    it = np.nditer(sharpness_arr, flags=['multi_index'])
    results = []
    while not it.finished:
        results.append([it.multi_index[0] * pixel_step[0] + pixel_step[0]/2,
                        it.multi_index[1] * pixel_step[1] + pixel_step[1]/2,
                        it[0]])
        it.iternext()

    return np.array(results)


def get_res(bgr_arr):
    """Get the x, y and total resolutions of the image in pixels.
    :param bgr_arr: The 3D array to split, in the format (no. of row pixels in
    image, no. of column pixels in image, 3 BGR values).
    :return: The list [x_pixels, y_pixels, total_square_pixels]."""
    return [float(bgr_arr.shape[1]), float(bgr_arr.shape[0]),
            float(np.product(bgr_arr.shape[:2]))]


def get_pixel_step(res, num_sub_imgs):
    """Calculates the number of pixels per subimage along x and y.
    :param res: A tuple of the resolution of the main image along (x, y).
    :param num_sub_imgs: A tuple of no. of subimages along x, y e.g. (4, 3).
    :return: A tuple of (number of x pixels per sub-image, number of y
    pixels per sub-image)."""
    return res[0] / num_sub_imgs[0], res[1] / num_sub_imgs[1]


def get_num_subimages(res, tot_subimages):
    """Returns a tuple of the number of subimages along x and y such that
    aspect ratio is maintained.
    :param res: (x_resolution, y_resolution).
    :param tot_subimages: Total number of subimages to split the main image
    into."""
    x_split = np.sqrt(res[0] / res[1] * tot_subimages)
    y_split = tot_subimages / x_split
    return x_split, y_split

# Code to test the sharpness vs position plot, buts needs modification.
#if __name__ == "__main__":
#    with picamera.PiCamera() as camera:
#        camera.resolution = (640, 480)
#        camera.start_preview()
#        time.sleep(3)   # Let camera to receive image clearly before
    # capturing.
#        capture_to_bgr_array(camera)
#        camera.stop_preview()   # Preview must be stopped afterwards to
#        # prevent Pi freezing on camera screen.
#
#        split_img_data = crop_img_into_n(capture_to_bgr_array(camera), 4800)
#        plotting_data = sharpness_vs_position(*split_img_data)
#
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        [X, Y, Z] = [plotting_data[:, 0], plotting_data[:, 1],
#                     plotting_data[:, 2]]
#        ax.plot_wireframe(X, Y, Z)
#
#        plt.show()
#


