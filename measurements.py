#!/usr/bin/env python

"""measurements.py
Contains functions to perform single measurements on BGR/greyscale arrays
that have been appropriately processed."""

import numpy as np
from scipy import ndimage as sn

import helpers as h
import image_proc


def sharpness_lap(bgr_array):
    """Calculate sharpness as the Laplacian of the BGR image array.
    :param bgr_array: The 3-channel image to calculate sharpness for.
    :return: The mean Laplacian.
    """
    image_bw = np.mean(bgr_array, 2)
    image_lap = sn.filters.laplace(image_bw)
    return np.mean(np.abs(image_lap))


def get_size(bgr_arr):
    """Get the x, y and total resolutions of the image in pixels. This is
    subtly different to getting a camera's resolution because it acts on an
    array, independently of any camera object.
    :param bgr_arr: The 3D array to split, with shape in the format (no. of
    row pixels in image, no. of column pixels in image, 3 BGR values). This
    is the format the images captured are in.
    :return: The list [x_pixels, y_pixels, total_square_pixels]."""
    # NOTE: The x and y pixel co-ordinates are swapped around in the bgr
    # array, so need to swap them around when calling the shape.
    return [float(bgr_arr.shape[1]), float(bgr_arr.shape[0]),
            float(np.product(bgr_arr.shape[:2]))]


def brightness(arr):
    """Calculates the mean brightness of an array.
    :param arr: A BGR or greyscale array.
    :return: The scalar brightness value."""

    # If the array is BGR, convert to greyscale before calculating brightness.
    if len(arr.shape) == 3:
        arr = image_proc.make_greyscale(arr, greyscale=True)
    elif len(arr.shape) != 2:
        raise ValueError('Array has invalid shape: {}'.format(arr.shape))

    return np.mean(arr)


def get_pixel_step(res, num_sub_imgs):
    """Calculates the number of pixels per sub-image along x and y.
    :param res: A tuple of the resolution of the main image along (x, y).
    :param num_sub_imgs: A tuple of no. of subimages along x, y e.g. (4, 3).
    :return: A tuple of (number of x pixels per sub-image, number of y
    pixels per sub-image)."""
    return res[0] / num_sub_imgs[0], res[1] / num_sub_imgs[1]


def get_num_subimages(res, tot_subimages):
    """Returns a tuple of the number of subimages along x and y such that
    aspect ratio is maintained. Used in crop_img_into_n.
    :param res: (x_resolution, y_resolution).
    :param tot_subimages: Total number of subimages to split the main image
    into."""
    x_split = np.sqrt(res[0] / res[1] * tot_subimages)
    y_split = tot_subimages / x_split
    return x_split, y_split
