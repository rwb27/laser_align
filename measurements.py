#!/usr/bin/env python

"""measurements.py
Contains functions to perform single measurements on BGR/greyscale arrays
that have been appropriately processed."""

import numpy as np
from scipy import ndimage as sn

import image_proc as proc


def sharpness_lap(bgr_array):
    """Calculate sharpness as the Laplacian of the BGR image array.
    :param bgr_array: The 3-channel image to calculate sharpness for.
    :return: The mean Laplacian.
    """
    image_bw = np.mean(bgr_array, 2)
    image_lap = sn.filters.laplace(image_bw)
    return np.mean(np.abs(image_lap))


def brightness(arr):
    """Calculates the mean brightness of an array.
    :param arr: A BGR or greyscale array.
    :return: The scalar brightness value."""

    # If the array is BGR, convert to greyscale before calculating brightness.
    if len(arr.shape) == 3:
        arr = proc.make_greyscale(arr, greyscale=True)
    elif len(arr.shape) != 2:
        raise ValueError('Array has invalid shape: {}'.format(arr.shape))

    return np.mean(arr)


