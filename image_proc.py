#!/usr/bin/env python

"""image_proc.py
Functions to process and measure image features."""

import numpy as np
import gen_helpers as h
from image_mmts import get_res


def crop_centre(bgr_arr, frac):
    """Crops the central portion of the image by a specified amount.
    :param bgr_arr: The 3D image array to split, in the format (no. of row
    pixels in image, no. of column pixels in image, 3 BGR values).
    :param frac: A tuple with the percentage of the image along (x,
    y) to retain. For example, x_frac = 30, y_frac = 50 would result in a
    cropped image 15% either side of the centre along x and 25% either side of
    the centre along y.
    :return: The cropped image BGR array."""

    (x_res, y_res) = get_res(bgr_arr)[:2]
    crop = bgr_arr[_frac_round(y_res, frac)[0]: _frac_round(y_res, frac)[1],
           _frac_round(x_res, frac)[0]: _frac_round(x_res, frac)[1], :]

    actual_fraction = crop.size/bgr_arr.size * 100.
    print r'Cropped the centre {}% of image.'.format(actual_fraction)
    return crop, actual_fraction


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

    [x_subimgs, y_subimgs] = _get_num_subimages((x_res, y_res), num_subimages)
    pixel_step = _get_pixel_step((x_res, y_res), (x_subimgs, y_subimgs))

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


def down_sample(array, factor_int):
    """Down sample a numpy array, such that the total number of pixels is
    reduced by a factor of factor_int**2. This is done by grouping the
    pixels into square blocks and taking the average of each of the B, G, R
    values in each block.
    :return: The down sampled array."""

    # Ensure that factor_int divides into the number of pixels on each side
    # - if not, round to the nearest factor.
    factor_int = h.ccf(array.shape[0:2], factor_int)
    print "Using {} x {} pixel blocks for down-sampling.".format(factor_int,
                                                                 factor_int)
    bin_y = np.mean(np.reshape(
        array, (array.shape[0], array.shape[1]/factor_int, factor_int, 3),
        order='C'), axis=2, dtype=np.uint16)
    binned = np.mean(np.reshape(bin_y, (factor_int, array.shape[0]/factor_int,
        array.shape[1]/factor_int, 3), order='F'), axis=0, dtype=np.uint16)
    return binned


def _frac_round(number, frac):
    """Function to aid readability, used in crop_centre."""
    frac /= 100.
    return int(np.round(number/2.*(1-frac))), int(np.round(number/2.*(1+frac)))


def _get_pixel_step(res, num_sub_imgs):
    """Calculates the number of pixels per subimage along x and y.
    :param res: A tuple of the resolution of the main image along (x, y).
    :param num_sub_imgs: A tuple of no. of subimages along x, y e.g. (4, 3).
    :return: A tuple of (number of x pixels per sub-image, number of y
    pixels per sub-image)."""
    return res[0] / num_sub_imgs[0], res[1] / num_sub_imgs[1]


def _get_num_subimages(res, tot_subimages):
    """Returns a tuple of the number of subimages along x and y such that
    aspect ratio is maintained.
    :param res: (x_resolution, y_resolution).
    :param tot_subimages: Total number of subimages to split the main image
    into."""
    x_split = np.sqrt(res[0] / res[1] * tot_subimages)
    y_split = tot_subimages / x_split
    return x_split, y_split
