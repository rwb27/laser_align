#!/usr/bin/env python

"""image_proc.py
Functions to process and measure image features."""

import numpy as np

import helpers as h
from measurements import get_size, get_pixel_step, get_num_subimages


def crop_section(bgr_arr, frac, centre_frac=(0, 0)):
    """Crops a portion of the image by a specified amount.
    :param bgr_arr: The 3D image array to split, in the format (no. of row
    pixels in image, no. of column pixels in image, 3 BGR values).
    :param frac: A tuple with the percentage of the image along (x,
    y) to retain. For example, x_frac = 30, y_frac = 50 would result in a
    cropped image 15% either side of the centre along x and 25% either side of
    the centre along y.
    :param centre_frac: The centre of the cropped image relative to the main
    image, as a fraction of the (x, y) length of the main image with centre
    at (0, 0). For example, (1/4., 1/2.) would result in the cropped image
    being centred on the top edge of the main image, 3/4 of the way along
    the edge from the top left corner. Checks exist to ensure the crop
    covers only the range of the main image.
    :return: The cropped image BGR array."""

    res = get_size(bgr_arr)[:2]
    if type(frac) is not tuple:
        frac = (frac, frac)
    for each in frac:
        assert 0. <= each <= 1., "{} is an invalid fraction of the image to " \
                          "crop.".format(each)
    for each in centre_frac:
        assert -1/2. <= each <= 1/2., "Centre lies outside range of image."
    for i in range(2):
        pass
    crop = bgr_arr[h.frac_round(res[1], frac[1], centre_frac[1])[0]:
                   h.frac_round(res[1], frac[1], centre_frac[1])[1],
                   h.frac_round(res[0], frac[0], centre_frac[0])[0]:
                   h.frac_round(res[0], frac[0], centre_frac[0])[1], ...]

    print crop.size, bgr_arr.size
    actual_fraction = float(crop.size)/bgr_arr.size * 100
    print r'Cropped {}% of image.'.format(actual_fraction)
    return crop, actual_fraction


def crop_region(grey_arr, dims, centre_pixels=(0, 0)):
    """Crops a region of a greyscaled array given a cropped image size and
    centre.
    :param grey_arr: The 3D image array to split, in the format (no. of row
    pixels in image, no. of column pixels in image, 3 BGR values).
    :param dims: A tuple of the dimensions of the cropped image along (x, y).
    :param centre_pixels: The centre of the cropped image relative to the main
    image in terms of the (x, y) length of the main image with centre
    at (0, 0). For example, (280, 300) would result in the cropped image
    being centred 280 pixels to the right of the centre and 300 pixels above.
    :return: The cropped image array."""

    res = get_size(grey_arr)[:2]
    for i in range(2):
        assert centre_pixels[i] + dims[i]/2. <= res[i]/2. and \
               centre_pixels[i] - dims[i]/2. >= -res[i]/2.
    return grey_arr[int(centre_pixels[1] - dims[1] / 2. + res[1] / 2.):
                    int(centre_pixels[1] + dims[1] / 2. + res[1] / 2.),
                    int(centre_pixels[0] - dims[0] / 2. + res[0] / 2.):
                    int(centre_pixels[0] + dims[0] / 2. + res[0] / 2.), ...]


def crop_img_into_n(bgr_arr, n):
    """Splits the bgr array into n equal sized chunks by array slicing.
    :param bgr_arr: The 3D array to split, in the format (no. of row pixels in
    image, no. of column pixels in image, 3 BGR values).
    :param n: The number of equal sized chunks to split the array into.
    :return: A list in the format [tuple of resolution of main image,
    tuple of number of subimages per (row, column), list of lists of each
    sub-image array."""

    [x_res, y_res, tot_res] = get_size(bgr_arr)

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


def down_sample(array, factor_int=4):
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
    if len(array.shape) == 3:
        # BGR array. Why is dtype uint16?
        bin_y = np.mean(np.reshape(
            array, (array.shape[0], array.shape[1]/factor_int, factor_int, 3),
            order='C'), axis=2, dtype=np.uint16)
        binned = np.mean(np.reshape(bin_y, (
            factor_int, array.shape[0]/factor_int, array.shape[1]/factor_int,
            3), order='F'), axis=0, dtype=np.uint16)
    elif len(array.shape) == 2:
        # Greyscale array.
        bin_y = np.mean(np.reshape(
            array, (array.shape[0], array.shape[1]/factor_int, factor_int),
            order='C'), axis=2, dtype=np.uint16)
        binned = np.mean(np.reshape(bin_y, (
            factor_int, array.shape[0]/factor_int, array.shape[1]/factor_int),
                                    order='F'), axis=0, dtype=np.uint16)
    else:
        raise ValueError('Array has incorrect dimensions.')

    return np.array(binned, dtype=np.uint8)


def crop_array(arr, mode='spec', **args):
    """Perform cropping on a BGR or greyscale array.

    :param arr: A BGR (M x N x 3) or greyscale (M x N) array.
    :param mode: 'spec' to crop a specific section of the image, 'split' to
    split the parent image into several sub-images.
    :param args: Keyword parameters satisfying the requirements of the mode
    used.
        'spec' mode:
        - mmts: 'frac' to measure the cropped image dimensions in terms of
          fractional size relative to the parent image, 'pixel' to
          measure them in raw pixel values.
        - dims: The size of the cropped image along (x, y) in the
          measurement system specified.
        - centre: a tuple of the (x, y) displacement of the centre of the
          cropped image relative to the centre of the parent image, measured
          in the units specified by 'mmts'. np.array([0, 0]) by default.
        - return_actual_crop: True to return the actual size of the
          cropped image (x*y), following the cropped image array. In 'frac'
          mode, this is the fraction of the total_area occupied by the
          cropped image. In 'pixel' mode, this is the size cropped area in
          square pixels. This is False by default.
        'split' mode:
        - n: The
    :return: A tuple of the cropped image array, and other parameters if
    'return_actual_crop' is specified."""

    res = get_size(arr)[:2]
    print res
    if mode == 'spec':
        assert all(var in args.keys() for var in
                   ['mmts', 'dims', 'centre']), "Invalid keyword arguments."
        if 'return_actual_crop' not in args.keys():
            args['return_actual_crop'] = False
        if 'centre' not in args.keys():
            args['centre'] = np.array([0, 0])
        if type(args['dims']) is tuple or type(args['dims']) is list:
            if len(args['dims']) == 2:
                args['dims'] = np.array(args['dims'])
            else:
                raise TypeError('The parameter \'dims\' is of incorrect '
                                'type.')
        elif len([args['dims']]) == 1:
            args['dims'] = np.array([args['dims'], args['dims']])
        else:
            raise TypeError('\'dims\' is of incorrect type.')

        if args['mmts'] == 'frac':
            # The format for these arrays is [[x_min, x_max], [y_min, y_max]].
            crop_limits = np.array([[0., 1.], [0., 1.]])
            cent_limits = np.array([[-1/2., 1/2.], [-1/2., 1/2.]])
            print args['dims'], args['centre']
            [final_xmin, final_xmax, final_ymin, final_ymax] = \
                [h.frac_round(res[1], args['dims'][1], args['centre'][1])[0],
                 h.frac_round(res[1], args['dims'][1], args['centre'][1])[1],
                 h.frac_round(res[0], args['dims'][0], args['centre'][0])[0],
                 h.frac_round(res[0], args['dims'][0], args['centre'][0])[1]]

        elif args['mmts'] == 'pixel':
            crop_limits = np.array([[0, res[0]], [0, res[1]]])
            cent_limits = np.array([[-res[0]/2., res[0]/2.],
                                    [-res[1]/2., res[1]/2.]])
            [final_xmin, final_xmax, final_ymin, final_ymax] = \
                [int(args['centre'][1] - args['dims'][1] / 2. + res[1] / 2.),
                 int(args['centre'][1] + args['dims'][1] / 2. + res[1] / 2.),
                 int(args['centre'][0] - args['dims'][0] / 2. + res[0] / 2.),
                 int(args['centre'][0] + args['dims'][0] / 2. + res[0] / 2.)]

        else:
            raise ValueError('\'mmts\' entry is invalid.')

        assert all(crop_limits[:, 0] <= args['dims']) and \
            all(args['dims'] <= crop_limits[:, 1]), \
            "Dimensions lie outside allowed crop range."
        assert all(cent_limits[:, 0] <= args['centre']) and \
            all(args['centre'] <= cent_limits[:, 1]), \
            "Centre of cropped images lies outside parent image."
        assert all(args['centre'] + args['dims'] / 2. <= cent_limits[:, 1]) \
            and all(args['centre'] - args['dims'] / 2. >= cent_limits[:, 0]), \
            "Cropped image only partially overlaps the parent image."

        crop = arr[final_xmin: final_xmax, final_ymin: final_ymax, ...]

        if args['return_actual_crop']:
            # If 'frac', return the fraction the crop occupies. If 'pixel',
            # return the number of square pixels in the cropped image.
            actual_crop_size = float(crop.size)/arr.size if \
                args['mmts'] == 'frac' else crop.size
            return crop, actual_crop_size
        else:
            return crop

    elif mode == 'split':
        pass
    else:
        raise ValueError('Invalid mode entry.')








if __name__ == '__main__':
    array = np.reshape(np.linspace(0, 750000, 750000), (500, 500, 3))

    print (crop_array(array, mode='spec', mmts='frac', dims=1/2.,
                      centre=np.array([0, 0])) == crop_section(array,
                                                               1/2.)[0]).all()
