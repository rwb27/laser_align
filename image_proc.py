#!/usr/bin/env python

"""image_proc.py
Functions to process and measure image features."""
import cv2
import numpy as np

import helpers as h


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

    res = h.get_size(arr)[:2]
    if mode == 'spec':
        assert all(var in args.keys() for var in
                   ['mmts', 'dims']), "Invalid keyword arguments."
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
        elif type(args['dims']) is int or type(args['dims']) is float:
            args['dims'] = np.array([args['dims'], args['dims']])
        elif not (type(args['dims']) is np.ndarray and args['dims'].size == 2):
            raise TypeError('\'dims\' is of incorrect type.')

        if args['mmts'] == 'frac':
            # The format for these arrays is [[x_min, x_max], [y_min, y_max]].
            crop_limits = np.array([[0., 1.], [0., 1.]])
            cent_limits = np.array([[-1/2., 1/2.], [-1/2., 1/2.]])
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

        assert (crop_limits[:, 0] <= args['dims']).all() and \
               (args['dims'] <= crop_limits[:, 1]).all(), \
            "Dimensions lie outside allowed crop range."
        assert (cent_limits[:, 0] <= args['centre']).all() and \
               (args['centre'] <= cent_limits[:, 1]).all(), \
            "Centre of cropped images lies outside parent image."
        assert (args['centre'] + args['dims'] / 2. <= cent_limits[:, 1]).all()\
            and (args['centre'] - args['dims'] / 2. >= cent_limits[:, 0]).all(), \
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


def make_greyscale(frame, greyscale=True):
    """Makes an image 'frame' greyscale if 'greyscale' is True."""
    greyscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return greyscaled if greyscale else frame
