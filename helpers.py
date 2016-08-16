#!/usr/bin/env python

"""Contains base-level functions that are required for the others to run."""

import numpy as np


def frac_round(number, frac, centre_frac):
    """Converts fraction 'frac' to crop a dimension of an image by, say x,
    centred at centre_frac, to an integer index for the array with 'number'
    indices along that axis. Refer to crop_array for more info."""
    lower_bound = (number/2.*(1-frac)) + (number * float(centre_frac))
    upper_bound = (number/2.*(1+frac)) + (number * float(centre_frac))

    return int(np.round(lower_bound)), int(np.round(upper_bound))


def verify_vector(vector):
    """Checks the input vector has 3 components."""
    r = np.array(vector)
    assert r.shape == (3,), "The variable 'vector' must have 3 components."
    return r


def ccf(list_int, n):
    """Returns the common factor of the integers in the list list_int that
    is closest to n. If two are equally close, the smallest is returned."""
    all_factors = [set(_factors(num)) for num in list_int]
    common_factors = list(set(all_factors[0]).intersection(*all_factors[1:]))
    return _one_disallowed(common_factors, n)


def closest_factor(f, n):
    """Returns the factor of f that is closest to n. If n is equidistant
    from two factors of f, the smallest of the two factors is returned."""
    return _one_disallowed(_factors(f), n)


def unchanged(arg):
    """Returns the single input argument; the default function for image
    post-processing to return the input array unchanged."""
    return arg


def bake(fun, args=None, kwargs=None, position_to_pass_through=0):
    """Returns an object given by the function 'fun' with its arguments,
    known as a curried function or closure. These objects can be passed into
    other functions to be evaluated.

    :param fun: The function object without any arguments specified.
    :param args: A list of the positional arguments.
    :param kwargs: A list of keyword arguments.
    :param position_to_pass_through: See docstring for 'wrapped'.
    :return: The object containing the function with its arguments."""

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    def wrapped(image):
        """Parameter position_to_pass_through specifies the index of the
        parameter 'image' in the sequence of positional arguments for 'fun'."""
        return fun(*(args[:position_to_pass_through] + [image] + args[(
            position_to_pass_through+1):]), **kwargs)

    return wrapped


def positions_maker(x=np.array([0]), y=np.array([0]), z=np.array([0]),
                    initial_pos=np.array([0, 0, 0])):
    """Generator to produce N x 3 array of all possible permutations of 1D
    arrays x and y, such that N = len(x) * len(y). For example x = [1,2] and
    y = [3,4] yields [1, 3, 0], [1, 4, 0], [2, 3, 0], [2, 4, 0] respectively.
    This is added to [0, 0, 0] before being output."""
    i = 0
    while i < x .size:
        j = 0
        while j < y.size:
            k = 0
            while k < z.size:
                yield np.array([x[i], y[j], z[k]]) + initial_pos
                k += 1
            j += 1
        i += 1


def check_defaults(list_of_vars, config_dict, list_of_keys):
    """Check if each variable in list_of_vars is None, and change it to the
    corresponding value in config_dict if so.
    :param list_of_vars: List of variables to check.
    :param config_dict: Configuration dictionary.
    :param list_of_keys: List of respective key names. If nested
    dictionaries, specify each entry as a list, e.g. ['hello', ['one',
    'two'], ...] for the entries dict['hello'], dict['one']['two'], ...
    respectively.
    :return: The modified list_of_vars."""

    assert len(list_of_vars) == len(list_of_keys)

    for i in range(len(list_of_vars)):
        if list_of_vars[i] is None:
            if type(list_of_keys[i]) is list or type(list_of_keys[i]) is tuple:
                # Attempt to treat like a list - if so it must have 2 entries.
                assert len(list_of_keys[i]) == 2
                list_of_vars[i] = config_dict[list_of_keys[i][0]][
                    list_of_keys[i][1]]
            else:
                # The key is not nested.
                list_of_vars[i] = config_dict[list_of_keys[i]]

    return list_of_vars


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


def _one_disallowed(factors, n):
    """For a list of integers 'factors' in ascending order, return the
    number closest to n (choose the smallest if 2 are equidistant) as long as
    it is not 1. If it is 1, return the second closest factor."""
    closest = min(factors, key=lambda x: abs(x - n))
    try:
        return closest if closest != 1 else factors[1]
    except:
        raise Exception('Only common factor is 1. Crop or zero-pad the image '
                        'before down-sampling.')


def _factors(num):
    """Returns the factors of a number, in ascending order as a list."""
    factor_list = list(reduce(list.__add__, ([j, num // j] for j in range(
        1, int(num ** 0.5) + 1) if num % j == 0)))
    factor_list.sort()
    return factor_list
