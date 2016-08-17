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
