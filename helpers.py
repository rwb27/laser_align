#!/usr/bin/env python

"""Contains base-level functions that are required for the others to run."""
import cv2
import numpy as np


def frac_round(number, frac, centre_frac):
    """Function to aid readability, used in crop_section. Note that frac and
    centre_frac are individual elements of the tuples defined in
    crop_section."""
    frac /= 100.
    lower_bound = (number/2.*(1-frac)) + (number * float(centre_frac))
    upper_bound = (number/2.*(1+frac)) + (number * float(centre_frac))

    if lower_bound < 0:
        lower_bound = 0
        raise Warning('Lower bound of cropped image exceeds main image '
                      'dimensions. Setting it to start of main image.')
    if upper_bound > number:
        upper_bound = 1
        raise Warning('Upper bound of cropped image exceeds main image '
                      'dimensions. Setting it to end of main image.')
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


def passer(*args):
    """Does nothing; the empty function object."""
    pass


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


def make_greyscale(frame, greyscale):
    """Makes an image 'frame' greyscale if 'greyscale' is True."""
    # TODO This function appears to flatten the array - it needs reshaping
    # TODO if greyscale is True.
    greyscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return greyscaled if greyscale else frame
