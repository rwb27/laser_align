"""Use functions here to calculate the Allan variance of brightness vs time
measurements."""

from scipy.integrate import simps
import numpy as np


def allan(x, y, tau):
    """Calculate Allan variance using Simpson's rule for numerical integration.
    :param x: The x-series as an array.
    :param y: The y-array.
    :param tau: The time chunks to divide the series into.
    :return: The data for the allan variance."""

    # Change tau to a value that is an close to an exact factor of the size of
    # the data series.
    print np.round(np.max(x) - np.min(x)), tau
    tau = closest_factor(np.round(np.max(x) - np.min(x)), tau)
    mean_indices = np.arange(0, np.round(np.max(x) - np.min(x))/tau, 1)
    for index in mean_indices:
        mean_positions = simps(y[])


def closest_factor(f, n):
    """Returns the factor of f that is closest to n. If n is equidistant
    from two factors of f, the smallest of the two factors is returned."""
    return _one_disallowed(_factors(f), n)


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

allan(np.array([0, 1,2,3, 4, 5, 6]), np.array([3, 4,5,6,7,8,9]), 2)