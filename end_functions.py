#!/usr/bin/env python

"""end_functions.py
Functions to perform at the end, after a set of measurements has been taken
from a class in experiments.py."""

import numpy as np


def max_fifth_col(results_arr, scope_obj):
    """Given a results array made of [[times], [x_pos], [y_pos], [z_pos],
    [quantity]] format, moves scope stage to the position with the
    maximum value of 'quantity'."""
    print results_arr
    new_position = results_arr[np.argmax(results_arr[:, 4]), :][1:4]
    print new_position
    scope_obj.stage.move_to_pos(new_position)
    print "Moved to " + str(new_position)
    return


def move_to_parmax(results_arr, scope_obj, axis):
    """Given a results array from _move_capture, and an axis to look at,
    fits that axis's positions with the measured quantities to a parabola,
    returns the error and moves the scope_obj stage to the maximum point of the
    parabola. Experiment can only be performed on a single axis."""
    x = results_arr[:, ['x', 'y', 'z'].index(axis)]
    y = results_arr[:, 3]
    assert np.where(y == np.max(y)) != 0 and \
        np.where(y == np.max(y)) != len(y), 'Extrapolation occurred - ' \
                                            'measure over a wider range.'

    coeffs = np.polyfit(x, y, 2) # Find some way to get errors from this.
    print "coeffs", coeffs
    coeffs_deriv = np.array([2, 1]) * coeffs[:2]    # Simulate differentiation.
    x_stat = -coeffs_deriv[1] / coeffs_deriv[0]

    new_pos = results_arr[0, :3]     # Get the values that stay the same.
    new_pos[['x', 'y', 'z'].index(axis)] = x_stat   # Overlay with max.
    print "new pos parmax", new_pos
    scope_obj.stage.move_to_pos(new_pos)
    return
