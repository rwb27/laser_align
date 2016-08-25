#!/usr/bin/env python

"""Contains base-level functions that are required for the others to run."""

import time as t
import numpy as np


# TODO MAKE THE MOVE_CAPTURE METHOD RECOGNISE ARBITRARY POSITIONS FUNCTIONS.
def move_capture(exp_obj, iter_dict, func_list=None,
                 save_mode='save_subset', end_func=None):
    """Function to carry out a sequence of measurements as per iter_list,
    take an image at each position, post-process it and return a final result.

    :param exp_obj: The experiment object.

    :param iter_dict: A dictionary of lists of 2-tuples to indicate all
    positions where images should be taken:
        {'x': [(n_x1, step_x1), ...], 'y': [(n_y1, step_y1), ...], 'z': ...],
    where each key indicates the axis to move, 'n' is the number of times
    to step (resulting in n+1 images) and 'step' is the number of microsteps
    between each subsequent image. So {'x': [(3, 100)]} would move only the
    x-axis of the microscope, taking 4 images at x=-150, x=-50, x=50,
    x=150 microsteps relative to the initial position. Note that:
    - Not all keys need to be specified.
    - All lists must be the same length.
    - All measurements will be taken symmetrically about the initial position.

    The position of each tuple in the list is important. If we have
    {'x': [(1, 100), (0, 100)],
     'y': [(2,  50), (3,  40)]},
    then tuples of the same index from each list will be combined into an
    array. This means that for the 0th index of the list, for x we have the
    positions [-50, 50] and [0] and for y [-50, 0, 50] and [-60, -20, 20, 60]
    respectively. [-50, 50] and [-50, 0, 50] will be combined to get the
    resulting array of [[-50, -50], [-50, 0], [-50, 50], [50, -50], [50, 0],
    [50, 50]], and the latter two to get [[0, -60], [0, -20], [0, 20],
    [0, 60]]. These are all the positions the stage will move to (the format
    here is [x, y]), iterating through each array in the order given.

    If you prefer to take images once for all the 'x' and, separately, once
    for all the 'y', run this function twice, once for 'x', once for 'y'.

    :param func_list: The post-processing curried function list, created using
    the bake function in the _experiments.py module.

    :param save_mode: How to save at the end of each iteration.
    - 'save_each': Every single measurement is saved: {'x': [(3, 100)]}
      would result in 4 post-processed results being saved, which is useful if
      the post-processed results are image arrays.
    - 'save_final': Every single measurement is made before the entire set of
      results, as an array, is saved along with their positions, in the
      format [[x-column], [y-column], [z-column], [measurements-column]]. This
      is good for the post-processed results that are single numerical value.
    - 'save_subset': Each array is measured before being saved (for example, in
      the description of iter_dict, there are two arrays being iterated
      through).
    - None: Data is not saved at all, but is returned. Might be useful if
      this is intermediate step.

    :param end_func: A curried function, which is executed on the array of
    final results. This can be useful to move to a position where the final
    measurement is maximised. Note: if save_mode is 'save_each', the results
    array will be empty so end_func must be None."""

    # Verify iter_dict format:
    valid_keys = ['x', 'y', 'z']
    len_lists = []
    assert len(iter_dict) <= len(valid_keys)
    for key in iter_dict.keys():
        for tup in iter_dict[key]:
            print tup
            assert len(tup) == 2, 'Invalid tuple format.'
        assert np.any(key == np.array(valid_keys)), 'Invalid key.'
        # For robustness, the lengths of the lists for each key must be
        # the same length.
        len_lists.append(len(iter_dict[key]))
    if len(len_lists) > 1:
        assert [len_lists[0] == element for element in len_lists[1:]], \
            'Lists of unequal lengths.'

    # Get initial position, which may not be [0, 0, 0] if scope object
    # has been used for something else prior to this experiment.
    initial_position = exp_obj.scope.stage.position
    # A set of results to be collected if save_mode == 'save_final'.
    results = []

    for i in xrange(len_lists[0]):
        # For the length of each list, combine every group of tuples in the
        # same position to get array of positions to move to.
        move_by = {}
        for key in valid_keys:
            try:
                (n, steps) = iter_dict[key][i]
                move_by[key] = np.linspace(-n / 2. * steps, n / 2. * steps,
                                           n + 1)
            except KeyError:
                # If key does not exist, then keep this axis fixed.
                move_by[key] = np.array([0])
        print "move-by", move_by
        # Generate array of positions to move to.
        pos = _positions_maker(x=move_by['x'], y=move_by['y'],
                               z=move_by['z'], initial_pos=initial_position)
        try:
            # For each position in the range specified, take an image, apply
            # all the functions in func_list on it, then either save the
            # measurement if save_mode = 'save_final', or append the
            # calculation to a results file and save it all at the end.
            while True:
                next_pos = next(pos)  # This returns StopIteration at end.
                print next_pos
                exp_obj.scope.stage.move_to_pos(next_pos)

                mmt = exp_obj.scope.sensor.read()
                processed = mmt

                # Post-process.
                if func_list is not None:
                    for function in func_list:
                        processed = function(processed)

                # Save this array in HDF5 file.
                if save_mode == 'save_each':
                    exp_obj.gr.create_dataset('post_processed', attrs={
                        'Position': exp_obj.scope.stage.position},
                        data=processed)
                else:
                    # The curried function and 'save_final' both use the
                    # array of final results.
                    results.append([elapsed(exp_obj.scope.start),
                                    exp_obj.scope.stage.position[0],
                                    exp_obj.scope.stage.position[1],
                                    exp_obj.scope.stage.position[2],
                                    processed])

        except StopIteration:
            # Iterations finished - save the subset of results and take the
            # next set.
            if save_mode == 'save_subset':
                # Save after every array of motions.
                results = np.array(results, dtype=np.float)
                exp_obj.gr.create_dataset('brightness_subset', data=results)

        except KeyboardInterrupt:
            print "Aborted, moving back to initial position."
            exp_obj.scope.stage.move_to_pos(initial_position)
            exp_obj.wait_or_stop(10)

    results = np.array(results, dtype=np.float)
    if save_mode == 'save_final':
        exp_obj.gr.create_dataset('brightness_final', data=results)
    elif save_mode is None:
        return results
    elif save_mode != 'save_each' and save_mode != 'save_subset':
        raise ValueError('Invalid save mode.')

    if end_func is not None:
        if save_mode != 'save_each':
            # Process the result and return it.
            try:
                return end_func(results)
            except:
                raise NameError('Invalid function name.')
        elif save_mode == 'save_each':
            raise ValueError('end_func must be None if save_mode = '
                             '\'save_each\', because the results array is '
                             'empty.')


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


def elapsed(start_time):
    return t.time() - start_time


def unchanged(arg):
    """Returns the single input argument; the default function for image
    post-processing to return the input array unchanged."""
    return arg


def _positions_maker(x=np.array([0]), y=np.array([0]), z=np.array([0]),
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


# Functions to perform on an array of results at the end.
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
    """Given a results array from move_capture, and an axis to look at,
    fits that axis's positions with the measured quantities to a parabola,
    returns the error and moves the scope_obj stage to the maximum point of the
    parabola. Experiment can only be performed on a single axis."""
    x = results_arr[:, ['x', 'y', 'z'].index(axis)]
    y = results_arr[:, 3]
    assert np.where(y == np.max(y)) != 0 and \
        np.where(y == np.max(y)) != len(y), 'Extrapolation occurred - ' \
                                            'measure over a wider range.'
    print "x", x, "y", y
    coeffs = np.polyfit(x, y, 2)    # Find some way to get errors from this.
    print "coeffs", coeffs
    coeffs_deriv = np.array([2, 1]) * coeffs[:2]    # Simulate differentiation.
    x_stat = -coeffs_deriv[1] / coeffs_deriv[0]

    new_pos = results_arr[0, :3]     # Get the values that stay the same.
    new_pos[['x', 'y', 'z'].index(axis)] = x_stat   # Overlay with max.
    print "new pos parmax", new_pos
    scope_obj.stage.move_to_pos(new_pos)
    return


def move_to_original(results_arr, scope_obj, initial_pos):
    scope_obj.stage.move_to_pos(initial_pos)
