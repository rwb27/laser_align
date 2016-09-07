#!/usr/bin/env python

"""Contains base-level functions that are required for the others to run."""

import sys
import time as t
import warnings as w
import numpy as np
from nplab.experiment import Experiment

import baking as b
import data_io as d
from baking import Saturation, NonZeroReading


class ScopeExp(Experiment):
    """Parent class of any experiments done using the SensorScope object."""

    def __init__(self, microscope, config_file, group, included_data,
                 **kwargs):
        super(ScopeExp, self).__init__()
        self.config_dict = d.make_dict(config_file, **kwargs)
        self.scope = microscope
        self.initial_position = self.scope.stage.position
        self.attrs = d.sub_dict(self.config_dict, included_data,
                                {'scope_object': str(self.scope.info.name),
                                 'initial_position': self.initial_position})
        self.gr = d.make_group(self, group=group,
                               group_name=self.__class__.__name__)

    def __del__(self):
        del self.scope


def move_capture(exp_obj, positions_dict, func_list=b.baker(b.unchanged),
                 order_gen=b.baker(b.raster, position_to_pass_through=(0, 3)),
                 save_mode='save_subset', end_func=b.baker(b.unchanged),
                 valid_keys=('x', 'y', 'z'), number=100, delay=0):
    """Function to carry out a sequence of measurements as per iter_list,
    take an image at each position, post-process it and return a final result.

    :param exp_obj: The experiment object.

    :param positions_dict: A dictionary of lists of 2-tuples to indicate all
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
    here is [x, y]), iterating through each array in the order given before
    applying end_func ONCE RIGHT AT THE END. To run after each array,
    run this function multiple times.

    If you prefer to take images once for all the 'x' and, separately, once
    for all the 'y', run this function twice, once for 'x', once for 'y'.

    :param func_list: The post-processing curried function list, created using
    the baker function in the _experiments.py module.

    :param order_gen: A generator object that determines the order to visit
    each position. Takes arguments (x_positions_array, y_positions_arry,
    z_positions_array, intial_position_vector).

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
    array will be empty so end_func must be None.

    :param valid_keys: A tuple of strings containing all keys that are valid
    ways to move. if rotational degrees of freedom are introduced, this can
    include theta, etc.

    :param number: The number of measurements to average over at each position.

    :param delay: The time delay between each individual measurement taken
    at a specific position."""

    b.ignore_saturation = False

    # Verify positions_dict format. The num_arrays is the number of arrays
    # of positions that will be measured in sequence.
    num_arrays = _verify_positions(positions_dict, valid_keys=valid_keys)

    # Get initial position, which may not be [0, 0, 0] if scope object
    # has been used for something else prior to this experiment.
    initial_position = exp_obj.scope.stage.position
    # A set of results to be collected if save_mode == 'save_final'.
    results = []

    for i in xrange(num_arrays):
        # For the length of each list, combine every group of tuples in the
        # same position to get array of positions to move to.
        move_by = {}
        for key in valid_keys:
            try:
                (n, steps) = positions_dict[key][i]
                move_by[key] = np.linspace(-n / 2. * steps, n / 2. * steps,
                                           n + 1)
            except KeyError:
                # If key does not exist, then keep this axis fixed.
                move_by[key] = np.array([0])
        print "move-by", move_by
        # Generate array of positions to move to.
        pos = order_gen(move_by['x'], move_by['y'],
                        move_by['z'], initial_position)
        attrs = {'mmts_per_reading': number, 'delay_between': delay}
        try:
            # For each position in the range specified, take an image, apply
            # all the functions in func_list on it, then either save the
            # measurement if save_mode = 'save_final', or append the
            # calculation to a results file and save it all at the end.
            while True:
                results = read_move_save(exp_obj, pos, func_list, save_mode,
                                         number, delay, results)

        except (StopIteration, KeyboardInterrupt) as e:
            # Iterations finished - save the subset of results.
            if save_mode == 'save_subset' or (save_mode == 'save_final'
                                              and e is KeyboardInterrupt):
                # Save after every array of motions.
                print "Saving captured results."
                results = np.array(results, dtype=np.float)
                exp_obj.gr.create_dataset('brightness_subset', data=results,
                                          attrs=attrs)
            if e is KeyboardInterrupt:
                # Move to original position and exit program.
                print "Aborted, moving back to initial position. Exiting " \
                      "program."
                exp_obj.scope.stage.move_to_pos(initial_position)
                sys.exit()

        except NonZeroReading:
            # After reading a non-zero value stay at that current position,
            # save all results up to that point.
            print "Non-zero value read."

        except Saturation:
            # If the max value has been read, then the user must turn down
            # the gain. Once turned down, this entire move_sequence must be
            # re-measured.
            return move_capture(exp_obj, positions_dict, func_list, order_gen,
                                save_mode, end_func, valid_keys, number, delay)

    results = np.array(results, dtype=np.float)
    b.ignore_saturation = False
    if save_mode != 'save_each':
        if save_mode == 'save_final':
            exp_obj.gr.create_dataset('brightness_final', data=results,
                                      attrs={'mmts_per_reading': number,
                                             'delay_between': delay})
        elif all(save_mode != valid_mode for valid_mode in
                 ['save_subset', None]):
            raise ValueError('Invalid save mode.')

        # Process the result and return it. Remember end_func is unchanged
        # by default, which returns the array as-is.
        return end_func(results)
    elif save_mode == 'save_each':
        if end_func is not None or end_func is not b.baker(b.unchanged):
            w.warn('end_func will not be used when the save_mode is '
                   '\'save_each\', as it is here.')
    else:
        raise ValueError('Invalid save_mode.')


def apply_functions(mmt, func_list):
    """Applies each function from func_list to the unprocessed mmt."""
    processed = mmt

    # Converting to list first allows single functions to be entered.
    try:
        len(func_list)
    except TypeError:
        func_list = [func_list]

    for function in func_list:
        processed = function(processed)

    return processed


def elapsed(starting):
    end = t.time()
    return end - starting


def _verify_positions(position_dict, valid_keys=('x', 'y', 'z')):
    """Function to verify positions_dict format in the function
    move_capture. All items in len_list must be the same for no AssertionErrors
    to be raised. Returns len_lists[0] as the number of position arrays is
    used for iteration in move_capture."""
    len_lists = []
    assert len(position_dict) <= len(valid_keys)
    for key in position_dict.keys():
        for tup in position_dict[key]:
            assert len(tup) == 2, 'Invalid tuple format.'
        assert np.any(key == np.array(valid_keys)), 'Invalid key.'
        # For robustness, the lengths of the lists for each key must be
        # the same length.
        len_lists.append(len(position_dict[key]))
    if len(len_lists) > 1:
        assert [len_lists[0] == element for element in len_lists[1:]], \
            'Lists of unequal lengths.'
    try:
        return len_lists[0]
    except IndexError:
        # An empty positions dictionary has been passed. We want to default
        # to the position (0, 0, 0) only. This is a single 1 x 1 x 1 array
        # of positions.
        return 1


def read_move_save(exp_obj, gen_obj, func_list, save_mode, number, delay,
                   results):
    """Function to repeatedly move to specific positions, take an average
    measurement, save it as appropriate, and move on.
    :param exp_obj: The experiment object.
    :param gen_obj: An instantiated generator function object.
    :param func_list: A list of functions to act on each measurement taken.
    :param save_mode: The string specifying how and when to save the data.
    :param number: The number of measurements to take before averaging these
    into a single result.
    :param delay: The time delay between individual measurements that will
    be averaged.
    :param results: The results list.
    :return: The modified results list."""

    next_pos = next(gen_obj)  # This returns StopIteration at end.
    exp_obj.scope.stage.move_to_pos(next_pos, backlash=5000)

    # Mmt is returned as a tuple of (mean brightness,
    # error_in_mean).
    mmt = exp_obj.scope.sensor.average_n(number, t=delay)

    # Apply functions.
    processed = apply_functions(mmt, func_list)

    # Save this array in HDF5 file.
    if save_mode == 'save_each':
        exp_obj.gr.create_dataset('post_processed', attrs={
            'Position': exp_obj.scope.stage.position,
            'mmts_per_reading': number, 'delay_between': delay},
                                  data=processed)
    else:
        # The curried function and 'save_final' both use the
        # array of final results.
        reading = [elapsed(exp_obj.scope.start),
                   exp_obj.scope.stage.position[0],
                   exp_obj.scope.stage.position[1],
                   exp_obj.scope.stage.position[2],
                   processed[0], processed[1]]
        print reading
        results.append(reading)

    return results
