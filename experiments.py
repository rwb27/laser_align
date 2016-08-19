#!/usr/bin/env python

"""experiments.py
Contains functions to perform a set of measurements and output the results to a
datafile or graph. These functions are built on top of measurements.py,
microscope.py, image_proc.py and data.py."""

import time as t
import numpy as np
from nplab.experiment.experiment import Experiment

import data as d
import end_functions as e
import helpers as h

start = t.time()


class ScopeExp(Experiment):
    """Parent class of any experiments done using the SensorScope object."""

    def __init__(self, microscope, config_file, **kwargs):
        super(ScopeExp, self).__init__()
        self.config_dict = d.make_dict(config_file, **kwargs)
        self.scope = microscope

    def __del__(self):
        del self.scope


class AlongZ(ScopeExp):
    """Measure brightness by varying z only, and move to the position of max
    brightness. Valid kwargs are: mmt_range"""

    def __init__(self, microscope, config_file, group=None,
                 group_name='AlongZ', **kwargs):
        super(AlongZ, self).__init__(microscope, config_file, **kwargs)
        self.attrs = d.sub_dict(self.config_dict, ['mmt_range'],
                                   {'scope_object': str(self.scope.info.name)})
        self.gr = d.make_group(self, group=group, group_name=group_name)
        print self.gr

    def run(self, save_mode='save_final'):
        # At the end, move to the position of maximum brightness.
        end = h.bake(e.max_fifth_col, args=['IMAGE_ARR', self.scope])

        for n_step in self.config_dict['mmt_range']:
            # Allow the iteration to take place as many times as specified
            # in the scope_dict file.
            _move_capture(self, {'z': [n_step]}, save_mode=save_mode,
                          end_func=end)
            print self.scope.stage.position


class RasterXY(ScopeExp):
    """Class to conduct experiments where a square raster scan is taken with
    the photo-diode and post-processed.
    Valid kwargs are: raster_n_step."""

    def __init__(self, microscope, config_file, group=None,
                 group_name='RasterXY', **kwargs):
        super(RasterXY, self).__init__(microscope, config_file, **kwargs)
        self.attrs = d.sub_dict(self.config_dict, ['raster_n_step'],
                                   {'scope_object': str(self.scope.info.name)})
        self.gr = d.make_group(self, group=group, group_name=group_name)

    def run(self, func_list=None, save_mode='save_final'):

        end = h.bake(e.max_fifth_col, args=['IMAGE_ARR', self.scope])

        # Take measurements and move to position of maximum brightness.
        _move_capture(self, {'x': self.config_dict['raster_n_step'],
                             'y': self.config_dict['raster_n_step']},
                      func_list=func_list, save_mode=save_mode, end_func=end)
        print self.scope.stage.position


class Align(ScopeExp):
    """Class to align the spot to position of maximum brightness."""

    def __init__(self, microscope, config_file, group=None,
                 group_name='Align', **kwargs):
        """Valid kwargs are n_steps, parabola_N, parabola_step,
        parabola_iterations."""
        super(Align, self).__init__(microscope, config_file, **kwargs)
        # Valid kwargs are n_steps, parabola_N, parabola_step,
        # parabola_iterations
        self.attrs = d.sub_dict(self.config_dict, [
            'n_steps', 'parabola_N', 'parabola_step', 'parabola_iterations'], {
            'scope_object': str(self.scope.info.name)})
        self.gr = d.make_group(self, group=group, group_name=group_name)

    def run(self, func_list=None, save_mode='save_final'):
        """Algorithm for alignment is to iterate the RasterXY procedure several
        times with decreasing width and increasing precision, and then using
        the parabola of brightness to try and find the maximum point by
        shifting slightly."""

        raster_set = RasterXY(self.scope, self.config_dict, group=self.gr,
                              raster_n_step=self.config_dict['n_steps'])
        # Take measurements and move to position of maximum brightness. All
        # arrays measurements will be taken before moving on.
        raster_set.run(func_list=func_list, save_mode=save_mode)

        par = ParabolicMax(self.scope, self.config_dict, group=self.gr)
        for i in range(self.config_dict["parabola_iterations"]):
            for ax in ['x', 'y']:
                par.run(func_list=func_list, save_mode=save_mode, axis=ax)


class ParabolicMax(ScopeExp):
    """Takes a sequence of N measurements, fits a parabola to them and moves to
    the maximum brightness value. Make sure the microstep size is not too
    small, otherwise noise will affect the parabola shape. kwargs: step_pair"""

    def __init__(self, microscope, config_file, group=None,
                 group_name='ParabolicMax', **kwargs):
        super(ParabolicMax, self).__init__(microscope, config_file, **kwargs)
        self.attrs = d.sub_dict(self.config_dict, [
            'parabola_N', 'parabola_step', 'parabola_iterations'], {
            'scope_object': str(self.scope.info.name)})
        self.gr = d.make_group(self, group=group, group_name=group_name)

    def run(self, func_list=None, save_mode='save_final', axis='x'):
        """Operates on one axis at a time."""
        # Get default values.
        step_pair = (self.config_dict["parabola_N"],
                     self.config_dict["parabola_step"])
        end = h.bake(e.move_to_parmax, args=['IMAGE_ARR', self.scope, axis])
        _move_capture(self, {axis: [step_pair]}, func_list=func_list,
                      save_mode=save_mode, end_func=end)


class DriftReCentre(ScopeExp):
    """Experiment to allow time for the spot to drift from its initial
    position for some time, then bring it back to the centre and measure
    the drift."""

    def __init__(self, microscope, config_file, group=None,
                 group_name='DriftReCentre', **kwargs):
        # kwargs means sleep_for.
        super(DriftReCentre, self).__init__(microscope, config_file, **kwargs)
        self.attrs = d.sub_dict(self.config_dict, [
            'sleep_times', 'n_steps', 'parabola_N', 'parabola_step',
            'parabola_iterations'], extra_entries={'scope_object': str(
            self.scope.info.name)})
        self.gr = d.make_group(self, group=group, group_name=group_name)

    def run(self, func_list=None, save_mode='save_final'):
        """Default is to sleep for 10 minutes."""
        # Do an initial alignment and then take that position as the initial
        # position.

        align = Align(self.scope, self.config_dict, group=self.gr,
                      group_name='DriftReCentre')
        sleep_times = self.config_dict['sleep_times']

        drifts = []
        for i in range(len(sleep_times)):
            # TODO CHANGE TO A QUICK ALIGNMENT FUNCTION
            align.run(func_list=func_list, save_mode=save_mode)
            pos = self.scope.stage.position
            t.sleep(sleep_times[i])
            if i == 0:
                last_pos = pos
            drift = pos - last_pos
            last_pos = pos
            drifts.append([sleep_times[i], drift])

        # Measure the position after it has drifted by working out how much
        # it needs to move by to re-centre it.
        self.gr.create_dataset('Drift', data=np.array(drifts))


class KeepCentred(ScopeExp):
    """Iterate the parabolic method repeatedly after the initial alignment."""

    def __init__(self, microscope, config_file, group=None,
                 group_name='KeepCentred', **kwargs):
        super(KeepCentred, self).__init__(microscope, config_file, **kwargs)
        self.attrs = d.sub_dict(
            self.config_dict, ['n_steps', 'parabola_N', 'parabola_step',
                               'parabola_iterations'], extra_entries={
                'scope_object': str(self.scope.info.name)})
        self.gr = d.make_group(self, group=group, group_name=group_name)

    def run(self, func_list=None, save_mode='save_final'):
        align = Align(self.scope, self.config_dict, group=self.gr,
                      group_name='KeepCentred')
        align.run(func_list=func_list, save_mode=save_mode)

        # TODO Insert a better algorithm here for fine alignment!
        align_fine = Align(self.scope, self.config_dict, group=self.gr,
                           group_name='KeepCentred', n_steps=[[2, 50]])
        while True:
            try:
                align_fine.run(func_list=func_list, save_mode=save_mode)
            except KeyboardInterrupt:
                break


# TODO add hill_climbing and simplex algorithm classes.


# TODO MAKE THE _MOVE_CAPTURE METHOD RECOGNISE ARBITRARY POSITIONS FUNCTIONS.
def _move_capture(exp_obj, iter_dict, func_list=None,
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
    the bake function in the helpers.py module.

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

    for i in range(len_lists[0]):
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
        pos = h.positions_maker(x=move_by['x'], y=move_by['y'],
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
                    results.append([h.elapsed(start),
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
