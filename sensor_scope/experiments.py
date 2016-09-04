#!/usr/bin/env python

"""experiments.py
Contains functions to perform a set of measurements and output the results to a
datafile or graph. These functions are built on top of _experiments.py and
data_io.py."""

import time as t
import numpy as np
from scipy.stats import linregress

import _experiments as _exp
import baking as b


class AlongZ(_exp.ScopeExp):
    """Measure brightness by varying z only, and move to the position of max
    brightness. Valid kwargs are: mmt_range"""

    def __init__(self, microscope, config_file, group=None,
                 included_data=('mmt_range',), **kwargs):
        super(AlongZ, self).__init__(microscope, config_file, group,
                                     included_data, **kwargs)

    def run(self, save_mode='save_final'):
        # At the end, move to the position of maximum brightness.
        end = b.baker(b.max_fifth_col, args=['IMAGE_ARR', self.scope,
                                             self.initial_position])

        for n_step in self.config_dict['mmt_range']:
            # Allow the iteration to take place as many times as specified
            # in the scope_dict file.
            _exp.move_capture(self, {'z': [n_step]}, save_mode=save_mode,
                              end_func=end)
            print self.scope.stage.position


class RasterXY(_exp.ScopeExp):
    """Class to conduct experiments where a square raster scan is taken with
    the photo-diode and post-processed.
    Valid kwargs are: raster_n_step."""

    def __init__(self, microscope, config_file, group=None,
                 included_data=('raster_n_step',), **kwargs):
        super(RasterXY, self).__init__(microscope, config_file, group,
                                       included_data, **kwargs)

    def run(self, func_list=b.baker(b.unchanged), save_mode='save_final'):

        end = b.baker(b.max_fifth_col, args=['IMAGE_ARR', self.scope,
                                             self.initial_position])

        # Take measurements and move to position of maximum brightness.
        _exp.move_capture(self, {'x': self.config_dict['raster_n_step'],
                                 'y': self.config_dict['raster_n_step']},
                          func_list=func_list, save_mode=save_mode,
                          end_func=end)
        print self.scope.stage.position


class RasterXYZ(_exp.ScopeExp):
    """Class to conduct experiments where a cubic raster scan is taken with
    the photo-diode and post-processed. MOVES TO ORIGINAL POSITION AFTER DONE
    Valid kwargs are: raster3d_n_step."""

    def __init__(self, microscope, config_file, group=None,
                 included_data=('raster3d_n_step',), **kwargs):
        super(RasterXYZ, self).__init__(microscope, config_file, group,
                                        included_data, **kwargs)

    def run(self, func_list=b.baker(b.unchanged), save_mode='save_final'):
        print self.scope.stage.position
        end = b.baker(b.max_fifth_col, args=['IMAGE_ARR', self.scope,
                                             self.initial_position])

        # Take measurements and move to position of maximum brightness.
        _exp.move_capture(self, {'x': self.config_dict['raster3d_n_step'],
                                 'y': self.config_dict['raster3d_n_step'],
                                 'z': self.config_dict['raster3d_n_step']},
                          func_list=func_list, save_mode=save_mode,
                          end_func=end)
        print self.scope.stage.position


class Align(_exp.ScopeExp):
    """Class to align the spot to position of maximum brightness."""

    def __init__(self, microscope, config_file, group=None, included_data=(
            'n_steps', 'parabola_N', 'parabola_step', 'parabola_iterations'),
                 **kwargs):
        """Valid kwargs are n_steps, parabola_N, parabola_step,
        parabola_iterations."""
        super(Align, self).__init__(microscope, config_file, group,
                                    included_data, **kwargs)

    def run(self, func_list=b.baker(b.unchanged), save_mode='save_final'):
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
        for i in xrange(self.config_dict["parabola_iterations"]):
            for ax in ['x', 'y']:
                par.run(func_list=func_list, save_mode=save_mode, axis=ax)


class ParabolicMax(_exp.ScopeExp):
    """Takes a sequence of N measurements, fits a parabola to them and moves to
    the maximum brightness value. Make sure the microstep size is not too
    small, otherwise noise will affect the parabola shape."""

    def __init__(self, microscope, config_file, group=None, included_data=(
            'parabola_N', 'parabola_step', 'parabola_iterations'), **kwargs):
        super(ParabolicMax, self).__init__(microscope, config_file, group,
                                           included_data, **kwargs)

    def run(self, func_list=b.baker(b.unchanged), save_mode='save_final',
            axis='x'):
        """Operates on one axis at a time."""
        # Get default values.
        step_pair = (self.config_dict["parabola_N"],
                     self.config_dict["parabola_step"])
        end = b.baker(b.move_to_parmax, args=['IMAGE_ARR', self.scope, axis])
        _exp.move_capture(self, {axis: [step_pair]}, func_list=func_list,
                          save_mode=save_mode, end_func=end)


class DriftReCentre(_exp.ScopeExp):
    """Experiment to allow time for the spot to drift from its initial
    position for some time, then bring it back to the centre and measure
    the drift."""

    # TODO ensure you average over the appropriate time for the noise not to
    # TODO matter
    def __init__(self, microscope, config_file, group=None, included_data=(
            'sleep_times', 'n_steps', 'parabola_N', 'parabola_step',
            'parabola_iterations'), **kwargs):
        super(DriftReCentre, self).__init__(microscope, config_file, group,
                                            included_data, **kwargs)

    def run(self, func_list=b.baker(b.unchanged), save_mode='save_final'):
        """Default is to sleep for 10 minutes."""
        # Do an initial alignment and then take that position as the initial
        # position.

        align = Align(self.scope, self.config_dict, group=self.gr)
        hillwalk = HillWalk(self.scope, self.config_dict, group=self.gr)
        sleep_times = self.config_dict['sleep_times']

        drifts = []
        for i in xrange(len(sleep_times)):
            if i == 0:
                align.run(func_list=func_list, save_mode=save_mode)
            else:
                hillwalk.run()
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


class KeepCentred(_exp.ScopeExp):
    """Iterate the parabolic method repeatedly after the initial alignment."""

    def __init__(self, microscope, config_file, group=None, included_data=(
            'n_steps', 'parabola_N', 'parabola_step', 'parabola_iterations'),
                 **kwargs):
        super(KeepCentred, self).__init__(microscope, config_file, group,
                                          included_data, **kwargs)

    def run(self, func_list=b.baker(b.unchanged), save_mode='save_final'):
        raster = RasterXY(self.scope, self.config_dict, group=self.gr,
                          group_name='KeepCentred')
        raster.run(func_list=func_list, save_mode=save_mode)

        # TODO Insert a better algorithm here for fine alignment!
        align_fine = HillWalk(self.scope, self.config_dict, group=self.gr)
        while True:
            try:
                align_fine.run()
            except KeyboardInterrupt:
                break


class TimedMeasurements(_exp.ScopeExp):
    """Experiment to repeatedly measure the average of N measurements at
    the same position, 'count' times with 'time' seconds between each set of 10
    measurements."""
    def __init__(self, microscope, config_file, group=None,
                 included_data=('N', 't'), **kwargs):
        # Valid kwargs are position, how many times and how long to wait
        # between each measurement.
        super(TimedMeasurements, self).__init__(microscope, config_file,
                                                group, included_data, **kwargs)

    def run(self, func_list=b.baker(b.unchanged), save_mode='save_final'):
        order_gen = b.baker(b.fixed_timer, kwargs={
            'count': self.config_dict["N"], 't': self.config_dict["t"]},
                            position_to_pass_through=(0, 3))
        _exp.move_capture(self, {}, order_gen=order_gen,
                          func_list=func_list, save_mode=save_mode,
                          number=300, delay=0)


class BeamWalk(_exp.ScopeExp):
    """Experiment to 'walk' the beam, consisting of a raster scan in XY,
    followed by homing in on the max brightness, adjusting Z slightly in the
    direction of increasing brightness, and repeating."""

    def __init__(self, microscope, config_file, group=None, included_data=(
            'raster_n_step', 'mmt_range'), **kwargs):
        super(BeamWalk, self).__init__(microscope, config_file, group,
                                       included_data, **kwargs)

    def run(self, func_list=b.baker(b.unchanged), save_mode='save_final'):
        for i in range(2):
            # TODO GENERALISE THE NUMBER
            if i == 0:
                raster_2d = RasterXY(self.scope, self.config_dict,
                                     group=self.gr, raster_n_step=[[39, 500]])
            else:
                raster_2d = RasterXY(self.scope, self.config_dict,
                                     group=self.gr)
            raster_2d.run(func_list=func_list, save_mode=save_mode)

            along_z = AlongZ(self.scope, self.config_dict, group=self.gr,
                             mmt_range=self.config_dict["mmt_range"])
            along_z.run(save_mode=save_mode)


class HillWalk(_exp.ScopeExp):
    """Experiment to walk until a 5-point peak is found, and then repeats
    this for each axis, after which the step size is reduced."""

    DRIFT_TIME = 1000

    def __init__(self, microscope, config_file, group=None, included_data=(
            'raster_n_step', 'mmt_range'), **kwargs):
        super(HillWalk, self).__init__(microscope, config_file, group,
                                       included_data, **kwargs)

    def run(self, max_step=100, number=100, delay=0):
        """Process for this method:
        1) For the max max_step size, start with the x axis and continuously
        repeat.
        2) Get position, get brightness as specified by number and delay, add
        this to results.
        3) For each reading, get the last 5 readings if possible. Sort them
        by the position axis that is varying (check that the others aren't,
        and get the sign of their changes. If it increases then decreases
        symmetrically, fit to parabola and move to the calculated maximum. Then
        break out of the loop and start with the next axis.
        4) If this is unsuccessful, get the last 3 readings and check
        that only one position is varying. If brightness has monotonically
        decreased for the last three readings, change direction.
        5) Work out the next position to go to by first ensuring that it has
        not been visited less than DRIFT_TIME seconds ago. Move to the new
        position and repeat for the next axis, and for a smaller max_step size
        when that is finished."""
        results = []
        step_size = max_step
        while step_size > 5:
            for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                direction = 1
                print "axis", axis

                while True:
                    pos = self.scope.stage.position
                    brightness = self.scope.sensor.average_n(number, delay)
                    print pos, brightness
                    results.append([_exp.elapsed(self.scope.start),
                                    pos[0], pos[1], pos[2], brightness[0],
                                    brightness[1]])

                    # To sort by position that has been modified, find the
                    # index of the axis that is changing.
                    axis_index = axis.index(1) + 1
                    try:
                        if self._try_fit_parabola(results, axis_index):
                            break
                    except (IndexError, AssertionError):
                        # Either not enough list elements for this to be a
                        # valid comparison, or more than one axis varies for
                        # the last three mmts.
                        pass

                    try:
                        direction = self._change_direction(
                            results, axis_index, direction)
                        print "direction", direction
                    except (IndexError, AssertionError):
                        # Either not enough list elements for this to be a
                        # valid comparison, or more than one axis varies for
                        # the last three mmts.
                        pass

                    # Move such that already measured positions are ignored
                    # unless a 'significant' time has passed (calculated
                    # using Allan deviation for drift).
                    pos_to_be = pos + (direction * step_size * np.array(axis))
                    i = 0
                    while i < len(results):
                        # TODO THIS DOESN'T WORK - ALREADY MEASURED POINTS
                        # TODO ARE STILL BEING MEASURED.
                        # For each row in the results, check if that
                        # position is present (i.e. it has already been
                        # measured less than DRIFT_TIME seconds ago. If it
                        # does exist, this position does not need to be
                        # re-measured so keep moving in the direction until
                        # this repetition no longer occurs. If pos_to_be is
                        # changed, the loop must restart to get around this
                        # problem. So reset i=0 and don't add 1 to it. The
                        # loop finally exists when that position has not
                        # been found in the entire set in a recent time.
                        if results[i][1:4] == list(pos_to_be) and _exp.elapsed(
                                results[i][0]) < self.DRIFT_TIME:
                            pos_to_be += direction * step_size * np.array(
                                axis)
                            i = 0
                        else:
                            i += 1

                    self.scope.stage.move_to_pos(pos_to_be)
                    print "new position", pos_to_be
            step_size /= 2
            print "step_size", step_size

        self.gr.create_dataset('hill_walk_brightness', data=results, attrs={
            'mmts_per_reading': number, 'delay_between': delay,
            'max_step': max_step })

    @staticmethod
    def _check_sliced_results(results, axis_index, slice_size):
        """Ensure the other position axes haven't changed, and the axis in
        question has all different positions. Return the appropriately
        sliced results.
        :param results: The results array.
        :param axis_index: The index of the position that is varying.
        :param slice_size: The number of the last rows of the results array to
        use."""
        last_rows = np.array(results[-slice_size:])
        assert last_rows.shape[0] == slice_size

        other_indices = []
        for ind in [1, 2, 3]:
            if ind != axis_index:
                other_indices.append(ind)
        assert [np.unique(last_rows[:, other_index]).size == 1 for other_index
                in other_indices]

        assert np.unique(last_rows[:, axis_index]).size == slice_size

        return last_rows

    def _try_fit_parabola(self, results, axis_index):
        last_five_rows = self._check_sliced_results(results, axis_index, 5)
        sort = last_five_rows[np.argsort(last_five_rows[:, axis_index])]

        if np.all(np.sign(np.ediff1d(sort[:, 4])) == np.array(
                [1, 1, -1, -1])):
            axis_string = ['x', 'y', 'z'][axis_index - 1]
            b.move_to_parmax(sort, self.scope, axis_string)
            return True
        else:
            return False

    def _change_direction(self, results, axis_index, direction):
        last_three_rows = self._check_sliced_results(results, axis_index, 3)

        # We want to find if the last 3 position/brightness
        # measurements have a positive/negative gradient. If
        # negative, reverse direction.
        x = last_three_rows[:, axis_index]
        y = last_three_rows[:, 4]
        if np.all(np.sign(np.ediff1d(y)) < 0) and np.ediff1d(np.sign(
                np.ediff1d(x))) == 0:
            # Change directions only if the positions are monotonically
            # changing.
            return direction * -1
        else:
            return direction


