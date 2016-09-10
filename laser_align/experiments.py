#!/usr/bin/env python

"""experiments.py
Contains functions to perform a set of measurements and output the results to a
datafile or graph. These functions are built on top of _experiments.py and
data_io.py."""

import time as t
import numpy as np
import sys

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
        # end_func is applied at the end of every set of (n, step).
        for n_step in self.config_dict['raster_n_step']:
            _exp.move_capture(self, {'x': [n_step], 'y': [n_step]},
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

        hilly = HillWalk(self.scope, self.config_dict, group=self.gr)
        hilly.run()


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

    def run(self, func_list=b.baker(b.unchanged), save_mode='save_final',
            number=1000, delay=0.1, initial_align=False):
        """Default is to measure for 100s. See the config file for sleep
        times."""
        # Do an initial alignment and then take that position as the initial
        # position.

        align = Align(self.scope, self.config_dict, group=self.gr)
        hill_walk = HillWalk(self.scope, self.config_dict, group=self.gr)
        timed_mmts = TimedMeasurements(self.scope, self.config_dict,
                                       group=self.gr, N=1, t=0)
        sleep_times = self.config_dict['sleep_times']

        drifts = []
        for i in xrange(len(sleep_times)):
            if i == 0 and initial_align:
                align.run(func_list=func_list, save_mode=save_mode)
            else:
                hill_walk.run()
            pos = self.scope.stage.position
            sleep_start = t.time()
            timed_list = []
            while _exp.elapsed(sleep_start) < sleep_times[i]:
                timed_list.append(timed_mmts.run(save_mode=None))
            self.gr.create_dataset('timed_run', data=np.array(timed_list),
                                   attrs={'number': number, 'delay': delay,
                                          'sleep_time': sleep_times[i]})
            if i == 0:
                last_pos = pos
            drift = pos - last_pos
            last_pos = pos
            drifts.append([sleep_times[i], drift[0], drift[1], drift[2]])

        # Measure the position after it has drifted by working out how much
        # it needs to move by to re-centre it.
        self.gr.create_dataset('Drift', data=np.array(drifts), attrs={
            'number': number, 'delay': delay})


class KeepCentred(_exp.ScopeExp):
    """After the initial alignment, keep hillwalking with a very small step
    size."""

    def __init__(self, microscope, config_file, group=None, included_data=(
            'n_steps', 'parabola_N', 'parabola_step', 'parabola_iterations'),
                 **kwargs):
        super(KeepCentred, self).__init__(microscope, config_file, group,
                                          included_data, **kwargs)

    def run(self, func_list=b.baker(b.unchanged), save_mode='save_final',
            max_step=10, min_step=1, number=100, delay=0):
        raster = RasterXY(self.scope, self.config_dict, group=self.gr,
                          group_name='KeepCentred')
        raster.run(func_list=func_list, save_mode=save_mode)

        hilly = HillWalk(self.scope, self.config_dict, group=self.gr)
        while True:
            try:
                hilly.run(max_step=max_step, min_step=min_step, number=number,
                          delay=delay)
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
        return _exp.move_capture(self, {}, order_gen=order_gen,
                                 func_list=func_list, save_mode=save_mode,
                                 number=1, delay=0)


class HillWalk(_exp.ScopeExp):
    """Experiment to walk until a 5-point peak is found, and then repeats
    this for each axis, after which the step size is reduced."""

    DRIFT_TIME = 1000

    def __init__(self, microscope, config_file, group=None, included_data=(
            'raster_n_step', 'mmt_range'), **kwargs):
        super(HillWalk, self).__init__(microscope, config_file, group,
                                       included_data, **kwargs)

    def run(self, max_step=25, number=100, delay=0.0, min_step=1):
        """ONLY use when you have a non-zero initial reading. Process for this
        method:
        1) For the max_step size, start with the x axis and continuously
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
        initial_pos = self.scope.stage.position

        while step_size > min_step:
            for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                direction = 1
                print "axis", axis

                while True:
                    try:
                        pos = self.scope.stage.position
                        mmt = self.scope.sensor.average_n(number, delay)
                        # Check for saturation of each measurement. Remember
                        # to set ignore_saturation to False at the end of
                        # all measurements.
                        mmt = b.saturation_reached(mmt, self.scope.sensor)
                    except b.Saturation:
                        self.scope.stage.move_to_pos(initial_pos)
                        self.run(max_step, number, delay)
                        return

                    print pos, mmt
                    results.append([_exp.elapsed(self.scope.start),
                                    pos[0], pos[1], pos[2], mmt[0],
                                    mmt[1]])

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
                        direction, step_size = self._change_direction(
                            results, axis_index, direction, 3, step_size)
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
                        # For each row in the results, check if that
                        # position is present (i.e. it has already been
                        # measured less than DRIFT_TIME seconds ago). To
                        # measure the time, work out the elapsed time
                        # since the start of the microscope object,
                        # and from this subtract the elapsed time since
                        # the start of the experiment. If it
                        # does exist, this position does not need to be
                        # re-measured so keep moving in the direction until
                        # this repetition no longer occurs. If pos_to_be is
                        # changed, the loop must restart to get around this
                        # problem. So reset i=0 and don't add 1 to it. The
                        # loop finally exists when that position has not
                        # been found in the entire set in a recent time.
                        if (np.array(results[i][1:4]) ==
                            np.array(pos_to_be)).all() and (_exp.elapsed(
                                self.scope.start) - results[i][0] <
                                self.DRIFT_TIME):
                            pos_to_be += (direction * step_size * np.array(
                                axis))
                            i = 0
                        else:
                            i += 1

                    self.scope.stage.move_to_pos(pos_to_be)
                    print "new position", pos_to_be
            step_size /= 2
            print "step_size", step_size

        self.gr.create_dataset('hill_walk_brightness', data=results, attrs={
            'mmts_per_reading': number, 'delay_between': delay,
            'max_step': max_step})
        self.scope.sensor.ignore_saturation = False

    @staticmethod
    def _check_sliced_results(results, slice_size, axis_index=None,
                              check_unique=True):
        """Ensure the other position axes haven't changed, and the axis in
        question has all different positions. Return the appropriately
        sliced results.
        :param results: The results array/list.
        :param axis_index: The index of the position that is varying.
        :param slice_size: The number of the last rows of the results array to
        use."""
        last_rows = np.array(results[-slice_size:])
        assert last_rows.shape[0] == slice_size

        if check_unique:
            other_indices = []
            for ind in [1, 2, 3]:
                if ind != axis_index:
                    other_indices.append(ind)
            assert [np.unique(last_rows[:, other_index]).size == 1 for
                    other_index in other_indices]

            assert np.unique(last_rows[:, axis_index]).size == slice_size

        return last_rows

    def _try_fit_parabola(self, results, axis_index):
        last_five_rows = self._check_sliced_results(results, 5, axis_index)
        sort = last_five_rows[np.argsort(last_five_rows[:, axis_index])]

        if np.all(np.sign(np.ediff1d(sort[:, 4])) == np.array(
                [1, 1, -1, -1])):
            axis_string = ['x', 'y', 'z'][axis_index - 1]
            b.move_to_parmax(sort, self.scope, axis_string)
            return True
        else:
            return False

    def _change_direction(self, results, axis_index, direction,
                          num_per_parabola, step_size):
        """Changes the direction of measurement if values are descending
        monotonically for the entire set of num_per_parabola measurements,
        such that decrease exceeds the noise error.
        :param results: The results array.
        :param axis_index: The index of the axis that is changing.
        :param direction: The current direction of motion.
        :return: The new direction of motion."""
        try:
            # Get between num_per_parabola and 2 * num_per_parabola -1 rows
            # (allows comparison of all combinations of consecutive
            # num_per_parabola results).
            # This section of last_rows needs checking for:
            # - noisy consecutive signals (consecutive meaning within n
            # sigma).
            # - descending
            last_rows = self._check_sliced_results(
                results, 3, axis_index)
            sort = last_rows[np.argsort(last_rows[:, axis_index])]
            # TODO WE NEED TO IMPLEMENT SORTING HERE. Check for monotonic
            # decreases.
            # We want to find if the last 3 position/brightness
            # measurements have a positive/negative gradient. If
            # negative, reverse direction.
            x = last_rows[:, axis_index]
            y = last_rows[:, 4]

            if np.all(np.sign(np.ediff1d(y)) <= 0) and np.ediff1d(np.sign(
                    np.ediff1d(x))) == 0:
                # Change directions only if the positions are monotonically
                # changing.
                direction *= -1

        except (IndexError, AssertionError):
            # The array is not large enough for this to work.
            pass

        return direction, step_size

    def _ignore_noisy_signal(self, results, n_rows=10):
        """If the last 10 readings along the same axis are within 2 sigma of
        their previous reading, the signal is noise-affected and hill-walk
        should not be used - instead a raster scan should be done."""
        last_ten_rows = self._check_sliced_results(results, n_rows,
                                                   check_unique=False)
        noisy_size = 0
        for i in xrange(last_ten_rows[:, 0].size):
            if ((last_ten_rows[i-1: 4] - 2 * last_ten_rows[i-1: 5]) <
                    last_ten_rows[i, 4] < (last_ten_rows[i-1: 4] + 2 *
                    last_ten_rows[i-1: 5])) and i != 0:
                noisy_size += 1

        if noisy_size == n_rows:
            raise b.NoisySignal(
                'The signal is noisy, so any measures of peaks are likely to '
                'be inaccurate.')


class HillWalk2(_exp.ScopeExp):
    """More sophisticated hill walk algorithm that measures in only one
    direction, accounts for noise, saturation, descending measurements,
    and adapts step size, averaging number and time to minimise noise and
    runtime, and maximise precision and accuracy."""

    DRIFT_TIME = 1000

    def __init__(self, microscope, config_file, group=None, included_data=(),
                 **kwargs):
        super(HillWalk2, self).__init__(microscope, config_file, group,
                                        included_data, **kwargs)

    def run(self, max_step=100, init_number=100, init_delay=0.0, min_step=5,
            num_per_parabola=7, sigma_multiples=2, sig_level=0.05,
            save_mode='save_final'):

        results = []
        step_size = max_step
        number = init_number
        delay = init_delay
        initial_pos = self.scope.stage.position
        self.scope.sensor.ignore_saturation = False

        while step_size > min_step:
            for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                axis_index = axis.index(1) + 1
                direction = 1

                while True:
                    try:
                        # Note down the position before the loop os measurements
                        # starts, to return there if something goes wrong.
                        current_pos = np.array([0, 10, 20])
                        move_positions = np.arange(
                            -step_size * num_per_parabola / 2.,
                            step_size * (num_per_parabola + 1) / 2., step_size)
                        positions = np.outer(move_positions, np.array(axis)) +\
                            current_pos
                        print "positions", positions
                        gen = b.do_not_revisit(results, positions)

                        func_list = b.baker(b.saturation_reached,
                                            args=['mmt-placeholder',
                                                  self.scope.sensor])

                        while True:
                            results = _exp.read_move_save(self, gen, func_list,
                                                          save_mode, number,
                                                          delay, results)
                            print results
                    except b.Saturation:
                        # If the measurement saturates, then take that set of
                        # measurements again, with the new values of parameters
                        # that change (step_size, number, delay). Don't ignore
                        # saturation exceptions here, as this is fine alignment!
                        # Note the ignore_saturations parameter is reset after
                        # every set of
                        # num_per_parabola measurements, as we advise against
                        # choosing the 'ignore all' parameter!
                        self.scope.stage.move_to_pos(current_pos, backlash=5000)
                        continue

                    except KeyboardInterrupt:
                        if save_mode == 'save_subset' or save_mode == 'save_final':
                            _exp.save_results(self, results, number, delay,
                                              why_ended='keyboard_interrupt')

                        # Move to original position and exit program.
                        print "Aborted, moving back to initial position. " \
                              "Exiting program."
                        self.scope.stage.move_to_pos(initial_pos,
                                                     backlash=5000)
                        sys.exit()

                    except StopIteration:
                        # Iterations finished - save the subset of results.
                        if save_mode == 'save_subset':
                            _exp.save_results(self, results, number, delay,
                                              why_ended=str(StopIteration))

                        # Test for whether signal is noisy.

                    self.scope.sensor.ignore_saturation = False
                    
    def _check_noisy(self, consec_results, num_per_parabola,
                     sigma_level=2, attempt=1):
        """The array 'consec_results' is the array of consecutive, same gain and 
        most recent, sorted etc. results. If, for this array, the >
        1/2*num_per_parabola of the brightness readings are within sigma_level 
        * error of their neighbours, then the signal is noisy. Repeat the 
        readings with a higher number and delay, and if results are still noisy 
        then raise a NoisySignal exception."""

        # The maximum number of allowed noisy signals in the array of
        # consecutive results.
        max_noisy_size = np.ceil(num_per_parabola/2.)
        noisy = 0

        # To check if noisy, compare each brightness to the previous one to
        # see if it is in the range.
        for i in xrange(consec_results[:, 0].size):
            min_value = consec_results[i-1: 4] - sigma_level * consec_results[i-1: 5]
            max_value = consec_results[i-1: 4] + sigma_level * consec_results[i-1: 5]
            if (min_value < consec_results[i, 4] < max_value) and i != 0:
                # The i != 0 condition prevents the first row being compared
                # to the last.
                noisy += 1

        if noisy >= max_noisy_size:
            # The signal has been found to be noisy this time around. If
            # this is the first attempt, re-measure with a 5 x higher number
            # per average (expected to reduce error in mean by sqrt(5)) and
            # increase the delay by 0.1s. We also need to ensure the total
            # measurement time is much less than the drift time. So ensure
            # that number * delay < 50s, the TODO approximate scale to
            # minimise Allan deviation. If this is not the case,
            # raise NoisySignal. Also raise NoisySignal if this is the
            # second attempt.
            number *= 5
            delay += 0.1
            if number * delay > 50 or attempt == 2:
                raise b.NoisySignal

        # Return the new positions to visit - the same ones that were
        # just measured if the signal was noisy. If the signal was not noisy,
        # return the next set to go to based on direction parameter.




    def get_proc_consec(self, results, axis_index, max_step):
        """From an array of results, select the appropriate ones to check for a
        parabola later.
        :param results: The results array, in the forat with column titles:
        time elapsed, x, y, z, brightness, error, number per average, delay per
        mmt, gain.
        :param axis_index: The index of the position axis in the array that was
        just measured - only values varying along this axis are considered.
        :param max_step: The maximum step size for measurements to be
        considered consecutive.
        :return: The measurements that the consecutive to the ones just taken.
        """

        # This function must be applied just after a series of measurements,
        # which are sorted by time.
        last_row = results[-1, :]
        example_pos = last_row[1:4]
        gain = last_row[-1]

        # To get the results ready for processing, the algorithm is:
        # 1) Only keep rows with the same values for the unchanged position
        #    axes as the last mmt set.
        # 2) Only keep rows with the same gain as the last mmt set.
        # 3) Sort the array by time and positions.
        # 4) Delete values older than DRIFT_TIME.
        # 5) Get the most recent versions of values if multiple have been taken
        #    at the same point.
        # 6) Find all values consecutive to the ones taken in the last set,
        #    along the varying axis.

        func_list = [b.baker(self.filter_unchanged_axes,
                             args=['results', axis_index, example_pos]),
                     b.baker(self.filter_gain, args=['results', gain]),
                     b.baker(self.sort_array, args=['results', axis_index]),
                     b.baker(self.delete_old_values, args=['results']),
                     b.baker(self.get_latest, args=['results', axis_index]),
                     b.baker(self.find_consecutive, args=[
                         'results', axis_index, max_step])]

        results = _exp.apply_functions(results, func_list)
        return results

    def delete_old_values(self, results):
        """Delete values older than those taken > DRIFT_TIME ago."""
        now_elapsed = _exp.elapsed(self.scope.start)
        deleted = np.delete(results, np.where(now_elapsed - results[:, 0] >
                                              self.DRIFT_TIME), axis=0)
        if deleted.size > 0:
            return deleted
        else:
            raise ValueError

    @staticmethod
    def filter_unchanged_axes(results, axis_index, example_position):
        """Return the part of the array where the values for the unchanged
        position axes are those in example_position (an example position array
        where only axis_index - 1 is allowed to vary)."""

        for i in xrange(1, 4):
            if i != axis_index:
                results = results[np.where(results[:, i] == example_position[
                    i - 1]), :][0]

        return results

    @staticmethod
    def filter_gain(results, gain):
        """Return the results array only containing the rows with same gain
        specified."""
        return results[np.where(results[:, -1] == gain)]

    @staticmethod
    def sort_array(results, axis_index):
        """Sort an array by the first 4 indices (time, 3-displacement).
        :param results: Results array filtered by one specific gain value.
        :param axis_index: The index of the position axis that varies."""

        # Get the order in which to sort the array. Sort first by the axes
        # that do not change. This should result in [0, axis_index,
        # other_pos_axes] at the end. So, sorting occurs by the unchanged
        # position axes first, then by the changing position, then by time.
        sort_by = []
        for axis in xrange(4):
            if axis != axis_index:
                sort_by.append(axis)
        sort_by.insert(1, axis_index)

        # This method sorts by the last element in the tuple first.
        row_order = np.lexsort((results[:, sort_by[0]],
                                results[:, sort_by[1]],
                                results[:, sort_by[2]],
                                results[:, sort_by[3]]))
        return results[row_order, :]

    @staticmethod
    def get_latest(results, axis_index):
        """Find the most recent value of each measurement in results, assuming
        it has already been sorted, filtered for gain, old values deleted,
        and filtered by position axes that do not change."""
        recent_array = []
        for position in np.nditer(np.unique(results[:, axis_index])):
            temp = results[np.where(results[:, axis_index] == position)]
            temp = temp[np.where(temp[:, 0] == np.max(temp[:, 0])), :][0][0]
            recent_array.append(temp)
        return np.array(recent_array)

    @staticmethod
    def find_consecutive(results, axis_index, max_step):
        """Find n consecutive positions on on either side of the positions
        array, in the sorted results array. Consecutive means they differ
        from their neighbours by <= max_step."""

        # Readings are taken as not consecutive when they differ from their
        # neighbours by > max_step. The code below gives the indices to split
        # the results array along, into different sets of measurements.
        split_along = \
        np.where(np.absolute(np.ediff1d(results[:, axis_index])) >
                 max_step)[0] + 1
        return np.split(results, split_along)

    @staticmethod
    def _check_sliced_results(results, slice_size, axis_index=None,
                              check_unique=True):
        """Ensure the other position axes haven't changed, and the axis in
        question has all different positions. Return the appropriately
        sliced results.
        :param results: The results array/list.
        :param axis_index: The index of the position that is varying.
        :param slice_size: The number of the last rows of the results array to
        use."""
        last_rows = np.array(results[-slice_size:])
        assert last_rows.shape[0] == slice_size

        if check_unique:
            other_indices = []
            for ind in [1, 2, 3]:
                if ind != axis_index:
                    other_indices.append(ind)
            assert [np.unique(last_rows[:, other_index]).size == 1 for
                    other_index in other_indices]

            assert np.unique(last_rows[:, axis_index]).size == slice_size

        return last_rows

