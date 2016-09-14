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
import matplotlib.pyplot as plt


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
        last_eleven_rows = self._check_sliced_results(results, 11, axis_index)
        sort = last_eleven_rows[np.argsort(last_eleven_rows[:, axis_index])]

        if np.all(np.sign(np.ediff1d(sort[:, 4])) == np.array(
                [1, 1, 1, 1, 1, -1, -1, -1, -1, -1])):
            axis_string = ['x', 'y', 'z'][axis_index - 1]
            b.to_parmax(sort, self.scope, axis_string)
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


class AdaptiveHillWalk(_exp.ScopeExp):
    """More sophisticated hill walk algorithm that measures in only one
    direction, accounts for noise, saturation, descending measurements,
    and adapts step size, averaging number and time to minimise noise and
    runtime, and maximise precision and accuracy."""

    DRIFT_TIME = 1000

    def __init__(self, microscope, config_file, group=None, included_data=(
            'max_step', 'init_number', 'init_delay', 'min_step',
            'num_per_parabola', 'sigma_level', 'sig_level'), **kwargs):
        super(AdaptiveHillWalk, self).__init__(microscope, config_file, group,
                                               included_data, **kwargs)
        self.step_size = None
        self.number = None
        self.delay = None
        self.num_per_parabola = None
        self.sigma_level = None
        self.reset()

        self.fig, self.axes = plt.subplots(3, 1, sharey=True)
        plt.ion()

    def reset(self):
        self.step_size = np.ones(3) * self.config_dict['max_step']
        self.number = self.config_dict['init_number']
        self.delay = self.config_dict['init_delay']
        self.num_per_parabola = self.config_dict['num_per_parabola']
        self.sigma_level = self.config_dict['sigma_level']

    def run(self, save_mode='save_final'):

        initial_pos = self.scope.stage.position
        self.scope.sensor.ignore_saturation = False

        while np.any(self.step_size > self.config_dict['min_step']):
            for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                axis_index = axis.index(1) + 1
                print 'axis', axis_index
                direction = 1
                while True:
                    try:
                        results = []
                        start_pos = self.scope.stage.position
                        positions, noisy_attempt = self.next_positions(
                            results, direction, axis)
                        gen = b.yield_pos(positions)

                        func_list = b.baker(b.saturation_reached,
                                            args=['mmt-placeholder',
                                                  self.scope.sensor])

                        while True:
                            results = _exp.read_move_save(
                                self, gen, func_list, save_mode, self.number,
                                self.delay, results)

                    except b.Saturation:
                        # If the measurement saturates, then take that set of
                        # measurements again, with the new values of parameters
                        # that change (step_size, number, delay). Don't ignore
                        # saturation exceptions here, as this is fine alignment!
                        # Note the ignore_saturations parameter is reset after
                        # every set of
                        # num_per_parabola measurements, as we advise against
                        # choosing the 'ignore all' parameter!
                        results = np.array(results)
                        _exp.save_results(self, results, self.number,
                                          self.delay, why_ended='Saturation')
                        self.scope.stage.move_to_pos(start_pos)
                        continue

                    except KeyboardInterrupt:
                        if save_mode == 'save_subset' or save_mode == \
                                'save_final':
                            _exp.save_results(
                                self, results, self.number, self.delay,
                                why_ended=str(KeyboardInterrupt))

                        # Move to original position and exit program.
                        print "Aborted, moving back to initial position. " \
                              "Exiting program."
                        self.scope.stage.move_to_pos(initial_pos)
                        sys.exit()

                    except StopIteration:
                        # Iterations finished - save the subset of results.
                        results = np.array(results)
                        if save_mode == 'save_subset':
                            _exp.save_results(
                                self, results, self.number, self.delay,
                                why_ended=str(StopIteration))

                            # Test for whether signals are all zero for the last
                            # set.
                            #try:
                            #    self.all_zeros(results, axis_index)
                            #except b.ZeroSignal:
                            #    # If all readings are zero, turn up the
                            #    # gain and retry a raster scan over a small area.
                            #    while True:
                            #        if self.scope.sensor.gain != 70:
                            #            answer = raw_input('Turn up the gain
                            # once '
                            #                               'and enter \'done\' '
                            #                               'when finished: ')
                            #            if answer == 'done':
                            #                self.scope.sensor.gain += 10
                            #                break
                        #
                        #    raster = RasterXY(self.scope, self.config_dict,
                        #                      raster_n_step=[[5, 100]])
                        #    raster.run()
                        #
                        #    # Then run this hillwalk again.
                        #    self.run()
                        #    return
                        #
                        # Test for whether signal is noisy.
                        #positions, noisy_attempt = self.act_if_noisy(
                        #    results, positions, direction, axis,
                        # noisy_attempt)

                        # Check if results ascending/descending and change
                        # direction if needed.
                        #direction = self.change_direction(results, axis_index,
                        #                                  direction)

                        # Check for parabolae and move to the appropriate
                        # maximum. Adjust step size, number and delay.

                        # Get consecutive results for later use.
                        #consec = self.get_consec(results, axis_index,
                        #                         self.config_dict['max_step'])

                        _exp.save_results(self, results, self.number,
                                          self.delay, why_ended='set')
                        #self.axes[axis_index - 1].plot(
                        #    results[:, axis_index], results[:, 4]/10 ** (
                        #        results[:, 8]/10) * 30000)
                        #plt.show()
                        #print "showing"
                        if self.fit_to_parabola(results, axis_index):
                            print "breaking"
                            break

                    self.scope.sensor.ignore_saturation = False

    def act_if_noisy(self, results, positions, direction, axis, attempt):
        """Tests for if the signal is noisy. If it is, on the first attempt
        the same positions are re-measured with 5 x the number per average
        and 0.1s extra delay between measurements, as long as this doesn't
        exceed the drift time factor. If it does, or this is the second
        attempt, raise an exception. If the signal is not noisy, return the
        next positions to visit."""
        if self._check_noisy(results, axis):
            # The signal has been found to be noisy this time around. If
            # this is the first attempt, re-measure with a 5 x higher number
            # per average (expected to reduce error in mean by sqrt(5)) and
            # increase the delay by 0.1s. We also need to ensure the total
            # measurement time is much less than the drift time. So ensure
            # that number * delay < 50s, the TODO approximate scale to
            # minimise Allan deviation. If this is not the case,
            # raise NoisySignal. Also raise NoisySignal if this is the
            # second attempt.
            if 5 * self.number * (self.delay + 0.1) > self.DRIFT_TIME/20. or \
                            attempt > 1:
                raise b.NoisySignal
            else:
                self.number *= 5
                self.delay += 0.1
                attempt += 1
            # Return the new positions to visit - the same ones that were
            # just measured if the signal was noisy.
            return positions, attempt
        else:
            # If the signal was not noisy, return the new positions to go to.
            return self.next_positions(results, direction, axis)

    def _check_noisy(self, results, axis):
        """The array 'results' is the array of unsorted results. If, for the
        last measured set, more than
        1/2*num_per_parabola of the brightness readings are within sigma_level
        * error of their neighbours, then the signal is noisy. Repeat the
        readings with a higher number and delay, and if results are still noisy
        then raise a NoisySignal exception."""
        try:
            last_rows = self._slice_results(
                results, self.num_per_parabola, axis.index(1) + 1)
        except (IndexError, AssertionError):
            pass

        # The maximum number of allowed noisy signals in the array of
        # consecutive results.
        max_noisy_size = np.ceil(self.num_per_parabola/2.)
        noisy = 0

        # To check if noisy, compare each brightness to the previous one to
        # see if it is in the range.
        for i in xrange(last_rows[:, 0].size):
            min_value = last_rows[i-1, 4] - self.sigma_level * \
                                                 last_rows[i-1, 5]
            max_value = last_rows[i-1, 4] + self.sigma_level * \
                                                 last_rows[i-1, 5]
            if (min_value <= last_rows[i, 4] <= max_value) and i != 0:
                # The i != 0 condition prevents the first row being compared
                # to the last.
                noisy += 1

        if noisy >= max_noisy_size:
            return True
        else:
            return False

    @staticmethod
    def _quad_bandwidth(coeffs, fraction):
        """For a quadratic function with coefficients given, calculate the
        width in x over which the function is a proportion of the maximum of
        this quadratic. NOTE Ensure the quadratic has a maximum,
        not a minimum!
        :param coeffs: The array of coefficients [a, b, c] for a*x**2 + b*x
        + c.
        :param fraction: The fraction for which to measure the width of x.
        For a value of 0.2, for example, the range over which y=0.8*y_max is
        found."""
        roots = np.roots(np.array([coeffs[0], coeffs[1], coeffs[2] - (
            1 - fraction)*(coeffs[2] - coeffs[1]**2/(4*coeffs[0]))]))
        x_range = np.max(roots) - np.min(roots)
        return x_range

    def all_zeros(self, results, axis_index):
        try:
            last_rows = self._slice_results(
                results, self.num_per_parabola, axis_index)
            if np.all(last_rows[:, 4] == 0):
                raise b.ZeroSignal
        except (IndexError, AssertionError):
            pass

    def change_direction(self, results, axis_index, direction):
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
            last_rows = self._slice_results(
                results, self.num_per_parabola, axis_index)
            sort = last_rows[np.argsort(last_rows[:, axis_index])]

            # We want to find if the last num_per_parabola position/brightness
            # measurements have a positive/negative gradient. If
            # negative, reverse direction. The change between consecutive
            # measurements must be negative or zero.
            y = sort[:, 4]

            if np.all(np.sign(np.ediff1d(y)) <= 0):
                # Change directions only if the positions are monotonically
                # changing - they have been pre-sorted beforehand.
                direction *= -1

        except (IndexError, AssertionError):
            # The array is not large enough for this to work.
            pass

        return direction

    def fit_to_parabola(self, results, axis_index):
        last_rows = self._slice_results(results, self.num_per_parabola,
                                        axis_index)

        # Fit last rows to parabola.
        params = b.to_parmax(last_rows, self.scope, ['x', 'y', 'z'][
            axis_index - 1], move=False)

        if params is False:
            # A parabola with a minimum point was found.
            within_range = False
            new_pos = last_rows[np.where(last_rows[:, 4] == np.max(
                last_rows[:, 4])), 1:4][0][0]
            print "Parabola fitting found a minimum, re-measuring"

        elif params[2][0] <= params[0][axis_index - 1] <= params[2][1]:
            # The calculated maximum lies within the range measured. Now
            # check how high this maximum is with the measured maximum.
            within_range = True
            # TODO change to have some factor of the residuals here
            if params[1]*1.5 >= np.max(last_rows[:, 4]):
                # Predicted maximum is higher, move to the predicted
                # maximum.
                new_pos = params[0]
                print "Moving to maximum of parabola: ", new_pos
                self.step_size[axis_index - 1] /= 2
            else:
                print "Predicted max. was ", params[1]
                new_pos = last_rows[np.where(last_rows[:, 4] == np.max(
                    last_rows[:, 4])), 1:4][0][0]
                print "Moving to maximum point as it's brighter than the " \
                      "parabola: ", np.max(last_rows[:, 4])
        else:
            # Measured parabola is an extrapolation.
            within_range = False
            new_pos = last_rows[np.where(last_rows[:, 4] == np.max(
                last_rows[:, 4])), 1:4][0][0]
            print "Predicted parabola maximum was too far away, moving to " \
                  "the highest-intensity point in the scan."

        self.scope.stage.move_to_pos(new_pos)
        return within_range

    def move_to_parabola(self, consec_results, axis_index):
        """Given the entire array of consecutive results, split them up,
        find parabolae and move to the appropriate maximum."""
        consec_list = self._get_consec_sets(consec_results)
        parabolae_param = self._find_parabolae(consec_list, axis_index)
        print "size", parabolae_param.size
        max_row = consec_results[np.where(consec_results[:, 4] ==
                                          np.max(consec_results[:, 4]))]
        print "max_row", max_row
        if parabolae_param.size > 0 and \
            np.any(parabolae_param[:, 2][0] < max_row[axis_index] <
                   parabolae_param[:, 2][1]):
            # If maximum is within the range of other parabolae, move to
            # the maximum of the parabola with the highest predicted
            # brightness.
            relevant_params = parabolae_param[np.where(
                parabolae_param[:, 1] == np.max(parabolae_param[:, 1]))]
            new_pos = relevant_params[1:4]

            # Modify number and delay based on residuals, and step_size
            # based on width of parabola.
            self.step_size = self._quad_bandwidth(relevant_params[4], 0.2)
        else:
            # The maximum exceeds the predicted parabola maximum,
            # or no parabolae were detected - move to the actual maximum.
            new_pos = max_row[1:4]
            self.step_size /= 2
        print new_pos

        # TODO MAKE THE RESPONSES DIFFERENT FOR MAX AND PARABOLA.
        self.delay += 0.1
        self.number += 50

        self.scope.stage.move_to_pos(new_pos)

    def _fit_parabola(self, consec_set, axis_index):
        """For a set of sorted, processed results, fit to a parabola and
        assess the degree of fit. If multiple parabolae are found in this
        set of results, move to the one with the highest maximum and lowest
        error."""

        #diffs_size = (self.num_per_parabola - 1)/2.
        #signs = np.hstack((np.ones(diffs_size), -np.ones(diffs_size)))

        #if np.all(np.sign(np.ediff1d(consec_set[:, 4])) == signs):
        axis_string = ['x', 'y', 'z'][axis_index - 1]
        return b.to_parmax(consec_set, self.scope, axis_string, move=True)
        #else:
        #    return False

    def _find_parabolae(self, consec_list, axis_index):
        """Pass in a set of consecutive, processed, sorted results in the
        form of a list of arrays of consecutive results using
        get_consec_sets."""
        parabolae_params = []
        for arr in consec_list:
            print consec_list
            fit = self._fit_parabola(arr, axis_index)
            print fit
            if fit is not False:
                new_pos, pred_y, x_range, residuals, coeffs = fit
                parabolae_params.append([new_pos, pred_y, x_range, residuals,
                                         coeffs])

        # When all sets have been checked, we have several outcomes. Either
        # there are no parabolae, in which case move to the maximum reading
        # and adjust step_size, number and delay, or there is one parabola
        # containing the measured maximum, or there is one parabola
        # not containing the measured maximum, or there are multiple
        # parabolae. In every case, move to the highest maximum possible
        # unless it lies outside a parabola, in which case re-measure with
        # finer parameters.
        print parabolae_params
        return np.array(parabolae_params)

    def _get_consec_sets(self, proc_res):
        """Get all sets of consecutive results of length num_per_parabola
        from the array of sorted, processed results. Return these as a list of
        arrays."""
        # To calculate the number of consecutive sets in results, note that
        # for an array with N rows, row indices go from 0 to N-1. The number
        # of times we can pick num_per_parabola consecutive rows is such
        # that the top index is between 0 and N-1-num_per_parabola. This is
        # the same as the size of the result rows for indexing,
        # which requires one extra index at the end:
        iterations = proc_res[:, 0].size - self.num_per_parabola
        consec_list = []
        for i in xrange(iterations):
            # So for num_per_parabola=7 we get rows 0:6, then 1:7, etc.
            consec_list.append(proc_res[i:i + self.num_per_parabola, :])
        return consec_list

    def next_positions(self, results, direction, axis, current_pos=None):
        attempt = 1
        new_pos = ()
        if current_pos is None:
            current_pos = self.scope.stage.position
        move_positions = np.sort(np.arange(-self.step_size[list(axis).index(
            1)] * (self.num_per_parabola)/2., self.step_size[list(axis).index(
            1)] * (self.num_per_parabola + 1)/2., self.step_size[list(
            axis).index(1)]) * direction)
        while len(new_pos) == 0:
            # If we want to move in the positive direction, the current
            # position is the max of those in the array of positions to go
            # to if there is overlap. For the negative direction, it is the
            # minimum. To get around this, multiply negative directions by
            # 'direction' (=-1)
            positions = np.outer(move_positions, np.array(axis)) + current_pos
            new_pos = b.revisit_check(results, positions, True)
            if direction == 1:
                current_pos = np.max(positions * axis)
            elif direction == -1:
                current_pos = np.min(positions * axis)
        print positions
        return positions, attempt

    def get_consec(self, results, axis_index, max_step):
        """From an array of results, select the appropriate ones to check for a
        parabola later. Process and find the consecutive ones.
        :param results: The results array, in the format with column titles:
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

        func_list = [b.baker(self._filter_unchanged_axes,
                             args=['results', axis_index, example_pos]),
                     b.baker(self._filter_gain, args=['results', gain]),
                     b.baker(self._sort_array, args=['results', axis_index]),
                     b.baker(self._del_old_values, args=['results']),
                     b.baker(self._get_latest, args=['results', axis_index]),
                     b.baker(self._get_consec, args=[
                         'results', axis_index, max_step])]

        results = _exp.apply_functions(results, func_list)
        return np.array(results)

    @staticmethod
    def _filter_unchanged_axes(results, axis_index, example_position):
        """Return the part of the array where the values for the unchanged
        position axes are those in example_position (an example position array
        where only axis_index - 1 is allowed to vary)."""

        for i in xrange(1, 4):
            if i != axis_index:
                results = results[np.where(results[:, i] == example_position[
                    i - 1]), :][0]

        return results

    @staticmethod
    def _filter_gain(results, gain):
        """Return the results array only containing the rows with same gain
        specified."""
        return results[np.where(results[:, -1] == gain)]

    @staticmethod
    def _sort_array(results, axis_index):
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

    def _del_old_values(self, results):
        """Delete values older than those taken > DRIFT_TIME ago."""
        now_elapsed = _exp.elapsed(self.scope.start)
        deleted = np.delete(results, np.where(now_elapsed - results[:, 0] >
                                              self.DRIFT_TIME), axis=0)
        if deleted.size > 0:
            return deleted
        else:
            raise ValueError

    @staticmethod
    def _get_latest(results, axis_index):
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
    def _get_consec(results, axis_index, max_step):
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
    def _slice_results(results, slice_size, axis_index,
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
