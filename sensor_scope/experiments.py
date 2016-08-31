#!/usr/bin/env python

"""experiments.py
Contains functions to perform a set of measurements and output the results to a
datafile or graph. These functions are built on top of _experiments.py and
data_io.py."""

import time as t
import numpy as np
from nplab.experiment.experiment import Experiment

import _experiments as _exp
import data_io as d
import baking as b


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


class AlongZ(ScopeExp):
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


class RasterXY(ScopeExp):
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


class RasterXYZ(ScopeExp):
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


class Align(ScopeExp):
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


class ParabolicMax(ScopeExp):
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


class DriftReCentre(ScopeExp):
    """Experiment to allow time for the spot to drift from its initial
    position for some time, then bring it back to the centre and measure
    the drift."""

    def __init__(self, microscope, config_file, group=None, included_data=(
            'sleep_times', 'n_steps', 'parabola_N', 'parabola_step',
            'parabola_iterations'), **kwargs):
        super(DriftReCentre, self).__init__(microscope, config_file, group,
                                            included_data, **kwargs)

    def run(self, func_list=b.baker(b.unchanged), save_mode='save_final'):
        """Default is to sleep for 10 minutes."""
        # Do an initial alignment and then take that position as the initial
        # position.

        align = Align(self.scope, self.config_dict, group=self.gr,
                      group_name='DriftReCentre')
        sleep_times = self.config_dict['sleep_times']

        drifts = []
        for i in xrange(len(sleep_times)):
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

    def __init__(self, microscope, config_file, group=None, included_data=(
            'n_steps', 'parabola_N', 'parabola_step', 'parabola_iterations'),
                 **kwargs):
        super(KeepCentred, self).__init__(microscope, config_file, group,
                                          included_data, **kwargs)

    def run(self, func_list=b.baker(b.unchanged), save_mode='save_final'):
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


class TimedMeasurements(ScopeExp):
    """Experiment to repeatedly measure the average of 10 measurements at
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
                          func_list=func_list, save_mode=save_mode)


class BeamWalk(ScopeExp):
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
                                     group=self.gr, raster_n_step=[[64, 1000]])
            else:
                raster_2d = RasterXY(self.scope, self.config_dict,
                                     group=self.gr)
            raster_2d.run(func_list=func_list, save_mode=save_mode)

            along_z = AlongZ(self.scope, self.config_dict, group=self.gr,
                             mmt_range=self.config_dict["mmt_range"])
            along_z.run(save_mode=save_mode)

# TODO add hill_climbing and simplex algorithm classes.
