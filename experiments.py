#!/usr/bin/env python

"""experiments.py
Contains functions to perform a set of measurements and output the results to a
datafile or graph. These functions are built on top of measurements.py,
microscope.py, image_proc.py and data_io.py."""

import time as t

import numpy as np
from nplab.experiment.experiment import Experiment

import data_io as d
import _experiments as _exp


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
        end = _exp.bake(_exp.max_fifth_col, args=['IMAGE_ARR', self.scope])

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
                 group_name='RasterXY', **kwargs):
        super(RasterXY, self).__init__(microscope, config_file, **kwargs)
        self.attrs = d.sub_dict(self.config_dict, ['raster_n_step'],
                                   {'scope_object': str(self.scope.info.name)})
        self.gr = d.make_group(self, group=group, group_name=group_name)

    def run(self, func_list=None, save_mode='save_final'):

        end = _exp.bake(_exp.max_fifth_col, args=['IMAGE_ARR', self.scope])

        # Take measurements and move to position of maximum brightness.
        _exp.move_capture(self, {'x': self.config_dict['raster_n_step'],
                                 'y': self.config_dict['raster_n_step']},
                          func_list=func_list, save_mode=save_mode,
                          end_func=end)
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
        end = _exp.bake(_exp.move_to_parmax, args=['IMAGE_ARR', self.scope,
                                                   axis])
        _exp.move_capture(self, {axis: [step_pair]}, func_list=func_list,
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



