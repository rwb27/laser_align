#!/usr/bin/env python

"""experiments.py
Contains functions to perform a set of measurements and output the results
to a datafile or graph. These functions are built on top of measurements.py,
microscope.py, image_proc.py and data_io.py and come with their own JSON
config files. NOTE: Whenever using microscope.py, ensure that its module-wide
variable 'scope_defs' is correctly defined.

Usage:
    experiments.py autofocus
    experiments.py centre
    experiments.py tiled
    experiments.py memoryleak
    experiments.py align
    experiments.py (-h | --help)

Options:
    -h, --help   Display this usage statement."""

import time
import cv2
import numpy as np
from docopt import docopt
from nplab.experiment.experiment import Experiment
from scipy import ndimage as sn

import data_io as d
import helpers as h
import image_proc as proc
import measurements as mmts
import microscope as micro

# Read the relevant config files. microscope.json is used to
# set everything about the microscope (see microscope.py), and
# autofocus.json for autofocusing. tiled_images.json is used for the tiled
# images procedure. align.json is for the laser alignment class.
focus_defs = d.config_read('./configs/autofocus.json')
tiled_defs = d.config_read('./configs/tiled_image.json')
align_defs = d.config_read('./configs/align.json')


class AutoFocus(Experiment):
    """Autofocus the image on the camera by varying z only, using the Laplacian
    method of calculating the sharpness and compressed JPEG image."""

    def __init__(self, microscope):
        super(AutoFocus, self).__init__()
        self.scope = microscope
        # Preview the microscope so we can see auto-focusing.
        self.scope.camera.preview()

    def run(self, backlash=micro.scope_defs["backlash"],
            capt_mode=micro.scope_defs["mode"]):
        # Set up the data recording.
        attributes = {'resolution':     self.scope.camera.resolution,
                      'backlash':       backlash,
                      'capture_func':   capt_mode}
        z_range = focus_defs["mmt_range"]

        funcs = [h.bake(proc.crop_array,
                        args=['IMAGE_ARR'],
                        kwargs={'mmts': 'frac',
                                'dims': focus_defs["crop_fraction"]}),
                 h.bake(mmts.sharpness_lap, args=['IMAGE_ARR'])]
        # At the end, move to the position of maximum brightness.
        end = h.bake(max_fourth_col, args=['IMAGE_ARR', self.scope])

        for n_step in z_range:
            # Allow the iteration to take place as many times as specified
            # in the config file.
            _move_capture(self, {'z': n_step}, 'bayer', func_list=funcs,
                          save_mode=None, end_func=end)
            print self.scope.stage.position


class TestMemoryLeak(Experiment):
    """Class to take lots of images, and crash the Pi."""
    def __init__(self, microscope):
        super(TestMemoryLeak, self).__init__()
        self.scope = microscope
        self.N = 1

    def run(self, N=None):
        if N is None:
            N = self.N
        prev_time = time.time()
        for i in range(N):
            print "acquiring image %d" % i
            image = self.scope.camera.get_frame(greyscale=False,
                                                mode='compressed')
            print "dt: {} seconds".format(time.time() - prev_time)
            prev_time = time.time()
        print "done"


class Tiled(Experiment):
    """Class to conduct experiments where a tiled sequence of images is 
    taken and post-processed."""

    def __init__(self, microscope):
        super(Tiled, self).__init__()
        self.scope = microscope
        self.scope.camera.preview()

    def run(self, func_list=None, save_mode='save_subset',
            step_pair=(tiled_defs["n"], tiled_defs["steps"])):
        # Set up the data recording.
        attributes = {'n': step_pair[0], 'steps': step_pair[1],
                      'backlash': micro.scope_defs["backlash"],
                      'focus': tiled_defs["focus"]}

        # Set mutable default values.
        if func_list is None:
            func_list = [h.bake(h.unchanged, args=['IMAGE_ARR'])]

        end = h.bake(max_fourth_col, args=['IMAGE_ARR', self.scope])

        # Take measurements and move to position of maximum brightness.
        _move_capture(self, {'x': [step_pair], 'y': [step_pair]},
                      image_mode='compressed', func_list=func_list,
                      save_mode=save_mode, end_func=end)
        print self.scope.stage.position


class Align(Experiment):
    """Class to align the spot to position of maximum brightness."""

    def __init__(self, microscope):
        super(Align, self).__init__()
        self.scope = microscope

    def run(self, func_list=None, save_mode='save_final'):
        """Algorithm for alignment is to iterate the Tiled procedure several
        times with decreasing width and increasing precision, and then using
        the parabola of brightness to try and find the maximum point by
        shifting slightly."""
        # Set mutable default values.
        if func_list is None:
            func_list = [h.bake(h.unchanged, args=['IMAGE_ARR'])]

        #tiled_set = Tiled(self.scope)
        #for step_pairs in align_defs["n_steps"]:
        #    # Take measurements and move to position of maximum brightness.
        #    tiled_set.run(func_list=func_list, save_mode=save_mode,
        #                  step_pair=step_pairs)

        par = ParabolicMax(self.scope)

        for i in range(3):
            for ax in ['x', 'y']:
                par.run(func_list=func_list, save_mode=save_mode, axis=ax)
        image = self.scope.camera.get_frame(greyscale=False)
        mod = proc.crop_array(image, mmts='pixel', dims=55)
        self.create_dataset('FINAL', data=mod)


class ParabolicMax(Experiment):
    """Takes a sequence of N measurements, fits a parabola to them and moves to
    the maximum brightness value."""

    def __init__(self, microscope):
        super(ParabolicMax, self).__init__()
        self.scope = microscope

    def run(self, func_list=None, save_mode='save_final', axis='x',
            step_pair=(align_defs["parabola_N"], align_defs["parabola_step"])):
        """Operates on one axis at a time."""
        # Set mutable default values.
        if func_list is None:
            func_list = [h.bake(h.unchanged, args=['IMAGE_ARR'])]

        end = h.bake(move_to_parmax, args=['IMAGE_ARR', self.scope, axis])
        _move_capture(self, {axis: [step_pair]},
                      image_mode='compressed', func_list=func_list,
                      save_mode=save_mode, end_func=end)


def centre_spot(scope_obj):
    """Find the spot on the screen, if it exists, and bring it to the
    centre. If no spot is found, an error is raised.
    :param scope_obj: A microscope object."""

    # TODO Need some way to identify if spot is not on screen.

    transform = scope_obj.calibrate()
    scope_obj.camera.preview()
    # TODO 'compressed' mode for speed, 'bayer' for accuracy.
    frame = scope_obj.camera.get_frame(mode='compressed', greyscale=True)

    # This is strongly affected by any other bright patches in the image -
    # need a better way to distinguish the bright spot.
    thresholded = cv2.threshold(frame, 180, 0, cv2.THRESH_TOZERO)[1]
    gr = scope_obj.datafile.new_group('crop')
    cropped = proc.crop_array(thresholded, mmts='pixel', dims=np.array([
        300, 300]), centre=np.array([0, 0]))
    peak = sn.measurements.center_of_mass(thresholded)
    half_dimensions = np.array(np.array(h.get_size(frame)[:2]) / 2., dtype=int)

    # Note that the stage moves in x and y, so to calculate how much to move
    # by to centre the spot, we need half_dimensions - peak.
    thing = np.dot(half_dimensions - peak[::-1], transform)
    move_by = np.concatenate((thing, np.array([0])))
    scope_obj.stage.move_rel(move_by)
    scope_obj.datafile.add_data(cropped, gr, 'cropped')
    return


def sharpness_vs_position(pixel_step, list_of_arrs):
    """Calculates the sharpness of a set of sub-images as a function of
    position.
    :param pixel_step: A tuple of no. of pixels per sub-image along (x, y).
    :param list_of_arrs: A list of lists containing the sub-image arrays,
    in the format returned by crop_img_into_n.
    :return: An array in the form [x_position, y_position, sharpness],
    where each of the three are column vectors."""

    sharpness_arr = []
    for arr_list in list_of_arrs:
        sharpness_col = []
        for arr in arr_list:
            sharpness_col.append(mmts.sharpness_lap(arr))
        sharpness_arr.append(sharpness_col)

    sharpness_arr = np.array(sharpness_arr)
    it = np.nditer(sharpness_arr, flags=['multi_index'])
    results = []
    while not it.finished:
        results.append([it.multi_index[0] * pixel_step[0] + pixel_step[0]/2,
                        it.multi_index[1] * pixel_step[1] + pixel_step[1]/2,
                        it[0]])
        it.iternext()

    return np.array(results)


def max_fourth_col(results_arr, scope_obj):
    """Given a results array made of [[x_pos], [y_pos], [z_pos], [quantity]]
    format, moves scope stage to the position with the maximum value of
    'quantity'."""
    print results_arr
    new_position = results_arr[np.argmax(results_arr[:, 3]), :][:3]
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


def grad_feedback(results_arr, scope_obj, axis):
    """Given three nearby points with only one axis's values varying,
    calculate a gradient by working out the straight line between the
    points. Move to make gradient equal 0."""


def _move_capture(exp_obj, iter_dict, image_mode, func_list=None,
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

    :param image_mode: The camera's capture mode: 'bayer', 'bgr' or
    'compressed'. Greyscale is off by default.

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

                image = exp_obj.scope.camera.get_frame(greyscale=False,
                                                       mode=image_mode)
                modified = image

                # Post-process.
                for function in func_list:
                    modified = function(modified)
                    exp_obj.create_dataset('modd', data=modified)

                # Save this array in HDF5 file.
                if save_mode == 'save_each':
                    exp_obj.create_dataset('modified_image', attrs={
                        'Position': exp_obj.scope.stage.position,
                        'Cropped size': 300}, data=modified)
                else:
                    # The curried function and 'save_final' both use the
                    # array of final results.
                    results.append([scope.stage.position[0],
                                    scope.stage.position[1],
                                    scope.stage.position[2], modified])

        except StopIteration:
            # Iterations finished - save the subset of results and take the
            # next set.
            if save_mode == 'save_subset':
                results = np.array(results, dtype=np.float)
                exp_obj.create_dataset('brightness_results', data=results)
                exp_obj.log("Test - brightness results added.")

        except KeyboardInterrupt:
            print "Aborted, moving back to initial position."
            exp_obj.scope.stage.move_to_pos(initial_position)
            exp_obj.wait_or_stop(10)

    results = np.array(results, dtype=np.float)
    if save_mode == 'save_final':
        exp_obj.create_dataset('brightness_results', data=results)
        exp_obj.log("Test - brightness results added.")
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


if __name__ == '__main__':
    sys_args = docopt(__doc__)

    # Control pre-processing manually.
    scope = micro.Microscope(filename='dat.hdf5', man=True)

    # Calculate brightness of central spot by taking a tiled section of
    # images, cropping the central 250 x 250 pixels, down sampling the bayer
    # image. Return an array of the positions and the brightness.
    fun_list = [h.bake(proc.crop_array, args=['IMAGE_ARR'],
                       kwargs={'mmts': 'pixel', 'dims': 55}),
                h.bake(mmts.brightness, args=['IMAGE_ARR'])]

    if sys_args['autofocus']:
        # auto_focus(scope)
        focus = AutoFocus(scope)
        focus.run()
    elif sys_args['centre']:
        centre_spot(scope)
    elif sys_args['tiled']:
        tiled = Tiled(scope)
        tiled.run(func_list=fun_list)
    elif sys_args['memoryleak']:
        leak = TestMemoryLeak(scope)
        leak.N = 100
        leak.run()
    elif sys_args['align']:
        align = Align(scope)
        align.run(func_list=fun_list)
