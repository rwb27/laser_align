#!/usr/bin/env python

"""experiments.py
Contains functions to perform a set of measurements and output the results
to a datafile or graph. These functions are built on top of measurements.py,
microscope.py, image_proc.py and data_io.py and come with their own JSON
config files. NOTE: Whenever using microscope.py, ensure that its module-wide
variable 'defaults' is correctly defined.

Usage:
    experiments.py autofocus
    experiments.py centre
    experiments.py tiled
"""

import time
import cv2
import numpy as np
from docopt import docopt
from scipy import ndimage
from nplab.experiment.experiment import Experiment

import data_io as d
import helpers as h
import image_proc as pro
import measurements as m
import microscope as micro
from measurements import sharpness_lap

# Read the relevant config files. microscope_defaults.json is used to
# set everything about the microscope (see microscope.py), and
# autofocus.json for autofocusing. tiled_images.json is used for the tiled
# images procedure.
autofocus_defaults = d.config_read('./configs/autofocus.json')
tiled_dict = d.config_read('./configs/tiled_image.json')


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
            sharpness_col.append(sharpness_lap(arr))
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


def auto_focus(microscope):
    """Autofocus the image on the camera by varying z only, using a given
    method of calculating the sharpness.
    :param microscope: The microscope object, containing the camera and stage.
    """

    # Preview the microscope so we can see auto-focusing.
    microscope.camera.preview()

    # Set up the data recording.
    attributes = {'resolution': microscope.camera.resolution,
                  'backlash':   micro.defaults["backlash"]}
    if autofocus_defaults["sharpness_func"] == 'laplacian':
        sharpness_func = m.sharpness_lap
        attributes["sharpness_func"] = 'laplacian'
    # Add more conditions here if other sharpness functions are added.
    else:
        raise ValueError('Invalid sharpness function entered.')

    attributes['capture_func'] = micro.defaults["mode"]
    focus_data = d.Datafile(filename=autofocus_defaults["filename"])
    tests = focus_data.new_group(autofocus_defaults["group_name"],
                                 attrs=attributes)

    # Take measurements and focus.
    for step, n in autofocus_defaults["mmt_range"]:
        sharpness_list = []
        positions = []

        for i in range(n + 1):
            if i == 0:
                # Initial capture
                microscope.stage.focus_rel(-step * n / 2)
            else:
                # Remaining n measurements.
                microscope.stage.focus_rel(step)

            raw_array = microscope.camera.get_frame()
            (cropped, actual_frac) = pro.crop_array(
                raw_array, mmts='frac', dims=autofocus_defaults[
                    "crop_fraction"], return_actual_crop=True)

            # Down-sample if the image is not already compressed.
            if micro.defaults["mode"] == 'bayer':
                compressed = pro.down_sample(
                    cropped, autofocus_defaults["pixel_block"])
            else:
                compressed = raw_array

            sharpness_list.append(sharpness_func(compressed))
            positions.append(microscope.stage.position[2])

        # Move to where the sharpness is maximised and measure about it.
        new_position = np.argmax(sharpness_list)
        microscope.stage.focus_rel(-(n - new_position) * step)

        data_arr = np.vstack((positions, sharpness_list)).T
        focus_data.add_data(data_arr, tests, '', attrs={
            'Step size': step, 'Repeats': n,
            'Pixel block': autofocus_defaults["pixel_block"],
            'Actual fraction cropped': actual_frac})


class Tiled(Experiment):
    """Class to conduct experiments where a tiled sequence of images is 
    taken and post-processed. Each such experiments consists of a microscope
    object and a datafile."""
    # TODO Modify the data collection and logging in this class.
    def __init__(self, microscope):
        super(Tiled, self).__init__()
        self.scope = microscope
    
    def run(self, func_list=None, save_every_mmt=True):
        # Set up the data recording.
        n = tiled_dict["n"]
        steps = tiled_dict["steps"]
        attributes = {'n': n, 'steps': steps,
                      'backlash': micro.defaults["backlash"],
                      'focus': tiled_dict["focus"]}

        # This may not be needed.
        image_set = self.create_data_group('tiled_images', attrs=attributes)
        
        # Set mutable default values.
        if func_list is None:
            func_list = [h.bake_in_args(h.unchanged, args=['IMAGE_ARR'])]
        direction = 1
        self.scope.stage.move_rel([-n / 2 * steps, -n / 2 * steps, 0])
        try:
            # A set of results to be collected if save_every_mmt is False.
            results = []

            # For each position in the range specified, take an image, apply
            # all the functions in func_list on it, then either save the
            # measurement if save_every_mmt is true, or append the calculation
            # to a results file and save it all at the end.
            prev_image = None
            for j in range(n):
                for i in range(n):

                    self.scope.stage.move_rel([steps * direction, 0, 0])
                    time.sleep(0.5)
                    image = self.scope.camera.get_frame(greyscale=False)

                    assert image is not prev_image, "Image didn't change."
                    prev_image = image

                    # Post-process.
                    modified = image

                    for function in func_list:
                        modified = function(modified)

                    # Save this array in HDF5 file.
                    if save_every_mmt:
                        #self.scope.datafile.add_data(
                        #    modified, image_set, 'img', attrs={
                        #        'Position': self.scope.stage.position,
                        #        'Cropped size': 300})
                        self.create_dataset('modified_image', attrs={
                                'Position': self.scope.stage.position,
                                'Cropped size': 300}, data=modified)
                    else:
                        results.append([self.scope.stage.position[0],
                                        self.scope.stage.position[1],
                                        self.scope.stage.position[2],
                                        modified])

                self.scope.stage.move_rel([0, steps, 0])
                direction *= -1

            # Move back to original position.
            self.scope.stage.move_rel([-n / 2 * steps, -n / 2 * steps, 0])

            # Save results if they have been incrementally collected.
            if not save_every_mmt:
                results = np.array(results, dtype=np.float)
                self.create_dataset('brightness_results', data=results)
                self.log_messages = ""  # This line is needed because
                # log_messages is only defined if run_in_background is run
                # in Experiment.
                self.log("Test - brightness results added.")
                # self.scope.datafile.add_data(results, image_set, 'data')

        except KeyboardInterrupt:
            print "Aborted, moving back to start"
            self.scope.stage.move_to_pos([0, 0, 0])


def centre_spot(scope):
    """Find the spot on the screen, if it exists, and bring it to the
    centre. If no spot is found, an error is raised.
    :param scope: A microscope object."""

    # TODO Need some way to identify if spot is not on screen.

    transform = scope.calibrate()
    scope.camera.preview()
    # TODO Need to change this to bayer mode to capture entire array.
    frame = scope.camera.get_frame(mode='compressed', greyscale=True)

    # This is strongly affected by any other bright patches in the image -
    # need a better way to distinguish the bright spot.
    thresholded = cv2.threshold(frame, 180, 0, cv2.THRESH_TOZERO)[1]
    gr = scope.datafile.new_group('crop')
    cropped = pro.crop_array(thresholded, mmts='pixel', dims=np.array([
        300, 300]), centre=np.array([0, 0]))
    peak = ndimage.measurements.center_of_mass(thresholded)
    half_dimensions = np.array(np.array(m.get_size(frame)[:2])/2., dtype=int)

    # Note that the stage moves in x and y, so to calculate how much to move
    # by to centre the spot, we need half_dimensions - peak.
    thing = np.dot(half_dimensions - peak[::-1], transform)
    move_by = np.concatenate((thing, np.array([0])))
    scope.stage.move_rel(move_by)
    scope.datafile.add_data(cropped, gr, 'cropped')
    return


if __name__ == '__main__':
    # Control pre-processing manually.
    sys_args = docopt(__doc__)
    print sys_args

    scope = micro.Microscope(filename='dat.hdf5', man=True)

    # Calculate brightness of central spot by taking a tiled section of
    # images, cropping the central 1/64, down sampling the bayer image.
    # Return an array of the positions and the brightness.
    fun_list = [h.bake_in_args(pro.crop_array, args=['IMAGE_ARR'],
                               kwargs={'mmts': 'frac', 'dims': 1/8.}),
                h.bake_in_args(m.brightness, args=['IMAGE_ARR'])]

    if sys_args['autofocus']:
        auto_focus(scope)
    elif sys_args['centre']:
        centre_spot(scope)
    elif sys_args['tiled']:
        tiled = Tiled(scope)
        tiled.run(func_list=fun_list, save_every_mmt=False)


# Code to test the sharpness vs position plot, buts needs modification.
#if __name__ == "__main__":
#    with picamera.PiCamera() as camera:
#        camera.resolution = (640, 480)
#        camera.start_preview()
#        time.sleep(3)   # Let camera to receive image clearly before
    # capturing.
#        capture_to_bgr_array(camera)
#        camera.stop_preview()   # Preview must be stopped afterwards to
#        # prevent Pi freezing on camera screen.
#
#        split_img_data = crop_img_into_n(capture_to_bgr_array(camera), 4800)
#        plotting_data = sharpness_vs_position(*split_img_data)
#
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        [X, Y, Z] = [plotting_data[:, 0], plotting_data[:, 1],
#                     plotting_data[:, 2]]
#        ax.plot_wireframe(X, Y, Z)
#
#        plt.show()



