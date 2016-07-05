"""experiments.py
Contains functions to perform a set of measurements and output the results
to a datafile or graph. These functions are built on top of measurements.py,
microscope.py, image_proc.py and data_io.py and come with their own JSON
config files. NOTE: Whenever using microscope.py, ensure that its module-wide
variable 'defaults' is correctly defined."""

import time
import numpy as np
import data_io as d
import image_proc as proc
import measurements as m
import microscope as micro


def auto_focus(microscope, config='./configs/autofocus.json'):
    """Autofocus the image on the camera by varying z only, using a given
    method of calculating the sharpness.
    :param microscope: The microscope object, containing the camera and stage.
    :param config: The JSON config file for this function."""

    # Read the relevant config files. microscope_defaults.json is used to
    # set everything about the microscope (see microscope.py), and
    # autofocus.json for this procedure.
    autofocus_defaults = d.config_read(config)

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
    autofocus_data = d.Datafile(filename=autofocus_defaults["filename"])
    tests = autofocus_data.new_group(autofocus_defaults["group_name"],
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
            (cropped, actual_percent) = proc.crop_section(
                raw_array, autofocus_defaults["crop_percent"])

            # Down-sample if the image is not already compressed.
            if micro.defaults["mode"] == 'bayer':
                compressed = proc.down_sample(
                    cropped, autofocus_defaults["pixel_block"])
            else:
                compressed = raw_array

            sharpness_list.append(sharpness_func(compressed))
            positions.append(microscope.stage.position[2])

        # Move to where the sharpness is maximised and measure about it.
        new_position = np.argmax(sharpness_list)
        microscope.stage.focus_rel(-(n - new_position) * step)

        data_arr = np.vstack((positions, sharpness_list)).T
        autofocus_data.add_data(data_arr, tests, '', attrs={
            'Step size': step, 'Repeats': n,
            'Pixel block': autofocus_defaults["pixel_block"],
            'Actual percentage cropped': actual_percent})


def tiled(microscope, config='./configs/tiled_image.json'):
    """Take a tiled image of a sample, taking account of backlash.
    :param microscope: The microscope object, containing the stage and
    camera objects to be controlled. Also can be used for calibration.
    :param config: The JSON config file for this function.
    :return: A combined array of the tiled image."""

    # Set up the data recording.
    tiled_dict = d.config_read(config)
    n = tiled_dict["n"]
    steps = tiled_dict["steps"]
    attributes = {'n': n, 'steps': steps,
                  'backlash': micro.defaults["backlash"],
                  'focus': tiled_dict["focus"]}
    tiled_data = d.Datafile(filename=tiled_dict["filename"])
    image_set = tiled_data.new_group('tiled_image', attrs=attributes)

    direction = 1
    microscope.stage.move_rel([-n / 2 * steps, -n / 2 * steps, 0])
    try:
        for j in range(n):
            for i in range(n):
                microscope.stage.move_rel([steps * direction, 0, 0])
                time.sleep(0.5)
                image = microscope.camera.get_frame()
                (image, actual_percent) = proc.crop_section(image, tiled_dict[
                    "crop_percent"])
                image = proc.down_sample(image, tiled_dict["pixel_block"])
                tiled_data.add_data(image, image_set, 'img', attrs={
                    'Position': microscope.stage.position,
                    'Cropped fraction': actual_percent})
            microscope.stage.move_rel([0, steps, 0])
            direction *= -1
        microscope.stage.move_rel([-n / 2 * steps, -n / 2 * steps, 0])
    except KeyboardInterrupt:
        print "Aborted, moving back to start"
        microscope.stage.move_to_pos([0, 0, 0])
