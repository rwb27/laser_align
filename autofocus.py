#!/usr/bin/env python

"""autofocus.py
Auto-focuses the camera using an algorithm that compares the Laplacian of
the image sharpness."""

import time

import numpy as np
import picamera.array

import data_output as d
import scope_stage as s
from capture import capture_raw, capture_compressed
from image_mmts import sharpness_lap
import image_proc as proc


def auto_focus(cam, sharpness_func=sharpness_lap, capture_func=capture_raw,
               mmt_range=None, backlash=128, res=(640, 480), crop_percent=50,
               pixel_block=4):
    """Autofocus the image on the camera by varying z only, using a given
    method of calculating the sharpness.
    :param cam: The camera object.
    :param sharpness_func: The function object used to calculate the
    sharpness. Calculate the sharpness using the down-sampled image?
    :param capture_func: The function object for the image capture method,
    which returns a BGR array.
    :param mmt_range: A list of tuples specifying the range to measure in
    the format [(step_size, repeats), ...]. This function measures from
    -step * n / 2 to +step * n / 2 in steps of n, allowing for backlash.
    :param backlash: The number of micro-steps to wind up by, so that
    measurements are made by rotating the motors in the same direction.
    :param res: A tuple of the camera resolution to set. Irrelevant for raw
    image capture.
    :param crop_percent: The percentage of the image to crop along each
    axis, i.e. 50% crops the central 50% of the image respectively along x
    and y.
    :param pixel_block: Forms a block of with pixel_block**2 pixels when
    down-sampling."""

    # TODO: Crop the centre of the image before sharpness calculation,
    # TODO: down-sample before capture, specify the range over which to
    # TODO: measure and in what steps.

    # Initialise camera and stage. TODO: Use the Microscope class instead.
    cam.resolution = res
    cam.start_preview()
    time.sleep(2)  # Let camera to receive image clearly before capturing.
    stage = s.ScopeStage()

    if mmt_range is None:
        mmt_range = [(1000, 20), (200, 10), (200, 10), (100, 12)]

    attributes = {'Resolution': res, 'Backlash': backlash}
    if sharpness_func == sharpness_lap:
        attributes['Sharpness function'] = 'Laplacian'
    if capture_func == capture_raw:
        attributes['Capture function'] = 'Raw Bayer'
    elif capture_func == capture_compressed:
        attributes['Capture function'] = 'Compressed'

    autofocus_data = d.Datafile(filename='autofocus.hdf5')
    tests = autofocus_data.new_group('tests', attrs=attributes)
    for step, n in mmt_range:
        sharpness_list = []
        positions = []

        # Initial capture
        stage.focus_rel(-step * n / 2 - backlash)
        stage.focus_rel(backlash)

        sharpness_list.append(sharpness_func(capture_func(cam)))
        positions.append(stage.position[2])

        # Remaining n measurements.
        for i in range(n):
            stage.focus_rel(step)
            raw_array = capture_func(cam)
            (cropped, actual_percent) = proc.crop_centre(
                raw_array, crop_percent)
            compressed = proc.down_sample(cropped, pixel_block)
            sharpness_list.append(sharpness_lap(compressed))
            positions.append(stage.position[2])

        # Move to where the sharpness is maximised and measure about it.
        new_position = np.argmax(sharpness_list)
        stage.focus_rel(-(n - new_position) * step - backlash)
        stage.focus_rel(backlash)

        data_arr = np.vstack((positions, sharpness_list)).T
        autofocus_data.add_data(data_arr, tests, '', attrs={
            'Step size': step, 'Repeats': n, 'Pixel block': pixel_block,
            'Actual percentage cropped': actual_percent})


if __name__ == "__main__":
    with picamera.PiCamera() as camera:
        auto_focus(camera)
        #    plt.plot(positions, sharpness_list, 'o-')
        #    plt.xlabel('position (Microsteps)')
        #    plt.ylabel('Sharpness (a.u.)')
        #    time.sleep(5)
        #
        #plt.show()
        #
        #print "Done :)"

'''
plt.figure()

plt.imshow(image_bw)

plt.figure()

plt.imshow(image_lap)

plt.show()
'''
