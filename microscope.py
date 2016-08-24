#!/usr/bin/env python

"""microscope.py
This script contains all the classes required to make the microscope work. This
includes the abstract LightSensor and Stage classes and their combination
into a single SensorScope class that allows both to be controlled together. It
is based on the script by James Sharkey."""

import re
import time
import smbus
import serial
import numpy as np
from nplab.instrument import Instrument

import data_io as d


class SensorScope(Instrument):
    """Class to combine camera and stage into a single usable class. The
    microscope may be slow to be created; it must wait for the camera,and stage
    to be ready for use."""
    
    def __init__(self, config, **kwargs):
        """Use this class instead of using the Camera and Stage classes!
        :param config: Either a string specifying a path to the config file,
        ending in .yaml, or the dictionary of default configuration parameters.
        :param kwargs: Specify optional keyword arguments, which will
        override the defaults specified in the config file. Valid kwargs are:
        - channel: Channel of I2C bus to connect to motors for the stage.
        - max_iterations:"""

        # TODO NEED MICROMETRES PER MICROSTEP!

        super(SensorScope, self).__init__()

        # If config is entered as a path string, the file from the path
        # will be read. If it is a dictionary, it is not changed. This
        # prevents the config file being read repeatedly by different
        # objects, rather than once and its value passed around.
        self.config_dict = d.make_dict(config, **kwargs)

        self.sensor = LightDetector(self.config_dict, **kwargs)
        self.stage = Stage(self.config_dict, **kwargs)
        # self.light = twoLED.Lightboard()

        # Set up data recording. Default values will be saved in the group
        # 'SensorScope'.
        self.start = time.time()
        self.attrs = {'startup_time': self.start}
        for key in ["channel", "xyz_bound", "microsteps", "backlash",
                    "override"]:
            self.attrs[key] = self.config_dict[key]
        self.info = self.create_dataset('ScopeSettings', data='',
                                        attrs=self.attrs)

    def __del__(self):
        del self.sensor
        del self.stage
        # del self.light


class LightDetector(Instrument):
    """Class to read brightness value from sensor by providing a Serial
    command to the Arduino. The logging feature of Instrument is not used,
    but can be invoked in future if this class is used directly."""

    def __init__(self, config_file, **kwargs):
        """An abstracted photo-diode class.
        :param config_file: A string with the path to the config file,
        or the times_data.
        :param kwargs:
            tty: The serial connection address as a string.
            baudrate: The baudrate through the port."""

        super(LightDetector, self).__init__()
        # If config_file is entered as a path string, the file from the path
        # will be read. If it is a dictionary, it is not changed. This
        # prevents the config file being read repeatedly by different
        # objects, rather than once and its value passed around.
        self.config_dict = d.make_dict(config_file, **kwargs)

        # Initialise connection as appropriate.
        self.ser = serial.Serial(self.config_dict['tty'],
                                 baudrate=self.config_dict['baudrate'],
                                 timeout=1)

    def __del__(self):
        self.ser.close()

    def read(self, n=1, t=0):
        """Take n measurements in total in the same position with a time delay
        of t seconds between each measurement. Returns them as a list. Note
        that the Arduino is also programmed to have a small delay between
        each measurement being taken."""
        readings = []
        times = _gen(n)
        output_string = 'a'

        while True:
            try:
                # Clear the input and output buffers before taking a reading.
                while self.ser.in_waiting > 0 or self.ser.out_waiting > 0:
                    self.ser.reset_output_buffer()
                    self.ser.reset_input_buffer()
                    print "input buffer size " + str(self.ser.in_waiting)
                    print "output buffer size" + str(self.ser.out_waiting)

                print "output to arduino " + str(self.ser.out_waiting)
                self.ser.write(output_string)
                print "output now " + str(self.ser.out_waiting)

                # Wait for 1s for data, else move on. Set timeout under
                # Serial port opening.
                reading = self.ser.readline()
                print "input now " + str(self.ser.in_waiting)

                if _formatter(reading):
                    # If the reading is a valid one, wait before taking
                    # another as specified, else take one again immediately.
                    times.next()
                    print 'reading', reading
                    reading = re.sub('[^\w]', '', reading.strip())
                    readings.append(int(reading))
                    time.sleep(t)

            except StopIteration:
                break
        return np.array(readings)

    def average_n(self, n, t=0):
        """Take n measurements and return their average value.
        :param n: Number of measurements to take in total.
        :param t: Time delay in seconds between each measurement. Ensure
        this is not too large otherwise time drift may affect the results.
        :return: The average value of the measurements."""
        return np.mean(self.read(n, t))

    # TODO LOOK UP WHAT OTHER ADCS CAN ALSO DO


class Stage(Instrument):

    def __init__(self, config_file, **kwargs):
        """Class representing a 3-axis microscope stage.
        :param config_file: Either file path or dictionary to config file.
        :param kwargs: Valid ones are the xyz_bound, microsteps and channel,
        backlash and override.
        Preferably, change them through the microscope class but in case of
        this class being used elsewhere, kwargs exists but is not logged.
        :param backlash: An array of the backlash along [x, y, z].
        :param override: Set to True to ignore the limits set by _XYZ_BOUND,
        otherwise an error is raised when the bounds are exceeded."""

        super(Stage, self).__init__()

        # If config_file is entered as a path string, the file from the path
        # will be read. If it is a dictionary, it is not changed. This
        # prevents the config file being read repeatedly by different
        # objects, rather than once and its value passed around.
        self.config_dict = d.make_dict(config_file, **kwargs)

        # Check these bounds.
        self._XYZ_BOUND = np.array(self.config_dict["xyz_bound"])
        # How many micro-steps per step?
        self._MICROSTEPS = self.config_dict["microsteps"]
        self.bus = smbus.SMBus(self.config_dict["channel"])
        self.position = np.array([0, 0, 0])

    def move_rel(self, vector):
        """Move the stage by (x,y,z) micro steps.
        :param vector: The increment to move by along [x, y, z]."""
        if type(vector) is not np.ndarray:
            vector = np.array(vector)

        [backlash, override] = [self.config_dict['backlash'],
                                self.config_dict['override']]

        # Check backlash  and the vector to move by have the correct format.
        assert np.all(backlash >= 0), "Backlash must >= 0 for all [x, y, z]."
        backlash = _verify_vector(backlash)
        r = _verify_vector(vector)

        # Generate the list of movements to make. If backlash is [0, 0, 0],
        # there is only one motion to make.
        movements = []
        if np.any(backlash != np.zeros(3)):
            # Subtract an extra amount where vector is negative.
            r[r < 0] -= backlash[np.where(r < 0)]
            r2 = np.zeros(3)
            r2[r < 0] = backlash[np.where(r < 0)]
            movements.append(r2)
        movements.insert(0, r)

        for movement in movements:
            new_pos = np.add(self.position, movement)
            # If all elements of the new position vector are inside bounds (OR
            # overridden):
            if np.all(np.less_equal(
                    np.absolute(new_pos), self._XYZ_BOUND)) or override:
                move_motors(self.bus, *movement)
                self.position = new_pos
            else:
                raise ValueError('New position is outside allowed range.')

    def move_to_pos(self, final):
        new_position = _verify_vector(final)
        rel_mov = np.subtract(new_position, self.position)
        return self.move_rel(rel_mov)

    def focus_rel(self, z):
        """Move the stage in the Z direction by z micro steps."""
        self.move_rel([0, 0, z])

    def centre_stage(self):
        """Move the stage such that self.position is (0,0,0) which in theory
        centres it."""
        self.move_to_pos([0, 0, 0])

    def _reset_pos(self):
        # Hard resets the stored position, just in case things go wrong.
        self.position = np.array([0, 0, 0])


def move_motors(bus, x, y, z):
    """Move the motors for the connected module (addresses hardcoded) by a
    certain number of steps.
    :param bus: The smbus.SMBus object connected to appropriate i2c channel.
    :param x: Move x-direction-controlling motor by specified number of steps.
    :param y: "
    :param z: "."""
    [x, y, z] = [int(x), int(y), int(z)]

    # The arguments for write_byte_data are: the I2C address of each motor,
    # the register and how much to move it by. Currently hardcoded in.
    bus.write_byte_data(0x50, x >> 8, x & 255)
    bus.write_byte_data(0x58, y >> 8, y & 255)
    bus.write_byte_data(0x6a, z >> 8, z & 255)

    # Empirical formula for how micro step values relate to rotational speed.
    # This is only valid for the specific set of motors tested.
    time.sleep(np.ceil(max([abs(x), abs(y), abs(z)])) * (1.5 / 1000) + 0.1)


def _verify_vector(vector):
    """Checks the input vector has 3 components."""
    r = np.array(vector)
    assert r.shape == (3,), "The variable 'vector' must have 3 components."
    return r


def _formatter(reading):
    """Checks a reading has a desired format and returns True if so,
    else returns False."""
    sub_readings = re.findall('(?<![0-9])([0-9]|[1-9][0-9]|[1-9][0-9][0-9]|10['
                              '0-1][0-9]|102[0-3])\s+', reading)
    if len(sub_readings) != 1:
        return False
    else:
        return True


def _gen(n):
    i = 0
    while i < n:
        yield i
        i += 1


if __name__ == '__main__':
    scope = SensorScope('./configs/config.yaml')
    print scope.sensor.read(5, t=10)
