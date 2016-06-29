#!/usr/bin/env python

"""scope_stage.py
Used to control the microscope translation stage from Python."""

import smbus
import time
from math import ceil
import numpy as np
import gen_helpers as h


class ScopeStage:
    # Check these bounds.
    _XYZ_BOUND = np.array([32000, 32000, 32000])
    _MICROSTEPS = 16  # How many micro-steps per step.

    def __init__(self, channel=1):
        """Class representing a 3-axis microscope stage."""
        self.bus = smbus.SMBus(channel)
        time.sleep(3)
        self.position = np.array([0, 0, 0])

    def move_rel(self, vector, override=False):
        """Move the stage by (x,y,z) micro steps.
        :param vector: The increment to move by along [x, y, z].
        :param override: Set to True to ignore the limits set by _XYZ_BOUND,
        otherwise an error is raised when the bounds are exceeded."""
        r = h.verify_vector(vector)
        new_pos = np.add(self.position, r)
        # If all elements of the new position vector are inside bounds (OR
        # overridden):
        if np.all(np.less_equal(np.absolute(new_pos), self._XYZ_BOUND)) or \
                override:
            _move_motors(self.bus, *r)
            self.position = new_pos
        else:
            raise ValueError('New position is outside allowed range.')

    def move_to_pos(self, final, override=False):
        new_position = h.verify_vector(final)
        rel_mov = np.subtract(new_position, self.position)
        return self.move_rel(rel_mov, override)

    def focus_rel(self, z):
        """Move the stage in the Z direction by z micro steps."""
        self.move_rel([0, 0, z])

    def centre_stage(self):
        """Move the stage such that self.position is (0,0,0) which in theory
        centres it."""
        self.move_to_pos([0, 0, 0])

    def current_pos(self):
        print self.position

    def _reset_pos(self):
        # Hard resets the stored position, just in case things go wrong.
        self.position = np.array([0, 0, 0])


def _move_motors(bus, x, y, z, mod_no=None):
    """Move the motors for an optionally specified module number by a
    certain number of steps.
    :param bus: The smbus.SMBus object connected to appropriate i2c channel.
    :param x: Move x-direction-controlling motor by specified number of steps.
    :param y: "
    :param z: "
    :param mod_no: The module number to control."""
    [x, y, z] = [int(x), int(y), int(z)]

    # The arguments for write_byte_data are: the I2C address of each motor,
    # the register and how much to move it by.
    # Currently hardcoded in, consider looking this up upon program run.
    bus.write_byte_data(0x50, x >> 8, x & 255)
    bus.write_byte_data(0x58, y >> 8, y & 255)
    bus.write_byte_data(0x6a, z >> 8, z & 255)

    # Empirical formula for how micro step values relate to rotational speed.
    # This is only valid for the specific set of motors tested.
    time.sleep(ceil(max([abs(x), abs(y), abs(z)]))/1000 + 2)

