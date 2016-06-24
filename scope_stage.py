#!/usr/bin/env python

"""scope_stage.py
Used to control the microscope translation stage from Python."""

import smbus
import time
from math import ceil


class ScopeStage:
    def __init__(self, channel=1):
        """Class representing a 3-axis microscope stage."""
        self.bus = smbus.SMBus(channel)
        time.sleep(3)
        self.position = [0, 0, 0]
        print self.position

    def move_rel(self, x, y, z):
        """Move the stage by (x,y,z) micro steps."""
        _move_motors(self.bus, x, y, z)
        for i, pos in enumerate([x, y, z]):
            self.position[i] += pos
        print self.position

    def focus_rel(self, z):
        """Move the stage in the Z direction by z micro steps."""
        self.move_rel(0, 0, z)


def _move_motors(bus, x, y, z, mod_no=None):
    """Move the motors for an optionally specified module number by a
    certain number of steps.
    :param bus: The smbus.SMBus object connected to the appropriate i2c
    channel.
    :param x: Move x-direction-controlling motor by specified number of steps.
    :param y: "
    :param z: "
    :param mod_no: The module number to control.
    """
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
