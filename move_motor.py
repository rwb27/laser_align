#!/usr/bin/env python

"""Code to connect to the 3 motors and test their motion."""

import smbus
import sys

module = int(sys.argv[1])

x = int(sys.argv[2])
y = int(sys.argv[3])
z = int(sys.argv[4])

bus = smbus.SMBus(1)

# The arguments for write_byte_data are: the I2C address of each motor, the register of the motion (?) and how much to
# move it by.
bus.write_byte_data(0x50, x >> 8, x & 255)
bus.write_byte_data(0x58, y >> 8, y & 255)
bus.write_byte_data(0x6a, z >> 8, z & 255)
