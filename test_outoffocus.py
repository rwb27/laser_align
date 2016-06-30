"""Makes image out of focus for testing."""

import smbus

z = -30000
bus = smbus.SMBus(1)
bus.write_byte_data(0x6a, z >>8, z & 255)