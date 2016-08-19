import serial
import re


def formatter(reading):
    """Checks a reading has a desired format and returns True if so,
    else returns False."""
    sub_readings = re.findall('(?<![0-9])([0-9]|[1-9][0-9]|[1-9][0-9][0-9]|10['
                              '0-1][0-9]|102[0-3])\s+', reading)
    if len(sub_readings) != 1:
        return False
    else:
        return True


def gen(n):
    i = 0
    while i < n:
        yield i
        i += 1

# Initialise connection as appropriate.
ser = serial.Serial(port='\\.\COM3', baudrate=9600,
                    bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE, timeout=1)
data=[]
times = gen(1000)
while True:
    try:
        reading = ser.readline()
        if formatter(reading):
            next(times)
            data.append(int(reading.strip()))
    except StopIteration:
        break

ser.close()
print data


