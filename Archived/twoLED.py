""" 
This control script switches on/off the illumination LEDs of the microscope.

Author: (c) Darryl Foo, 2015

It was used for the paper in Review of Scientific Instruments titled:
A one-piece 3D printed flexure translation stage for open-source microscopy 

This script is released under the GNU General Public License v3.0
"""
import RPi.GPIO as GPIO
import time


class Lightboard:

    #pin numbers
    global _GPIO1
    _GPIO1 = 8
    global _GPIO2
    _GPIO2 = 10

    #using pin numbers
    GPIO.setmode(GPIO.BOARD)

    #set up all pins to off
    GPIO.setup(_GPIO1, GPIO.OUT)
    GPIO.setup(_GPIO2, GPIO.OUT)
    GPIO.output(_GPIO1, False)
    GPIO.output(_GPIO2, False)

    def _close(self):
        GPIO.output(_GPIO1, False)
        GPIO.output(_GPIO2, False)
        GPIO.cleanup()

    def bright(self):
        self._close()
        time.sleep(1)
        GPIO.output(_GPIO1, True)

    def flores(self):
        self._close()
        time.sleep(1)
        GPIO.output(_GPIO2, True)

l=Lightboard()
#default to brightfield imaging for focusing
l.bright()
