""" REVISION 08-02-2015 - jps79

This control script communicates with an Arduino microcontroller that 
drives the stepper motors controlling the stage.

Author: (c) Darryl Foo, James Sharkey, 2015

It was used for the paper in Review of Scientific Instruments titled:
A one-piece 3D printed flexure translation stage for open-source microscopy 

This script is released under the GNU General Public License v3.0
"""
# To use as a test class on a computer without a serial connection to an
# Arduino stage: simply comment out all lines containing self._ser
import numpy as np
import serial


class Stage:
    # Check these bounds.
    _XYZ_BOUND = np.array([32000, 32000, 5000])
    _MICROSTEPS = 16  # How many micro-steps per step

    # Probably nothing to do here, KIV.
    def __init__(self, tty="/dev/ttyACM0"):
        """Class representing a 3-axis microscope stage.

            If the serial device is not found, it will be emulated by default(!)
            and a warning message printed."""
        self._emulate = False
        self._pos = np.array([0, 0, 0])
        try:  # Attempt to open the stage:
            self._ser = serial.Serial(tty)
        except serial.serialutil.SerialException:
            print "Emulating Stage!"
            self._emulate = True  # If it fails, emulate a stage
        if self._emulate:
            self.ver = "Emulated Stage"
        else:  # When the Arduino is connected, it sends code version details to say it is ready:
            self.ver = self._ser.readline()  # When opening, read in the start-up line
            self.ver = self.ver.replace("\r\n", "")  # Remove extra characters and store

    def _close(self):
        """Close serial comms, turn off motors if necessary."""
        self.release()
        if not self._emulate:
            self._ser.close()

    def __del__(self):
        self._close()

    def _motor_coord(self, x, y, z):
        # In case the microscope configuration changes or to alter directions; not very important
        return x, y, z

    def _query(self, command):
        """Send a command to the Arduino, clearing input buffer and waiting
           for command to complete."""
        if not self._emulate:  # Don't send serial commands if we're emulating
            self._ser.read(self._ser.inWaiting())  # Flush the input buffer
            self._ser.write(command)  # Send the command
            ret = self._ser.readline()  # Wait for it to stop moving
        else:
            ret = "emulated"
        return ret.replace("\r\n", "")  # Return the stage message removing junk chars
##need to change this to fit new syntax for movement
    def move_rel(self, vector, release=True, override=False):##maybe change the release default
        """Move the stage by vector=[x,y,z] microsteps.

            - If precision is required, set release to False: this keeps the motor
              magnets on, holding slide in position. Use release() to turn off again.
            - Does slide XYZ range checking: override with extreme caution!"""
        r = np.array(vector)
        assert r.shape==(3, ), "move_rel must have a 3 component vector."
        new_pos = np.add(self._pos, r)
        # If all elements of the new position vector are inside bounds (OR overridden):
        if np.all(np.less_equal(np.absolute(new_pos), self._XYZ_BOUND)) or override:
            ret = self._query("move x%d y%d z%d\n" % self._motor_coord(r[0], r[1], r[2]))
            self._pos = new_pos
            if release:
                self.release()
        else:
            ret = "bounds_error"
#        return ret  # To aid error checking, returns Arduino message or error
##keep this? not sure if this will ever be used
    def fast_move(self, vector, release=True, override=False):
        """Move the stage by vector=[x,y,z] microsteps, but move using whole steps.

            - If x, y, z are NOT multiples of 16 - will ignore remaining microsteps.
            - Does slide XYZ range checking: override with extreme caution!"""
        r = np.array(vector)
        assert r.shape==(3, ), "fast_move must have a 3 component vector."
        new_pos = np.add(self._pos, r)
        # If all elements of the new position vector are inside bounds (OR overridden):
        if np.all(np.less_equal(np.absolute(new_pos), self._XYZ_BOUND)) or override:
            (step_x, step_y, step_z) = (r[0]/self._MICROSTEPS, r[1]/self._MICROSTEPS, r[2]/self._MICROSTEPS)
            ret = self._query("fast_move %d %d %d\n" % self._motor_coord(step_x, step_y, step_z))
            self._pos = new_pos
            if release:
                self.release()
        else:
            ret = "bounds_error"
#        return ret  # To aid error checking, returns Arduino message or error

    def move_to_pos(self, vector, release=True, override=False):
        r = np.array(vector)
        assert r.shape==(3, ), "move_to_pos must have a 3 component vector."
        rel_mov = np.subtract(r, self._pos)
        return self.move_rel(rel_mov, release, override)

    def focus_rel(self, z, release=True):
        """Move the stage in the Z direction by z microsteps."""
        self.move_rel(np.array([0, 0, z]), release)

    def centre_stage(self):
        """Move the stage such that self._pos is (0,0,0) which in theory centres it."""
        new_pos = -1 * self._pos
#        self.fast_move(new_pos)
        self.move_rel(new_pos)

    def release(self):
        """Manually turn off the motors, if left on using optional argument in
        move_rel and focus_rel."""
        ret = self._query("release\n")  # Turn off the motors
#        return ret

    def _reset_pos(self):
        # Hard resets the stored position, just in case things go wrong
        self._pos = np.array([0, 0, 0])

    def current_pos(self):
        print self._pos
