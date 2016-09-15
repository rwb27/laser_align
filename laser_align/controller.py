#!/usr/bin/env python

"""Script containing class to control and take measurements using the
microscope via the keyboard."""

import cv2

import _experiments as _exp
import data_io as d
import numpy as np


class KeyboardControls:
    """Class to control the movement of the microscope using the keyboard."""

    # Key codes for Windows (W) and Linux (L), to allow conversion:
    _GUI_W_KEYS = {2490368: "UP", 2621440: "DOWN", 2424832: "LEFT",
                   2555904: "RIGHT"}
    _GUI_L_KEYS = {"UP": 82, "DOWN": 84, "LEFT": 81, "RIGHT": 83}

    # Some useful text key code constants to avoid unreadable code:
    _GUI_KEY_UP = 82
    _GUI_KEY_DOWN = 84
    _GUI_KEY_LEFT = 81
    _GUI_KEY_RIGHT = 83
    _GUI_KEY_SPACE = 32
    _GUI_KEY_ENTER = 13

    def __init__(self, microscope, config_file, **kwargs):
        """Optionally specify a width and height for Camera object, the channel
        for the Stage object and a filename for the attached datafile.
        cv2camera allows non-RPi systems to be tested also.
        Valid kwargs are:
            resolution, cv2camera, channel, manual and um_per_pixel and
            camera_stage_transform, mode, tolerance, max_iterations,
            max_resolution, key_stepsize. Keypress controls can only be
            changed from the config file itself."""

        self.config_dict = d.make_dict(config_file, **kwargs)

        # Create a new SensorScope object.
        self.microscope = microscope

        # Set up the GUI variables:
        self._ARROW_STEP_SIZE = self.config_dict["key_stepsize"]
        self._gui_quit = False
        self._gui_drag_start = None
        self._gui_sel = None

    def __del__(self):
        """Closes the attached objects properly by deleting them."""
        cv2.destroyAllWindows()
        del self.microscope

    @staticmethod
    def _create_gui():
        """Initialises the things needed for the GUI."""
        # Create the necessary GUI elements
        print "Creating GUI - use QWEASD to move and X to quit."
        cv2.namedWindow('Controls', cv2.WINDOW_AUTOSIZE)

    def _update_gui(self):
        """Run the code needed to update the GUI to latest frame."""
        # Now process keyboard input:
        keypress = cv2.waitKey()

        # Skip all the unnecessary if statements if no keypress
        if keypress != -1:
            # This converts Windows arrow keys to Linux
            if keypress in self._GUI_W_KEYS:
                keypress = self._GUI_L_KEYS[self._GUI_W_KEYS[keypress]]
            else:
                # The 0xFF allows ordinary Linux keys to work too
                keypress &= 0xFF

            # Now process the keypress:
            if keypress == ord(self.config_dict["exit"]):
                self._gui_quit = True
            elif keypress == ord(self.config_dict["save"]):
                # The g key will 'grab' (save) the time, position and a
                # singly-sampled brightness value.
                time = _exp.elapsed(self.microscope.start)
                brightness = self.microscope.sensor.average_n(10)
                position = self.microscope.stage.position
                reading = np.hstack((np.array(time), position,
                                    brightness))
                print reading
                self.microscope.create_dataset('GUI_save', data=reading)

            # QWEASD for 3D motion: WASD are motion in +y, -x, -y,
            # +x directions, QE are motion in +z, -z directions. Note these
            # need to be CHANGED DEPENDING ON THE AXIS ORIENTATION. NOT
            # SURE ABOUT Q AND E YET.
            elif keypress == ord(self.config_dict["+x"]):
                self.microscope.stage.move_rel([self._ARROW_STEP_SIZE, 0, 0])
            elif keypress == ord(self.config_dict["-x"]):
                self.microscope.stage.move_rel([-self._ARROW_STEP_SIZE, 0, 0])
            elif keypress == ord(self.config_dict["+y"]):
                self.microscope.stage.move_rel([0, self._ARROW_STEP_SIZE, 0])
            elif keypress == ord(self.config_dict["-y"]):
                self.microscope.stage.move_rel([0, -self._ARROW_STEP_SIZE, 0])
            elif keypress == ord(self.config_dict["-z"]):
                self.microscope.stage.move_rel([0, 0, -self._ARROW_STEP_SIZE])
            elif keypress == ord(self.config_dict["+z"]):
                self.microscope.stage.move_rel([0, 0, self._ARROW_STEP_SIZE])

    def run_gui(self):
        """Run the GUI."""
        self._create_gui()
        while not self._gui_quit:
            self._update_gui()

        # Triggered when the GUI quit key is pressed.
        # self.microscope.stage.centre_stage() Uncomment to move scope back
        # to original position upon closing.
        cv2.destroyWindow('Controls')
        self._gui_quit = False  # This allows restarting of the GUI
