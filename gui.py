#!/usr/bin/env python

"""gui.py
Script containing class to control and take measurements using the
microscope via a GUI."""

import datetime
import cv2

import caller
import data_io
import microscope as micro


class ScopeGUI:
    """Class to control the GUI used for the microscope."""

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

    _ARROW_STEP_SIZE = data_io.scope_defs["key_stepsize"]

    def __init__(self, width=data_io.scope_defs["resolution"][0],
                 height=data_io.scope_defs["resolution"][1], cv2camera=False,
                 channel=data_io.scope_defs["channel"]):
        """Optionally specify a width and height for Camera object, the channel
        for the Stage object and a filename for the attached datafile.
        cv2camera allows non-RPi systems to be tested also."""

        # Create a new Microscope object.
        self.microscope = micro.Microscope(width, height, cv2camera, channel)

        # Set up the GUI variables:
        self._gui_quit = False
        self._gui_greyscale = True
        self._gui_img = None
        self._gui_pause_img = None
        self._gui_drag_start = None
        self._gui_sel = None
        self._gui_tracking = False
        self._gui_bead_pos = None
        self._gui_color = (0, 0, 0)  # BGR colour

        # And the rest:
        self.template_selection = None

    def __del__(self):
        """Closes the attached objects properly by deleting them."""
        cv2.destroyAllWindows()
        del self.microscope

    @staticmethod
    def _gui_nothing(x):
        """GUI needs callbacks for some functions: this is a blank one."""
        pass

    def _create_gui(self):
        """Initialises the things needed for the GUI."""
        # Create the necessary GUI elements
        # cv2.namedWindow('Preview', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Controls', cv2.WINDOW_AUTOSIZE)
        self.microscope.camera.preview()
        cv2.createTrackbar('Greyscale', 'Controls', 0, 1, self._gui_nothing)
        cv2.createTrackbar('Tracking', 'Controls', 0, 1, self._gui_nothing)
        # Set default values
        cv2.setTrackbarPos('Greyscale', 'Controls', 1)
        cv2.setTrackbarPos('Tracking', 'Controls', 0)
        # Add mouse functionality on image click:
        # cv2.setMouseCallback('Preview', self._on_gui_mouse)
        # For the sake of speed, use the RPi iterator:
        self.microscope.camera.use_iterator(True)

    def _read_gui_track_bars(self):
        """Read in and process the track bar values."""
        self._gui_greyscale = bool(cv2.getTrackbarPos('Greyscale', 'Controls'))
        self._gui_tracking = \
            (bool(cv2.getTrackbarPos('Tracking', 'Controls')) and
             (self._gui_sel is not None) and (self._gui_drag_start is None))

    def _stop_gui_tracking(self):
        """Run the code necessary to cleanup after tracking stopped."""
        self._gui_sel = None
        self._gui_drag_start = None
        self.template_selection = None
        cv2.setTrackbarPos('Tracking', 'Controls', 0)
        self._gui_tracking = False
        self._gui_bead_pos = None

    def _update_gui(self):
        """Run the code needed to update the GUI to latest frame."""
        # Take image if not paused:
        #if self._gui_pause_img is None:
        #    self._gui_img = self.microscope.camera.get_frame(
        #        greyscale=self._gui_greyscale)
        #else:
        #    # If paused, use a fresh copy of the pause frame
        #    self._gui_img = self._gui_pause_img.copy()
        #
        ## Now do the tracking, before the rectangle is drawn!
        #if self._gui_tracking:
        #    self._update_gui_tracker()
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
            if keypress == ord(data_io.scope_defs["exit"]):
                self._gui_quit = True
            elif keypress == ord(data_io.scope_defs["save"]):
                # The g key will 'grab' (save) the box region or the whole
                # frame if nothing selected.
                fname = datetime.datetime.now().strftime("%Y%mmts%d_%H%M%S")
                if self._gui_sel is None:
                    cv2.imwrite("microscope_img_%s.jpg" % fname, self._gui_img)
                else:
                    w, h = self._gui_sel[2] - self._gui_sel[0], \
                           self._gui_sel[3] - self._gui_sel[1]
                    crop = self._gui_img[self._gui_sel[1]: self._gui_sel[1]+h,
                           self._gui_sel[0]: self._gui_sel[0]+w]
                    cv2.imwrite("microscope_img_%s.jpg" % fname, crop)

            elif keypress == ord(data_io.scope_defs["save_stored_image"]):
                # The t key will save the stored template image.
                fname = datetime.datetime.now().strftime("%Y%mmts%d_%H%M%S")
                cv2.imwrite("template_%s.jpg" % fname, self.template_selection)
            elif keypress == ord(data_io.scope_defs["stop_tracking"]):
                # Reset the template selection box and stop tracking.
                self._stop_gui_tracking()
            # QWEASD for 3D motion: WASD are motion in +y, -x, -y,
            # +x directions, QE are motion in +z, -z directions. Note these
            # need to be CHANGED DEPENDING ON THE CAMERA ORIENTATION. NOT
            # SURE ABOUT Q AND E YET.
            elif keypress == ord(data_io.scope_defs["+x"]):
                # The arrow keys will move the stage
                self.microscope.stage.move_rel([self._ARROW_STEP_SIZE, 0, 0])
            elif keypress == ord(data_io.scope_defs["-x"]):
                self.microscope.stage.move_rel([-self._ARROW_STEP_SIZE, 0, 0])
            elif keypress == ord(data_io.scope_defs["+y"]):
                self.microscope.stage.move_rel([0, self._ARROW_STEP_SIZE, 0])
            elif keypress == ord(data_io.scope_defs["-y"]):
                self.microscope.stage.move_rel([0, -self._ARROW_STEP_SIZE, 0])
            elif keypress == ord(data_io.scope_defs["-z"]):
                self.microscope.stage.move_rel([0, 0, -self._ARROW_STEP_SIZE])
            elif keypress == ord(data_io.scope_defs["+z"]):
                self.microscope.stage.move_rel([0, 0, self._ARROW_STEP_SIZE])
            elif keypress == ord(data_io.scope_defs["invert_colour"]):
                # Inverts the selection box colour.
                if self._gui_color == (0, 0, 0):
                    self._gui_color = (255, 255, 255)  # White
                else:
                    self._gui_color = (0, 0, 0)        # Black

                    # Finally process the image, drawing boxes etc:
                    #if self._gui_sel is not None:
                    #    cv2.rectangle(self._gui_img, (self._gui_sel[0], self._gui_sel[
                    #        1]), (self._gui_sel[2], self._gui_sel[3]), self._gui_color)
                    #cv2.imshow('Preview', self._gui_img)

    def _update_gui_tracker(self):
        """Code to update the position of the selection box if tracking is
        enabled."""
        assert ((self.template_selection is not None) and
                self._gui_tracking and (self._gui_bead_pos is not None))
        w, h = self.template_selection.shape[::-1]
        try:
            if (w >= 100) or (h >= 100):
                # If template bigger than the default search box, enlarge it.
                d = max(w, h) + 50
                centre = self.microscope.camera.find_template(
                    self.template_selection, self._gui_img, self._gui_bead_pos,
                    box_d=d)
            else:
                centre = self.microscope.camera.find_template(
                    self.template_selection, self._gui_img, self._gui_bead_pos)

        except RuntimeError:
            # find_template raises RuntimeError if region exceeds image
            # bounds. If this occurs: just stop following it for now!
            self._stop_gui_tracking()
            return

        self._gui_bead_pos = centre
        # The template top left corner.
        x1, y1 = int(centre[0] - w/2), int(centre[1] - h/2)
        # The template bottom right corner.
        x2, y2 = int(centre[0] + w/2), int(centre[1] + h/2)
        # The selection is top left to bottom right.
        self._gui_sel = (x1, y1, x2, y2)

    #def _on_gui_mouse(self, event, x, y, flags):
    #    """Code to run on mouse action on GUI preview image."""
    #    # This is the bounding box selection: the start, end and
    #    # intermediate parts respectively.
#
    #    if (event == cv2.EVENT_LBUTTONDOWN) and (self._gui_sel is None):
    #        # Pause the display, and set initial coords for the bounding box:
    #        self._gui_pause_img = self._gui_img
    #        self._gui_drag_start = (x, y)
    #        self._gui_sel = (x, y, x, y)
#
    #    elif (event == cv2.EVENT_LBUTTONUP) and \
    #            (self._gui_drag_start is not None):
    #        # Finish setting the bounding box coords and unpause.
    #        self._gui_sel = (min(self._gui_drag_start[0], x),
    #                         min(self._gui_drag_start[1], y),
    #                         max(self._gui_drag_start[0], x),
    #                         max(self._gui_drag_start[1], y))
#
    #        w, h = self._gui_sel[2] - self._gui_sel[0], \
    #            self._gui_sel[3] - self._gui_sel[1]
#
    #        self.template_selection = self._gui_pause_img[
    #                                  self._gui_sel[1]: self._gui_sel[1]+h,
    #                                  self._gui_sel[0]:self._gui_sel[0]+w]
#
    #        if not self._gui_greyscale:
    #            self.template_selection = cv2.cvtColor(
        # self.template_selection,
    #                                                   cv2.COLOR_BGR2GRAY)
#
    #        self._gui_bead_pos = (int((self._gui_sel[0]+self._gui_sel[
        # 2])/2.0),
    #                              int((self._gui_sel[1]+self._gui_sel[
        # 3])/2.0))
    #        self._gui_pause_img = None
    #        self._gui_drag_start = None
#
    #    elif (event == cv2.EVENT_MOUSEMOVE) and \
    #            (self._gui_drag_start is not None) and \
    #            (flags == cv2.EVENT_FLAG_LBUTTON):
    #        # Set the bounding box to some intermediate value; don't unpause.
    #        self._gui_sel = (min(self._gui_drag_start[0], x),
    #                         min(self._gui_drag_start[1], y),
    #                         max(self._gui_drag_start[0], x),
    #                         max(self._gui_drag_start[1], y))

    def run_gui(self):
        """Run the GUI."""
        self._create_gui()
        while not self._gui_quit:
            self._read_gui_track_bars()
            self._update_gui()

        # Triggered when the GUI quit key is pressed.
        # self.microscope.stage.centre_stage() Uncomment to move scope back
        # to original position upon closing.
        #cv2.destroyWindow('Preview')
        cv2.destroyWindow('Controls')
        self.microscope.camera.preview()
        self._gui_quit = False  # This allows restarting of the GUI
