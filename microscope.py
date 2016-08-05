#!/usr/bin/env python

"""microscope.py
This script contains all the classes required to make the microscope
work. This includes the abstract Camera and ScopeStage classes and
their combination into a single Microscope class that allows both to be
controlled together. It is based on the script by James Sharkey, which was used
for the paper in Review of Scientific Instruments titled: A one-piece 3D
printed flexure translation stage for open-source microscopy. NOTE: Whenever
using microscope.py, ensure that its module-wide variable 'defaults' is
correctly defined."""

import datetime
import io
import sys
import time

import cv2
import numpy as np
import smbus
from scipy import ndimage

import data_io
import helpers as h
import image_proc as proc

# import twoLED
import image_proc

try:
    import picamera
    import picamera.array
except ImportError:
    pass  # Don't fail on error; simply force cv2 camera later

# Read defaults from config file.
defaults = data_io.config_read('./configs/microscope_defaults.json')


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

    _ARROW_STEP_SIZE = defaults["key_stepsize"]

    def __init__(self, width=defaults["resolution"][0],
                 height=defaults["resolution"][1], cv2camera=False,
                 channel=defaults["channel"],
                 filename=defaults["filename"]):
        """Optionally specify a width and height for Camera object, the channel
        for the Stage object and a filename for the attached datafile.
        cv2camera allows non-RPi systems to be tested also."""

        # Create a new Microscope object.
        self.microscope = Microscope(width, height, cv2camera, channel,
                                     filename)

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
            if keypress == ord(defaults["exit"]):
                self._gui_quit = True
            elif keypress == ord(defaults["save"]):
                # The g key will 'grab' (save) the box region or the whole
                # frame if nothing selected.
                fname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                if self._gui_sel is None:
                    cv2.imwrite("microscope_img_%s.jpg" % fname, self._gui_img)
                else:
                    w, h = self._gui_sel[2] - self._gui_sel[0], \
                           self._gui_sel[3] - self._gui_sel[1]
                    crop = self._gui_img[self._gui_sel[1]: self._gui_sel[1]+h,
                           self._gui_sel[0]: self._gui_sel[0]+w]
                    cv2.imwrite("microscope_img_%s.jpg" % fname, crop)

            elif keypress == ord(defaults["save_stored_image"]):
                # The t key will save the stored template image.
                fname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite("template_%s.jpg" % fname, self.template_selection)
            elif keypress == ord(defaults["stop_tracking"]):
                # Reset the template selection box and stop tracking.
                self._stop_gui_tracking()
            # QWEASD for 3D motion: WASD are motion in +y, -x, -y,
            # +x directions, QE are motion in +z, -z directions. Note these
            # need to be CHANGED DEPENDING ON THE CAMERA ORIENTATION. NOT
            # SURE ABOUT Q AND E YET.
            elif keypress == ord(defaults["+x"]):
                # The arrow keys will move the stage
                self.microscope.stage.move_rel([self._ARROW_STEP_SIZE, 0, 0])
            elif keypress == ord(defaults["-x"]):
                self.microscope.stage.move_rel([-self._ARROW_STEP_SIZE, 0, 0])
            elif keypress == ord(defaults["+y"]):
                self.microscope.stage.move_rel([0, self._ARROW_STEP_SIZE, 0])
            elif keypress == ord(defaults["-y"]):
                self.microscope.stage.move_rel([0, -self._ARROW_STEP_SIZE, 0])
            elif keypress == ord(defaults["-z"]):
                self.microscope.stage.move_rel([0, 0, -self._ARROW_STEP_SIZE])
            elif keypress == ord(defaults["+z"]):
                self.microscope.stage.move_rel([0, 0, self._ARROW_STEP_SIZE])
            elif keypress == ord(defaults["invert_colour"]):
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


class Microscope:
    """Class to combine camera and stage into a single usable class. The
    microscope may be slow to be created; it must wait for the camera,
    stage and the datafile to be ready for use."""

    _UM_PER_PIXEL = defaults["microns_per_pixel"]
    CAMERA_TO_STAGE_MATRIX = np.array(defaults["camera_stage_transform"])

    def __init__(self, width=defaults["resolution"][0],
                 height=defaults["resolution"][1], cv2camera=False,
                 channel=defaults["channel"],
                 filename=defaults["filename"], man=True):
        """Create the Microscope object containing a camera, stage and
        datafile. Use this instead of using the Camera and ScopeStage classes!
        :param width: Resolution along x.
        :param height: " along y.
        :param cv2camera: Set to True if cv2-type camera will be used.
        :param channel: Channel of I2C bus to connect to motors for the stage.
        :param filename: The name of the data file.
        :param man: Whether to control pre-processing manually or not."""

        self.camera = Camera(width, height, manual=man, cv2camera=cv2camera)
        self.stage = ScopeStage(channel)
        # self.light = twoLED.Lightboard()

        h.make_dirs(filename)   # Create parent directories if needed.
        self.datafile = data_io.Datafile(filename)

    def __del__(self):
        del self.camera
        del self.stage
        # del self.light
        del self.datafile

    def camera_centre_move(self, template, mode=defaults["mode"]):
        """Code to return the movement in pixels needed to centre a template
        image, as well as the actual camera position of the template."""
        if mode == 'bayer':
            width, height = self.camera.FULL_RPI_WIDTH, \
                            self.camera.FULL_RPI_HEIGHT
        else:
            width, height = self.camera.resolution

        template_pos = self.camera.find_template(template, box_d=-1,
                                                 decimal=True)
        # The camera needs to move (-delta_x, -delta_y); given (0,0) is top
        # left, not centre as needed.
        camera_move = (-(template_pos[0] - (width/2.0)),
                       -(template_pos[1] - (height/2.0)))
        assert ((camera_move[0] >= -(width/2.0)) and
                (camera_move[0] <= (width/2.0)))
        assert ((camera_move[1] >= -(height/2.0)) and
                (camera_move[1] <= (height/2.0)))
        return camera_move, template_pos

    def camera_move_distance(self, camera_move):
        """Code to convert an (x,y) displacement in pixels to a distance in
        microns."""
        camera_move = np.array(camera_move)
        assert camera_move.shape == (2,)
        return np.power(np.sum(np.power(camera_move, 2.0)), 0.5) * \
            self._UM_PER_PIXEL

    def centre_on_template(self, template, tolerance=1, max_iterations=10):
        """Given a template image, move the stage until the template is
        centred. Returns a tuple containing the number of iterations, the
        camera positions and the stage moves as (number, camera_positions,
        stage_moves), where number is returned as -1 * max_iterations if failed
        to converge.
        - If a tolerance is specified, keep iterating until the template is
        within this distance from the centre or the maximum number of
        iterations is exceeded.
        - The max_iterations is how many times the code will run to attempt to
        centre the template image to within tolerance before aborting.
        - The stage will be held in position after motion, unless release is
        set to True.
        - A return value for iteration less than zero denotes failure, with the
        absolute value denoting the maximum number of iterations.
       - If centre_on_template(...)[0] < 0 then failure."""

        stage_move = np.array([0, 0, 0])
        stage_moves = []
        camera_move, position = self.camera_centre_move(template)
        camera_positions = [position]
        iteration = 0
        while ((self.camera_move_distance(camera_move)) > tolerance) and \
                (iteration < max_iterations):
            iteration += 1
            stage_move = np.dot(camera_move, self.CAMERA_TO_STAGE_MATRIX)   # Rotate to stage coords
            stage_move = np.append(stage_move, [0], axis=1)                 # Append the z-component of zero
            stage_move = np.trunc(stage_move).astype(int)                   # Need integer microsteps (round to zero)
            self.stage.move_rel(stage_move)
            stage_moves.append(stage_move)
            time.sleep(0.5)
            camera_move, position = self.camera_centre_move(template)
            camera_positions.append(position)
        if iteration == max_iterations:
            print "Abort: Tolerance not reached in %d iterations" % iteration
            iteration *= -1
        return iteration, np.array(camera_positions), np.array(stage_moves)

    def calibrate(self, template=None, d=1000):
        """Calibrate the stage-camera coordinates by finding the transformation
        between them.
        - If a template is specified, it will be used as the calibration track
        which is searched for in each image. The central half of the image will
        be used if one is not specified.
        - The size of the calibration square can be adjusted using d,
        in microsteps. Care should be taken that the template or central part
        of the image does not leave the field of view!"""

        # Set up the necessary variables:
        self.camera.preview()
        gr = self.datafile.new_group('templates')
        pos = [np.array([d, d, 0]), np.array([d, -d, 0]),
               np.array([-d, -d, 0]), np.array([-d, d, 0])]
        camera_displacement = []
        stage_displacement = []

        # Move to centre (scope_stage takes account of backlash already).
        self.stage.move_to_pos([0, 0, 0])
        if template is None:
            # Default capture mode is bayer.
            template = self.camera.get_frame(mode='compressed')
            # Crop the central 1/2 of the image - can replace by my central
            # crop function or the general crop function (to be written).
            frac = 0.8
            template = proc.crop_array(template, mmts='frac', dims=frac)
            self.datafile.add_data(template, gr, 'template', attrs={
                'crop_frac': frac})
        time.sleep(1)

        # Store the initial configuration:
        init_cam_pos = np.array(self.camera.find_template(template, box_d=-1,
                                                          decimal=True))
        init_stage_vector = self.stage.position  # 3 component form
        init_stage_pos = init_stage_vector[0:2]  # xy part
        time.sleep(1)

        # Now make the motions in square specified by pos
        for p in pos:
            # Relate the microstep motion to the pixels measured on the
            # camera. To do it with millimetres, you need to relate the
            # pixels to the graticule image. Use ImageJ to make manual
            # measurements.
            self.stage.move_to_pos(np.add(init_stage_vector, p))
            time.sleep(1)
            print "sleeping"
            cam_pos = np.array(self.camera.find_template(template, box_d=-1,
                                                         decimal=True))
            cam_pos = np.subtract(cam_pos, init_cam_pos)
            print "subtraction", cam_pos
            stage_pos = np.subtract(self.stage.position[0:2],
                                    init_stage_pos)
            print "stage_pos", stage_pos
            camera_displacement.append(cam_pos)
            stage_displacement.append(stage_pos)
        self.stage.centre_stage()

        # Do the required analysis:
        camera_displacement = np.array(camera_displacement)
        camera_displacement -= np.mean(camera_displacement, axis=0)
        print camera_displacement
        stage_displacement = np.array(stage_displacement)
        print stage_displacement

        a, res, rank, s = np.linalg.lstsq(camera_displacement,
                                          stage_displacement)
        print "residuals:  ", res
        print "norm:  ", np.linalg.norm(a)
        self.camera.preview()
        self.CAMERA_TO_STAGE_MATRIX = a
        return a


class Camera:

    (FULL_RPI_WIDTH, FULL_RPI_HEIGHT) = defaults["max_resolution"]

    def __init__(self, width=defaults["resolution"][0],
                 height=defaults["resolution"][1], cv2camera=False,
                 manual=False):
        """An abstracted camera class. Use through the Microscope class
        wherever possible.
           - Optionally specify an image width and height.
           - Choosing cv2camera=True allows testing on non RPi systems,
             though code will detect if picamera is not present and assume
             that cv2 must be used instead.
           - Set greyscale to False if measurements are to be done with a
           full colour image.
           - manual specifies whether pre-processing (ISO, white balance,
           exposure) are to be manually controlled or not."""

        if "picamera" not in sys.modules:  # If cannot use picamera, force cv2
            cv2camera = True
        self._usecv2 = cv2camera
        self._view = False
        self._camera = None
        self._stream = None
        self.latest_frame = None

        # Check the resolution is valid.

        if 0 < width <= self.FULL_RPI_WIDTH and 0 < height <= \
                self.FULL_RPI_HEIGHT:
            # Note this is irrelevant for bayer images which always capture
            # at full resolution.
            self.resolution = (width, height)
        elif (width <= 0 or height <= 0) and not cv2camera:
            # Negative dimensions - use full sensor
            self.resolution = (self.FULL_RPI_WIDTH, self.FULL_RPI_HEIGHT)
        else:
            raise ValueError('Camera resolution has incorrect dimensions.')

        if self._usecv2:
            self._camera = cv2.VideoCapture(0)
            self._camera.set(3, width)  # Set width
            self._camera.set(4, height)  # Set height
        else:
            self._camera = picamera.PiCamera()
            self._rgb_stream = picamera.array.PiRGBArray(self._camera)
            self._bayer_stream = picamera.array.PiBayerArray(self._camera)
            self._camera.resolution = (width, height)
            self._fast_capture_iterator = None

        if manual and not cv2camera:
            # This only works with RPi camera.
            self._make_manual()

    def _close(self):
        """Closes the camera devices correctly. Called on deletion, do not call
         explicitly."""
        del self.latest_frame

        if self._usecv2:
            self._camera.release()
        else:
            if self._fast_capture_iterator is not None:
                del self._fast_capture_iterator
            self._camera.close()
            self._rgb_stream.close()

    def __del__(self):
        self._close()

    def _cv2_frame(self, greyscale):
        """Uses the cv2 VideoCapture method to obtain an image. Use get_frame()
        to access."""
        if not self._usecv2:
            raise TypeError("_cv2_frame() should ONLY be used when camera is "
                            "cv2.VideoCapture(0)")
        # We seem to be one frame behind always. So simply get current frame by
        # updating twice .
        frame = self._camera.read()[1]
        frame = self._camera.read()[1]
        if greyscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame

    def _jpeg_frame(self, greyscale, videoport):
        """Captures via a jpeg, code may be adapted to save jpeg. Use
        get_frame() to access."""

        if self._fast_capture_iterator is not None:
            raise Warning("_jpeg_frame cannot be used while use_iterator(True)"
                          " is set.")

        stream = io.BytesIO()
        stream.seek(0)
        self._camera.capture(stream, format='jpeg', use_video_port=videoport)
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(data, 1)
        return image_proc.make_greyscale(frame, greyscale)

    def _bayer_frame(self, greyscale):
        """Capture a raw bayer image, de-mosaic it and output a BGR numpy
        array."""
        # Normally bayer images are not processed via white balance, etc in
        # the camera and are thus of much worse quality if this were done.
        # But the combination of lenses in the Pi means that the reverse is
        # true.
        self._camera.capture(self._bayer_stream, 'jpeg', bayer=True)
        frame = (self._bayer_stream.demosaic() >> 2).astype(np.uint8)
        return image_proc.make_greyscale(frame, greyscale)

    def _bgr_frame(self, greyscale, videoport):
        """Captures straight to a BGR array object; a raw format. Use
        get_frame() to access."""

        if self._fast_capture_iterator is not None:
            raise Warning("_bgr_frame cannot be used while use_iterator(True) "
                          "is set.")
        self._rgb_stream.seek(0)
        self._camera.capture(self._rgb_stream, 'bgr', use_video_port=videoport)
        frame = self._rgb_stream.array
        return image_proc.make_greyscale(frame, greyscale)

    def _fast_frame(self, greyscale):
        """Captures really fast with the iterator method. Must be set up to run
        using use_iterator(True). Use get_frame() to access."""
        if self._fast_capture_iterator is None:
            raise Warning("_fast_frame cannot be used while use_iterator(True)"
                          " is not set.")
        self._rgb_stream.seek(0)
        self._fast_capture_iterator.next()
        frame = self._rgb_stream.array
        return image_proc.make_greyscale(frame, greyscale)

    def _make_manual(self):
        # Set ISO to the desired value
        self._camera.iso = 100
        # Wait for the automatic gain control to settle
        time.sleep(2)
        # Now fix the values
        self._camera.shutter_speed = self._camera.exposure_speed
        self._camera.exposure_mode = 'off'
        g = self._camera.awb_gains
        self._camera.awb_mode = 'off'
        self._camera.awb_gains = g

    def preview(self):
        """If using picamera, turn preview on and off."""
        if not self._usecv2:
            if self._view:
                self._camera.stop_preview()
                self._view = False
            else:
                self._camera.start_preview(fullscreen=False, window=(
                    20, 20, int(640*1.5), int(480*1.5)))
                self._view = True
                time.sleep(2)   # Let the image be properly received.

    def get_frame(self, greyscale=defaults["greyscale"],
                  videoport=defaults["videoport"], mode=defaults["mode"]):
        """Manages obtaining a frame from the camera device.
            - Toggle greyscale to obtain either a grey frame or BGR colour one.
            - Use videoport to select RPi option "use_video_port",
            which speeds up capture of images but has an offset compared to not
            using it.
            - mode allows choosing RPi camera method; via a
            compressed 'compressed', 'bgr' or 'bayer'. 'bgr' is less CPU
            intensive than 'compressed'.
            - If use_iterator(True) has been used to initiate the iterator
            method of capture, this method will be overridden to use that,
            regardless of jpg/array choice."""

        if self._usecv2:
            frame = self._cv2_frame(greyscale)
        elif self._fast_capture_iterator is not None:
            frame = self._fast_frame(greyscale)
        elif mode == 'compressed':
            frame = self._jpeg_frame(greyscale, videoport)
        elif mode == 'bgr':
            frame = self._bgr_frame(greyscale, videoport)
        elif mode == 'bayer':
            frame = self._bayer_frame(greyscale)
        else:
            raise ValueError('The parameter \'mode\' has an invalid value: '
                             '{}.'.format(mode))
        self.latest_frame = frame
        return frame

    def use_iterator(self, iterator=defaults["iterator"]):
        """For the RPi camera only, use the capture_continuous iterator to
        capture frames many times faster.
           - Call this function with iterator=True to turn on the method, and
           use get_frame() as usual. To turn off the iterator and allow
           capture via jpeg/raw then call with iterator=False."""
        if self._usecv2:
            return
        if iterator:
            if self._fast_capture_iterator is None:
                self._fast_capture_iterator = self._camera.capture_continuous(
                    self._rgb_stream, 'bgr', use_video_port=True)
        else:
            self._fast_capture_iterator = None

    def set_roi(self, (x, y, w, h)=(0, 0, -1, -1), normed=False):
        """For the RPi camera only, set the Region of Interest on the sensor
        itself.
            - The tuple should be (x,y,w,gen) so x,y position then width and
            height in pixels. Setting w,gen negative will use maximum size.
            Reset by calling as set_roi().
            - Take great care: changing this will change the camera coordinate
            system, since the zoomed in region will be treated as the whole
            image afterwards.
            - Will NOT behave as expected if applied when already zoomed!
            - Set normed to True to adjust raw normalise coordinates."""
        if self._usecv2:
            pass
        else:
            # TODO Binning and native resolution hard coding
            (frame_w, frame_h) = self.resolution
            if w <= 0:
                w = frame_w
            if h <= 0:
                h = frame_h
            if not normed:
                self._camera.zoom = (x*1.0/frame_w, y*1.0/frame_h,
                                     w*1.0/frame_w, h*1.0/frame_h)
            else:
                self._camera.zoom = (x, y, w, h)

    def find_template(self, template, frame=None, bead_pos=(-1, -1),
                      box_d=-1,
                      centre_mass=True, cross_corr=True, tolerance=0.05,
                      decimal=False):
        """Finds a dot given a camera and a template image. Returns a camera
        coordinate for the centre of where the template has matched a part
        of the image.
        - Default behaviour is to search a 100x100px box at the centre of the
        image.
        - Providing a frame as an argument will allow searching of an existing
        image, which avoids taking a frame from the camera.
        - Specifying a bead_pos will centre the search on that location; use
        when a previous location is known and bead has not moved. The camera
        co-ordinate system should be used. Default is (-1,-1) which actually
        looks at the centre.
        - Specifying box_d allows the dimensions of the search box to be
        altered. A negative or zero value will search the whole image. box_d
        ought to be larger that the template dimensions.
        - Toggle centremass to use Centre of Mass searching (default: True) or
        Maximum Value (False).
        - Use either Cross Correlation (cross_corr=True, the default) or
        Square Difference (False) to find the likely position of the template.
        - Fraction is the tolerance in the thresholding when filtering.
        - decimal determines whether a float or int is returned."""

        # If the template is a colour image (3 channels), make greyscale.
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if frame is None:
            # The default mode is a bayer image.
            frame = self.get_frame(greyscale=True, mode='compressed')
        else:
            frame = frame.copy()

        # These offsets are needed to find position in uncropped image.
        frame_x_off, frame_y_off = 0, 0
        temp_w, temp_h = template.shape[::-1]
        if box_d > 0:  # Only crop if box_d is positive
            if bead_pos == (-1, -1):  # Search the centre if default:
                frame_w, frame_h = frame.shape[::-1]
                frame_x_off, frame_y_off = int(frame_w / 2 - box_d / 2), int(
                    frame_h / 2 - box_d / 2)
            else:  # Otherwise search centred on bead_pos
                frame_x_off, frame_y_off = int(bead_pos[0] - box_d / 2), \
                                           int(bead_pos[1] - box_d / 2)
            frame = frame[frame_y_off: frame_y_off + box_d,
                          frame_x_off: frame_x_off + box_d]

        # Check the size of the frame is bigger than the template to avoid
        # OpenCV Error:
        frame_w, frame_h = frame.shape[::-1]
        if (frame_w < temp_w) or (frame_h < temp_h):
            raise RuntimeError("Template larger than Frame dimensions! %dx%d "
                               "> %dx%d" % (temp_w, temp_h, frame_w, frame_h))

        # If all good, then do the actual correlation:
        # Use either Cross Correlation or Square Difference to match
        if cross_corr:
            corr = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED)
        else:
            corr = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF_NORMED)
            corr *= -1.0  # Actually want minima with this method so reverse
            # values.
        corr += (corr.max()-corr.min()) * tolerance - corr.max()

        """Use this section of the code for brightness centering."""
        corr = cv2.threshold(corr, 0, 0, cv2.THRESH_TOZERO)[1]
        if centre_mass:  # Either centre of mass:
            # Get co-ordinates of peak from origin at top left of array.
            peak = ndimage.measurements.center_of_mass(corr)
            """Ends here."""
            # Array indexing means peak has (y,x) not (x,y):
            centre = (peak[1] + temp_w/2.0, peak[0] + temp_h/2.0)
        else:  # or crudely using max pixel
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)
            centre = (max_loc[0] + temp_w/2.0, max_loc[1] + temp_h/2.0)
        centre = (centre[0]+frame_x_off, frame_y_off+centre[1])

        # To see what the correlations look like.
        corr *= 255/corr.max()
        #cv2.imwrite("corr_%f.jpg" % time.time(), corr)
        if not decimal:
            centre = (int(centre[0]), int(centre[1]))

        print "tem plate found at {}".format(centre)
        return centre


class ScopeStage:
    # Check these bounds.
    _XYZ_BOUND = np.array(defaults["xyz_bound"])
    _MICROSTEPS = defaults["microsteps"]  # How many micro-steps per step.

    def __init__(self, channel=defaults["channel"]):
        """Class representing a 3-axis microscope stage."""
        self.bus = smbus.SMBus(channel)
        time.sleep(3)
        self.position = np.array([0, 0, 0])

    def move_rel(self, vector, back=defaults["backlash"],
                 override=defaults["override"]):
        """Move the stage by (x,y,z) micro steps.
        :param vector: The increment to move by along [x, y, z].
        :param back: An array of the backlash along [x, y, z]. If None,
        it is set to the default value of [128, 128, 128].
        :param override: Set to True to ignore the limits set by _XYZ_BOUND,
        otherwise an error is raised when the bounds are exceeded."""

        # Check back has correct format.
        assert np.all(back >= 0), "Backlash must >= 0 for all [x, y, z]."
        back = h.verify_vector(back)

        r = h.verify_vector(vector)

        # Generate the list of movements to make. If back is [0, 0, 0],
        # there is nly one motion to make.
        movements = []
        if np.any(back != np.zeros(3)):
            # Subtract an extra amount where vector is negative.
            r[r < 0] -= back[np.where(r < 0)]
            r2 = np.zeros(3)
            r2[r < 0] = back[np.where(r < 0)]
            movements.append(r2)
        movements.insert(0, r)

        for movement in movements:
            new_pos = np.add(self.position, movement)
            # If all elements of the new position vector are inside bounds (OR
            # overridden):
            if np.all(np.less_equal(
                    np.absolute(new_pos), self._XYZ_BOUND)) or override:
                _move_motors(self.bus, *movement)
                self.position = new_pos
            else:
                raise ValueError('New position is outside allowed range.')

    def move_to_pos(self, final, over=defaults["override"]):
        new_position = h.verify_vector(final)
        rel_mov = np.subtract(new_position, self.position)
        return self.move_rel(rel_mov, override=over)

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


def _move_motors(bus, x, y, z):
    """Move the motors for the connected module (addresses hardcoded) by a
    certain number of steps.
    :param bus: The smbus.SMBus object connected to appropriate i2c channel.
    :param x: Move x-direction-controlling motor by specified number of steps.
    :param y: "
    :param z: "."""
    [x, y, z] = [int(x), int(y), int(z)]
    print "Moving by [{}, {}, {}]".format(x, y, z)

    # The arguments for write_byte_data are: the I2C address of each motor,
    # the register and how much to move it by.
    # Currently hardcoded in, consider looking this up upon program run.
    bus.write_byte_data(0x50, x >> 8, x & 255)
    bus.write_byte_data(0x58, y >> 8, y & 255)
    bus.write_byte_data(0x6a, z >> 8, z & 255)

    # Empirical formula for how micro step values relate to rotational speed.
    # This is only valid for the specific set of motors tested.
    time.sleep(np.ceil(max([abs(x), abs(y), abs(z)]))/1000 + 2)


if __name__ == '__main__':
    scope = ScopeGUI()
    scope.run_gui()
