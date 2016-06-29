""" REVISION 10-03-2015 - jps79

This control script communicates with the Raspberry Pi camera module
and also contains workaround code to allow testing with other cameras.

Author: (c) James Sharkey, 2015

It was used for the paper in Review of Scientific Instruments titled:
A one-piece 3D printed flexure translation stage for open-source microscopy 

This script is released under the GNU General Public License v3.0."""

import cv2
import numpy as np
from scipy import ndimage
import io
import sys
# Try and import picamera:
try:
    import picamera
    import picamera.array
except ImportError:
    pass  # Don't fail on error; simply force cv2 camera later


class Camera:

    _FULL_RPI_WIDTH = 2592
    _FULL_RPI_HEIGHT = 1944

    def __init__(self, width=640, height=480, cv2camera=False):
        """An abstracted camera class.
           - Optionally specify an image width and height.
           - Choosing cv2camera=True allows testing on non RPi systems,
             though code will detect if picamera is not present and assume
             that cv2 must be used instead."""

        if "picamera" not in sys.modules:  # If cannot use picamera, force cv2
            cv2camera = True

        self._use_cv2 = cv2camera
        self._view = False
        self._camera = None
        self._stream = None
        self.latest_frame = None
        self.resolution = (width, height)

        if ((width <= 0) or (height <= 0)) and not cv2camera:
            width = self._FULL_RPI_WIDTH  # Negative dimensions use full sensor
            height = self._FULL_RPI_HEIGHT

        if self._use_cv2:
            self._camera = cv2.VideoCapture(0)
            self._camera.set(3, width)   # Set width
            self._camera.set(4, height)  # Set height

        else:
            self._camera = picamera.PiCamera()
            self._stream = picamera.array.PiRGBArray(self._camera)
            self._camera.resolution = (width, height)
            self._fast_capture_iterator = None

    def _close(self):
        """Closes the camera devices correctly. Called on deletion, do not call
         explicitly."""
        del self.latest_frame
        if self._use_cv2:
            self._camera.release()
        else:
            if self._fast_capture_iterator is not None:
                del self._fast_capture_iterator
            self._camera.close()
            self._stream.close()

    def __del__(self):
        self._close()

    def _cv2_frame(self, greyscale):
        """Uses the cv2 VideoCapture method to obtain an image. Use get_frame()
         to access."""

        if not self._use_cv2:
            raise TypeError("_cv2_frame() should ONLY be used when camera is "
                            "cv2.VideoCapture(0)")

        # We always seem to be one frame behind. So simply get current frame by
        # updating twice.
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
        if greyscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _raw_frame(self, greyscale, videoport):
        """Captures straight to an array object; a raw format. Use get_frame()
        to access."""

        if self._fast_capture_iterator is not None:
            raise Warning("_raw_frame cannot be used while use_iterator(True) "
                          "is set.")
        self._stream.seek(0)
        self._camera.capture(self._stream, 'bgr', use_video_port=videoport)
        frame = self._stream.array
        if greyscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _fast_frame(self, greyscale):
        """Captures really fast with the iterator method. Must be set up to run
        using use_iterator(True). Use get_frame() to access."""
        if self._fast_capture_iterator is None:
            raise Warning("_fast_frame cannot be used while use_iterator(True)"
                          " is not set.")
        self._stream.seek(0)
        self._fast_capture_iterator.next()
        frame = self._stream.array
        if greyscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def preview(self):
        """If using picamera, turn preview on and off."""
        if not self._use_cv2:
            if self._view:
                self._camera.stop_preview()
                self._view = False
            else:
                self._camera.start_preview(fullscreen=False, window=(
                    20, 20, int(640*1.5), int(480*1.5)))
                self._view = True

    def get_frame(self, greyscale=True, videoport=True, rawformat=True):
        """Manages obtaining a frame from the camera device.
            - Toggle greyscale to obtain either a grey frame or BGR colour one.
            - Use videoport to select RPi option "use_video_port",
            which speeds up capture of images but has an offset compared to not
            using it.
            - rawformat allows choosing RPi camera method; via a jpeg or
            straight to array. Array is less CPU intensive.
            - If use_iterator(True) has been used to initiate the iterator
            method of capture, this method will be overridden to use that,
            regardless of jpg/array choice."""

        if self._use_cv2:
            frame = self._cv2_frame(greyscale)
        elif self._fast_capture_iterator is not None:
            frame = self._fast_frame(greyscale)
        elif rawformat:
            frame = self._raw_frame(greyscale, videoport)
        else:
            frame = self._jpeg_frame(greyscale, videoport)
        self.latest_frame = frame
        return frame

    def use_iterator(self, iterator):
        """For the RPi camera only, use the capture_continuous iterator to
        capture frames many times faster.
           - Call this function with iterator=True to turn on the method, and
           use get_frame() as usual. To turn off the iterator and allow
           capture via jpeg/raw then call with iterator=False."""

        if self._use_cv2:
            return
        if iterator:
            if self._fast_capture_iterator is None:
                self._fast_capture_iterator = self._camera.capture_continuous(
                    self._stream, 'bgr', use_video_port=True)
        else:
            self._fast_capture_iterator = None

    def set_roi(self, (x, y, w, h)=(0, 0, -1, -1), normed=False):
        """For the RPi camera only, set the Region of Interest on the sensor
        itself.
            - The tuple should be (x,y,w,h) so x,y position then width and
            height in pixels. Setting w,h negative will use maximum size. Reset
            by calling as set_roi().
            - Take great care: changing this will change the camera coordinate
            system, since the zoomed in region will be treated as the whole
            image afterwards.
            - Will NOT behave as expected if applied when already zoomed!
            - Set normed to True to adjust raw normalise coordinates."""

        if self._use_cv2:
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

    def find_template(self, template, frame=None, bead_pos=(-1, -1), box_d=100,
                      centre_mass=True, cross_corr=True, fraction=0.05,
                      decimal=False):
        """Finds a dot given a camera and a template image. Returns a camera
        coordinate.
        - Default behaviour is to search a 100x100px box at the centre of the
        image.
        - Providing a frame as an argument will allow searching of an existing
        image, which avoids taking a frame from the camera.
        - Specifying a bead_pos will centre the search on that location; use
        when a previous location is known and bead has not moved. The camera
        co-ordinate system should be used. Default is (-1,-1) which actually
        looks at the centre.
        - Specifying boxD allows the dimensions of the search box to be
        altered. A negative or zero value will search the whole image. boxD
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
            frame = self.get_frame(greyscale=True, videoport=True,
                                   rawformat=True)
        else:
            frame = frame.copy()

        # These offsets are needed to find position in uncropped image
        frame_x_off, frame_y_off = 0, 0
        temp_w, temp_h = template.shape[::-1]
        if box_d > 0:  # Only crop if boxD is positive
            if bead_pos == (-1, -1):  # Search the centre if default:
                frame_w, frame_h = frame.shape[::-1]
                frame_x_off, frame_y_off = int(
                    frame_w / 2 - box_d / 2), int(frame_h / 2 - box_d / 2)
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
        corr += (corr.max()-corr.min())*fraction - corr.max()
        corr = cv2.threshold(corr, 0, 0, cv2.THRESH_TOZERO)[1]
        if centre_mass:  # Either centre of mass:
            peak = ndimage.measurements.center_of_mass(corr)
            # Array indexing means peak has (y,x) not (x,y):
            centre = (peak[1] + temp_w/2.0, peak[0] + temp_h/2.0)
        else:  # or crudely using max pixel
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)
            centre = (max_loc[0] + temp_w/2.0, max_loc[1] + temp_h/2.0)
        centre = (centre[0]+frame_x_off, frame_y_off+centre[1])
# To see what the correlations look like, uncomment:
#        corr*=255/corr.max()
#        cv2.imwrite("corr_%f.jpg"%time.time(), corr)
        if not decimal:
            centre = (int(centre[0]), int(centre[1]))
        return centre
