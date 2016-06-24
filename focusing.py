import picamera
import cv2
import io
import numpy as np
from scope_stage import ScopeStage


# my algorithm will be to 1. move, 2.autofocus 3. capture


def bgr2gray(rgb):
    b, g, r = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def capture_to_BGR_array(camera):
    """Capture a frame from a camera and output to a numpy array"""
    stream = io.BytesIO()
    camera.capture(stream, format='jpeg')  # get an image, see picamera.readthedocs.org/en/latest/recipes1.html
    data = np.fromstring(stream.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, 1)


def autofocus(camera, stage, npoints=10, steps=1000):
    lsdev = []

    stage.focus_rel(-steps * npoints / 2)
    for i in range(npoints):
        image = capture_to_BGR_array(camera)
        contrast = np.std(bgr2gray(image))
        lsdev.append(contrast)
        stage.focus_rel(steps)
    # take one last image after moving
    image = capture_to_BGR_array(camera)
    contrast = np.std(bgr2gray(image))
    lsdev.append(contrast)

    focuspic = np.argmax(lsdev)
    stage.focus_rel(-steps * npoints + steps * focuspic)


if __name__ == "__main__":

    stage = ScopeStage()
    lsdev = []
    steps = 1000
    n = 3

    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.start_preview()
        autofocus(camera, stage, npoints=20, steps=1000)
        time.sleep(2)
        direction = 1
        stage.move_rel(-n / 2 * steps, -n / 2 * steps, 0)
        try:
            for j in range(n):
                for i in range(n):
                    stage.move_rel(steps * direction, 0, 0)
                    time.sleep(0.5)
                    autofocus(camera, stage, npoints=4, steps=500)
                    time.sleep(0.5)
                    camera.capture('{0}_{1}.jpg'.format(j, i))
                stage.move_rel(0, steps, 0)
                direction = -1 * direction
            stage.move_rel(-n / 2 * steps, -n / 2 * steps, 0)
        except KeyboardInterrupt:
            print "Aborted, moving back to start"
            stage.move_rel((n / 2 - i) * steps, (n / 2 - j) * steps, 0)

