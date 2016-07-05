"""image_mmts.py
Contains functions to perform measurements on a BGR array that has been
appropriately processed."""

import numpy as np
from scipy import ndimage as sn


def sharpness_lap(rgb_image):
    """Calculate sharpness as the Laplacian of the black and white image.
    :param rgb_image: The 3-channel image to calculate sharpness for.
    :return: The mean Laplacian.
    """
    image_bw = np.mean(rgb_image, 2)
    image_lap = sn.filters.laplace(image_bw)    # Look up this filter.
    return np.mean(np.abs(image_lap))


def sharpness_vs_position(pixel_step, list_of_arrs):
    """Calculates the sharpness of a set of sub-images as a function of
    position.
    :param pixel_step: A tuple of no. of pixels per sub-image along (x, y).
    :param list_of_arrs: A list of lists containing the sub-image arrays,
    in the format returned by crop_img_into_n.
    :return: An array in the form [x_position, y_position, sharpness],
    where each of the three are column vectors."""

    sharpness_arr = []
    for arr_list in list_of_arrs:
        sharpness_col = []
        for arr in arr_list:
            sharpness_col.append(sharpness_lap(arr))
        sharpness_arr.append(sharpness_col)

    sharpness_arr = np.array(sharpness_arr)

    it = np.nditer(sharpness_arr, flags=['multi_index'])
    results = []
    while not it.finished:
        results.append([it.multi_index[0] * pixel_step[0] + pixel_step[0]/2,
                        it.multi_index[1] * pixel_step[1] + pixel_step[1]/2,
                        it[0]])
        it.iternext()

    return np.array(results)


def get_res(bgr_arr):
    """Get the x, y and total resolutions of the image in pixels.
    :param bgr_arr: The 3D array to split, with shape in the format (no. of
    row pixels in image, no. of column pixels in image, 3 BGR values). This
    is the format the images captured are in.
    :return: The list [x_pixels, y_pixels, total_square_pixels]."""
    # NOTE: The x and y pixel co-ordinates are swapped around in the bgr
    # array, so need to swap them around when calling the shape.
    return [float(bgr_arr.shape[1]), float(bgr_arr.shape[0]),
            float(np.product(bgr_arr.shape[:2]))]

# Code to test the sharpness vs position plot, buts needs modification.
#if __name__ == "__main__":
#    with picamera.PiCamera() as camera:
#        camera.resolution = (640, 480)
#        camera.start_preview()
#        time.sleep(3)   # Let camera to receive image clearly before
    # capturing.
#        capture_to_bgr_array(camera)
#        camera.stop_preview()   # Preview must be stopped afterwards to
#        # prevent Pi freezing on camera screen.
#
#        split_img_data = crop_img_into_n(capture_to_bgr_array(camera), 4800)
#        plotting_data = sharpness_vs_position(*split_img_data)
#
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        [X, Y, Z] = [plotting_data[:, 0], plotting_data[:, 1],
#                     plotting_data[:, 2]]
#        ax.plot_wireframe(X, Y, Z)
#
#        plt.show()
