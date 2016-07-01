"""
This is the control script for repeatability experiments

It moves the stage in a random direction for a given distance, then
moves it back again.

It was written by (c) Darryl Foo, James Sharkey, 2015

It was used for the paper in Review of Scientific Instruments titled:
A one-piece 3D printed flexure translation stage for open-source microscopy 

This script is released under the GNU General Public License v3.0."""
import numpy as np
import microscope
import random
import time
import cv2

m = microscope.MicroscopeGUI(filename="repeatability_backlash_corrected.hdf5") # create a microscope object
m.camera.preview() # start the camera preview window

# we correct for backlash by moving away and back again, such that we are
# always approaching from the same direction.
def backlash_corr(correction_dist):
    m.stage.move_rel([correction_dist,correction_dist,0], release=False)
    time.sleep(1)
    m.stage.move_rel([-correction_dist,-correction_dist,0], release=False)

corr_dist = 500 # 500 usteps is 500/16 steps, or 500/16/200 = 5/32 of one revolution, about 40um
m.stage._reset_pos()
backlash_corr(corr_dist) # ensure we're backlash-corrected to start with.
m.stage._reset_pos() # set the stage's position counter to zero

# Get a template of the central part of the image:
template = m.camera.get_frame()
w, h = template.shape
template = template[w/3:2*w/3, h/3:2*h/3]

def repeatability(move_dist, repeats=10, samples=10):
    """Take a repeatability measurement.

    We move a distance `move_dist` in a random direction, then move
    back again, and measure the error between the two.  All positions
    are saved in the HDF5 file.
    """
    global template

    results = np.zeros((repeats*samples, 3))
    moves = []
    end_cam_pos = []
    for i in range(repeats):
        init_cam_pos = []
        for j in range(samples): # measure the initial position
            init_cam_pos.append(m.camera.find_template(template, box_d=-1, decimal=True))
        # move in a random direction
        move_vect = random_point(move_dist)
        moves.append(move_vect)
        m.stage.move_rel(move_vect, release=False)
        time.sleep(1)
        # correct for backlash
        backlash_corr(corr_dist)
        time.sleep(1)
        # move back again
        m.stage.move_rel(np.negative(move_vect), release=False)
        time.sleep(1)
        # correct for backlash
        backlash_corr(corr_dist)
        for j in range(samples): # measure the final position, several times
            end_cam_pos = (m.camera.find_template(template, box_d=-1, decimal=True))
            results[i*samples+j, 0] = i
            results[i*samples+j, 1] = end_cam_pos[0] - init_cam_pos[j][0]
            results[i*samples+j, 2] = end_cam_pos[1] - init_cam_pos[j][1]
        time.sleep(5)
    m.stage.release()
    return (results, moves)

def random_point(move_dist):
    """Generate a random displacement of a given length"""
    angle = random.randrange(0, 360) * np.pi / 180
    vector = np.array([move_dist*np.cos(angle), move_dist*np.sin(angle), 0])
    vector = np.rint(vector)
    return vector

# the global template was used to re-centre.  Now we just re-take the template image.
global_template = m.camera.get_frame()
w, h = global_template.shape
global_template = global_template[2*w/5:3*w/5, 2*h/5:3*h/5]

# make a group in the HDF5 file to store the data
group = m.datafile.new_group("repeatability_backlash_corrected", "Repeatability_Backlash_Corrected")
# run the experiment for a number of different distances, repeating 50 times for each.
for dist in [7,12,21,36,63,109,189,327,567,982,1701,2946,5103,8839,15309]:
    results, moves = repeatability(dist, repeats=50)
    m.datafile.add_data(results, group, "camera_pos", "Camera coords after movement of %d microsteps in random direction." % dist)
    m.datafile.add_data(moves, group, "stage_moves", "Stage movements before each camera coord recorded")
    time.sleep(5)   
    # Get a template of the central part of the image (this ensures we're always starting at zero)
    template = m.camera.get_frame()
    w, h = template.shape
    template = template[w/3:2*w/3, h/3:2*h/3]

# turn off the light and the camera
m.camera.preview()
m.light._close()
