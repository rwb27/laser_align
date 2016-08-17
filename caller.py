#!/usr/bin/env python

"""caller.py
Main front-end of script, to call experiments and GUI to run, and reads 
config files.

Usage:
    caller.py align
    caller.py autofocus
    caller.py tiled
    caller.py gui
    caller.py (-h | --help)

Options:
    -h, --help   Display this usage statement."""

from docopt import docopt
import nplab

import experiments as exp
import gui as g
import helpers as h
import image_proc as proc
import measurements as mmts
import microscope as micro

# Edit the paths of the config files.
CONFIG_PATH = './configs/config.yaml'


if __name__ == '__main__':
    sys_args = docopt(__doc__)

    # Calculate brightness of central spot by taking a tiled section of
    # images, cropping the central 55 x 55 pixels, moving to the region of 
    # maximum brightness and repeating. Return an array of the positions and 
    # the brightness.
    fun_list = [h.bake(proc.crop_array, args=['IMAGE_ARR'],
                       kwargs={'mmts': 'pixel', 'dims': 55}),
                h.bake(mmts.brightness, args=['IMAGE_ARR'])]

    scope = micro.SensorScope(CONFIG_PATH)
    if sys_args['autofocus']:
        focus = exp.AutoFocus(scope, CONFIG_PATH)
        focus.run()
    elif sys_args['tiled']:
        tiled = exp.Tiled(scope, CONFIG_PATH)
        tiled.run(func_list=fun_list)
    elif sys_args['align']:
        align = exp.Align(scope, CONFIG_PATH)
        align.run(func_list=fun_list)
    elif sys_args['gui']:
        gui = g.KeyboardControls(scope, CONFIG_PATH)
        gui.run_gui()

    nplab.close_current_datafile()
