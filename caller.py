#!/usr/bin/env python

"""caller.py
Main front-end of script, to call experiments and GUI to run, and reads 
config files.

Usage:
    caller.py align
    caller.py autofocus
    caller.py calibrate
    caller.py centre
    caller.py tiled
    caller.py gui
    caller.py (-h | --help)

Options:
    -h, --help   Display this usage statement."""

from docopt import docopt

import experiments as exp
import gui as g
import helpers as h
import image_proc as proc
import measurements as mmts
import microscope as micro
from data_io import scope_defs, focus_defs, tiled_defs, align_defs

if __name__ == '__main__':
    sys_args = docopt(__doc__)

    # Control pre-processing manually.
    scope = micro.Microscope(man=True)

    # Calculate brightness of central spot by taking a tiled section of
    # images, cropping the central 55 x 55 pixels, moving to the region of 
    # maximum brightness and repeating. Return an array of the positions and 
    # the brightness.
    fun_list = [h.bake(proc.crop_array, args=['IMAGE_ARR'],
                       kwargs={'mmts': 'pixel', 'dims': 55}),
                h.bake(mmts.brightness, args=['IMAGE_ARR'])]

    if sys_args['autofocus']:
        focus = exp.AutoFocus(scope, scope_defs, focus_defs)
        focus.run()
    elif sys_args['centre']:
        exp.centre_spot(scope)
    elif sys_args['calibrate']:
        scope.calibrate()
    elif sys_args['tiled']:
        tiled = exp.Tiled(scope, scope_defs, tiled_defs)
        tiled.run(func_list=fun_list)
    elif sys_args['align']:
        align = exp.Align(scope, scope_defs, tiled_defs, align_defs)
        align.run(func_list=fun_list)
    elif sys_args['gui']:
        gui = g.ScopeGUI()
        gui.run_gui()
