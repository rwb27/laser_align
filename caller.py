#!/usr/bin/env python

"""caller.py
Main front-end of script, to call experiments and GUI to run, and reads 
config files.[

Usage:
    caller.py align [<configs>...]
    caller.py autofocus [<configs>...]
    caller.py tiled [<configs>...]
    caller.py raster_3d [<configs>...]
    caller.py controller [<configs>...]
    caller.py move --x=<x> --y=<y> --z=<z>
    caller.py (-h | --help)

Options:
    -h, --help   Display this usage statement."""

from docopt import docopt
import nplab

import experiments as exp
import controller as c
import microscope as micro

# Edit the paths of the config files.
DEFAULT_PATH = './configs/config.yaml'

if __name__ == '__main__':
    sys_args = docopt(__doc__)

    if not sys_args['<configs>']:
        # configs is a list of paths of config files, each of which will be
        # tested in turn.
        configs = [DEFAULT_PATH]
    else:
        configs = sys_args['<configs>']

    # List of baked functions to do post-processing, if any.
    fun_list = None

    for config in configs:
        if not sys_args['move']:
            scope = micro.SensorScope(config)
            if sys_args['autofocus']:
                focus = exp.AlongZ(scope, config)
                focus.run()
            elif sys_args['tiled']:
                tiled = exp.RasterXY(scope, config)
                tiled.run(func_list=fun_list)
            elif sys_args['align']:
                align = exp.Align(scope, config)
                align.run(func_list=fun_list)
            elif sys_args['controller']:
                gui = c.KeyboardControls(scope, config)
                gui.run_gui()
            elif sys_args['raster_3d']:
                raster3d = exp.RasterXYZ(scope, config)
                raster3d.run(func_list=fun_list)
        elif sys_args['move']:
            positions = []
            [positions.append(int(sys_args[axis])) for axis in ['--x', '--y',
                                                                '--z']]
            print "Moved by {}".format(positions)
            stage = micro.Stage(config)
            stage.move_rel(positions)

    # nplab.close_current_datafile()
