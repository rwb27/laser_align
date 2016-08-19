#!/usr/bin/env python

"""caller.py
Main front-end of script, to call experiments and GUI to run, and reads 
config files.

Usage:
    caller.py align [<configs>...]
    caller.py autofocus [<configs>...]
    caller.py tiled [<configs>...]
    caller.py gui [<configs>...]
    caller.py (-h | --help)

Options:
    -h, --help   Display this usage statement."""

from docopt import docopt
import nplab

import experiments as exp
import gui as g
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
    print configs
    # Calculate brightness of central spot by taking a tiled section of
    # images, cropping the central 55 x 55 pixels, moving to the region of
    # maximum brightness and repeating. Return an array of the positions
    # and the brightness.
    fun_list = None

    for config in configs:
        try:
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
            elif sys_args['gui']:
                gui = g.KeyboardControls(scope, config)
                gui.run_gui()
        except:
            raise ValueError('Invalid config file path {}'.format(config))

    nplab.close_current_datafile()
