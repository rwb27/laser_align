#!/usr/bin/env python

"""Main front-end of script, to call experiments, run GUI, read config files.

Usage:
    caller.py align [<configs>...] [--output=<output>]
    caller.py autofocus [<configs>...] [--output=<output>]
    caller.py raster [<configs>...] [--output=<output>]
    caller.py raster_3d [<configs>...] [--output=<output>]
    caller.py controller [<configs>...] [--output=<output>]
    caller.py move [--x=<x>] [--y=<y>] [--z=<z>] [--output=<output>]
    caller.py timed [--output=<output>]
    caller.py hill_walk [--output=<output>]
    caller.py measure
    caller.py drift_recentre [--output=<output>]
    caller.py hill_walk2 [--output=<output>] [--initial_gain=<initial_gain>] [--gain_step=<gain_step>] [--max_step=<max_step>] [--init_number=<num>] [--init_delay=<delay>] [--min_step=<min_step>]
    caller.py (-h | --help)



Options:
    -h, --help                      Display this usage statement.
    --output=<output>               The HDF5 file to store data [default: tests.hdf5].
    --initial_gain=<initial_gain>   The initial gain on the photodiode.
    --gain_step=<gain_step>         The increment to reduce the gain by upon each saturation.
"""

from docopt import docopt
import nplab

import experiments as exp
import controller as c
import microscope as micro

# Edit the paths of the config files.
DEFAULT_PATH = '../configs/config.yaml'

if __name__ == '__main__':
    sys_args = docopt(__doc__)
    nplab.datafile.set_current(sys_args['--output'])

    if not sys_args['<configs>']:
        # configs is a list of paths of config files, each of which will be
        # tested in turn.
        configs = [DEFAULT_PATH]
    else:
        configs = sys_args['<configs>']

    adaptive_kwargs = {}
    for custom in ['--initial_gain', '--gain_step', '--max_step',
                   '--init_number', '--init_delay', '--min_step']:
        if sys_args[custom]:
            adaptive_kwargs[custom] = sys_args[custom]

    for config in configs:
        if not sys_args['move']:
            scope = micro.SensorScope(config)
            if sys_args['autofocus']:
                focus = exp.AlongZ(scope, config)
                focus.run()
            elif sys_args['raster']:
                tiled = exp.RasterXY(scope, config)
                tiled.run()
            elif sys_args['align']:
                align = exp.Align(scope, config)
                align.run()
            elif sys_args['controller']:
                gui = c.KeyboardControls(scope, config)
                gui.run_gui()
            elif sys_args['raster_3d']:
                raster3d = exp.RasterXYZ(scope, config)
                raster3d.run()
            elif sys_args['timed']:
                timed = exp.TimedMeasurements(scope, config)
                timed.run()
            elif sys_args['measure']:
                print scope.sensor.average_n(1, 0)
            elif sys_args['hill_walk']:
                hilly = exp.HillWalk(scope, config)
                hilly.run()
            elif sys_args['hill_walk2']:
                hill_walk = exp.AdaptiveHillWalk(scope, config,
                                                 **adaptive_kwargs)
                hill_walk.run()
            elif sys_args['drift_recentre']:
                drift = exp.DriftReCentre(scope, config)
                drift.run()

        elif sys_args['move']:
            positions = []
            for axis in ['--x', '--y', '--z']:
                if sys_args[axis] is None:
                    sys_args[axis] = 0
                positions.append(int(sys_args[axis]))
            print "Moved by {}".format(positions)
            stage = micro.Stage(config)
            stage.move_rel(positions)

    nplab.close_current_datafile()
