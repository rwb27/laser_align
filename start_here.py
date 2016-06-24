#!/usr/bin/env python
"""start_here.py

Contains command line parser for controlling the microscope translation stage
and camera. This script MUST be run in the terminal.

Usage:
    start_here.py <x> <y> <z> [<module_no>]
"""

from docopt import docopt
from scope_stage import *

if __name__ == '__main__':
    args = docopt(__doc__)
    stage = ScopeStage()
    stage.focus_rel(-5000)
    print stage.position
    stage.focus_rel(5000)
    print stage.position

