#!/usr/bin/env python

"""data_output.py
This script contains classes and function to ease data output from the
microscope. Adapted from the script by James Sharkey, 2015, which was used for
the paper in Review of Scientific Instruments titled: A one-piece 3D printed
flexure translation stage for open-source microscopy."""


import simplejson as json
from jsmin import jsmin


def config_read(json_file):
    """Parses a JSON config file with comments and returns the output
    dictionary.
    :param json_file: The JSON file name, ending in .json."""
    with open(json_file) as js_file:
        no_comments = jsmin(js_file.read(), quote_chars="'\"`")
    return json.loads(no_comments)
