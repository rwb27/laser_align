#!/usr/bin/env python

"""data_output.py
This script contains classes and function to ease data output from the
microscope. Adapted from the script by James Sharkey, 2015, which was used for
the paper in Review of Scientific Instruments titled: A one-piece 3D printed
flexure translation stage for open-source microscopy."""

import yaml
import pprint

def yaml_read(yaml_file):
    """Parses a YAML file and returns the resulting dictionary.
    :param yaml_file: the YAML file path, ending in .yaml."""
    with open(yaml_file, 'r') as f:
        config_dict = yaml.load(f)
    return config_dict


def make_dict(config, **kwargs):
    """Read the dictionary, or create one from a YAML file, and replace any
    optionally specified named parameters by those from kwargs. Returns the
    modified dictionary.
    :param config: Either a string specifying a path to the config file,
    or the dictionary of config parameters."""

    if type(config) is str:
        # Read default values from dictionary.
        config = yaml_read(config)
    elif type(config) is not dict:
        raise TypeError('Variable \'config\' is neither a string nor a '
                        'dictionary.')
    for key in kwargs:
        # Only existing parameters can be changed.
        assert key in config.keys(), "The key \'{}\' is not present in the " \
                                     "file. Ensure it has been typed " \
                                     "correctly.".format(key)
        config[key] = kwargs[key]

    return config


pprint.pprint(yaml_read('./configs/config.yaml'))