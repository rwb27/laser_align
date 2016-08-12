#!/usr/bin/env python

"""data_output.py
This script contains classes and function to ease data output from the
microscope. Adapted from the script by James Sharkey, 2015, which was used for
the paper in Review of Scientific Instruments titled: A one-piece 3D printed
flexure translation stage for open-source microscopy."""

import yaml

# Edit the paths of the config files. DO NOT CHANGE ANYTHING ELSE DURING USE!
SCOPE_CONFIG_PATH = './configs/config.json'
FOCUS_CONFIG_PATH = './configs/autofocus.json'
TILED_CONFIG_PATH = './configs/tiled_image.json'
ALIGN_CONFIG_PATH = './configs/align.json'


def yaml_read(yaml_file):
    """Parses a YAML file and returns the resulting dictionary.
    :param yaml_file: the YAML file path, ending in .yaml."""
    with open(yaml_file, 'r') as f:
        config_dict = yaml.load(f)
    return config_dict


def make_dict(config_path, **kwargs):
    """Given a path to configuration dictionary from a JSON file, read the
    dictionary and replace any optionally specified named parameters by those
    from kwargs. Returns the modified dictionary."""

    # Read default values from dictionary.
    config_dict = yaml_read(config_path)

    for key in kwargs:
        assert key in config_dict.keys(), "The key {} is not present in the " \
                                      "file {}. Ensure it has been typed " \
                                      "correctly.".format(key, config_path)
        config_dict[key] = kwargs[key]

    return config_dict
