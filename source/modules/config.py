# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import argparse
import yaml
import re
import os

class AttrDict(dict):
    """ Nested Attribute Dictionary

    A class to convert a nested Dictionary into an object with key-values
    accessible using attribute notation (AttrDict.attribute) in addition to
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse into nested dicts (like: AttrDict.attr.attr)
    """

    def __init__(self, mapping=None):
        super(AttrDict, self).__init__()
        if mapping is not None:
            for key, value in mapping.items():
                self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)
        self.__dict__[key] = value  # for code completion in editors

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
            
        except KeyError:
            raise AttributeError(item)

    __setattr__ = __setitem__


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def execute():
    return parse_args(create_parser())


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--config_file', dest='config_file', type=argparse.FileType(mode='r'))
    parser.add_argument('-d', '--data_list', help="List to use in the run")

    return parser

env_matcher = re.compile(r'\$\{([^}^{]+)\}')
#pylint: disable=unused-argument
def env_constructor(loader, node):
    ''' Extract the matched value, expand env variable, and replace the match '''
    value = node.value
    match = env_matcher.match(value)
    env_var = match.group()[2:-1]

    if os.environ.get(env_var) is None:
        raise Exception("Expected {0} environment variable to be set.".format(env_var))

    return os.environ.get(env_var) + value[match.end():]

def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        loader.add_implicit_resolver('!env', env_matcher, None)
        loader.add_constructor('!env', env_constructor)

        data = AttrDict(yaml.load(args.config_file, Loader=loader))

        if args.data_list is not None:
            data["data_list"] = args.data_list

        data["config"] = args.config_file.name

    return data