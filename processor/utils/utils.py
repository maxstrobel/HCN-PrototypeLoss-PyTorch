import argparse
import sys
import traceback

import torch


def import_class(import_str):
    """Returns a class from a string including module and class."""
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'Class {class_str} cannot be found ({traceback.format_exception(*sys.exc_info())})')


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f'dict({values})')
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)


def euclidean_dist(x, y):
    """
    Compute pairwise euclidean distance
    :param x: batch of data
    :param y: batch of data
    :return: pairwise euclidean distances
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def comparison_mask(a_labels, b_labels):
    """
    Computes two batches of labels regarding equality
    :param a_labels: labels of first batch
    :param b_labels: labels of second batch
    :return:
    """
    return torch.eq(a_labels.unsqueeze(1), b_labels.unsqueeze(0))
