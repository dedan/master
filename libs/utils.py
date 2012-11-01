#!/usr/bin/env python
# encoding: utf-8
"""
    some utilities for plotting and analysis
"""
import numpy as np

def ceiled_root(value):
    """get the next larger integer root for a value

        this is usefull for example if plots should be aranged in a quadratic
        way with shape sqrt(n) x sqrt(n)
    """
    return int(np.ceil(value ** 0.5))

def max_in_values(value_dict):
    """maximum of non NANs in the values of a dict"""
    stacker = lambda x,y: np.hstack((x,y))
    all_values = reduce(stacker, value_dict.values(), np.array([]))
    return np.max(all_values[~np.isnan(all_values)])
