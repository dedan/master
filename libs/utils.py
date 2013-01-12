#!/usr/bin/env python
# encoding: utf-8
"""
    some utilities for plotting and analysis
"""
import numpy as np
from collections import defaultdict


class RUDict(dict):
    """code for this class from http://stackoverflow.com/a/8447781/515807"""

    def __init__(self, *args, **kw):
        super(RUDict,self).__init__(*args, **kw)

    def update(self, E=None, **F):
        if E is not None:
            if 'keys' in dir(E) and callable(getattr(E, 'keys')):
                for k in E:
                    if k in self:  # existing ...must recurse into both sides
                        self.r_update(k, E)
                    else: # doesn't currently exist, just update
                        self[k] = E[k]
            else:
                for (k, v) in E:
                    self.r_update(k, {k:v})

        for k in F:
            self.r_update(k, {k:F[k]})

    def r_update(self, key, other_dict):
        if isinstance(self[key], dict) and isinstance(other_dict[key], dict):
            od = RUDict(self[key])
            nd = other_dict[key]
            od.update(nd)
            self[key] = od
        else:
            self[key] = other_dict[key]

def pdist_1d(values):
    """like pdist but for 1-d lists"""
    res = []
    for i in range(len(values)):
        for j in range(i+1, len(values)):
            res.append(np.abs(values[i] - values[j]))
    return res


def recursive_defaultdict():
    """get a defaultdict of defaultdicts of defaultdicts of ..."""
    l=lambda:defaultdict(l)
    return l()

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

def run_from_ipython():
    """check if a script is run in ipython

    this can be used e.g. to show plots only when run in ipython
    """
    try:
        __IPYTHON__
        return True
    except NameError:
        return False