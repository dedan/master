#!/usr/bin/env python
# encoding: utf-8
"""
prepend a string to the files described by a glob string

usage:
    prepend.py /path/to/*.json prepend_string

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import glob

def prepend(globstring, prefix):
    """docstring for prepend"""
    for path in glob.glob(globstring):
        location = os.path.dirname(path)
        fname = os.path.basename(path)
        os.rename(path, os.path.join(location, prefix + fname))

if __name__ == '__main__':
    prepend(sys.argv[1], sys.argv[2])
