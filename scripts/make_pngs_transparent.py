#!/usr/bin/env python
# encoding: utf-8
"""
make all images in a folder transparent using imagemagick

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import glob
import subprocess as sp

for fname in glob.glob(os.path.join(sys.argv[1], '*.png')):
    command = ['convert', fname, '-transparent', 'white', fname]
    print command
    sp.call(command)
