#!/usr/bin/env python
# encoding: utf-8
"""
compare normalized and unnormalized response matrices

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import numpy as np
import pylab as plt
from master.libs import read_data_lib as rdl

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')


_, _, rm = rdl.load_response_matrix(os.path.join(data_path, 'response_matrix.csv'))
_, _, urm = rdl.load_response_matrix(os.path.join(data_path, 'unnorm_response_matrix.csv'))

plt.imshow(rm)
plt.axis('off')
plt.savefig('bla1.png')
plt.imshow(urm)
plt.axis('off')
plt.savefig('bla2.png')
