#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import __builtin__
import sys
import os
import pickle, json
import itertools as it
import numpy as np
import pylab as plt
from collections import defaultdict
from master.libs import run_lib
reload(run_lib)


glomerulus = 'Or22a'
active_thresh = 0.5
fpath = '/Users/dedan/projects/master/data/spectral_features/large_base/'

plt.close('all')
features = pickle.load(open(os.path.join(fpath, 'parsed.pckl')))

# only for molecules we always use and for a single glomerulus
config = {'data_path': os.path.join(os.path.dirname(__file__), '..', 'data'),
          'glomerulus': glomerulus}
data, targets, molids = run_lib.load_data_targets(config, features)
freqs = {k: v['freq'] for k, v in features.items() if k in molids}

# histogram of frequencies
fig = plt.figure()
all_freqs = list(it.chain(*freqs.values()))
ax = fig.add_subplot(111)
ax.hist(all_freqs, 4000, range=[0, 4000])

# target value histogram
fig = plt.figure()
ax = fig.add_subplot(211)
ax.hist(targets)

# frequency distribution of active targets
active = [m for i, m in enumerate(molids) if targets[i] > active_thresh]
act_freqs = list(it.chain(*[v for k, v in freqs.items() if k in active]))
ax = fig.add_subplot(212)
# do ligand share bins for resolution of 0.1?
ax.hist(act_freqs, 8000, range=[0, 4000])


plt.show()
