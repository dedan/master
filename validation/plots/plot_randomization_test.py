#!/usr/bin/env python
# encoding: utf-8
"""
plot the results of a randomization test

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
import glob
import numpy as np
import pylab as plt
from master.libs import read_data_lib as rdl
from master.libs import utils

method = 'forest'
inpath = '/Users/dedan/projects/master/results/validation/'

if not os.path.exists(os.path.join(inpath, 'collected_res.json')):

    res = {d: {} for d in os.listdir(inpath) if os.path.isdir(os.path.join(inpath, d))}
    for subdir in res:
        true_res = json.load(open(os.path.join(inpath, subdir, 'true.json')))
        for glom, value in true_res.items():
            if not glom in res:
                res[subdir][glom] = {"rand_res_dist": []}
            res[subdir][glom]['true_res'] = value
        rand_res_files = glob.glob(os.path.join(inpath, subdir, 'run_*.json'))
        for rand_res_file in rand_res_files:
            rand_res = json.load(open(rand_res_file))
            for glom, value in rand_res.items():
                res[subdir][glom]['rand_res_dist'].append(value)
        # compute statistics
        for r in res[subdir].values():
            sum_larger_true = np.sum(np.array(r['rand_res_dist']) > r['true_res'])
            n_repetitions = float(len(r['rand_res_dist']))
            r['p'] = (1 + sum_larger_true) / n_repetitions
    json.dump(res, open(os.path.join(inpath, 'collected_res.json'), 'w'))
else:
    res = json.load(open(os.path.join(inpath, 'collected_res.json')))



fig = plt.figure()
ax = fig.add_subplot(111)
for i, (subdir, tmp_res) in enumerate(res.items()):
    plt.plot([r['true_res'] for r in tmp_res.values() if r['true_res'] > 0],
             [r['p'] for r in tmp_res.values() if r['true_res'] > 0],
             'ko')
    ax.set_title(subdir)
fig.savefig(os.path.join(inpath, 'rand_test_overview.png'))
if utils.run_from_ipython():
    plt.show()

