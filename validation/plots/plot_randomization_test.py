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

res = {}
method = 'forest'
inpath = '/Users/dedan/projects/master/results/validation/svrlin_haddad'

outpath = os.path.join(inpath, 'plots')
if not os.path.exists(outpath):
    os.mkdir(outpath)

true_res = json.load(open(os.path.join(inpath, 'true.json')))
for glom, value in true_res.items():
    if not glom in res:
        res[glom] = {"rand_res_dist": []}
    res[glom]['true_res'] = value

rand_res_files = glob.glob(os.path.join(inpath, 'run_*.json'))
for rand_res_file in rand_res_files:
    rand_res = json.load(open(rand_res_file))
    for glom, value in rand_res.items():
        res[glom]['rand_res_dist'].append(value)


# compute statistics
for r in res.values():
    r['p'] = np.sum(np.array(r['rand_res_dist']) > r['true_res']) / float(len(r['rand_res_dist']))

# create a plot of the resulting distribution and the original value
fig = plt.figure(figsize=(8,4))
for i, glom in enumerate(res):

    ax = fig.add_subplot(len(res), 1, i+1)
    ax.hist(res[glom]['rand_res_dist'])
    ax.plot([res[glom]['true_res']], [5], 'r*')
    ax.set_ylabel('{} - {:.3f}'.format(glom, res[glom]['p']), rotation='0')
    ax.set_yticks([])
    ax.set_xlim([-0.4, 0.8])
    if not i == len(res) - 1:
        ax.set_xticks([])
    print('{} - p: {:.5f}'.format(glom, res[glom]['p']))
fig.subplots_adjust(hspace=0.3)
fig.savefig(os.path.join(outpath, 'rand_test.png'))
if utils.run_from_ipython():
    plt.show()

