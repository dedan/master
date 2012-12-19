#!/usr/bin/env python
# encoding: utf-8
"""
There are a few CAS numbers that we mapped to the same molecule. We have to
check that these CAS numbers have similar target values, otherwise it might
confuse our model during learning

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
from collections import defaultdict
import itertools as it
import numpy as np
import pylab as plt
from master.libs import read_data_lib as rdl
from master.libs import utils

N = 10

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
csv_path = os.path.join(data_path, 'response_matrix.csv')
door2id = json.load(open(os.path.join(data_path, 'door2id.json')))
cas_numbers, glomeruli, rm = rdl.load_response_matrix(csv_path, door2id)

# count which molids have 2 CAS numbers assigned
molid_count = defaultdict(int)
for c in door2id:
    if door2id[c]:
        molid_count[door2id[c][0]] += 1
double_molids = [k for k, v in molid_count.items() if v > 1]
double_cas = {m: [k for k, v in door2id.items() if v and v[0] == m]
              for m in double_molids}
all_double_cas = list(it.chain(*double_cas.values()))

# get mask for non double cas targets
non_double_idx = np.ones(len(cas_numbers), dtype=bool)
non_double_idx[[i for i in range(len(cas_numbers)) if cas_numbers[i] in all_double_cas]] = 0

# pick the N glomeruli with most molecules available
n_avail = np.sum(~np.isnan(rm), axis=0)
N_best = np.argsort(n_avail)[-N:]

# plot distributions
for i in range(N):
    cur_glom = rm[:, N_best[i]]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    non_doubles = cur_glom[non_double_idx]
    t_dists = utils.pdist_1d(non_doubles[~np.isnan(non_doubles)])
    ax.hist(t_dists, 100)

    for d_cas in double_cas.values():
        cas_idx1 = cas_numbers.index(d_cas[0])
        cas_idx2 = cas_numbers.index(d_cas[1])
        t_diff = np.abs(cur_glom[cas_idx1] - cur_glom[cas_idx2])
        ax.plot([t_diff], [100], 'r*')
    ax.set_title(glomeruli[N_best[i]])
plt.show()



