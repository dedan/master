#!/usr/bin/env python
# encoding: utf-8
"""

reproduce the results of Haddad 2009

to run this file you need to extract the Hallem dataset from the DoOR database
R module. This can be done by the script:

    scripts/extract_door2csv.py

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
import numpy as np
import pylab as plt
from master.libs import run_lib
from master.libs import read_data_lib as rdl
from master.libs import utils
from sklearn.cross_validation import KFold
from scipy.spatial.distance import pdist
from scipy import stats
import csv, glob
from collections import defaultdict

inpath = '/Users/dedan/projects/master/data/door_csvs'
hallems = ['Or2a', 'Or7a', 'Or9a', 'Or10a', 'Or19a', 'Or22a', 'Or23a',
           'Or33b', 'Or35a', 'Or43a', 'Or43b', 'Or47a', 'Or47b', 'Or49b',
           'Or59b', 'Or65a', 'Or67a', 'Or67c', 'Or82a', 'Or85a', 'Or85b',
           'Or85f', 'Or88a', 'Or98a']

res = defaultdict(dict)
for glom in hallems:
    dreader = csv.DictReader(open(os.path.join(inpath, glom + '.csv')))
    for entry in dreader:
        if not entry['CAS'] == 'SFR':
            res[entry['CAS']][glom] = entry['Hallem.2006.EN']

cases = res.keys()
rm = np.array([[int(c[g]) if not c[g] == 'NA' else np.nan for g in hallems] for c in res.values()])
all_avail = np.sum(np.isnan(rm), axis=1) == 0
rm = rm[all_avail]
cases = [cases[i] for i in range(len(cases)) if all_avail[i]]
assert len(cases) == rm.shape[0]

# haddad dismissed 63 odors which did not have a large variance over glomeruli
to_take = np.std(rm, axis=1) > 46
rm = rm[to_take]
cases = [cases[i] for i in range(len(cases)) if to_take[i]]
assert len(cases) == rm.shape[0]

config = {
    'data_path': '/Users/dedan/projects/master/data/',
    'features': {
        'type': 'spectral',
        'descriptor': 'haddad_desc',
        'spec_type': 'ir',
        'use_intensity': False,
        'kernel_width': 75,
        'bin_width': 110,
        'properties_to_add': [],
        'normalize': True
    }
}


features = run_lib.prepare_features(config)
door2id = json.load(open('data/door2id.json'))
available = [True if door2id[c][0] in features else False for c in cases]
cases = [cases[i] for i in range(len(available)) if available[i]]
rm = rm[np.array(available)]
rm[np.isnan(rm)] = 0

eva_space = np.array([features[door2id[c][0]] for c in cases])
assert eva_space.shape[0] == rm.shape[0]

def greedy_selection(space, measure):
    """implement greedy feature selection"""

    all_idx = range(space.shape[1]-1)
    chosen, res = [], []
    for i in all_idx:

        to_try = set(all_idx) - set(chosen)
        tmp = [(chosen + [tt], measure(space[:, chosen + [tt]])) for tt in to_try]
        sorted_tmp = sorted(tmp, key=lambda t: t[1], reverse=True)
        chosen = sorted_tmp[0][0]
        res.append(sorted_tmp[0])
    return res, chosen


val_res = []
chosens = []
for _ in range(100):
    kf = KFold(rm.shape[0], 5, indices=False, shuffle=True)
    for train, test in kf:
        measure = lambda x: stats.pearsonr(pdist(rm[train], 'correlation'), pdist(x))[0]
        greedy_res, chosen = greedy_selection(eva_space[train], measure)
        sorted_res = sorted(greedy_res, key=lambda t: t[1], reverse=True)
        best_chosen = sorted_res[0][0]
        chosens.append(best_chosen)
        perf = stats.pearsonr(pdist(rm[test], 'correlation'), pdist(eva_space[np.ix_(test, best_chosen)]))[0]
        val_res.append(perf)


fig = plt.figure()
ax = fig.add_subplot(211)
ax.hist(val_res, color='0.5')
ax.set_xlabel('histogram over r, mean: {}'.format(np.mean(val_res)))

ax = fig.add_subplot(212)
ax.hist(utils.flatten(chosens), bins=range(eva_space.shape[1]+1), color='0.5')
ax.set_xlabel('dimension selection histogram')
fig.subplots_adjust(hspace=0.2)


