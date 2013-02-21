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
        'kernel_width': 100,
        'bin_width': 150,
        'properties_to_add': [],
        'normalize': True
    }
}

for k_width in range(50, 100, 5):

    config['features']['kernel_width'] = k_width
    config['features']['bin_width'] = int(1.5 * k_width)
    features = run_lib.prepare_features(config)
    door2id = json.load(open('data/door2id.json'))
    available = [True if door2id[c][0] in features else False for c in cases]
    cases = [cases[i] for i in range(len(available)) if available[i]]
    rm = rm[np.array(available)]
    rm[np.isnan(rm)] = 0

    eva_space = np.array([features[door2id[c][0]] for c in cases])
    assert eva_space.shape[0] == rm.shape[0]

    response_dists = pdist(rm, 'correlation')
    nan_cors = np.isnan(response_dists)
    eva_dists = pdist(eva_space)

    response_dists = response_dists[~nan_cors]
    eva_dists = eva_dists[~nan_cors]
    print k_width
    print stats.pearsonr(response_dists, eva_dists)
