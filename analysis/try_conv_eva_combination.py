#!/usr/bin/env python
# encoding: utf-8
'''
The sensors review paper on the swipe-card theory proposes that odor receptors
mainly react to a combination of molecule shape and vibrational frequencies.
If this is true, shouldn't I get the best performance for a combination of
a eDragon based descriptor with EVA?

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
'''

import os
import copy
import json
from master.libs import run_lib
import numpy as np
import pylab as plt

config = {
    'data_path': os.path.join(os.path.dirname(__file__), '..', 'data'),
    'features': {
        'type': 'conventional',
        'descriptor': 'haddad_desc',
        'normalize': True,
        "properties_to_add": []
    },
    "feature_selection": {
        "method": 'linear'
    },
    "methods": {
        "svr": {
            "C": 1.0,
            "n_folds": 50
        }
    },
    "randomization_test": False
}

used_gloms = json.load(open(os.path.join(config['data_path'], 'used_glomeruli.json')))

alone_haddad, alone_vib, together = [], [], []
for glom in used_gloms:

    config['glomerulus'] = glom

    # prepare haddad features
    features_h = run_lib.prepare_features(config)
    data_h, targets_h, molids_h = run_lib.load_data_targets(config, features_h)
    config['feature_selection']['k_best'] = data_h.shape[1]
    tmp = run_lib.run_runner(config, data_h, targets_h)
    print glom, tmp
    alone_haddad.append(tmp['svr']['gen_score'])

    # prepare vib100
    config_spec = copy.deepcopy(config)
    config_spec['features']['type'] = 'spectral'
    config_spec['features']['kernel_width'] = 100
    config_spec['features']['bin_width'] = 150
    config_spec['features']['use_intensity'] = False
    config_spec['features']['spec_type'] = 'ir'

    features_v = run_lib.prepare_features(config_spec)
    data_v, targets_v, molids_v = run_lib.load_data_targets(config_spec, features_v)
    config['feature_selection']['k_best'] = data_v.shape[1]
    tmp = run_lib.run_runner(config, data_v, targets_v)
    alone_vib.append(tmp['svr']['gen_score'])

    # together
    both_avail = set(molids_v).intersection(molids_h)
    idx_h = [i for i, m in enumerate(molids_h) if m in both_avail]
    idx_v = [i for i, m in enumerate(molids_v) if m in both_avail]
    data_combined = np.hstack((data_h[idx_h], data_v[idx_v]))
    targets_combined = targets_h[idx_h]
    config['feature_selection']['k_best'] = data_combined.shape[1]
    tmp = run_lib.run_runner(config, data_combined, targets_combined)
    together.append(tmp['svr']['gen_score'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(range(0, len(used_gloms)*3, 3), alone_haddad, color='b')
ax.set_xlabel('haddad')

ax.bar(range(1, len(used_gloms)*3+1, 3), alone_vib, color='r')
ax.set_xlabel('vib')

ax.bar(range(2, len(used_gloms)*3+2, 3), together, color='g')
ax.set_xlabel('together')
plt.show()