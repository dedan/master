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
        "forest": {
            "n_estimators": 500,
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
    features = run_lib.prepare_features(config)
    data_haddad, targets, _ = run_lib.load_data_targets(config, features)
    config['feature_selection']['k_best'] = data_haddad.shape[1]
    tmp = run_lib.run_runner(config, data_haddad, targets, False, False)
    alone_haddad.append(tmp['forest']['gen_score'])

    # prepare vib100
    config['features']['type'] = 'spectral'
    config['features']['kernel_width'] = 50
    config['features']['bin_width'] = 75
    config['features']['use_intensity'] = False
    config['features']['spec_type'] = 'ir'
    data_vib100, _, _ = run_lib.load_data_targets(config, features)
    config['feature_selection']['k_best'] = data_vib100.shape[1]
    tmp = run_lib.run_runner(config, data_vib100, targets, False, False)
    alone_vib.append(tmp['forest']['gen_score'])

    # together
    data_combined = np.hstack((data_haddad, data_vib100))
    config['feature_selection']['k_best'] = data_combined.shape[1]
    tmp = run_lib.run_runner(config, data_combined, targets, False, False)
    together.append(tmp['forest']['gen_score'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(range(0, len(used_gloms)*3, 3), alone_haddad, color='b')
ax.set_xlabel('haddad')

ax.bar(range(1, len(used_gloms)*3+1, 3), alone_vib, color='r')
ax.set_xlabel('vib')

ax.bar(range(2, len(used_gloms)*3+2, 3), together, color='g')
ax.set_xlabel('together')
plt.show()