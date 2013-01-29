#!/usr/bin/env python
# encoding: utf-8
'''
I'm still surprised that the SVR performs that good with all featues together,
I would expect the metric for support vector distance to break in such
a high D space with many noisy dimensions. But the randomization test tells me
the high generalization scores are not just a result of chance. To be really sure
I decided to combine the saito descriptor with a randomized version of the all
descriptor. This is then also very high dimensional and has data that can
be used for prediction only in a few dimensions. Let's see how this performs.

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
        'descriptor': 'all',
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

for glom in used_gloms:

    config['glomerulus'] = glom
    features = run_lib.prepare_features(config)
    data_all, targets, _ = run_lib.load_data_targets(config, features)
    config['features']['descriptor'] = 'saito_desc'
    data_saito, _, _ = run_lib.load_data_targets(config, features)
    np.random.seed()
    # map(np.random.shuffle, data_all.T)
    new_data = np.hstack((data_saito, data_all))
    config['feature_selection']['k_best'] = data_all.shape[1]
    tmp = run_lib.run_runner(config, data_all, targets, False, False)
    print glom, tmp['svr']['gen_score']
