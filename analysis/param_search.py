#!/usr/bin/env python
# encoding: utf-8
'''

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
'''
import glob
import sys
import os
import json
from master.libs import run_lib
from master.libs import features_lib as flib
import numpy as np
import copy
reload(run_lib)
reload(flib)

configs = []
# search config
sc = json.load(open(sys.argv[1]))
config = sc['runner_config_content']
config['data_path'] = os.path.join(os.path.dirname(__file__), '..', 'data')

if config['features']['type'] == 'conventional':
    files = glob.glob(os.path.join(config['data_path'], 'conventional_features', '*.csv'))
    for f in files:
        desc = os.path.splitext(os.path.basename(f))[0]
        if 'descriptors' in sc and not desc in sc['descriptors']:
            continue
        config['features']['descriptor'] = desc
        config['run_name'] = desc
        configs.append(copy.deepcopy(config))
elif config['features']['type'] == 'spectral':
    config['features']['descriptor'] = 'large_base'
    for kwidth in sc['kernel_widths']:
        config['features']['kernel_width'] = kwidth
        config['run_name'] = repr(kwidth)
        configs.append(copy.deepcopy(config))
else:
    assert False


for config in configs:

    sc['runner_config_content'] = config
    # if result file already exists, load it to append new glomeruli
    if os.path.exists(os.path.join(sc['outpath'], config['run_name'] + '.json')):
        print('load existing results from: {}'.format(config['run_name']))
        res = json.load(open(os.path.join(sc['outpath'], config['run_name'] + '.json')))["res"]
    else:
        res = {sel: {} for sel in sc['selection']}

    # load the features
    features = run_lib.prepare_features(config)
    n_features = len(features[features.keys()[0]])
    max_expo = int(np.floor(np.log2(n_features)))
    sc['k_best'] = [2**i for i in range(max_expo)] + [n_features]

    print 'working on: ', config['run_name']
    for selection in sc['selection']:
        print selection
        config['feature_selection']['method'] = selection
        for glomerulus in sc['glomeruli']:
            if not glomerulus in res[selection]:
                res[selection][glomerulus] = {}
            config['glomerulus'] = glomerulus
            res[selection][glomerulus] = run_lib.do_paramsearch(sc, config, features)
            print('param search for {} done'.format(glomerulus))
            json.dump({'sc': sc, 'res': res}, open(os.path.join(sc['outpath'], config['run_name'] + '.json'), 'w'))
