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
reload(run_lib)
reload(flib)

# search config
sc = json.load(open(sys.argv[1]))
config = json.load(open(sc['runner_config']))
config['features']['properties_to_add'] = sc['properties_to_add']
sc['runner_config_content'] = config

feature_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'conventional_features')
files = glob.glob(os.path.join(feature_path, '*.csv'))
for f in files:
    desc = os.path.splitext(os.path.basename(f))[0]
    config['features']['descriptor'] = desc
    config['features']['type'] = 'conventional'

    # if result file already exists, load it to append new glomeruli
    if os.path.exists(os.path.join(sc['outpath'], desc + '.json')):
        print('load existing results from: {}'.format(desc))
        res = json.load(open(os.path.join(sc['outpath'], desc + '.json')))["res"]
    else:
        res = {sel: {} for sel in sc['selection']}

    # load the features
    features = run_lib.prepare_features(config)
    n_features = len(features[features.keys()[0]])
    sc['k_best'] = [2**i for i in range(10) if 2**i < n_features] + [n_features]

    print 'working on: ', desc

    for selection in sc['selection']:
        print selection
        config['feature_selection']['method'] = selection
        for glomerulus in sc['glomeruli']:
            if not glomerulus in res[selection]:
                res[selection][glomerulus] = {}
            config['glomerulus'] = glomerulus
            data, targets = run_lib.load_data_targets(config, features)
            sel_scores = run_lib.get_selection_score(config, data, targets)
            for k_b in sc['k_best']:
                if not str(k_b) in res[selection][glomerulus]:
                    res[selection][glomerulus][str(k_b)] = {}
                config['feature_selection']['k_best'] = k_b
                for i in range(len(sc['forest'])):
                    if str(i) in res[selection][glomerulus][str(k_b)]:
                        continue
                    config['methods']['svr']['C'] = sc['svr'][i]
                    config['methods']['svr_ens']['C'] = sc['svr'][i]
                    config['methods']['forest']['max_depth'] = sc['forest'][i]
                    print('running {} {} {}'.format(glomerulus, k_b, i))
                    data = flib.select_k_best(data, sel_scores, k_b)
                    tmp_res = run_lib.run_runner(config, data, targets)
                    tmp_res['n_features'] = n_features
                    res[selection][glomerulus][str(k_b)][str(i)] = tmp_res
            print('param search for {} done'.format(glomerulus))
    json.dump({'sc': sc, 'res': res}, open(os.path.join(sc['outpath'], desc + '.json'), 'w'))
