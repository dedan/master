#!/usr/bin/env python
# encoding: utf-8
"""

    this file will train models for different parameters and store
    them in pickles for later plotting

    * one pickle always contains a dict of models for different feature thresholds and glomeruli
    * each model is annotated with its settings
    * for each pickle we also write a json file containing the settings in a human readable format

    TODO: maybe make it work on job files in a folder (batch mode)

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""


"""
    !!! move everything in small functions, with many options this script might
    !!! become very complex later
"""
import sys
import os
import json
import pickle
import time
from collections import defaultdict
from master.libs import read_data_lib as rdl
from master.libs import features as flib
from master.libs import learning_lib as llib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
reload(rdl)
reload(flib)
reload(llib)

# read from a config file, this might become a job file later
config = json.load(open(sys.argv[1]))
door2id = json.load(open(os.path.join(config['data_path'], 'door2id.json')))
csv_path = os.path.join(config['data_path'], 'response_matrix.csv')
cas_numbers, glomeruli, rm = rdl.load_response_matrix(csv_path, door2id)


# feature related stuff
if config['feature_type'] == 'conventional':
    feature_file = os.path.join(config['data_path'], 'conventional_features',
                                config['descriptor'] + '.csv')
    features = rdl.read_feature_csv(feature_file)

elif config['feature_type'] == 'spectral':
    feature_file = os.path.join(config['data_path'], 'spectral_features',
                                config['descriptor'], 'parsed.pckl')
    spectra = pickle.load(open(feature_file))
    features = flib.get_spectral_features(spectra, config['resolution'],
                                          spec_type=config['spec_type'],
                                          use_intensity=config['use_intensity'],
                                          kernel_width=config['kernel_width'])
features = rdl.remove_invalid_features(features)
if config['normalize_features']:
    features = rdl.normalize_features(features)

res = defaultdict(dict)
for glom in config['glomeruli']:

    print glom
    glom_idx = glomeruli.index(glom)

    # select molecules available for the glomerulus
    targets , tmp_cas_numbers = rdl.get_avail_targets_for_glom(rm, cas_numbers, glom_idx)
    molids = [str(door2id[cas_number][0]) for cas_number in tmp_cas_numbers]
    assert len(molids) == len(targets)

    # for some of them the spectra are not available
    avail = [i for i in range(len(molids)) if molids[i] in features]
    targets = np.array([targets[i] for i in avail])
    data = np.array([features[molids[i]] for i in avail])
    assert targets.shape[0] == data.shape[0]

    # # feature selection
    # sel_data = rfr.transform(data, config['feature_threshold'])
    # rfr.fit(sel_data, targets)

    # random forest
    rfr = RandomForestRegressor(n_estimators=config['n_estimators'],
                                compute_importances=True,
                                oob_score=True,
                                random_state=0)
    rfr.fit(data, targets)
    res[glom]['forest'] = {'params': rfr.get_params(),
                           'train_score': rfr.score(data, targets),
                           'gen_score': rfr.oob_score_}
    del(res[glom]['forest']['params']['random_state'])

    # SVR
    svr = llib.MySVR(cross_val=True, n_folds=config['n_folds'])
    svr.fit(data, targets)
    res[glom]['svr'] = {'params': svr.get_params(),
                        'train_score': svr.score(data, targets),
                        'gen_score': svr.r2_score_}

    svr_ens = llib.SVREnsemble(n_estimators=config['n_estimators'],
                               oob_score=True)
    svr_ens.fit(data, targets)
    res[glom]['svr_ens'] = {'params': svr_ens.get_params(),
                            'train_score': svr_ens.score(data, targets),
                            'gen_score': svr_ens.oob_score_}


timestamp = time.strftime("%d%m%Y_%H%M%S", time.localtime())
json.dump(dict(res), open(os.path.join(config['results_path'], timestamp + '.json'), 'w'))
json.dump(config, open(os.path.join(config['results_path'], timestamp + '_config.json'), 'w'))



















