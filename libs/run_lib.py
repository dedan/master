#!/usr/bin/env python
# encoding: utf-8
"""

    this file will train models for different parameters and store
    them in pickles for later plotting

    * one pickle always contains a dict of models for different feature thresholds and glomeruli
    * each model is annotated with its settings
    * for each pickle we also write a json file containing the settings in a human readable format

    TODO: maybe make it work on job files in a folder (batch mode)

    List of parameters that can be explored with this file

    * parameters for each ML method (e.g. C for the SVR)
    * different features
    * spectral features with or without use_intensity
    * feature selection threshold
    * different feature selection methods
    * depth of a tree
    * combination of different spectral bands (kernel_widths)
    * per sample normalization


Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
import pickle
import time
import glob
from collections import defaultdict
from master.libs import read_data_lib as rdl
from master.libs import features_lib as flib
from master.libs import learning_lib as llib
from sklearn.preprocessing import normalize
import numpy as np
reload(rdl)
reload(flib)
reload(llib)

ml_methods = {'forest': llib.MyRFR,
              'svr': llib.MySVR,
              'svr_ens': llib.SVREnsemble}

def do_paramsearch(sc, config, features):
    """docstring for do_paramsearch"""
    tmp_res = {}
    data, targets, _ = load_data_targets(config, features)
    for k_b in sc['k_best']:
        if not str(k_b) in tmp_res:
            tmp_res[str(k_b)] = {}
        config['feature_selection']['k_best'] = k_b
        for i in range(len(sc['svr'])):
            if str(i) in tmp_res[str(k_b)]:
                continue
            if 'svr' in config['methods']:
                config['methods']['svr']['C'] = sc['svr'][i]
            if 'svr_ens' in config['methods']:
                config['methods']['svr_ens']['C'] = sc['svr'][i]
            if 'forest' in config['methods']:
                config['methods']['forest']['max_depth'] = sc['forest'][i]
            print('running for {} - {}'.format(k_b, sc['svr'][i]))
            tmp = run_runner(config, data, targets)
            tmp_res[str(k_b)][str(i)] = tmp
    return tmp_res


def prepare_features(config):
    """load and prepare the features, either conventional or spectral"""

    if config['features']['type'] == 'conventional':
        features_path = os.path.join(config['data_path'], 'conventional_features')
        if config['features']['descriptor'] == 'all':
            all_features = defaultdict(lambda: np.array([]))
            feature_files = glob.glob(os.path.join(features_path, '*.csv'))
            for feature_file in feature_files:
                if 'haddad' in feature_file or 'saito' in feature_file:
                    continue
                features = flib.read_feature_csv(feature_file)
                for key, value in features.items():
                    all_features[key] = np.hstack((all_features[key], value))
            features = all_features
            max_len = max([len(val) for val in features.values()])
            for key in list(features.keys()):
                if len(features[key]) < max_len:
                    del features[key]
                    print 'deleted a molecule because not all features available'
        else:
            feature_file = os.path.join(config['data_path'], 'conventional_features',
                                        config['features']['descriptor'] + '.csv')
            features = flib.read_feature_csv(feature_file)
    elif config['features']['type'] == 'spectral':
        feature_file = os.path.join(config['data_path'], 'spectral_features',
                                    'large_base', 'parsed.pckl')
        spectra = pickle.load(open(feature_file))
        features = flib.get_spectral_features(spectra, config['features']['resolution'],
                                              spec_type=config['features']['spec_type'],
                                              use_intensity=config['features']['use_intensity'],
                                              kernel_widths=config['features']['kernel_width'])
    features = flib.remove_invalid_features(features)
    if config['features']['properties_to_add']:
        flib.add_molecule_properties(features, config['features']['properties_to_add'])
    if config['features']['normalize']:
        features = flib.normalize_features(features)
    return features


def load_data_targets(config, features):
    """load the targets for a glomerulus"""
    door2id = json.load(open(os.path.join(config['data_path'], 'door2id.json')))
    csv_path = os.path.join(config['data_path'], 'response_matrix.csv')
    cas_numbers, glomeruli, rm = rdl.load_response_matrix(csv_path, door2id)
    glom_idx = glomeruli.index(config['glomerulus'])

    # select molecules available for the glomerulus
    targets , tmp_cas_numbers = rdl.get_avail_targets_for_glom(rm, cas_numbers, glom_idx)
    molids = [str(door2id[cas_number][0]) for cas_number in tmp_cas_numbers]
    assert len(molids) == len(targets)

    # for some of them the spectra are not available
    avail = [i for i in range(len(molids)) if molids[i] in features]
    targets = np.array([targets[i] for i in avail])
    data = np.array([features[molids[i]] for i in avail])
    molids = [m for i, m in enumerate(molids) if i in avail]
    assert targets.shape[0] == data.shape[0]
    assert targets.shape[0] == len(molids)
    return data, targets, molids


def run_runner(config, data, targets, get_models=False):
    """docstring for run"""
    res = {}
    if config['features']['normalize_samples']:
        data = normalize(data, norm='l2', axis=1, copy=True)
    if config['randomization_test']:
        np.random.seed()
        map(np.random.shuffle, data.T)
    # train models and get results
    for method in config['methods']:
        regressor = ml_methods[method](**config['methods'][method])
        regressor.fit(data, targets,
                      config['feature_selection']['method'],
                      config['feature_selection']['k_best'])
        res[method] = {'train_score': regressor.score(data, targets),
                       'gen_score': regressor.gen_score}
        if get_models:
            res[method]['model'] = regressor
    return res


if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))
    features = prepare_features(config)
    res = run_runner(config, features)
    timestamp = time.strftime("%d%m%Y_%H%M%S", time.localtime())
    json.dump(res, open(os.path.join(config['results_path'], timestamp + '.json'), 'w'))
    json.dump(config, open(os.path.join(config['results_path'], timestamp + '_config.json'), 'w'))

