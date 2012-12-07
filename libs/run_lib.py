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
from collections import defaultdict
from master.libs import read_data_lib as rdl
from master.libs import features_lib as flib
from master.libs import learning_lib as llib
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.preprocessing import normalize
import numpy as np
reload(rdl)
reload(flib)
reload(llib)

ml_methods = {'forest': RandomForestRegressor,
              'svr': llib.MySVR,
              'svr_ens': llib.SVREnsemble}

def do_paramsearch(sc, config, features):
    """docstring for do_paramsearch"""
    tmp_res = {}
    data, targets, _ = load_data_targets(config, features)
    sel_scores = get_selection_score(config, data, targets)
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
            print('running {} {} {}'.format(config['glomerulus'], k_b, i))
            data_sel = flib.select_k_best(data, sel_scores, k_b)
            tmp = run_runner(config, data_sel, targets)
            tmp_res[str(k_b)][str(i)] = tmp
    return tmp_res

def randomization_test(config, n_repetitions):
    """validate results via randomization_test"""
    assert len(config['methods']) == 1  # only one method a time
    config['randomization_test'] = False
    k_best = config['feature_selection']['k_best']
    features = prepare_features(config)
    data, targets, _ = load_data_targets(config, features)
    print data.shape
    sel_scores = get_selection_score(config, data, targets)
    data = flib.select_k_best(data, sel_scores, k_best)
    true_res = run_runner(config, data, targets).values()[0]['gen_score']

    # add shuffle data to the config and run the runner for N times
    config['randomization_test'] = True
    rand_res = []
    orig_data, orig_targets, _ = load_data_targets(config, features)
    for i in range(n_repetitions):
        data, targets = orig_data.copy(), orig_targets.copy()
        sel_scores = get_selection_score(config, data, targets)
        data = flib.select_k_best(data, sel_scores, k_best)
        tmp_res = run_runner(config, data, targets)
        rand_res.append(tmp_res.values()[0]['gen_score'])
    return rand_res, true_res

def prepare_features(config):
    """load and prepare the features, either conventional or spectral"""

    if config['features']['type'] == 'conventional':
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
    assert targets.shape[0] == data.shape[0]
    return data, targets, molids


def get_selection_score(config, data, targets):
    """select k_best features"""
    # feature selection
    if config['feature_selection']['method'] == 'linear':
        sel_scores, _ = f_regression(data, targets)
    elif config['feature_selection']['method'] == 'forest':
        rfr_sel = RandomForestRegressor(compute_importances=True, random_state=0)
        sel_scores = rfr_sel.fit(data, targets).feature_importances_
    return sel_scores


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
        method_params = config['methods'][method]
        for p in method_params:
            if isinstance(method_params[p], unicode):
                method_params[p] = str(method_params[p])
        regressor = ml_methods[method](**method_params)
        regressor.fit(data, targets)
        res[method] = {'train_score': regressor.score(data, targets),
                       'gen_score': regressor.oob_score_}
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

