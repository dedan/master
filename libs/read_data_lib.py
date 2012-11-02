#!/usr/bin/env python
# encoding: utf-8
"""
library for all the feature analysis and selection stuff

all longer functions should be moved in here to make the individual scripts
more readable

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import os, glob
from collections import defaultdict
import csv
from scipy.stats import zscore
import numpy as np
import pylab as plt
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.rinterface import NARealType


def read_feature_csv(feature_file):
    """read one feature CSV into a dictionary structure

    csvs have molid in the 1st column, identifiere in 2nd and then features
    """
    features = {}
    molid_idx = 0
    identifiere_idx = 1
    feature_start_idx = 2
    features = defaultdict(list)

    with open(feature_file) as f:
        reader = csv.reader(f)
        header = reader.next()

        for row in reader:
            if 'Error' in row[identifiere_idx]:
                continue
            mol = row[molid_idx]
            for f_id in range(feature_start_idx, len(row)):
                features[mol].append(float(row[f_id]))
    mols = features.keys()
    for i in range(len(mols) -1):
        assert(len(features[mols[i]]) == len(features[mols[i+1]]))
    return features

def remove_invalid_features(features):
    """remove features with 0 variance"""
    valid = np.var(np.array(features.values()), axis=0) != 0.
    for feature in features.keys():
        features[feature] = features[feature][valid]
    return features

def normalize_features(features):
    """z-transform the features to make individual dimensions comparable"""
    normed = zscore(np.array(features.values()))
    orig_shape = np.array(features.values()).shape
    for i, feature in enumerate(features):
        features[feature] = normed[i]
    assert(np.array(features.values()).shape == orig_shape)
    return features

def get_features_for_molids(f_space, molids):
    """get all features for the given molecule IDs

        result is returnd as array: molecules x features
    """
    mol_fspace = [[f_space[f][molid] for f in f_space if molid in f_space[f]]
                                     for molid in molids]
    # remove empty entries (features for molid not available)
    available = [i for i in range(len(mol_fspace)) if mol_fspace[i]]
    mol_fspace = [elem if elem else [0] * len(f_space) for elem in mol_fspace]
    return np.array(mol_fspace), available

def get_data_from_r(path_to_csv):
    """extract the response matrix from the R package and save it as a CSV"""
    importr('DoOR.function')
    importr('DoOR.data')
    load_data = robjects.r['loadRD']
    load_data()
    rm = robjects.r['response.matrix']
    rm.to_csvfile(path_to_csv)

def load_response_matrix(path_to_csv, door2id=None):
    """load the DoOR response matrix from the R package

        if door2id given, return only the stimuli for which we have a molID
    """
    with open(path_to_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        glomeruli = reader.next()
        cas_numbers, data = [], []
        for row in reader:
            if not row[0] in ['SFR', 'solvent']:
                cas_numbers.append(row[0])
                data.append(row[1:])
    rm = np.zeros((len(cas_numbers), len(glomeruli)))
    for i in range(len(cas_numbers)):
        for j in range(len(glomeruli)):
            rm[i, j] = float(data[i][j]) if data[i][j] != 'NA' else np.nan

    if door2id:
        # check whether we have molids for the CAS number and if not remove them
        stim_idx = [i for i in range(len(cas_numbers)) if door2id[cas_numbers[i]]]
        rm = rm[stim_idx, :]
        cas_numbers = [cas_numbers[i] for i in stim_idx]

    return cas_numbers, glomeruli, rm

def select_n_best_glomeruli(response_matrix, glomeruli, n_glomeruli):
    """select the glomeruli with most stimuli available"""
    glom_available = np.sum(~np.isnan(response_matrix), axis=0)
    glom_available_idx = np.argsort(glom_available)[::-1]
    return [glomeruli[i] for i in glom_available_idx[:n_glomeruli]]

def get_avail_targets_for_glom(rm, cas_numbers, glom_idx):
    """filter response matrix and cas numbers for availability in a glomerulus"""
    avail_cas_idx = np.where(~np.isnan(rm[:, glom_idx]))[0]
    tmp_rm = rm[avail_cas_idx, glom_idx]
    tmp_cas_numbers = [cas_numbers[i] for i in avail_cas_idx]
    return tmp_rm, tmp_cas_numbers
