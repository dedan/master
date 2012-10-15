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


def read_feature_csvs(features_path):
    """read the feature CSVs into a dictionary structure

    csvs have molid in the 1st column, identifiere in 2nd and then features
    """
    features = {}
    molid_idx = 0
    identifiere_idx = 1
    feature_start_idx = 2
    all_feature_files = glob.glob(os.path.join(features_path, '*.csv'))
    for feature_file in all_feature_files:

        f_space = os.path.splitext(os.path.basename(feature_file))[0]
        features[f_space] = defaultdict(dict)

        with open(feature_file) as f:
            reader = csv.reader(f)
            header = reader.next()

            for row in reader:
                if 'Error' in row[identifiere_idx]:
                    continue
                mol = row[molid_idx]
                for f_id in range(feature_start_idx, len(row)):
                    try:
                        features[f_space][header[f_id]][mol] = float(row[f_id])
                    except:
                        features[f_space][header[f_id]][mol] = 0.
    return features

def remove_invalid_features(features):
    """remove features with 0 variance"""
    for f_space in features:
        for feature in list(features[f_space].keys()):
            if np.var(features[f_space][feature].values()) == 0:
                del(features[f_space][feature])
    return features


def normalize_features(features):
    """z-transform the features to make individual dimensions comparable"""
    for f_space in features:
        for feature in features[f_space]:
            normed = zscore(features[f_space][feature].values())
            keys = features[f_space][feature].keys()
            for key, value in zip(keys, normed):
                features[f_space][feature][key] = value
    return features

def get_features_for_molids(f_space, molids):
    """get all features for the given molecule IDs

        result is returnd as array: molecules x features
    """
    mol_fspace = [[f_space[f][molid] for f in f_space if molid in f_space[f]]
                                     for molid in molids]
    # remove empty entries (features for molid not available)
    mol_fspace = [elem for elem in mol_fspace if elem]
    return np.array(mol_fspace)

def load_response_matrix():
    """load the DoOR response matrix from the R package"""
    importr('DoOR.function')
    importr('DoOR.data')
    load_data = robjects.r['loadRD']
    load_data()
    rm = robjects.r['response.matrix']
    row_idx = [i for i in range(len(rm.rownames))
                    if not rm.rownames[i] in ['SFR', 'solvent']]
    np_rm = np.zeros((len(row_idx), len(rm)))
    for col_i, col in enumerate(rm):
        for new_row_i, row_i in enumerate(row_idx):
            try:
                np_rm[new_row_i, col_i] = col[row_i]
            except:
                np_rm[new_row_i, col_i] = NARealType()
    row_names = [rm.rownames[i] for i in row_idx]
    return row_names, rm.colnames, np_rm

