#!/usr/bin/env python
# encoding: utf-8
"""
    functions to create spektral features or do any other feature transforms

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import __builtin__
import os, glob
from collections import defaultdict
import csv
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import zscore

MAX_FREQ = 4000

def add_molecule_properties(features, properties):
    """add additional molecule properties like vapor pressure or size"""
    feature_start_idx = 3
    feature_file = os.path.join(os.path.dirname(__file__),
                                '..', 'data', 'basic_molecule_properties.csv')
    with open(feature_file) as f:
        reader = csv.reader(f)
        header = reader.next()

        idx = [header.index(prop) for prop in properties]
        for row in reader:
            if row[0] in features:
                data_str = ','.join([row[i] for i in idx])
                to_add = np.fromstring(data_str, dtype=float, sep=',')
                features[row[0]] = np.hstack((features[row[0]], to_add))
    return features


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
            data_str = ','.join(row[feature_start_idx:])
            features[mol] = np.fromstring(data_str, dtype=float, sep=',')
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


def get_spectral_features(spectra, resolution, use_intensity=True,
                                               spec_type='ir',
                                               kernel_widths=1,
                                               bin_width=1):
    """bining after convolution

        combine several binings if kernel_width is a list of widths
    """
    if not isinstance(kernel_widths, list):
        kernel_widths = [kernel_widths]
    combined = np.zeros((len(spectra),0))
    for k_width in kernel_widths:
        as_vectors = _place_waves_in_vector(spectra, resolution, use_intensity, spec_type)
        as_vectors = gaussian_filter(as_vectors, [0, k_width], 0)
        bined = _bining(as_vectors, bin_width)
        # combined = np.hstack((combined, bined))
    combined = bined
    features = {}
    for i, molid in enumerate(spectra):
        features[molid] = combined[i]
    assert(len(spectra) == len(features))
    return features

def _place_waves_in_vector(spectra, resolution, use_intensity, spec_type):
    """from gaussian we only get the wavenumbers, place them in vector for convolution"""

    x = np.zeros((len(spectra), int(np.ceil(MAX_FREQ/resolution))))
    for i, molid in enumerate(spectra):
        idx = np.round(np.array(spectra[molid]['freq']) / resolution).astype(int)
        if use_intensity:
            x[i, idx] = spectra[molid][spec_type]
        else:
            x[i, idx] = 1
    return x

def _bining(vectors, kernel_width):
    """divide the *continous* spectrum into bins"""
    factor = vectors.shape[1] / kernel_width
    rest = vectors.shape[1] % kernel_width
    if rest:
        return np.mean(vectors[:,:-rest].reshape((vectors.shape[0], factor, -1)), axis=2)
    else:
        return np.mean(vectors.reshape((vectors.shape[0], factor, -1)), axis=2)
