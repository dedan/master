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

def _gaussian(x, mu, sigma):
    """docstring for gaussian"""
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x-mu)/float(sigma))**2)

def _sum_of_gaussians(x_range, positions, heights, sigma):
    """docstring for sum_of_gaussians"""
    assert len(positions) == len(heights)
    return [np.sum(heights * _gaussian(x, positions, sigma)) for x in x_range]

def get_spectral_features(spectra, use_intensity=True, spec_type='ir',
                          kernel_widths=10, bin_width=10, BFS_max=4000):
    """ sum of gaussians, sampled at regular intervals """
    x_range = range(0, BFS_max, bin_width)
    if not isinstance(kernel_widths, list):
        kernel_widths = [kernel_widths]
    if not isinstance(spec_type, list):
        spec_type = [spec_type]
    features = defaultdict(list)
    for molid, spectrum in spectra.items():
        # remove negative vibrations (result of imaginary frequencies)
        valid_freqs = spectrum['freq'][spectrum['freq'] > 0]
        tmp = []
        for k_width in kernel_widths:
            for st in spec_type:
                intensities = spectrum[st] if use_intensity else np.ones(len(spectrum[st]))
                sampled = _sum_of_gaussians(x_range, spectrum['freq'], intensities, k_width)
                tmp += sampled
        features[molid] = np.array(tmp)
    assert(len(spectra) == len(features))
    return features

