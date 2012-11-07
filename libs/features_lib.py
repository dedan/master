#!/usr/bin/env python
# encoding: utf-8
"""
    functions to create spektral features or do any other feature transforms

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import __builtin__
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def get_k_best(scores, k):
    """get indices for the k best features depending on the scores"""
    assert not (scores < 0).any()
    scores[np.isnan(scores)] = 0
    return np.argsort(scores)[-k:]

def get_spectral_features(spectra, resolution, use_intensity=True,
                                               spec_type='ir',
                                               kernel_width=1):
    """bining after convolution"""
    all_freq = __builtin__.sum([spectra[molid]['freq'] for molid in spectra], [])
    max_freq = np.max(all_freq)

    x = np.zeros((len(spectra), int(np.ceil(np.max(all_freq)/resolution)) + 1))
    for i, molid in enumerate(spectra):
        idx = np.round(np.array(spectra[molid]['freq']) / resolution).astype(int)
        if use_intensity:
            x[i, idx] = spectra[molid][spec_type]
        else:
            x[i, idx] = 1
    x = gaussian_filter(x, [0, kernel_width], 0)
    # bining
    factor, rest = x.shape[1] / kernel_width, x.shape[1] % kernel_width
    if rest:
        data = np.mean(x[:,:-rest].reshape((x.shape[0], factor, -1)), axis=2)
    else:
        data = np.mean(x.reshape((x.shape[0], factor, -1)), axis=2)
    features = {}
    for i, molid in enumerate(spectra):
        features[molid] = data[i]
    assert(len(spectra) == len(features))
    return features
