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

def get_spectral_features(spectra, molids, resolution, kernel_width=1, max_freq=None):
    """bining after convolution"""
    if not max_freq:
        all_freq = __builtin__.sum([spectra[str(molid)]['freq'] for molid in molids], [])
        max_freq = np.max(all_freq)

    x = np.zeros((len(molids), int(np.ceil(np.max(all_freq)/resolution)) + 1))
    for i, molid in enumerate(molids):
        idx = np.round(np.array(spectra[str(molid)]['freq']) / resolution).astype(int)
        x[i, idx] = spectra[str(molid)]['ir']
    x = gaussian_filter(x, [0, kernel_width], 0)
    # bining
    factor, rest = x.shape[1] / kernel_width, x.shape[1] % kernel_width
    if rest:
        data = np.mean(x[:,:-rest].reshape((x.shape[0], factor, -1)), axis=2)
    else:
        data = np.mean(x.reshape((x.shape[0], factor, -1)), axis=2)
    return data
