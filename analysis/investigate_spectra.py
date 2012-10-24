#!/usr/bin/env python
# encoding: utf-8
"""
    which band tells us most about our data
"""

import sys, os, pickle, json, __builtin__
import numpy as np
import pylab as plt
from master.libs import read_data_lib as rdl
from scipy.ndimage.filters import gaussian_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, f_regression

plt.close('all')

base_path = '/Users/dedan/projects/master/'
ir_file = '/Users/dedan/projects/master/results/gamess/ir.pckl'
out_folder = '/Users/dedan/projects/master/results/spectra'
format = 'png'
n_glomeruli = 5
resolution = 0.5

# read in the IR spectra TODO: move them to data when final version exists
spectra = pickle.load(open(ir_file))
door2id = json.load(open(os.path.join(base_path, 'data', 'door2id.json')))

# investigate only the glomeruli for which we have most molecules available
csv_path = os.path.join(base_path, 'data', 'response_matrix.csv')
cas_numbers, glomeruli, rm = rdl.load_response_matrix(csv_path, door2id)
best_glom = rdl.select_n_best_glomeruli(rm, glomeruli, n_glomeruli)
print best_glom

# # histogram of number of available frequencies
# plt.figure()
# plt.hist([len(mol['freq']) for mol in spectra.values()])
# plt.title('number of frequencies available')
# plt.savefig(os.path.join(out_folder, 'frequencies_hist.' + format))

kernel_widths = [2, 3, 5, 10, 20, 30, 50]

def get_spectral_features(spectra, molids, kernel_width=1, max_freq=None):
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

def max_in_values(value_dict):
    """maximum of non NANs in the values of a dict"""
    stacker = lambda x,y: np.hstack((x,y))
    all_values = reduce(stacker, value_dict.values(), np.array([]))
    return np.max(all_values[~np.isnan(all_values)])

res = {}
for glom in best_glom:

    print glom
    glom_idx = glomeruli.index(glom)

    # select molecules available for the glomerulus
    targets , tmp_cas_numbers = rdl.get_avail_targets_for_glom(rm, cas_numbers, glom_idx)
    molids = [door2id[cas_number][0] for cas_number in tmp_cas_numbers]

    # overlay of all spectra
    for molid in molids:
        assert len(spectra[str(molid)]['freq']) == len(spectra[str(molid)]['ir'])
    all_freq = __builtin__.sum([spectra[str(molid)]['freq'] for molid in molids], [])
    all_ir = __builtin__.sum([spectra[str(molid)]['ir'] for molid in molids], [])

    # distribution of distances between frequences (helps to decide for resolution)
    # plt.figure()
    # plt.hist(np.diff(sorted(all_freq)), bins=1000)
    # plt.xlim([0,2])

    res[glom] = {'data': {}, 'regression': {}, 'forest': {}}
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(glom)
    for i, kernel_width in enumerate(kernel_widths):

        data = get_spectral_features(spectra, molids, kernel_width=kernel_width)
        res[glom]['data'][kernel_width] = data

        # univariate test
        _, p = f_regression(data, targets)
        res[glom]['regression'][kernel_width] = -np.log10(p)

        # random forest regression
        rfr = RandomForestRegressor(n_estimators=10, compute_importances=True)
        rfr.fit(data,targets)
        res[glom]['forest'][kernel_width] = rfr.feature_importances_

    # normalize both methods to their maximum value to make them comparable
    max_reg = max_in_values(res['regression'])
    max_for = max_in_values(res['forest'])
    for i, kernel_width in enumerate(kernel_widths):
        ax = fig.add_subplot(len(kernel_widths)*2, 1, (i*2)+1)
        ax.imshow(res['data'][kernel_width], aspect='auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax = fig.add_subplot(len(kernel_widths)*2, 1, (i*2)+2)
        ax.plot(np.array(res['regression'][kernel_width]) / max_reg, 'b')
        ax.plot(np.array(res['forest'][kernel_width]) / max_for, 'r')
        ax.set_ylabel(kernel_width)
        ax.set_xlim([0, res['regression'][kernel_width].shape[0]])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([])

    fig.savefig(os.path.join(out_folder, 'spectral_features_' + glom + '.' + format))

# compare kernel width 20 for different glomeruli

