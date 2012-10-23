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

    fig = plt.figure()
    for i, kernel_width in enumerate(kernel_widths):

        ax = fig.add_subplot(len(kernel_widths)*2, 1, (i*2)+1)
        data = get_spectral_features(spectra, molids, kernel_width=kernel_width)
        ax.imshow(data, aspect='auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        _, p = f_regression(data, targets)

        ax = fig.add_subplot(len(kernel_widths)*2, 1, (i*2)+2)
        # rfr = RandomForestRegressor(n_estimators=10, compute_importances=True)
        # rfr.fit(data,targets)
        # ax.plot(rfr.feature_importances_)
        ax.plot(-np.log10(p))
        ax.set_xlim([0, data.shape[1]])


    # kernel_width = 10
    # # compute features for each molecule
    # max_freq = np.max(frequencies)



    # rfr = RandomForestRegressor(n_estimators=10, compute_importances=True)
    # rfr.fit(data,targets)
    # res[descriptor][glom]['rf'] = rfr.feature_importances_



