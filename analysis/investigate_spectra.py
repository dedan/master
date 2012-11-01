#!/usr/bin/env python
# encoding: utf-8
"""
    which band tells us most about our data
"""

import sys, os, pickle, json, __builtin__
import numpy as np
import pylab as plt
from master.libs import read_data_lib as rdl
from master.libs import utils
from master.libs import features
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
reload(features)

plt.close('all')

base_path = '/Users/dedan/projects/master/'
ir_file = '/Users/dedan/projects/master/data/spectra/gamess_am1/parsed.pckl'
out_folder = '/Users/dedan/projects/master/results/spectra/plots'
format = 'png'
# selected via the basic statistics script
interesting_glomeruli = ['Or19a', 'Or22a', 'Or35a', 'Or43b', 'Or67a',
                         'Or67b', 'Or7a', 'Or85b', 'Or98a', 'Or9a']
n_glomeruli = 5
resolution = 0.5
recompute = False

# read in the IR spectra TODO: move them to data when final version exists
spectra = pickle.load(open(ir_file))
door2id = json.load(open(os.path.join(base_path, 'data', 'door2id.json')))

# investigate only the glomeruli for which we have most molecules available
csv_path = os.path.join(base_path, 'data', 'response_matrix.csv')
cas_numbers, glomeruli, rm = rdl.load_response_matrix(csv_path, door2id)
# best_glom = rdl.select_n_best_glomeruli(rm, glomeruli, n_glomeruli)
# print best_glom

# # histogram of number of available frequencies
# plt.figure()
# plt.hist([len(mol['freq']) for mol in spectra.values()])
# plt.title('number of frequencies available')
# plt.savefig(os.path.join(out_folder, 'frequencies_hist.' + format))

kernel_widths = [2, 3, 5, 10, 20, 30, 50]

res = {}
# data collection
if recompute:
    for glom in interesting_glomeruli:

        print glom
        glom_idx = glomeruli.index(glom)

        # select molecules available for the glomerulus
        targets , tmp_cas_numbers = rdl.get_avail_targets_for_glom(rm, cas_numbers, glom_idx)
        molids = [door2id[cas_number][0] for cas_number in tmp_cas_numbers]

        # for some of them the spectra are not available
        targets = [targets[i] for i in range(len(tmp_cas_numbers)) if str(molids[i]) in spectra]
        molids = [m for m in molids if str(m) in spectra]

        res[glom] = {'data': {}, 'regression': {}, 'forest': {}, 'oob': {},
                     'targets': targets, 'oob_prediction': {}}
        for i, kernel_width in enumerate(kernel_widths):

            data = features.get_spectral_features(spectra, molids, resolution, kernel_width=kernel_width)
            # res[glom]['data'][kernel_width] = data

            # univariate test
            _, p = f_regression(data, targets)
            res[glom]['regression'][kernel_width] = -np.log10(p)

            # random forest regression
            rfr = RandomForestRegressor(n_estimators=500,
                                        compute_importances=True,
                                        oob_score=True)

            # TODO: feature selection step
            rfr.fit(data,targets)
            res[glom]['forest'][kernel_width] = rfr.feature_importances_
            res[glom]['oob'][kernel_width] = rfr.oob_score_
            res[glom]['oob_prediction'][kernel_width] = rfr.oob_prediction_
    pickle.dump(res, open(os.path.join(out_folder, 'res.pckl'), 'w'))
else:
    res = pickle.load(open(os.path.join(out_folder, 'res.pckl')))

# plotting
for glom in interesting_glomeruli:
    print 'plotting: ', glom
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(glom)
    fig1 = plt.figure()
    # normalize both methods to their maximum value to make them comparable
    max_reg = utils.max_in_values(res[glom]['regression'])
    max_for = utils.max_in_values(res[glom]['forest'])
    for i, kernel_width in enumerate(kernel_widths):

        ax = fig.add_subplot(len(kernel_widths), 1, i+1)
        ax.plot(np.array(res[glom]['regression'][kernel_width]) / max_reg, 'b')
        ax.plot(np.array(res[glom]['forest'][kernel_width]) / max_for, 'r')
        ax.set_ylabel(kernel_width, rotation='0')
        ax.set_xlim([0, res[glom]['regression'][kernel_width].shape[0]])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylabel('%.2f' % res[glom]['oob'][kernel_width], rotation='0')

        w_c = utils.ceiled_root(len(kernel_widths))
        ax = fig1.add_subplot(w_c, w_c, i+1)
        ax.scatter(res[glom]['targets'], res[glom]['oob_prediction'][kernel_width])


    fig.savefig(os.path.join(out_folder, 'spectral_features_' + glom + '.' + format))
    fig1.savefig(os.path.join(out_folder, 'target_vs_prediction_' + glom + '.' + format))

# compare kernel width for different glomeruli
fig = plt.figure(figsize=(10,10))
for i, kernel_width in enumerate(kernel_widths):
    ax1 = fig.add_subplot(len(kernel_widths)+1, 2, (i*2)+1)
    ax2 = fig.add_subplot(len(kernel_widths)+1, 2, (i*2)+2)
    for glom in interesting_glomeruli:
        ax1.plot(res[glom]['regression'][kernel_width])
        ax1.set_xticklabels([])
        ax1.set_yticks([0, ax1.get_yticks()[-1]])
        ax1.set_yticklabels(['', ax1.get_yticks()[-1]])
        ax1.set_title('glom: {}'.format(glom))
        ax1.set_xlim([0, res[glom]['forest'][kernel_width].shape[0]])
        ax2.plot(res[glom]['forest'][kernel_width])
        ax2.set_yticks([0, ax2.get_yticks()[-1]])
        ax2.set_yticklabels(['', ax2.get_yticks()[-1]])
        ax2.set_xticklabels([])
        ax2.set_title('width: {}'.format(kernel_width))
        ax2.set_xlim([0, res[glom]['forest'][kernel_width].shape[0]])

ax1.set_xlabel('regression')
ax2.set_xlabel('forest')
fig.subplots_adjust(hspace=0.4)
fig.savefig(os.path.join(out_folder, 'glom_comparison.' + format))


