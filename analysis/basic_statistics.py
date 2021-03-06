#!/usr/bin/env python
# encoding: utf-8
"""
Print and plot basic statistics on the data I am using.

How many data I have, its distribution, etc. The idea is to give an overview
of all things we use in this analysis

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys, os, pickle, json
import master.libs.read_data_lib as rdl
from master.libs import utils
import numpy as np
import pylab as plt
from scipy.stats import scoreatpercentile
reload(rdl)

data_path = '/Users/dedan/projects/master/data'
results_path = '/Users/dedan/projects/master/results/summary/'
descriptor = 'ATOMCENTRED_FRAGMENTS'
format = 'png'
N = 50
percentile = 75
percentile_thres = 0.2

door2id = json.load(open(os.path.join(data_path, 'door2id.json')))
cas_numbers, glomeruli, rm = rdl.load_response_matrix(os.path.join(data_path, 'response_matrix.csv'))

# which molecules are missing in door2id?
print 'molecues missing in door2id: \n%s' % [r for r in cas_numbers if not door2id[r]]

# number of measurements available
fig = plt.figure(figsize=(20, 5))
ax = fig.add_subplot(111)
ax.bar(range(len(cas_numbers)), np.sum(~np.isnan(rm), axis=1))
ax.set_xticks(np.arange(len(cas_numbers)) + 1)
bla = []
for i, g in enumerate(cas_numbers):
    bla.append(g + ' ' * 40 if i % 2 == 0 else '' + g)
ax.set_xticklabels(bla, rotation='90', ha='right')
ax.set_title('number of glomeruli available for a stimulus')
fig.savefig(os.path.join(results_path, 'glomeruli_per_stimulus.' + format))

# histograms for the glomeruli with more than N stimuli
print '\n target distribution of glomeruli with more than %d stimuli' % N
interesting = []
fig = plt.figure(figsize=(10, 10))
larger_n_idx = np.where(np.sum(~np.isnan(rm), axis=0) > 50)[0]
n_avail_with_all_conditions = []
for i, idx in enumerate(larger_n_idx):
    w_c = utils.ceiled_root(len(larger_n_idx))
    ax = fig.add_subplot(w_c, w_c, i + 1)
    glom = rm[:, idx]
    data = glom[~np.isnan(glom)]
    if scoreatpercentile(data, 75) < 0.2:
        ax.hist(data, bins=np.arange(0, 1.1, 0.1), color='#990000')
    else:
        n_avail_with_all_conditions.append(np.sum(~np.isnan(rm[:, idx])))
        interesting.append(glomeruli[idx])
        ax.hist(data, bins=np.arange(0, 1.1, 0.1), color='#38761d')
    ax.set_title(glomeruli[idx])
    ax.set_xticklabels([])  # can be switched of because all values 0 < x < 1
fig.subplots_adjust(hspace=0.3)
fig.savefig(os.path.join(results_path, 'target_quality.' + format))
print ('\tlist of glomeruli with more than %d stimuli and %d percentile > %f' %
       (N, percentile, percentile_thres))
print interesting

print 'mean number of mols available for the 10 glom with all conds fulfilled'
print np.mean(n_avail_with_all_conditions), n_avail_with_all_conditions

# number of stimuli availabe for each glomerulus, colored according to selected or not
fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)
n_stimuli = np.sum(~np.isnan(rm), axis=0)
sorted_idx = list(reversed(np.argsort(n_stimuli)))
patches = ax.bar(range(len(glomeruli)), n_stimuli[sorted_idx])
for i, p in enumerate(patches):
    if glomeruli[sorted_idx[i]] in interesting:
        p.set_facecolor('#38761d')
    else:
        p.set_facecolor('#990000')
ax.set_xticks(np.arange(len(glomeruli)) + 1)
ax.set_xticklabels([glomeruli[i] for i in sorted_idx], rotation='45', ha='right')
ax.set_title('number of stimuli available for a glomerulus')
fig.savefig(os.path.join(results_path, 'stimuli_per_glomerulus.' + format))
for i in sorted_idx:
    print('{}: {}'.format(glomeruli[i], n_stimuli[i]))

# number of molecules available for all glomeruli
print 'glomerulus for which all stimuli are available'
all_stims_idx = np.where(np.sum(~np.isnan(rm), axis=0) == rm.shape[0])
print [glomeruli[i] for i in all_stims_idx[0]]
