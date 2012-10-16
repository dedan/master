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
from sklearn import svm
from sklearn import linear_model
import numpy as np
import pylab as plt
reload(rdl)

features_path = '/Users/dedan/projects/master/data/features/features.json'
door_path = '/Users/dedan/projects/master/data/response_matrix.pckl'
data_path = '/Users/dedan/projects/master/'
results_path = '/Users/dedan/projects/master/results/summary/'
descriptor = 'ATOMCENTRED_FRAGMENTS'
glomeruli_id = 0
format = 'png'

door2id = json.load(open(os.path.join(data_path, 'data', 'door2id.json')))
features = json.load(open(features_path))
if os.path.exists(door_path):
    print 'loading from: %s' % door_path
    rm_dict = pickle.load(open(door_path))
    cas_numbers = rm_dict['cas_numbers']
    glomeruli = rm_dict['glomeruli']
    rm = rm_dict['rm']
else:
    cas_numbers, glomeruli, rm = rdl.load_response_matrix()
    rm_dict = {'cas_numbers': cas_numbers, 'glomeruli': glomeruli, 'rm': rm}
    pickle.dump(rm_dict, open(door_path, 'w'))


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


fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)
ax.bar(range(len(glomeruli)), np.sum(~np.isnan(rm), axis=0))
ax.set_xticks(np.arange(len(glomeruli)) + 1)
ax.set_xticklabels(glomeruli, rotation='45', ha='right')
ax.set_title('number of stimuli available for a glomerulus')
fig.savefig(os.path.join(results_path, 'stimuli_per_glomerulus.' + format))

# use a good glomerulus (many molecules available) for training and feature selection,

# number of molecules available for all glomeruli

# get the feature vectors for all available responses

