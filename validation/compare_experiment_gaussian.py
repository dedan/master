#!/usr/bin/env python
# encoding: utf-8
"""
    we recomputed the IR spectra also using GAUSSIAN

    this script is to compare the results of GAMESS and GAUSSIAN
"""
import pickle, os, json
import numpy as np
import pylab as plt
import __builtin__

# settings
n_to_plot = 10

# load all the results
gaussian_spectra = '/Users/dedan/projects/master/results/soroban/large_base'
experimental_spectra = '/Users/dedan/projects/master/data/experimental_spectra.pckl'
door2id = json.load(open('/Users/dedan/projects/master/data/door2id.json'))
headers = json.load(open('/Users/dedan/projects/master/data/headers.json'))
gaussian = pickle.load(open(os.path.join(gaussian_spectra, 'parsed.pckl')))
gaussian_problems = json.load(open(os.path.join(gaussian_spectra, 'problems.json')))
gaussian_settings = json.load(open(os.path.join(gaussian_spectra, 'config.json')))
exp_spec = pickle.load(open(experimental_spectra))
scaling_factor = float(headers[gaussian_settings['header'] + '_factor'])

print 'gaussian_problems', sorted(gaussian_problems)

# normalize to one
for mol in gaussian:
    for dim in ['ir', 'raman']:
        gaussian[mol][dim] = np.array(gaussian[mol][dim]) / np.max(np.array(gaussian[mol][dim]))

# select all molecules for which we have a cas number and experimental spectra
all_with_cas = __builtin__.sum(door2id.values(), [])
gauss_with_cas = set(gaussian.keys()).intersection(all_with_cas)
experimental_molids = [str(k) for k in exp_spec]
with_experimental = list(gauss_with_cas.intersection(experimental_molids))

plt.close('all')
for i in range(n_to_plot):
    molid = with_experimental[i]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(scaling_factor * np.array(gaussian[molid]['freq']),
           gaussian[molid]['ir'], edgecolor='b', width=3, label='gaussian')
    plt.hold(True)
    ax.bar(exp_spec[int(molid)]['wavenumber'],
           -np.array(exp_spec[int(molid)]['spec']), edgecolor='r', width=3, label='exp')
    ax.set_yticklabels([])
    ax.set_title(molid)
plt.show()




