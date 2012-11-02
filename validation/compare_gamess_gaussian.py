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

gamess_spectra = '/Users/dedan/projects/master/data/spectra/gamess_am1'
gaussian_spectra = '/Users/dedan/projects/master/data/spectra/gaussian_am1'

door2id = json.load(open('/Users/dedan/projects/master/data/door2id.json'))
gamess = pickle.load(open(os.path.join(gamess_spectra, 'parsed.pckl')))
gaussian = pickle.load(open(os.path.join(gaussian_spectra, 'parsed.pckl')))
gamess_problems = json.load(open(os.path.join(gamess_spectra, 'problems.json')))
gaussian_problems = json.load(open(os.path.join(gaussian_spectra, 'problems.json')))

plt.close('all')

print 'keys'
print len(set(gamess.keys()).difference(set(gaussian.keys())))
print sorted(list(set(gamess.keys()).difference(set(gaussian.keys()))))
print

print 'gamess_problems', sorted(gamess_problems.keys())
print 'gaussian_problems', sorted(gaussian_problems)
print len(set(gamess_problems.keys()).difference(set(gaussian_problems)))
print set(gamess_problems.keys()).difference(set(gaussian_problems))
print set(gamess_problems.keys()).intersection(set(gaussian_problems))

for mol in gamess:
    for dim in ['ir']:
        gamess[mol][dim] = np.array(gamess[mol][dim]) / np.max(np.array(gamess[mol][dim]))
for mol in gaussian:
    for dim in ['ir', 'raman']:
        gaussian[mol][dim] = np.array(gaussian[mol][dim]) / np.max(np.array(gaussian[mol][dim]))

all_with_cas = __builtin__.sum(door2id.values(), [])
for_both = set(gamess.keys()).intersection(set(gaussian.keys()))
for_both_with_cas = list(for_both.intersection(all_with_cas))
for i in range(10):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    molid = for_both_with_cas[i]
    ax.bar(gaussian[molid]['freq'], gaussian[molid]['ir'], edgecolor='b', width=3)
    plt.hold(True)
    ax.bar(gamess[molid]['freq'], -np.array(gamess[molid]['ir']), edgecolor='r', width=3)
    ax.set_yticklabels([])
    ax.set_title(molid)






