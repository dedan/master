#!/usr/bin/env python
# encoding: utf-8
"""
    do a randomization test on a feature_selection-preprocessing-model combination

    by shuffling the data within columns and the re-evaluating the result N times.
    The idea is that the result should be much worse for shuffled data because
    the observations should now be meaningless and not helpful to predict any
    target values. It would only perform equally well if the unshuffled
    observations were already meaningless (without target related structure)

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import json
import sys
import os
from master.libs import run_lib
import pylab as plt
reload(run_lib)

plt.close('all')
n_repetitions = 100
outpath = '/Users/dedan/projects/master/results/validations/'
name = 'svr_lin_saito'
glomeruli = ["Or19a", "Or22a", "Or35a", "Or43b", "Or67a", "Or67b", "Or7a", "Or85b", "Or98a", "Or9a"]
config = json.load(open(sys.argv[1]))

fig = plt.figure()
for i, glom in enumerate(glomeruli):

    config['glomerulus'] = glom
    rand_res, true_res = run_lib.randomization_test(config, n_repetitions)

    # create a plot of the resulting distribution and the original value
    ax = fig.add_subplot(len(glomeruli), 1, i+1)
    ax.hist(rand_res)
    ax.plot([true_res], [1], 'r*')
    ax.set_ylabel(glom)
    ax.set_xlim([-2, 0.8])
fig.subplots_adjust(h_space=0.3)
fig.savefig(os.path.join(outpath, name + '.png'))
plt.show()

