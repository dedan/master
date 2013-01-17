#!/usr/bin/env python
# encoding: utf-8
"""
plot the results obtained by validate_gen_score

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import os
import json
from master.libs import utils
import numpy as np
import pylab as plt

inpath = '/Users/dedan/projects/master/results/validation/gen_score_svr'

res = json.load(open(os.path.join(inpath, 'res.json')))

fig = plt.figure(figsize=(1, 5))
for i, glom in enumerate(res):
    ax = fig.add_subplot(len(res), 1, i+1)
    sorted_keys = sorted(res[glom], key=lambda k: int(k))
    values = [np.std(res[glom][key]) for key in sorted_keys]
    ax.bar(np.arange(len(values)), values)
    ax.set_yticks([0, ax.get_yticks()[-1]])
    if i == len(res) -1:
        ax.set_xticklabels(sorted_keys, rotation='90')
    else:
        ax.set_xticks([])
    ax.set_ylabel(glom, rotation='0')
fig.subplots_adjust(hspace=0.6)
plt.show()
