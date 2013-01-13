import os
import glob
import json
import numpy as np
import pylab as plt
from sklearn.metrics import r2_score


base_path = '/Users/dedan/projects/master/results/new_param_search'
input_folders = ['test', 'conv_newxval']

res2comp = [{}, {}]
for i, input_folder in enumerate(input_folders):

    files = glob.glob(os.path.join(base_path, input_folder, '*.json'))
    for fname in files:
        res = json.load(open(fname))['res']
        # TODO: treat the two selection methods differently?
        for sel in res:
            for glom in res[sel]:
                for k_best in res[sel][glom]:
                    for reg in res[sel][glom][k_best]:
                        key = '{}_{}_{}_{}_{}'.format(os.path.basename(fname), sel, glom, k_best, reg)
                        res2comp[i][key] = res[sel][glom][k_best][reg]['forest']['gen_score']

print 'n scores for {}: {}'.format(input_folders[0], len(res2comp[0]))
print 'n scores for {}: {}'.format(input_folders[1], len(res2comp[1]))

avail_for_all = set(res2comp[0]).intersection(res2comp[1])
x_vals = [res2comp[0][entry] for entry in avail_for_all]
y_vals = [res2comp[1][entry] for entry in avail_for_all]

plt.plot(x_vals, y_vals, 'x')
plt.plot([0, 1], [0, 1], color='0.5')
plt.xlabel(input_folders[0])
plt.ylabel(input_folders[1])
plt.show()

