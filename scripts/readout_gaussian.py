#!/usr/bin/env python
# encoding: utf-8
"""
    file to read the results produced by by GAUSSIAN that I run on soroban
"""

import pickle, os, glob, json
from collections import defaultdict

inpath = '/Users/dedan/projects/master/results/spectra/gaussian_am1'
n_lines_after_warning = 0
outdict = defaultdict(dict)
problem_files = []

outfiles=glob.glob(os.path.join(inpath, '*.log'))
for outfilename in outfiles:

    freq, ir, raman = [], [], []
    outfile = open(outfilename)
    molid = os.path.splitext(os.path.basename(outfilename))[0]
    molid = os.path.splitext(molid)[0]

    ok = False
    for line in outfile:

        # collect frequencies and intensities
        if 'Frequencies' in line:
            freq.extend(line.split()[2:])
        if 'IR Inten' in line:
            ir.extend(line.split()[3:])
        if 'Raman Activ' in line:
            raman.extend(line.split()[3:])
        if 'Normal termination' in line:
            ok = True

    assert len(freq) == len(ir)
    assert len(freq) == len(raman)
    if ok:
        outdict[molid]['freq'] = [float(f) for f in freq]
        outdict[molid]['ir'] = [float(f) for f in ir]
        outdict[molid]['raman'] = [float(f) for f in raman]
    else:
        problem_files.append(molid)

pickle.dump(dict(outdict), open(os.path.join(inpath, 'parsed.pckl'),'w'))
json.dump(problem_files, open(os.path.join(inpath, 'problems.json'),'w'))

