#!/usr/bin/env python
# encoding: utf-8
"""
    file to read the results produced by by GAUSSIAN that I run on soroban
"""

import pickle, os, glob, json
from collections import defaultdict

inpath = '/Users/dedan/projects/master/results/soroban/tmp/am1'
n_lines_after_warning = 0
outdict = defaultdict(dict)
problem_files = []

outfiles=glob.glob(os.path.join(inpath, '*.log'))
for outfilename in outfiles:

    print 'reading from: ', outfilename
    freq, ir, raman = [], [], []
    outfile = open(outfilename)
    molid = os.path.splitext(os.path.basename(outfilename))[0]

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
        outdict[molid]['freq'] = freq
        outdict[molid]['ir'] = ir
        outdict[molid]['raman'] = raman
    else:
        problem_files.append(outfilename)

pickle.dump(dict(outdict), open(os.path.join(inpath, 'parsed.pckl'),'w'))
json.dump(dict(problem_files), open(os.path.join(inpath, 'problems.json'),'w'))

