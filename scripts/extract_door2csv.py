#!/usr/bin/env python
# encoding: utf-8
"""
    extract only the Hallem dataset from the DoOR database (R module)

    I needed this to reproduce the results of Haddad 2008

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.rinterface import NARealType
from master.libs import read_data_lib as rdl
import os,glob

outpath = '/Users/dedan/projects/master/data/door_csvs'

importr('DoOR.function')
importr('DoOR.data')
load_data = robjects.r['loadRD']
load_data()

hallems = ['Or2a', 'Or7a', 'Or9a', 'Or10a', 'Or19a', 'Or22a', 'Or23a',
           'Or33b', 'Or35a', 'Or43a', 'Or43b', 'Or47a', 'Or47b', 'Or49b',
           'Or59b', 'Or65a', 'Or67a', 'Or67c', 'Or82a', 'Or85a', 'Or85b',
           'Or85f', 'Or88a', 'Or98a']
for glom in hallems:
    glom_data = robjects.r[glom]
    glom_data.to_csvfile(os.path.join(outpath, glom + '.csv'))

for fname in glob.glob(os.path.join(outpath, '*.csv')):
    content = open(fname).readlines()
    content[0] = ',' + content[0]
    open(fname, 'w').writelines(content)
