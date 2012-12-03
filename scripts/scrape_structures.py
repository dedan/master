#!/usr/bin/env python
# encoding: utf-8
"""
    get structure drawing images form chemicalize

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import os
import pybel
import urllib2

mol_file = '/Users/dedan/projects/master/data/molecules.sdf'
out_path = '/Users/dedan/projects/master/data/structures/'
chem_url = 'http://www.chemicalize.org/tomcat-files/imggen.jsp?mol={}'


print('reading files from: {}'.format(mol_file))
molecules = pybel.readfile('sdf', mol_file)
opener = urllib2.build_opener()
for i, mol in enumerate(molecules):

    # read molid from database entry
    molid = mol.data['CdId']
    f_name = os.path.join(out_path, molid + '.png')

    if not os.path.exists(f_name):
        data_str = mol.write().split('\t')[0]
        url = chem_url.format(data_str)
        try:
            pic = opener.open(url).read()
            with open(f_name, 'wb') as f:
                f.write(pic)
        except Exception, e:
            print url, '400'
