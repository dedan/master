#!/usr/bin/env python
# encoding: utf-8
"""
This script reads data from all the csv files and save them in a format
convenient for analysis.

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import sys, os, json
import master.libs.read_data_lib as rdl

data_path = '/Users/dedan/projects/master/data/'

# TODO: make it work with the new read_feature_csv function
features = rdl.read_feature_csvs(os.path.join(data_path, 'features'))
features = rdl.normalize_features(rdl.remove_invalid_features(features))
json.dump(features, open(os.path.join(data_path, 'features.json'), 'w'))

rdl.get_data_from_r(os.path.join(data_path, 'response_matrix.csv'))
