#!/usr/bin/env python
# encoding: utf-8
"""

    this file will train models for different parameters and store
    them in pickles for later plotting

    * one pickle always contains a dict of models for different feature thresholds and glomeruli
    * each model is annotated with its settings
    * for each pickle we also write a json file containing the settings in a human readable format

    TODO: maybe make it work on job files in a folder (batch mode)

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""


"""
    !!! move everything in small functions, with many options this script might
    !!! become very complex later
"""
import sys
import os
import json
from master.libs import read_data_lib as rdl
reload(rdl)

# read from a config file, this might become a job file later
config = json.load(open(sys.argv[1]))

# feature related stuff
if config['features']['type'] == 'conventional':
    feature_file = os.path.join(config['data_path'],
                                'conventional_features',
                                config['features']['descriptor'] + '.csv')
    features = rdl.read_feature_csv(feature_file)
    features = rdl.remove_invalid_features(features)
    if config['features']['normalize']:
        features = rdl.normalize_features(features)





















