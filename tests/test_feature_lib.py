#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import unittest, sys, os
import master.features.feature_lib as fl
import numpy as np

class TestFeatureLib(unittest.TestCase):

    def setUp(self):
        path = os.path.join(os.path.dirname(__file__), 'data', 'features')
        self.features = fl.read_feature_csvs(path)

    def test_feature_set_loaded(self):
        """name of the CSV file should be key in the features dict"""
        self.assertIn('test_features', self.features)

    def test_error_removed(self):
        """for some molecules the descriptor could not be computed

        this is marked by *Error* in the identifiere column and this molecules
        should not be in the result of read_feature_csvs
        """
        test_features = self.features['test_features']
        for feature_name in test_features:
            self.assertNotIn('297', test_features[feature_name])

    def test_invalid_column_removed(self):
        """features with zero variance should be removed"""
        self.assertIn('C-005', self.features['test_features'])
        features = fl.remove_invalid_features(self.features)
        self.assertNotIn('C-005', features)

    def test_normalization(self):
        """normalize to zero mean and unit variance"""
        unnormed_features = fl.remove_invalid_features(self.features)
        features = fl.normalize_features(unnormed_features)['test_features']
        for fname in features:
            self.assertAlmostEqual(np.var(features[fname].values()), 1.0, 5)
            self.assertAlmostEqual(np.mean(features[fname].values()), 0, 5)


if __name__ == '__main__':
    unittest.main()