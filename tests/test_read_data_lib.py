#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import unittest, sys, os
import master.libs.read_data_lib as rdl
import numpy as np

class TestFeatureLib(unittest.TestCase):

    def setUp(self):
        path = os.path.join(os.path.dirname(__file__), 'data', 'features')
        self.tmp_path = os.path.join(os.path.dirname(__file__), 'data')
        self.features = rdl.read_feature_csvs(path)

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
        features = rdl.remove_invalid_features(self.features)
        self.assertNotIn('C-005', features)

    def test_normalization(self):
        """normalize to zero mean and unit variance"""
        unnormed_features = rdl.remove_invalid_features(self.features)
        features = rdl.normalize_features(unnormed_features)['test_features']
        for fname in features:
            self.assertAlmostEqual(np.var(features[fname].values()), 1.0, 5)
            self.assertAlmostEqual(np.mean(features[fname].values()), 0, 5)

    def test_get_cas_numbers(self):
        """read the CAS numbers from the R package (rownames)"""
        csv_path = os.path.join(self.tmp_path, 'rm.csv')
        rdl.get_data_from_r(csv_path)
        cas_numbers, _, _ = rdl.load_response_matrix(csv_path)
        self.assertEqual(len(cas_numbers), 249)
        self.assertIn('89-78-1', cas_numbers)
        self.assertNotIn('solvent', cas_numbers)
        os.remove(csv_path)

    def test_get_data_from_r(self):
        """read the data from r and write it in a CSV file"""
        csv_path = os.path.join(self.tmp_path, 'rm.csv')
        self.assertFalse(os.path.exists(csv_path))
        rdl.get_data_from_r(csv_path)
        self.assertTrue(os.path.exists(csv_path))
        os.remove(csv_path)

    def test_get_response_matrix(self):
        """read the response matrix from the DoOR R package"""
        csv_path = os.path.join(self.tmp_path, 'rm.csv')
        rdl.get_data_from_r(csv_path)
        row_names, col_names, rm = rdl.load_response_matrix(csv_path)
        self.assertEqual(249, rm.shape[0])
        self.assertEqual(67, rm.shape[1])
        self.assertEqual(249, len(row_names))
        self.assertEqual(67, len(col_names))
        os.remove(csv_path)

    # TODO: make this a test
        #     data, avail_features = rdl.get_features_for_molids(features[descriptor], molids)
        # assert data.shape[0] == len(molids)
        # for i in np.where(np.sum(data, axis=1) == 0)[0]:
        #     assert not i in avail_features


if __name__ == '__main__':
    unittest.main()