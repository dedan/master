#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import unittest, sys, os
import master.libs.read_data_lib as rdl
import master.libs.features_lib as flib
import numpy as np
reload(flib)

class TestLibs(unittest.TestCase):

    def setUp(self):
        path = os.path.join(os.path.dirname(__file__), 'data', 'features')
        self.tmp_path = os.path.join(os.path.dirname(__file__), 'data')
        self.features = flib.read_feature_csv(os.path.join(path, 'test_features.csv'))

    def test_conventional_feature_correct(self):
        """test at least the first line for correctness"""
        self.assertTrue(np.array_equal(np.array([2., 2., 1., 0., -999., 0., 0., 0.]),
                                       np.array(self.features.values())[0]))

    def test_error_removed(self):
        """for some molecules the descriptor could not be computed

        this is marked by *Error* in the identifiere column and this molecules
        should not be in the result of read_feature_csvs
        """
        for feature_name in self.features:
            self.assertNotIn('297', self.features[feature_name])

    def test_invalid_column_removed(self):
        """features with zero variance should be removed"""
        self.assertEqual(np.array(self.features.values()).shape[1], 8)
        features = flib.remove_invalid_features(self.features)
        self.assertEqual(np.array(self.features.values()).shape[1], 6)

    def test_normalization(self):
        """normalize to zero mean and unit variance"""
        unnormed_features = flib.remove_invalid_features(self.features)
        features = flib.normalize_features(unnormed_features)
        feature_mat = np.array(self.features.values())
        for i in range(feature_mat.shape[1]):
            self.assertAlmostEqual(np.var(feature_mat[:, i]), 1.0, 5)
            self.assertAlmostEqual(np.mean(feature_mat[:, i]), 0, 5)

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