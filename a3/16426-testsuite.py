from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import unittest

# Inputs
X = [[0, np.nan], [1, np.nan]]
Y = [[0, np.nan], [1, 1]]

# Expected output
Y_expected = [[0., np.nan], [ 1.,  1.]]

# Error message in case if test case failed
message = "Transform data is incorrect, should be [[0., np.nan], [ 1.,  1.]]!"

class TestMethods(unittest.TestCase):

    # Test function to test if nan values are not kept in SimpleImputer()
    def test_nan_values_not_kept_in_SimpleImputer(self):
        impute = SimpleImputer(remove_features=False)
        impute.fit(X)
        Y_actual = impute.transform(Y)

        self.assertTrue(np.array_equal(Y_actual, Y_expected, equal_nan=True), message) 

    # Test function to test if nan values are not kept in KNNImputer()
    def test_nan_values_not_kept_in_KNNImputer(self):
        impute = KNNImputer(remove_features=False)
        impute.fit(X)
        Y_actual = impute.transform(Y)

        self.assertTrue(np.array_equal(Y_actual, Y_expected, equal_nan=True), message) 

    # Test function to test if nan values are not kept in IterativeImputer()
    def test_nan_values_not_kept_in_IterativeImputer(self):
        impute = IterativeImputer(remove_features=False)
        impute.fit(X)
        Y_actual = impute.transform(Y)

        self.assertTrue(np.array_equal(Y_actual, Y_expected, equal_nan=True), message)
