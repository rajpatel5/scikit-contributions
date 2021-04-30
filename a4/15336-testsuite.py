from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting import binning
import numpy as np
from scipy import sparse
import unittest

X = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
X = sparse.csr_matrix(X)
Y = [1]

class TestMethods(unittest.TestCase):

    # Test function to test compressing a sparse matrix
    def test_predicting_sparse_matrix(self):
        Y_expected = [1]

        clf = HistGradientBoostingClassifier()
        clf.fit(X, Y)
        Y_actual = clf.predict(X)

        # Error message in case if test case failed
        message = "Predict data is incorrect, should be [1]!"

        self.assertTrue(np.array_equal(Y_actual, Y_expected, equal_nan=True), message) 

    # Test function to test compressing a sparse matrix
    def test_compressing_sparse_matrix(self):
        Y_expected = [[1, 1, 1, 1]]

        compressed_X = binning._compress_matrix(X)
        Y_actual = compressed_X[0]

        # Error message in case if test case failed
        message = "Compressed matrix data is incorrect, should be [[1, 1, 1, 1]]!"

        self.assertTrue(np.array_equal(Y_actual, Y_expected, equal_nan=True), message) 

    # Test function to test decompressing a dense matrix
    def test_decompress_dense_matrix(self):
        compressed_X = binning._compress_matrix(X)
        x = compressed_X[0]
        row_indices = compressed_X[1]
        Y_expected = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])

        decompressed_X = binning._decompress_matrix(x, row_indices, X.shape)
        Y_actual = decompressed_X.toarray()

        # Error message in case if test case failed
        message = "Decompressed matrix data is incorrect, should be [[0, 1, 0, 1], [1, 0, 1, 0]]!"

        self.assertTrue(np.array_equal(Y_actual, Y_expected, equal_nan=True), message) 