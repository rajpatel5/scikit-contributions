from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
import unittest
import numpy as np

kernel = MiniSeqKernel(baseline_similarity_bounds='fixed')
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3)

class TestMethods(unittest.TestCase):
    # Test function to test multiple non-zero-values in y
    def test_multiple_non_zero_values(self):
        X = [[1.,2.],[2.,2.]]
        Y = [5, 10]
        Y_expected = [-1., 1.]

        Y_actual = gpr.fit(X, Y).y_train_

        # error message in case if test case got failed
        message = "gpr.fit(X,Y) data is incorrect, should be [-1., 1.]!"
        self.assertTrue(np.array_equal(Y_actual, Y_expected), message) 

    # Test function to test single non-zero value in y
    def test_single_non_zero_value(self):
        X = [[1., 2.]]
        Y = [5]
        Y_expected = [0.]

        Y_actual = gpr.fit(X, Y).y_train_

        # error message in case if test case got failed
        message = "gpr.fit(X,Y) data is incorrect, should be [0.]!"

        # self.assertEqual(Y_actual, Y_expected, message)
        self.assertTrue(np.array_equal(Y_actual, Y_expected), message) 

    # Test function to test multiple zero value in y
    def test_multiple_zero_values(self):
        X = [[1.,2.],[2.,2.]]
        Y = [0,0]
        Y_expected = [0., 0.]

        Y_actual = gpr.fit(X, Y).y_train_
        
        # error message in case if test case got failed
        message = "gpr.fit(X,Y) data is incorrect, should be [0., 0.]!"
        self.assertTrue(np.array_equal(Y_actual, Y_expected), message) 

    # Test function to test single zero value in y
    def test_single_zero_value(self):
        X = [[1., 2.]]
        Y = [0]
        Y_expected = [0.]

        Y_actual = gpr.fit(X, Y).y_train_

        # error message in case if test case got failed
        message = "gpr.fit(X,Y) data is incorrect, should be [0.]!"
        self.assertTrue(np.array_equal(Y_actual, Y_expected), message) 


if __name__ == '__main__':
    unittest.main()