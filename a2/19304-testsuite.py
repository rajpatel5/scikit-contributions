from sklearn.ensemble import RandomForestRegressor
import unittest
import numpy as np

rf = RandomForestRegressor(criterion="poisson", random_state=10)

class TestMethods(unittest.TestCase):
    # Test function to test array with multiple positive values
    def test_multiple_positive(self):
        X = np.array([[1, 2, 3]]).T
        Y = [0, 1, 2]
        Y_expected = [0.82833333, 0.82833333, 1.40333333]

        rf.fit(X, Y)
        Y_actual = rf.predict(X)

        # Error message in case if test case got failed
        message = "rf.predict(X) data is incorrect, should be [0.82833333 0.82833333 1.40333333]!"
        self.assertTrue(np.allclose(Y_actual, Y_expected), message)

    # Test function to test array with multiple all zeros
    def test_multiple_all_zeros(self):
        X = np.array([[1, 2, 3]]).T
        Y = [0, 0, 0]
        Y_expected = [0., 0., 0.]

        rf.fit(X, Y)
        Y_actual = rf.predict(X)

        # Error message in case if test case got failed
        message = "rf.predict(X) data is incorrect, should be [0., 0., 0.]!"
        self.assertTrue(np.allclose(Y_actual, Y_expected), message)

    # Test function to test array with one negative value at the start
    def test_multiple_with_negative_at_start(self):
        X = np.array([[1, 2, 3]]).T
        Y = [-1, 2, 3]

        # Error message in case if test case got failed
        message = "rf.fit(X, Y) should raise a ValueError!"
        self.assertRaises(ValueError, lambda: rf.fit(X, Y))

    # Test function to test array with one negative value in the middle
    def test_multiple_with_negative_in_middle(self):
        X = np.array([[1, 2, 3]]).T
        Y = [1, -2, 3]

        # Error message in case if test case got failed
        message = "rf.fit(X, Y) should raise a ValueError!"
        self.assertRaises(ValueError, lambda: rf.fit(X, Y))

    # Test function to test array with one negative value at the end
    def test_multiple_with_negative_at_end(self):
        X = np.array([[1, 2, 3]]).T
        Y = [1, 2, -3]

        rf = RandomForestRegressor(criterion="poisson", random_state=10)

        # Error message in case if test case got failed
        message = "rf.fit(X, Y) should raise a ValueError!"
        self.assertRaises(ValueError, lambda: rf.fit(X, Y))


if __name__ == '__main__':
    unittest.main()

