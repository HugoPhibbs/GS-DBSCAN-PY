import gs_dbscan as gsd
import unittest
import numpy as np
from numba import cuda
import deepcomparer as dc


class TestConstructingIMatrix(unittest.TestCase):
    # TODO Test edge cases, this function only does blue sky scenario - more of a proof it isn't completely messed up!

    def test_normally(self):
        # FTI example uses, n=2, k=4, m=5

        A = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [0, 1, 3, 3], [0, 1, 2, 3]])
        B = np.array([[17, 18, 19, 20, 21], [22, 23, 24, 25, 26], [27, 28, 29, 30, 31], [32, 33, 34, 35, 36]])

        A_d = cuda.to_device(A)
        B_d = cuda.to_device(B)

        n = 2
        k = 4
        m = 5

        I = np.zeros(shape=(2, 2 * k * m))

        I_d = cuda.to_device(I)

        gsd.construct_I_matrix[2 * n, k * m](I_d, A_d, B_d)

        I = I_d.copy_to_host()

        expected = [
            [22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 17.0, 18.0, 19.0,
             20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 17.0,
             18.0, 19.0, 20.0, 21.0],
            [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 32.0, 33.0, 34.0, 35.0, 36.0, 32.0, 33.0, 34.0,
             35.0, 36.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
             33.0, 34.0, 35.0, 36.0]]

        self.assertTrue(dc.deep_compare(I.tolist(), expected))


class TestConstructingABMatrices(unittest.TestCase):

    def test_normally(self):
        pass # TODO

