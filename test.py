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


class TestCompareDistances(unittest.TestCase):

    def test_normally(self):
        C = np.full((8, 6), -1)
        E = np.zeros((8,), dtype=np.int32)
        R = np.array([
            [0, 1, 2, 3, 4, 1],
            [5, 4, 3, 2, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [2, 1, 1, 0, 2, 0],
            [0, 5, 0, 4, 2, 0],
            [0, 2, 0, 0, 0, 5],
            [0, 0, 3, 0, 3, 0],
            [0, 3, 0, 5, 0, 0]
        ])
        I = np.array([
            [0, 1, 2, 3, 4, 5],
            [7, 5, 6, 4, 3, 2],
            [0, 0, 0, 1, 0, 0],
            [2, 1, 1, 0, 2, 0],
            [0, 5, 0, 4, 2, 8],
            [0, 2, 0, 0, 0, 5],
            [6, 0, 3, 0, 3, 0],
            [0, 3, 0, 5, 0, 0]
        ])

       

        # Getting into the habit of memory copying manually is apparently faster see https://numba.pydata.org/numba-doc/0.13/CUDAJit.html
        C_d = cuda.to_device(C)
        E_d = cuda.to_device(E)
        R_d = cuda.to_device(R)
        I_d = cuda.to_device(I)
    
        n = 8
        k = 1
        m = 3
        eps_squared = 2

        gsd.compare_distances[2*n, k*m](C_d, E_d, R_d, I_d, eps_squared)

        C = C_d.copy_to_host()
        E = E_d.copy_to_host()
    
        # Gives list when copying back, but we want numpy array
        C = np.array(C)
        E = np.array(E)         
        print(C)
        print(E)

        C_non_empty = np.sum(C != -1, axis=1)
        self.assertTrue(np.all(C_non_empty == E))

        C_expected = np.full_like(C, -1)       
        C_expected[R < eps_squared] = I[R < eps_squared]
        self.assertTrue(np.all(C_expected == C))