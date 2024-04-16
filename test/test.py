import gs_dbscan as gsd
import unittest
import numpy as np
from numba import cuda
import deepcomparer as dc

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