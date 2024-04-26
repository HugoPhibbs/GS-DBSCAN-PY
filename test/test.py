import src.gs_dbscan as gsd
import unittest
import numpy as np
from numba import cuda
import deepcomparer as dc
import ThrustRTC as trtc
from timeit import default_timer as timer
import cupy as cp

class TestConstructingABMatrices(unittest.TestCase):

    def test_normally(self):
        pass # TODO

class TestSquareDistances(unittest.TestCase):

    def test_normally(self):
        X = np.array([
            [0, 1, 3],
            [1, 2, 0],
            [2, 0, 3],
            [3, 0, 1],
            [0, 0, 1]
        ])
       

        A = np.array([
            [0, 1],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 0]
        ])

        B = np.array([
            [1, 2],
            [0, 4],
            [3, 1],
            [1, 0],
            [0, 2]
        ])

        start = timer()
        distances_cp = gsd.find_squared_distances_cupy(X, A, B)

        print(f"Cupy took {timer() - start} seconds")

    def test_normally_large_mats(self):
        X = cp.random.uniform(size=(10000, 50))
        A = cp.random.randint(size=(10000, 20), low=0, high=50-1)
        B = cp.random.randint(size=(50, 50), low=0, high=X.shape[0] - 1)

        start = timer()

        gsd.find_squared_distances_cupy(X, A, B)

        print(f"Cupy took {timer() - start} seconds")


class TestContructQueryVectorDegreeArray(unittest.TestCase):

    def test_normally(self):
        distances = np.array([
            [0, 1, 2, 3],
            [0, 2, 1, 0],
            [1, 8, 9, 11],
            [15, 2, 6, 7]
        ])
        eps_squared = 2.1

        E = gsd.construct_query_vector_degree_array(distances, eps_squared)

        expected = np.array([3, 4, 1, 1])

        self.assertTrue(dc.deep_compare(E, expected))

class TestProcessQueryVectorDegreeArray(unittest.TestCase):

    def test_normally(self):

        E = cp.array([3, 4, 1, 1], dtype=cp.int32)
        V_expected = cp.array([0, 3, 7, 8])
        V = gsd.process_query_vector_degree_array(E)

        self.assertTrue(dc.deep_compare(V, V_expected))


class TestAssembleAdjacencyList(unittest.TestCase):

    def test_normally(self):
        V = np.array([0, 3, 7, 8])
        E = [3, 4, 1, 1]

        distances = np.array([
            [0, 1, 2, 3],
            [0, 2, 1, 0],
            [1, 8, 9, 11],
            [15, 2, 6, 7]
        ])

        adj_list = gsd.assemble_adjacency_list(V, E, distances)

        expected = []

        self.assertTrue(dc.deep_compare(adj_list, expected))


class TestContructClusterGraph(unittest.TestCase):

    pass # TODO

class TestPerformBFS(unittest.TestCase):

    pass # TODO