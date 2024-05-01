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
            [0, 3],
            [2, 5],
            [4, 1],
            [0, 7],
            [2, 1]
        ])

        B = np.array([
            [1, 2, 3],
            [0, 4, 1],
            [3, 1, 0],
            [1, 0, 2],
            [0, 2, 3],
            [1, 2, 0],
            [0, 4, 1],
            [3, 1, 2],
            [1, 0, 4],
            [0, 2, 1]
        ])

        start = timer()
        distances = gsd.find_squared_distances(X, A, B)

        expected = np.array([
            [11,  5, 14, 11,  0,  5],
            [ 9,  0, 11,  0, 14, 11],
            [ 5,  0,  5,  5,  8, 14],
            [ 9,  5,  0,  0,  9,  5],
            [ 9,  6,  5,  5,  0,  6]
        ])
        
        self.assertTrue(distances.shape == (X.shape[0], A.shape[1] * B.shape[1])) # Check shape is (N, 2*k*m)
        self.assertTrue(dc.deep_compare(distances, expected))

        print(f"Cupy took {timer() - start} seconds")

    def test_normally_large_mats(self):
        X = cp.random.uniform(size=(10000, 50))
        A = cp.random.randint(size=(10000, 20), low=0, high=50-1)
        B = cp.random.randint(size=(50, 50), low=0, high=X.shape[0] - 1)

        start = timer()

        distances = gsd.find_squared_distances(X, A, B)
        self.assertTrue(distances.shape==(X.shape[0], A.shape[1] * B.shape[1]))

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

        self.assertTrue(cp.array_equal(V, V_expected))


class TestConstructAdjacencyListForQueryVector(unittest.TestCase):

    def test_normally(self):

        X = cp.array([
            [0, 1, 3],
            [1, 2, 0],
            [2, 0, 3],
            [3, 0, 1],
            [0, 0, 1]
        ])
       
        A = np.array([
            [0, 3],
            [2, 5],
            [4, 1],
            [0, 7],
            [2, 1]
        ])

        B = np.array([
            [1, 2, 3],
            [0, 4, 1],
            [3, 1, 0],
            [1, 0, 2],
            [0, 2, 3],
            [1, 2, 0],
            [0, 4, 1],
            [3, 1, 2],
            [1, 0, 4],
            [0, 2, 1]
        ])
        distances = gsd.find_squared_distances(X, A, B)

        # Distances should be:
        # array([[11,  5, 14, 11,  0,  5],
        #        [ 9,  0, 11,  0, 14, 11],
        #        [ 5,  0,  5,  5,  8, 14],
        #        [ 9,  5,  0,  0,  9,  5],
        #        [ 9,  6,  5,  5,  0,  6]])

        eps_squared = 5.1

        E = gsd.construct_query_vector_degree_array(distances, eps_squared)
        V = gsd.process_query_vector_degree_array(E)

        distances_d = cuda.to_device(distances)
        A_d = cuda.to_device(A)
        B_d = cuda.to_device(B)
        V_d = cuda.to_device(V)
    
        # Figured out adj myself by using B[A]
        expected_adj_list = np.array([2, 0, 2, 1, 1, 0, 2, 3, 0, 2, 3, 3, 2, 0, 0, 4])
        adjacency_list = cp.full(shape=(int(E[-1] + V[-1])), fill_value=-1, dtype=cp.int32)
        adjacency_list_d = cuda.to_device(adjacency_list)

        gsd.construct_adjacency_list_for_query_vector[1, 5](distances_d, adjacency_list_d, V_d, A_d, B_d, eps_squared)
        adjacency_list = adjacency_list_d.copy_to_host()

        self.assertTrue(cp.array_equal(adjacency_list, expected_adj_list))


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