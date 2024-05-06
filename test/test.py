import src.gs_dbscan as gsd
import unittest
import numpy as np
from numba import cuda
import deepcomparer as dc
import ThrustRTC as trtc
from timeit import default_timer as timer
import cupy as cp
import matplotlib.pyplot as plt

class TestConstructingABMatrices(unittest.TestCase):

    def test_normally(self):
        pass # TODO

class TestFindDistances(unittest.TestCase):

    def test_normally(self):
        X = cp.array([
            [0, 1, 3],
            [1, 2, 0],
            [2, 0, 3],
            [3, 0, 1],
            [0, 0, 1]
        ])
       
        A = cp.array([
            [0, 3],
            [2, 5],
            [4, 1],
            [0, 7],
            [2, 1]
        ])

        B = cp.array([
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
        distances = gsd.find_distances(X, A, B)

        expected = cp.sqrt(cp.array([
            [11,  5, 14, 11,  0,  5],
            [ 9,  0, 11,  0, 14, 11],
            [ 5,  0,  5,  5,  8, 14],
            [ 9,  5,  0,  0,  9,  5],
            [ 9,  6,  5,  5,  0,  6]
        ], dtype=cp.float16))
        
        self.assertTrue(distances.shape == (X.shape[0], A.shape[1] * B.shape[1])) # Check shape is (N, 2*k*m)
        self.assertTrue(cp.array_equal(distances, expected))

        print(f"Cupy took {timer() - start} seconds")

    def test_normally_mnist(self):
        X, A, B = create_mock_mnist_dataset()
        n = X.shape[0]
        m = B.shape[1]
        k = A.shape[1] // 2

        start = timer()
        distances = gsd.find_distances(X, A, B)
        print(f"Test took {timer() - start} seconds")

        self.assertTrue(distances.shape == (n, 2*k*m))

class TestContructQueryVectorDegreeArray(unittest.TestCase):

    def test_normally(self):
        distances = np.array([
            [0, 1, 2, 3],
            [0, 2, 1, 0],
            [1, 8, 9, 11],
            [15, 2, 6, 7]
        ])
        eps = cp.array(2.1, dtype = distances.dtype)

        E = gsd.construct_query_vector_degree_array(distances, eps)

        expected = np.array([3, 4, 1, 1])

        self.assertTrue(dc.deep_compare(E, expected))

    def test_normally_mnist(self):
        X, A, B = create_mock_mnist_dataset()

        distances = gsd.find_distances(X, A, B)
        eps = cp.array(2.1, dtype=cp.float16)

        start = timer()
        E = gsd.construct_query_vector_degree_array(distances, eps) # Why does this take so long???

        print(f"Test took {timer() - start} seconds")

        # self.assertTrue(E.shape == (n,))

    def test_normally_large_mnist(self):
        X, A, B = create_mock_mnist_dataset(n=int(1e6))
        n = X.shape[0]

        distances = gsd.find_distances(X, A, B, alpha = 1)
        eps = cp.array(2.1, dtype=cp.float16)

        start = timer()

        E = gsd.construct_query_vector_degree_array(distances, eps)

        self.assertTrue(E.shape == (n,))

        print("Test took", timer() - start, "seconds")


class TestProcessQueryVectorDegreeArray(unittest.TestCase):

    def test_normally(self):

        E = cp.array([3, 4, 1, 1], dtype=cp.int32)
        V_expected = cp.array([0, 3, 7, 8])
        V = gsd.process_query_vector_degree_array(E)

        self.assertTrue(cp.array_equal(V, V_expected))


    def test_normally_mnist(self):
        X, A, B = create_mock_mnist_dataset()

        distances = gsd.find_distances(X, A, B)

        eps = cp.array(2.1, dtype = distances.dtype)
        E = gsd.construct_query_vector_degree_array(distances, eps)

        start = timer()

        V = gsd.process_query_vector_degree_array(E)

        print(f"Test took {timer() - start} seconds")

    @unittest.skip("Takes way too long")
    def test_normally_large_mnist(self):
        X, A, B = create_mock_mnist_dataset(n=int(1e6))

        distances = gsd.find_distances(X, A, B, alpha = 100)

        eps = cp.array(2.1, dtype = distances.dtype)
        E = gsd.construct_query_vector_degree_array(distances, eps)

        start = timer()

        V = gsd.process_query_vector_degree_array(E)

        start = timer()

        print(f"Test took {timer() - start} seconds")

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
        distances = gsd.find_distances(X, A, B)

        # Distances should be:
        # array([[11,  5, 14, 11,  0,  5],
        #        [ 9,  0, 11,  0, 14, 11],
        #        [ 5,  0,  5,  5,  8, 14],
        #        [ 9,  5,  0,  0,  9,  5],
        #        [ 9,  6,  5,  5,  0,  6]]) ** 2

        eps = cp.array(cp.sqrt(5.1), dtype=distances.dtype)

        E = gsd.construct_query_vector_degree_array(distances, eps)
        V = gsd.process_query_vector_degree_array(E)

        distances_d = cuda.to_device(distances)
        A_d = cuda.to_device(A)
        B_d = cuda.to_device(B)
        V_d = cuda.to_device(V)
    
        # Figured out adj myself by using B[A]
        expected_adj_list = np.array([2, 0, 2, 1, 1, 0, 2, 3, 0, 2, 3, 3, 2, 0, 0, 4])
        adjacency_list = cp.full(shape=(int(E[-1] + V[-1])), fill_value=-1, dtype=cp.int32)
        adjacency_list_d = cuda.to_device(adjacency_list)

        gsd.construct_adjacency_list_for_query_vector[1, 5](distances_d, adjacency_list_d, V_d, A_d, B_d, eps)
        adjacency_list = adjacency_list_d.copy_to_host()

        self.assertTrue(cp.array_equal(adjacency_list, expected_adj_list))


class TestAssembleAdjacencyList(unittest.TestCase):

    def test_normally(self):
        X = cp.array([
            [0, 1, 3],
            [1, 2, 0],
            [2, 0, 3],
            [3, 0, 1],
            [0, 0, 1]
        ])

        A = cp.array([
            [0, 3],
            [2, 5],
            [4, 1],
            [0, 7],
            [2, 1]
        ])

        B = cp.array([
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

        eps = cp.array(2, dtype = distances.dtype)
        distances = gsd.find_distances(X, A, B)
        E = gsd.construct_query_vector_degree_array(distances, eps)
        V = gsd.process_query_vector_degree_array(E)

        adj_list = gsd.assemble_adjacency_list(distances, E, A, B, V, eps)

        expected_adj_list = np.array([2, 0, 2, 1, 1, 0, 2, 3, 0, 2, 3, 3, 2, 0, 0, 4])

        self.assertTrue(np.array_equal(adj_list, expected_adj_list))

    def test_normally_large_mats(self):
        X, A, B = create_mock_mnist_dataset()

        distances = gsd.find_distances(X, A, B)
        eps = cp.array(2.1, dtype = distances.dtype)

        E = gsd.construct_query_vector_degree_array(distances, eps)
        V = gsd.process_query_vector_degree_array(E)

        start = timer()

        adj_list = gsd.assemble_adjacency_list(distances, E, A, B, V, eps)

        print(f"Test took {timer() - start} seconds")

class TestContructClusterGraph(unittest.TestCase):

    pass # TODO

class TestPerformBFS(unittest.TestCase):

    pass # TODO

def create_mock_mnist_dataset(n=70_000, d= 784, m = 2000, k = 2, D = 1024):
    """
    Creates a mock MNIST dataset for testing purposes, along with accompanying A, B matrices

    Args:
        n: size of dataset
        d: dimensionality of dataset
        m: Number of query vectors per random vector
        k: number of random vectors per dataset vector
        D: number of random vectors

    Returns:
        X, A, B: dataset, A matrix, B matrix
    """
    X = cp.random.randint(size=(n, d), low=0, high=255, dtype=cp.int16) / 255
    X = X.astype(cp.float16)#  Since eps is only 2DP, float16 will do
    A = cp.random.randint(size=(n, 2*k), low=0, high=2*(D-1), dtype=cp.uint32) # uint32 goes up to 4 billion, more than enough for max n
    B = cp.random.randint(size=(2*D, m), low=0, high=n-1, dtype=cp.uint32)
    return X, A, B
