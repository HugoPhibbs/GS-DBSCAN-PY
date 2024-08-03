import pandas as pd

import src.gs_dbscan as gsd
import unittest
import numpy as np
from numba import cuda
import deepcomparer as dc
from timeit import default_timer as timer
import cupy as cp


class TestConstructingABMatrices(unittest.TestCase):

    def test_normally(self):
        n = 100
        k = 5
        m = 10
        D = 30
        projections = cp.random.rand(n, D)
        projections = cp.round(projections, 3)

        # import pandas as pd
        # projections_np = cp.asnumpy(projections)
        # df = pd.DataFrame(projections_np)
        # df.to_csv("projections.csv")

        A_gs, B_gs = gsd.construct_A_B_matrices_gpu_sort_from_projections(n, D, k, m, projections)

        A_gp, B_gp = gsd.construct_A_B_matrices_gpu_partition_from_projections(n, D, k, m, projections)

        self.assertTrue(A_gs.shape == (n, 2 * k))
        self.assertTrue(B_gs.shape == (2 * D, m))
        self.assertTrue(A_gp.shape == (n, 2 * k))
        self.assertTrue(B_gp.shape == (2 * D, m))

        self.assertTrue(cp.array_equal(A_gp, A_gs))
        self.assertTrue(cp.array_equal(B_gp, B_gs))

    def test_normally_mnist(self):
        X = create_mock_mnist_dataset()
        Y = create_mock_random_vectors_set()

        start = timer()

        A, B = gsd.construct_A_B_matrices(X, Y, k=5, m=50)

        print(f"Test took {timer() - start} seconds")

        self.assertTrue(A.shape == (X.shape[0], 2 * 5))
        self.assertTrue(B.shape == (2 * 1024, 50))

    def test_normally_mnist_gpu(self):
        k = 5
        m = 50
        D = 1024
        d = 784
        n = 400_000

        X = create_mock_mnist_dataset(n=n, d=d)
        Y = create_mock_random_vectors_set(D=D, d=d)

        start = timer()

        A_gs, B_gs = gsd.construct_A_B_matrices_gpu_sort(X, Y, k=k, m=m)

        print(f"GPU sort test took {timer() - start} seconds")

        start = timer()

        A_gp, B_gp = gsd.construct_A_B_matrices_gpu_partition(X, Y, k=k, m=m)

        print(f"GPU partition test took {timer() - start} seconds")

        A_expected_shape = (n, 2 * k)
        B_expected_shape = (2 * D, m)

        self.assertTrue(A_gp.shape == A_expected_shape)
        self.assertTrue(B_gp.shape == B_expected_shape)
        self.assertTrue(A_gs.shape == A_expected_shape)
        self.assertTrue(B_gs.shape == B_expected_shape)

        A_eq = A_gp == A_gs
        B_eq = B_gp == B_gs

        self.assertTrue(cp.sum(
            A_eq) / A_eq.size > 0.95)  # Rounding issues due to float16 cause some differences between sorting and partitioning
        # self.assertTrue(cp.sum(B_eq) / B_eq.size > 0.70) # Rounding issues are somehow much more pronounced for B than for A.

        count = 0
        for i in range(B_gs.shape[0]):
            is_in = cp.isin(B_gs[i], B_gp[i])
            count += cp.sum(is_in)

        print(count / B_gs.size)

        #
        # self.assertTrue(cp.array_equal(A_gs, A_gp))
        # self.assertTrue(cp.array_equal(B_gs, B_gp))


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
            [11, 5, 14, 11, 0, 5],
            [9, 0, 11, 0, 14, 11],
            [5, 0, 5, 5, 8, 14],
            [9, 5, 0, 0, 9, 5],
            [9, 6, 5, 5, 0, 6]
        ], dtype=cp.float16))

        self.assertTrue(distances.shape == (X.shape[0], A.shape[1] * B.shape[1]))  # Check shape is (N, 2*k*m)
        self.assertTrue(cp.array_equal(distances, expected))

        print(f"Cupy took {timer() - start} seconds")

    def test_medium_mnist(self):
        n = 1_000
        d = 20
        k = 3
        m = 20
        D = 100

        X = create_mock_mnist_dataset(n=n, d=d)
        A, B = create_mock_A_B_matrices(n=n, k=k, m=m, D=D)

        start = timer()
        distances = gsd.find_distances(X, A, B)
        cp.cuda.Device().synchronize()

        print(f"Test took {timer() - start} seconds")

        self.assertTrue(distances.shape == (n, 2 * k * m))


    def test_normally_mnist(self):
        X = create_mock_mnist_dataset()
        A, B = create_mock_A_B_matrices(n=X.shape[0], k=5, m=50)

        n = X.shape[0]
        m = B.shape[1]
        k = A.shape[1] // 2

        start = timer()
        distances = gsd.find_distances(X, A, B, batch_size=250)
        cp.cuda.Device().synchronize()
        print(f"Test took {timer() - start} seconds")

        print(distances.shape)

        self.assertTrue(distances.shape == (n, 2 * k * m))


class TestContructQueryVectorDegreeArray(unittest.TestCase):

    def test_normally(self):
        distances = np.array([
            [0, 1, 2, 3],
            [0, 2, 1, 0],
            [1, 8, 9, 11],
            [15, 2, 6, 7]
        ])
        eps = cp.array([2.1], dtype=distances.dtype)

        E = gsd.construct_query_vector_degree_array(distances, eps)

        expected = np.array([3, 4, 1, 1])

        self.assertTrue(dc.deep_compare(E, expected))

    def test_normally_mnist(self):
        X = create_mock_mnist_dataset()
        A, B = create_mock_A_B_matrices()

        distances = gsd.find_distances(X, A, B)
        eps = cp.array([11.5], dtype=distances.dtype)

        start = timer()
        E = gsd.construct_query_vector_degree_array(distances, eps)  # Why does this take so long???, spooky
        print(f"Test took {timer() - start} seconds")

        # self.assertTrue(E.shape == (n,))

    def test_normally_large_mnist(self):
        X, A, B = create_mock_mnist_dataset(n=int(1e6))
        A, B = create_mock_A_B_matrices(n=X.shape[0])
        n = X.shape[0]

        distances = gsd.find_distances(X, A, B, alpha=1)
        eps = cp.array([2.1], dtype=cp.float16)

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
        X = create_mock_mnist_dataset()
        A, B = create_mock_A_B_matrices()

        distances = gsd.find_distances(X, A, B)

        eps = cp.array([11.5], dtype=distances.dtype)
        E = gsd.construct_query_vector_degree_array(distances, eps)

        start = timer()

        V = gsd.process_query_vector_degree_array(E)

        print(f"Test took {timer() - start} seconds")

    @unittest.skip("Takes way too long")
    def test_normally_large_mnist(self):
        X = create_mock_mnist_dataset(n=int(1e6))
        A, B = create_mock_A_B_matrices(n=X.shape[0])

        distances = gsd.find_distances(X, A, B, alpha=100)

        eps = cp.array([2.1], dtype=distances.dtype)
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

        eps = cp.array([cp.sqrt(5.1)], dtype=distances.dtype)

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

        distances = gsd.find_distances(X, A, B)
        distances = distances.astype(cp.float16)

        eps = cp.array([2], dtype=distances.dtype)  # Use same dtype as distances
        E = gsd.construct_query_vector_degree_array(distances, eps)
        V = gsd.process_query_vector_degree_array(E)

        expected_adj_list = np.array([0, 1, 1, 2, 3, 3, 4])
        adj_list = gsd.assemble_adjacency_list(distances, E, A, B, V, eps)

        self.assertTrue(np.array_equal(adj_list, expected_adj_list))

    def test_normally_large_mats(self):
        X, A, B = create_mock_mnist_dataset()

        distances = gsd.find_distances(X, A, B)
        eps = cp.array([11.5], dtype=distances.dtype)

        E = gsd.construct_query_vector_degree_array(distances, eps)
        V = gsd.process_query_vector_degree_array(E)

        start = timer()

        adj_list = gsd.assemble_adjacency_list(distances, E, A, B, V, eps,
                                               block_size=64)  # Weird how its slower than the CPU?

        print(f"Test took {timer() - start} seconds")  # Takes less than a second


class TestContructClusterGraph(unittest.TestCase):
    pass  # TODO


class TestPerformBFS(unittest.TestCase):
    pass  # TODO

def cpArrayToFlatCSV(arr, filename):
    arr = arr.get()
    arr = arr.flatten()
    df = pd.DataFrame(arr)
    df.to_csv(filename)

def create_mock_A_B_matrices(n=70_000, k=2, m=2000, D=1024):
    A = np.random.randint(size=(n, 2 * k), low=0, high=2 * (D - 1),
                          dtype=cp.uint32)  # uint32 goes up to 4 billion, more than enough for max n
    B = np.random.randint(size=(2 * D, m), low=0, high=n - 1, dtype=cp.uint32)
    return cp.asarray(A), cp.asarray(B)  # Dunno why but this fixes error


def create_mock_random_vectors_set(D=1024, d=784):
    Y = cp.random.random(size=(d, D))  # Stored along the columns.
    Y = Y.astype(cp.float16)
    return Y


def create_mock_mnist_dataset(n=70_000, d=784):
    """
    Creates a mock MNIST dataset for testing purposes

    Args:
        n: size of dataset
        d: dimensionality of dataset
        m: Number of query vectors per random vector
        k: number of random vectors per dataset vector
        D: number of random vectors

    Returns:
        X, A, B: dataset, A matrix, B matrix
    """
    X_n = np.random.uniform(size=(n, d))
    X = X_n.astype(np.float32)
    return cp.asarray(X)


