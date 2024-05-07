from test import create_mock_mnist_dataset
import src.gs_dbscan as gsd
from timeit import default_timer as timer
import cupy as cp

X, A, B = create_mock_mnist_dataset()

distances = gsd.find_distances(X, A, B)
eps = cp.array([11.5], dtype=distances.dtype)

start = timer()
E = gsd.construct_query_vector_degree_array(distances, eps)  # Why does this take so long???

print(f"Test took {timer() - start} seconds")

# self.assertTrue(E.shape == (n,))