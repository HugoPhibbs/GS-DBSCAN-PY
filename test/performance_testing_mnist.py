from test import create_mock_mnist_dataset
import src.gs_dbscan as gsd
from timeit import default_timer as timer
import cupy as cp

k = 2
m = 2000
n = 70_000
d = 784
D = 1024

X, A, B = create_mock_mnist_dataset(n=n, m=m, k=k, d=d, D=1024)

# start_t = timer()

# Finding distances
# start = timer()

distances = gsd.find_distances(X, A, B)

# print(f"Calculating distances took {timer() - start} seconds")

eps = cp.array([11.5], dtype=distances.dtype)

# Using distances to calculate E
start = timer()

cp.sum(distances < eps[0], axis=1) # No idea why this takes 9 seconds, doing in terminals with random arrays is very very fast

# E = gsd.construct_query_vector_degree_array(distances, eps)  # Why does this take so long???

print(f"Creating E took {timer() - start} seconds")

distances_r = cp.random.random(size=distances.shape)
distances_r = distances_r.astype(dtype=distances.dtype)

eps = cp.array([0.5], dtype=distances.dtype)

start = timer()

cp.sum(distances < eps[0], axis=1) # Why is this thousands of times faster?

print(f"{timer() - start}")

# # Processing E to create V
# start = timer()
#
# V = gsd.process_query_vector_degree_array(E)
#
# print(f"Processing E took {timer() - start} seconds")
#
#
# # Adj list
# start = timer()
#
# adj_list = gsd.assemble_adjacency_list(distances, E, A, B, V, eps, block_size = 64) # Weird how its slower than the CPU?
#
# print(f"Assembling adjacency list took {timer() - start} seconds")
#
# print(f"Complete pipeline took {timer() - start_t} seconds")