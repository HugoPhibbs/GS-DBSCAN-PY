from numba import cuda
import numpy as np
import heapq
import numba
from limited_priority_queue import LimitedPriorityQueue
import math

"""
Preprocessing step for s-dbscan. 

Equiv to algo 3 from paper

Args:
    X: point dataset
    D: random vectors
    k: k param
    m: m param
"""
def pre_processing(X, D, k, m):
    pass

"""
Finding core points and their neighbourhoods

Equiv to algo 4 from paper

Args:
    X: point dataset
    D: random vectors
    k: k param
    m: m param
    eps: cluster radius tolerance
"""
def core_points(X, D, k, m, eps):
    pass


def gs_dbscan(X, D, minPts, k, m, eps):
    """
    Main algo for s-dbscan

    Equiv to algo 1 from paper

    Args:
        X: point dataset
        D: random vectors
        minPts: Minimum number of points for a point to be clasified as a core point
        k: k param
        m: m param
        eps: Cluster radius for identifying core points
    """
    for q in X:
        product = D * q
        sorted_indices = product.arg

    pass

def construct_A_B_matrices(X, D, k, m):
    """
    Constructs the A and B matrices for the dataset and random vectors

    A has a shape of 2*len(X) x k. Each even row (row index of 2*i) of A is the indices of the top k closest points to the ith point in X. Each odd row (row index of form 2*i) of A is the indices of the top k furthest points to the ith point in X.

    B has a shape of 2*len(D) x m. Each even row (row index of 2*j) of B is the indices of the top m closest points to the jth random vector in D. Each odd row (row index of form 2*j) of B is the indices of the top m furthest points (in X) to the jth random vector in D.

    Args:
        X: point dataset A matrix of points stored along the ROWS
        D: random vectors A matrix of random vectors stored along the COLS
        k: k param k parameter as described above
        m: m param m parameter as described above

    Returns:
        A tuple of two matrices. The first matrix is the A matrix and the second matrix is the B matrix
    """
    product = dataset_random_projections(X, D)

    A = np.zeros(size=(len(X)*2, k))
    B = np.zeros(size=(len(D)*2, m))

    for j in range(len(A)):
        top_k, bottom_k = top_closest_and_furthest_indices(product[j], k)
        A[2*j] = top_k
        A[2*j+1] = bottom_k

    for j in range(len(B)):
        top_m, bottom_m = top_closest_and_furthest_indices(product[:, j], m)
        B[2*j] = top_m
        B[2*j+1] = bottom_m

    return A, B

def top_closest_and_furthest_indices(vector, n):
    """
    Finds the indices of the top k largest and smallest entries from a vector

    Args:
        vector: the vector as mentioned above
        k: k as mentioned

    Returns:
        A tuple of two lists. The first list is the top k closest indices and the second list is the top m furthest indices
    """
    top_heap = LimitedPriorityQueue(n)
    bottom_heap = LimitedPriorityQueue(n, reverse=True)
    for i in range(len(vector)):
        entry = vector[i]
        top_heap.push(i, entry)
        bottom_heap.push(i, entry)
    return top_heap.heap_items(), bottom_heap.heap_items()

def dataset_random_projections(X, D):
    """
    Computes the random projections of the dataset onto the random vectors

    Intend to make this faster in the future by using the FHT. For now its an overbloated method doing one line of code :D

    Args:
        X: point dataset A matrix of points stored along the ROWS
        D: random vectors. A matrix with random vectors stored along the COLS

    Returns:
        A matrix of the random projections of the dataset onto the random vectors. An entry (i, j) is the inner product of the ith dataset vector with the jth random vector
    """
    return np.dot(X, D)

def random_vecs(num, dim):
    """
    Creates a matrix with random values representing random vectors.

    Samples from the gaussian distribution

    Vectors are stored along the COLS of this matrix (not the rows)

    Args:
        num: Number of random vectors to create
        dim: Dimension of the random vectors to create
    Returns:
        A num*dim matrix
    """
    return np.random.normal(size=(num, dim))

@cuda.jit
def main_kernel(A, B, X, D, k, m, eps, start_row, start_col, n, d):
    block_idx = cuda.blockIdx.x # Block index within the grid. Grid is flat to simplify things
    thread_idx = cuda.threadIdx.x # Thread index within the block. Block is flat to simplify things

    row = start_row + block_idx

    if row >= n:
        row %= n
        col = start_col + 1
    else:
        col = start_col
    
    if col >= d:
        return
    
    q_minus_x = cuda.local.array(shape=(X.shape[0],), dtype=numba.float32) 

    vec_minus_across_row(X, row, B[col, thread_idx], q_minus_x)


    
    # q = X[row // 2] ̌# TODO, this is wrong, need to allocate array differently.
    # x = X[B[col, thread_idx]]

@cuda.jit
def vec_minus_across_row(X, i, j, w):
    """
    Subtracts the ith row of X from the jth row of X
    """
    thread_idx = cuda.threadIdx.x

    w[thread_idx] = X[i][thread_idx] - X[j][thread_idx] 

u = np.array([[1, 2, 3],[4, 5, 6]])
v = np.array([[7, 8, 9],[10, 11, 12]])

print(euclidean_distance(u, v))



    





