from numba import cuda
import numpy as np
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

@cuda.jit
def construct_I_matrix(I, A, B):
    """
    Constructs the I matrix. 

    When using, set a grid size to be the twice the number of query vectors.

    In essence the I matrix stores the vector indices for the comparisons between query and candidate vectors. The block size should be k*m. These both assume flat grids and blocks.

    The idea of this kernel is that each block (with indices of the form 2*i+a) will closest vectors to closest random vectors if a is 0 and furthest vectors to furthest random vectors if a is 1.

    Matrices A and B are passed in since they contain the indices of the top k closest and furthest points to the query and random vectors respectively. This can't exactly be done in this method

    Args:
        I: result matrix to store the pairs of (q_idx, x_idx). Has shape (dataset_size, 2*k*m)
        X: point dataset
        A: A matrix as described in notes. Has shape (2*dataset_size, k)
        B: B matrix as described in notes. Has shape (2*num_random_vecs, m)    

    """
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    k = A.shape[1]
    m = B.shape[1]

    A_col = thread_idx // m
    B_col = thread_idx % m
    
    q_idx = block_idx // 2

    A_entry = A[block_idx][A_col]
    B_entry = B[A_entry][B_col]

    if not (block_idx & 1): # Use LSB to find if even, not modulo
        # Store closest candidates to closest random vecs in the first half of I
        I[q_idx][thread_idx] = B_entry
    else:
        # Store furthest candidates to furthest random vecs in the second half of I
        I[q_idx][k*m + thread_idx] = B_entry

def main_algorithm(I, X, batch_size=500):
    """TODO"""
    pass


# TODO complete this
# @cuda.jit
# def construct_A_B_matrices(X, D, A, B, projections, k, m):
#     pass

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
def main_kernel(A, B, X, D, C, eps, start_row, start_col, num_random_vecs):
    """
    Main cuda kernel for the alpha algorithm

    Yes this function has a lot of parameters, not much I can do about it :/. 

    TODO Remove superfluous parameters that can be calculated from the others

    Args:
        A: A matrix as described in notes. Has shape (2*dataset_size, k)
        B: B matrix as described in notes. Has shape (2*num_random_vecs, m)
        C: C matrix as described in notes. Has shape (dataset_size, 2*k*m)
        D: D vector as described in notes. Has (dataset_size,)
        X: point dataset stored along the ROWS. Has shape (dataset_size, dataset_dim)
        D: random vectors stored along the COLS. Has shape (random_vecs_dim, num_random_vecs)
        eps: Cluster radius tolerance
        start_row: The row that this grid starts at within A
        start_col: The column that this grid starts at within A
    """
    block_idx = cuda.blockIdx.x # Block index within the grid. Grid is flat to simplify things
    thread_idx = cuda.threadIdx.x # Thread index within the block. Block is flat to simplify things

    A_size = A.shape[0]
    k = A.shape[1]
    dataset_dim = X.shape[1]

    A_row = start_row
    A_col = start_col + block_idx

    if A_col >= k:
        q = A_col // k
        A_row += q
        A_col %= k
    
    if A_row >= A_size:
        return
    
    x_idx = B[A[A_row][A_col]][thread_idx]
    q_idx = A_row // 2

    # TODO idea. Use one kernel to calculate x_idx, q_idx

    distance = euclidean_dist_main_helper(X, dataset_dim, q_idx, x_idx=x_idx)

    if distance < eps:
        C[q_idx][D[q_idx]] = x_idx
        D[q_idx] += 1

def euclidean_dist_main_helper(X, dataset_dim, q_idx, x_idx):
    """
    Helper method to calculate the euclidean distance between two points in the dataset. Specialised to help the main kernel

    Args:
        X: The dataset
        dataset_dim: The dimension of the dataset
        q_idx: The index of the query vector in the dataset. I
        x_idx: The index of the vector to be figured out if it is within eps range of the query vector

    Returns:
        The euclidean distance between the two points. 
    """

    q_minus_x = cuda.local.array(shape=(dataset_dim,), dtype=numba.float32) 

    minus_across_mat_row(X, q_idx, x_idx, q_minus_x)

    q_minus_x_squared = cuda.local.array(shape=(dataset_dim,), dtype=numba.float32)

    square_vector[1,  ](q_minus_x, q_minus_x_squared)

    summed = cuda.local.array(shape=(1,), dtype=numba.float32) # Need an array for atomic addition across threads.

    sum_vector(q_minus_x_squared, summed)

    summed_sqrt = math.sqrt(summed[0])

    return summed_sqrt


@cuda.jit
def process_i_j_vectors(X, i, j, ):
    """
    Combines minusing, squaring and summing in one kernel

     TODO complete me!
    Returns:

    """
    pass


@cuda.jit
def sum_vector(vec, result):
    """
    Sums the entires of a vector
    
    Args:
        vec: The vector to sum
        result: The result of the sum
    """
    tid = cuda.threadIdx.x
    cuda.atomic.add(result, 0, vec[tid])

@cuda.jit
def square_vector(vec, result):
    """
    Squares the entries of a vector. I.e. performs a dot product with itself

    Args:
        vec: The vector to square
        result: The result of the squared vector
    """
    thread_idx = cuda.threadIdx.x
    result[thread_idx] = vec[thread_idx] * vec[thread_idx]

@cuda.jit
def minus_across_mat_row(X, i, j, w):
    """
    Subtracts the ith row of X from the jth row of X

    Args:
        X: The matrix to subtract from
        i: The row to subtract from
        j: The row to subtract
        w: The resulting vector
    """
    thread_idx = cuda.threadIdx.x

    w[thread_idx] = X[i][thread_idx] - X[j][thread_idx] 


    





