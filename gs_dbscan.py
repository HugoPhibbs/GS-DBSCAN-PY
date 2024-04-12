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

def main_algorithm(I, X, batch_size=500):
    """TODO"""
    pass

@cuda.jit
def construct_A_matrix():
    """
    Constructs the A matrix for the dataset

    A has a shape of 2*len(X) x k. Each even row (row index of 2*i) of A is the indices of the top k closest points to the ith point in X. Each odd row (row index of form 2*i) of A is the indices of the top k furthest points to the ith point in X.

    Each block is responsible for a single query vector. The block size should be k. The grid size should be len(X)
    """
    pass # Not sure how to implement this yet, not until I figure out how to do a priority Queue on the GPU

@cuda.jit
def construct_B_matrix():
    """
    Constructs the B matrix for the random vectors

    B has a shape of 2*len(D) x m. Each even row (row index of 2*j) of B is the indices of the top m closest points to the jth random vector in D. Each odd row (row index of form 2*j) of B is the indices of the top m furthest points to the jth random vector in D.

    Each block is responsible for a single random vector. The block size should be m. The grid size should be len(D)
    """
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
def process_i_j_vectors(X, I, R, start_i):
    """
    Combines subtraction, squaring and summing steos in one kernel. 

    Assumes that processing is done in batches (say 500 query vectors per grid at a time)

    Idea is that you organise the grid with query vectors along x direction, and candidate vectors along the y direction. This way you can process the distances between the query and candidate vectors in parallel.

    Each thread within a block handles a different entry of the query vectors to candidate vector. I.e the ith thread processes the ith entry of the query vector to the candidate vector.

    A single thread does the subtraction, squaring and summing for a single entry between a query and candidate vector, then adds it to the result matrix R.

    Effectively finds the euclidean distance squared between two vectors in the dataset.

    Args:
        X: The dataset containing the vectors, stored along the ROWS
        R: The matrix to store the result of the subtraction, squaring and summing. Has shape (batch_size, 2*k*m). Each entry should be atomic. And should be initialised with zero entries
        start_i: Starting index of the q'ith vector to start at in the dataset. This is added to allow for batching of this process, since it can't all be done at once
    Returns:
    """
    thread_idx = cuda.threadId.x # Entry within vectors to process
    block_idx_x = cuda.blockIdx.x # Query vector index
    block_idx_y = cuda.blockIdx.y # Candidate vector index

    R_cols = R.shape[1]

    i = start_i + block_idx_x
    j = I[i][block_idx_y]

    subtracted_entry = X[i][thread_idx] - X[j][thread_idx]
    squared_entry = subtracted_entry ** 2
    cuda.atomic.add(R, block_idx_x * R_cols + block_idx_y, squared_entry) # CUDA flattens array so we use this trick

@cuda.jit
def compare_distances(C, E, R, I, eps_squared):
    """
    Compares the distances between query vectors and candidate vectors to see if they are within epsilon range. Updates a adjacency matrix if so

    There should be a grid size of 2 * len(X), with two blocks per query vector. The block size should be k*m. This is because each block is responsible for a single query vector. The first block is responsible for the closest candidate vectors to the query vector and the second block is responsible for the furthest candidate vectors to the query vector. This is done to have the total thread count per block to be less than 1024 (assuming k*m <= 1024)
    
    Yes this essentially does a mask operation that is inbuilt in NumPy - but we have total control here over block and grid sizes. Also we effectively count entries of this mask at the same time

    Args:
        C: Adjacency matrix storing the cluster data for each query vector. Has shape (len(X), 2*k*m). Should have default entries of -1. For non -1 entries on the ith row, the entry is the index of a vector that is with epsilon range of the ith query vector
        E: Stores the number of neighbourhood vectors per candidate vector. Used to allow for easier processing of C in the later steps. Has shape (len(X), 1)
        R: The matrix containing the squared distances between the query and candidate vectors. Has shape (dataset_size, 2*k*m). Essentially the distances between query and candidate vectors as stored in the I matrix.
        I: The matrix containing the indices of the candidate vectors. Has shape (dataset_size, 2*k*m). The indices of the candidate vectors are stored in the same order as the distances in R. This is used to find the indices of the candidate vectors that are within epsilon range of the query vectors. Col entries are the indices of the candidate vectors
        eps_squared: The square of the epsilon value. Take epsilon squared to avoid the square root operation of the euclidean distance. I.e. we are using for x, y possitive, if x^2 < y then x < sqrt(y)
    """
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x

    i = block_idx // 2

    if not (block_idx & 1): # Check if even. True if even
        # Closest candidate vectors
        j = thread_idx
        dist_squared = R[i][j]

    else:
        # Furthest candidate vectors
        j = thread_idx + R.shape[1] // 2
        dist_squared = R[i][j]

    if dist_squared < eps_squared:
        C[i][j] = I[i][j]
        cuda.atomic.add(E, i, 1)


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

def find_B_entry(A, B, thread_idx, block_idx):
    """
    Finds the entry from B to retrieve, given the current thread and block index

    Assumes that there is a grid size (number of blocks) of 2*len(X) and a block size of k*m

    Args:
        A:
        B:
        thread_idx:
        block_idx:

    Returns:

    """


    # TODO integrate this process with everything else

    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    k = A.shape[1]
    m = B.shape[1]

    A_col = thread_idx // m
    B_col = thread_idx % m

    q_idx = block_idx // 2

    A_entry = A[block_idx][A_col]
    return B[A_entry][B_col]









    





