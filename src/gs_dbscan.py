from numba import cuda
import numpy as np
from src.limited_priority_queue import LimitedPriorityQueue
import ThrustRTC as trtc
import cupy as cp
import networkx as nx
from timeit import default_timer as timer

def gs_dbscan(X, D, minPts, k, m, eps, skip_pre_checks=False):
    """
    Main algo for gs-dbscan

    Equiv to algo 1 from paper

    Args:s
        X: point dataset. Stored along the ROWS
        D: number of random vectors to generate for projections
        minPts: Minimum number of points for a point to be clasified as a core point
        k: k param as per S-DBSCAN paper
        m: m param as per S-DBSCAN paper
        eps: Cluster radius for identifying core points
    """
    if not skip_pre_checks:
        prechecks(X, D, minPts, k, m, eps)
    
    A, B = preprocessing(X, D)

    distances = find_distances(X, A, B)

    graph_adjacency_list = construct_cluster_graph(distances, eps, k, m)

    connected_components = perform_BFS(graph_adjacency_list)

    return connected_components

def prechecks(X, D, minPts, k, m, eps):
    """
    Prechecks for the GS-DBSCAN algorithm

    Args:
        X: point dataset. Stored along the ROWS
        D: number of random vectors to generate for projections
        minPts: Minimum number of points for a point to be clasified as a core point
        k: k param as per S-DBSCAN paper
        m: m param as per S-DBSCAN paper
        eps: Cluster radius for identifying core points
    """
    assert X.shape[0] > 0, "Dataset is empty"
    assert X.shape[1] > 0, "Dataset has no features"
    assert D > 0, "Number of random vectors must be greater than 0"
    assert D >= k, "Number of random vectors must be greater than or equal to k"

def preprocessing(X, D, k, m):
    """
    Pre-processing steps for GS-DBSCAN

    Args:
        X: point dataset. Stored along the ROWS
        D: random vectors. Stored along the COLS

    Returns:
        A tuple of 2 matrices. The first matrix is the A matrix and the second matrix is the B matrix 
    """
    Y = create_random_vec_mat(D, X.shape[1])
    return construct_A_B_matrices(X, D, k, m)

### Calculating the random projections of the dataset ###

def create_random_vec_mat(D, d):
    """
    Creates a matrix with random values representing random vectors.

    Samples from the gaussian distribution

    Vectors are stored along the COLS of this matrix (not the rows)

    Args:
        D: Number of random vectors to create
        d: Dimension of the random vectors to create

    Returns:
        A d*D matrix. Each column is a random vector
    """
    return np.random.normal(size=(d, D))

def random_projections(X):
    """
    Uses FHT to compute the random projections of the dataset

    Args:
        X: point dataset. Stored along the ROWS
    """

    # TODO Implement me!
    pass

def construct_A_B_matrices(X, Y, k, m):
    """
    Constructs the A and B matrices for the dataset and random vectors

    A has a shape of (len(X), 2 * k). The i'th row of A stores the indices of the 2*k closest and furthest random vectors to the i'th point in X. Closest in first k slots and furthest in the last k slots.

    B has a shape of (2*len(D), m). The (2j)'th row of B stores the indices of vectors closest to the j'th random vector. And the (2j+1)'th row of B stores the indices of the vectors furthest from the j'th random vector

    Args: 
        X: point dataset, a matrix of points stored along the ROWS
        Y: random vectors, a matrix of random vectors stored along the COLS
        k: k param k parameter as described above
        m: m param m parameter as described above

    Returns:
        A tuple of two matrices. The first matrix is the A matrix and the second matrix is the B matrix
    """

    # TODO, do this on the GPU
    
    product = dataset_random_projections(X, Y)

    A = np.zeros(size=(len(X), 2*k))
    B = np.zeros(size=(len(Y), 2*m))

    for j in range(len(A)):
        top_k, bottom_k = top_closest_and_furthest_indices(product[j], k)
        A[j, :k] = 2*top_k
        A[j, k:] = 2*bottom_k + 1

    for j in range(len(B)):
        top_m, bottom_m = top_closest_and_furthest_indices(product[:, j], m)
        B[2*j, :m] = top_m
        B[2*j+1, m:] = bottom_m

    return A, B

def top_closest_and_furthest_indices(vector, n):
    """
    Finds the indices of the top k largest and smallest entries from a vector

    Args:
        vector: the vector as mentioned above
        k: k as mentioned   

    Returns:
        A tuple of two np arrays. The first array is the top k closest indices and the second array is the top m furthest indices
    """
    top_heap = LimitedPriorityQueue(n)
    bottom_heap = LimitedPriorityQueue(n, reverse=True)
    for i in range(len(vector)):
        entry = vector[i]
        top_heap.push(i, entry)
        bottom_heap.push(i, entry)
    return np.array(top_heap.heap_items()), np.array(bottom_heap.heap_items())

def dataset_random_projections(X, Y):
    """
    Computes the random projections of the dataset onto the random vectors

    Intend to make this faster in the future by using the FHT. For now its an overbloated method doing one line of code :D

    Args:
        X: point dataset A matrix of points stored along the ROWS
        Y: random vectors. A matrix with random vectors stored along the COLS

    Returns:
        A matrix of the random projections of the dataset onto the random vectors. An entry (i, j) is the inner product of the ith dataset vector with the jth random vector
    """
    return np.dot(X, Y)

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

### Calculating distances between query and candidate vectors ###

def find_distances(X, A, B, alpha=1.2):
    """
    Finds the distances between each query vector and their candidate vectors

    I'm curious to see if this is faster than the Numba CUDA implementation
    
    # TODO add batch processing for X with large dimensions

    Args:
        X: dataset of vectors stored along the ROWS
        A: matrix of indices of the closest and furthest random vectors to each point in X. Has shape (len(X), 2*k)
        B: matrix of indices of the closest and furthest points to each random vector in D. Has shape (2*len(D), m)
        alpha: a scaling factor to adjust the batch size. See docs for get_batch_size for more info
    Returns:
        A matrix of distances between the query and candidate vectors. Has shape (len(X), 2*k*m)
    """
    start = timer()

    k = A.shape[1] // 2
    m = B.shape[1]

    n = X.shape[0]
    d = X.shape[1]

    batch_size = get_batch_size(n, X.shape[1], k, m, alpha=alpha)

    distances = cp.empty(shape=(n, 2*k*m), dtype=cp.float16) # float32 causes a memory overload. float16 is fine (for eps 2DP)
    # Z_batch = cp.empty(shape=(batch_size, 2*k, m, d), dtype=cp.float16)
    # Z_batch_adj = cp.empty(shape=(batch_size, 2*k*m, d), dtype=cp.float16)
    # Y_batch = cp.empty(shape=Z_batch_adj.shape, dtype=cp.float16)

    for i in range(0, n, batch_size):
        max_batch_idx = min(i+batch_size, X.shape[0])

        Z_batch = X[B[A[i:max_batch_idx]]]
        Z_batch_adj = Z_batch.reshape(Z_batch.shape[0], Z_batch.shape[1] * Z_batch.shape[2], Z_batch.shape[3])

        Y_batch = Z_batch_adj - X[i:max_batch_idx, cp.newaxis, :]

        distances[i:max_batch_idx] = cp.linalg.norm(Y_batch, axis=2) # using linalg.norm faster than cp.sum(Y**2, axis=2)
    
    return distances

def get_batch_size(n, d, k, m, alpha = 1.2):
    """
    Calculates the batch size. 

    Batch size is best a divisor of n, to maximise GPU utilisation (its better when batches are all the same size and the last one isn't a special case)

    Args:
        n: number of query vectors
        d: dimension of the query vectors
        k: k as described above
        m: m as described above
        alpha: a scaling factor to adjust the batch size. Default is 1.2. Mainly here as a toggle to maximise batch size without overloading the memory
    """

    # For some reason, no matter what batch size you choose, after 200 iterations
    # Of the above forloop, the performance falls of a cliff # Can be adjusted to make the batch size smaller or larger

    batch_size = int((n*d*2*k*m) // (1024**3) // alpha) # Make batch size such that the total memory used is less than 1GB. 32 here is for float32. 1024**3 is 1GB

    if batch_size == 0:
        return n

    for divisor in range(batch_size - 1, 0, -1):
        if n % divisor == 0:
            batch_size = divisor
            break

    return batch_size


### Constructing the cluster graph ###

def construct_cluster_graph(distances, eps, k, m):
    """
    Constructs the cluster graph as per the G-DBSCAN algorithm

    Args:
        distances: The matrix containing the distances between the query and candidate vectors. Stored on GPU memory via CUDA
        eps: The epsilon value. Used to determine if the candidate vectors are within epsilon range of the query vectors
        k: k as described above
        m: m as described above
        n: number of query vectors

    Returns:
        The adjacency list as per the G-DBSCAN algorithm
        And the processed degree array of the query vectors. Stored on GPU memory (within the thrust library)
    """

    E = construct_query_vector_degree_array(distances, eps)

    V = process_query_vector_degree_array(E) 

    adjacency_list = assemble_adjacency_list(distances, E, V, eps)

    return adjacency_list, V

def construct_query_vector_degree_array(distances, eps):
    """
    Finds the degree of the query vectors as per the G-DBSCAN algorithm.

    This is used in later steps for the construction of the cluster graph

    Args:
        distances: The matrix containing the distances between the query and candidate vectors. Has shape (dataset_size, 2*k*m)
        eps: The epsilon value for DBSCAN
    
    Returns:
        The degree array of the query vectors. Has shape (len(X), 1)
    """
    return cp.sum(distances < eps, axis=1)
    
def process_query_vector_degree_array(E):
    """
    Processes the query vector degree array into a form that can be used to create the cluster graph

    Creates a new vector V which stores the cumulative sum of the degree array E. This is used to find the starting index of the neighbourhood vectors for each query vector when creating the cluster graph.

    Inspired by step 2 of the G-DBSCAN algorthm. 

    Uses the ThrustRTC library to perform an exclusive scan

    Args:
        E: A CuPy Array. The degree array of the query vectors. Has shape (len(X), 1)
    
    Returns:
        The processed E array. As a cupy array
    """
    return cp.cumsum(E) - E # Effectively does an exclusive scan, very fast anwyays, probably not worth the effort to use ThrustRTC

    #E_d = trtc.DVCupyVector(E)
    #V_d = trtc.DVCupyVector(cp.zeros(len(E), dtype=cp.int32))
    #trtc.Exclusive_Scan(E_d, V_d)
    #return cp.array(V_d.to_host())

def assemble_adjacency_list(distances, E, A, B, V, eps, block_size=1024):
    """
    Assembles the adjacency list as per the G-DBSCAN algorithm

    Some parameters are saved onto cuda device memory. Just to avoid copying them back to CPU memory for processing

    Args:
        distances: The matrix containing the distances between the query and candidate vectors. Has shape (dataset_size, 2*k*m)
        E: CuPy array. The degree array of the query vectors. Has shape (len(X), 1)
        A: CuPy array. The matrix of indices of the closest and furthest random vectors to each point in X. Has shape (len(X), 2*k)
        B: CuPy array. The matrix of indices of the closest and furthest points to each random vector in D. Has shape (2*len(D), m)
        V: CuPy array. The processed degree array of the query vectors (as per the G-DBSCAN algorithm). Has shape (len(X), 1)
        eps: The epsilon value for DBSCAN.
        k: k as described above
        m: m as described above
        n: number of query vectors
    
    Returns:
        An adjacency list as per the G-DBSCAN algorithm. Stored on CPU memory. Has shape (E[-1] + V[-1],). The indices of the candidate vectors that are within epsilon range of the query vectors are stored in this list. The indices are stored in the same order as the query vectors. The first E[0] entries are the indices of the closest candidate vectors to the query vectors. The next E[1] entries are the indices of the next closest candidate vectors to the query vectors and so on

    """
    adjacency_list = cp.full(shape=(int(E[-1] + V[-1]),), fill_value=-1)
    adjacency_list_d = cuda.as_cuda_array(adjacency_list)
    distances_d = cuda.as_cuda_array(distances)
    	
    A_d = cuda.as_cuda_array(A)
    B_d = cuda.as_cuda_array(B)
    V_d = cuda.as_cuda_array(V)

    construct_adjacency_list_for_query_vector[len(E) // block_size, block_size](distances_d, adjacency_list_d, V_d, A_d, B_d, eps)
    cuda.synchronize()
    
    return adjacency_list_d.copy_to_host()
    

@cuda.jit
def construct_adjacency_list_for_query_vector(distances_d, adjacency_list_d, V_d, A_d, B_d, eps):
    #TODO add A and B to above signature
    """
    Constructs a segment of the adjacency list for a single query vector

    Thread is responsible for a single query vector. The block size should be the maximum - around 1024. Assumes that a query vector doesn't have more than 1024 vectors in its epsilon neighbourhood.
    
    Modifies G in place
    
    Args:
        distances_d: The matrix containing the distances between the query and candidate vectors. Has shape (dataset_size, 2*k*m). Essentially the distances between query and candidate vectors as stored in the I matrix.
        adjacency_list_d: The adjacency list to store the neighbourhood vectors of the query vectors. Has shape (E[-1] + V[-1],). The indices of the candidate vectors that are within epsilon range of the query vectors are stored in this list. The indices are stored in the same order as the query vectors. The first E[0] entries are the indices of the closest candidate vectors to the query vectors. The next E[1] entries are the indices of the next closest candidate vectors to the query vectors and so on
        A_d: The matrix of indices of the closest and furthest random vectors to each point in X. Has shape (len(X), 2*k). Stored on GPU memory
        B_d: The matrix of indices of the closest and furthest points to each random vector in D. Has shape (2*len(D), m). Stored on GPU memory
        V_d: The processed degree array of the query vectors. Has shape (len(X), 1). Stored on GPU memory
        eps: The epsilon value
    """
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_size = cuda.blockDim.x

    i = block_idx * block_size + thread_idx
    curr_index = V_d[i]
    m = B_d.shape[1]

    # TODO can make this more efficient if E[i] == 0 then this loop can be skipped altogether
    # Can also have a break condition if you have already added E[i] neighbourhood vectors


    for j in range(distances_d.shape[1]):
        if distances_d[i][j] <  eps:
            A_col = j // m
            B_col = j % m
            if i == 0:
                print(j, A_col, B_col)
            B_row = A_d[i][A_col]
            neighbourhood_vector_idx = B_d[B_row][B_col]

            adjacency_list_d[curr_index] = neighbourhood_vector_idx 
            curr_index += 1

### Performing BFS to find connected components ###

def perform_BFS(adjacency_list, V):
    """
    Performs a BFS on the adjacency list to find the connected components

    Currently this is done with a CPU using the NetworkX library, just as a proof of concept. 

    For C++ version, we can use a parallel BFS implementation

    Args:
        adjacency_list: The adjacency list as per the G-DBSCAN algorithm. Stored on CPU memory. Has shape (E[-1] + V[-1],). The indices of the candidate vectors that are within epsilon range of the query vectors are stored in this list. The indices are stored in the same order as the query vectors. The first E[0] entries are the indices of the closest candidate vectors to the query vectors. The next E[1] entries are the indices of the next closest candidate vectors to the query vectors and so on
    
    Returns:
        The connected components of the cluster graph

    """
    # Done this with a CPU for now. Just as a proof of concept. Will be done on the GPU with C++ in the future
    adjacency_mat = np.zeros(shape=(len(V), len(V)), dtype=bool)
    for i in range(len(V)):
        for j in range(V[i], V[i+1]):
            adjacency_mat[i, adjacency_list[j]] = 1

    G = nx.from_numpy_matrix(adjacency_mat)
    return nx.connected_components(G)








    





