from numba import cuda
import numpy as np
from limited_priority_queue import LimitedPriorityQueue
import ThrustRTC as trtc

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

    distances_d = find_squared_distances(X, A, B)

    graph_adjacency_list = construct_cluster_graph(len(X), distances_d, eps, k, m)

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

    B has a shape of (len(D), 2 * m). The j'th row of B stores the indices of the 2*m closest and furthest points to the j'th random vector in D. Closest in first m slots and furthest in the last m slots.

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
        A[2*j, :k] = top_k
        A[2*j+1, k:] = bottom_k

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
        A tuple of two lists. The first list is the top k closest indices and the second list is the top m furthest indices
    """
    top_heap = LimitedPriorityQueue(n)
    bottom_heap = LimitedPriorityQueue(n, reverse=True)
    for i in range(len(vector)):
        entry = vector[i]
        top_heap.push(i, entry)
        bottom_heap.push(i, entry)
    return top_heap.heap_items(), bottom_heap.heap_items()

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

def find_squared_distances(X, A, B, batch_size = 500):
    """
    Handles finding the squared distances between each of the query vectors and their candidate neighbourhood vectors

    Args:
        X dataset of vectors stored along the ROWS
        A matrix of indices of the closest and furthest random vectors to each point in X. Has shape (len(X), 2*k)
        B matrix of indices of the closest and furthest points to each random vector in D. Has shape (len(D), 2*m)
        batch_size: The number of query vectors to process at a time. This is to avoid running out of memory on the GPU. Defaults to 500
    
    Returns:
        A matrix of squared distances between the query and candidate vectors. Has shape (len(X), 2*k*m). IMPORTANT This is a reference to the matrix stored on GPU memory (i.e. a cuda device array)
    """
    dataset_size = len(X)
    k = A.shape[1] // 2
    m = B.shape[1] // 2

    distances = np.array(shape=(dataset_size, 2*k*m))
    
    X_d = cuda.to_device(X)
    A_d = cuda.to_device(A)
    B_d = cuda.to_device(B)
    distances_d = cuda.to_device(distances)

    dataset_dim = X.shape[1] 

    for i in range(0, dataset_size, batch_size):
        find_squared_distances_at_start_i[batch_size, dataset_dim](X_d, A_d, B_d, distances_d, i)
    
    return distances_d


@cuda.jit
def find_squared_distances_at_start_i(X_d, A_d, B_d, distances_d, start_i):
    """
    Combines subtraction, squaring and summing steos in one kernel. 

    Assumes that processing is done in batches (say 500 query vectors per grid at a time)

    Idea is that you organise the grid with query vectors along x direction, and candidate vectors along the y direction. This way you can process the distances between the query and candidate vectors in parallel.

    Each thread within a block handles a different entry of the query vectors to candidate vector. I.e the ith thread processes the ith entry of the query vector to the candidate vector.

    A single thread does the subtraction, squaring and summing for a single entry between a query and candidate vector, then adds it to the result matrix R.

    Effectively finds the euclidean distance squared between two vectors in the dataset.

    Args:
        X_d: The dataset containing the vectors, stored along the ROWS
        A_d: The matrix containing the indices of the closest and furthest random vectors to each point in X. Has shape (len(X), 2*k)
        B_d: The matrix containing the indices of the closest and furthest points to each random vector in D. Has shape (len(D), 2*m)
        distances_d: The matrix to store the result of the subtraction, squaring and summing. Has shape (dataset_size, 2*k*m). Each entry should be atomic. And should be initialised with zero entries
        start_i: Starting index of the q'ith vector to start at in the dataset. This is added to allow for batching of this process, since it can't all be done at once
    """
    thread_idx = cuda.threadIdx.x # Entry within vectors to process
    block_idx_x = cuda.blockIdx.x # Query vector index
    block_idx_y = cuda.blockIdx.y # Candidate vector index

    distance_cols = distances_d.shape[1]

    i = start_i + block_idx_x
    A_col = block_idx_y // B_d.shape[1]
    B_col = block_idx_y % B_d.shape[1]

    j = B_d[A_d[i][A_col]][B_col] # NTS: Removes usage of I matrix

    subtracted_entry = X_d[i][thread_idx] - X_d[j][thread_idx]
    squared_entry = subtracted_entry ** 2
    cuda.atomic.add(distances_d, block_idx_x * distance_cols + block_idx_y, squared_entry) # CUDA flattens array so we use this trick

### Constructing the cluster graph ###

def construct_cluster_graph(dataset_size, distances_d, eps, k, m):
    """
    Constructs the cluster graph as per the G-DBSCAN algorithm

    Args:
        dataset_size: The number of vectors in the dataset
        distances: The matrix containing the squared distances between the query and candidate vectors. Stored on GPU memory via CUDA
        eps: The epsilon value. Used to determine if the candidate vectors are within epsilon range of the query vectors
        k: k as described above
        m: m as described above
        n: number of query vectors

    Returns:
        The adjacency list as per the G-DBSCAN algorithm
    """
    eps_squared = eps ** 2

    E = np.zeros(shape=(dataset_size, 1))
    E_d = cuda.to_device(E)

    construct_query_vector_degree_array[2*dataset_size, k*m](E_d, distances_d, eps_squared)

    E = E_d.copy_to_host()

    V_trtc = process_query_vector_degree_array(E) # Stored with ThrustRTC

    adjacency_list = assemble_adjacency_list(distances_d, E, V_trtc, eps_squared)

    return adjacency_list, V_trtc

@cuda.jit
def construct_query_vector_degree_array(E_d, distances_d, eps_squared):
    """
    Compares the distances between query vectors and candidate vectors to see if they are within epsilon range. Updates an array containing the degree (number of neighbouring cluster points) of each query vector

    There should be a grid size of 2 * len(X), with two blocks per query vector. The block size should be k*m. This is because each block is responsible for a single query vector. The first block is responsible for the closest candidate vectors to the query vector and the second block is responsible for the furthest candidate vectors to the query vector. This is done to have the total thread count per block to be less than 1024 (assuming k*m <= 1024)

    Args:
        E_d: Stores the number of neighbourhood vectors per candidate vector. Used to allow for easier processing of C in the later steps. Has shape (len(X), 1)
        distances_d: The matrix containing the squared distances between the query and candidate vectors. Has shape (dataset_size, 2*k*m). Essentially the distances between query and candidate vectors as stored in the I matrix.
        eps_squared: The square of the epsilon value. Take epsilon squared to avoid the square root operation of the euclidean distance. I.e. we are using for x, y possitive, if x^2 < y then x < sqrt(y)
    """
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x

    i = block_idx // 2

    if not (block_idx & 1): # Check if even. True if even
        # Closest candidate vectors
        j = thread_idx
        dist_squared = distances_d[i][j]

    else:
        # Furthest candidate vectors
        j = thread_idx + distances_d.shape[1] // 2
        dist_squared = distances_d[i][j]

    if dist_squared < eps_squared:
        cuda.atomic.add(E_d, i, 1)

def process_query_vector_degree_array(E):
    """
    Processes the query vector degree array into a form that can be used to create the cluster graph

    Creates a new vector V which stores the cumulative sum of the degree array E. This is used to find the starting index of the neighbourhood vectors for each query vector when creating the cluster graph.

    Inspired by step 2 of the G-DBSCAN algorthm. 

    Uses the ThrustRTC library to perform an exclusive scan

    Args:
        E: The degree array of the query vectors. Has shape (len(X), 1)
    
    Returns:
        The processed E array. Stored on GPU memory
    """
    E_d = trtc.device_vector_from_list(E, "int32_t") # These operations are expensive to copy back and forth. In C++ implementation could use ThrustRTC for all kernel implementations to avoid having to copy back and forth.
    V_d = trtc.device_vector("int32_t", E.shape[0])
    trtc.Exclusive_Scan(E_d, V_d)
    return V_d

def assemble_adjacency_list(distances_d, E, V_d, eps_squared, block_size=1024):
    """
    Assembles the adjacency list as per the G-DBSCAN algorithm

    Some parameters are saved onto cuda device memory. Just to avoid copying them back to CPU memory for processing

    Args:
        distances_d: The matrix containing the squared distances between the query and candidate vectors. Has shape (dataset_size, 2*k*m). Stored in GPU memory
        E: The degree array of the query vectors. Has shape (len(X), 1)
        V: The processed degree array of the query vectors (as per the G-DBSCAN algorithm). Has shape (len(X), 1). Stored in GPU memory with the ThrustRTC library
        eps_squared: The square of the epsilon value. Take epsilon squared to avoid the square root operation of the euclidean distance. I.e. we are using for x, y possitive, if x^2 < y then x < sqrt(y)
        k: k as described above
        m: m as described above
        n: number of query vectors
    
    Returns:
        An adjacency list as per the G-DBSCAN algorithm. Stored on CPU memory. Has shape (E[-1] + V[-1],). The indices of the candidate vectors that are within epsilon range of the query vectors are stored in this list. The indices are stored in the same order as the query vectors. The first E[0] entries are the indices of the closest candidate vectors to the query vectors. The next E[1] entries are the indices of the next closest candidate vectors to the query vectors and so on

    """
    dataset_size = len(E)
    last_V_entry = V_d.to_host(begin=dataset_size-1, end=dataset_size)[0]

    adjacency_list = np.full(shape=(E[-1] + last_V_entry,), fill_value=-1)
    adjacency_list_d = cuda.to_device(adjacency_list)

    construct_adjacency_list_for_query_vector[len(E), block_size](distances_d, adjacency_list_d, V_d, eps_squared)
    
    return adjacency_list_d.copy_to_host()
    

@cuda.jit
def construct_adjacency_list_for_query_vector(distances_d, adjacency_list_d, V_d, eps_squared):

    #TODO add A and B to above signature
    """
    Constructs a segment of the adjacency list for a single query vector

    Thread is responsible for a single query vector. The block size should be the maximum - around 1024. Assumes that a query vector doesn't have more than 1024 vectors in its epsilon neighbourhood.
    
    Modifies G in place
    
    Args:
        distances_d: The matrix containing the squared distances between the query and candidate vectors. Has shape (dataset_size, 2*k*m). Essentially the distances between query and candidate vectors as stored in the I matrix.
        G: The adjacency list to store the neighbourhood vectors of the query vectors. Has shape (E[-1] + V[-1],). The indices of the candidate vectors that are within epsilon range of the query vectors are stored in this list. The indices are stored in the same order as the query vectors. The first E[0] entries are the indices of the closest candidate vectors to the query vectors. The next E[1] entries are the indices of the next closest candidate vectors to the query vectors and so on
        V: The processed degree array of the query vectors. Has shape (len(X), 1)
        eps_squared: The square of the epsilon value. Take epsilon squared to avoid the square root operation of the euclidean distance. I.e. we are using for x, y possitive, if x^2 < y then x < sqrt(y)
    """
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_size = cuda.blockDim.x

    i = block_idx * block_size + thread_idx
    curr_index = V_d[i]
    for j in range(distances_d.shape[1]):
        if distances_d[i][j] < eps_squared:
            adjacency_list_d[curr_index] = j #T
            curr_index += 1

### Performing BFS to find connected components ###

def perform_BFS(G):
    """
    Performs a BFS on the adjacency list to find the connected components

    As per the G-DBSCAN algorithm. Inspired by "Accelerating large graph algorithms on the GPU with CUDA" Hawish et al
    """
    pass # TODO Implement me!









    





