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
def gs_dbscan(X, D, minPts, k, m, eps):
    pass
